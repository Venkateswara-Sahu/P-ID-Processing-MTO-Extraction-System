"""
Engineering Rule-Based Validator for P&ID MTO data.

Deterministic validation rules based on engineering standards.
These checks don't require an LLM — they're fast, reliable, and auditable.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import networkx as nx

from src.graph.entity_mapper import EngineeringEntity

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation finding."""

    rule_id: str
    severity: Severity
    entity_id: Optional[str]
    message: str
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "entity_id": self.entity_id,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationReport:
    """Aggregated validation results."""

    issues: List[ValidationIssue] = field(default_factory=list)
    total_entities: int = 0
    entities_with_tags: int = 0
    entities_without_tags: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    @property
    def pass_rate(self) -> float:
        if self.total_entities == 0:
            return 0.0
        failed = len(set(i.entity_id for i in self.issues if i.severity == Severity.ERROR))
        return round((self.total_entities - failed) / self.total_entities * 100, 1)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_entities": self.total_entities,
                "entities_with_tags": self.entities_with_tags,
                "entities_without_tags": self.entities_without_tags,
                "errors": self.error_count,
                "warnings": self.warning_count,
                "pass_rate": f"{self.pass_rate}%",
            },
            "issues": [i.to_dict() for i in self.issues],
        }


class EngineeringRuleValidator:
    """Deterministic rule-based validator for P&ID MTO data."""

    def validate(
        self,
        entities: List[EngineeringEntity],
        graph: nx.DiGraph,
    ) -> ValidationReport:
        """
        Run all engineering validation rules.

        Args:
            entities: Extracted engineering entities.
            graph: P&ID relationship graph.

        Returns:
            ValidationReport with findings.
        """
        report = ValidationReport()
        report.total_entities = len(entities)
        report.entities_with_tags = sum(1 for e in entities if e.tag_id)
        report.entities_without_tags = report.total_entities - report.entities_with_tags

        # Run each rule
        report.issues.extend(self._rule_tag_completeness(entities))
        report.issues.extend(self._rule_unique_tags(entities))
        report.issues.extend(self._rule_instrument_loop_integrity(entities, graph))
        report.issues.extend(self._rule_low_confidence(entities))
        report.issues.extend(self._rule_orphan_nodes(graph))
        report.issues.extend(self._rule_valve_connections(entities, graph))

        logger.info(
            f"Validation complete: {report.error_count} errors, "
            f"{report.warning_count} warnings, "
            f"pass rate: {report.pass_rate}%"
        )

        return report

    def _rule_tag_completeness(self, entities: List[EngineeringEntity]) -> List[ValidationIssue]:
        """Rule R001: Every instrument and equipment should have a tag."""
        issues = []
        for entity in entities:
            if entity.entity_type in ("instrument", "equipment", "valve"):
                if not entity.tag_id:
                    issues.append(
                        ValidationIssue(
                            rule_id="R001",
                            severity=Severity.WARNING,
                            entity_id=entity.entity_id,
                            message=f"{entity.entity_type.title()} at position "
                                    f"{entity.position} has no tag identifier.",
                            suggestion="Verify OCR accuracy or check if the tag is "
                                       "located further from the symbol.",
                        )
                    )
        return issues

    def _rule_unique_tags(self, entities: List[EngineeringEntity]) -> List[ValidationIssue]:
        """Rule R002: Tag IDs should be unique (no duplicates)."""
        issues = []
        tag_map: Dict[str, List[str]] = {}

        for entity in entities:
            if entity.tag_id:
                if entity.tag_id not in tag_map:
                    tag_map[entity.tag_id] = []
                tag_map[entity.tag_id].append(entity.entity_id)

        for tag_id, entity_ids in tag_map.items():
            if len(entity_ids) > 1:
                issues.append(
                    ValidationIssue(
                        rule_id="R002",
                        severity=Severity.ERROR,
                        entity_id=entity_ids[0],
                        message=f"Duplicate tag '{tag_id}' found on "
                                f"{len(entity_ids)} entities: {entity_ids}",
                        suggestion="Check if the same symbol was detected "
                                   "multiple times or if tags are genuinely duplicated.",
                    )
                )
        return issues

    def _rule_instrument_loop_integrity(
        self,
        entities: List[EngineeringEntity],
        graph: nx.DiGraph,
    ) -> List[ValidationIssue]:
        """Rule R003: Instrument loops should have measurement + control elements."""
        issues = []

        # Group instruments by loop number
        loop_instruments: Dict[str, List[EngineeringEntity]] = {}
        for entity in entities:
            if entity.entity_type == "instrument" and entity.parsed_tags:
                for tag in entity.parsed_tags:
                    if tag.loop_number:
                        key = tag.loop_number
                        if key not in loop_instruments:
                            loop_instruments[key] = []
                        loop_instruments[key].append(entity)

        for loop_num, instruments in loop_instruments.items():
            # A complete loop typically has: Element/Transmitter + Controller + Valve
            functions = set()
            for inst in instruments:
                for tag in inst.parsed_tags:
                    if tag.function:
                        functions.update(tag.function.split("/"))

            if len(instruments) == 1:
                issues.append(
                    ValidationIssue(
                        rule_id="R003",
                        severity=Severity.INFO,
                        entity_id=instruments[0].entity_id,
                        message=f"Loop {loop_num} has only 1 instrument. "
                                f"Typical loops have 2-3 elements.",
                        suggestion="Check if other loop members are on a "
                                   "different sheet or were missed by detection.",
                    )
                )

        return issues

    def _rule_low_confidence(
        self,
        entities: List[EngineeringEntity],
        threshold: float = 0.5,
    ) -> List[ValidationIssue]:
        """Rule R004: Flag entities with low extraction confidence."""
        issues = []
        for entity in entities:
            if entity.confidence < threshold:
                issues.append(
                    ValidationIssue(
                        rule_id="R004",
                        severity=Severity.WARNING,
                        entity_id=entity.entity_id,
                        message=f"Low confidence ({entity.confidence:.2f}) for "
                                f"entity '{entity.tag_id or entity.description}' "
                                f"at {entity.position}.",
                        suggestion="Flagged for human review — verify detection "
                                   "and OCR accuracy manually.",
                    )
                )
        return issues

    def _rule_orphan_nodes(self, graph: nx.DiGraph) -> List[ValidationIssue]:
        """Rule R005: Detect isolated nodes (no connections)."""
        issues = []
        for node_id in graph.nodes():
            if graph.degree(node_id) == 0:
                data = graph.nodes[node_id]
                entity_type = data.get("entity_type", "unknown")

                # Lines and descriptions can be standalone
                if entity_type in ("line", "unknown"):
                    continue

                issues.append(
                    ValidationIssue(
                        rule_id="R005",
                        severity=Severity.INFO,
                        entity_id=node_id,
                        message=f"Orphan {entity_type} '{data.get('tag_id', node_id)}' "
                                f"has no connections.",
                        suggestion="Isolated components may indicate missed "
                                   "piping connections or edge-of-sheet continuations.",
                    )
                )
        return issues

    def _rule_valve_connections(
        self,
        entities: List[EngineeringEntity],
        graph: nx.DiGraph,
    ) -> List[ValidationIssue]:
        """Rule R006: Valves should typically have 2 connections (in and out)."""
        issues = []
        for entity in entities:
            if entity.entity_type == "valve" and entity.entity_id in graph:
                degree = graph.degree(entity.entity_id)
                if degree < 2:
                    issues.append(
                        ValidationIssue(
                            rule_id="R006",
                            severity=Severity.WARNING,
                            entity_id=entity.entity_id,
                            message=f"Valve '{entity.tag_id or entity.entity_id}' "
                                    f"has only {degree} connection(s). "
                                    f"Valves typically have 2 (inlet + outlet).",
                            suggestion="Check if piping connections were missed "
                                       "or if this is a boundary valve.",
                        )
                    )
        return issues
