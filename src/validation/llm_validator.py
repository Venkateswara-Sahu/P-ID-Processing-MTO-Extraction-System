"""
LLM-based Validator using LangGraph and Groq.

Implements a 3-node agentic validation pipeline:
1. Extractor Node — Summarizes extracted P&ID data
2. Validator Node — LLM checks engineering consistency
3. Reporter Node — Generates human-readable validation report

Uses Groq's ``openai/gpt-oss-120b`` for fast, free inference. The model
name used to be ``llama-3.3-70b-versatile``, which Groq retired -- it
returns HTTP 404, so LLM validation silently never ran.

Every failure mode (no API key, retired model, missing langgraph, network
error) degrades to a clearly-flagged rule-only report rather than raising.
"""

import logging
import os
import re
from typing import Annotated, Dict, List, Optional, TypedDict

from config.settings import settings

logger = logging.getLogger(__name__)


# ---- State Schema ----

class ValidationState(TypedDict):
    """State that flows through the LangGraph validation pipeline."""

    # Input data
    extraction_summary: str
    entity_count: int
    graph_summary: dict
    rule_based_issues: list

    # LLM outputs
    llm_analysis: str
    llm_issues: list
    final_report: str

    # Control
    validation_passed: bool


class LLMValidator:
    """
    LangGraph-based LLM validator for P&ID extraction results.

    Uses a 3-node graph:
    [Summarize] -> [Validate] -> [Report]
    """

    def __init__(self, api_key: str = None, model: str = None):
        """
        Args:
            api_key: Groq API key. Falls back to settings/env. An explicit
                empty string means "no key" and is NOT replaced by the
                configured one.
            model: Groq model name.
        """
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or settings.groq_model
        self.graph = None
        self._llm = None

    # ------------------------------------------------------------------ #
    # Setup                                                                 #
    # ------------------------------------------------------------------ #

    def _init_llm(self):
        """Initialize the Groq LLM. Returns False if it cannot be used."""
        if not self.api_key:
            logger.warning(
                "No Groq API key configured. LLM validation will be skipped. "
                "Set GROQ_API_KEY in .env or get a free key from console.groq.com"
            )
            return False

        try:
            from langchain_groq import ChatGroq

            self._llm = ChatGroq(
                model=self.model,
                api_key=self.api_key,
                temperature=settings.groq_temperature,
                max_tokens=2048,
                timeout=settings.groq_timeout,
                max_retries=settings.groq_max_retries,
            )
            logger.info(f"Groq LLM initialized: {self.model}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {e}")
            return False

    def _build_graph(self):
        """
        Build the LangGraph validation pipeline.

        Raises:
            ImportError: if langgraph is unavailable. The caller handles it.
        """
        from langgraph.graph import StateGraph, START, END

        builder = StateGraph(ValidationState)

        # Add nodes
        builder.add_node("summarize", self._summarize_node)
        builder.add_node("validate", self._validate_node)
        builder.add_node("report", self._report_node)

        # Define edges
        builder.add_edge(START, "summarize")
        builder.add_edge("summarize", "validate")
        builder.add_edge("validate", "report")
        builder.add_edge("report", END)

        self.graph = builder.compile()
        logger.info("LangGraph validation pipeline built")

    # ------------------------------------------------------------------ #
    # Entry point                                                           #
    # ------------------------------------------------------------------ #

    def validate(
        self,
        entities_data: List[dict],
        graph_summary: dict,
        rule_based_issues: List[dict],
    ) -> dict:
        """
        Run the full LLM validation pipeline.

        Args:
            entities_data: List of entity dicts from extraction.
            graph_summary: Graph structure summary.
            rule_based_issues: Issues from deterministic rule checks.

        Returns:
            Validation result dict with analysis, issues, and report. The
            ``degraded`` flag is True when the LLM stage did not run.
        """
        if not self._init_llm():
            return self._fallback_report(
                rule_based_issues,
                reason="No Groq API key is configured.",
            )

        # Prepare extraction summary for the LLM
        extraction_summary = self._format_extraction_summary(
            entities_data, graph_summary
        )

        # Initial state
        initial_state = ValidationState(
            extraction_summary=extraction_summary,
            entity_count=len(entities_data),
            graph_summary=graph_summary,
            rule_based_issues=rule_based_issues,
            llm_analysis="",
            llm_issues=[],
            final_report="",
            validation_passed=True,
        )

        try:
            # Building the graph is INSIDE the try: a broken or missing
            # langgraph must degrade to a rule-only report, not kill the
            # whole pipeline.
            if self.graph is None:
                self._build_graph()

            result = self.graph.invoke(initial_state)
            return {
                "llm_analysis": result["llm_analysis"],
                "llm_issues": result["llm_issues"],
                "report": result["final_report"],
                "passed": result["validation_passed"],
                "degraded": False,
            }
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return self._fallback_report(
                rule_based_issues,
                reason=f"The LLM validation stage failed: {e}",
            )

    # ---- Graph Nodes ----

    def _summarize_node(self, state: ValidationState) -> dict:
        """Node 1: Summarize the extraction data for LLM analysis."""
        prompt = f"""You are an engineering P&ID (Piping & Instrumentation Diagram) validation expert.

Below is a summary of data extracted from a P&ID drawing using AI (YOLOv8 for symbol detection + Tesseract for text).
Analyze this extraction and identify any potential issues or inconsistencies.

## Extracted Data Summary
{state['extraction_summary']}

## Graph Structure
- Total nodes: {state['graph_summary'].get('total_nodes', 0)}
- Total edges: {state['graph_summary'].get('total_edges', 0)}
- Entity types: {state['graph_summary'].get('entity_types', {})}
- Connected components: {state['graph_summary'].get('connected_components', 0)}

## Rule-Based Issues Already Found
{self._format_issues(state['rule_based_issues'])}

Provide a concise technical analysis of the extraction quality. Focus on:
1. Completeness — are typical P&ID components present?
2. Consistency — do tag patterns, line numbers, and instrument loops make sense?
3. Potential errors — any obviously incorrect classifications or tags?
"""
        response = self._llm.invoke(prompt)
        return {"llm_analysis": response.content}

    def _validate_node(self, state: ValidationState) -> dict:
        """Node 2: Validate engineering consistency using LLM reasoning."""
        prompt = f"""Based on the following analysis of P&ID extraction data, identify specific engineering validation issues.

## Previous Analysis
{state['llm_analysis']}

For each issue found, provide:
- Issue description
- Severity (error/warning/info)
- Affected entity or component
- Recommended action

Format your response as a numbered list of issues, one per line, in the form:
<number>. <description> | Severity: <error|warning|info> | Entity: <entity id or name> | Action: <recommended action>

If no additional issues are found beyond rule-based checks, write exactly:
No additional issues found.

Focus only on engineering-relevant issues, NOT on AI/ML model accuracy.
"""
        response = self._llm.invoke(prompt)

        llm_issues = self._parse_issues(response.content)

        return {
            "llm_issues": llm_issues,
            "validation_passed": self._compute_passed(llm_issues),
        }

    def _report_node(self, state: ValidationState) -> dict:
        """Node 3: Generate a human-readable validation report."""
        prompt = f"""Generate a concise, professional P&ID extraction validation report.

## Extraction Summary
- Total entities extracted: {state['entity_count']}
- Graph: {state['graph_summary'].get('total_nodes', 0)} nodes, {state['graph_summary'].get('total_edges', 0)} edges

## AI Analysis
{state['llm_analysis']}

## LLM-Identified Issues
{chr(10).join(state['llm_issues']) if state['llm_issues'] else 'None'}

## Rule-Based Issues
{self._format_issues(state['rule_based_issues'])}

Generate a report with:
1. **Overall Assessment** — Pass/Fail with brief justification
2. **Key Findings** — Top 3-5 most important findings
3. **Recommendations** — Actionable next steps
4. **Confidence Level** — Your assessment of extraction quality (High/Medium/Low)

Keep the report under 300 words. Use markdown formatting.
"""
        response = self._llm.invoke(prompt)
        return {"final_report": response.content}

    # ------------------------------------------------------------------ #
    # Response parsing                                                      #
    # ------------------------------------------------------------------ #

    # A line that starts a new issue: an optional markdown marker, a number
    # and a period. '**1. Something**' must start an issue, and a sub-bullet
    # '- Severity: error' must NOT.
    _ISSUE_START = re.compile(r"^\s*(?:[*_#>\s]*)?(\d+)[.)]\s+(.*)$")
    _SEVERITY = re.compile(r"severity\s*[:=]\s*\**\s*(error|warning|info)\b", re.IGNORECASE)

    def _parse_issues(self, raw: str) -> List[str]:
        """
        Split a numbered issue list into individual issues.

        Handles markdown-bold headers ('**1. Missing tag**') and keeps
        sub-bullets ('- Severity: error') attached to their parent issue.
        """
        if not raw:
            return []

        issues: List[str] = []
        current: Optional[str] = None

        for line in raw.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            match = self._ISSUE_START.match(stripped)
            if match:
                if current:
                    issues.append(current.strip())
                current = stripped
                continue

            if current is not None:
                # Continuation: fold into the current issue.
                current += " " + stripped
                continue

            # Text before any numbered issue: treat the whole block as one
            # issue so nothing is silently dropped.
            current = stripped

        if current:
            issues.append(current.strip())

        return issues

    def _compute_passed(self, issues: List[str]) -> bool:
        """
        Decide pass/fail from the parsed severity fields.

        The previous implementation searched for the word 'error' anywhere
        in the text, so a reassuring "No errors detected" was read as a
        failure.
        """
        for issue in issues:
            match = self._SEVERITY.search(issue)
            if match and match.group(1).lower() == "error":
                return False
        return True

    # ------------------------------------------------------------------ #
    # Helpers                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_extraction_summary(entities_data: List[dict], graph_summary: dict) -> str:
        """Format entity data as a readable summary for the LLM."""
        # Group by type
        type_groups: Dict[str, list] = {}
        for entity in entities_data:
            etype = entity.get("entity_type", "unknown")
            if etype not in type_groups:
                type_groups[etype] = []
            type_groups[etype].append(entity)

        lines = []
        for etype, entities in type_groups.items():
            lines.append(f"\n### {etype.title()} ({len(entities)} items)")
            for e in entities[:10]:  # Limit to first 10 per type
                tag = e.get("tag_id", "no-tag")
                desc = e.get("description", "")
                conf = e.get("confidence", 0)
                lines.append(f"  - {tag}: {desc} (confidence: {conf:.2f})")
            if len(entities) > 10:
                lines.append(f"  ... and {len(entities) - 10} more")

        return "\n".join(lines)

    @staticmethod
    def _format_issues(issues: List[dict]) -> str:
        """Format rule-based issues for LLM context."""
        if not issues:
            return "No rule-based issues found."

        lines = []
        for i, issue in enumerate(issues[:15], 1):
            severity = issue.get("severity", "info")
            message = issue.get("message", "Unknown issue")
            lines.append(f"{i}. [{severity.upper()}] {message}")

        if len(issues) > 15:
            lines.append(f"... and {len(issues) - 15} more issues")

        return "\n".join(lines)

    @staticmethod
    def _fallback_report(rule_based_issues: List[dict], reason: str = "") -> dict:
        """
        Generate a report when the LLM stage could not run.

        The result is explicitly flagged ``degraded`` so the API and the
        Excel sheet can say so, rather than passing off a rule-only report
        as a full validation result.
        """
        error_count = sum(1 for i in rule_based_issues if i.get("severity") == "error")
        warning_count = sum(1 for i in rule_based_issues if i.get("severity") == "warning")

        reason_line = f"\n\n**Reason:** {reason}" if reason else ""

        report = f"""# P&ID Extraction Validation Report

## Overall Assessment
**LLM validation did not run** — showing rule-based results only.{reason_line}

## Rule-Based Findings
- **Errors:** {error_count}
- **Warnings:** {warning_count}
- **Total issues:** {len(rule_based_issues)}

## Recommendations
1. Review entities flagged with low confidence scores
2. Verify duplicate tags and orphan nodes manually
3. Check the LLM validation configuration if AI validation was expected

*This report contains deterministic rule findings only. The LLM review did not run.*
"""
        return {
            "llm_analysis": "LLM validation did not run",
            "llm_issues": [],
            "report": report,
            "passed": error_count == 0,
            "degraded": True,
            "degraded_reason": reason,
        }
