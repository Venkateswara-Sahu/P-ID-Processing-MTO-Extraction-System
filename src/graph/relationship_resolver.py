"""
Relationship Resolver for P&ID graph connections.

Resolves and enriches relationships between engineering entities
using spatial analysis, line-following heuristics, and engineering
domain knowledge. Assigns line numbers to entities based on
proximity to line labels.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import networkx as nx

from src.graph.entity_mapper import EngineeringEntity
from src.ocr.tag_parser import TagType

logger = logging.getLogger(__name__)


class RelationshipResolver:
    """Resolves and enriches connections in the P&ID graph."""

    def __init__(self, alignment_tolerance: float = 40.0):
        """
        Args:
            alignment_tolerance: Pixel tolerance for considering
                                entities as horizontally/vertically aligned.
        """
        self.alignment_tolerance = alignment_tolerance

    def resolve(
        self,
        graph: nx.DiGraph,
        entities: List[EngineeringEntity],
    ) -> nx.DiGraph:
        """
        Resolve and enrich relationships in the graph.

        Steps:
        1. Assign line numbers to entities based on proximity
        2. Strengthen connections between same-line entities
        3. Detect and annotate branching/tee points
        4. Resolve instrument signal connections

        Args:
            graph: The P&ID graph to enrich.
            entities: List of engineering entities.

        Returns:
            Enriched graph with resolved relationships.
        """
        logger.info("Resolving relationships...")

        # Step 1: Assign line numbers to entities near line labels
        self._assign_line_numbers(graph, entities)

        # Step 2: Strengthen same-line connections
        self._strengthen_line_connections(graph)

        # Step 3: Detect instrument loops
        self._detect_instrument_loops(graph)

        # Step 4: Identify flow paths
        self._identify_flow_paths(graph)

        logger.info("Relationship resolution complete")
        return graph

    def _assign_line_numbers(
        self,
        graph: nx.DiGraph,
        entities: List[EngineeringEntity],
    ):
        """Assign line numbers to nearby entities that don't have one."""
        # Find all entities that are line labels
        line_entities = [
            e for e in entities
            if e.entity_type == "line" and e.parsed_tags
            and any(t.tag_type == TagType.LINE_NUMBER for t in e.parsed_tags)
        ]

        if not line_entities:
            logger.debug("No line number labels found")
            return

        # For each entity without a line number, find nearest line label
        for node_id, data in graph.nodes(data=True):
            if data.get("line_number") is not None:
                continue

            entity = data.get("entity")
            if entity is None or entity.position is None:
                continue

            # Find nearest line label
            nearest_line = None
            nearest_dist = float("inf")

            for line_entity in line_entities:
                if line_entity.position is None:
                    continue
                dist = self._distance(entity.position, line_entity.position)
                if dist < nearest_dist and dist < 300:  # Max association distance
                    nearest_dist = dist
                    for tag in line_entity.parsed_tags:
                        if tag.tag_type == TagType.LINE_NUMBER:
                            nearest_line = tag.tag_id
                            break

            if nearest_line:
                graph.nodes[node_id]["line_number"] = nearest_line
                entity.line_number = nearest_line

        assigned = sum(
            1 for _, d in graph.nodes(data=True) if d.get("line_number") is not None
        )
        logger.info(f"Assigned line numbers to {assigned} entities")

    def _strengthen_line_connections(self, graph: nx.DiGraph):
        """Strengthen connections between entities on the same line."""
        edges_to_update = []

        for source, target, data in graph.edges(data=True):
            source_line = graph.nodes[source].get("line_number")
            target_line = graph.nodes[target].get("line_number")

            if source_line and target_line and source_line == target_line:
                edges_to_update.append((source, target, source_line))

        for source, target, line_number in edges_to_update:
            graph[source][target]["line_number"] = line_number
            graph[source][target]["weight"] *= 0.5  # Strengthen connection

        logger.debug(f"Strengthened {len(edges_to_update)} same-line connections")

    def _detect_instrument_loops(self, graph: nx.DiGraph):
        """
        Detect instrument loops (measurement + control groupings).
        E.g., FT-101 → FIC-101 → FV-101 form a flow control loop.
        """
        instruments = [
            (nid, data)
            for nid, data in graph.nodes(data=True)
            if data.get("entity_type") == "instrument"
        ]

        # Group instruments by loop number
        loop_groups: Dict[str, List[str]] = {}
        for node_id, data in instruments:
            entity = data.get("entity")
            if entity and entity.parsed_tags:
                for tag in entity.parsed_tags:
                    if tag.loop_number:
                        key = tag.loop_number
                        if key not in loop_groups:
                            loop_groups[key] = []
                        loop_groups[key].append(node_id)

        # Add loop metadata
        loop_count = 0
        for loop_num, members in loop_groups.items():
            if len(members) > 1:
                loop_count += 1
                loop_id = f"LOOP-{loop_num}"
                for node_id in members:
                    graph.nodes[node_id]["instrument_loop"] = loop_id

                # Ensure loop members are connected
                for i in range(len(members) - 1):
                    if not graph.has_edge(members[i], members[i + 1]):
                        graph.add_edge(
                            members[i],
                            members[i + 1],
                            connection_type="signal",
                            weight=50,
                            loop_id=loop_id,
                        )

        if loop_count:
            logger.info(f"Detected {loop_count} instrument loops")

    def _identify_flow_paths(self, graph: nx.DiGraph):
        """Identify continuous flow paths through the graph."""
        if graph.number_of_nodes() == 0:
            return

        # Find path segments along each line
        line_groups: Dict[str, List[str]] = {}
        for node_id, data in graph.nodes(data=True):
            line = data.get("line_number")
            if line:
                if line not in line_groups:
                    line_groups[line] = []
                line_groups[line].append(node_id)

        # Sort entities on each line by x-position (left-to-right flow)
        for line, node_ids in line_groups.items():
            positions = []
            for nid in node_ids:
                pos = graph.nodes[nid].get("position")
                if pos:
                    positions.append((nid, pos[0]))

            positions.sort(key=lambda x: x[1])

            # Mark flow order
            for order, (nid, _) in enumerate(positions):
                graph.nodes[nid]["flow_order"] = order

        logger.debug(f"Identified flow paths on {len(line_groups)} lines")

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
