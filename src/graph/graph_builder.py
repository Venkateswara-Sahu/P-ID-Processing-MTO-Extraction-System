"""
Graph Builder for P&ID engineering relationships.

Constructs a NetworkX directed graph where nodes represent
engineering entities and edges represent piping connections.
Enables graph queries like "find all valves on line 1001".
"""

import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx

from src.graph.entity_mapper import EngineeringEntity

logger = logging.getLogger(__name__)


class PIDGraphBuilder:
    """Builds a NetworkX graph from P&ID engineering entities."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def build(self, entities: List[EngineeringEntity]) -> nx.DiGraph:
        """
        Build a directed graph from engineering entities.

        Args:
            entities: List of EngineeringEntity objects.

        Returns:
            NetworkX DiGraph with entities as nodes and connections as edges.
        """
        self.graph = nx.DiGraph()

        # Step 1: Add all entities as nodes
        for entity in entities:
            self._add_entity_node(entity)

        # Step 2: Infer connections based on spatial proximity and line numbers
        self._infer_connections(entities)

        logger.info(
            f"Built graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

        return self.graph

    def _add_entity_node(self, entity: EngineeringEntity):
        """Add an engineering entity as a node in the graph."""
        self.graph.add_node(
            entity.entity_id,
            tag_id=entity.tag_id,
            entity_type=entity.entity_type,
            description=entity.description,
            category=entity.component_category,
            position=entity.position,
            confidence=entity.confidence,
            size=entity.size,
            specification=entity.specification,
            line_number=entity.line_number,
            # Store full entity reference
            entity=entity,
        )

    def _infer_connections(
        self,
        entities: List[EngineeringEntity],
        proximity_threshold: float = 200.0,
    ):
        """
        Infer piping connections between entities based on:
        1. Spatial proximity (nearby entities are likely connected)
        2. Horizontal/vertical alignment (piping runs are typically straight)
        3. Shared line numbers

        Args:
            entities: List of entities.
            proximity_threshold: Max pixel distance for connection inference.
        """
        import math

        # Sort entities by x-coordinate for left-to-right flow
        positioned = [(e, e.position) for e in entities if e.position is not None]

        for i, (entity_a, pos_a) in enumerate(positioned):
            for j, (entity_b, pos_b) in enumerate(positioned):
                if i >= j:
                    continue

                # Calculate distance
                distance = math.sqrt(
                    (pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2
                )

                if distance > proximity_threshold:
                    continue

                # Check alignment (horizontal or vertical)
                dx = abs(pos_a[0] - pos_b[0])
                dy = abs(pos_a[1] - pos_b[1])

                # Entities roughly aligned horizontally or vertically
                # are more likely connected by piping
                alignment_score = 1.0
                if dy < 30:  # Horizontal alignment
                    alignment_score = 1.5
                elif dx < 30:  # Vertical alignment
                    alignment_score = 1.3

                # Connection weight (lower = stronger connection)
                weight = distance / alignment_score

                # Determine connection type
                conn_type = "piping"
                if entity_a.entity_type == "instrument" or entity_b.entity_type == "instrument":
                    conn_type = "signal" if distance < 80 else "piping"

                # Shared line number strengthens connection
                shared_line = None
                if entity_a.line_number and entity_a.line_number == entity_b.line_number:
                    weight *= 0.5  # Halve weight for same-line entities
                    shared_line = entity_a.line_number

                # Add edge (direction: left-to-right or top-to-bottom)
                if pos_a[0] <= pos_b[0]:
                    source, target = entity_a.entity_id, entity_b.entity_id
                else:
                    source, target = entity_b.entity_id, entity_a.entity_id

                self.graph.add_edge(
                    source,
                    target,
                    weight=round(weight, 2),
                    distance=round(distance, 2),
                    connection_type=conn_type,
                    line_number=shared_line,
                )

    # ---- Graph Query Methods ----

    def get_entities_by_type(self, entity_type: str) -> List[dict]:
        """Get all entities of a specific type (valve, instrument, equipment)."""
        result = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("entity_type") == entity_type:
                result.append({"node_id": node_id, **data})
        return result

    def get_entities_by_category(self, category: str) -> List[dict]:
        """Get all entities in a category (valves, instruments, equipment, piping)."""
        result = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("category") == category:
                result.append({"node_id": node_id, **data})
        return result

    def get_connected_entities(self, entity_id: str) -> List[dict]:
        """Get all entities directly connected to a given entity."""
        if entity_id not in self.graph:
            return []

        neighbors = []
        for neighbor_id in nx.all_neighbors(self.graph, entity_id):
            data = dict(self.graph.nodes[neighbor_id])
            data["node_id"] = neighbor_id

            # Get edge data
            if self.graph.has_edge(entity_id, neighbor_id):
                edge_data = self.graph[entity_id][neighbor_id]
            else:
                edge_data = self.graph[neighbor_id][entity_id]
            data["connection"] = edge_data

            neighbors.append(data)

        return neighbors

    def get_entities_on_line(self, line_number: str) -> List[dict]:
        """Get all entities associated with a specific line number."""
        result = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("line_number") == line_number:
                result.append({"node_id": node_id, **data})
        return result

    def get_graph_summary(self) -> dict:
        """Get a summary of the graph structure."""
        type_counts = {}
        category_counts = {}

        for _, data in self.graph.nodes(data=True):
            etype = data.get("entity_type", "unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1

            cat = data.get("category", "other")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "entity_types": type_counts,
            "categories": category_counts,
            "is_connected": nx.is_weakly_connected(self.graph)
            if self.graph.number_of_nodes() > 0
            else False,
            "connected_components": nx.number_weakly_connected_components(self.graph)
            if self.graph.number_of_nodes() > 0
            else 0,
        }

    def to_serializable(self) -> dict:
        """Convert the graph to a JSON-serializable format."""
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            node_data = {k: v for k, v in data.items() if k != "entity"}
            node_data["id"] = node_id
            nodes.append(node_data)

        edges = []
        for source, target, data in self.graph.edges(data=True):
            edge_data = dict(data)
            edge_data["source"] = source
            edge_data["target"] = target
            edges.append(edge_data)

        return {"nodes": nodes, "edges": edges}

    def export_for_visualization(self) -> dict:
        """Export graph data formatted for pyvis/streamlit-agraph visualization."""
        vis_nodes = []
        vis_edges = []

        # Color map by category
        color_map = {
            "valves": "#4CAF50",       # Green
            "instruments": "#FF9800",  # Orange
            "equipment": "#F44336",    # Red
            "piping": "#2196F3",       # Blue
            "other": "#9E9E9E",        # Gray
        }

        # Size map by type
        size_map = {
            "equipment": 30,
            "valve": 20,
            "instrument": 20,
            "piping_component": 15,
            "line": 10,
            "unknown": 12,
        }

        for node_id, data in self.graph.nodes(data=True):
            category = data.get("category", "other")
            entity_type = data.get("entity_type", "unknown")
            label = data.get("tag_id") or data.get("description", node_id)

            vis_nodes.append({
                "id": node_id,
                "label": str(label),
                "color": color_map.get(category, "#9E9E9E"),
                "size": size_map.get(entity_type, 12),
                "title": f"{data.get('description', '')} | Conf: {data.get('confidence', 0):.2f}",
                "group": category,
            })

        for source, target, data in self.graph.edges(data=True):
            conn_type = data.get("connection_type", "piping")
            vis_edges.append({
                "from": source,
                "to": target,
                "color": "#00BCD4" if conn_type == "signal" else "#607D8B",
                "width": 2 if conn_type == "piping" else 1,
                "dashes": conn_type == "signal",
                "title": f"Type: {conn_type} | Distance: {data.get('distance', '?')}px",
            })

        return {"nodes": vis_nodes, "edges": vis_edges}
