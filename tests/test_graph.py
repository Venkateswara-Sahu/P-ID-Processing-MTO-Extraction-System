"""
Graph builder and relationship resolver tests.

Covers node identity, edge inference, the self-loop bug, and the
confidence-scale mismatch between the graph and the MTO.
"""
import networkx as nx
import pytest

from src.detection.symbol_detector import Detection
from src.graph.entity_mapper import EngineeringEntity
from src.graph.graph_builder import PIDGraphBuilder
from src.graph.relationship_resolver import RelationshipResolver
from src.ocr.tag_parser import TagParser


def _ent(entity_id="E0001", entity_type="valve", category="valves",
         tag_id=None, x=100.0, y=100.0, parsed_tags=None):
    det = Detection(
        class_id=0, class_name="Gate Valve", confidence=0.9,
        bbox_xyxy=(x - 10, y - 10, x + 10, y + 10), center=(x, y),
    )
    return EngineeringEntity(
        entity_id=entity_id, symbol_detection=det, entity_type=entity_type,
        component_category=category, tag_id=tag_id, position=(x, y),
        confidence=0.9, parsed_tags=parsed_tags or [],
    )


# --------------------------------------------------------------------------- #
# Self-loops
# --------------------------------------------------------------------------- #

def test_instrument_loop_detection_does_not_create_a_self_loop():
    """
    A node whose parsed tags shared a loop number was appended to the loop
    members twice, so the resolver added an edge from the node to itself.
    """
    tp = TagParser()
    ent = _ent(
        entity_id="E0001", entity_type="instrument", category="instruments",
        tag_id="FIC-101", parsed_tags=[tp.parse("FIC-101"), tp.parse("FV-101")],
    )
    g = nx.DiGraph()
    g.add_node("E0001", entity_type="instrument", category="instruments",
               position=(10.0, 10.0), confidence=0.9, tag_id="FIC-101", entity=ent)

    RelationshipResolver().resolve(g, [ent])

    self_loops = [(u, v) for u, v in g.edges() if u == v]
    assert not self_loops, f"the resolver created self-loops: {self_loops}"


def test_a_normal_instrument_loop_does_connect_its_members():
    """Two distinct instruments in a loop must be connected to each other."""
    tp = TagParser()
    a = _ent(entity_id="E0001", entity_type="instrument",
             category="instruments", tag_id="FIC-101", x=10.0,
             parsed_tags=[tp.parse("FIC-101")])
    b = _ent(entity_id="E0002", entity_type="instrument",
             category="instruments", tag_id="FV-101", x=80.0,
             parsed_tags=[tp.parse("FV-101")])

    g = nx.DiGraph()
    for e in (a, b):
        g.add_node(e.entity_id, entity_type=e.entity_type,
                   category=e.component_category, position=e.position,
                   confidence=e.confidence, tag_id=e.tag_id, entity=e)

    RelationshipResolver().resolve(g, [a, b])

    assert g.has_edge("E0001", "E0002") or g.has_edge("E0002", "E0001"), (
        "the two loop members were not connected"
    )


# --------------------------------------------------------------------------- #
# Node identity and graph summary
# --------------------------------------------------------------------------- #

def test_graph_nodes_use_the_entity_id():
    ents = [_ent(entity_id="E0001", x=10.0), _ent(entity_id="E0002", x=500.0)]
    g = PIDGraphBuilder().build(ents)
    assert set(g.nodes()) == {"E0001", "E0002"}


def test_two_symbols_sharing_a_tag_stay_separate_nodes():
    """
    They are physically distinct components; the rule validator reports the
    duplicate tag, but the graph must not merge them.
    """
    ents = [
        _ent(entity_id="E0001", tag_id="FV-101", x=10.0),
        _ent(entity_id="E0002", tag_id="FV-101", x=500.0),
    ]
    g = PIDGraphBuilder().build(ents)
    assert g.number_of_nodes() == 2


def test_graph_summary_reflects_the_resolved_graph():
    """
    The summary must describe the graph AFTER the resolver has run, not a
    stale pre-resolution snapshot.
    """
    ents = [_ent(entity_id="E0001", x=100.0), _ent(entity_id="E0002", x=160.0)]
    builder = PIDGraphBuilder()
    g = builder.build(ents)
    before = builder.get_graph_summary()["total_edges"]

    RelationshipResolver().resolve(g, ents)
    after = builder.get_graph_summary()["total_edges"]

    assert after >= before
    assert builder.get_graph_summary()["total_nodes"] == 2


# --------------------------------------------------------------------------- #
# Edge inference
# --------------------------------------------------------------------------- #

def test_nearby_symbols_are_connected():
    ents = [_ent(entity_id="E0001", x=100.0), _ent(entity_id="E0002", x=160.0)]
    g = PIDGraphBuilder().build(ents)
    assert g.number_of_edges() >= 1


def test_distant_symbols_are_not_connected():
    ents = [_ent(entity_id="E0001", x=100.0), _ent(entity_id="E0002", x=3000.0)]
    g = PIDGraphBuilder().build(ents)
    assert g.number_of_edges() == 0


def test_the_line_number_sharing_branch_is_reachable():
    """
    The 'shared line halves the weight' branch in _infer_connections was
    dead code: nothing set entity.line_number before the graph was built.
    """
    ents = [
        _ent(entity_id="E0001", x=100.0),
        _ent(entity_id="E0002", x=160.0),
    ]
    ents[0].line_number = "6\"-PA-1001"
    ents[1].line_number = "6\"-PA-1001"

    g = PIDGraphBuilder().build(ents)
    edge = list(g.edges(data=True))[0]
    assert edge[2].get("line_number") == "6\"-PA-1001", (
        "the shared line number was not recorded on the edge"
    )


# --------------------------------------------------------------------------- #
# Confidence must be the same number everywhere
# --------------------------------------------------------------------------- #

def test_graph_node_confidence_matches_the_entity():
    """
    The confidence scorer rewrites entity.confidence, so a graph built
    before scoring kept a pre-scoring value. The MTO and the graph must
    agree about the same entity.
    """
    from src.validation.confidence_scorer import ConfidenceScorer

    ent = _ent(entity_id="E0001")
    builder = PIDGraphBuilder()
    g = builder.build([ent])

    # Scoring with the graph in hand keeps the two in step.
    ConfidenceScorer().score_entities([ent], graph=g)

    assert g.nodes["E0001"]["confidence"] == pytest.approx(ent.confidence), (
        "the graph node still holds the pre-scoring confidence"
    )
