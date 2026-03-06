"""
Mermaid flowchart builder.

Converts a labeled ControlFlowGraph into a Mermaid `flowchart TD` script.

Rules (DO NOT CHANGE):
  - Structural truth (edges, topology) comes only from the CFG.
  - All Mermaid-unsafe characters in labels are escaped.
  - <br/> line breaks within labels are preserved.
  - START/END nodes use stadium shape  ([...])
  - DECISION / LOOP_HEAD / SWITCH_HEAD nodes use diamond  {...}
  - All other nodes use rectangle  [...]
  - CATCH nodes use subroutine shape  [[...]]
  - Edge labels are quoted to prevent Mermaid parsing errors.
"""

import re
from typing import Dict, List, Optional

from mermaid.normalizer import normalize_edge_label
from models import CfgEdge, CfgNode, ControlFlowGraph, NodeType


# ---------------------------------------------------------------------------
# Shape mapping
# ---------------------------------------------------------------------------

_DIAMOND_TYPES = frozenset({
    NodeType.DECISION,
    NodeType.LOOP_HEAD,
    NodeType.SWITCH_HEAD,
})

_STADIUM_TYPES = frozenset({
    NodeType.START,
    NodeType.END,
})

_SUBROUTINE_TYPES = frozenset({
    NodeType.CATCH,
})


def build_mermaid(cfg: ControlFlowGraph) -> str:
    """
    Render a ControlFlowGraph as a Mermaid flowchart TD script.
    Returns the complete Mermaid string.
    """
    lines: List[str] = ["flowchart TD"]

    # Node definitions
    for node in _topo_order(cfg):
        lines.append(f"    {_node_def(node)}")

    lines.append("")  # blank line between defs and edges

    # Edge definitions
    for edge in cfg.edges:
        lines.append(f"    {_edge_def(edge)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node definition rendering
# ---------------------------------------------------------------------------

def _node_def(node: CfgNode) -> str:
    """Render a single Mermaid node definition line."""
    label = node.label or node.raw_code[:60] or node.node_id
    escaped = _escape_label(label)

    nid = node.node_id
    t = node.node_type

    if t in _STADIUM_TYPES:
        return f'{nid}(["{escaped}"])'
    if t in _DIAMOND_TYPES:
        return f'{nid}{{"{escaped}"}}'
    if t in _SUBROUTINE_TYPES:
        return f'{nid}[["{escaped}"]]'
    return f'{nid}["{escaped}"]'


# ---------------------------------------------------------------------------
# Edge definition rendering
# ---------------------------------------------------------------------------

def _edge_def(edge: CfgEdge) -> str:
    """Render a Mermaid edge, with optional label."""
    norm_label = normalize_edge_label(edge.label)
    if norm_label:
        escaped = _escape_edge_label(norm_label)
        return f'{edge.source} -->|"{escaped}"| {edge.target}'
    return f"{edge.source} --> {edge.target}"


# ---------------------------------------------------------------------------
# Topological ordering (deterministic: entry node first, then BFS)
# ---------------------------------------------------------------------------

def _topo_order(cfg: ControlFlowGraph) -> List[CfgNode]:
    """
    Return nodes in a stable order: entry first, then BFS.
    Deterministic output (no random set iteration).
    """
    if not cfg.nodes:
        return []

    adjacency: Dict[str, List[str]] = {nid: [] for nid in cfg.nodes}
    for edge in cfg.edges:
        if edge.source in adjacency:
            adjacency[edge.source].append(edge.target)

    visited: List[str] = []
    seen = set()
    queue = [cfg.entry_node_id] if cfg.entry_node_id in cfg.nodes else [
        next(iter(cfg.nodes))
    ]

    while queue:
        nid = queue.pop(0)
        if nid in seen:
            continue
        seen.add(nid)
        if nid in cfg.nodes:
            visited.append(nid)
        for child in adjacency.get(nid, []):
            if child not in seen:
                queue.append(child)

    # Append any nodes not reached by BFS (isolated or back-edge-only nodes)
    for nid in cfg.nodes:
        if nid not in seen:
            visited.append(nid)

    return [cfg.nodes[nid] for nid in visited]


# ---------------------------------------------------------------------------
# Label escaping
# ---------------------------------------------------------------------------

def _escape_label(text: str) -> str:
    """
    Escape characters that break Mermaid node label syntax.
    The label will be wrapped in double quotes, so we must escape " inside.
    <br/> line breaks are preserved verbatim.
    """
    if not text:
        return text

    # Temporarily protect <br/> tags
    text = text.replace("<br/>", "\x00BR\x00")
    text = text.replace("<br />", "\x00BR\x00")

    # Escape double quotes
    text = text.replace('"', "#quot;")

    # Remove characters that are structurally dangerous in Mermaid
    # even inside quotes: { } [ ] ( ) | are mostly safe inside quotes
    # but angle brackets need entity encoding
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    # Restore <br/> placeholders
    text = text.replace("\x00BR\x00", "<br/>")

    return text


def _escape_edge_label(text: str) -> str:
    """Escape edge label text (simpler — no <br/> needed)."""
    text = text.replace('"', "#quot;")
    text = text.replace("|", "&#124;")
    return text
