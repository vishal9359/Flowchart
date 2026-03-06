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
#
# Mermaid uses #NNN; (decimal ASCII code) entity syntax inside node labels.
# All characters that can confuse Mermaid's parser — even inside double-quoted
# labels — must be replaced with their #NNN; equivalents so that a script can
# be pasted directly into mermaid.live without parse errors.
#
# Reference (from user examples):
#   store[prg index]  →  store#91;prg index#93;
#   back(info[index]) →  back#40;info#91;index#93;#41;
# ---------------------------------------------------------------------------

# Characters that must be replaced with Mermaid #NNN; entity codes.
# Order matters: & must come first to avoid double-encoding existing entities.
_NODE_LABEL_ESCAPES = [
    ("&",  "#38;"),    # ampersand — must be first
    ('"',  "#quot;"),  # double quote (Mermaid's own named entity)
    ("[",  "#91;"),    # left square bracket
    ("]",  "#93;"),    # right square bracket
    ("(",  "#40;"),    # left parenthesis
    (")",  "#41;"),    # right parenthesis
    ("{",  "#123;"),   # left curly brace
    ("}",  "#125;"),   # right curly brace
    ("|",  "#124;"),   # pipe / vertical bar
    (";",  "#59;"),    # semicolon
    ("%",  "#37;"),    # percent
    ("^",  "#94;"),    # caret
    ("~",  "#126;"),   # tilde
    # < and > are applied AFTER protecting <br/> below
]

_ANGLE_ESCAPES = [
    ("<",  "#60;"),
    (">",  "#62;"),
]

# Edge-label characters — edge labels are simpler (no <br/> line breaks)
_EDGE_LABEL_ESCAPES = [
    ("&",  "#38;"),
    ('"',  "#quot;"),
    ("|",  "#124;"),
    ("[",  "#91;"),
    ("]",  "#93;"),
    ("(",  "#40;"),
    (")",  "#41;"),
    ("{",  "#123;"),
    ("}",  "#125;"),
    (";",  "#59;"),
    ("<",  "#60;"),
    (">",  "#62;"),
]

_BR_PLACEHOLDER = "\x00BR\x00"
_NL_PLACEHOLDER = "\x00NL\x00"


def _escape_label(text: str) -> str:
    """
    Escape all Mermaid-unsafe characters in a node label using #NNN; codes.

    - <br/> line-break tags are preserved verbatim (not escaped).
    - Newline characters (\\n) are converted to <br/> for multi-line labels.
    - All other special characters are replaced with their #NNN; equivalents
      so the output can be pasted into mermaid.live without errors.
    """
    if not text:
        return text

    # Protect <br/> line-break tags
    text = text.replace("<br/>", _BR_PLACEHOLDER)
    text = text.replace("<br />", _BR_PLACEHOLDER)

    # Convert real newlines to <br/>
    text = text.replace("\n", _BR_PLACEHOLDER)
    text = text.replace("\r", "")

    # Apply character escapes
    for char, entity in _NODE_LABEL_ESCAPES:
        text = text.replace(char, entity)

    # Escape angle brackets (safe now that <br/> is protected)
    for char, entity in _ANGLE_ESCAPES:
        text = text.replace(char, entity)

    # Restore line-break placeholders
    text = text.replace(_BR_PLACEHOLDER, "<br/>")

    return text


def _escape_edge_label(text: str) -> str:
    """Escape characters in Mermaid edge labels using #NNN; codes."""
    if not text:
        return text
    for char, entity in _EDGE_LABEL_ESCAPES:
        text = text.replace(char, entity)
    return text
