"""
Mermaid flowchart builder.

Converts a labeled ControlFlowGraph into a Mermaid `flowchart TD` script.

Rules (DO NOT CHANGE):
  - Structural truth (edges, topology) comes only from the CFG.
  - All Mermaid-unsafe characters in labels are escaped with #NNN; entity codes.
  - <br/> line breaks within labels are preserved.
  - START/END nodes use stadium shape  ([...])
  - DECISION / LOOP_HEAD / SWITCH_HEAD nodes use diamond  {...}
  - All other nodes use rectangle  [...]
  - CATCH nodes use subroutine shape  [[...]]
  - Node/edge labels are NOT wrapped in double-quotes, to avoid \" sequences
    when the Mermaid script is stored in a JSON output file.
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
    """
    Render a single Mermaid node definition line.

    Labels are NOT wrapped in double-quotes.  Wrapping in " causes the
    Mermaid script stored in JSON to contain \" (JSON-escaped quotes),
    which breaks mermaid.live when users copy-paste the raw value.
    All special characters are already encoded with #NNN; entity codes
    by _escape_label(), so no delimiters are needed.
    """
    label = node.label or node.raw_code[:60] or node.node_id
    label = _enforce_line_length(label)
    escaped = _escape_label(label)

    nid = node.node_id
    t = node.node_type

    if t in _STADIUM_TYPES:
        return f"{nid}([{escaped}])"
    if t in _DIAMOND_TYPES:
        return f"{nid}{{{escaped}}}"
    if t in _SUBROUTINE_TYPES:
        return f"{nid}[[{escaped}]]"
    return f"{nid}[{escaped}]"


# ---------------------------------------------------------------------------
# Edge definition rendering
# ---------------------------------------------------------------------------

def _edge_def(edge: CfgEdge) -> str:
    """
    Render a Mermaid edge with an optional label.

    Edge labels are NOT wrapped in double-quotes for the same reason as
    node labels — JSON encoding would turn them into \" sequences.
    All special characters are already #NNN;-encoded.
    """
    norm_label = normalize_edge_label(edge.label)
    if norm_label:
        escaped = _escape_edge_label(norm_label)
        return f"{edge.source} -->|{escaped}| {edge.target}"
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

# Single-pass character maps for Mermaid entity encoding.
#
# Sequential str.replace() calls cause double-encoding: e.g. "(" → "#40;" and
# then ";" → "#59;", which turns "#40;" into "#40#59;".  Using re.sub() with a
# single compiled pattern ensures every original character is encoded exactly
# once, regardless of the order entries appear in the map.

_NODE_LABEL_CHAR_MAP: dict = {
    "&":  "#38;",    # ampersand
    '"':  "#quot;",  # double quote
    "[":  "#91;",    # left square bracket
    "]":  "#93;",    # right square bracket
    "(":  "#40;",    # left parenthesis
    ")":  "#41;",    # right parenthesis
    "{":  "#123;",   # left curly brace
    "}":  "#125;",   # right curly brace
    "|":  "#124;",   # pipe / vertical bar
    ";":  "#59;",    # semicolon
    "%":  "#37;",    # percent
    "^":  "#94;",    # caret
    "~":  "#126;",   # tilde
    "<":  "#60;",    # less-than  (applied after protecting <br/>)
    ">":  "#62;",    # greater-than
}

_EDGE_LABEL_CHAR_MAP: dict = {
    "&":  "#38;",
    '"':  "#quot;",
    "|":  "#124;",
    "[":  "#91;",
    "]":  "#93;",
    "(":  "#40;",
    ")":  "#41;",
    "{":  "#123;",
    "}":  "#125;",
    ";":  "#59;",
    "<":  "#60;",
    ">":  "#62;",
}

# Pre-compiled single-pass patterns — each original character is matched and
# replaced exactly once.
_NODE_LABEL_RE  = re.compile("[&\"\\[\\](){}|;%^~<>]")
_EDGE_LABEL_RE  = re.compile("[&\"\\[\\](){}|;<>]")

_BR_PLACEHOLDER = "\x00BR\x00"

# Maximum characters per visual line in a Mermaid node label.
# Mermaid's SVG renderer does NOT word-wrap within a line; it only breaks
# at explicit <br/> tags.  If a line is too long it extends beyond the node
# boundary and wraps awkwardly when the browser wraps the SVG text element.
# This limit is applied mechanically AFTER the LLM generates labels so that
# even labels that violate the system-prompt guideline are fixed before render.
_MAX_LINE_CHARS = 40


def _enforce_line_length(label: str, max_chars: int = _MAX_LINE_CHARS) -> str:
    """
    Ensure no <br/>-separated segment exceeds max_chars characters.

    Algorithm:
      1. Split label on existing <br/> tags.
      2. For each segment, if it is longer than max_chars, word-wrap it:
         break at the last space that keeps the current chunk ≤ max_chars,
         inserting a <br/> at each break point.
      3. Rejoin all segments with <br/>.

    Words longer than max_chars are never broken mid-word to avoid
    corrupting function names or technical terms.
    """
    if not label:
        return label

    # Normalise <br /> → <br/>
    label = label.replace("<br />", "<br/>")
    segments = label.split("<br/>")

    result_segments: List[str] = []
    for seg in segments:
        seg = seg.strip()
        if len(seg) <= max_chars:
            result_segments.append(seg)
            continue

        # Word-wrap this segment
        words = seg.split(" ")
        current_line: List[str] = []
        current_len = 0
        wrapped: List[str] = []

        for word in words:
            # +1 for the space before the word (except at start of line)
            addition = len(word) + (1 if current_line else 0)
            if current_line and current_len + addition > max_chars:
                wrapped.append(" ".join(current_line))
                current_line = [word]
                current_len = len(word)
            else:
                current_line.append(word)
                current_len += addition

        if current_line:
            wrapped.append(" ".join(current_line))

        result_segments.extend(wrapped)

    return "<br/>".join(result_segments)


def _escape_label(text: str) -> str:
    """
    Escape all Mermaid-unsafe characters in a node label using #NNN; codes.

    - <br/> line-break tags are preserved verbatim.
    - Newline characters (\\n) are converted to <br/> for multi-line labels.
    - A single-pass re.sub() is used so that entity semicolons introduced by
      earlier replacements are never re-encoded (no double-encoding).
    """
    if not text:
        return text

    # Protect <br/> line-break tags
    text = text.replace("<br/>", _BR_PLACEHOLDER)
    text = text.replace("<br />", _BR_PLACEHOLDER)

    # Convert real newlines to <br/>
    text = text.replace("\n", _BR_PLACEHOLDER)
    text = text.replace("\r", "")

    # Single-pass entity encoding — each original character replaced exactly once
    text = _NODE_LABEL_RE.sub(lambda m: _NODE_LABEL_CHAR_MAP[m.group(0)], text)

    # Restore line-break placeholders
    text = text.replace(_BR_PLACEHOLDER, "<br/>")

    return text


def _escape_edge_label(text: str) -> str:
    """
    Escape characters in Mermaid edge labels using #NNN; codes.

    Single-pass substitution to prevent double-encoding of semicolons.
    """
    if not text:
        return text
    return _EDGE_LABEL_RE.sub(lambda m: _EDGE_LABEL_CHAR_MAP[m.group(0)], text)
