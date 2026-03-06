"""
Mermaid flowchart validator.

Validates a ControlFlowGraph and its rendered Mermaid script for
structural correctness before writing output.

Checks (DO NOT CHANGE):
  - START node exists and is reachable
  - END node exists
  - No node references an undefined node_id in edges
  - No empty labels on non-sentinel nodes
  - No node is completely unreachable (warning, not error)
  - Mermaid script contains the flowchart keyword
"""

import logging
import re
from typing import List, Tuple

from models import ControlFlowGraph, NodeType

logger = logging.getLogger(__name__)


class ValidationResult:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:
        parts = []
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARN:  {w}")
        return "\n".join(parts) if parts else "  OK"


def validate_cfg(cfg: ControlFlowGraph) -> ValidationResult:
    """Validate the CFG structure."""
    result = ValidationResult()

    if not cfg.nodes:
        result.errors.append("CFG has no nodes")
        return result

    # Check START node exists
    start_nodes = [n for n in cfg.nodes.values()
                   if n.node_type == NodeType.START]
    if not start_nodes:
        result.errors.append("No START node found")
    elif len(start_nodes) > 1:
        result.warnings.append(f"Multiple START nodes: {[n.node_id for n in start_nodes]}")

    # Check END node exists
    end_nodes = [n for n in cfg.nodes.values()
                 if n.node_type == NodeType.END]
    if not end_nodes:
        result.errors.append("No END node found")

    # Validate edge references
    all_ids = set(cfg.nodes.keys())
    for edge in cfg.edges:
        if edge.source not in all_ids:
            result.errors.append(f"Edge source '{edge.source}' is not a known node")
        if edge.target not in all_ids:
            result.errors.append(f"Edge target '{edge.target}' is not a known node")

    # Check for empty labels (skip START/END which have fixed labels)
    skip_types = {NodeType.START, NodeType.END}
    for node in cfg.nodes.values():
        if node.node_type not in skip_types and not node.label.strip():
            result.warnings.append(f"Node {node.node_id} ({node.node_type.value}) has empty label")

    # Check reachability from entry
    if cfg.entry_node_id:
        reachable = _reachable(cfg)
        unreachable = all_ids - reachable
        if unreachable:
            result.warnings.append(f"Unreachable nodes: {sorted(unreachable)}")

    return result


def validate_mermaid(script: str) -> ValidationResult:
    """Validate a rendered Mermaid script for basic syntax."""
    result = ValidationResult()

    if not script.strip():
        result.errors.append("Mermaid script is empty")
        return result

    if not script.strip().startswith("flowchart"):
        result.errors.append("Mermaid script does not start with 'flowchart'")

    # Check for unmatched quotes (crude but catches common escaping errors)
    for i, line in enumerate(script.splitlines(), 1):
        # Count unescaped double quotes outside of node defs
        # Simple heuristic: odd count of " on a line is suspicious
        dq_count = line.count('"') - line.count('#quot;') * 0
        if dq_count % 2 != 0:
            result.warnings.append(f"Possible unmatched quote on line {i}: {line.strip()[:60]}")

    return result


def _reachable(cfg: ControlFlowGraph) -> set:
    """BFS from entry node; returns set of reachable node IDs."""
    visited = set()
    queue = [cfg.entry_node_id]
    adj = {}
    for edge in cfg.edges:
        adj.setdefault(edge.source, []).append(edge.target)

    while queue:
        nid = queue.pop()
        if nid in visited:
            continue
        visited.add(nid)
        queue.extend(adj.get(nid, []))
    return visited
