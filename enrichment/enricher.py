"""
Node enricher.

Annotates every CfgNode in a ControlFlowGraph with project-specific
context drawn from the PKB:
  - Called function signatures / descriptions for function-call nodes
  - Variable type meanings for declarations
  - Source-level inline comments near the node's lines
  - Enum/macro hints extracted from raw_code tokens

This enriched context is later injected into the LLM prompt so that
generated labels are accurate and project-vocabulary-consistent.
"""

import logging
import re
from typing import Dict, List, Optional

from models import CfgNode, ControlFlowGraph, FunctionEntry, NodeType
from pkb.builder import ProjectKnowledgeBase
from pkb.knowledge import ProjectKnowledge

logger = logging.getLogger(__name__)

# Patterns that identify a function call in raw C++ code
_CALL_PATTERN = re.compile(r'(\w[\w:~<>]*)\s*\(')
# Patterns for simple variable declarations: type varname = ...
_DECL_PATTERN = re.compile(r'(\w[\w:*&<> ]+?)\s+(\w+)\s*[=;{(]')


class NodeEnricher:
    """
    Annotates CFG nodes with PKB context.

    Each node's enriched_context dict is filled with:
      "function_calls": [{signature, description}, ...]
      "inline_comment": str
    """

    def __init__(self, pkb: ProjectKnowledgeBase,
                 source_lines_by_file: Dict[str, List[str]],
                 knowledge: Optional[ProjectKnowledge] = None) -> None:
        self._pkb = pkb
        self._src: Dict[str, List[str]] = source_lines_by_file
        self._knowledge: Optional[ProjectKnowledge] = knowledge

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(self, cfg: ControlFlowGraph,
               func_entry: FunctionEntry) -> None:
        """Enrich all nodes in the CFG in-place."""
        src_lines = self._src.get(func_entry.file, [])
        for node in cfg.nodes.values():
            node.enriched_context = self._enrich_node(node, func_entry, src_lines)

    # ------------------------------------------------------------------
    # Per-node enrichment
    # ------------------------------------------------------------------

    def _enrich_node(self, node: CfgNode,
                     func_entry: FunctionEntry,
                     src_lines: List[str]) -> Dict:
        ctx: Dict = {}

        # Skip structural sentinel nodes
        if node.node_type in (NodeType.START, NodeType.END,
                               NodeType.BREAK, NodeType.CONTINUE):
            return ctx

        # Function calls present in this node's raw code
        calls = self._resolve_calls(node.raw_code, func_entry.calls_ids)
        if calls:
            ctx["function_calls"] = calls

        # Inline source comment near this node's lines
        comment = self._nearest_comment(src_lines, node.start_line)
        if comment:
            ctx["inline_comment"] = comment

        # Enum constant meanings found in raw_code (from project knowledge)
        if self._knowledge:
            enum_ctx = self._lookup_enums(node.raw_code)
            if enum_ctx:
                ctx["enum_context"] = enum_ctx

            macro_ctx = self._lookup_macros(node.raw_code)
            if macro_ctx:
                ctx["macro_context"] = macro_ctx

            typedef_ctx = self._lookup_typedefs(node.raw_code)
            if typedef_ctx:
                ctx["typedef_context"] = typedef_ctx

        return ctx

    # ------------------------------------------------------------------
    # Called-function resolution
    # ------------------------------------------------------------------

    def _resolve_calls(self, raw_code: str,
                       calls_ids: List[str]) -> List[Dict]:
        """
        Find which function calls from calls_ids appear in raw_code.
        Returns list of {signature, description} for matched callees.
        """
        # Extract simple names used in this code snippet
        names_in_code = set(_CALL_PATTERN.findall(raw_code))
        if not names_in_code:
            return []

        result = []
        for cid in calls_ids:
            entry = self._pkb.get(cid)
            if not entry:
                continue
            simple = entry.qualified_name.split("::")[-1].split("<")[0]
            if simple in names_in_code:
                param_str = ", ".join(
                    f"{p.get('type','')} {p.get('name','')}".strip()
                    for p in entry.params
                )
                result.append({
                    "signature": f"{entry.qualified_name}({param_str})",
                    "description": entry.description or "",
                })
        return result

    # ------------------------------------------------------------------
    # Source comment extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _nearest_comment(lines: List[str], target_line: int) -> str:
        """
        Find the nearest inline or preceding comment for a source line.
        Looks at the target line itself, then scans upward up to 3 lines.
        """
        if not lines:
            return ""

        idx = target_line - 1  # convert to 0-based

        # Inline comment on the same line
        if 0 <= idx < len(lines):
            inline = _extract_inline_comment(lines[idx])
            if inline:
                return inline

        # Preceding comment block (up to 3 lines above, skip blank lines)
        i = idx - 1
        comment_lines: List[str] = []
        while i >= 0 and i >= idx - 4:
            stripped = lines[i].strip()
            if not stripped:
                i -= 1
                continue
            if stripped.startswith("//"):
                comment_lines.insert(0, stripped.lstrip("/ "))
                i -= 1
            elif stripped.endswith("*/") or stripped.startswith("*"):
                raw = stripped.lstrip("*/ ").rstrip("*/ ")
                if raw:
                    comment_lines.insert(0, raw)
                i -= 1
            else:
                break

        return " ".join(comment_lines).strip()


    # ------------------------------------------------------------------
    # Project knowledge lookups
    # ------------------------------------------------------------------

    def _lookup_enums(self, raw_code: str) -> List[str]:
        """Find enum types and their values referenced in raw_code."""
        if not self._knowledge:
            return []
        results: List[str] = []
        tokens = set(re.findall(r'\b([A-Za-z_]\w*)\b', raw_code))
        seen: set = set()
        for token in tokens:
            for enum_name, ek in self._knowledge.enums.items():
                if enum_name in seen:
                    continue
                # Match if token is the enum type name, or a value of this enum
                if token == enum_name or token in ek.values:
                    seen.add(enum_name)
                    results.append(f"{enum_name} (enum): {ek.summary()}")
                    break
        return results

    def _lookup_macros(self, raw_code: str) -> List[str]:
        """Find macro constants referenced in raw_code."""
        if not self._knowledge:
            return []
        results: List[str] = []
        tokens = set(re.findall(r'\b([A-Z_][A-Z0-9_]{2,})\b', raw_code))
        for token in tokens:
            mk = self._knowledge.macros.get(token)
            if mk:
                results.append(mk.summary())
        return results

    def _lookup_typedefs(self, raw_code: str) -> List[str]:
        """Find typedef/using names referenced in raw_code."""
        if not self._knowledge:
            return []
        results: List[str] = []
        tokens = set(re.findall(r'\b([A-Za-z_]\w*)\b', raw_code))
        for token in tokens:
            tk = self._knowledge.typedefs.get(token)
            if tk:
                results.append(tk.summary())
        return results


def _extract_inline_comment(line: str) -> str:
    """Extract a trailing // comment from a source line."""
    # Skip string literals to avoid false positives
    in_str = False
    i = 0
    while i < len(line) - 1:
        if line[i] == '"' and (i == 0 or line[i - 1] != "\\"):
            in_str = not in_str
        if not in_str and line[i] == "/" and line[i + 1] == "/":
            return line[i + 2:].strip()
        i += 1
    return ""
