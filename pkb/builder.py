"""
Project Knowledge Base (PKB) builder.

Builds a structured lookup of all functions in functions.json.
Provides context packets for LLM label generation so every function's
LLM call includes relevant project-specific knowledge.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from models import FunctionEntry
from pkb.knowledge import ProjectKnowledge

logger = logging.getLogger(__name__)


class ProjectKnowledgeBase:
    """
    In-memory index of all project functions.

    Provides:
    - Fast lookup by key or qualifiedName
    - Context packets for LLM prompts (callee signatures, descriptions)
    - Source-level comment extraction fallback for non-PKB callees
    """

    def __init__(self) -> None:
        self._functions: Dict[str, FunctionEntry] = {}
        self._by_qualified_name: Dict[str, List[str]] = {}
        self._knowledge: Optional[ProjectKnowledge] = None

    def load_project_knowledge(self, knowledge: ProjectKnowledge) -> None:
        """Attach a ProjectKnowledge (from project_scanner.py) for richer context."""
        self._knowledge = knowledge
        logger.info("Project knowledge attached to PKB: %s", knowledge.stats())

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, functions_data: Dict) -> None:
        """Populate PKB from the parsed functions.json dict."""
        for key, data in functions_data.items():
            location = data.get("location", {})
            entry = FunctionEntry(
                key=key,
                qualified_name=data.get("qualifiedName", ""),
                file=location.get("file", ""),
                line=location.get("line", 0),
                end_line=location.get("endLine", 0),
                # functions.json uses "parameters"; cache serialisation uses
                # "params".  Accept either so both sources work correctly.
                params=data.get("parameters", data.get("params", [])),
                calls_ids=data.get("callsIds", []),
                called_by_ids=data.get("calledByIds", []),
                interface_id=data.get("interfaceId", ""),
                description=data.get("description", ""),
            )
            self._functions[key] = entry
            self._by_qualified_name.setdefault(entry.qualified_name, []).append(key)

        logger.info("PKB built: %d functions indexed", len(self._functions))

    def to_dict(self) -> Dict:
        """Serialize PKB for disk caching."""
        return {
            key: {
                "qualifiedName": e.qualified_name,
                "file": e.file,
                "line": e.line,
                "endLine": e.end_line,
                "params": e.params,
                "callsIds": e.calls_ids,
                "calledByIds": e.called_by_ids,
                "interfaceId": e.interface_id,
                "description": e.description,
            }
            for key, e in self._functions.items()
        }

    def from_dict(self, data: Dict) -> None:
        """Restore PKB from a cached dict (same shape as to_dict output)."""
        for key, d in data.items():
            entry = FunctionEntry(
                key=key,
                qualified_name=d.get("qualifiedName", ""),
                file=d.get("file", ""),
                line=d.get("line", 0),
                end_line=d.get("endLine", 0),
                params=d.get("params", []),
                calls_ids=d.get("callsIds", []),
                called_by_ids=d.get("calledByIds", []),
                interface_id=d.get("interfaceId", ""),
                description=d.get("description", ""),
            )
            self._functions[key] = entry
            self._by_qualified_name.setdefault(entry.qualified_name, []).append(key)
        logger.info("PKB restored from cache: %d functions", len(self._functions))

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[FunctionEntry]:
        return self._functions.get(key)

    def get_by_qualified_name(self, qualified_name: str) -> Optional[FunctionEntry]:
        keys = self._by_qualified_name.get(qualified_name, [])
        return self._functions[keys[0]] if keys else None

    def all_entries(self) -> List[FunctionEntry]:
        return list(self._functions.values())

    def all_keys(self) -> List[str]:
        return list(self._functions.keys())

    # ------------------------------------------------------------------
    # Context packet construction (for LLM prompt injection)
    # ------------------------------------------------------------------

    def build_context_packet(self, func_entry: FunctionEntry,
                              base_path: str) -> str:
        """
        Construct a concise, LLM-ready context paragraph for a function.

        Priority order for function description:
          1. project_knowledge.json comment (from source — most accurate)
          2. functions.json description field
          3. (omitted)

        Also includes:
        - Parameter types resolved to enum/typedef definitions
        - Signatures + descriptions of all directly called functions
        - Enum / macro context for any types appearing in params
        """
        lines: List[str] = []

        # -- Parameters --
        if func_entry.params:
            param_desc = ", ".join(
                f"{p.get('type', '')} {p.get('name', '')}".strip()
                for p in func_entry.params
            )
            lines.append(f"Parameters: {param_desc}")

            # Resolve enum/typedef meanings for parameter types
            type_context = self._resolve_param_types(func_entry.params)
            if type_context:
                lines.append("Parameter type context:")
                for tc in type_context:
                    lines.append(f"  {tc}")

        # -- Function purpose: prefer scanner comment over functions.json --
        purpose = ""
        if self._knowledge:
            fk = self._knowledge.functions.get(func_entry.qualified_name)
            if fk and fk.comment:
                purpose = fk.comment
        if not purpose and func_entry.description:
            purpose = func_entry.description
        if purpose:
            lines.append(f"Purpose: {purpose}")

        # -- Called functions --
        callees = self._resolve_callees(func_entry.calls_ids, base_path)
        if callees:
            lines.append("\nCalled functions context:")
            for c in callees:
                sig = c["signature"]
                desc = c.get("description", "")
                if desc:
                    lines.append(f"  - {sig}  →  {desc}")
                else:
                    lines.append(f"  - {sig}")

        return "\n".join(lines)

    def _resolve_param_types(self, params: List[Dict]) -> List[str]:
        """
        For each parameter type, look up enum definitions or typedef meanings
        from project knowledge and return human-readable descriptions.
        """
        if not self._knowledge:
            return []
        result = []
        seen = set()
        for param in params:
            raw_type = param.get("type", "")
            # Extract simple type name (strip pointers, refs, templates)
            simple = re.sub(r"[*&<> ].*", "", raw_type.split("::")[-1]).strip()
            if not simple or simple in seen:
                continue
            seen.add(simple)

            enum_k = self._knowledge.enums.get(simple)
            if enum_k:
                result.append(f"{simple} (enum): {enum_k.summary()}")
                continue

            typedef_k = self._knowledge.typedefs.get(simple)
            if typedef_k:
                result.append(f"{simple}: {typedef_k.summary()}")

        return result

    def _resolve_callees(self, calls_ids: List[str],
                         base_path: str) -> List[Dict]:
        """
        Resolve callee context.  Priority:
          1. functions.json entry (has full param list)
          2. project_knowledge.json function comment
          3. Source file parsing fallback
        """
        result = []
        for cid in calls_ids:
            entry = self._functions.get(cid)
            if entry:
                param_str = ", ".join(
                    f"{p.get('type','')} {p.get('name','')}".strip()
                    for p in entry.params
                )
                # Prefer scanner comment over functions.json description
                description = entry.description or ""
                if self._knowledge:
                    fk = self._knowledge.functions.get(entry.qualified_name)
                    if fk and fk.comment:
                        description = fk.comment
                result.append({
                    "signature": f"{entry.qualified_name}({param_str})",
                    "description": description,
                    "file": entry.file,
                })
            else:
                # Callee not in PKB — try project knowledge then source parsing
                if self._knowledge:
                    qname = cid.split("|")[2] if "|" in cid else cid
                    fk = self._knowledge.functions.get(qname)
                    if fk:
                        result.append({
                            "signature": fk.signature,
                            "description": fk.comment,
                            "file": fk.file,
                        })
                        continue
                callee_context = _extract_callee_from_source(cid, base_path)
                if callee_context:
                    result.append(callee_context)
        return result


# ------------------------------------------------------------------
# Source-level callee extraction (fallback for unknown functions)
# ------------------------------------------------------------------

def _extract_callee_from_source(callee_id: str, base_path: str) -> Optional[Dict]:
    """
    Parse the callee id (format: src|file|qualifiedName|params) to locate
    the source file, then extract function signature + nearest comment.
    Returns a context dict or None if not found.
    """
    parts = callee_id.split("|")
    if len(parts) < 3:
        return None

    relative_file_hint = parts[0] + "/" + parts[1]
    qualified_name = parts[2]

    # Try candidate file paths
    candidate_paths = [
        Path(base_path) / f"{relative_file_hint}.cpp",
        Path(base_path) / f"{relative_file_hint}.h",
        Path(base_path) / f"{relative_file_hint}.hpp",
        Path(base_path) / f"{relative_file_hint}.cc",
    ]

    simple_name = qualified_name.split("::")[-1].split("<")[0]

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            sig, comment = _find_function_in_source(content, simple_name)
            if sig:
                return {
                    "signature": sig,
                    "description": comment or "",
                    "file": str(path),
                }
        except Exception:
            continue

    return None


def _find_function_in_source(source: str, simple_name: str):
    """
    Search source text for a function definition matching simple_name.
    Returns (signature_line, preceding_comment) or (None, None).
    """
    lines = source.splitlines()
    for i, line in enumerate(lines):
        # Match function definition: name followed by (
        if re.search(rf'\b{re.escape(simple_name)}\s*\(', line):
            # Extract the signature (may span multiple lines up to {)
            sig_lines = [line.strip()]
            j = i + 1
            while j < len(lines) and "{" not in "".join(sig_lines):
                sig_lines.append(lines[j].strip())
                j += 1
            signature = " ".join(sig_lines).split("{")[0].strip()

            # Look for preceding comment (/** */ or //)
            comment = _extract_preceding_comment(lines, i)
            return signature, comment

    return None, None


def _extract_preceding_comment(lines: List[str], func_line_idx: int) -> str:
    """Extract the nearest doc comment above a function definition."""
    comment_lines = []
    i = func_line_idx - 1

    # Skip blank lines
    while i >= 0 and not lines[i].strip():
        i -= 1

    if i < 0:
        return ""

    line = lines[i].strip()

    # Single-line comment
    if line.startswith("//"):
        while i >= 0 and lines[i].strip().startswith("//"):
            comment_lines.insert(0, lines[i].strip().lstrip("/ "))
            i -= 1
        return " ".join(comment_lines)

    # Block comment end */
    if line.endswith("*/"):
        while i >= 0 and "/*" not in lines[i]:
            raw = lines[i].strip().lstrip("* ")
            if raw:
                comment_lines.insert(0, raw)
            i -= 1
        return " ".join(comment_lines)

    return ""
