"""
Project Knowledge Base (PKB) builder.

Builds a structured lookup of all functions in functions.json.
Provides context packets for LLM label generation so every function's
LLM call includes:

  1. Hierarchical project context   (project → module → file → function)
  2. 4-level callee call-graph      (BFS up to depth 4, project-only, deduplicated)
  3. Parameter type resolution      (enum + typedef meanings from project_knowledge)
"""

import logging
import re
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from models import FunctionEntry
from pkb.knowledge import FunctionKnowledge, ProjectKnowledge

logger = logging.getLogger(__name__)

# Maximum number of callee entries injected per depth level in the prompt.
# Prevents the context window from being overwhelmed on wide call graphs.
_MAX_CALLEES_PER_LEVEL = 10

# BFS depth limit for callee traversal
_CALLEE_BFS_DEPTH = 4


class ProjectKnowledgeBase:
    """
    In-memory index of all project functions.

    Provides:
    - Fast lookup by key or qualifiedName
    - Context packets for LLM prompts (hierarchy + callee graph + types)
    - Source-level comment extraction fallback for non-PKB callees
    """

    def __init__(self) -> None:
        self._functions: Dict[str, FunctionEntry] = {}
        self._by_qualified_name: Dict[str, List[str]] = {}
        self._knowledge: Optional[ProjectKnowledge] = None
        # Short-name → FunctionKnowledge (built lazily in load_project_knowledge)
        # Enables O(1) lookup in build_targeted_callee_context instead of O(n) scan.
        self._knowledge_by_short_name: Dict[str, "FunctionKnowledge"] = {}

    def load_project_knowledge(self, knowledge: ProjectKnowledge) -> None:
        """Attach a ProjectKnowledge (from project_scanner.py) for richer context."""
        self._knowledge = knowledge
        # Build short-name index: last segment after '::' → FunctionKnowledge
        # On collision (two functions with the same short name), keep the first
        # match (arbitrary but stable). build_targeted_callee_context falls back
        # to exact qualified-name lookup, so collisions rarely matter.
        self._knowledge_by_short_name = {}
        for qname, fk in knowledge.functions.items():
            short = qname.split("::")[-1].split("(")[0].strip()
            if short and short not in self._knowledge_by_short_name:
                self._knowledge_by_short_name[short] = fk
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

    def get_function_phases(self, func_entry: FunctionEntry) -> List[Dict]:
        """Return phase breakdown list for a function (empty if not available)."""
        if not self._knowledge:
            return []
        fk = self._knowledge.functions.get(func_entry.qualified_name)
        return fk.phases if fk and fk.phases else []

    # ------------------------------------------------------------------
    # Context packet construction (for LLM prompt injection)
    # ------------------------------------------------------------------

    def build_base_context_packet(self, func_entry: FunctionEntry,
                                   base_path: str) -> str:
        """
        Static portion of the context packet — same for all batches of a function.

        Includes:
          1. Project hierarchy context  (project → module → file)
          2. Caller context             (who calls this function and why)
          3. Parameters + type meanings
          4. Function purpose
          5. Function execution phases

        Does NOT include callee BFS — that is done per-batch via
        build_targeted_callee_context() so only relevant callees are injected.
        """
        sections: List[str] = []

        # ── 1. Hierarchical context ────────────────────────────────────
        hierarchy = self._build_hierarchy_context(func_entry)
        if hierarchy:
            sections.append(hierarchy)

        # ── 2. Caller context (1-level upward from calledByIds) ────────
        caller = self._build_caller_context(func_entry)
        if caller:
            sections.append(caller)

        # ── 3. Parameters ─────────────────────────────────────────────
        if func_entry.params:
            param_desc = ", ".join(
                f"{p.get('type', '')} {p.get('name', '')}".strip()
                for p in func_entry.params
            )
            sections.append(f"Parameters: {param_desc}")

            type_context = self._resolve_param_types(func_entry.params)
            if type_context:
                sections.append("Parameter type context:")
                for tc in type_context:
                    sections.append(f"  {tc}")

        # ── 4. Function purpose ────────────────────────────────────────
        purpose = ""
        if self._knowledge:
            fk = self._knowledge.functions.get(func_entry.qualified_name)
            if fk and fk.comment:
                purpose = fk.comment
        if not purpose and func_entry.description:
            purpose = func_entry.description
        if purpose:
            sections.append(f"Purpose: {purpose}")

        # ── 5. Function phases (logical sections) ─────────────────────
        phases = self.get_function_phases(func_entry)
        if phases:
            phase_lines = ["Function execution phases:"]
            for i, phase in enumerate(phases, 1):
                sl = phase.get("start_line", "?")
                el = phase.get("end_line", "?")
                desc = phase.get("description", "")
                phase_lines.append(f"  Phase {i} (lines {sl}–{el}): {desc}")
            sections.append("\n".join(phase_lines))

        return "\n".join(sections)

    def build_targeted_callee_context(self, func_entry: FunctionEntry,
                                       callee_names: Set[str]) -> str:
        """
        Build callee context for only the functions actually called in a batch.

        Looks up each name in the knowledge base (exact match then short-name
        fallback).  Much more focused than the broad 4-level BFS — prevents
        injecting dozens of unrelated callees when only 2–3 are relevant.

        Returns an empty string if no project callees match.
        """
        if not callee_names or not self._knowledge:
            return ""

        entries: List[str] = []
        seen: Set[str] = set()

        for name in callee_names:
            # 1. Exact qualified-name match
            fk = self._knowledge.functions.get(name)
            if not fk:
                # 2. Short-name fallback — O(1) via pre-built index
                short = name.split("::")[-1].split("(")[0].strip()
                fk = self._knowledge_by_short_name.get(short)

            if fk and fk.qualified_name not in seen:
                seen.add(fk.qualified_name)
                entries.append(_format_callee_entry(_make_callee_info(fk.qualified_name, fk)))

        if not entries:
            return ""

        return "\n=== Relevant Called Functions ===\n" + "\n".join(entries)

    def build_context_packet(self, func_entry: FunctionEntry,
                              base_path: str) -> str:
        """
        Full context packet including 4-level BFS callee graph.

        Kept for backward compatibility.  The label generator now uses
        build_base_context_packet() + build_targeted_callee_context() instead,
        which gives more focused, per-batch callee context.
        """
        base = self.build_base_context_packet(func_entry, base_path)
        callee = self._build_callee_bfs_context(func_entry, base_path)
        if callee:
            return base + "\n" + callee
        return base

    # ------------------------------------------------------------------
    # Caller context builder  (1-level upward from calledByIds)
    # ------------------------------------------------------------------

    def _build_caller_context(self, func_entry: FunctionEntry) -> str:
        """
        Inject 1-level upward context: which project functions call this one
        and what they are trying to accomplish.

        This gives the LLM the "why" — the higher-level intent of the callers
        — which is critical for labeling helper functions whose internal code
        looks opaque without knowing the calling context.

        Caps at 5 callers to avoid bloating the context packet.
        """
        if not func_entry.called_by_ids:
            return ""

        caller_lines: List[str] = []
        seen: Set[str] = set()

        for cid in func_entry.called_by_ids[:8]:  # scan up to 8, keep first 5 hits
            qname = _callsid_to_qname(cid)
            if not qname or qname in seen:
                continue
            seen.add(qname)

            fk = self._knowledge.functions.get(qname) if self._knowledge else None
            if fk:
                sig = fk.signature or qname
                desc = fk.comment or ""
                line = f"  - {sig}" + (f"  →  {desc}" if desc else "")
            else:
                line = f"  - {qname}"

            caller_lines.append(line)
            if len(caller_lines) >= 5:
                break

        if not caller_lines:
            return ""

        return "Called by (callers of this function):\n" + "\n".join(caller_lines)

    # ------------------------------------------------------------------
    # Hierarchy context builder
    # ------------------------------------------------------------------

    def _build_hierarchy_context(self, func_entry: FunctionEntry) -> str:
        """
        Build the project → module → file hierarchy block.

        Uses summaries stored in ProjectKnowledge (populated by
        project_scanner.py --llm-summarize).  Gracefully omits any
        level whose summary is empty.
        """
        if not self._knowledge:
            return ""

        lines: List[str] = []

        # Project summary
        if self._knowledge.project_summary:
            name = self._knowledge.project_name or "Project"
            lines.append(f"[Project] {name}: {self._knowledge.project_summary}")

        # Module summary — derive module path from function's file
        if func_entry.file and self._knowledge.module_summaries:
            module_path = _parent_dir(func_entry.file)
            module_summary = self._knowledge.module_summaries.get(module_path, "")
            if not module_summary and "/" in module_path:
                # Try parent of parent (e.g. src/qos/detail → src/qos)
                module_summary = self._knowledge.module_summaries.get(
                    _parent_dir(module_path), ""
                )
            if module_summary:
                lines.append(f"[Module] {module_path}/: {module_summary}")

        # File summary
        if func_entry.file and self._knowledge.file_summaries:
            file_summary = self._knowledge.file_summaries.get(func_entry.file, "")
            if file_summary:
                file_name = func_entry.file.split("/")[-1]
                lines.append(f"[File] {file_name}: {file_summary}")

        if not lines:
            return ""

        return "=== Project Context ===\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # 4-level BFS callee context
    # ------------------------------------------------------------------

    def _build_callee_bfs_context(self, func_entry: FunctionEntry,
                                   base_path: str) -> str:
        """
        BFS traversal of the call graph up to _CALLEE_BFS_DEPTH levels.

        Filters to project-only functions (must exist in project_knowledge).
        Deduplicates via a visited set so the same function only appears once.

        Returns a formatted multi-section string for prompt injection,
        or an empty string if no project callees are found.
        """
        if not self._knowledge:
            # Fall back to level-1 only from PKB
            level1 = self._resolve_level1_from_pkb(func_entry, base_path)
            if not level1:
                return ""
            lines = ["\n=== Called Functions Context ==="]
            lines.append("Direct calls:")
            for info in level1:
                lines.append(_format_callee_entry(info))
            return "\n".join(lines)

        # BFS across up to 4 levels
        visited: Set[str] = {func_entry.qualified_name}
        by_level: Dict[int, List[Dict]] = {}

        # Level 1 seed: from functions.json callsIds
        current_qnames: List[str] = []
        for cid in func_entry.calls_ids:
            qname = _callsid_to_qname(cid)
            if qname and qname not in visited:
                fk = self._knowledge.functions.get(qname)
                if fk:
                    info = _make_callee_info(qname, fk)
                    by_level.setdefault(1, []).append(info)
                    visited.add(qname)
                    current_qnames.append(qname)
                    if len(by_level.get(1, [])) >= _MAX_CALLEES_PER_LEVEL:
                        break
                else:
                    # Callee not in project_knowledge — try source fallback for level 1
                    fallback = _extract_callee_from_source(cid, base_path)
                    if fallback:
                        by_level.setdefault(1, []).append(fallback)
                        visited.add(qname)

        # Levels 2–4: follow calls stored in FunctionKnowledge.calls
        for depth in range(2, _CALLEE_BFS_DEPTH + 1):
            next_qnames: List[str] = []
            for qname in current_qnames:
                fk = self._knowledge.functions.get(qname)
                if not fk:
                    continue
                for callee_qname in fk.calls:
                    if callee_qname in visited:
                        continue
                    callee_fk = self._knowledge.functions.get(callee_qname)
                    if not callee_fk:
                        continue  # not a project function — skip
                    info = _make_callee_info(callee_qname, callee_fk)
                    by_level.setdefault(depth, []).append(info)
                    visited.add(callee_qname)
                    next_qnames.append(callee_qname)
                    if len(by_level.get(depth, [])) >= _MAX_CALLEES_PER_LEVEL:
                        break

            if not next_qnames:
                break
            current_qnames = next_qnames

        if not by_level:
            return ""

        lines = ["\n=== Called Functions Context ==="]
        depth_labels = {
            1: "Direct calls",
            2: "Calls made by direct callees (depth 2)",
            3: "Calls at depth 3",
            4: "Calls at depth 4",
        }
        for depth in sorted(by_level.keys()):
            entries = by_level[depth]
            lines.append(f"\n{depth_labels.get(depth, f'Depth {depth}')}:")
            for info in entries:
                lines.append(_format_callee_entry(info))

        return "\n".join(lines)

    def _resolve_level1_from_pkb(self, func_entry: FunctionEntry,
                                   base_path: str) -> List[Dict]:
        """Level-1 callee resolution without project knowledge (PKB only)."""
        result = []
        for cid in func_entry.calls_ids:
            entry = self._functions.get(cid)
            if entry:
                param_str = ", ".join(
                    f"{p.get('type','')} {p.get('name','')}".strip()
                    for p in entry.params
                )
                description = entry.description or ""
                result.append({
                    "signature": f"{entry.qualified_name}({param_str})",
                    "description": description,
                    "file": entry.file,
                })
            else:
                fallback = _extract_callee_from_source(cid, base_path)
                if fallback:
                    result.append(fallback)
        return result

    # ------------------------------------------------------------------
    # Parameter type resolution
    # ------------------------------------------------------------------

    def _resolve_param_types(self, params: List[Dict]) -> List[str]:
        if not self._knowledge:
            return []
        result = []
        seen: Set[str] = set()
        for param in params:
            raw_type = param.get("type", "")
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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parent_dir(file_path: str) -> str:
    """Return the parent directory path of a relative file path."""
    parts = file_path.replace("\\", "/").split("/")
    if len(parts) <= 1:
        return "."
    return "/".join(parts[:-1])


def _callsid_to_qname(calls_id: str) -> str:
    """
    Extract the qualified function name from a callsId string.
    Format: "src|file|qualified::Name|param,types"
    """
    if "|" in calls_id:
        parts = calls_id.split("|")
        if len(parts) >= 3:
            return parts[2]
    return calls_id


def _make_callee_info(qname: str, fk: FunctionKnowledge) -> Dict:
    """Build a callee info dict from a FunctionKnowledge entry."""
    return {
        "signature": fk.signature or qname,
        "description": fk.comment or "",
        "file": fk.file,
    }


def _format_callee_entry(info: Dict) -> str:
    """Format one callee info dict as a single prompt line."""
    sig = info.get("signature", "")
    desc = info.get("description", "")
    if desc:
        return f"  - {sig}  →  {desc}"
    return f"  - {sig}"


# ---------------------------------------------------------------------------
# Source-level callee extraction (fallback for unknown functions)
# ---------------------------------------------------------------------------

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


def _find_function_in_source(source: str, simple_name: str) -> Tuple[str, str]:
    """
    Search source text for a function definition matching simple_name.
    Returns (signature_line, preceding_comment) or ("", "").
    """
    lines = source.splitlines()
    for i, line in enumerate(lines):
        if re.search(rf'\b{re.escape(simple_name)}\s*\(', line):
            sig_lines = [line.strip()]
            j = i + 1
            while j < len(lines) and "{" not in "".join(sig_lines):
                sig_lines.append(lines[j].strip())
                j += 1
            signature = " ".join(sig_lines).split("{")[0].strip()
            comment = _extract_preceding_comment(lines, i)
            return signature, comment

    return "", ""


def _extract_preceding_comment(lines: List[str], func_line_idx: int) -> str:
    """Extract the nearest doc comment above a function definition."""
    comment_lines: List[str] = []
    i = func_line_idx - 1

    while i >= 0 and not lines[i].strip():
        i -= 1

    if i < 0:
        return ""

    line = lines[i].strip()

    if line.startswith("//"):
        while i >= 0 and lines[i].strip().startswith("//"):
            comment_lines.insert(0, lines[i].strip().lstrip("/ "))
            i -= 1
        return " ".join(comment_lines)

    if line.endswith("*/"):
        while i >= 0 and "/*" not in lines[i]:
            raw = lines[i].strip().lstrip("* ")
            if raw:
                comment_lines.insert(0, raw)
            i -= 1
        return " ".join(comment_lines)

    return ""
