"""
project_scanner.py — C++ project knowledge builder.

Scans every C++ source file in a project using libclang and extracts
rich semantic context that flowchart_engine.py uses when prompting the LLM:

  - Function signatures + Doxygen / inline comments
  - Function call graph (which project functions each function calls)
  - Enum declarations with all values and per-value comments
  - #define macro definitions with values and comments
  - typedef / using type aliases with underlying types and comments
  - Struct / class member field declarations

Optionally, with --llm-summarize, runs a 4-level LLM summarization pass:
  1. Function level  — one-sentence summary for each undocumented function
  2. File level      — 2-3 sentence description per source file
  3. Module level    — 2-3 sentence description per directory module
  4. Project level   — overall project description (README preferred)

Run this ONCE per project (or whenever the project changes) to build
project_knowledge.json, then pass it to flowchart_engine.py via
--knowledge-json.

Usage (scan only):
    python project_scanner.py \\
        --project-dir /path/to/cpp-project \\
        --std         c++17 \\
        --clang-arg="-I/path/to/includes" \\
        --out         project_knowledge.json

Usage (scan + LLM summarization):
    python project_scanner.py \\
        --project-dir /path/to/cpp-project \\
        --std         c++17 \\
        --clang-arg="-I/path/to/includes" \\
        --out         project_knowledge.json \\
        --llm-summarize \\
        --llm-url     http://localhost:11434/api/generate \\
        --llm-model   gpt-oss
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import clang.cindex as ci

from pkb.knowledge import (
    EnumKnowledge,
    EnumValueKnowledge,
    FunctionKnowledge,
    MacroKnowledge,
    ProjectKnowledge,
    StructKnowledge,
    StructMemberKnowledge,
    TypedefKnowledge,
    save_knowledge,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("project_scanner")

# ---------------------------------------------------------------------------
# Cursor kinds
# ---------------------------------------------------------------------------

_FUNCTION_KINDS = frozenset({
    ci.CursorKind.FUNCTION_DECL,
    ci.CursorKind.CXX_METHOD,
    ci.CursorKind.CONSTRUCTOR,
    ci.CursorKind.DESTRUCTOR,
    ci.CursorKind.FUNCTION_TEMPLATE,
})

_CONTAINER_KINDS = frozenset({
    ci.CursorKind.NAMESPACE,
    ci.CursorKind.CLASS_DECL,
    ci.CursorKind.STRUCT_DECL,
    ci.CursorKind.CLASS_TEMPLATE,
    ci.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
    ci.CursorKind.UNION_DECL,
})

# Parse options:
#   PARSE_DETAILED_PROCESSING_RECORD  — expose MACRO_DEFINITION cursors
#   PARSE_INCOMPLETE                  — tolerate missing headers/types
#
#   PARSE_SKIP_FUNCTION_BODIES is intentionally NOT included so that
#   CALL_EXPR nodes inside bodies are visible for call-graph extraction.
_PARSE_OPTIONS = (
    ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
    | ci.TranslationUnit.PARSE_INCOMPLETE
)

_DEFAULT_EXTENSIONS = {".cpp", ".h", ".hpp", ".cc", ".cxx", ".hh", ".h++"}

# Qualified-name prefixes that are definitely not project functions
_LIBRARY_PREFIXES: Tuple[str, ...] = (
    "std::", "__", "boost::", "spdlog::", "fmt::",
    "google::", "testing::", "gtest::", "absl::",
    "llvm::", "clang::", "folly::",
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _norm_path(p: str) -> str:
    """
    Normalise a file path for comparison.

    Uses os.path.abspath (makes absolute, collapses ..) WITHOUT following
    symlinks.  This is critical: libclang reports cursor file names using
    the same path string that was passed to index.parse().  If we resolve
    symlinks in one place but not the other, every single cursor would be
    silently skipped.
    """
    try:
        return os.path.normcase(
            os.path.normpath(os.path.abspath(str(p)))
        ).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def _is_system_path(p: str) -> bool:
    """Return True if p is a system or compiler-internal header."""
    lp = p.replace("\\", "/").lower()
    return (
        lp.startswith("/usr/")
        or lp.startswith("/lib/")
        or "/clang/" in lp
        or "/c++/v1/" in lp
        or lp.startswith("c:/windows/")
        or lp.startswith("c:/program files/")
    )


def _relative(abs_path: str, base_path: str) -> str:
    """Return a relative path, falling back to abs_path if not under base_path."""
    try:
        return str(Path(abs_path).relative_to(base_path)).replace("\\", "/")
    except ValueError:
        return abs_path.replace("\\", "/")


def _is_library_call(qname: str) -> bool:
    """Return True if qname looks like a standard-library or well-known framework call."""
    return any(qname.startswith(p) for p in _LIBRARY_PREFIXES)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files(project_dir: str,
                   extensions: Set[str],
                   exclude_dirs: Set[str]) -> List[Path]:
    """Walk project_dir and return all C++ files, skipping excluded dirs."""
    result: List[Path] = []
    root = Path(project_dir)

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in exclude_dirs and not d.startswith(".")
        ]
        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix.lower() in extensions:
                result.append(p)

    result.sort()
    logger.info("Discovered %d C++ file(s) in %s", len(result), project_dir)
    return result


# ---------------------------------------------------------------------------
# Comment extraction
# ---------------------------------------------------------------------------

def _preceding_comment(lines: List[str], func_line_idx: int) -> str:
    """
    Extract the doc comment immediately above func_line_idx (0-based).
    Handles // and /* */ styles.
    """
    i = func_line_idx - 1

    while i >= 0 and not lines[i].strip():
        i -= 1

    if i < 0:
        return ""

    line = lines[i].strip()

    if line.startswith("//"):
        comment_lines: List[str] = []
        while i >= 0 and lines[i].strip().startswith("//"):
            comment_lines.insert(0, lines[i].strip().lstrip("/ "))
            i -= 1
        return " ".join(l for l in comment_lines if l)

    if line.endswith("*/"):
        comment_lines = []
        while i >= 0 and "/*" not in lines[i]:
            raw = lines[i].strip().lstrip("* ").rstrip()
            if raw and raw != "*/":
                comment_lines.insert(0, raw)
            i -= 1
        if i >= 0:
            opener = re.sub(r"/\*+\s*", "", lines[i]).strip().rstrip("*/").strip()
            if opener:
                comment_lines.insert(0, opener)
        return " ".join(l for l in comment_lines if l)

    return ""


def _inline_comment(line: str) -> str:
    """Extract a trailing // comment from a single source line."""
    in_str = False
    i = 0
    while i < len(line) - 1:
        if line[i] == '"' and (i == 0 or line[i - 1] != "\\"):
            in_str = not in_str
        if not in_str and line[i] == "/" and line[i + 1] == "/":
            return line[i + 2:].strip()
        i += 1
    return ""


def _safe_line(lines: List[str], line_number: int) -> Optional[str]:
    """Return source line for a 1-based line_number, or None if out of range."""
    idx = line_number - 1
    if 0 <= idx < len(lines):
        return lines[idx]
    return None


def _funclike_macro_body(value: str) -> str:
    """
    Given the text after a function-like macro name  — which starts with
    the parameter list  e.g. '(eventid, ...) logger()->iboflog(...)'  —
    return the expansion body (the part after the closing ')').

    Continuation backslashes and excess whitespace are normalised so the
    result is a compact single-line string suitable for LLM context.
    Returns an empty string if no body follows the parameter list.
    """
    if not value.startswith("("):
        return value
    depth = 0
    for i, ch in enumerate(value):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                body = value[i + 1:]
                # Remove line-continuation backslashes and collapse whitespace
                body = re.sub(r"\\\s*\n\s*", " ", body)
                body = re.sub(r"\s+", " ", body).strip()
                return body[:200]
    return ""


# ---------------------------------------------------------------------------
# Source text extraction
# ---------------------------------------------------------------------------

def _extent_text(lines: List[str],
                 start_line: int, end_line: int,
                 start_col: int, end_col: int) -> str:
    if not lines:
        return ""
    sl = max(1, min(start_line, len(lines)))
    el = max(sl, min(end_line, len(lines)))
    if sl == el:
        return lines[sl - 1][start_col - 1: end_col - 1].strip()
    parts = [lines[sl - 1][start_col - 1:].rstrip()]
    for i in range(sl, el - 1):
        parts.append(lines[i].rstrip())
    parts.append(lines[el - 1][: end_col - 1].rstrip())
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Call-graph extraction helper
# ---------------------------------------------------------------------------

def _collect_calls(func_cursor: ci.Cursor) -> List[str]:
    """
    Traverse a function cursor's body for CALL_EXPR nodes.

    For each call, attempts to resolve the referenced declaration.
    Only includes calls where:
      - The referenced declaration is from a non-system file
      - The qualified name does not match a known library prefix

    Returns a deduplicated list of qualified callee names (project-only).
    """
    calls: List[str] = []
    seen: Set[str] = set()

    def _walk(cursor: ci.Cursor) -> None:
        for child in cursor.get_children():
            if child.kind == ci.CursorKind.CALL_EXPR:
                qname = ""
                try:
                    ref = child.referenced
                    if ref:
                        loc = ref.location
                        if loc.file and not _is_system_path(loc.file.name):
                            qname = ref.spelling or ""
                except Exception:
                    pass

                if qname and qname not in seen and not _is_library_call(qname):
                    seen.add(qname)
                    calls.append(qname)

            # Recurse into all children (but not into nested function definitions)
            if child.kind not in _FUNCTION_KINDS:
                _walk(child)

    _walk(func_cursor)
    return calls


# ---------------------------------------------------------------------------
# Per-file knowledge extractor
# ---------------------------------------------------------------------------

class FileKnowledgeExtractor:
    """
    Parses one source file and extracts all knowledge artifacts.

    Traversal design
    ────────────────
    Container cursors (namespace / class / struct):
        Always recurse — regardless of which file they report as their
        location.  Why: in C++, a namespace reopened in block_manager.cpp
        may look like the one first defined in a header.  libclang might
        report the namespace cursor's location as the header file.
        Filtering containers by file would cause all method definitions
        inside that namespace to be silently skipped.
        System headers are skipped for performance.

    Leaf cursors (function / enum / macro / typedef):
        Only extract when location.file matches the file being scanned.
        This prevents duplication when each file is processed separately.

    is_definition() is NOT used as a gate for functions:
        Method declarations in .h files (is_definition()=False) and
        out-of-line definitions in .cpp files are both valuable.
        Additionally, PARSE_SKIP_FUNCTION_BODIES (not used here) causes
        is_definition() to be unreliable for .cpp definitions.
    """

    def __init__(self, index: ci.Index, std: str,
                 extra_args: List[str],
                 verbose: bool = False) -> None:
        self._index = index
        self._std = std
        self._extra_args = extra_args
        self._verbose = verbose

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def extract(self, file_path: Path,
                knowledge: ProjectKnowledge,
                base_path: str) -> None:
        """Parse file_path and populate knowledge."""
        abs_path = _norm_path(str(file_path))
        rel_path = _relative(abs_path, base_path)

        try:
            lines = file_path.read_text(encoding="utf-8",
                                        errors="replace").splitlines(keepends=True)
        except Exception as exc:
            logger.warning("Cannot read %s: %s", abs_path, exc)
            return

        args = [f"-std={self._std}", "-x", "c++"] + self._extra_args
        try:
            tu = self._index.parse(abs_path, args=args, options=_PARSE_OPTIONS)
        except Exception as exc:
            logger.warning("libclang failed on %s: %s", abs_path, exc)
            return

        if tu is None:
            logger.warning("libclang returned None TU for %s", abs_path)
            return

        if self._verbose:
            diags = [d for d in tu.diagnostics
                     if d.severity >= ci.Diagnostic.Warning]
            if diags:
                logger.debug("  %d diagnostic(s) in %s:", len(diags), rel_path)
                for d in diags[:5]:
                    logger.debug("    [clang] %s", d.spelling)

        before = (len(knowledge.functions), len(knowledge.enums),
                  len(knowledge.macros), len(knowledge.typedefs),
                  len(knowledge.structs))

        self._traverse(tu.cursor, lines, abs_path, rel_path,
                       knowledge, visited=set(), depth=0)

        after = (len(knowledge.functions), len(knowledge.enums),
                 len(knowledge.macros), len(knowledge.typedefs),
                 len(knowledge.structs))
        added = tuple(a - b for a, b in zip(after, before))

        if self._verbose or any(x > 0 for x in added):
            logger.debug(
                "  %-55s  +func=%-3d +enum=%-3d +macro=%-3d +typedef=%-3d +struct=%-3d",
                rel_path, *added,
            )

    # ------------------------------------------------------------------
    # AST traversal
    # ------------------------------------------------------------------

    def _traverse(self, cursor: ci.Cursor, lines: List[str],
                  abs_path: str, rel_path: str,
                  knowledge: ProjectKnowledge,
                  visited: Set[int], depth: int) -> None:
        """Walk the AST, extracting knowledge for the current file."""
        if depth > 30:
            return  # safety cap

        for child in cursor.get_children():
            chash = child.hash
            if chash in visited:
                continue
            visited.add(chash)

            loc = child.location
            kind = child.kind

            # ── Containers: always recurse, skip system headers ───────────
            if kind in _CONTAINER_KINDS:
                if loc.file and _is_system_path(loc.file.name):
                    continue
                self._traverse(child, lines, abs_path, rel_path,
                                knowledge, visited, depth + 1)
                # For struct/class definitions in THIS file, also extract members
                if kind in (ci.CursorKind.CLASS_DECL, ci.CursorKind.STRUCT_DECL):
                    if (loc.file
                            and _norm_path(loc.file.name) == abs_path
                            and child.is_definition()):
                        self._extract_struct(child, lines, rel_path, knowledge)
                continue

            # ── Leaf items: only extract if from THIS file ────────────────
            if not loc.file:
                continue

            child_file = _norm_path(loc.file.name)
            if child_file != abs_path:
                continue

            if loc.line < 1 or loc.line > len(lines):
                continue

            # ── Dispatch ──────────────────────────────────────────────────
            if kind == ci.CursorKind.MACRO_DEFINITION:
                self._extract_macro(child, lines, rel_path, knowledge)

            elif kind == ci.CursorKind.ENUM_DECL and child.is_definition():
                self._extract_enum(child, lines, rel_path, knowledge)

            elif kind in (ci.CursorKind.TYPEDEF_DECL,
                          ci.CursorKind.TYPE_ALIAS_DECL):
                self._extract_typedef(child, lines, rel_path, knowledge)

            elif kind in _FUNCTION_KINDS:
                self._extract_function(child, lines, rel_path, knowledge)

    # ------------------------------------------------------------------
    # Function extraction
    # ------------------------------------------------------------------

    def _extract_function(self, cursor: ci.Cursor, lines: List[str],
                           rel_path: str,
                           knowledge: ProjectKnowledge) -> None:
        qname = cursor.spelling or cursor.displayname
        if not qname:
            return

        try:
            ret = cursor.result_type.spelling if cursor.result_type else ""
        except Exception:
            ret = ""

        params: List[str] = []
        for c in cursor.get_children():
            if c.kind == ci.CursorKind.PARM_DECL:
                ptype = c.type.spelling or ""
                pname = c.spelling or ""
                params.append(f"{ptype} {pname}".strip())

        sig = f"{ret} {qname}({', '.join(params)})".strip()

        line_idx = cursor.location.line - 1
        comment = _preceding_comment(lines, line_idx)
        src_line = _safe_line(lines, cursor.location.line)
        if not comment and src_line:
            comment = _inline_comment(src_line)

        # Collect call graph only from definitions (bodies present)
        calls: List[str] = []
        if cursor.is_definition():
            calls = _collect_calls(cursor)

        fk = FunctionKnowledge(
            qualified_name=qname,
            signature=sig,
            file=rel_path,
            line=cursor.location.line,
            comment=comment,
            calls=calls,
        )

        existing = knowledge.functions.get(qname)
        if existing is None:
            knowledge.functions[qname] = fk
        elif not existing.comment and comment:
            # Upgrade to version that has a doc comment; preserve calls if richer
            fk.calls = fk.calls if fk.calls else existing.calls
            knowledge.functions[qname] = fk
        elif cursor.is_definition() and not existing.calls and calls:
            # Upgrade to add call-graph info from the definition
            existing.calls = calls
        elif cursor.is_definition() and not existing.comment:
            knowledge.functions[qname] = fk

    # ------------------------------------------------------------------
    # Enum extraction
    # ------------------------------------------------------------------

    def _extract_enum(self, cursor: ci.Cursor, lines: List[str],
                       rel_path: str,
                       knowledge: ProjectKnowledge) -> None:
        name = cursor.spelling
        if not name:
            return

        line_idx = cursor.location.line - 1
        comment = _preceding_comment(lines, line_idx)

        ek = EnumKnowledge(
            qualified_name=name,
            file=rel_path,
            comment=comment,
        )

        for child in cursor.get_children():
            if child.kind != ci.CursorKind.ENUM_CONSTANT_DECL:
                continue
            const_name = child.spelling
            const_val = str(child.enum_value)
            src_line = _safe_line(lines, child.location.line)
            c_comment = _inline_comment(src_line) if src_line else ""
            if not c_comment:
                c_comment = _preceding_comment(lines, child.location.line - 1)
            ek.values[const_name] = EnumValueKnowledge(
                value=const_val,
                comment=c_comment,
            )

        if ek.values:
            knowledge.enums[name] = ek
            simple = name.split("::")[-1]
            if simple != name:
                knowledge.enums[simple] = ek

    # ------------------------------------------------------------------
    # Macro extraction
    # ------------------------------------------------------------------

    def _extract_macro(self, cursor: ci.Cursor, lines: List[str],
                        rel_path: str,
                        knowledge: ProjectKnowledge) -> None:
        name = cursor.spelling
        if not name or name.startswith("_"):
            return

        ext = cursor.extent
        if ext.start.line < 1 or ext.start.line > len(lines):
            return
        if ext.end.line < 1 or ext.end.line > len(lines):
            return

        full_text = _extent_text(lines,
                                  ext.start.line, ext.end.line,
                                  ext.start.column, ext.end.column)
        value = full_text[len(name):].strip() if full_text.startswith(name) else full_text

        # Skip include guards  e.g. #define BLOCK_MANAGER_H_
        if re.match(r"^[A-Z_]+_H[_H]?$", name):
            return

        # For function-like macros  e.g. #define TRACE_DEBUG(id, ...)  body...
        if value.startswith("("):
            value = _funclike_macro_body(value)
            if not value:
                return

        if not value:
            return

        src_line = _safe_line(lines, ext.start.line)
        comment = _inline_comment(src_line) if src_line else ""
        if not comment:
            comment = _preceding_comment(lines, ext.start.line - 1)

        knowledge.macros[name] = MacroKnowledge(
            name=name,
            value=value[:120],
            file=rel_path,
            comment=comment,
        )

    # ------------------------------------------------------------------
    # Typedef / using extraction
    # ------------------------------------------------------------------

    def _extract_typedef(self, cursor: ci.Cursor, lines: List[str],
                          rel_path: str,
                          knowledge: ProjectKnowledge) -> None:
        name = cursor.spelling
        if not name:
            return

        underlying = ""
        if cursor.kind == ci.CursorKind.TYPEDEF_DECL:
            underlying = cursor.underlying_typedef_type.spelling or ""
        elif cursor.kind == ci.CursorKind.TYPE_ALIAS_DECL:
            underlying = cursor.type.spelling or ""

        if not underlying or underlying == name:
            return

        line_idx = cursor.location.line - 1
        comment = _preceding_comment(lines, line_idx)
        src_line = _safe_line(lines, cursor.location.line)
        if not comment and src_line:
            comment = _inline_comment(src_line)

        knowledge.typedefs[name] = TypedefKnowledge(
            name=name,
            underlying=underlying,
            file=rel_path,
            comment=comment,
        )

    # ------------------------------------------------------------------
    # Struct / class member extraction
    # ------------------------------------------------------------------

    def _extract_struct(self, cursor: ci.Cursor, lines: List[str],
                         rel_path: str,
                         knowledge: ProjectKnowledge) -> None:
        qname = cursor.type.spelling or cursor.spelling
        if not qname or qname.startswith("("):
            return

        line_idx = cursor.location.line - 1
        struct_comment = _preceding_comment(lines, line_idx)

        members: Dict[str, StructMemberKnowledge] = {}
        for child in cursor.get_children():
            if child.kind != ci.CursorKind.FIELD_DECL:
                continue
            field_name = child.spelling
            if not field_name:
                continue
            field_type = child.type.spelling or ""
            field_line = child.location.line

            comment = ""
            src_line = _safe_line(lines, field_line)
            if src_line:
                comment = _inline_comment(src_line)
            if not comment and field_line > 0:
                comment = _preceding_comment(lines, field_line - 1)

            members[field_name] = StructMemberKnowledge(
                field_type=field_type,
                comment=comment,
            )

        if not members:
            return

        sk = StructKnowledge(
            qualified_name=qname,
            file=rel_path,
            comment=struct_comment,
            members=members,
        )
        knowledge.structs[qname] = sk
        simple = qname.split("::")[-1].split("<")[0].strip()
        if simple and simple != qname:
            knowledge.structs[simple] = sk


# ---------------------------------------------------------------------------
# Hierarchy LLM Summarizer
# ---------------------------------------------------------------------------

class HierarchySummarizer:
    """
    4-level LLM summarization pass.

    After project_scanner.py has completed its structural scan, this class
    runs LLM calls to fill in semantic understanding at every level:
      1. Function  — one-sentence summary for undocumented functions
      2. File      — 2-3 sentence description per source file
      3. Module    — 2-3 sentence description per directory
      4. Project   — overall description (README preferred, else from modules)

    All results are stored back into the ProjectKnowledge object and
    persisted to the output JSON by the caller.

    LLM calls are batched to minimise round-trips:
      - Functions : up to batch_size functions per call
      - Files     : 1 call per file
      - Modules   : 1 call per module directory
      - Project   : 1 call total
    """

    _FUNC_SYSTEM = (
        "You are a C++ code analyst. "
        "For each function below, write ONE concise sentence (max 20 words) "
        "describing what it does. "
        "Return ONLY a JSON object: {\"FunctionName\": \"sentence.\", ...}. "
        "No markdown, no code fences, no extra text."
    )
    _PHASE_SYSTEM = (
        "You are a C++ code analyst. "
        "Break the given function body into 2–6 logical phases (sequential sections). "
        "For each phase, provide:\n"
        "  - start_line: first line number (1 = first line of the body shown)\n"
        "  - end_line: last line number of that phase\n"
        "  - description: concise high-level description (max 12 words) of "
        "what that section achieves — describe intent, not individual statements.\n"
        "Return ONLY a valid JSON array: "
        '[{"start_line": 1, "end_line": 5, "description": "..."}, ...]. '
        "No markdown, no code fences, no extra text."
    )
    _FILE_SYSTEM = (
        "You are a C++ software architect. "
        "Summarize the responsibility of the given source file in 2-3 sentences. "
        "Output only the sentences, no headings or lists."
    )
    _MODULE_SYSTEM = (
        "You are a C++ software architect. "
        "Summarize the purpose of the given software module (directory) in 2-3 sentences. "
        "Output only the sentences, no headings or lists."
    )
    _PROJECT_SYSTEM = (
        "You are a software architect. "
        "Summarize the overall purpose of the given software project in 2-3 sentences. "
        "Output only the sentences, no headings or lists."
    )

    def __init__(self, knowledge: ProjectKnowledge,
                 llm_client,          # LlmClient instance
                 project_dir: str,
                 batch_size: int = 8,
                 verbose: bool = False) -> None:
        self._k = knowledge
        self._client = llm_client
        self._project_dir = project_dir
        self._batch_size = batch_size
        self._verbose = verbose

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def summarize(self) -> None:
        """Run all four summarization levels in order."""
        logger.info("── Hierarchy summarization starting ──")
        self._summarize_functions()
        self._summarize_function_phases()
        self._summarize_files()
        self._summarize_modules()
        self._summarize_project()
        logger.info("── Hierarchy summarization complete ──")

    # ------------------------------------------------------------------
    # Level 1 — Function summaries
    # ------------------------------------------------------------------

    def _summarize_functions(self) -> None:
        # Collect unique unsummarized functions (dedup by qualified_name)
        seen_qnames: Set[str] = set()
        unique: List[FunctionKnowledge] = []
        for fk in self._k.functions.values():
            if not fk.comment and fk.qualified_name not in seen_qnames:
                seen_qnames.add(fk.qualified_name)
                unique.append(fk)

        if not unique:
            logger.info("  All functions already have comments — skipping function summarization")
            return

        logger.info("  Summarizing %d undocumented functions (batch=%d)...",
                    len(unique), self._batch_size)

        done = 0
        for i in range(0, len(unique), self._batch_size):
            batch = unique[i: i + self._batch_size]
            summaries = self._summarize_function_batch(batch)
            for fk in batch:
                key = fk.qualified_name.split("::")[-1]  # short name for JSON key
                summary = summaries.get(key) or summaries.get(fk.qualified_name, "")
                if summary:
                    # Update ALL entries with this qualified name
                    for stored in self._k.functions.values():
                        if stored.qualified_name == fk.qualified_name and not stored.comment:
                            stored.comment = summary.strip().rstrip(".")+ "."
            done += len(batch)
            if done % 50 == 0 or done == len(unique):
                logger.info("  Function summaries: %d/%d", done, len(unique))

    def _summarize_function_batch(self,
                                   batch: List[FunctionKnowledge]) -> Dict[str, str]:
        """Single LLM call for up to batch_size functions. Returns {shortName: summary}."""
        parts = []
        for fk in batch:
            body_lines = self._read_body(fk, max_lines=12)
            short_name = fk.qualified_name.split("::")[-1]
            block = f"Function [{short_name}]:\n  Signature: {fk.signature}\n  File: {fk.file}"
            if body_lines:
                block += "\n  Body (first lines):\n"
                block += "\n".join(f"    {l}" for l in body_lines)
            parts.append(block)

        user_prompt = "\n\n".join(parts)
        user_prompt += (
            "\n\nReturn a JSON object where each key is the function name in "
            "brackets above (the part inside [...]) and the value is a ONE-sentence summary."
        )

        raw = self._client.generate(self._FUNC_SYSTEM, user_prompt)
        if not raw:
            return {}
        return self._parse_json_dict(raw)

    # ------------------------------------------------------------------
    # Level 1b — Function phase breakdown
    # ------------------------------------------------------------------

    # Minimum function body lines to justify phase generation.
    _PHASE_MIN_LINES = 8

    def _summarize_function_phases(self) -> None:
        """
        Generate phase-by-phase breakdowns for functions with enough body lines.

        Phases describe what each logical section of a function achieves at a
        high level (e.g., "Build JSON request body", "Validate and send request").
        Only runs for functions with >= _PHASE_MIN_LINES lines.
        """
        seen_qnames: Set[str] = set()
        eligible: List[FunctionKnowledge] = []
        for fk in self._k.functions.values():
            if fk.qualified_name not in seen_qnames and not fk.phases:
                seen_qnames.add(fk.qualified_name)
                body_lines = self._read_body(fk, max_lines=70)
                if len(body_lines) >= self._PHASE_MIN_LINES:
                    eligible.append(fk)

        if not eligible:
            logger.info("  No functions eligible for phase generation (need %d+ lines)",
                        self._PHASE_MIN_LINES)
            return

        logger.info("  Generating phases for %d function(s) (>= %d lines)...",
                    len(eligible), self._PHASE_MIN_LINES)

        done = 0
        for fk in eligible:
            try:
                phases = self._generate_phases(fk)
                if phases:
                    # Update all entries that share this qualified name
                    for stored in self._k.functions.values():
                        if stored.qualified_name == fk.qualified_name:
                            stored.phases = phases
            except Exception as exc:
                logger.debug("  Phase generation failed for %s: %s",
                             fk.qualified_name, exc)
            done += 1
            if done % 20 == 0 or done == len(eligible):
                logger.info("  Phases: %d/%d", done, len(eligible))

    def _generate_phases(self, fk: FunctionKnowledge) -> List[Dict]:
        """Generate phase breakdown for one function via one LLM call."""
        body_lines = self._read_body(fk, max_lines=60)
        if not body_lines:
            return []

        numbered = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(body_lines))
        prompt = (
            f"Function: {fk.qualified_name}\n"
            f"Signature: {fk.signature}\n\n"
            f"Body (line numbers relative to function start):\n"
            f"{numbered}\n\n"
            "Identify 2–6 logical phases in this function body. "
            "Each phase should describe what that section achieves at a high level."
        )

        raw = self._client.generate(self._PHASE_SYSTEM, prompt)
        if not raw:
            return []

        # Parse JSON array from response
        raw = re.sub(r"```(?:json)?\s*", "", raw)
        raw = re.sub(r"```", "", raw)
        start = raw.find("[")
        if start == -1:
            return []
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "[":
                depth += 1
            elif raw[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        phases = json.loads(raw[start:i + 1])
                        if isinstance(phases, list):
                            return [
                                p for p in phases
                                if isinstance(p, dict)
                                and isinstance(p.get("start_line"), int)
                                and isinstance(p.get("end_line"), int)
                                and isinstance(p.get("description"), str)
                                and p["description"]
                            ]
                    except Exception:
                        pass
                    break
        return []

    # ------------------------------------------------------------------
    # Level 2 — File summaries
    # ------------------------------------------------------------------

    def _summarize_files(self) -> None:
        # Group unique functions by file
        by_file: Dict[str, List[FunctionKnowledge]] = {}
        seen_qnames: Set[str] = set()
        for fk in self._k.functions.values():
            if fk.file and fk.qualified_name not in seen_qnames:
                seen_qnames.add(fk.qualified_name)
                by_file.setdefault(fk.file, []).append(fk)

        logger.info("  Summarizing %d files...", len(by_file))
        for file_path, funcs in sorted(by_file.items()):
            try:
                summary = self._summarize_one_file(file_path, funcs)
                if summary:
                    self._k.file_summaries[file_path] = summary
            except Exception as exc:
                logger.debug("  File summary failed for %s: %s", file_path, exc)

    def _summarize_one_file(self, file_path: str,
                             funcs: List[FunctionKnowledge]) -> str:
        func_lines = []
        for fk in funcs[:30]:
            line = f"  - {fk.signature}"
            if fk.comment:
                line += f": {fk.comment}"
            func_lines.append(line)

        prompt = (
            f"File: {file_path}\n"
            f"Functions ({len(funcs)} total, showing up to 30):\n"
            + "\n".join(func_lines)
            + "\n\nSummarize the responsibility of this file in 2-3 sentences."
        )
        raw = self._client.generate(self._FILE_SYSTEM, prompt)
        return raw.strip()[:600] if raw else ""

    # ------------------------------------------------------------------
    # Level 3 — Module summaries
    # ------------------------------------------------------------------

    def _summarize_modules(self) -> None:
        # Group files by their immediate parent directory (the module)
        by_module: Dict[str, List[str]] = {}
        for file_path in self._k.file_summaries:
            parts = file_path.replace("\\", "/").split("/")
            module = "/".join(parts[:-1]) if len(parts) > 1 else "."
            by_module.setdefault(module, []).append(file_path)

        logger.info("  Summarizing %d modules...", len(by_module))
        for module_path, files in sorted(by_module.items()):
            try:
                summary = self._summarize_one_module(module_path, files)
                if summary:
                    self._k.module_summaries[module_path] = summary
            except Exception as exc:
                logger.debug("  Module summary failed for %s: %s", module_path, exc)

    def _summarize_one_module(self, module_path: str,
                               file_paths: List[str]) -> str:
        file_lines = []
        for fp in file_paths[:20]:
            fs = self._k.file_summaries.get(fp, "")
            line = f"  - {fp}"
            if fs:
                # Use only first sentence to keep prompt compact
                first_sentence = fs.split(".")[0].strip()
                line += f": {first_sentence}"
            file_lines.append(line)

        prompt = (
            f"Module directory: {module_path}\n"
            f"Files ({len(file_paths)} total, showing up to 20):\n"
            + "\n".join(file_lines)
            + "\n\nSummarize the purpose of this module in 2-3 sentences."
        )
        raw = self._client.generate(self._MODULE_SYSTEM, prompt)
        return raw.strip()[:600] if raw else ""

    # ------------------------------------------------------------------
    # Level 4 — Project summary
    # ------------------------------------------------------------------

    def _summarize_project(self) -> None:
        logger.info("  Summarizing project...")

        # Prefer README.md if it exists
        for readme_name in ("README.md", "readme.md", "README.txt", "README"):
            readme_path = Path(self._project_dir) / readme_name
            if readme_path.exists():
                try:
                    readme_text = readme_path.read_text(
                        encoding="utf-8", errors="replace"
                    )[:3000]
                    prompt = (
                        f"Project: {self._k.project_name}\n"
                        f"README:\n{readme_text}\n\n"
                        "Summarize what this project does in 2-3 sentences."
                    )
                    raw = self._client.generate(self._PROJECT_SYSTEM, prompt)
                    if raw and raw.strip():
                        self._k.project_summary = raw.strip()[:600]
                        logger.info("  Project summary generated from README")
                        return
                except Exception as exc:
                    logger.debug("  README read failed: %s", exc)

        # Fall back: aggregate top module summaries
        if not self._k.module_summaries:
            logger.info("  No module summaries available — skipping project summary")
            return

        module_lines = [
            f"  - {mod}: {summary[:100]}"
            for mod, summary in list(self._k.module_summaries.items())[:20]
        ]
        prompt = (
            f"Project: {self._k.project_name}\n"
            f"Modules ({len(self._k.module_summaries)} total, showing up to 20):\n"
            + "\n".join(module_lines)
            + "\n\nSummarize the overall purpose of this project in 2-3 sentences."
        )
        raw = self._client.generate(self._PROJECT_SYSTEM, prompt)
        if raw and raw.strip():
            self._k.project_summary = raw.strip()[:600]
            logger.info("  Project summary generated from module descriptions")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_body(self, fk: FunctionKnowledge, max_lines: int = 12) -> List[str]:
        """Read the first max_lines of a function body from source."""
        try:
            file_path = Path(self._project_dir) / fk.file
            if not file_path.exists():
                return []
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            start_idx = max(0, fk.line - 1)
            end_idx = min(len(all_lines), start_idx + max_lines + 3)
            return [l.rstrip() for l in all_lines[start_idx:end_idx]]
        except Exception:
            return []

    @staticmethod
    def _parse_json_dict(raw: str) -> Dict[str, str]:
        """Extract a JSON object from an LLM response."""
        raw = re.sub(r"```(?:json)?\s*", "", raw)
        raw = re.sub(r"```", "", raw)
        start = raw.find("{")
        if start == -1:
            return {}
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start: i + 1])
                    except Exception:
                        return {}
        return {}


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

def scan_project(project_dir: str,
                 std: str,
                 clang_args: List[str],
                 extensions: Set[str],
                 exclude_dirs: Set[str],
                 project_name: str = "",
                 verbose: bool = False) -> ProjectKnowledge:
    """Scan the entire project and return a ProjectKnowledge object."""
    knowledge = ProjectKnowledge(
        project_name=project_name or Path(project_dir).name,
        base_path=project_dir,
    )

    files = discover_files(project_dir, extensions, exclude_dirs)
    if not files:
        logger.warning("No C++ files found in %s", project_dir)
        return knowledge

    index = ci.Index.create()
    extractor = FileKnowledgeExtractor(index, std, clang_args, verbose=verbose)

    total = len(files)
    for i, file_path in enumerate(files, 1):
        if i % 50 == 0 or i == total:
            logger.info("Scanning [%d/%d] %s", i, total, file_path.name)
        try:
            extractor.extract(file_path, knowledge, project_dir)
        except Exception as exc:
            logger.error("Error scanning %s: %s", file_path, exc, exc_info=verbose)

    logger.info("Scan complete — %s", knowledge.stats())
    return knowledge


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="C++ project knowledge builder for flowchart_engine.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project-dir", required=True,
                   help="Root directory of the C++ project to scan")
    p.add_argument("--std", default="c++17",
                   help="C++ standard (default: c++17)")
    p.add_argument("--clang-arg", dest="clang_args", action="append",
                   default=[], metavar="ARG",
                   help="Extra clang argument (repeatable, e.g. -I/path)")
    p.add_argument("--out", default="project_knowledge.json",
                   help="Output JSON file (default: project_knowledge.json)")
    p.add_argument("--project-name", default="",
                   help="Project name (defaults to directory name)")
    p.add_argument("--extensions", default=".cpp,.h,.hpp,.cc,.cxx,.hh",
                   help="Comma-separated file extensions")
    p.add_argument("--exclude-dirs",
                   default="build,_build,out,dist,third_party,external,"
                           "vendor,test,tests,googletest,gtest",
                   help="Comma-separated directory names to skip")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable debug logging (shows per-file extraction counts)")

    # LLM summarization arguments
    p.add_argument("--llm-summarize", action="store_true",
                   help="Run 4-level LLM summarization after scanning "
                        "(function → file → module → project)")
    p.add_argument("--llm-url", default="http://localhost:11434/api/generate",
                   help="LLM endpoint URL for summarization (default: Ollama local)")
    p.add_argument("--llm-model", default="gpt-oss",
                   help="LLM model name for summarization (default: gpt-oss)")
    p.add_argument("--llm-timeout", type=int, default=120,
                   help="LLM request timeout in seconds (default: 120)")
    p.add_argument("--summarize-batch-size", type=int, default=8,
                   help="Number of functions per LLM summarization call (default: 8)")
    p.add_argument("--use-openai-format", action="store_true",
                   help="Use OpenAI-compatible chat/completions API format")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    extensions = {e.strip() for e in args.extensions.split(",") if e.strip()}
    exclude_dirs = {d.strip() for d in args.exclude_dirs.split(",") if d.strip()}

    if not Path(args.project_dir).exists():
        logger.error("project-dir does not exist: %s", args.project_dir)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("project_scanner starting")
    logger.info("  project-dir  : %s", args.project_dir)
    logger.info("  std          : %s", args.std)
    logger.info("  extensions   : %s", extensions)
    logger.info("  exclude-dirs : %s", exclude_dirs)
    logger.info("  output       : %s", args.out)
    if args.llm_summarize:
        logger.info("  llm-summarize: ON  model=%s  url=%s", args.llm_model, args.llm_url)
    logger.info("=" * 60)

    knowledge = scan_project(
        project_dir=args.project_dir,
        std=args.std,
        clang_args=args.clang_args,
        extensions=extensions,
        exclude_dirs=exclude_dirs,
        project_name=args.project_name,
        verbose=args.verbose,
    )

    # Optional LLM summarization pass
    if args.llm_summarize:
        try:
            from llm.client import LlmClient  # import here to keep scanner standalone
            llm_client = LlmClient(
                url=args.llm_url,
                model=args.llm_model,
                timeout=args.llm_timeout,
                temperature=0.1,
                use_openai_format=args.use_openai_format,
            )
            summarizer = HierarchySummarizer(
                knowledge=knowledge,
                llm_client=llm_client,
                project_dir=args.project_dir,
                batch_size=args.summarize_batch_size,
                verbose=args.verbose,
            )
            summarizer.summarize()
        except ImportError:
            logger.error("Cannot import LlmClient — ensure llm/client.py is accessible")
        except Exception as exc:
            logger.error("LLM summarization failed: %s", exc, exc_info=args.verbose)

    save_knowledge(knowledge, args.out)

    logger.info("=" * 60)
    logger.info("Done. Output: %s", args.out)
    logger.info("=" * 60)
