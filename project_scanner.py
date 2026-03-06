"""
project_scanner.py — C++ project knowledge builder.

Scans every C++ source file in a project using libclang and extracts
rich semantic context that flowchart_engine.py uses when prompting the LLM:

  - Function signatures + Doxygen / inline comments
  - Enum declarations with all values and per-value comments
  - #define macro definitions with values and comments
  - typedef / using type aliases with underlying types and comments

Run this ONCE per project (or whenever the project changes) to build
project_knowledge.json, then pass it to flowchart_engine.py via
--knowledge-json.

Usage:
    python project_scanner.py \\
        --project-dir /path/to/cpp-project \\
        --std         c++17 \\
        --clang-arg="-I/path/to/includes" \\
        --out         project_knowledge.json \\
        [--project-name poseidonos] \\
        [--extensions  .cpp,.h,.hpp,.cc,.cxx] \\
        [--exclude-dirs build,third_party,external,test] \\
        [--verbose]
"""

import argparse
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
#   PARSE_SKIP_FUNCTION_BODIES is intentionally NOT included.
#   With that flag, libclang marks out-of-line method definitions as
#   non-definitions, so is_definition() returns False for methods defined
#   in .cpp files — which would cause the scanner to miss them entirely.
_PARSE_OPTIONS = (
    ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
    | ci.TranslationUnit.PARSE_INCOMPLETE
)

_DEFAULT_EXTENSIONS = {".cpp", ".h", ".hpp", ".cc", ".cxx", ".hh", ".h++"}


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
        # IMPORTANT: use _norm_path (abspath, no symlink resolution) so the
        # path we pass to index.parse() is the exact same string that
        # libclang will put in cursor.location.file.name.
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
                  len(knowledge.macros), len(knowledge.typedefs))

        self._traverse(tu.cursor, lines, abs_path, rel_path,
                       knowledge, visited=set(), depth=0)

        after = (len(knowledge.functions), len(knowledge.enums),
                 len(knowledge.macros), len(knowledge.typedefs))
        added = tuple(a - b for a, b in zip(after, before))

        if self._verbose or any(x > 0 for x in added):
            logger.debug("  %-55s  +func=%-3d +enum=%-3d +macro=%-3d +typedef=%-3d",
                         rel_path, *added)

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
                # No is_definition() filter — we want both .h declarations
                # and .cpp definitions for the knowledge base.
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

        fk = FunctionKnowledge(
            qualified_name=qname,
            signature=sig,
            file=rel_path,
            line=cursor.location.line,
            comment=comment,
        )

        existing = knowledge.functions.get(qname)
        if existing is None:
            knowledge.functions[qname] = fk
        elif not existing.comment and comment:
            # Upgrade to version that has a doc comment
            knowledge.functions[qname] = fk
        elif not existing.comment and cursor.is_definition():
            # Upgrade to the actual definition (has full signature)
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

        # Skip function-like macros  e.g. #define MAX(a,b) ...
        if value.startswith("("):
            return
        # Skip empty macros and include guards  e.g. #define BLOCK_MANAGER_H_
        if not value or re.match(r"^[A-Z_]+_H[_H]?$", name):
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

    save_knowledge(knowledge, args.out)

    logger.info("=" * 60)
    logger.info("Done. Output: %s", args.out)
    logger.info("=" * 60)
