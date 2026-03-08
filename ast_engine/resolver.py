"""
Function cursor resolver.

Finds the libclang cursor for a specific C++ function definition inside
a parsed TranslationUnit using a two-strategy approach:

  Strategy 1 — Direct position lookup (fast, parse-error-resilient)
    Uses Cursor.from_location() to jump straight to (file, start_line)
    then walks up through semantic parents until a function-kind cursor
    is found.  This succeeds even when the TU has diagnostic errors
    (missing headers, unresolved types) because it does not depend on
    cursor.location.file being non-None.

  Strategy 2 — AST traversal (comprehensive fallback)
    Walks the AST collecting every function-kind cursor whose extent
    contains the target range.  Two file-matching sub-strategies are
    used in order:
      a) Normal: cursor.location.file matches target file path.
      b) Null-file fallback: cursor.location.file is None — this happens
         when libclang error-recovery creates a cursor for a method whose
         owning class could not be resolved (e.g. a missing header).
         We still accept these cursors if their extent closely matches.
    System header subtrees are skipped for performance.
"""

import logging
from typing import List, Optional, Tuple

import clang.cindex as ci

from models import FunctionEntry

logger = logging.getLogger(__name__)

# Cursor kinds that represent function / method definitions.
_FUNCTION_KINDS = frozenset({
    ci.CursorKind.FUNCTION_DECL,
    ci.CursorKind.CXX_METHOD,
    ci.CursorKind.CONSTRUCTOR,
    ci.CursorKind.DESTRUCTOR,
    ci.CursorKind.FUNCTION_TEMPLATE,
    ci.CursorKind.CONVERSION_FUNCTION,
})

# How many lines of "slop" we allow when accepting a null-file cursor.
_NULL_FILE_SLOP = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_function_cursor(tu: ci.TranslationUnit,
                         func_entry: FunctionEntry,
                         abs_path: Optional[str] = None) -> Optional[ci.Cursor]:
    """
    Locate the cursor for func_entry inside the TU.

    Parameters
    ----------
    tu          : libclang TranslationUnit (must be parsed with bodies).
    func_entry  : target function metadata (file, line, end_line, qualifiedName).
    abs_path    : absolute path of the source file that was passed to
                  index.parse().  Providing this enables Strategy 1 and
                  improves Strategy 2 path matching.

    Returns the best-matching cursor, or None if not found.
    """
    target_file = _normalise_path(func_entry.file)
    target_line = func_entry.line
    target_end  = func_entry.end_line
    simple_name = _simple_name(func_entry.qualified_name)

    # ------------------------------------------------------------------
    # Strategy 1: direct position-based lookup via Cursor.from_location()
    # ------------------------------------------------------------------
    if abs_path:
        hit = _position_lookup(tu, abs_path, target_line, target_end, simple_name)
        if hit is not None:
            logger.debug("Resolved '%s' via position lookup → '%s'",
                         func_entry.qualified_name, hit.spelling)
            return hit

    # ------------------------------------------------------------------
    # Strategy 2: full AST traversal (with null-file fallback)
    # ------------------------------------------------------------------
    norm_abs = _normalise_path(abs_path) if abs_path else None
    candidates: List[Tuple[int, ci.Cursor]] = []

    def _visit(cursor: ci.Cursor) -> None:
        if cursor.kind in _FUNCTION_KINDS:
            loc = cursor.location
            ext = cursor.extent

            # Sub-strategy 2a: normal file-path match
            file_match = False
            if loc.file is not None:
                n = _normalise_path(loc.file.name)
                file_match = (
                    n.endswith(target_file)
                    or (norm_abs is not None and n == norm_abs)
                )

            # Sub-strategy 2b: null-file fallback
            # libclang sets loc.file = None when error-recovery constructs a
            # cursor whose owning class is unresolved (missing header).
            # Accept if the extent is very close to our target range.
            null_file_match = (
                not file_match
                and loc.file is None
                and ext.start.line > 0
                and abs(ext.start.line - target_line) <= _NULL_FILE_SLOP
                and abs(ext.end.line   - target_end)  <= _NULL_FILE_SLOP
            )

            if file_match or null_file_match:
                if ext.start.line <= target_line and ext.end.line >= target_end:
                    score = _score(cursor, simple_name, target_line, target_end)
                    if null_file_match:
                        score -= 10  # small penalty for uncertain file
                    candidates.append((score, cursor))

        # Recurse into children, but skip system-header subtrees entirely.
        # (The comment in the original code promised this but never implemented
        # it — fixed here.)
        for child in cursor.get_children():
            try:
                child_loc = child.location
                if child_loc.file is not None and child_loc.is_in_system_header:
                    continue
            except Exception:
                pass
            _visit(child)

    _visit(tu.cursor)

    if not candidates:
        logger.warning("No cursor found for '%s' (%s:%d-%d)",
                       func_entry.qualified_name, func_entry.file,
                       target_line, target_end)
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best = candidates[0]
    logger.debug("Resolved '%s' → cursor '%s' (score %d)",
                 func_entry.qualified_name, best.spelling, best_score)
    return best


def get_function_body(cursor: ci.Cursor) -> Optional[ci.Cursor]:
    """Return the COMPOUND_STMT body of a function cursor, or None."""
    for child in cursor.get_children():
        if child.kind == ci.CursorKind.COMPOUND_STMT:
            return child
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _position_lookup(tu: ci.TranslationUnit,
                     abs_path: str,
                     target_line: int,
                     target_end: int,
                     simple_name: str) -> Optional[ci.Cursor]:
    """
    Try to find the function cursor by jumping directly to (abs_path, line)
    using Cursor.from_location(), then walking up semantic parents.

    We probe several lines near the function start so that we succeed even
    if the exact first line is a template parameter list, preprocessor macro,
    or other non-cursor position.
    """
    try:
        tu_file = ci.File.from_name(tu, abs_path)
        if tu_file is None:
            return None

        probe_lines = [target_line, target_line + 1, target_line + 2]
        for line in probe_lines:
            try:
                loc = ci.SourceLocation.from_position(tu, tu_file, line, 1)
                cursor = ci.Cursor.from_location(tu, loc)
            except Exception:
                continue

            if cursor is None:
                continue
            try:
                if cursor.kind.is_invalid():
                    continue
            except Exception:
                continue

            # Walk up through semantic parents to find the enclosing function.
            c = cursor
            depth = 0
            while c is not None and depth < 20:
                depth += 1
                try:
                    if c.kind in _FUNCTION_KINDS:
                        ext = c.extent
                        # Accept if the function starts within a few lines of
                        # the target (allowing for leading annotations/macros)
                        if (ext.start.line <= target_line + 3
                                and ext.end.line >= target_end - 1):
                            return c
                    if c.kind == ci.CursorKind.TRANSLATION_UNIT:
                        break
                    parent = c.semantic_parent
                    if parent is None or parent == c:
                        break
                    c = parent
                except Exception:
                    break
    except Exception as exc:
        logger.debug("Position lookup failed for %s: %s", abs_path, exc)

    return None


def _score(cursor: ci.Cursor, target_simple: str,
           target_line: int, target_end: int) -> int:
    """Score a candidate cursor — higher is better."""
    score = 0

    spelling = cursor.spelling or cursor.displayname or ""
    if spelling == target_simple:
        score += 100
    elif target_simple.lower() in spelling.lower():
        score += 40

    ext = cursor.extent
    score -= abs(ext.start.line - target_line)
    score -= abs(ext.end.line   - target_end)

    try:
        if cursor.is_definition():
            score += 20
    except Exception:
        pass

    return score


def _simple_name(qualified_name: str) -> str:
    """Extract simple function name: 'ns::A::foo<T>' → 'foo'."""
    name = qualified_name.split("<")[0]
    parts = name.split("::")
    return parts[-1] if parts else qualified_name


def _normalise_path(path: str) -> str:
    return path.replace("\\", "/").lower()
