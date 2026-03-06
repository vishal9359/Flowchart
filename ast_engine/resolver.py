"""
Function cursor resolver.

Finds the libclang cursor for a specific C++ function definition inside
a parsed TranslationUnit using:
  1. File path matching
  2. Extent containment (start/end line)
  3. Spelling/displayname matching
  4. Best-fit scoring with tie-breaking
"""

import logging
import re
from typing import List, Optional, Tuple

import clang.cindex as ci

from models import FunctionEntry

logger = logging.getLogger(__name__)

# Cursor kinds that represent function definitions
_FUNCTION_KINDS = frozenset({
    ci.CursorKind.FUNCTION_DECL,
    ci.CursorKind.CXX_METHOD,
    ci.CursorKind.CONSTRUCTOR,
    ci.CursorKind.DESTRUCTOR,
    ci.CursorKind.FUNCTION_TEMPLATE,
    ci.CursorKind.CONVERSION_FUNCTION,
})


def find_function_cursor(tu: ci.TranslationUnit,
                         func_entry: FunctionEntry) -> Optional[ci.Cursor]:
    """
    Locate the cursor for func_entry inside the TU.
    Returns the best-matching cursor, or None if not found.
    """
    target_file = _normalise_path(func_entry.file)
    target_line = func_entry.line
    target_end = func_entry.end_line
    simple_name = _simple_name(func_entry.qualified_name)

    candidates: List[Tuple[int, ci.Cursor]] = []

    def _visit(cursor: ci.Cursor) -> None:
        if cursor.kind in _FUNCTION_KINDS:
            loc = cursor.location
            if loc.file and _normalise_path(loc.file.name).endswith(target_file):
                ext = cursor.extent
                if ext.start.line <= target_line and ext.end.line >= target_end:
                    score = _score(cursor, simple_name, target_line, target_end)
                    candidates.append((score, cursor))

        # Recurse — but skip included system files to stay fast
        for child in cursor.get_children():
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

def _score(cursor: ci.Cursor, target_simple: str,
           target_line: int, target_end: int) -> int:
    score = 0

    spelling = cursor.spelling or cursor.displayname or ""
    if spelling == target_simple:
        score += 100
    elif target_simple.lower() in spelling.lower():
        score += 40

    ext = cursor.extent
    score -= abs(ext.start.line - target_line)
    score -= abs(ext.end.line - target_end)

    if cursor.is_definition():
        score += 20

    return score


def _simple_name(qualified_name: str) -> str:
    """Extract simple function name: 'ns::A::foo<T>' → 'foo'."""
    name = qualified_name.split("<")[0]
    parts = name.split("::")
    return parts[-1] if parts else qualified_name


def _normalise_path(path: str) -> str:
    return path.replace("\\", "/").lower()
