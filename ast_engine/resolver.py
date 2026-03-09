"""
Function cursor resolver.

Finds the libclang cursor for a specific C++ function definition inside
a parsed TranslationUnit using a three-strategy approach:

  Strategy 1 — Direct position lookup  (fast, parse-error-resilient)
    Uses Cursor.from_location() to jump straight to (file, start_line)
    and walks up through LEXICAL parents until a function-kind cursor is
    found.  Also handles UNEXPOSED_DECL cursors that arise when the
    owning class is unresolved (missing header).

  Strategy 2 — Full AST traversal  (comprehensive fallback)
    Walks the AST collecting every function-kind cursor, including
    UNEXPOSED_DECL cursors whose spelling matches the target function
    name (error-recovery stand-ins for CXX_METHOD when the class header
    is missing).  Matching uses two independent criteria so that EITHER
    is sufficient:

      Tight match  — cursor extent fully contains [target_line, target_end].
                     Preferred when the AST is complete.

      Loose match  — cursor starts within ±SLOP lines of target_line AND
                     the cursor's spelling contains simple_name.
                     Handles truncated extents caused by parse errors: when
                     libclang encounters an unresolved class (missing header)
                     it still creates a cursor whose extent.start is correct
                     but extent.end stops at the first error inside the body.

    Two file-matching sub-strategies:
      a) Normal: loc.file name matches target file.
      b) Null-file: loc.file is None (happens when the owning class is
         unresolved).  Accepted on start-line proximity alone (no end-line
         check) because end is always wrong in that case.

    System-header subtrees are skipped for performance.

  Strategy 3 — Broad line-range scan  (last-resort fallback)
    When Strategies 1 and 2 find no candidates at all (e.g. clang places
    the method under an UNEXPOSED_DECL with no recognised children), this
    pass accepts ANY cursor kind that:
      • lives in the target file (or has null file with ext.start near the
        target line), AND
      • has ext.start.line within ±BROAD_SLOP of target_line, AND
      • whose spelling contains simple_name.
    A small penalty is applied so Strategy 2 results always win on ties.

Known libclang behaviours addressed here
-----------------------------------------
* When a class header is missing (file-not-found include error) libclang
  may create the method cursor with CursorKind.UNEXPOSED_DECL rather than
  CXX_METHOD because the owning class is unknown.  Strategies 1 and 2
  handle this via _UNEXPOSED_KINDS.

* In some clang versions / error-recovery paths the method cursor has
  loc.file pointing to the *missing header* (not the .cpp being parsed).
  Strategy 3 therefore does NOT hard-reject on file mismatch; it only
  applies a score penalty so correctly-located cursors still win.

* ci.File.from_name(tu, path) requires an exact string match against the
  TU's internal file table.  We try tu.spelling first (the exact path
  libclang used) then abs_path as fallback.

* semantic_parent of a statement jumps to TRANSLATION_UNIT, skipping the
  enclosing function.  We use lexical_parent instead.

* With PARSE_INCOMPLETE + missing headers, a CXX_METHOD cursor's
  extent.end.line can be far less than the real closing-brace line.
  All extent-end checks are therefore treated as soft (scoring only),
  not hard filters.
"""

import logging
from typing import List, Optional, Tuple

import clang.cindex as ci

from models import FunctionEntry

logger = logging.getLogger(__name__)

# Cursor kinds that represent callable definitions.
_FUNCTION_KINDS = frozenset({
    ci.CursorKind.FUNCTION_DECL,
    ci.CursorKind.CXX_METHOD,
    ci.CursorKind.CONSTRUCTOR,
    ci.CursorKind.DESTRUCTOR,
    ci.CursorKind.FUNCTION_TEMPLATE,
    ci.CursorKind.CONVERSION_FUNCTION,
})

# Cursor kinds that may appear as error-recovery stand-ins for a method
# definition when the owning class header is missing.  libclang creates
# UNEXPOSED_DECL in place of CXX_METHOD when the class is unresolved.
_UNEXPOSED_KINDS = frozenset({
    ci.CursorKind.UNEXPOSED_DECL,
})

# Union used wherever we want to consider both real functions and
# error-recovery unexposed declarations.
_ALL_CALLABLE_KINDS = _FUNCTION_KINDS | _UNEXPOSED_KINDS

# Lines of tolerance for "loose" start-line matching.
# Raised from 5 → 10 so that off-by-a-few-lines headers / macros
# in the function signature don't cause a miss.
_START_SLOP = 10

# Extra-wide slop used only in the Strategy-3 broad fallback.
_BROAD_SLOP = 25


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_function_cursor(tu: ci.TranslationUnit,
                         func_entry: FunctionEntry,
                         abs_path: Optional[str] = None) -> Optional[ci.Cursor]:
    """
    Locate the cursor for func_entry inside tu.

    Parameters
    ----------
    tu          : libclang TranslationUnit parsed with function bodies.
    func_entry  : target function metadata.
    abs_path    : absolute path passed to index.parse() for this TU.
                  Providing this enables Strategy 1 and exact path matching.

    Returns the best-matching cursor, or None.
    """
    target_file = _normalise_path(func_entry.file)
    target_line = func_entry.line
    target_end  = func_entry.end_line
    simple_name = _simple_name(func_entry.qualified_name)

    # ------------------------------------------------------------------ #
    #  Strategy 1: direct position lookup — fast and error-resilient      #
    # ------------------------------------------------------------------ #
    if abs_path:
        hit = _position_lookup(tu, abs_path, target_line, target_end,
                               simple_name)
        if hit is not None:
            logger.debug("Resolved '%s' via position lookup → '%s'",
                         func_entry.qualified_name, hit.spelling)
            return hit

    # ------------------------------------------------------------------ #
    #  Strategy 2: full AST traversal                                     #
    # ------------------------------------------------------------------ #
    norm_abs = _normalise_path(abs_path) if abs_path else None
    candidates: List[Tuple[int, ci.Cursor]] = []

    def _accept(cursor: ci.Cursor) -> None:
        """Evaluate one function-kind (or unexposed) cursor and add to candidates."""
        is_func    = cursor.kind in _FUNCTION_KINDS
        is_unexposed = cursor.kind in _UNEXPOSED_KINDS

        # For UNEXPOSED_DECL, require that the spelling contains the
        # simple function name to avoid adding thousands of noisy candidates.
        if is_unexposed:
            sp = cursor.spelling or cursor.displayname or ""
            if simple_name.lower() not in sp.lower():
                return

        loc = cursor.location
        ext = cursor.extent

        # ---- file matching ----
        file_match = False
        if loc.file is not None:
            n = _normalise_path(loc.file.name)
            file_match = (
                n.endswith(target_file)
                or (norm_abs is not None and n == norm_abs)
            )

        # Null-file fallback: libclang error-recovery sets loc.file = None
        # when the owning class is unresolved.  Accept on start-line
        # proximity alone — no end-line check (end is always truncated here).
        null_file_match = (
            not file_match
            and loc.file is None
            and ext.start.line > 0
            and abs(ext.start.line - target_line) <= _START_SLOP
        )

        if not (file_match or null_file_match):
            return

        # ---- extent / name matching ----
        #
        # Tight match: extent fully contains [target_line, target_end].
        tight = (ext.start.line <= target_line
                 and ext.end.line >= target_end)

        # Loose match: start is near target AND spelling contains simple_name.
        # This handles truncated extents caused by parse errors.
        spelling = cursor.spelling or cursor.displayname or ""
        loose = (
            abs(ext.start.line - target_line) <= _START_SLOP
            and (spelling == simple_name
                 or simple_name.lower() in spelling.lower())
        )

        if not (tight or loose):
            return

        score = _score(cursor, simple_name, target_line, target_end)
        # Penalise loose matches slightly so tight matches win on tie-break.
        if not tight:
            score -= 15
        if null_file_match:
            score -= 10
        # Penalise unexposed-decl matches so real function cursors win when both exist.
        if is_unexposed:
            score -= 20
        candidates.append((score, cursor))

    def _visit(cursor: ci.Cursor) -> None:
        if cursor.kind in _ALL_CALLABLE_KINDS:
            _accept(cursor)

        # Recurse, skipping system-header subtrees for performance.
        for child in cursor.get_children():
            try:
                cl = child.location
                if cl.file is not None and cl.is_in_system_header:
                    continue
            except Exception:
                pass
            _visit(child)

    _visit(tu.cursor)

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best = candidates[0]
        logger.debug("Resolved '%s' → cursor '%s' (score=%d)",
                     func_entry.qualified_name, best.spelling, best_score)
        return best

    # ------------------------------------------------------------------ #
    #  Strategy 3: broad line-range scan (last resort)                    #
    # ------------------------------------------------------------------ #
    # Triggered when Strategies 1 and 2 found nothing — e.g. libclang
    # places the method under an unusual cursor kind in extreme error-
    # recovery scenarios.  Accept ANY cursor kind whose location and name
    # are consistent with the target function.
    broad_candidates: List[Tuple[int, ci.Cursor]] = []

    def _accept_broad(cursor: ci.Cursor) -> None:
        """
        Last-resort acceptance: require only function kind + name + line
        proximity.  File matching is intentionally relaxed here because clang
        error-recovery can assign loc.file to the *missing* header rather than
        the .cpp being parsed, which would wrongly reject valid cursors in the
        tighter strategies above.
        """
        # Only accept callable-kind cursors to avoid expression noise.
        if cursor.kind not in _ALL_CALLABLE_KINDS:
            return

        ext = cursor.extent

        # A zero start-line is a sentinel for an invalid/phantom cursor.
        if ext.start.line == 0:
            return

        if abs(ext.start.line - target_line) > _BROAD_SLOP:
            return

        spelling = cursor.spelling or cursor.displayname or ""
        if simple_name.lower() not in spelling.lower():
            return

        # Penalise cursors whose file clearly doesn't match the target .cpp.
        # We don't *reject* them — just score them lower — because clang may
        # have set loc.file to the (missing) header under error-recovery.
        file_penalty = 0
        loc = cursor.location
        if loc.file is not None:
            n = _normalise_path(loc.file.name)
            if not (n.endswith(target_file)
                    or (norm_abs is not None and n == norm_abs)):
                file_penalty = 30

        score = _score(cursor, simple_name, target_line, target_end) - 50 - file_penalty
        broad_candidates.append((score, cursor))

    def _visit_broad(cursor: ci.Cursor) -> None:
        _accept_broad(cursor)
        for child in cursor.get_children():
            try:
                cl = child.location
                if cl.file is not None and cl.is_in_system_header:
                    continue
            except Exception:
                pass
            _visit_broad(child)

    _visit_broad(tu.cursor)

    if broad_candidates:
        broad_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best = broad_candidates[0]
        logger.debug(
            "Resolved '%s' via broad scan → cursor kind=%s spelling='%s' (score=%d)",
            func_entry.qualified_name, best.kind, best.spelling, best_score,
        )
        return best

    logger.warning("No cursor found for '%s' (%s:%d-%d)",
                   func_entry.qualified_name, func_entry.file,
                   target_line, target_end)
    return None


def get_function_body(cursor: ci.Cursor) -> Optional[ci.Cursor]:
    """Return the COMPOUND_STMT body of a function cursor, or None."""
    for child in cursor.get_children():
        if child.kind == ci.CursorKind.COMPOUND_STMT:
            return child
    return None


# ---------------------------------------------------------------------------
# Strategy 1 helper
# ---------------------------------------------------------------------------

def _position_lookup(tu: ci.TranslationUnit,
                     abs_path: str,
                     target_line: int,
                     target_end: int,
                     simple_name: str) -> Optional[ci.Cursor]:
    """
    Direct cursor lookup by source position.

    Fixes applied vs the previous version:
    * Try tu.spelling first for File.from_name — libclang stores the
      exact string it was given; abs_path may differ after symlink
      resolution or path normalisation.
    * Use lexical_parent instead of semantic_parent.  For a statement
      cursor inside a function body, semantic_parent jumps to
      TRANSLATION_UNIT (skipping the function).  lexical_parent correctly
      returns the enclosing function cursor.
    * Remove the ext.end.line hard check — extent ends are unreliable
      when the TU has parse errors.  Accept on start-line proximity and
      name match only.
    * Probe multiple lines and columns so we land inside the function even
      if the first line is a macro, annotation, or blank.
    * Accept UNEXPOSED_DECL cursors with matching name — these appear when
      the owning class header is missing (clang error-recovery).
    """
    # Build list of filenames to try with File.from_name.
    # tu.spelling is the path libclang stored internally; try it first.
    candidates_filenames: List[str] = []
    if tu.spelling:
        candidates_filenames.append(tu.spelling)
    if abs_path and abs_path not in candidates_filenames:
        candidates_filenames.append(abs_path)

    # Lines and columns to probe: try the function start line first, then
    # a few lines into the body (avoids hitting blank lines / annotations).
    probe_positions = [
        (target_line,     1),
        (target_line,     5),
        (target_line + 1, 5),
        (target_line + 2, 5),
        (target_line + 3, 5),
        (target_line + 4, 5),
        (target_line + 5, 5),
    ]

    for filename in candidates_filenames:
        try:
            tu_file = ci.File.from_name(tu, filename)
        except Exception:
            continue
        if tu_file is None:
            continue

        for line, col in probe_positions:
            try:
                loc    = ci.SourceLocation.from_position(tu, tu_file, line, col)
                cursor = ci.Cursor.from_location(tu, loc)
            except Exception:
                continue

            if cursor is None:
                continue
            try:
                if cursor.kind.is_invalid():
                    continue
                if cursor.kind == ci.CursorKind.TRANSLATION_UNIT:
                    continue
            except Exception:
                continue

            # Check the cursor itself first (might already be a function).
            if _is_function_match(cursor, simple_name, target_line):
                return cursor

            # Walk up via LEXICAL parents.
            # lexical_parent follows textual nesting (correct for finding the
            # enclosing function of a statement).  semantic_parent follows
            # semantic scoping and skips straight to TRANSLATION_UNIT for
            # statement nodes.
            depth = 0
            c = cursor
            while depth < 25:
                depth += 1
                try:
                    p = c.lexical_parent
                except Exception:
                    break
                if p is None or p == c:
                    break
                c = p
                try:
                    if c.kind == ci.CursorKind.TRANSLATION_UNIT:
                        break
                    if _is_function_match(c, simple_name, target_line):
                        return c
                except Exception:
                    break

    return None


def _is_function_match(cursor: ci.Cursor,
                       simple_name: str,
                       target_line: int) -> bool:
    """
    True if cursor is a function-kind (or matching unexposed-decl) cursor
    whose start line is near target_line AND whose spelling contains
    simple_name.

    Deliberately does NOT check extent.end so truncated extents
    (from parse errors) are still accepted.
    """
    try:
        is_func     = cursor.kind in _FUNCTION_KINDS
        is_unexposed = cursor.kind in _UNEXPOSED_KINDS
        if not (is_func or is_unexposed):
            return False
        ext = cursor.extent
        if abs(ext.start.line - target_line) > _START_SLOP:
            return False
        spelling = cursor.spelling or cursor.displayname or ""
        # For unexposed decls require that the simple name appears in the
        # spelling to avoid accepting arbitrary unexposed nodes.
        if is_unexposed and simple_name.lower() not in spelling.lower():
            return False
        return (spelling == simple_name
                or simple_name.lower() in spelling.lower())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Scoring and utility helpers
# ---------------------------------------------------------------------------

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
    # extent.end is unreliable with parse errors, so weight it less.
    score -= min(abs(ext.end.line - target_end), 20)

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
