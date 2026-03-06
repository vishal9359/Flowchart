"""
Source extraction and libclang TranslationUnit management.

- SourceExtractor: reads and caches source files, extracts line ranges and
  cursor extents as raw text.
- TranslationUnitParser: creates and caches libclang TUs with the correct
  std and include args.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import clang.cindex as ci

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------

class SourceExtractor:
    """Reads source files and provides text extraction helpers."""

    def __init__(self, base_path: str) -> None:
        self._base_path = Path(base_path)
        self._cache: Dict[str, List[str]] = {}

    def get_lines(self, relative_file: str) -> List[str]:
        """Return all lines for a source file (cached)."""
        if relative_file not in self._cache:
            abs_path = self._base_path / relative_file
            if not abs_path.exists():
                raise FileNotFoundError(f"Source file not found: {abs_path}")
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                self._cache[relative_file] = f.readlines()
        return self._cache[relative_file]

    def extract_by_lines(self, relative_file: str,
                         start_line: int, end_line: int) -> str:
        """Extract function source strictly by 1-indexed line range."""
        lines = self.get_lines(relative_file)
        return "".join(lines[start_line - 1: end_line])

    @staticmethod
    def get_extent_text(lines: List[str],
                        start_line: int, end_line: int,
                        start_col: int, end_col: int) -> str:
        """
        Extract text for a cursor extent.
        All line/column values are 1-indexed (as returned by libclang).
        """
        if not lines:
            return ""

        # Clamp to valid range
        start_line = max(1, min(start_line, len(lines)))
        end_line = max(start_line, min(end_line, len(lines)))

        if start_line == end_line:
            row = lines[start_line - 1]
            return row[start_col - 1: end_col - 1].strip()

        result: List[str] = []
        result.append(lines[start_line - 1][start_col - 1:].rstrip())
        for i in range(start_line, end_line - 1):
            result.append(lines[i].rstrip())
        result.append(lines[end_line - 1][: end_col - 1].rstrip())
        return "\n".join(result)

    def abs_path(self, relative_file: str) -> str:
        return str(self._base_path / relative_file)


# ---------------------------------------------------------------------------
# Translation Unit management
# ---------------------------------------------------------------------------

class TranslationUnitParser:
    """Creates and caches libclang TranslationUnits."""

    _PARSE_OPTIONS = (
        ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        | ci.TranslationUnit.PARSE_INCOMPLETE
        | ci.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES
    )

    def __init__(self, std: str, extra_clang_args: List[str]) -> None:
        self._std = std
        self._extra_args = extra_clang_args
        self._index = ci.Index.create()
        self._tu_cache: Dict[str, ci.TranslationUnit] = {}

    def _build_args(self) -> List[str]:
        return [f"-std={self._std}", "-x", "c++"] + self._extra_args

    def get_tu(self, abs_path: str) -> ci.TranslationUnit:
        """Return (cached) TranslationUnit for a source file."""
        if abs_path not in self._tu_cache:
            args = self._build_args()
            logger.debug("Parsing TU: %s", abs_path)
            tu = self._index.parse(abs_path, args=args,
                                   options=self._PARSE_OPTIONS)
            if tu is None:
                raise RuntimeError(f"libclang failed to parse: {abs_path}")
            self._log_diagnostics(tu, abs_path)
            self._tu_cache[abs_path] = tu
        return self._tu_cache[abs_path]

    def get_tu_full(self, abs_path: str) -> ci.TranslationUnit:
        """
        Return a TranslationUnit parsed WITHOUT skipping function bodies.
        Used when we need to traverse the actual function body for CFG building.
        """
        cache_key = abs_path + "__full"
        if cache_key not in self._tu_cache:
            args = self._build_args()
            logger.debug("Parsing full TU (with bodies): %s", abs_path)
            options = (
                ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
                | ci.TranslationUnit.PARSE_INCOMPLETE
            )
            tu = self._index.parse(abs_path, args=args, options=options)
            if tu is None:
                raise RuntimeError(f"libclang failed to parse: {abs_path}")
            self._log_diagnostics(tu, abs_path)
            self._tu_cache[cache_key] = tu
        return self._tu_cache[cache_key]

    @staticmethod
    def _log_diagnostics(tu: ci.TranslationUnit, path: str) -> None:
        errors = [d for d in tu.diagnostics
                  if d.severity >= ci.Diagnostic.Error]
        if errors:
            logger.warning("%d error(s) in %s (AST may be incomplete)",
                           len(errors), path)
            for d in errors[:5]:
                logger.debug("  clang: %s", d.spelling)
