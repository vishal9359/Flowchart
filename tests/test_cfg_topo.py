"""
test_cfg_topo.py — Layer-1 and Layer-2 test runner for flowchart_engine.

Tests
-----
  Layer 1a — CFG structural invariants
    Validates the Control Flow Graph produced by ast_engine is well-formed.
    No LLM call is made.

  Layer 1b — Topological sort invariants
    Validates that _topo_sort produces a valid execution order and correctly
    identifies loop back-edges.

  Layer 2  — CFG node-type counts vs Mermaid shape counts  (cross-layer)
    Reads the previously-generated output JSON from --out-dir and compares
    the number of each shape kind (diamond / oval / rectangle / subroutine)
    in the Mermaid script against the matching counts in the CFG.
    Requires --out-dir to be supplied.

Usage
-----
  # Layer 1 only (no output JSON needed):
  python tests/test_cfg_topo.py \\
      --interface-json functions.json \\
      --metadata-json  metadata.json

  # Layer 1 + Layer 2 (with output JSON):
  python tests/test_cfg_topo.py \\
      --interface-json functions.json \\
      --metadata-json  metadata.json \\
      --out-dir        output/

  # Test a single function:
  python tests/test_cfg_topo.py \\
      --interface-json functions.json \\
      --metadata-json  metadata.json \\
      --out-dir        output/ \\
      --function-key   "src|myfile|MyClass::myMethod|int"

  # Pass extra include paths to libclang:
  python tests/test_cfg_topo.py \\
      --interface-json functions.json \\
      --metadata-json  metadata.json \\
      --clang-arg -I/path/to/headers \\
      --clang-arg -I/another/path
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Allow running from project root or from tests/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import FunctionEntry, NodeType
from ast_engine.cfg_builder import CFGBuilder
from ast_engine.parser import SourceExtractor, TranslationUnitParser
from ast_engine.resolver import find_function_cursor
from llm.generator import _topo_sort


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str = ""


@dataclass
class FunctionTestResult:
    function_key: str
    qualified_name: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed]


# ---------------------------------------------------------------------------
# Layer 1a — CFG structural invariants
# ---------------------------------------------------------------------------

def check_cfg_invariants(cfg) -> List[CheckResult]:
    """
    Invariants that must hold for every well-formed ControlFlowGraph:

    1. At least one node exists.
    2. The entry node ID is set and points to a real node.
    3. Every edge's source and target are valid node IDs (no dangling refs).
    4. Exactly one START node exists.
    5. At least one END node exists.
    """
    results: List[CheckResult] = []
    node_ids: Set[str] = set(cfg.nodes.keys())

    # 1. Non-empty
    results.append(CheckResult(
        "cfg.nodes_not_empty",
        len(cfg.nodes) > 0,
        f"{len(cfg.nodes)} node(s)",
    ))

    # 2. Entry node is set and exists
    entry_ok = bool(cfg.entry_node_id) and cfg.entry_node_id in cfg.nodes
    results.append(CheckResult(
        "cfg.entry_node_exists",
        entry_ok,
        f"entry_node_id={cfg.entry_node_id!r}",
    ))

    # 3. No dangling edge endpoints
    bad_edges = [
        (e.source, e.target)
        for e in cfg.edges
        if e.source not in node_ids or e.target not in node_ids
    ]
    results.append(CheckResult(
        "cfg.edges_reference_valid_nodes",
        len(bad_edges) == 0,
        f"dangling edges: {bad_edges[:3]}" if bad_edges else f"{len(cfg.edges)} edge(s) all valid",
    ))

    # 4. Exactly one START node
    start_count = sum(1 for n in cfg.nodes.values() if n.node_type == NodeType.START)
    results.append(CheckResult(
        "cfg.exactly_one_start_node",
        start_count == 1,
        f"START count={start_count}",
    ))

    # 5. At least one END node
    end_count = sum(1 for n in cfg.nodes.values() if n.node_type == NodeType.END)
    results.append(CheckResult(
        "cfg.has_end_node",
        end_count >= 1,
        f"END count={end_count}",
    ))

    return results


# ---------------------------------------------------------------------------
# Layer 1b — Topological sort invariants
# ---------------------------------------------------------------------------

def check_topo_sort_invariants(
    cfg,
    topo_order: List[str],
    back_edges: Set[Tuple[str, str]],
) -> List[CheckResult]:
    """
    Invariants that must hold for a correct topological sort of the CFG:

    1. Every CFG node appears in the output exactly once.
    2. The entry node is first in the order.
    3. For every forward edge A→B (not a back-edge): A appears before B.
    4. Every back-edge points from a later node to an earlier node
       (i.e. it truly is a backward arc in the sort order).
    """
    results: List[CheckResult] = []
    node_ids: Set[str] = set(cfg.nodes.keys())

    # 1. All nodes present, no extras, no duplicates
    order_ids = list(topo_order)
    missing = node_ids - set(order_ids)
    extra   = set(order_ids) - node_ids
    dupes   = {nid for nid in order_ids if order_ids.count(nid) > 1}

    results.append(CheckResult(
        "topo.all_nodes_present_no_duplicates",
        not missing and not extra and not dupes,
        (
            f"missing={missing} extra={extra} dupes={dupes}"
            if (missing or extra or dupes)
            else f"{len(order_ids)} node(s) in order"
        ),
    ))

    # 2. Entry node is first
    if cfg.entry_node_id and cfg.entry_node_id in node_ids:
        entry_first = bool(order_ids) and order_ids[0] == cfg.entry_node_id
        results.append(CheckResult(
            "topo.entry_node_is_first",
            entry_first,
            f"first={order_ids[0] if order_ids else None!r}  entry={cfg.entry_node_id!r}",
        ))

    # 3. Forward edges respect the order (source before target)
    pos: Dict[str, int] = {nid: i for i, nid in enumerate(order_ids)}
    forward_violations = [
        (e.source, e.target)
        for e in cfg.edges
        if (e.source, e.target) not in back_edges
        and e.source in pos
        and e.target in pos
        and pos[e.source] >= pos[e.target]
    ]
    results.append(CheckResult(
        "topo.forward_edges_respect_order",
        len(forward_violations) == 0,
        (
            f"violations (first 3): {forward_violations[:3]}"
            if forward_violations
            else "all forward edges ordered correctly"
        ),
    ))

    # 4. Back-edges truly go backward (src after tgt in order)
    bad_back = [
        (src, tgt)
        for src, tgt in back_edges
        if src in pos and tgt in pos and pos[src] <= pos[tgt]
    ]
    results.append(CheckResult(
        "topo.back_edges_are_backward",
        len(bad_back) == 0,
        (
            f"bad back-edges: {bad_back[:3]}"
            if bad_back
            else f"{len(back_edges)} back-edge(s) all valid"
        ),
    ))

    return results


# ---------------------------------------------------------------------------
# Layer 2 — CFG node-type counts vs Mermaid shape counts
# ---------------------------------------------------------------------------

# Mermaid shape groupings by NodeType
_DIAMOND_TYPES   = {NodeType.DECISION, NodeType.LOOP_HEAD, NodeType.SWITCH_HEAD}
_OVAL_TYPES      = {NodeType.START, NodeType.END}
_SUBROUTINE_TYPES = {NodeType.CATCH}
# Everything else → rectangle
_RECT_TYPES      = {
    NodeType.ACTION, NodeType.RETURN, NodeType.BREAK,
    NodeType.CONTINUE, NodeType.CASE, NodeType.DEFAULT_CASE,
    NodeType.TRY_HEAD,
}


def _count_cfg_shapes(cfg) -> Dict[str, int]:
    counts = {"diamond": 0, "oval": 0, "rectangle": 0, "subroutine": 0}
    for node in cfg.nodes.values():
        if node.node_type in _DIAMOND_TYPES:
            counts["diamond"] += 1
        elif node.node_type in _OVAL_TYPES:
            counts["oval"] += 1
        elif node.node_type in _SUBROUTINE_TYPES:
            counts["subroutine"] += 1
        else:
            counts["rectangle"] += 1
    return counts


def _count_mermaid_shapes(mermaid: str) -> Dict[str, int]:
    """
    Parse the Mermaid script line-by-line and count node shapes.

    Shape syntax (from mermaid/builder.py):
      Oval       →  N0([label])     — START / END
      Diamond    →  N1{label}       — DECISION / LOOP_HEAD / SWITCH_HEAD
      Subroutine →  N2[[label]]     — CATCH
      Rectangle  →  N3[label]       — ACTION / RETURN / BREAK / CONTINUE / ...

    Lines containing '-->' are edges and are skipped.
    The first line 'flowchart TD' is also skipped.

    Check order matters: subroutine ([[ ) must be checked before
    rectangle ( [ ), and oval ( ([ ) must be checked before both.
    """
    counts = {"diamond": 0, "oval": 0, "rectangle": 0, "subroutine": 0}
    for line in mermaid.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("flowchart") or "-->" in stripped:
            continue
        # Order: oval > subroutine > diamond > rectangle
        if re.search(r'\(\[', stripped):       # ([...])
            counts["oval"] += 1
        elif re.search(r'\[\[', stripped):     # [[...]]
            counts["subroutine"] += 1
        elif re.search(r'\{', stripped):       # {...}
            counts["diamond"] += 1
        elif re.search(r'\[', stripped):       # [...]
            counts["rectangle"] += 1
    return counts


def check_cfg_vs_mermaid(cfg, mermaid: str) -> List[CheckResult]:
    """
    Compare the count of each node shape between the CFG and the Mermaid script.
    A mismatch means a node was dropped or rendered with the wrong shape.
    """
    results: List[CheckResult] = []
    cfg_counts     = _count_cfg_shapes(cfg)
    mermaid_counts = _count_mermaid_shapes(mermaid)

    for shape in ("oval", "diamond", "rectangle", "subroutine"):
        c = cfg_counts[shape]
        m = mermaid_counts[shape]
        results.append(CheckResult(
            f"mermaid.{shape}_count_matches_cfg",
            c == m,
            f"CFG={c}  Mermaid={m}" + ("  MISMATCH" if c != m else "  OK"),
        ))

    return results


# ---------------------------------------------------------------------------
# Output JSON loader
# ---------------------------------------------------------------------------

def _load_mermaid_for_function(out_dir: str, func_entry: FunctionEntry) -> Optional[str]:
    """
    Locate the output JSON for func_entry's source file (by stem) and return
    the 'flowchart' field for this specific function key, or None if not found.

    Example: func_entry.file = "src/qos/qos_event_manager.cpp"
             → looks for <out_dir>/qos_event_manager.json
    """
    source_stem = Path(func_entry.file).stem
    out_path    = Path(out_dir) / f"{source_stem}.json"
    if not out_path.exists():
        return None

    with open(out_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    for entry in entries:
        if entry.get("functionKey") == func_entry.key:
            return entry.get("flowchart", "")

    return None


# ---------------------------------------------------------------------------
# functions.json → FunctionEntry (mirrors pkb/builder.py PKB.build)
# ---------------------------------------------------------------------------

def _parse_function_entries(functions_data: Dict) -> List[FunctionEntry]:
    """
    Parse the raw functions.json dict into FunctionEntry objects using the
    same field mapping as ProjectKnowledgeBase.build().

    functions.json schema (per entry):
      {
        "qualifiedName": "...",
        "location": { "file": "...", "line": N, "endLine": N },
        "parameters": [...],
        "callsIds": [...],
        "calledByIds": [...],
        "description": "..."
      }
    """
    entries: List[FunctionEntry] = []
    for key, data in functions_data.items():
        location = data.get("location", {})
        entries.append(FunctionEntry(
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
        ))
    return entries


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"


def _colored(text: str, color: str) -> str:
    if sys.stdout.isatty():
        return f"{color}{text}{_RESET}"
    return text


def _print_function_result(result: FunctionTestResult) -> None:
    status = _colored("PASS", _GREEN) if result.passed else _colored("FAIL", _RED)
    print(f"\n[{status}] {result.qualified_name}")
    print(f"       key: {result.function_key}")
    for check in result.checks:
        icon = _colored("  OK ", _GREEN) if check.passed else _colored("FAIL ", _RED)
        print(f"    {icon} {check.name}")
        if check.message:
            print(f"           → {check.message}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="CFG + Topological Sort test runner (Layer 1 & 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--interface-json", required=True,
                    help="Path to functions.json (same file used by flowchart_engine)")
    ap.add_argument("--metadata-json",  required=True,
                    help="Path to metadata.json (provides basePath and projectName)")
    ap.add_argument("--out-dir", default=None,
                    help="Directory containing generated flowchart JSON files. "
                         "When provided, Layer-2 CFG-vs-Mermaid checks are enabled.")
    ap.add_argument("--std", default="c++17",
                    help="C++ standard for libclang (default: c++17)")
    ap.add_argument("--clang-arg", action="append", default=[], dest="clang_args",
                    help="Extra argument forwarded to libclang "
                         "(e.g. -I/path/to/headers). Repeat for multiple paths.")
    ap.add_argument("--function-key", default=None,
                    help="Run tests for this single function key only.")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    with open(args.interface_json, "r", encoding="utf-8") as f:
        functions_data: Dict = json.load(f)

    with open(args.metadata_json, "r", encoding="utf-8") as f:
        meta: Dict = json.load(f)

    base_path = meta.get("basePath", ".")
    entries   = _parse_function_entries(functions_data)

    if args.function_key:
        entries = [e for e in entries if e.key == args.function_key]
        if not entries:
            print(f"ERROR: --function-key not found in functions.json: {args.function_key!r}")
            return 1

    print(f"Testing {len(entries)} function(s) from {args.interface_json}")
    if args.out_dir:
        print(f"Layer-2 (CFG vs Mermaid) enabled — reading from: {args.out_dir}")
    print(f"base_path={base_path}  std={args.std}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Shared libclang objects (reused across functions for performance)
    # ------------------------------------------------------------------
    source_extractor = SourceExtractor(base_path)
    tu_parser        = TranslationUnitParser(args.std, args.clang_args)

    # ------------------------------------------------------------------
    # Run tests per function
    # ------------------------------------------------------------------
    results: List[FunctionTestResult] = []

    for func_entry in entries:
        checks: List[CheckResult] = []

        try:
            # Parse source file and resolve function cursor
            source_lines = source_extractor.get_lines(func_entry.file)
            abs_path     = source_extractor.abs_path(func_entry.file)
            tu           = tu_parser.get_tu_full(abs_path)
            func_cursor  = find_function_cursor(tu, func_entry, abs_path)

            if func_cursor is None:
                checks.append(CheckResult(
                    "cfg.cursor_resolved", False,
                    f"could not resolve cursor — check line numbers in functions.json",
                ))
                results.append(FunctionTestResult(func_entry.key, func_entry.qualified_name, checks))
                continue

            checks.append(CheckResult("cfg.cursor_resolved", True, "cursor found"))

            # Build CFG (no LLM, no enrichment)
            cfg = CFGBuilder(source_lines).build(func_cursor, func_entry)

            # Layer 1a — CFG structural invariants
            checks.extend(check_cfg_invariants(cfg))

            # Layer 1b — Topological sort invariants
            topo_order, back_edges = _topo_sort(cfg)
            checks.extend(check_topo_sort_invariants(cfg, topo_order, back_edges))

            # Layer 2 — CFG node-type counts vs Mermaid shape counts
            if args.out_dir:
                mermaid = _load_mermaid_for_function(args.out_dir, func_entry)
                if mermaid:
                    checks.extend(check_cfg_vs_mermaid(cfg, mermaid))
                else:
                    checks.append(CheckResult(
                        "mermaid.output_found", False,
                        f"no matching entry in {Path(func_entry.file).stem}.json "
                        f"under {args.out_dir}",
                    ))

        except Exception as exc:
            checks.append(CheckResult(
                "cfg.build_no_exception", False, str(exc),
            ))

        results.append(FunctionTestResult(func_entry.key, func_entry.qualified_name, checks))

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    for result in results:
        _print_function_result(result)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total  = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    failed_checks_total = sum(len(r.failed_checks) for r in results)

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(_colored(f"ALL PASSED  {passed}/{total} functions", _GREEN))
    else:
        print(_colored(f"FAILED  {failed}/{total} functions  ({failed_checks_total} check(s) failed)", _RED))

    # List functions that failed for easy grep
    if failed > 0:
        print("\nFailed functions:")
        for r in results:
            if not r.passed:
                print(f"  {r.qualified_name}")
                for c in r.failed_checks:
                    print(f"    - {c.name}: {c.message}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
