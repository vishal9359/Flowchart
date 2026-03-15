"""
LLM-based flowchart label generator.

Batching strategy
-----------------
Large functions produce prompts that exceed local LLM context windows.  Two
complementary strategies keep every prompt within any model's context window:

  1. Smaller defaults — BATCH_SIZE=4, MAX_PROMPT_CHARS=6000 (~1500 tokens).
     At ~3 chars/token for C++ code this fits a 2048-token default window
     with room for JSON output.

  2. Auto-halving — if ALL retries for a batch return "no LLM response"
     (empty string from server, the Ollama signal that num_ctx was exceeded),
     the batch is automatically split in half and each sub-batch is retried.
     This recurses up to depth 3 (minimum 1 node per call), so the generator
     adapts to any model's actual context window without manual tuning.

Retry discipline
----------------
There are TWO distinct failure modes; they need different retry strategies:

  * "no LLM response" (raw=None) — the server returned an empty string.
    Root cause: prompt exceeded num_ctx.  Retrying with the SAME or LARGER
    prompt never helps.  Retry with the SAME prompt (server may be temporarily
    busy) but do NOT append a retry note (which makes the prompt larger).
    After all retries fail this way, trigger auto-halving.

  * "bad response" (raw is not None but JSON is wrong / nodes missing) —
    the LLM returned something but it was malformed or incomplete.  Append a
    targeted retry note listing the exact failing node_ids and reasons so the
    LLM corrects precisely.

Budget logic
------------
MAX_PROMPT_CHARS  — hard cap on system+user prompt characters.
CONTEXT_BUDGET    — max chars for the PKB context packet.
SOURCE_PADDING    — source lines above/below batch nodes in excerpt.
BATCH_SIZE        — default nodes per LLM call.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from llm.client import LlmClient
from llm.prompts import SYSTEM_PROMPT, build_user_prompt
from models import CfgNode, ControlFlowGraph, FunctionEntry, NodeType
from pkb.builder import ProjectKnowledgeBase

logger = logging.getLogger(__name__)

# ── Prompt-size budget constants ─────────────────────────────────────────────

# Target: fits in a 2048-token model with room for ~400 tokens of JSON output.
# System prompt ~850 tokens + user prompt ~1150 tokens = ~2000 tokens total.
# At 4 chars/token: 2000 tokens * 4 = 8000 chars.  Use 6000 to be safe for
# code-heavy content which tokenizes less efficiently (~2-3 chars/token).
MAX_PROMPT_CHARS = 6000

# Max chars for the PKB context packet (hierarchy + callee graph).
# The 4-level BFS can be very large; cap it so it doesn't dominate.
CONTEXT_BUDGET = 1200

# Lines of source above/below the batch node range for the excerpt.
SOURCE_PADDING = 5

# Default nodes per LLM call.  4 is a safe default for 2048-token models.
# Increase via --llm-batch-size for models with larger context windows.
BATCH_SIZE = 4

# Max label length accepted from LLM
_MAX_LABEL_LEN = 300

# Max depth for auto-halving recursion (prevents infinite split)
_MAX_SPLIT_DEPTH = 3


class LabelGenerator:
    """
    Generates LLM labels for all CFG nodes.

    Batches nodes BATCH_SIZE per call.  On persistent "no LLM response"
    failures, auto-halves the batch until a working size is found.
    """

    def __init__(self, client: LlmClient, pkb: ProjectKnowledgeBase,
                 max_retries: int = 2,
                 batch_size: int = BATCH_SIZE) -> None:
        self._client = client
        self._pkb = pkb
        self._max_retries = max_retries
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label_cfg(self, cfg: ControlFlowGraph,
                  func_entry: FunctionEntry,
                  source_code: str,
                  base_path: str) -> None:
        """Fill cfg.nodes[*].label for every non-sentinel node. In-place."""
        labelable = [n for n in cfg.nodes.values()
                     if n.node_type not in (NodeType.START, NodeType.END)]
        if not labelable:
            return

        # Build static (per-function) context — hierarchy + callers + purpose + phases
        # + full BFS callee graph (always present as baseline callee context).
        # Per-batch targeted callee context is ADDED ON TOP in _label_batch(), so we
        # never lose callee context even when enriched_context["function_calls"] is empty.
        base_context = self._pkb.build_context_packet(func_entry, base_path)
        phases = self._pkb.get_function_phases(func_entry)

        # Order labelable nodes in CFG execution order (topological sort).
        # This ensures neighbor context (preceding/following_node_code) is meaningful
        # and region batching groups semantically related sequential nodes together.
        # back_edges is returned so _make_region_batches can exclude loop back-edges
        # from predecessor counts (otherwise loop headers appear as merge points).
        topo_ids, back_edges = _topo_sort(cfg)
        labelable_ids = {n.node_id for n in labelable}
        node_by_id = {n.node_id: n for n in labelable}
        ordered_labelable = [node_by_id[nid] for nid in topo_ids if nid in labelable_ids]

        # CFG-region-aware batching: group nodes that belong to the same sequential
        # region (basic-block-like), so the LLM labels a coherent block at a time.
        batches = _make_region_batches(ordered_labelable, cfg, self._batch_size, back_edges)
        label_map: Dict[str, str] = {}

        for batch_idx, batch in enumerate(batches):
            batch_labels = self._label_batch_with_split(
                batch=batch,
                func_entry=func_entry,
                base_context=base_context,
                source_code=source_code,
                all_nodes=ordered_labelable,
                phases=phases,
            )
            label_map.update(batch_labels)
            logger.debug("Batch %d/%d done (%d nodes) for '%s'",
                         batch_idx + 1, len(batches), len(batch),
                         func_entry.qualified_name)

        # Coherence pass: normalize terminology and phrasing across all labels.
        if len(label_map) >= 5:
            label_map = self._coherence_pass(cfg, func_entry, label_map, ordered_labelable)

        self._apply_labels(cfg, label_map)

        # START / END get simple fixed labels
        for node in cfg.nodes.values():
            if node.node_type == NodeType.START:
                node.label = f"Start: {func_entry.qualified_name.split('::')[-1]}"
            elif node.node_type == NodeType.END:
                node.label = "End"

        # Report fallback usage
        fallback_count = sum(1 for n in labelable if _looks_like_fallback(n))
        if fallback_count > 0:
            logger.warning("'%s': %d/%d node(s) used fallback labels",
                           func_entry.qualified_name, fallback_count, len(labelable))
        else:
            logger.debug("'%s': all %d nodes labeled by LLM",
                         func_entry.qualified_name, len(labelable))

    # ------------------------------------------------------------------
    # Auto-halving wrapper
    # ------------------------------------------------------------------

    def _label_batch_with_split(
        self,
        batch: List[CfgNode],
        func_entry: FunctionEntry,
        base_context: str,
        source_code: str,
        all_nodes: Optional[List[CfgNode]] = None,
        phases: Optional[List[Dict]] = None,
        depth: int = 0,
    ) -> Dict[str, str]:
        """
        Label a batch.  If ALL retries returned no LLM response and the batch
        has >1 node, split in half and retry each sub-batch independently.
        Recurses up to _MAX_SPLIT_DEPTH times (minimum 1 node per call).
        """
        result, all_no_response = self._label_batch(
            batch=batch,
            func_entry=func_entry,
            base_context=base_context,
            source_code=source_code,
            all_nodes=all_nodes,
            phases=phases,
        )

        if all_no_response and len(batch) > 1 and depth < _MAX_SPLIT_DEPTH:
            logger.warning(
                "All %d attempt(s) returned no LLM response for a %d-node batch. "
                "Auto-splitting in half (depth %d of %d)…",
                self._max_retries + 1, len(batch), depth + 1, _MAX_SPLIT_DEPTH,
            )
            mid = len(batch) // 2
            combined: Dict[str, str] = {}
            for sub_batch in [batch[:mid], batch[mid:]]:
                sub_result = self._label_batch_with_split(
                    sub_batch, func_entry, base_context, source_code,
                    all_nodes=all_nodes, phases=phases,
                    depth=depth + 1,
                )
                combined.update(sub_result)
            return combined

        return result

    # ------------------------------------------------------------------
    # Core per-batch labeling
    # ------------------------------------------------------------------

    def _label_batch(
        self,
        batch: List[CfgNode],
        func_entry: FunctionEntry,
        base_context: str,
        source_code: str,
        all_nodes: Optional[List[CfgNode]] = None,
        phases: Optional[List[Dict]] = None,
    ) -> Tuple[Dict[str, str], bool]:
        """
        Label one batch of nodes.

        Builds a batch-specific context packet by combining the static base
        context (hierarchy, callers, purpose, phases) with targeted callee
        context for only the functions actually called in this batch.

        Returns (label_map, all_no_response) where:
          all_no_response=True  → every attempt returned None (server issue /
                                  context overflow) — caller should auto-halve.
          all_no_response=False → at least one attempt got a response (even if
                                  partial) — fallback labels are already set.
        """
        # Targeted callee context: only inject callees used in THIS batch.
        callee_names = _extract_batch_callee_names(batch)
        callee_ctx = self._pkb.build_targeted_callee_context(func_entry, callee_names)
        context_packet = base_context + ("\n" + callee_ctx if callee_ctx else "")

        base_prompt = _build_size_aware_prompt(
            batch=batch,
            func_entry=func_entry,
            context_packet=context_packet,
            source_code=source_code,
            all_nodes=all_nodes,
            phases=phases,
        )

        total_chars = len(SYSTEM_PROMPT) + len(base_prompt)
        logger.debug("Batch prompt: %d chars (~%d tokens) for %d nodes",
                     total_chars, total_chars // 4, len(batch))
        # Print full prompt at TRACE level (set env FLOWCHART_TRACE=1 to enable)
        if os.environ.get("FLOWCHART_TRACE"):
            print("\n" + "="*60)
            print(f"[TRACE] SYSTEM PROMPT:\n{SYSTEM_PROMPT[:500]}...")
            print(f"[TRACE] USER PROMPT:\n{base_prompt}")
            print("="*60 + "\n")

        required_ids = {n.node_id for n in batch}
        accumulated: Dict[str, str] = {}
        remaining: Set[str] = set(required_ids)
        last_failures: Dict[str, str] = {}
        no_response_attempts = 0

        for attempt in range(1, self._max_retries + 2):
            # ── Choose prompt for this attempt ────────────────────────
            if attempt == 1 or no_response_attempts > 0:
                # First attempt, or previous attempt(s) had no response.
                # Do NOT append retry note — it only makes the prompt larger,
                # which is exactly wrong when the server is returning empty
                # because the prompt already exceeds num_ctx.
                prompt = base_prompt
            else:
                # Previous attempt returned a response but it was incomplete.
                # Add a targeted retry note so the LLM fixes specific nodes.
                prompt = base_prompt + _build_retry_note(last_failures)

            # ── Call LLM ──────────────────────────────────────────────
            raw = self._client.generate(SYSTEM_PROMPT, prompt)

            if raw is None:
                no_response_attempts += 1
                prompt_chars = len(SYSTEM_PROMPT) + len(prompt)
                logger.warning(
                    "Batch attempt %d/%d: no LLM response "
                    "(prompt=%d chars ~%d tokens). "
                    "Will auto-split if all attempts fail. "
                    "Or pass: --llm-batch-size 2 --llm-num-ctx 16384",
                    attempt, self._max_retries + 1,
                    prompt_chars, prompt_chars // 4,
                )
                last_failures = {nid: "LLM returned no response" for nid in remaining}
                continue

            # ── Parse response ────────────────────────────────────────
            partial, failures = _parse_partial(raw, remaining)
            accumulated.update(partial)
            remaining -= set(partial.keys())
            last_failures = failures

            if not remaining:
                logger.debug("Batch fully labeled on attempt %d", attempt)
                return accumulated, False

            logger.warning(
                "Batch attempt %d/%d: %d node(s) still need labels — %s",
                attempt, self._max_retries + 1, len(remaining),
                {nid: failures.get(nid, "missing") for nid in remaining},
            )

        # ── All retries exhausted ─────────────────────────────────────
        all_no_response = (no_response_attempts == self._max_retries + 1)

        if remaining:
            node_by_id = {n.node_id: n for n in batch}
            for nid in remaining:
                node = node_by_id.get(nid)
                if node:
                    accumulated[nid] = _fallback_label(node)
            if not all_no_response:
                # Only warn about fallback here; auto-halving path warns separately
                logger.warning(
                    "Fallback labels applied for %d node(s): %s",
                    len(remaining), sorted(remaining),
                )

        return accumulated, all_no_response

    # ------------------------------------------------------------------
    # Coherence pass — normalize all labels after per-batch generation
    # ------------------------------------------------------------------

    _COHERENCE_SYSTEM = (
        "You are a technical editor reviewing flowchart labels for a C++ function. "
        "Fix ONLY: inconsistent terminology, non-active-voice phrasing, or labels "
        "that break the narrative flow when read in sequence. "
        "Do NOT change meanings, do NOT re-label nodes that are already good. "
        'Return ONLY a JSON object for labels you changed: {"node_id": "improved_label"}. '
        "Return {} if nothing needs changing. No markdown, no explanations."
    )

    def _coherence_pass(
        self,
        cfg: ControlFlowGraph,
        func_entry: FunctionEntry,
        label_map: Dict[str, str],
        ordered_nodes: List[CfgNode],
    ) -> Dict[str, str]:
        """
        Second LLM call per function to normalize terminology and phrasing.

        Sees all labels in execution order and can spot inconsistencies that
        per-batch generation misses (e.g., mixing 'request' / 'req', passive
        voice, inconsistent subject across labels).  Only changes what needs
        changing — returns the same label_map if all is consistent.
        """
        # Build ordered list of (node_id, label, type) for the prompt
        ordered_pairs = [
            (n.node_id, label_map[n.node_id], n.node_type.value)
            for n in ordered_nodes
            if n.node_id in label_map
        ]
        if not ordered_pairs:
            return label_map

        # Determine purpose string
        purpose = func_entry.description or ""
        if self._pkb._knowledge:
            fk = self._pkb._knowledge.functions.get(func_entry.qualified_name)
            if fk and fk.comment:
                purpose = fk.comment

        node_lines = "\n".join(
            f'  "{nid}" ({ntype}): "{lbl}"'
            for nid, lbl, ntype in ordered_pairs
        )

        prompt = (
            f"Function: {func_entry.qualified_name}\n"
            + (f"Purpose: {purpose}\n" if purpose else "")
            + f"\nFlowchart labels ({len(ordered_pairs)} nodes in execution order):\n"
            + node_lines
            + "\n\nReview as a set:\n"
            "1. Fix inconsistent terminology (e.g. 'request'/'req' — pick one).\n"
            "2. Ensure active voice throughout (e.g. 'Initialize X' not 'X is initialized').\n"
            "3. Make labels read as a coherent step-by-step narrative.\n"
            "4. Do NOT change node meaning — only improve phrasing/consistency.\n"
            "5. Return only labels you changed.\n"
            'Return ONLY: {"node_id": "improved_label", ...} or {} if nothing needs changing.'
        )

        # Cap prompt size — coherence pass is best-effort
        if len(self._COHERENCE_SYSTEM) + len(prompt) > MAX_PROMPT_CHARS:
            logger.debug("Coherence pass skipped for '%s': prompt too large",
                         func_entry.qualified_name)
            return label_map

        raw = self._client.generate(self._COHERENCE_SYSTEM, prompt)
        if not raw:
            logger.debug("Coherence pass: no LLM response for '%s'",
                         func_entry.qualified_name)
            return label_map

        cleaned = _extract_json(raw)
        if not cleaned:
            return label_map

        try:
            changes = json.loads(cleaned)
        except json.JSONDecodeError:
            return label_map

        if not isinstance(changes, dict) or not changes:
            return label_map

        updated = dict(label_map)
        changed_count = 0
        for nid, new_label in changes.items():
            if (nid in updated
                    and isinstance(new_label, str)
                    and new_label.strip()
                    and len(new_label) <= _MAX_LABEL_LEN):
                logger.debug("Coherence: %s  %r → %r", nid, updated[nid], new_label)
                updated[nid] = new_label.strip()
                changed_count += 1

        if changed_count:
            logger.info("Coherence pass updated %d label(s) for '%s'",
                        changed_count, func_entry.qualified_name)
        return updated

    # ------------------------------------------------------------------
    # Apply labels back to CFG nodes
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_labels(cfg: ControlFlowGraph,
                      label_map: Dict[str, str]) -> None:
        for node_id, label in label_map.items():
            if node_id in cfg.nodes:
                cfg.nodes[node_id].label = label.strip()


# ---------------------------------------------------------------------------
# Size-aware prompt builder
# ---------------------------------------------------------------------------

def _build_size_aware_prompt(
    batch: List[CfgNode],
    func_entry: FunctionEntry,
    context_packet: str,
    source_code: str,
    all_nodes: Optional[List[CfgNode]] = None,
    phases: Optional[List[Dict]] = None,
) -> str:
    """
    Build a prompt that fits within MAX_PROMPT_CHARS.

    Strategy (applied in order until the prompt fits):
      1. Try WITHOUT source excerpt (nodes have raw_code; excerpt is a bonus).
         If this already fits, optionally add a compact source excerpt.
      2. If still too large, trim the context packet to CONTEXT_BUDGET.
      3. Log a warning if the prompt is still large after trimming (the
         auto-halving mechanism will handle it upstream).
    """
    # ── Attempt 1: no source excerpt ─────────────────────────────────
    prompt_no_src = build_user_prompt(
        qualified_name=func_entry.qualified_name,
        params=func_entry.params,
        description=func_entry.description,
        context_packet=context_packet,
        source_code="",          # omit by default
        nodes=batch,
        all_nodes=all_nodes,
        phases=phases,
        func_start_line=func_entry.line,
    )

    total_no_src = len(SYSTEM_PROMPT) + len(prompt_no_src)

    if total_no_src <= MAX_PROMPT_CHARS:
        # We have headroom — try to squeeze in a compact source excerpt
        excerpt = _extract_batch_source(source_code, batch, func_entry.line)
        prompt_with_src = build_user_prompt(
            qualified_name=func_entry.qualified_name,
            params=func_entry.params,
            description=func_entry.description,
            context_packet=context_packet,
            source_code=excerpt,
            nodes=batch,
            all_nodes=all_nodes,
            phases=phases,
            func_start_line=func_entry.line,
        )
        if len(SYSTEM_PROMPT) + len(prompt_with_src) <= MAX_PROMPT_CHARS:
            return prompt_with_src   # full prompt with excerpt fits
        return prompt_no_src         # excerpt didn't fit; use version without

    # ── Attempt 2: trim context packet ───────────────────────────────
    user_without_context = len(prompt_no_src) - len(context_packet)
    available = max(
        300,   # always keep at least the first few hierarchy lines
        MAX_PROMPT_CHARS - len(SYSTEM_PROMPT) - user_without_context,
    )
    trimmed_context = _trim_context(context_packet, min(available, CONTEXT_BUDGET))

    prompt_trimmed = build_user_prompt(
        qualified_name=func_entry.qualified_name,
        params=func_entry.params,
        description=func_entry.description,
        context_packet=trimmed_context,
        source_code="",
        nodes=batch,
        all_nodes=all_nodes,
        phases=phases,
        func_start_line=func_entry.line,
    )

    final_total = len(SYSTEM_PROMPT) + len(prompt_trimmed)
    if final_total > MAX_PROMPT_CHARS:
        logger.debug(
            "Prompt is %d chars (~%d tokens) after trimming — "
            "auto-halving will handle it if LLM returns no response.",
            final_total, final_total // 4,
        )

    return prompt_trimmed


def _extract_batch_source(
    full_source: str,
    batch: List[CfgNode],
    func_start_line: int,
    padding: int = SOURCE_PADDING,
) -> str:
    """
    Return only the source lines covering this batch's nodes ±padding lines.
    Each line is annotated with its absolute line number for LLM orientation.
    """
    if not full_source or not batch:
        return ""

    lines = full_source.splitlines()
    min_abs = min(n.start_line for n in batch)
    max_abs = max(n.end_line for n in batch)

    offset = func_start_line
    start_idx = max(0, min_abs - offset - padding)
    end_idx   = min(len(lines), max_abs - offset + padding + 1)

    excerpt_lines = lines[start_idx:end_idx]
    abs_first = offset + start_idx
    numbered = [f"{abs_first + i}: {ln}" for i, ln in enumerate(excerpt_lines)]
    return "\n".join(numbered)


def _trim_context(context: str, budget: int) -> str:
    """Trim context packet to budget chars, appending a truncation marker."""
    if len(context) <= budget:
        return context
    cutoff = max(0, budget - 60)
    newline_pos = context.rfind("\n", 0, cutoff)
    cut_at = newline_pos if newline_pos > cutoff // 2 else cutoff
    return context[:cut_at] + "\n… [context trimmed to fit model window]"


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def _make_batches(nodes: List[CfgNode], batch_size: int) -> List[List[CfgNode]]:
    """Simple sequential batching (kept for reference / fallback)."""
    return [nodes[i: i + batch_size] for i in range(0, len(nodes), batch_size)]


# ---------------------------------------------------------------------------
# CFG topological sort
# ---------------------------------------------------------------------------

def _topo_sort(cfg: ControlFlowGraph) -> Tuple[List[str], Set[Tuple[str, str]]]:
    """
    Iterative DFS post-order topological sort of the CFG.

    Handles cycles (loop back-edges) by detecting nodes already on the DFS
    stack.  Back-edges are recorded and returned so callers can exclude them
    from predecessor-count calculations (avoiding false merge-point detection
    on loop headers).

    Returns:
        (order, back_edges) where:
          order      — node IDs in execution order (entry first)
          back_edges — set of (source, target) pairs that are back-edges
    """
    succ: Dict[str, List[str]] = {nid: [] for nid in cfg.nodes}
    for edge in cfg.edges:
        if edge.source in succ:
            succ[edge.source].append(edge.target)

    visited: Set[str] = set()
    in_stack: Set[str] = set()
    order: List[str] = []
    back_edges: Set[Tuple[str, str]] = set()

    def _iterative_dfs(start: str) -> None:
        # Each stack frame: (node_id, iterator_over_children, entered_stack)
        # We push (nid, iter(children), False) and when we pop with
        # entered_stack=True we finalize the node (post-order).
        stack: List[Tuple[str, object, bool]] = [(start, iter(succ.get(start, [])), True)]
        in_stack.add(start)
        visited.add(start)

        while stack:
            nid, children_iter, _ = stack[-1]
            try:
                child = next(children_iter)  # type: ignore[arg-type]
                if child not in cfg.nodes:
                    continue
                if child in in_stack:
                    # Back-edge — record it but do NOT recurse
                    back_edges.add((nid, child))
                    continue
                if child in visited:
                    continue
                visited.add(child)
                in_stack.add(child)
                stack.append((child, iter(succ.get(child, [])), True))
            except StopIteration:
                # All children processed — finalize this node
                stack.pop()
                in_stack.discard(nid)
                order.append(nid)

    entry = cfg.entry_node_id
    if entry and entry in cfg.nodes:
        _iterative_dfs(entry)
    for nid in cfg.nodes:
        if nid not in visited:
            _iterative_dfs(nid)

    order.reverse()
    return order, back_edges


# ---------------------------------------------------------------------------
# CFG-region-aware batching
# ---------------------------------------------------------------------------

# Node types that mark the start/end of a decision region.
# When we encounter one, we flush the current batch and start a new one
# so that sequential ACTION sequences are never split across batches.
_BRANCH_NODE_TYPES: Set[NodeType] = {
    NodeType.DECISION,
    NodeType.LOOP_HEAD,
    NodeType.SWITCH_HEAD,
}


def _make_region_batches(
    ordered_nodes: List[CfgNode],
    cfg: ControlFlowGraph,
    batch_size: int,
    back_edges: Optional[Set[Tuple[str, str]]] = None,
) -> List[List[CfgNode]]:
    """
    Group CFG nodes into batches that respect basic-block boundaries.

    A new batch begins when:
      - A merge point is reached (node has multiple predecessors in the CFG)
      - A branch node is encountered (DECISION / LOOP_HEAD / SWITCH_HEAD)
      - The current batch reaches batch_size

    This keeps sequential ACTION sequences (e.g., all the statements that
    build a JSON document) together in one batch, giving the LLM the full
    context it needs to write coherent high-level labels.

    back_edges: set of (source, target) back-edges from _topo_sort().
    These are excluded from predecessor counting so that loop headers are
    NOT falsely treated as merge points (they only have one forward-edge
    predecessor; the back-edge comes from the loop body).

    Tiny single-node batches are merged with their neighbour where possible.
    """
    # Count forward-edge predecessors only (exclude back-edges).
    # A true merge point is one where control flows in from 2+ different
    # forward paths (e.g., after an if/else).  Loop headers only have ONE
    # forward predecessor (the code before the loop) — the second edge is
    # a back-edge from the loop body and must not inflate the count.
    _back = back_edges or set()
    pred_count: Dict[str, int] = {nid: 0 for nid in cfg.nodes}
    for edge in cfg.edges:
        if (edge.source, edge.target) not in _back and edge.target in pred_count:
            pred_count[edge.target] += 1

    batches: List[List[CfgNode]] = []
    current: List[CfgNode] = []

    for node in ordered_nodes:
        is_merge = pred_count.get(node.node_id, 0) > 1
        is_branch = node.node_type in _BRANCH_NODE_TYPES

        # Flush if we hit a merge point or the batch is full
        if current and (is_merge or len(current) >= batch_size):
            batches.append(current)
            current = []

        current.append(node)

        # Branch node ends the current region — flush after adding it
        if is_branch and current:
            batches.append(current)
            current = []

    if current:
        batches.append(current)

    return _merge_small_batches(batches, batch_size)


def _merge_small_batches(
    batches: List[List[CfgNode]],
    batch_size: int,
) -> List[List[CfgNode]]:
    """
    Merge single-node batches with a neighbour to avoid pointless 1-node calls.
    Only merges when the combined size stays within batch_size.
    """
    if not batches:
        return batches
    result: List[List[CfgNode]] = [batches[0]]
    for batch in batches[1:]:
        last = result[-1]
        if (len(last) == 1 or len(batch) == 1) and len(last) + len(batch) <= batch_size:
            result[-1] = last + batch
        else:
            result.append(batch)
    return result


# ---------------------------------------------------------------------------
# Batch callee name extractor (for targeted callee context)
# ---------------------------------------------------------------------------

def _extract_batch_callee_names(batch: List[CfgNode]) -> Set[str]:
    """
    Collect function names actually called in this batch's nodes.

    Pulls names from each node's enriched_context['function_calls'] list
    (populated by the CFG enricher).  Returns both qualified names and
    short names so build_targeted_callee_context() can find matches.
    """
    names: Set[str] = set()
    for node in batch:
        for call in node.enriched_context.get("function_calls", []):
            sig = call.get("signature", "")
            if not sig:
                continue
            # qualified name: everything before '('
            full = sig.split("(")[0].strip()
            if full:
                names.add(full)
                # Also add the short name (last segment after '::')
                short = full.split("::")[-1]
                if short:
                    names.add(short)
    return names


# ---------------------------------------------------------------------------
# Targeted retry note (only used for parse / missing-node failures)
# ---------------------------------------------------------------------------

def _build_retry_note(failures: Dict[str, str]) -> str:
    if not failures:
        return ""
    lines = [
        "\n\n=== CORRECTION REQUIRED ===",
        "Your previous response was incomplete or had invalid labels.",
        "Return ONLY a JSON object for the following node_ids that still need labels:",
    ]
    for nid, reason in sorted(failures.items()):
        lines.append(f"  {nid}: {reason}")
    lines.append("\nDo NOT include node_ids that were already successfully labeled.")
    lines.append('Return ONLY valid JSON: {"node_id": "label", ...}')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Partial response parser
# ---------------------------------------------------------------------------

def _parse_partial(
    raw: str,
    required_ids: Set[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Parse LLM response and extract valid labels.
    Returns (accepted, failures) maps.
    """
    accepted: Dict[str, str] = {}
    failures: Dict[str, str] = {}

    cleaned = _extract_json(raw)
    if not cleaned:
        for nid in required_ids:
            failures[nid] = "No JSON object found in LLM response"
        return accepted, failures

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        for nid in required_ids:
            failures[nid] = f"JSON parse error: {exc}"
        return accepted, failures

    if not isinstance(data, dict):
        for nid in required_ids:
            failures[nid] = "LLM response was not a JSON object"
        return accepted, failures

    for nid in required_ids:
        if nid not in data:
            failures[nid] = f"node_id '{nid}' missing from response"
            continue
        label = data[nid]
        if not isinstance(label, str):
            failures[nid] = f"label must be a string, got {type(label).__name__}"
            continue
        if not label.strip():
            failures[nid] = "label is empty"
            continue
        if len(label) > _MAX_LABEL_LEN:
            failures[nid] = f"label too long ({len(label)} > {_MAX_LABEL_LEN} chars)"
            continue
        accepted[nid] = label

    return accepted, failures


def _extract_json(text: str) -> Optional[str]:
    """Extract the first complete JSON object from a potentially noisy response."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return None


# ---------------------------------------------------------------------------
# Fallback detection helper
# ---------------------------------------------------------------------------

def _looks_like_fallback(node: CfgNode) -> bool:
    label = node.label or ""
    fallback_prefixes = ("Check: ", "Loop: ", "Switch on: ", "Case: ",
                         "Handle exception: ")
    return label.startswith(tuple(fallback_prefixes))


# ---------------------------------------------------------------------------
# Rule-based fallback label generation
# ---------------------------------------------------------------------------

def _fallback_label(node: CfgNode) -> str:
    """Generate a minimal correct label without LLM."""
    raw = node.raw_code.strip()[:120]

    if node.node_type == NodeType.DECISION:
        return f"Check: {raw}?" if not raw.endswith("?") else raw
    if node.node_type == NodeType.LOOP_HEAD:
        return f"Loop: {raw}?"
    if node.node_type == NodeType.SWITCH_HEAD:
        return f"Switch on: {raw}?"
    if node.node_type == NodeType.CASE:
        return f"Case: {raw}"
    if node.node_type == NodeType.DEFAULT_CASE:
        return "Default case"
    if node.node_type == NodeType.RETURN:
        return f"Return {raw.removeprefix('return').strip().rstrip(';')}"
    if node.node_type == NodeType.BREAK:
        return "Exit loop"
    if node.node_type == NodeType.CONTINUE:
        return "Continue to next iteration"
    if node.node_type == NodeType.TRY_HEAD:
        return "Execute with exception handling"
    if node.node_type == NodeType.CATCH:
        return f"Handle exception: {raw}"

    first_line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), raw)
    return first_line[:80]
