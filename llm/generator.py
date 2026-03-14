"""
LLM-based flowchart label generator.

Batching strategy
-----------------
Large functions (many CFG nodes) produce prompts that exceed local LLM context
windows (~4 k–8 k tokens).  The old approach sent the FULL function source code
in every batch call.  For a 72-line function this is ~3200 chars — repeated for
every batch of 8 nodes — consuming most of the available context budget before
even the nodes are included.

Fix: each batch receives only the source lines that cover its own nodes (plus
±SOURCE_PADDING lines for context), not the full function source.  The context
packet (hierarchy + callee graph + function purpose) is kept in full, but is
capped at CONTEXT_BUDGET chars if the total prompt still exceeds MAX_PROMPT_CHARS.

Budget logic
------------
MAX_PROMPT_CHARS    — total system + user prompt character limit.
                      Conservative for 4 k-token local models.
                      (chars ÷ 4 ≈ tokens;  6000 chars ≈ 1500 tokens → fits
                       in 4096-token window with room for the JSON output.)
CONTEXT_BUDGET      — max chars for the context packet; trimmed if over budget.
SOURCE_PADDING      — lines above/below batch nodes included in excerpt.
BATCH_SIZE          — default nodes per LLM call.

Targeted retry
--------------
On validation failure the retry prompt lists the exact failing node_ids and
the reason so the LLM corrects precisely rather than regenerating everything.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from llm.client import LlmClient
from llm.prompts import SYSTEM_PROMPT, build_user_prompt
from models import CfgNode, ControlFlowGraph, FunctionEntry, NodeType
from pkb.builder import ProjectKnowledgeBase

logger = logging.getLogger(__name__)

# ── Prompt-size budget constants ─────────────────────────────────────────────

# Maximum total characters (system + user prompt) sent to the LLM.
# The system prompt alone is ~3400 chars.  8 nodes × 400 chars = 3200 chars.
# Source excerpt + context adds ~3000 chars.  Total ≈ 9600 chars ≈ 2400 tokens.
# Modern local models (llama3, etc.) support 8k–32k token windows.
# Even the smallest common window (4096 tokens) has room for 2400-token prompts
# with ~1700 tokens left for JSON output.
#
# Set to 10000 chars (≈ 2500 tokens).  This is the realistic working budget.
# Context and source excerpt are still trimmed proportionally above this limit
# so the generator handles even edge cases gracefully.
MAX_PROMPT_CHARS = 10000

# Maximum chars reserved for the context packet (hierarchy + callee graph).
# The 4-level BFS can produce hundreds of entries; cap at 2500 chars so it
# doesn't dominate the context window.
CONTEXT_BUDGET = 2500

# Lines of source code to include above and below the batch's node range.
SOURCE_PADDING = 8

# ── Batch size ────────────────────────────────────────────────────────────────

# Default nodes per LLM call.  Can be overridden via --llm-batch-size.
# Smaller values reduce prompt size; larger values reduce LLM call count.
BATCH_SIZE = 8

# Maximum characters per label (validation threshold)
_MAX_LABEL_LEN = 300


class LabelGenerator:
    """
    Generates LLM labels for all nodes in a CFG.

    Batches nodes (BATCH_SIZE per call).  Each batch sends:
      - Full context packet (hierarchy + callee graph) — capped at CONTEXT_BUDGET
      - Source excerpt covering only this batch's node lines (±SOURCE_PADDING)
      - This batch's nodes only

    Targeted retry per batch: lists exactly which node_ids failed and why.
    Falls back to rule-based labels only for nodes that exhaust all retries.
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
        """
        Fill cfg.nodes[*].label for every non-sentinel node.
        Modifies nodes in-place.
        """
        labelable = [n for n in cfg.nodes.values()
                     if n.node_type not in (NodeType.START, NodeType.END)]
        if not labelable:
            return

        # Build project context packet once — shared across all batch calls
        context_packet = self._pkb.build_context_packet(func_entry, base_path)

        # Split into batches and label each independently
        batches = _make_batches(labelable, self._batch_size)
        label_map: Dict[str, str] = {}

        for batch_idx, batch in enumerate(batches):
            batch_labels = self._label_batch(
                batch=batch,
                func_entry=func_entry,
                context_packet=context_packet,
                source_code=source_code,
            )
            label_map.update(batch_labels)
            logger.debug(
                "Batch %d/%d done (%d nodes) for '%s'",
                batch_idx + 1, len(batches), len(batch), func_entry.qualified_name,
            )

        self._apply_labels(cfg, label_map)

        # START / END get simple fixed labels
        for node in cfg.nodes.values():
            if node.node_type == NodeType.START:
                node.label = f"Start: {func_entry.qualified_name.split('::')[-1]}"
            elif node.node_type == NodeType.END:
                node.label = "End"

        # Log fallback usage
        fallback_count = sum(1 for n in labelable if _looks_like_fallback(n))
        if fallback_count > 0:
            logger.warning(
                "'%s': %d/%d node(s) used fallback labels",
                func_entry.qualified_name, fallback_count, len(labelable),
            )
        else:
            logger.debug(
                "'%s': all %d nodes labeled by LLM",
                func_entry.qualified_name, len(labelable),
            )

    # ------------------------------------------------------------------
    # Per-batch labeling with targeted retry
    # ------------------------------------------------------------------

    def _label_batch(
        self,
        batch: List[CfgNode],
        func_entry: FunctionEntry,
        context_packet: str,
        source_code: str,
    ) -> Dict[str, str]:
        """
        Label a single batch of nodes.  Returns node_id → label map.
        Falls back to rule-based labels only for nodes that keep failing.
        """
        # ── Build size-aware prompt ───────────────────────────────────
        base_prompt = _build_size_aware_prompt(
            batch=batch,
            func_entry=func_entry,
            context_packet=context_packet,
            source_code=source_code,
        )

        # Log estimated token count at DEBUG level
        total_chars = len(SYSTEM_PROMPT) + len(base_prompt)
        logger.debug(
            "Batch prompt size: %d chars (~%d tokens) for %d nodes",
            total_chars, total_chars // 4, len(batch),
        )

        required_ids = {n.node_id for n in batch}
        accumulated: Dict[str, str] = {}
        remaining: Set[str] = set(required_ids)
        last_failures: Dict[str, str] = {}

        for attempt in range(1, self._max_retries + 2):
            prompt = base_prompt if attempt == 1 else base_prompt + _build_retry_note(last_failures)

            raw = self._client.generate(SYSTEM_PROMPT, prompt)
            if raw is None:
                prompt_chars = len(SYSTEM_PROMPT) + len(prompt)
                logger.warning(
                    "Batch attempt %d/%d: no LLM response "
                    "(prompt=%d chars ~%d tokens). "
                    "If this keeps failing, try: --llm-batch-size 4 --llm-timeout 180",
                    attempt, self._max_retries + 1,
                    prompt_chars, prompt_chars // 4,
                )
                last_failures = {nid: "LLM returned no response" for nid in remaining}
                continue

            partial, failures = _parse_partial(raw, remaining)
            accumulated.update(partial)
            remaining -= set(partial.keys())
            last_failures = failures

            if not remaining:
                logger.debug("Batch fully labeled on attempt %d", attempt)
                return accumulated

            logger.warning(
                "Batch attempt %d/%d: %d node(s) still need labels — %s",
                attempt, self._max_retries + 1, len(remaining),
                {nid: failures.get(nid, "missing") for nid in remaining},
            )

        # All retries exhausted — fallback for remaining nodes only
        if remaining:
            node_by_id = {n.node_id: n for n in batch}
            for nid in remaining:
                node = node_by_id.get(nid)
                if node:
                    accumulated[nid] = _fallback_label(node)
            logger.warning(
                "Fallback labels applied for %d node(s): %s",
                len(remaining), sorted(remaining),
            )

        return accumulated

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
) -> str:
    """
    Build a prompt that fits within MAX_PROMPT_CHARS.

    Reduction strategy applied in order:
      1. Source excerpt — send only the lines covering this batch's nodes
         (±SOURCE_PADDING lines) instead of the full function source.
         This reduces source contribution from ~3000 chars to ~500–1500 chars.
      2. Context cap — trim the context packet to CONTEXT_BUDGET chars if the
         combined prompt still exceeds MAX_PROMPT_CHARS.
      3. Hard minimum — always send at least 400 chars of context (first few
         lines contain the most important hierarchy info).

    The node list raw_code is capped at 200 chars per node throughout.
    """
    # Step 1 — build with source excerpt instead of full source
    source_excerpt = _extract_batch_source(source_code, batch, func_entry.line)

    prompt = build_user_prompt(
        qualified_name=func_entry.qualified_name,
        params=func_entry.params,
        description=func_entry.description,
        context_packet=context_packet,
        source_code=source_excerpt,
        nodes=batch,
    )

    total = len(SYSTEM_PROMPT) + len(prompt)
    if total <= MAX_PROMPT_CHARS:
        return prompt

    # Step 2 — trim context packet
    # Calculate how many chars the user prompt occupies WITHOUT context,
    # then figure out how much room is left for context.
    user_without_context = len(prompt) - len(context_packet)
    available_for_context = max(
        400,  # hard minimum — always include at least the hierarchy lines
        MAX_PROMPT_CHARS - len(SYSTEM_PROMPT) - user_without_context,
    )
    trimmed_context = _trim_context(
        context_packet,
        min(available_for_context, CONTEXT_BUDGET),
    )

    prompt = build_user_prompt(
        qualified_name=func_entry.qualified_name,
        params=func_entry.params,
        description=func_entry.description,
        context_packet=trimmed_context,
        source_code=source_excerpt,
        nodes=batch,
    )

    final_total = len(SYSTEM_PROMPT) + len(prompt)
    if final_total > MAX_PROMPT_CHARS:
        # Prompt is still large but within reason for models with 8k+ windows.
        # Log so the user knows they can tune --llm-batch-size if needed.
        logger.debug(
            "Prompt is %d chars (~%d tokens) after trimming. "
            "If LLM times out, try --llm-batch-size 4 to reduce further.",
            final_total, final_total // 4,
        )

    return prompt


def _extract_batch_source(
    full_source: str,
    batch: List[CfgNode],
    func_start_line: int,
    padding: int = SOURCE_PADDING,
) -> str:
    """
    Return only the source lines that cover this batch's nodes ±padding lines.

    Line numbers in CfgNode are absolute (matching the source file).
    full_source starts at func_start_line.

    Prepends each line with its absolute line number so the LLM can orient
    itself within the function (e.g. "line 347: if (diskId == \"\")").
    """
    if not full_source or not batch:
        return full_source

    lines = full_source.splitlines()

    # Absolute line range for nodes in this batch
    min_abs = min(n.start_line for n in batch)
    max_abs = max(n.end_line for n in batch)

    # Convert to 0-based indices within full_source
    offset = func_start_line  # absolute line of full_source[0]
    start_idx = max(0, min_abs - offset - padding)
    end_idx   = min(len(lines), max_abs - offset + padding + 1)

    excerpt_lines = lines[start_idx:end_idx]

    # Annotate with absolute line numbers for LLM orientation
    abs_first = offset + start_idx
    numbered = [f"{abs_first + i}: {ln}" for i, ln in enumerate(excerpt_lines)]

    return "\n".join(numbered)


def _trim_context(context: str, budget: int) -> str:
    """Trim context packet to budget chars, appending a truncation marker."""
    if len(context) <= budget:
        return context
    cutoff = max(0, budget - 60)
    # Try to cut at a newline boundary
    newline_pos = context.rfind("\n", 0, cutoff)
    cut_at = newline_pos if newline_pos > cutoff // 2 else cutoff
    return context[:cut_at] + "\n... [context trimmed to fit model window]"


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def _make_batches(nodes: List[CfgNode], batch_size: int) -> List[List[CfgNode]]:
    return [nodes[i: i + batch_size] for i in range(0, len(nodes), batch_size)]


# ---------------------------------------------------------------------------
# Targeted retry note
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
    Parse LLM response and extract valid labels for as many nodes as possible.
    Returns (accepted, failures) where failures maps node_id → reason string.
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
