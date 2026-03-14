"""
LLM-based flowchart label generator.

Processes one function at a time using batched LLM calls.

Batching strategy
-----------------
Large functions (many CFG nodes) would produce prompts that exceed a local
LLM's context window (~4 k–8 k tokens).  When the prompt is too large the LLM
either truncates its response (missing required node_ids) or times out — both
of which previously caused silent fallback to rule-based labels.

Fix: split the labelable node list into batches of at most BATCH_SIZE nodes.
Every batch call sends the FULL context (hierarchy + callee graph + source
code) so the LLM always understands the function, but labels only a small
slice of nodes.  Partial results from each batch are merged into a single
label map.

Targeted retry
--------------
On validation failure the retry prompt includes the exact list of failing
node_ids and the specific reason for each failure so the LLM can correct
precisely rather than regenerating everything.
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

# Maximum characters for a single label
_MAX_LABEL_LEN = 300

# Maximum nodes per LLM call.  Keeps prompts well within 4 k-token windows
# even for functions with long source code and rich context packets.
# Each node contributes ~200–400 chars of JSON; 8 nodes ≈ 2 k–3 k chars of
# node list, leaving plenty of room for the context packet + source code.
BATCH_SIZE = 8


class LabelGenerator:
    """
    Generates LLM labels for all nodes in a CFG.

    Batches nodes into groups of BATCH_SIZE per LLM call so that every call
    fits within the local model's context window.  Full project context
    (hierarchy, callee graph, source code) is included in every batch call.

    Retry strategy per batch:
      - Attempt 1: normal prompt
      - Attempt 2+: targeted prompt listing exactly which node_ids failed and why
      - After max_retries exhausted for a batch: rule-based fallback for
        that batch only (other batches are unaffected)
    """

    def __init__(self, client: LlmClient, pkb: ProjectKnowledgeBase,
                 max_retries: int = 2) -> None:
        self._client = client
        self._pkb = pkb
        self._max_retries = max_retries

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
        batches = _make_batches(labelable, BATCH_SIZE)
        label_map: Dict[str, str] = {}

        for batch_idx, batch in enumerate(batches):
            logger.debug(
                "Labeling batch %d/%d (%d nodes) for '%s'",
                batch_idx + 1, len(batches), len(batch), func_entry.qualified_name,
            )
            batch_labels = self._label_batch(
                batch=batch,
                func_entry=func_entry,
                context_packet=context_packet,
                source_code=source_code,
            )
            label_map.update(batch_labels)

        self._apply_labels(cfg, label_map)

        # START / END get simple fixed labels
        for node in cfg.nodes.values():
            if node.node_type == NodeType.START:
                node.label = f"Start: {func_entry.qualified_name.split('::')[-1]}"
            elif node.node_type == NodeType.END:
                node.label = "End"

        # Log how many nodes used LLM vs fallback
        llm_count = sum(1 for n in labelable if n.label and not _is_fallback(n))
        fallback_count = len(labelable) - llm_count
        if fallback_count > 0:
            logger.warning(
                "'%s': %d/%d nodes used fallback labels (LLM failed for those nodes)",
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
        Label a single batch of nodes.  Returns node_id → label map for the batch.
        Falls back to rule-based labels only for nodes that keep failing.
        """
        # Build the base prompt for this batch (full context + this batch's nodes)
        base_prompt = build_user_prompt(
            qualified_name=func_entry.qualified_name,
            params=func_entry.params,
            description=func_entry.description,
            context_packet=context_packet,
            source_code=source_code,
            nodes=batch,
        )

        required_ids = {n.node_id for n in batch}
        accumulated: Dict[str, str] = {}   # partial results saved across attempts
        remaining: Set[str] = set(required_ids)
        last_failures: Dict[str, str] = {} # node_id → reason, for targeted retry

        for attempt in range(1, self._max_retries + 2):
            if attempt == 1:
                prompt = base_prompt
            else:
                # Build targeted retry prompt: only ask for the still-failing nodes
                retry_note = _build_retry_note(last_failures)
                prompt = base_prompt + retry_note

            raw = self._client.generate(SYSTEM_PROMPT, prompt)
            if raw is None:
                logger.warning(
                    "Batch attempt %d/%d: no LLM response",
                    attempt, self._max_retries + 1,
                )
                last_failures = {nid: "LLM returned no response" for nid in remaining}
                continue

            # Parse whatever the LLM returned — accept partial results
            partial, failures = _parse_partial(raw, remaining)
            accumulated.update(partial)
            remaining -= set(partial.keys())
            last_failures = failures

            if not remaining:
                logger.debug("Batch fully labeled on attempt %d", attempt)
                return accumulated

            # Some nodes still missing/invalid — log and retry
            logger.warning(
                "Batch attempt %d/%d: %d node(s) still need labels: %s",
                attempt, self._max_retries + 1,
                len(remaining),
                {nid: failures.get(nid, "missing") for nid in remaining},
            )

        # All retries exhausted — use fallback for remaining nodes only
        if remaining:
            node_by_id = {n.node_id: n for n in batch}
            for nid in remaining:
                node = node_by_id.get(nid)
                if node:
                    accumulated[nid] = _fallback_label(node)
            logger.warning(
                "Fallback labels used for %d node(s): %s",
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
# Batch construction
# ---------------------------------------------------------------------------

def _make_batches(nodes: List[CfgNode], batch_size: int) -> List[List[CfgNode]]:
    """Split node list into batches of at most batch_size."""
    return [nodes[i: i + batch_size] for i in range(0, len(nodes), batch_size)]


# ---------------------------------------------------------------------------
# Targeted retry note builder
# ---------------------------------------------------------------------------

def _build_retry_note(failures: Dict[str, str]) -> str:
    """
    Build the targeted retry instruction appended to the base prompt.

    Lists exactly which node_ids failed and why so the LLM can correct
    precisely without regenerating labels that already succeeded.
    """
    if not failures:
        return ""

    lines = [
        "\n\n=== CORRECTION REQUIRED ===",
        "Your previous response was incomplete or invalid.",
        "Return ONLY a JSON object containing the following node_ids that still need labels:",
    ]
    for nid, reason in sorted(failures.items()):
        lines.append(f"  {nid}: {reason}")
    lines.append(
        "\nDo NOT include node_ids that were already successfully labeled."
    )
    lines.append(
        "Return ONLY valid JSON: {\"node_id\": \"label\", ...}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Partial response parser
# ---------------------------------------------------------------------------

def _parse_partial(
    raw: str,
    required_ids: Set[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Parse an LLM response and extract valid labels for as many nodes as possible.

    Returns:
        accepted  — node_id → label for nodes that passed validation
        failures  — node_id → failure reason for nodes that are still missing/invalid
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

    # Validate each required node individually so partial success is possible
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
    """
    Extract the first complete JSON object from a potentially
    noisy LLM response (may contain markdown fences or extra text).
    """
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
# Fallback detection helper (for logging)
# ---------------------------------------------------------------------------

def _is_fallback(node: CfgNode) -> bool:
    """Heuristic: label looks like it came from the rule-based fallback."""
    label = node.label or ""
    raw = node.raw_code.strip()[:80]
    # If label starts with raw code or fallback prefixes, likely a fallback
    fallback_prefixes = ("Check: ", "Loop: ", "Switch on: ", "Case: ",
                         "Handle exception: ", "Return ")
    return label.startswith(tuple(fallback_prefixes)) or label == raw


# ---------------------------------------------------------------------------
# Rule-based fallback label generation
# ---------------------------------------------------------------------------

def _fallback_label(node: CfgNode) -> str:
    """Generate a minimal but correct label without LLM."""
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
