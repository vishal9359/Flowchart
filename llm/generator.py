"""
LLM-based flowchart label generator.

Processes one function at a time (all nodes in a single LLM call).
Validates the response, retries if invalid, falls back to rule-based
labels on repeated failure — ensuring the pipeline never stalls.
"""

import json
import logging
import re
from typing import Dict, List, Optional

from llm.client import LlmClient
from llm.prompts import SYSTEM_PROMPT, build_user_prompt
from models import CfgNode, ControlFlowGraph, FunctionEntry, NodeType
from pkb.builder import ProjectKnowledgeBase

logger = logging.getLogger(__name__)

# Maximum characters for a single label before it is considered invalid
_MAX_LABEL_LEN = 300


class LabelGenerator:
    """
    Generates LLM labels for all nodes in a CFG.

    One LLM call per function:
      - Injects function context packet from PKB
      - Injects enriched per-node context (called functions, comments)
      - Validates response schema
      - Retries up to max_retries with targeted feedback
      - Falls back to rule-based labels if LLM repeatedly fails
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

        # Build project context packet from PKB
        context_packet = self._pkb.build_context_packet(func_entry, base_path)

        user_prompt = build_user_prompt(
            qualified_name=func_entry.qualified_name,
            params=func_entry.params,
            description=func_entry.description,
            context_packet=context_packet,
            source_code=source_code,
            nodes=labelable,
        )

        label_map = self._call_with_retry(user_prompt, labelable)
        self._apply_labels(cfg, label_map)

        # START / END get simple fixed labels
        for node in cfg.nodes.values():
            if node.node_type == NodeType.START:
                node.label = f"Start: {func_entry.qualified_name.split('::')[-1]}"
            elif node.node_type == NodeType.END:
                node.label = "End"

    # ------------------------------------------------------------------
    # LLM call + retry logic
    # ------------------------------------------------------------------

    def _call_with_retry(self, user_prompt: str,
                         nodes: List[CfgNode]) -> Dict[str, str]:
        """Call LLM, validate, retry on failure. Returns node_id → label map."""
        required_ids = {n.node_id for n in nodes}
        last_error = ""

        for attempt in range(1, self._max_retries + 2):  # +2 = retries + 1 final
            if attempt > 1:
                retry_note = (
                    f"\n\nPREVIOUS ATTEMPT FAILED: {last_error}\n"
                    "Fix ONLY the failing nodes. Return the complete JSON."
                )
                prompt = user_prompt + retry_note
            else:
                prompt = user_prompt

            raw = self._client.generate(SYSTEM_PROMPT, prompt)
            if raw is None:
                last_error = "LLM returned no response"
                logger.warning("Attempt %d: no LLM response", attempt)
                continue

            label_map, error = _parse_and_validate(raw, required_ids)
            if label_map is not None:
                logger.debug("Labels generated on attempt %d", attempt)
                return label_map

            last_error = error or "unknown validation error"
            logger.warning("Attempt %d validation failed: %s", attempt, last_error)

        # All retries exhausted — fall back to rule-based labels
        logger.warning("LLM labeling failed after %d attempts; using fallback labels",
                       self._max_retries + 1)
        return {n.node_id: _fallback_label(n) for n in nodes}

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
# Response parsing and validation
# ---------------------------------------------------------------------------

def _parse_and_validate(raw: str,
                        required_ids: set) -> tuple:
    """
    Parse LLM response as JSON and validate it.
    Returns (label_map, None) on success, (None, error_str) on failure.
    """
    cleaned = _extract_json(raw)
    if not cleaned:
        return None, "No JSON object found in response"

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"

    if not isinstance(data, dict):
        return None, "Response is not a JSON object"

    # All required node_ids must be present
    missing = required_ids - set(data.keys())
    if missing:
        return None, f"Missing node IDs: {sorted(missing)}"

    # Validate each value
    for nid, label in data.items():
        if not isinstance(label, str):
            return None, f"Label for {nid} is not a string"
        if not label.strip():
            return None, f"Empty label for {nid}"
        if len(label) > _MAX_LABEL_LEN:
            return None, f"Label for {nid} exceeds {_MAX_LABEL_LEN} chars"

    return data, None


def _extract_json(text: str) -> Optional[str]:
    """
    Extract the first complete JSON object from a potentially
    noisy LLM response (may contain markdown fences or extra text).
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)

    # Find first { ... } block
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
# Rule-based fallback label generation
# ---------------------------------------------------------------------------

def _fallback_label(node: CfgNode) -> str:
    """Generate a minimal but correct label without LLM."""
    raw = node.raw_code.strip()[:120]  # truncate for safety

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

    # ACTION — first non-empty line of raw code
    first_line = next((l.strip() for l in raw.splitlines() if l.strip()), raw)
    return first_line[:80]
