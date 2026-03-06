"""
Prompt templates and builder for LLM label generation.

Design principles:
  - System prompt defines the engineer role + strict label rules
  - User prompt injects function-scoped context (params, callees, source)
  - Output is always a JSON object: { "node_id": "label", ... }
  - No invented logic — LLM only translates code constructs to English
"""

import json
from typing import Dict, List

from models import CfgNode, NodeType


# ---------------------------------------------------------------------------
# System prompt (stable across all functions / projects)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior C++ software engineer writing flowchart labels for formal technical documentation.

Your sole task is to convert C++ code constructs into clear, concise English flowchart labels.
You have full context of the project: function purpose, called function descriptions, and inline comments.

=== LABEL WRITING RULES ===

1. VOICE & TENSE
   - Always use present-tense active voice.
   - Examples: "Check if X", "Initialize Y", "Iterate over list", "Return result"

2. ACTION nodes  (non-decision sequential code)
   - Short imperative sentence(s), max 10 words per line.
   - Use <br/> to separate distinct actions if the node contains multiple statements.
   - Maximum 3 lines per label.
   - Preserve exact function call names exactly as they appear in code.

3. DECISION nodes  (if-conditions, loop conditions, switch)
   - Must be phrased as a Yes/No question ending with "?".
   - Keep it concise — one line if possible.
   - Example: "Is rate limit exceeded for this event?"

4. LOOP_HEAD nodes  (for / while / do-while conditions)
   - Phrase as a question or iteration statement ending with "?".
   - Example: "For each worker in worker list?" or "While queue is not empty?"

5. SWITCH_HEAD nodes
   - Describe what is being switched on.
   - Example: "Based on event type?"

6. CASE / DEFAULT_CASE nodes
   - State the case value in plain English.
   - Example: "Case: BackendEvent::GC" or "Default case"

7. RETURN nodes
   - Format: "Return <what is returned>"
   - Example: "Return true (limit exceeded)" or "Return nullptr"

8. TRY_HEAD nodes
   - Label: "Execute with exception handling"

9. CATCH nodes
   - Format: "Handle <exception type> exception"

10. BREAK / CONTINUE
    - Label: "Exit loop" for break, "Continue to next iteration" for continue

=== STRICT RULES (NEVER VIOLATE) ===
- Never rename or paraphrase function call names.
- Never invent logic not present in the source code.
- Use project terminology exactly as provided in the context section.
- Do not add explanatory text outside the label itself.

=== OUTPUT FORMAT ===
Return ONLY a valid JSON object — no markdown, no code fences, no explanation.
Keys are node_id strings. Values are label strings.

Example:
{"N2": "Is rate limit exceeded for event?", "N3": "Return 1 (limit exceeded)", "N4": "Return 0 (within limit)"}
"""


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(
    qualified_name: str,
    params: List[Dict],
    description: str,
    context_packet: str,
    source_code: str,
    nodes: List[CfgNode],
) -> str:
    """
    Build the per-function user prompt.

    Includes:
      - Function identity (name + params)
      - Project context (PKB context packet)
      - Full function source
      - Structured node list for labeling
    """
    parts: List[str] = []

    param_str = ", ".join(
        f"{p.get('type', '')} {p.get('name', '')}".strip() for p in params
    )
    parts.append(f"Function: {qualified_name}({param_str})")

    if description:
        parts.append(f"Purpose: {description}")

    if context_packet:
        parts.append(f"\n--- Project Context ---\n{context_packet}")

    parts.append(f"\n--- Function Source Code ---\n{source_code}")

    # Build node descriptions for labeling
    node_list = _build_node_list(nodes)
    node_json = json.dumps(node_list, indent=2)
    parts.append(f"\n--- Nodes to Label ---\n{node_json}")

    parts.append(
        "\nReturn ONLY the JSON object mapping each node_id to its label."
    )

    return "\n".join(parts)


def _build_node_list(nodes: List[CfgNode]) -> List[Dict]:
    """Convert CfgNodes into a structured list for the LLM prompt."""
    result = []
    for node in nodes:
        # Skip structural nodes that don't need LLM labels
        if node.node_type in (NodeType.START, NodeType.END):
            continue

        entry: Dict = {
            "node_id": node.node_id,
            "type": node.node_type.value,
            "raw_code": node.raw_code[:300],  # cap to avoid huge prompts
        }

        # Inject enriched context inline if available
        ctx = node.enriched_context
        if ctx.get("function_calls"):
            entry["called_functions"] = [
                f"{c['signature']}: {c['description']}"
                if c.get("description")
                else c["signature"]
                for c in ctx["function_calls"]
            ]
        if ctx.get("inline_comment"):
            entry["source_comment"] = ctx["inline_comment"]

        result.append(entry)

    return result
