"""
Prompt templates and builder for LLM label generation.

Design principles:
  - System prompt defines the engineer role + strict label rules
  - User prompt injects function-scoped context (params, callees, source)
  - Output is always a JSON object: { "node_id": "label", ... }
  - No invented logic — LLM only translates code constructs to English
"""

import json
import re
from typing import Dict, List, Optional, Set

from models import CfgNode, NodeType


# ---------------------------------------------------------------------------
# C++ keywords to exclude from data-flow identifier extraction
# ---------------------------------------------------------------------------

_CPP_KEYWORDS: Set[str] = {
    "if", "else", "while", "for", "do", "switch", "case", "default",
    "return", "break", "continue", "goto", "throw", "try", "catch",
    "new", "delete", "nullptr", "true", "false", "this", "class", "struct",
    "public", "private", "protected", "virtual", "override", "const",
    "static", "auto", "int", "void", "bool", "char", "double", "float",
    "long", "short", "unsigned", "signed", "namespace", "using", "typedef",
    "template", "typename", "operator", "sizeof", "decltype", "explicit",
    "inline", "extern", "volatile", "enum", "union", "std", "nullptr",
    "and", "or", "not", "xor", "bitand", "bitor", "compl",
}


# ---------------------------------------------------------------------------
# System prompt (stable across all functions / projects)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior C++ software engineer writing flowchart labels for formal technical documentation.

Your sole task is to convert C++ code constructs into clear, concise English flowchart labels.

You are provided with rich project context at four levels of understanding:
  [Project]  — overall purpose of the software system
  [Module]   — purpose of the directory/subsystem this file belongs to
  [File]     — responsibility of the specific source file being flowcharted
  [Function] — what the function being flowcharted does

Use this hierarchy to understand domain terminology and intent before writing any label.
For example: if the project is a storage OS and the module is QoS management, then
"activeJRts >= MAX_IU_COUNT" should become "Is active journal task count at maximum I/O unit limit?"
— not a literal copy of the code.

You are also provided with a 4-level call graph context showing what functions are called,
what those callees call, and so on up to 4 levels deep.  Use callee descriptions to
understand what each function call in the code actually does.

=== LABEL WRITING RULES ===

1. VOICE & TENSE
   - Always use present-tense active voice.
   - Examples: "Check if X", "Initialize Y", "Iterate over list", "Return result"

2. ACTION nodes  (non-decision sequential code)
   - Short imperative sentence(s), max 10 words per line.
   - Each line separated by <br/> must be AT MOST 40 characters long.
     If a description naturally exceeds 40 characters, split it at a logical
     word boundary and continue on the next <br/> line.
   - Maximum 3 lines per label (3 <br/>-separated segments).
   - Preserve exact function call names exactly as they appear in code.
   - MANDATORY: If the node has a "called_functions" list, your label MUST
     explicitly name at least one of those functions.  Never omit a project
     function call that appears in called_functions.
     Example: called_functions contains "ServerReplicate" → label must say
     "Call ServerReplicate(...)" or "Replicate server state via ServerReplicate"
   - For assignment statements (obj.field = val, var = expr), write as:
       "Set <left-hand side description> to <right-hand side description>"
     Use struct_member_context (when provided) to find the meaning of the field.
     Example: "prt.repFactor = repFactor" → "Set partition replication factor"
   - For format strings containing positional placeholders ({}, {0}, %s, %d, etc.),
     replace each placeholder with the corresponding argument variable name.
     Example: TRACE_DEBUG("Item {} active", ldu) → "Log debug: Item ldu active"
   - For logging macro calls (debug/info/warning/error), write as:
       "Log <level>: <message with placeholders resolved>"

3. DECISION nodes  (if-conditions, loop conditions, switch)
   - Must be phrased as a Yes/No question ending with "?".
   - Keep it concise — one line if possible, at most 40 characters.
   - Use project hierarchy context to translate domain variables and constants to English.
   - If callee context or macro context explains a constant, use that meaning.
   - For unknown function calls, infer intent from camelCase decomposition of the name.
     Example: IsLimitExceeded → "Is the limit exceeded?"
   - Example: "Is rate limit exceeded for this event?"

4. LOOP_HEAD nodes  (for / while / do-while conditions)
   - Phrase as a question or iteration statement ending with "?".
   - At most 40 characters.
   - Example: "For each worker in worker list?" or "While queue is not empty?"

5. SWITCH_HEAD nodes
   - Describe what is being switched on.
   - Example: "Based on event type?"

6. RETURN nodes
   - Format: "Return <what is returned>"
   - Example: "Return true (limit exceeded)" or "Return nullptr"

7. TRY_HEAD nodes
   - Label: "Execute with exception handling"

8. CATCH nodes
   - Format: "Handle <exception type> exception"

9. BREAK / CONTINUE
   - Label: "Exit loop" for break, "Continue to next iteration" for continue

=== ABSTRACTION GUIDELINE ===

Think like a senior engineer reviewing this function — describe WHAT the code
achieves, not HOW each statement is written.

When consecutive ACTION nodes collectively accomplish one high-level task (e.g.,
building a JSON object, initializing a struct, populating a request), each node's
label should describe its contribution to that goal — not just repeat the
statement literally.

  Bad (too literal):
    N3: "Call doc.SetObject()"
    N4: "Call doc.AddMember with key initiator"
    N5: "Call doc.AddMember with key count"
  Good (high-level, using phase_hint and context):
    N3: "Initialize JSON document as object"
    N4: "Add initiator ID to JSON body"
    N5: "Add I/O unit count to JSON body"

If a node has a "phase_hint" field, use that description as the thematic
framing for the label — it tells you what this section of code is trying to
accomplish at a high level.

If a node has "preceding_node_code" or "following_node_code", use them to
understand whether this node is part of a sequence — and label it accordingly
as a step in that sequence rather than in isolation.

If a node has "data_flow_shared", those are variable/object names that appear
in multiple nodes in this batch — they represent shared state being built up
across the sequence.  Use this to understand the collective purpose of the
group and label each node as a step contributing to that shared goal.
Example: if "doc" appears in N3, N4, N5 as data_flow_shared, label them as
sequential steps building the same document: "Initialize doc", "Add X to doc",
"Add Y to doc" — not as isolated statements.

If a node has "called_by" context, those are the higher-level functions that
invoke the current function.  Use them to understand the semantic purpose
of this function from the caller's perspective.

Non-negotiable output rule:
  EVERY node_id listed in "Nodes to Label" MUST appear as a key in your JSON
  output.  Do not skip any node, even simple ones.

=== STRICT RULES (NEVER VIOLATE) ===
- Never rename or paraphrase function call names.
- Never invent logic not present in the source code.
- Use project hierarchy terminology ([Project], [Module], [File]) to inform labels.
- Use callee descriptions from "Called Functions Context" to understand what calls do.
- When a node has called_functions, the label MUST name at least one of those functions.
- When struct_member_context is provided, use the member descriptions to enrich labels.
- When macro_context is provided, use the macro value/meaning to enrich labels.
- Do not add explanatory text outside the label itself.
- Each <br/>-separated line in a label must be at most 40 characters.
- Every node_id in "Nodes to Label" MUST have an entry in the output JSON.
- Never skip a DECISION, LOOP_HEAD, SWITCH_HEAD, CASE, RETURN, BREAK, or CONTINUE node.

=== OUTPUT FORMAT ===
Return ONLY a valid JSON object — no markdown, no code fences, no explanation.
Keys are node_id strings. Values are label strings.

Example:
{"N2": "Is rate limit exceeded?", "N3": "Return 1 (limit exceeded)", "N4": "Return 0 (within limit)"}
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
    all_nodes: Optional[List[CfgNode]] = None,
    phases: Optional[List[Dict]] = None,
    func_start_line: int = 0,
) -> str:
    """
    Build the per-function user prompt.

    Includes:
      - Function identity (name + params)
      - Project context (PKB context packet)
      - Full function source
      - Structured node list for labeling (with neighbor context and phase hints)
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

    if source_code and source_code.strip():
        parts.append(f"\n--- Function Source Code ---\n{source_code}")

    # Build node descriptions for labeling
    node_list = _build_node_list(
        nodes,
        all_nodes=all_nodes,
        phases=phases,
        func_start_line=func_start_line,
    )
    node_json = json.dumps(node_list, indent=2)
    parts.append(f"\n--- Nodes to Label ---\n{node_json}")

    parts.append(
        "\nReturn ONLY the JSON object mapping each node_id to its label."
    )

    return "\n".join(parts)


def _build_node_list(
    nodes: List[CfgNode],
    all_nodes: Optional[List[CfgNode]] = None,
    phases: Optional[List[Dict]] = None,
    func_start_line: int = 0,
) -> List[Dict]:
    """Convert CfgNodes into a structured list for the LLM prompt.

    Adds:
      - preceding_node_code / following_node_code  (neighbor context)
      - phase_hint                                  (phase annotation)
      - data_flow_shared                            (identifiers shared across batch)
    """
    # Build position map for neighbor lookup (all_nodes = full ordered list)
    pos_map: Dict[str, int] = {}
    if all_nodes:
        pos_map = {n.node_id: i for i, n in enumerate(all_nodes)}

    # ── Data-flow hint: find identifiers shared across ≥2 nodes in this batch ──
    # Identifier → set of node_ids that reference it
    ident_to_nodes: Dict[str, Set[str]] = {}
    for node in nodes:
        if node.node_type in (NodeType.START, NodeType.END):
            continue
        for ident in _extract_code_identifiers(node.raw_code):
            ident_to_nodes.setdefault(ident, set()).add(node.node_id)
    # Only keep identifiers that appear in 2+ nodes (the "shared" ones)
    shared_idents: Set[str] = {
        ident for ident, node_ids in ident_to_nodes.items()
        if len(node_ids) >= 2
    }

    result = []
    for node in nodes:
        # Skip structural nodes that don't need LLM labels
        if node.node_type in (NodeType.START, NodeType.END):
            continue

        entry: Dict = {
            "node_id": node.node_id,
            "type": node.node_type.value,
            "raw_code": node.raw_code[:200],  # cap per node
        }

        # ── Neighbor context (Change 2) ──────────────────────────────
        if all_nodes and node.node_id in pos_map:
            idx = pos_map[node.node_id]
            if idx > 0:
                prev_node = all_nodes[idx - 1]
                if prev_node.node_type not in (NodeType.START, NodeType.END):
                    entry["preceding_node_code"] = _truncate(prev_node.raw_code, 80)
            if idx < len(all_nodes) - 1:
                next_node = all_nodes[idx + 1]
                if next_node.node_type not in (NodeType.START, NodeType.END):
                    entry["following_node_code"] = _truncate(next_node.raw_code, 80)

        # ── Phase hint ───────────────────────────────────────────────
        if phases and func_start_line > 0:
            rel_line = node.start_line - func_start_line + 1
            for phase in phases:
                ps = phase.get("start_line", 0)
                pe = phase.get("end_line", 0)
                if isinstance(ps, int) and isinstance(pe, int) and ps <= rel_line <= pe:
                    desc = phase.get("description", "")
                    if desc:
                        entry["phase_hint"] = _truncate(desc, 80)
                    break

        # ── Data-flow hint ───────────────────────────────────────────
        # Variables/objects this node shares with other nodes in the batch.
        # Tells the LLM "these nodes all work on the same object/state" so it
        # labels them as steps in a coherent sequence rather than in isolation.
        node_shared = _extract_code_identifiers(node.raw_code) & shared_idents
        if node_shared:
            entry["data_flow_shared"] = sorted(node_shared)[:5]

        ctx = node.enriched_context

        # Function calls — cap to 3 entries, each truncated to 120 chars
        if ctx.get("function_calls"):
            entry["called_functions"] = [
                _truncate(
                    f"{c['signature']}: {c['description']}"
                    if c.get("description") else c["signature"],
                    120,
                )
                for c in ctx["function_calls"][:3]
            ]

        if ctx.get("inline_comment"):
            entry["source_comment"] = ctx["inline_comment"][:120]

        # Cap list-valued context fields to 2 entries, each 80 chars
        if ctx.get("enum_context"):
            entry["enum_context"] = [_truncate(s, 80) for s in ctx["enum_context"][:2]]

        if ctx.get("macro_context"):
            entry["macro_context"] = [_truncate(s, 80) for s in ctx["macro_context"][:2]]

        if ctx.get("typedef_context"):
            entry["typedef_context"] = [_truncate(s, 80) for s in ctx["typedef_context"][:2]]

        if ctx.get("struct_member_context"):
            entry["struct_member_context"] = [_truncate(s, 80) for s in ctx["struct_member_context"][:2]]

        result.append(entry)

    return result


def _extract_code_identifiers(code: str) -> Set[str]:
    """
    Extract variable/object identifiers from C++ raw_code for data-flow hints.

    Crucially: filters out function/method call names (identifiers followed by
    '(').  We want objects that carry state across nodes (e.g. 'doc', 'req'),
    not method names (e.g. 'SetObject', 'AddMember') which appear shared only
    because multiple nodes call methods on the same object.

    Also filters: C++ keywords, ALL-CAPS macros/constants, names shorter than 3
    characters.
    """
    # Match identifiers NOT immediately followed by '(' — excludes call names
    tokens = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()', code)
    return {
        t for t in tokens
        if t not in _CPP_KEYWORDS and len(t) > 2 and not t.isupper()
    }


def _truncate(s: str, max_chars: int) -> str:
    """Truncate a string to max_chars, appending '…' if truncated."""
    if len(s) <= max_chars:
        return s
    return s[:max_chars - 1] + "…"
