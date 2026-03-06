"""
Condition label normalizer.

Converts raw C++ condition expressions into readable English phrases
for use in DECISION / LOOP_HEAD / SWITCH_HEAD labels when they are not
already human-readable (i.e., when the LLM fallback is invoked, or as
a pre-processing step before LLM prompting).

Rules (DO NOT CHANGE):
  - Logical operators: && → AND, || → OR, ! → NOT / negation
  - Comparisons: == → equals, != → does not equal, < → less than, etc.
  - Predicate-style methods (is*, has*, can*, should*) detected and preserved
  - Negation of predicates: !isValid() → "is not valid"
  - No reordering or merging of conditions
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Operator replacement tables  (order matters — longer patterns first)
# ---------------------------------------------------------------------------

_LOGICAL_OPS = [
    (r"\s*&&\s*", " AND "),
    (r"\s*\|\|\s*", " OR "),
]

_COMPARISON_OPS = [
    (r"\s*==\s*", " equals "),
    (r"\s*!=\s*", " does not equal "),
    (r"\s*>=\s*", " is greater than or equal to "),
    (r"\s*<=\s*", " is less than or equal to "),
    (r"\s*>\s*", " is greater than "),
    (r"\s*<\s*", " is less than "),
]

# Predicate method prefixes that indicate a boolean check
_PREDICATE_PREFIXES = ("is", "has", "can", "should", "will", "was",
                       "are", "have", "did", "does", "check", "needs")


def normalize_condition(raw: str) -> str:
    """
    Convert a raw C++ condition expression to a readable English phrase.

    Examples:
        "retval == STATUS_OK"           → "retval equals STATUS_OK"
        "!manager->isConnected()"       → "manager is not connected"
        "count > 0 && buf != nullptr"   → "count is greater than 0 AND buf does not equal nullptr"
        "isLimitExceeded(id, event)"    → "Is limit exceeded for id event"
    """
    s = raw.strip()
    if not s:
        return s

    # Handle simple negated predicate: !foo->isBar()
    negated = _try_negated_predicate(s)
    if negated:
        return negated

    # Handle direct predicate call: isBar() / obj->isBar()
    predicate = _try_predicate(s)
    if predicate:
        return predicate

    # Apply comparison operators first (before logical, so && isn't broken)
    for pattern, replacement in _COMPARISON_OPS:
        s = re.sub(pattern, replacement, s)

    # Apply logical operators
    for pattern, replacement in _LOGICAL_OPS:
        s = re.sub(pattern, replacement, s)

    # Clean up remaining C++ symbols
    s = _clean_cpp_symbols(s)

    return s.strip()


def normalize_edge_label(label: Optional[str]) -> Optional[str]:
    """
    Normalize a flowchart edge label (Yes/No, case values, etc.).
    Keeps short edge labels clean and human-readable.
    """
    if label is None:
        return None
    label = label.strip()
    if not label:
        return None

    # Standardise Yes/No / True/False
    if label.lower() in ("yes", "true", "1"):
        return "Yes"
    if label.lower() in ("no", "false", "0"):
        return "No"

    # case labels: normalise "case 0" / "case BackendEvent::GC"
    if label.lower().startswith("case "):
        return label  # keep as-is; LLM will label the CASE node

    return label


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_negated_predicate(expr: str) -> Optional[str]:
    """Handle !obj->isPredicate() → 'obj is not predicate'."""
    m = re.match(
        r"^!\s*([\w\->:.*]+?)\s*->\s*(\w+)\s*\(", expr
    )
    if m:
        obj, method = m.group(1), m.group(2)
        english = _method_to_english(method, negated=True)
        if english:
            return f"{obj} {english}"

    # !isPredicate() with no object
    m = re.match(r"^!\s*(\w+)\s*\(", expr)
    if m:
        method = m.group(1)
        english = _method_to_english(method, negated=True)
        if english:
            return english

    return None


def _try_predicate(expr: str) -> Optional[str]:
    """Handle obj->isPredicate(...) → 'obj is predicate'."""
    m = re.match(r"^([\w\->:.*]+?)\s*->\s*(\w+)\s*\(", expr)
    if m:
        obj, method = m.group(1), m.group(2)
        english = _method_to_english(method, negated=False)
        if english:
            return f"{obj} {english}"

    # isPredicate() with no object
    m = re.match(r"^(\w+)\s*\(", expr)
    if m:
        method = m.group(1)
        english = _method_to_english(method, negated=False)
        if english:
            return english

    return None


def _method_to_english(method: str, negated: bool) -> Optional[str]:
    """Convert a predicate method name to an English phrase."""
    lower = method.lower()
    for prefix in _PREDICATE_PREFIXES:
        if lower.startswith(prefix) and len(lower) > len(prefix):
            body = method[len(prefix):]
            # CamelCase split: isLimitExceeded → limit exceeded
            words = re.sub(r"([A-Z])", r" \1", body).strip().lower()
            if negated:
                return f"is not {words}" if prefix in ("is", "are") else f"does not {prefix} {words}"
            return f"is {words}" if prefix in ("is", "are") else f"{prefix}s {words}"
    return None


def _clean_cpp_symbols(s: str) -> str:
    """Remove or replace remaining C++ syntax that looks ugly in labels."""
    # nullptr / NULL → null
    s = re.sub(r"\bnullptr\b", "null", s)
    s = re.sub(r"\bNULL\b", "null", s)
    # Dereference operators
    s = s.replace("->", ".")
    # Address-of
    s = s.replace("&", " ")
    # Scope resolution — keep last component
    s = re.sub(r"\w+::", "", s)
    # Remove trailing semicolons
    s = s.rstrip(";").strip()
    return s
