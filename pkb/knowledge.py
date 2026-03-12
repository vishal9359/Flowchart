"""
ProjectKnowledge — data structures and loader for project_scanner.py output.

Holds the rich semantic context extracted by scanning the entire C++ project:
  - Function signatures + doc comments + call graph (project-only calls)
  - Enum declarations with all values + per-value comments
  - #define macro definitions with values + comments
  - typedef / using type aliases with underlying types + comments
  - struct / class member field declarations
  - Hierarchical summaries: project → module → file → function

Used by ProjectKnowledgeBase.build_context_packet() and NodeEnricher
to inject project-vocabulary into LLM prompts.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FunctionKnowledge:
    qualified_name: str
    signature: str
    file: str
    line: int
    comment: str = ""
    # Qualified names of project functions this function calls (non-system).
    # Populated by project_scanner.py via CALL_EXPR traversal.
    calls: List[str] = field(default_factory=list)


@dataclass
class EnumValueKnowledge:
    value: str           # integer value as string
    comment: str = ""    # per-constant comment if any


@dataclass
class EnumKnowledge:
    qualified_name: str
    file: str
    comment: str = ""
    values: Dict[str, EnumValueKnowledge] = field(default_factory=dict)

    def summary(self, max_values: int = 8) -> str:
        """Return a compact summary string for LLM injection."""
        lines = []
        if self.comment:
            lines.append(self.comment)
        value_parts = []
        for name, ev in list(self.values.items())[:max_values]:
            simple = name.split("::")[-1]
            entry = f"{simple}={ev.value}"
            if ev.comment:
                entry += f" ({ev.comment})"
            value_parts.append(entry)
        if value_parts:
            lines.append("values: " + ", ".join(value_parts))
            if len(self.values) > max_values:
                lines.append(f"  ... and {len(self.values) - max_values} more")
        return "; ".join(lines)


@dataclass
class MacroKnowledge:
    name: str
    value: str
    file: str
    comment: str = ""

    def summary(self) -> str:
        # Classify intent from expansion body keywords
        v = self.value.lower()
        intent = ""
        if "level::err" in v or "level::critical" in v or "level::warn" in v:
            intent = "error/warning logging macro"
        elif "level::info" in v:
            intent = "info logging macro"
        elif "level::debug" in v or "level::trace" in v:
            intent = "debug logging macro"
        elif "assert" in v:
            intent = "assertion macro"
        elif "throw" in v or "exception" in v:
            intent = "exception-throwing macro"

        desc = f"{self.name}"
        if intent:
            desc += f" ({intent})"
        elif self.value and len(self.value) < 60:
            desc += f" = {self.value}"
        if self.comment:
            desc += f"  [{self.comment}]"
        return desc


@dataclass
class TypedefKnowledge:
    name: str
    underlying: str
    file: str
    comment: str = ""

    def summary(self) -> str:
        desc = f"{self.name} → {self.underlying}"
        if self.comment:
            desc += f"  ({self.comment})"
        return desc


@dataclass
class StructMemberKnowledge:
    """One field/member variable inside a struct or class."""
    field_type: str
    comment: str = ""

    def summary(self, field_name: str) -> str:
        desc = f"{field_name}: {self.field_type}"
        if self.comment:
            desc += f"  ({self.comment})"
        return desc


@dataclass
class StructKnowledge:
    """A C++ struct or class with its member variable declarations."""
    qualified_name: str
    file: str
    comment: str = ""
    members: Dict[str, StructMemberKnowledge] = field(default_factory=dict)

    def member_summary(self, member_name: str) -> str:
        """Return a compact description of one member for LLM injection."""
        m = self.members.get(member_name)
        if not m:
            return ""
        base = f"{self.qualified_name}::{member_name} ({m.field_type})"
        return base + (f"  [{m.comment}]" if m.comment else "")


@dataclass
class ProjectKnowledge:
    project_name: str = ""
    base_path: str = ""

    # Hierarchical summaries (populated by --llm-summarize in project_scanner.py)
    project_summary: str = ""
    # Keyed by module/directory relative path (e.g. "src/qos")
    module_summaries: Dict[str, str] = field(default_factory=dict)
    # Keyed by relative file path (e.g. "src/qos/qos_manager.cpp")
    file_summaries: Dict[str, str] = field(default_factory=dict)

    # Keyed by qualified_name
    functions: Dict[str, FunctionKnowledge] = field(default_factory=dict)
    # Keyed by simple or qualified enum name
    enums: Dict[str, EnumKnowledge] = field(default_factory=dict)
    # Keyed by macro name
    macros: Dict[str, MacroKnowledge] = field(default_factory=dict)
    # Keyed by typedef/alias name
    typedefs: Dict[str, TypedefKnowledge] = field(default_factory=dict)
    # Keyed by simple or qualified struct/class name
    structs: Dict[str, StructKnowledge] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return (not self.functions and not self.enums
                and not self.macros and not self.typedefs
                and not self.structs)

    def stats(self) -> str:
        return (
            f"functions={len(self.functions)}, enums={len(self.enums)}, "
            f"macros={len(self.macros)}, typedefs={len(self.typedefs)}, "
            f"structs={len(self.structs)}, "
            f"file_summaries={len(self.file_summaries)}, "
            f"module_summaries={len(self.module_summaries)}, "
            f"project_summary={'yes' if self.project_summary else 'no'}"
        )


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

def save_knowledge(knowledge: ProjectKnowledge, path: str) -> None:
    """Serialize ProjectKnowledge to JSON file."""
    data = {
        "project_name": knowledge.project_name,
        "base_path": knowledge.base_path,
        "project_summary": knowledge.project_summary,
        "module_summaries": knowledge.module_summaries,
        "file_summaries": knowledge.file_summaries,
        "functions": {
            k: {
                "qualifiedName": v.qualified_name,
                "signature": v.signature,
                "file": v.file,
                "line": v.line,
                "comment": v.comment,
                "calls": v.calls,
            }
            for k, v in knowledge.functions.items()
        },
        "enums": {
            k: {
                "qualifiedName": v.qualified_name,
                "file": v.file,
                "comment": v.comment,
                "values": {
                    vk: {"value": vv.value, "comment": vv.comment}
                    for vk, vv in v.values.items()
                },
            }
            for k, v in knowledge.enums.items()
        },
        "macros": {
            k: {
                "name": v.name,
                "value": v.value,
                "file": v.file,
                "comment": v.comment,
            }
            for k, v in knowledge.macros.items()
        },
        "typedefs": {
            k: {
                "name": v.name,
                "underlying": v.underlying,
                "file": v.file,
                "comment": v.comment,
            }
            for k, v in knowledge.typedefs.items()
        },
        "structs": {
            k: {
                "qualifiedName": v.qualified_name,
                "file": v.file,
                "comment": v.comment,
                "members": {
                    mk: {"type": mv.field_type, "comment": mv.comment}
                    for mk, mv in v.members.items()
                },
            }
            for k, v in knowledge.structs.items()
        },
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Project knowledge saved: %s  (%s)", path, knowledge.stats())


def load_knowledge(path: str) -> Optional[ProjectKnowledge]:
    """Load ProjectKnowledge from a JSON file. Returns None if file missing."""
    p = Path(path)
    if not p.exists():
        logger.warning("Knowledge file not found: %s", path)
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.error("Failed to load knowledge file %s: %s", path, exc)
        return None

    k = ProjectKnowledge(
        project_name=data.get("project_name", ""),
        base_path=data.get("base_path", ""),
        project_summary=data.get("project_summary", ""),
        module_summaries=data.get("module_summaries", {}),
        file_summaries=data.get("file_summaries", {}),
    )

    for key, v in data.get("functions", {}).items():
        k.functions[key] = FunctionKnowledge(
            qualified_name=v.get("qualifiedName", key),
            signature=v.get("signature", ""),
            file=v.get("file", ""),
            line=v.get("line", 0),
            comment=v.get("comment", ""),
            calls=v.get("calls", []),
        )

    for key, v in data.get("enums", {}).items():
        enum_values = {
            vk: EnumValueKnowledge(
                value=vv.get("value", ""),
                comment=vv.get("comment", ""),
            )
            for vk, vv in v.get("values", {}).items()
        }
        k.enums[key] = EnumKnowledge(
            qualified_name=v.get("qualifiedName", key),
            file=v.get("file", ""),
            comment=v.get("comment", ""),
            values=enum_values,
        )

    for key, v in data.get("macros", {}).items():
        k.macros[key] = MacroKnowledge(
            name=v.get("name", key),
            value=v.get("value", ""),
            file=v.get("file", ""),
            comment=v.get("comment", ""),
        )

    for key, v in data.get("typedefs", {}).items():
        k.typedefs[key] = TypedefKnowledge(
            name=v.get("name", key),
            underlying=v.get("underlying", ""),
            file=v.get("file", ""),
            comment=v.get("comment", ""),
        )

    for key, v in data.get("structs", {}).items():
        members = {
            mk: StructMemberKnowledge(
                field_type=mv.get("type", ""),
                comment=mv.get("comment", ""),
            )
            for mk, mv in v.get("members", {}).items()
        }
        k.structs[key] = StructKnowledge(
            qualified_name=v.get("qualifiedName", key),
            file=v.get("file", ""),
            comment=v.get("comment", ""),
            members=members,
        )

    logger.info("Project knowledge loaded: %s  (%s)", path, k.stats())
    return k
