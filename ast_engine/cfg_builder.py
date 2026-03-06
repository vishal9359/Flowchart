"""
Control Flow Graph (CFG) builder.

Traverses the libclang AST of a function body and produces a ControlFlowGraph
(nodes + directed edges) that precisely represents the C++ control flow.

Rules (ABSOLUTE — never change):
  - Structural truth comes only from the AST; no heuristics.
  - Loop back-edges are always explicit.
  - break → after-loop / after-switch.
  - continue → loop head.
  - return → END node.
  - All open (fall-through) exits connect to the next sequential node.
"""

import logging
from typing import Dict, List, Optional, Tuple

import clang.cindex as ci

from ast_engine.parser import SourceExtractor
from models import CfgEdge, CfgNode, ControlFlowGraph, FunctionEntry, NodeType

logger = logging.getLogger(__name__)

# Cursor kinds that break sequential grouping and need dedicated CFG nodes
_CONTROL_FLOW_KINDS = frozenset({
    ci.CursorKind.IF_STMT,
    ci.CursorKind.FOR_STMT,
    ci.CursorKind.WHILE_STMT,
    ci.CursorKind.DO_STMT,
    ci.CursorKind.CXX_FOR_RANGE_STMT,
    ci.CursorKind.SWITCH_STMT,
    ci.CursorKind.RETURN_STMT,
    ci.CursorKind.BREAK_STMT,
    ci.CursorKind.CONTINUE_STMT,
    ci.CursorKind.CXX_TRY_STMT,
})

# (node_id, edge_label_or_None) — edges waiting to connect to next node
OpenExits = List[Tuple[str, Optional[str]]]


class CFGBuilder:
    """
    Builds a ControlFlowGraph for a single C++ function.

    Usage:
        builder = CFGBuilder(source_lines, max_stmts, max_lines)
        cfg = builder.build(func_cursor, func_entry)
    """

    def __init__(self, source_lines: List[str],
                 max_stmts: int = 5, max_lines: int = 10) -> None:
        self._src = source_lines
        self._max_stmts = max_stmts
        self._max_lines = max_lines

        self._counter = 0
        self._nodes: Dict[str, CfgNode] = {}
        self._edges: List[CfgEdge] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self, func_cursor: ci.Cursor,
              func_entry: FunctionEntry) -> ControlFlowGraph:
        """Build and return the CFG for func_entry."""

        param_str = _params_str(func_entry.params)
        start = self._new_node(
            NodeType.START,
            f"{func_entry.qualified_name}({param_str})",
            func_entry.line, func_entry.line,
        )
        end = self._new_node(
            NodeType.END, "End",
            func_entry.end_line, func_entry.end_line,
        )

        body = _get_body(func_cursor)
        if body is None:
            self._edge(start.node_id, end.node_id)
        else:
            entry_id, opens, returns, _, _ = self._process_compound(body)

            if entry_id:
                self._edge(start.node_id, entry_id)
            else:
                self._edge(start.node_id, end.node_id)

            for (nid, lbl) in opens:
                self._edge(nid, end.node_id, lbl)
            for nid in returns:
                self._edge(nid, end.node_id)

        return ControlFlowGraph(
            function_key=func_entry.key,
            qualified_name=func_entry.qualified_name,
            source_file=func_entry.file,
            start_line=func_entry.line,
            end_line=func_entry.end_line,
            nodes=self._nodes,
            edges=self._edges,
            entry_node_id=start.node_id,
            exit_node_ids=[end.node_id],
        )

    # ------------------------------------------------------------------
    # Node / edge helpers
    # ------------------------------------------------------------------

    def _new_id(self) -> str:
        self._counter += 1
        return f"N{self._counter}"

    def _new_node(self, ntype: NodeType, raw: str,
                  sl: int, el: int) -> CfgNode:
        nid = self._new_id()
        node = CfgNode(node_id=nid, node_type=ntype,
                       raw_code=raw.strip(), start_line=sl, end_line=el)
        self._nodes[nid] = node
        return node

    def _edge(self, src: str, tgt: str,
              label: Optional[str] = None) -> None:
        self._edges.append(CfgEdge(source=src, target=tgt, label=label))

    def _connect_open(self, opens: OpenExits, to_id: str) -> None:
        for (nid, lbl) in opens:
            self._edge(nid, to_id, lbl)

    def _src_text(self, cursor: ci.Cursor) -> str:
        ext = cursor.extent
        return SourceExtractor.get_extent_text(
            self._src,
            ext.start.line, ext.end.line,
            ext.start.column, ext.end.column,
        )

    # ------------------------------------------------------------------
    # Statement dispatcher
    # ------------------------------------------------------------------

    def _process_stmt(self, cursor: ci.Cursor):
        """
        Process any statement cursor.
        Returns (entry_id, open_exits, return_nodes, break_nodes, continue_nodes).
        """
        k = cursor.kind
        if k == ci.CursorKind.COMPOUND_STMT:
            return self._process_compound(cursor)
        if k == ci.CursorKind.IF_STMT:
            return self._process_if(cursor)
        if k in (ci.CursorKind.FOR_STMT, ci.CursorKind.CXX_FOR_RANGE_STMT):
            return self._process_for(cursor)
        if k == ci.CursorKind.WHILE_STMT:
            return self._process_while(cursor)
        if k == ci.CursorKind.DO_STMT:
            return self._process_do_while(cursor)
        if k == ci.CursorKind.SWITCH_STMT:
            return self._process_switch(cursor)
        if k == ci.CursorKind.RETURN_STMT:
            return self._process_return(cursor)
        if k == ci.CursorKind.BREAK_STMT:
            return self._process_break(cursor)
        if k == ci.CursorKind.CONTINUE_STMT:
            return self._process_continue(cursor)
        if k == ci.CursorKind.CXX_TRY_STMT:
            return self._process_try(cursor)
        # Default: treat as an action node
        raw = self._src_text(cursor)
        node = self._new_node(NodeType.ACTION, raw,
                              cursor.extent.start.line,
                              cursor.extent.end.line)
        return (node.node_id, [(node.node_id, None)], [], [], [])

    # ------------------------------------------------------------------
    # Compound statement (sequential block)
    # ------------------------------------------------------------------

    def _process_compound(self, cursor: ci.Cursor):
        pending: List[ci.Cursor] = []
        current_open: OpenExits = []
        first_entry: Optional[str] = None
        all_returns: List[str] = []
        all_breaks: List[str] = []
        all_continues: List[str] = []

        def flush():
            nonlocal current_open, first_entry
            if not pending:
                return
            for seg in self._segment(pending):
                raw = self._seg_text(seg)
                node = self._new_node(NodeType.ACTION, raw,
                                      seg[0].extent.start.line,
                                      seg[-1].extent.end.line)
                self._connect_open(current_open, node.node_id)
                if first_entry is None:
                    first_entry = node.node_id
                current_open = [(node.node_id, None)]
            pending.clear()

        for child in cursor.get_children():
            if child.kind in _CONTROL_FLOW_KINDS:
                flush()
                entry, opens, rets, brks, conts = self._process_stmt(child)
                if entry is not None:
                    self._connect_open(current_open, entry)
                    if first_entry is None:
                        first_entry = entry
                current_open = opens
                all_returns.extend(rets)
                all_breaks.extend(brks)
                all_continues.extend(conts)
            else:
                pending.append(child)

        flush()
        return (first_entry, current_open, all_returns, all_breaks, all_continues)

    # ------------------------------------------------------------------
    # if / else-if / else
    # ------------------------------------------------------------------

    def _process_if(self, cursor: ci.Cursor):
        children = list(cursor.get_children())
        if len(children) < 2:
            return (None, [], [], [], [])

        cond = children[0]
        then_c = children[1]
        else_c = children[2] if len(children) > 2 else None

        dec = self._new_node(NodeType.DECISION, self._src_text(cond),
                             cond.extent.start.line, cond.extent.end.line)

        then_entry, then_opens, then_rets, then_brks, then_conts = \
            self._process_stmt(then_c)
        if then_entry:
            self._edge(dec.node_id, then_entry, "Yes")

        all_rets = list(then_rets)
        all_brks = list(then_brks)
        all_conts = list(then_conts)

        if else_c:
            else_entry, else_opens, else_rets, else_brks, else_conts = \
                self._process_stmt(else_c)
            if else_entry:
                self._edge(dec.node_id, else_entry, "No")
            open_exits = then_opens + else_opens
            all_rets.extend(else_rets)
            all_brks.extend(else_brks)
            all_conts.extend(else_conts)
        else:
            # No-else: decision "No" path goes forward
            open_exits = [(dec.node_id, "No")] + then_opens

        return (dec.node_id, open_exits, all_rets, all_brks, all_conts)

    # ------------------------------------------------------------------
    # for / range-based for
    # ------------------------------------------------------------------

    def _process_for(self, cursor: ci.Cursor):
        is_range = cursor.kind == ci.CursorKind.CXX_FOR_RANGE_STMT
        header = _for_header_text(cursor, self._src_text(cursor))

        loop = self._new_node(NodeType.LOOP_HEAD, header,
                              cursor.extent.start.line,
                              cursor.extent.start.line)

        children = list(cursor.get_children())
        body = children[-1] if children else None

        if body is None:
            return (loop.node_id, [(loop.node_id, "No")], [], [], [])

        b_entry, b_opens, b_rets, b_brks, b_conts = self._process_stmt(body)
        if b_entry:
            self._edge(loop.node_id, b_entry, "Yes")

        # Body fall-through and continues → back to loop head
        for (nid, lbl) in b_opens:
            self._edge(nid, loop.node_id, lbl)
        for nid in b_conts:
            self._edge(nid, loop.node_id)

        open_exits = [(loop.node_id, "No")] + [(b, None) for b in b_brks]
        return (loop.node_id, open_exits, b_rets, [], [])

    # ------------------------------------------------------------------
    # while
    # ------------------------------------------------------------------

    def _process_while(self, cursor: ci.Cursor):
        children = list(cursor.get_children())
        if len(children) < 2:
            return (None, [], [], [], [])

        cond, body = children[0], children[1]
        loop = self._new_node(NodeType.LOOP_HEAD, self._src_text(cond),
                              cond.extent.start.line, cond.extent.end.line)

        b_entry, b_opens, b_rets, b_brks, b_conts = self._process_stmt(body)
        if b_entry:
            self._edge(loop.node_id, b_entry, "Yes")

        for (nid, lbl) in b_opens:
            self._edge(nid, loop.node_id, lbl)
        for nid in b_conts:
            self._edge(nid, loop.node_id)

        open_exits = [(loop.node_id, "No")] + [(b, None) for b in b_brks]
        return (loop.node_id, open_exits, b_rets, [], [])

    # ------------------------------------------------------------------
    # do-while
    # ------------------------------------------------------------------

    def _process_do_while(self, cursor: ci.Cursor):
        children = list(cursor.get_children())
        if len(children) < 2:
            return (None, [], [], [], [])

        body_c, cond_c = children[0], children[1]

        b_entry, b_opens, b_rets, b_brks, b_conts = self._process_stmt(body_c)

        loop = self._new_node(NodeType.LOOP_HEAD,
                              f"do-while: {self._src_text(cond_c)}",
                              cond_c.extent.start.line, cond_c.extent.end.line)

        # Body fall-through and continues → condition check
        for (nid, lbl) in b_opens:
            self._edge(nid, loop.node_id, lbl)
        for nid in b_conts:
            self._edge(nid, loop.node_id)

        # Condition True → back to body start
        if b_entry:
            self._edge(loop.node_id, b_entry, "Yes")

        open_exits = [(loop.node_id, "No")] + [(b, None) for b in b_brks]
        return (b_entry, open_exits, b_rets, [], [])

    # ------------------------------------------------------------------
    # switch / case / default
    # ------------------------------------------------------------------

    def _process_switch(self, cursor: ci.Cursor):
        children = list(cursor.get_children())
        if not children:
            return (None, [], [], [], [])

        cond = children[0]
        sw = self._new_node(NodeType.SWITCH_HEAD, self._src_text(cond),
                            cond.extent.start.line, cond.extent.end.line)

        body = children[1] if len(children) > 1 else None
        all_rets: List[str] = []
        open_exits: OpenExits = []

        if body and body.kind == ci.CursorKind.COMPOUND_STMT:
            open_exits, all_rets = self._process_switch_body(sw.node_id, body)

        # If no cases connected, switch head is open exit
        if not open_exits:
            open_exits = [(sw.node_id, None)]

        return (sw.node_id, open_exits, all_rets, [], [])

    def _process_switch_body(self, sw_id: str,
                              body: ci.Cursor) -> Tuple[OpenExits, List[str]]:
        """Process the COMPOUND_STMT body of a switch statement."""
        open_exits: OpenExits = []
        all_rets: List[str] = []
        prev_open: OpenExits = []   # fallthrough from previous case

        for child in body.get_children():
            if child.kind == ci.CursorKind.CASE_STMT:
                case_children = list(child.get_children())
                val_cursor = case_children[0] if case_children else None
                val_text = self._src_text(val_cursor) if val_cursor else "?"

                case_node = self._new_node(NodeType.CASE, val_text,
                                           child.extent.start.line,
                                           child.extent.start.line)
                self._edge(sw_id, case_node.node_id, f"case {val_text}")
                # Fallthrough from prior case
                self._connect_open(prev_open, case_node.node_id)

                body_stmts = case_children[1:]
                if body_stmts:
                    e, opens, rets, brks, _ = self._process_case_stmts(body_stmts)
                    if e:
                        self._edge(case_node.node_id, e)
                    all_rets.extend(rets)
                    open_exits.extend([(b, None) for b in brks])
                    prev_open = opens
                else:
                    prev_open = [(case_node.node_id, None)]

            elif child.kind == ci.CursorKind.DEFAULT_STMT:
                def_children = list(child.get_children())
                def_node = self._new_node(NodeType.DEFAULT_CASE, "default",
                                          child.extent.start.line,
                                          child.extent.start.line)
                self._edge(sw_id, def_node.node_id, "default")
                self._connect_open(prev_open, def_node.node_id)

                if def_children:
                    e, opens, rets, brks, _ = self._process_case_stmts(def_children)
                    if e:
                        self._edge(def_node.node_id, e)
                    all_rets.extend(rets)
                    open_exits.extend([(b, None) for b in brks])
                    prev_open = opens
                else:
                    prev_open = [(def_node.node_id, None)]

        # Last case without break falls through to after-switch
        open_exits.extend(prev_open)
        return open_exits, all_rets

    def _process_case_stmts(self, stmts: List[ci.Cursor]):
        """Process statement list inside a case body (may contain sub-control-flow)."""
        pending: List[ci.Cursor] = []
        current_open: OpenExits = []
        first_entry: Optional[str] = None
        all_rets: List[str] = []
        all_brks: List[str] = []
        all_conts: List[str] = []

        def flush():
            nonlocal current_open, first_entry
            if not pending:
                return
            for seg in self._segment(pending):
                raw = self._seg_text(seg)
                node = self._new_node(NodeType.ACTION, raw,
                                      seg[0].extent.start.line,
                                      seg[-1].extent.end.line)
                self._connect_open(current_open, node.node_id)
                if first_entry is None:
                    first_entry = node.node_id
                current_open = [(node.node_id, None)]
            pending.clear()

        for s in stmts:
            # Nested CASE_STMT inside a case means fallthrough — stop here
            if s.kind in (ci.CursorKind.CASE_STMT, ci.CursorKind.DEFAULT_STMT):
                break
            if s.kind in _CONTROL_FLOW_KINDS:
                flush()
                entry, opens, rets, brks, conts = self._process_stmt(s)
                if entry is not None:
                    self._connect_open(current_open, entry)
                    if first_entry is None:
                        first_entry = entry
                current_open = opens
                all_rets.extend(rets)
                all_brks.extend(brks)
                all_conts.extend(conts)
            else:
                pending.append(s)

        flush()
        return (first_entry, current_open, all_rets, all_brks, all_conts)

    # ------------------------------------------------------------------
    # return / break / continue
    # ------------------------------------------------------------------

    def _process_return(self, cursor: ci.Cursor):
        node = self._new_node(NodeType.RETURN, self._src_text(cursor),
                              cursor.extent.start.line, cursor.extent.end.line)
        return (node.node_id, [], [node.node_id], [], [])

    def _process_break(self, cursor: ci.Cursor):
        node = self._new_node(NodeType.BREAK, "break",
                              cursor.extent.start.line, cursor.extent.end.line)
        return (node.node_id, [], [], [node.node_id], [])

    def _process_continue(self, cursor: ci.Cursor):
        node = self._new_node(NodeType.CONTINUE, "continue",
                              cursor.extent.start.line, cursor.extent.end.line)
        return (node.node_id, [], [], [], [node.node_id])

    # ------------------------------------------------------------------
    # try / catch
    # ------------------------------------------------------------------

    def _process_try(self, cursor: ci.Cursor):
        children = list(cursor.get_children())
        if not children:
            return (None, [], [], [], [])

        try_node = self._new_node(NodeType.TRY_HEAD, "try",
                                  cursor.extent.start.line,
                                  cursor.extent.start.line)

        # First child = try body
        b_entry, b_opens, b_rets, b_brks, b_conts = \
            self._process_stmt(children[0])
        if b_entry:
            self._edge(try_node.node_id, b_entry)

        all_opens: OpenExits = list(b_opens)
        all_rets = list(b_rets)

        for catch_c in children[1:]:
            if catch_c.kind != ci.CursorKind.CXX_CATCH_STMT:
                continue

            exc_type = _catch_exception_type(catch_c)
            catch_node = self._new_node(NodeType.CATCH,
                                        f"catch ({exc_type})",
                                        catch_c.extent.start.line,
                                        catch_c.extent.end.line)
            self._edge(try_node.node_id, catch_node.node_id, "exception")

            catch_body = _get_body(catch_c)
            if catch_body:
                ce, co, cr, _, _ = self._process_compound(catch_body)
                if ce:
                    self._edge(catch_node.node_id, ce)
                all_opens.extend(co)
                all_rets.extend(cr)
            else:
                all_opens.append((catch_node.node_id, None))

        return (try_node.node_id, all_opens, all_rets, b_brks, b_conts)

    # ------------------------------------------------------------------
    # Statement segmentation
    # ------------------------------------------------------------------

    def _segment(self, stmts: List[ci.Cursor]) -> List[List[ci.Cursor]]:
        """Split a list of sequential statements into size-bounded segments."""
        segments: List[List[ci.Cursor]] = []
        current: List[ci.Cursor] = []
        line_count = 0

        for s in stmts:
            s_lines = s.extent.end.line - s.extent.start.line + 1
            if current and (len(current) >= self._max_stmts
                            or line_count + s_lines > self._max_lines):
                segments.append(current)
                current = []
                line_count = 0
            current.append(s)
            line_count += s_lines

        if current:
            segments.append(current)

        return segments or [[]]

    def _seg_text(self, stmts: List[ci.Cursor]) -> str:
        """Combine source text of a segment."""
        return "\n".join(self._src_text(s) for s in stmts if s)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _get_body(cursor: ci.Cursor) -> Optional[ci.Cursor]:
    """Find the COMPOUND_STMT child of a function or catch cursor."""
    for child in cursor.get_children():
        if child.kind == ci.CursorKind.COMPOUND_STMT:
            return child
    return None


def _params_str(params: List[dict]) -> str:
    return ", ".join(
        f"{p.get('type', '')} {p.get('name', '')}".strip()
        for p in params
    )


def _for_header_text(cursor: ci.Cursor, full_text: str) -> str:
    """Extract the for() header (without the body braces)."""
    for child in cursor.get_children():
        if child.kind == ci.CursorKind.COMPOUND_STMT:
            body_start = child.extent.start
            cur_start = cursor.extent.start
            if body_start.line == cur_start.line:
                offset = body_start.column - cur_start.column
                return full_text[:offset].strip()
            else:
                line_offset = body_start.line - cur_start.line
                return "\n".join(full_text.splitlines()[:line_offset]).strip()
    return full_text.split("{")[0].strip()


def _catch_exception_type(catch_cursor: ci.Cursor) -> str:
    """Extract exception type from a CXX_CATCH_STMT cursor."""
    for child in catch_cursor.get_children():
        if child.kind in (ci.CursorKind.VAR_DECL, ci.CursorKind.PARM_DECL):
            t = child.type.spelling
            return t if t else child.spelling
    return "..."
