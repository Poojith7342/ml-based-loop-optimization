"""Phase 4 - Intermediate representation: three-address code (TAC).

Lowers the loop-nest AST into TAC.  Array subscripts that are simple affine
expressions are kept inline (e.g. x[i-1]); every other sub-expression gets a
temporary, so each instruction has at most one operator.
"""
from loop_parser import Num, Var, Index, BinOp, UnOp, Call, Assign, For, If, Return


def show(e):
    """Pretty-print an expression AST."""
    if isinstance(e, Num):
        return e.text
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Index):
        return e.base + "".join(f"[{show(s)}]" for s in e.subs)
    if isinstance(e, BinOp):
        return f"{show(e.left)} {e.op} {show(e.right)}"
    if isinstance(e, UnOp):
        return f"{e.op}{show(e.operand)}"
    if isinstance(e, Call):
        return f"{e.name}({', '.join(show(a) for a in e.args)})"
    return "?"


def _simple_affine(e):
    if isinstance(e, (Num, Var)):
        return True
    if isinstance(e, BinOp) and e.op in "+-*":
        return _simple_affine(e.left) and _simple_affine(e.right)
    return False


class TACGen:
    def __init__(self):
        self.code, self.ntemp, self.nlabel = [], 0, 0

    def temp(self):
        self.ntemp += 1
        return f"t{self.ntemp}"

    def label(self):
        self.nlabel += 1
        return f"L{self.nlabel}"

    def emit(self, s):
        self.code.append(s)

    # expression -> operand string
    def expr(self, e):
        if isinstance(e, Num):
            return e.text
        if isinstance(e, Var):
            return e.name
        if isinstance(e, Index):
            subs = [show(s) if _simple_affine(s) else self.expr(s) for s in e.subs]
            t = self.temp()
            self.emit(f"{t} = {e.base}" + "".join(f"[{s}]" for s in subs))
            return t
        if isinstance(e, UnOp):
            a = self.expr(e.operand)
            t = self.temp()
            self.emit(f"{t} = {e.op}{a}")
            return t
        if isinstance(e, Call):
            args = [self.expr(a) for a in e.args]
            t = self.temp()
            self.emit(f"{t} = call {e.name}, {', '.join(args)}")
            return t
        a, b = self.expr(e.left), self.expr(e.right)
        t = self.temp()
        self.emit(f"{t} = {a} {e.op} {b}")
        return t

    def lvalue(self, e):
        if isinstance(e, Index):
            subs = [show(s) if _simple_affine(s) else self.expr(s) for s in e.subs]
            return e.base + "".join(f"[{s}]" for s in subs)
        return e.name

    def stmts(self, body):
        for s in body:
            self.stmt(s)

    def stmt(self, s):
        if isinstance(s, Assign):
            if s.op == "=":
                v = self.expr(s.rhs)
            else:
                cur = self.expr(s.lhs) if isinstance(s.lhs, Index) else s.lhs.name
                r = self.expr(s.rhs)
                v = self.temp()
                self.emit(f"{v} = {cur} {s.op[0]} {r}")
            self.emit(f"{self.lvalue(s.lhs)} = {v}")
        elif isinstance(s, If):
            c = self.expr(s.cond)
            l_else, l_end = self.label(), self.label()
            self.emit(f"ifFalse {c} goto {l_else}")
            self.stmts(s.then)
            if s.els:
                self.emit(f"goto {l_end}")
            self.emit(f"{l_else}:")
            if s.els:
                self.stmts(s.els)
                self.emit(f"{l_end}:")
        elif isinstance(s, For):
            top, out = self.label(), self.label()
            self.emit(f"{s.var} = {show(s.lo)}")
            self.emit(f"{top}:")
            rel = {"<": ">=", "<=": ">", ">": "<=", ">=": "<"}[s.cmp]
            c = self.temp()
            self.emit(f"{c} = {s.var} {rel} {show(s.hi)}")
            self.emit(f"if {c} goto {out}")
            self.stmts(s.body)
            self.emit(f"{s.var} = {s.var} + {s.step}")
            self.emit(f"goto {top}")
            self.emit(f"{out}:")
        elif isinstance(s, Return):
            self.emit(f"return {self.expr(s.expr)}")


def lower(stmts):
    g = TACGen()
    g.stmts(stmts)
    return g.code


def body_instruction_count(stmts):
    """Number of TAC instructions for a loop *body* (excludes loop control)."""
    return len([c for c in lower(stmts) if not c.endswith(":") and not c.startswith("goto")
                and not c.startswith("ifFalse")])
