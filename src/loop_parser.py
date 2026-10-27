"""Phase 2 - Syntax analysis.

Recursive-descent parser (with precedence climbing for expressions) for the
loop-kernel subset of C.  It builds an abstract syntax tree (AST).

Grammar (informal)
    block  -> '{' stmt* '}'
    stmt   -> for | if | block | return | decl | assign ';'
    for    -> 'for' '(' [type] ID '=' expr ';' ID relop expr ';' update ')' stmt
    if     -> 'if' '(' expr ')' stmt ['else' stmt]
    assign -> lvalue ('='|'+='|'-='|'*='|'/=') expr
    expr   -> unary (binop unary)*          (precedence climbing)
    unary  -> ('-'|'!') unary | primary
    primary-> NUMBER | ID ('[' expr ']')* | ID '(' args ')' | '(' expr ')'
"""
from dataclasses import dataclass
from lexer import tokenize


# ----------------------------- AST nodes -----------------------------------
@dataclass
class Num:
    text: str

    @property
    def value(self):
        return float(self.text.rstrip("fFuUlL"))


@dataclass
class Var:
    name: str


@dataclass
class Index:
    base: str
    subs: list


@dataclass
class BinOp:
    op: str
    left: object
    right: object


@dataclass
class UnOp:
    op: str
    operand: object


@dataclass
class Call:
    name: str
    args: list


@dataclass
class Assign:
    lhs: object
    op: str
    rhs: object
    decl: bool = False


@dataclass
class For:
    var: str
    lo: object
    hi: object
    cmp: str
    step: int
    body: list


@dataclass
class If:
    cond: object
    then: list
    els: list


@dataclass
class Return:
    expr: object


PREC = {"||": 1, "&&": 2, "==": 3, "!=": 3, "<": 4, ">": 4, "<=": 4, ">=": 4,
        "+": 5, "-": 5, "*": 6, "/": 6, "%": 6}
ASSIGN_OPS = {"=", "+=", "-=", "*=", "/="}


class Parser:
    def __init__(self, tokens):
        self.t, self.p = tokens, 0

    # -- helpers
    def peek(self, k=0):
        i = self.p + k
        return self.t[i] if i < len(self.t) else None

    def next(self):
        tok = self.t[self.p]
        self.p += 1
        return tok

    def accept(self, text):
        tok = self.peek()
        if tok and tok.text == text:
            self.p += 1
            return True
        return False

    def expect(self, text):
        tok = self.peek()
        if not tok or tok.text != text:
            got = tok.text if tok else "EOF"
            line = tok.line if tok else "?"
            raise SyntaxError(f"syntax error: expected '{text}' but found '{got}' (line {line})")
        self.p += 1

    # -- statements
    def block(self):
        self.expect("{")
        out = []
        while not self.accept("}"):
            out += self.stmt()
        return out

    def stmt(self):
        tok = self.peek()
        if tok.text == "{":
            return self.block()
        if tok.text == "for":
            return [self.for_stmt()]
        if tok.text == "if":
            return [self.if_stmt()]
        if tok.text == "return":
            self.next()
            e = self.expr()
            self.expect(";")
            return [Return(e)]
        if tok.kind == "TYPE":                       # declaration
            self.next()
            out = []
            while True:
                name = self.next().text
                if self.accept("="):
                    out.append(Assign(Var(name), "=", self.expr(), decl=True))
                if not self.accept(","):
                    break
            self.expect(";")
            return out
        lhs = self.primary()                          # assignment statement
        op = self.next()
        if op.text in ("++", "--"):
            self.expect(";")
            return [Assign(lhs, "+=" if op.text == "++" else "-=", Num("1"))]
        if op.text not in ASSIGN_OPS:
            raise SyntaxError(f"syntax error: expected assignment operator, found '{op.text}' (line {op.line})")
        rhs = self.expr()
        self.expect(";")
        return [Assign(lhs, op.text, rhs)]

    def for_stmt(self):
        self.expect("for")
        self.expect("(")
        if self.peek().kind == "TYPE":
            self.next()
        var = self.next().text
        self.expect("=")
        lo = self.expr()
        self.expect(";")
        v2 = self.next().text
        cmp_ = self.next().text
        if v2 != var or cmp_ not in ("<", "<=", ">", ">="):
            raise SyntaxError("unsupported for-loop condition")
        hi = self.expr()
        self.expect(";")
        step = 1
        if self.peek().text == "++":
            self.next()
            self.next()
        elif self.peek().text == "--":
            self.next()
            self.next()
            step = -1
        else:
            self.next()                               # loop variable
            op = self.next().text
            if op == "+=":
                step = int(float(self.next().text))
            elif op == "-=":
                step = -int(float(self.next().text))
            elif op == "++":
                step = 1
            else:
                raise SyntaxError("unsupported loop increment")
        self.expect(")")
        return For(var, lo, hi, cmp_, step, self.stmt())

    def if_stmt(self):
        self.expect("if")
        self.expect("(")
        cond = self.expr()
        self.expect(")")
        then = self.stmt()
        els = self.stmt() if self.accept("else") else []
        return If(cond, then, els)

    # -- expressions
    def expr(self, minp=1):
        left = self.unary()
        while True:
            tok = self.peek()
            if not tok or tok.kind != "OP" or tok.text not in PREC or PREC[tok.text] < minp:
                return left
            self.next()
            left = BinOp(tok.text, left, self.expr(PREC[tok.text] + 1))

    def unary(self):
        tok = self.peek()
        if tok.kind == "OP" and tok.text in ("-", "!", "+"):
            self.next()
            return UnOp(tok.text, self.unary())
        return self.primary()

    def primary(self):
        tok = self.next()
        if tok.kind == "NUMBER":
            return Num(tok.text)
        if tok.text == "(":
            if self.peek().kind == "TYPE" and self.peek(1).text == ")":   # cast
                self.next()
                self.next()
                return self.unary()
            e = self.expr()
            self.expect(")")
            return e
        if tok.kind == "IDENT":
            if self.peek() and self.peek().text == "(":
                self.next()
                args = []
                if not self.accept(")"):
                    args.append(self.expr())
                    while self.accept(","):
                        args.append(self.expr())
                    self.expect(")")
                return Call(tok.text, args)
            subs = []
            while self.peek() and self.peek().text == "[":
                self.next()
                subs.append(self.expr())
                self.expect("]")
            return Index(tok.text, subs) if subs else Var(tok.text)
        raise SyntaxError(f"syntax error: unexpected '{tok.text}' (line {tok.line})")


def parse_expr_text(text):
    return Parser(tokenize(text)).expr()


def parse_function(src, name="kernel"):
    """Parse the body of function `name` and return its statement list."""
    toks = tokenize(src)
    starts = [i for i, t in enumerate(toks)
              if t.kind == "IDENT" and t.text == name and i + 1 < len(toks) and toks[i + 1].text == "("]
    if not starts:
        raise SyntaxError(f"function '{name}' not found")
    i = starts[0]
    depth = 0
    while True:                                   # skip the parameter list
        if toks[i].text == "(":
            depth += 1
        elif toks[i].text == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    p = Parser(toks)
    p.p = i + 1
    return p.block()
