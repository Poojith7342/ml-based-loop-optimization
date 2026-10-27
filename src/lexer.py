"""Phase 1 - Lexical analysis.

A small regular-expression based scanner for the C subset used by the loop
kernels.  It plays the same role as a Lex/Flex specification: the source text
is converted into a stream of (kind, lexeme, line) tokens.
"""
import re
from dataclasses import dataclass

TYPES = {"int", "long", "float", "double", "void", "unsigned", "size_t", "DT"}
KEYWORDS = {"for", "if", "else", "return", "while", "static", "typedef"}

_SPEC = [
    ("COMMENT", r"//[^\n]*|/\*.*?\*/"),
    ("PREPROC", r"\#[^\n]*"),
    ("STRING",  r'"(?:\\.|[^"\\])*"' + r"|'(?:\\.|[^'\\])*'"),
    ("NUMBER",  r"\d+\.\d*(?:[eE][+-]?\d+)?[fF]?|\.\d+(?:[eE][+-]?\d+)?[fF]?"
                r"|\d+[eE][+-]?\d+[fF]?|\d+[uUlL]*"),
    ("IDENT",   r"[A-Za-z_]\w*"),
    ("OP",      r"\+\+|--|\+=|-=|\*=|/=|<=|>=|==|!=|&&|\|\||[+\-*/%<>=!&|^~?:]"),
    ("PUNCT",   r"[()\[\]{};,.]"),
    ("WS",      r"\s+"),
    ("BAD",     r"."),
]
_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _SPEC), re.S)


@dataclass
class Token:
    kind: str
    text: str
    line: int

    def __repr__(self):
        return f"{self.kind:<8}{self.text}"


def tokenize(src, keep=("NUMBER", "IDENT", "OP", "PUNCT")):
    """Return the token list.  Whitespace/comments/preprocessor lines are
    discarded; identifiers are re-classified as KEYWORD or TYPE."""
    out, line = [], 1
    for m in _RE.finditer(src):
        kind, text = m.lastgroup, m.group()
        if kind == "BAD":
            raise SyntaxError(f"lexical error: illegal character {text!r} at line {line}")
        if kind in keep:
            if kind == "IDENT":
                kind = "KEYWORD" if text in KEYWORDS else "TYPE" if text in TYPES else "IDENT"
            out.append(Token(kind, text, line))
        line += text.count("\n")
    return out


if __name__ == "__main__":
    import sys
    for t in tokenize(open(sys.argv[1]).read()):
        print(t)
