"""Phase 3 - Semantic analysis and loop analysis (feature extraction).

Takes the C source of a loop kernel and produces a numeric feature vector:

  lexical analysis  -> tokens                      (lexer.py)
  syntax analysis   -> AST                         (loop_parser.py)
  semantic analysis -> declarations, trip counts, memory strides,
                       loop-carried dependences, reductions
  IR generation     -> three-address code length   (tac.py)

The feature vector is the input of the ML optimisation decision model.
"""
import math
import re
from lexer import tokenize
from loop_parser import (Num, Var, Index, BinOp, UnOp, Call, Assign, For, If,
                         Return, parse_expr_text, parse_function)
from tac import lower, body_instruction_count, show

DT_BYTES = {"double": 8, "float": 4, "int": 4, "long": 8}


class SemanticError(Exception):
    pass


# ------------------------------------------------------------------ evaluation
def eval_expr(e, env):
    """Evaluate a constant/affine expression; None if a name is unknown."""
    if isinstance(e, Num):
        return e.value
    if isinstance(e, Var):
        return env.get(e.name)
    if isinstance(e, UnOp):
        v = eval_expr(e.operand, env)
        return None if v is None else (-v if e.op == "-" else v)
    if isinstance(e, BinOp):
        a, b = eval_expr(e.left, env), eval_expr(e.right, env)
        if a is None or b is None:
            return None
        try:
            return {"+": a + b, "-": a - b, "*": a * b,
                    "/": a / b if b else None,
                    "%": a % b if b else None}.get(e.op)
        except Exception:
            return None
    return None


def scan_declarations(src):
    """#define constants, typedef of DT and the static array declarations."""
    raw = {m.group(1): m.group(2)
           for m in re.finditer(r"^\s*#\s*define\s+(\w+)\s+(.+?)\s*$", src, re.M)}
    env = {}
    for _ in range(6):
        for k, v in raw.items():
            if k not in env:
                val = eval_expr(parse_expr_text(v), env)
                if val is not None:
                    env[k] = val
    m = re.search(r"typedef\s+(double|float)\s+DT\s*;", src)
    dtype = m.group(1) if m else "double"
    arrays = {}
    for m in re.finditer(r"^\s*static\s+(DT|float|double)\s+([^;()]+);", src, re.M):
        et = dtype if m.group(1) == "DT" else m.group(1)
        for decl in m.group(2).split(","):
            mm = re.match(r"\s*(\w+)((?:\s*\[[^\]]+\])*)", decl)
            if not mm:
                continue
            dims = [eval_expr(parse_expr_text(d), env)
                    for d in re.findall(r"\[([^\]]+)\]", mm.group(2))]
            if None in dims or not dims:
                continue
            arrays[mm.group(1)] = {"dims": [int(d) for d in dims], "elem": DT_BYTES[et]}
    return env, dtype, arrays


# ------------------------------------------------------------------ AST walkers
def walk_exprs(e, into_subs=True):
    """Pre-order walk of an expression tree.  With into_subs=False the
    subscript (address) arithmetic is skipped, so only data-path operations
    are visited."""
    yield e
    if isinstance(e, BinOp):
        yield from walk_exprs(e.left, into_subs)
        yield from walk_exprs(e.right, into_subs)
    elif isinstance(e, UnOp):
        yield from walk_exprs(e.operand, into_subs)
    elif isinstance(e, Call):
        for a in e.args:
            yield from walk_exprs(a, into_subs)
    elif isinstance(e, Index) and into_subs:
        for s in e.subs:
            yield from walk_exprs(s)


def all_stmts(stmts):
    for s in stmts:
        yield s
        if isinstance(s, For):
            yield from all_stmts(s.body)
        elif isinstance(s, If):
            yield from all_stmts(s.then)
            yield from all_stmts(s.els)


def count_ops(stmts):
    """Arithmetic operations executed per iteration of a loop body.
    For an if/else, the more expensive branch is counted (static estimate)."""
    n = {"+": 0, "-": 0, "*": 0, "/": 0, "call": 0}
    for s in stmts:
        if isinstance(s, Assign):
            if s.op != "=":
                n[s.op[0]] += 1
            for x in walk_exprs(s.rhs, into_subs=False):
                if isinstance(x, BinOp) and x.op in "+-*/":
                    n[x.op] += 1
                elif isinstance(x, Call):
                    n["call"] += 1
        elif isinstance(s, If):
            a, b = count_ops(s.then), count_ops(s.els)
            pick = a if sum(a.values()) >= sum(b.values()) else b
            for k in n:
                n[k] += pick[k]
    return n


def n_branches(stmts):
    return sum(1 for s in all_stmts(stmts) if isinstance(s, If))


def collect_refs(stmts):
    """All array references in a body as (Index, is_write)."""
    refs = []

    def reads(e):
        for x in walk_exprs(e):
            if isinstance(x, Index):
                refs.append((x, False))

    for s in stmts:
        if isinstance(s, Assign):
            if isinstance(s.lhs, Index):
                refs.append((s.lhs, True))
                if s.op != "=":
                    refs.append((s.lhs, False))
                for sub in s.lhs.subs:
                    reads(sub)
            reads(s.rhs)
        elif isinstance(s, If):
            reads(s.cond)
            refs += collect_refs(s.then) + collect_refs(s.els)
    return refs


# ------------------------------------------------------------------ loop nest
def trip_count(loop, env):
    lo, hi = eval_expr(loop.lo, env), eval_expr(loop.hi, env)
    if lo is None or hi is None:
        return None
    s = abs(loop.step)
    d = (hi - lo) if loop.step > 0 else (lo - hi)
    n = math.ceil(d / s) if loop.cmp in ("<", ">") else math.floor(d / s) + 1
    return max(0, n)


def loop_chain(loop):
    chain = [loop]
    while True:
        inner = [s for s in chain[-1].body if isinstance(s, For)]
        if not inner:
            return chain
        chain.append(inner[0])


def linear_form(e, var, env, loopvars):
    """(coefficient, offset) of expression `e` in loop variable `var`
    (other loop variables are fixed to 0); None if e is not affine in var."""
    scope = dict(env)
    for lv in loopvars:
        scope[lv] = 0
    vals = []
    for t in (0, 1, 2):
        scope[var] = t
        v = eval_expr(e, scope)
        if v is None:
            return None
        vals.append(v)
    if abs((vals[2] - vals[1]) - (vals[1] - vals[0])) > 1e-9:
        return None
    return (vals[1] - vals[0], vals[0])


def mem_stride(ref, var, env, loopvars, arrays):
    """Memory stride (in elements) of `ref` when `var` is incremented."""
    dims = arrays[ref.base]["dims"]
    total = 0
    for k, sub in enumerate(ref.subs):
        lf = linear_form(sub, var, env, loopvars)
        if lf is None:
            return None
        total += lf[0] * math.prod(dims[k + 1:])
    return total


def carried_distance(w, r, var, env, loopvars):
    """Distance of a loop-carried flow dependence write w -> read r."""
    if len(w.subs) != len(r.subs):
        return None
    lw = [linear_form(s, var, env, loopvars) for s in w.subs]
    lr = [linear_form(s, var, env, loopvars) for s in r.subs]
    if None in lw or None in lr:
        return None
    if lw == lr:
        return 0
    if lw[:-1] != lr[:-1]:
        return None
    (cw, ow), (cr, orr) = lw[-1], lr[-1]
    if cw != cr or cw == 0:
        return None
    return (ow - orr) / cw


def check_semantics(stmts, arrays, env, params):
    declared = set(env) | set(params)
    for s in all_stmts(stmts):
        if isinstance(s, For):
            declared.add(s.var)
        elif isinstance(s, Assign) and s.decl:
            declared.add(s.lhs.name)
    problems = []

    def check(e):
        for x in walk_exprs(e):
            if isinstance(x, Index):
                if x.base not in arrays:
                    problems.append(f"array '{x.base}' is not declared")
                elif len(x.subs) != len(arrays[x.base]["dims"]):
                    problems.append(f"array '{x.base}' used with {len(x.subs)} subscript(s), "
                                    f"declared with {len(arrays[x.base]['dims'])}")
            elif isinstance(x, Var) and x.name not in declared:
                problems.append(f"identifier '{x.name}' is not declared")

    for s in all_stmts(stmts):
        if isinstance(s, Assign):
            check(s.lhs)
            check(s.rhs)
        elif isinstance(s, If):
            check(s.cond)
    if problems:
        raise SemanticError("; ".join(sorted(set(problems))))


# ------------------------------------------------------------------ analysis
FEATURES = [
    "dtype_bytes", "nest_depth", "inner_trip_log2", "total_iters_log2",
    "n_add", "n_mul", "n_div", "n_flops", "n_loads", "n_stores", "n_arrays",
    "n_branches", "n_unit_stride", "n_nonunit_stride", "n_invariant",
    "max_stride_log2", "has_reduction", "has_carried_dep", "dep_distance",
    "footprint_log2", "bytes_per_iter", "arith_intensity", "tac_len",
]


def analyze_source(src, func="kernel"):
    env, dtype, arrays = scan_declarations(src)
    region = src
    if "KERNEL_BEGIN" in src and "KERNEL_END" in src:      # analyse only the kernel
        region = src.split("KERNEL_BEGIN", 1)[1].split("KERNEL_END", 1)[0]
    stmts = parse_function(region, func)
    m = re.search(rf"\b{func}\s*\(([^)]*)\)", region)
    params = [p.strip().split()[-1] for p in m.group(1).split(",") if p.strip() and p.strip() != "void"] if m else []
    check_semantics(stmts, arrays, env, params)

    top = [s for s in stmts if isinstance(s, For)]
    if not top:
        raise SemanticError("no loop found in kernel")

    def work(l):
        c = loop_chain(l)
        return math.prod((trip_count(x, env) or 1000) for x in c)

    nest = max(top, key=work)
    chain = loop_chain(nest)
    loopvars = [l.var for l in chain]
    inner = chain[-1]
    ivar = inner.var
    trips = [(trip_count(l, env) or 1000) for l in chain]
    body = inner.body

    ops = count_ops(body)
    refs = collect_refs(body)

    # distinct references (the compiler performs CSE on identical loads)
    uniq_reads, uniq_writes = {}, {}
    for r, w in refs:
        (uniq_writes if w else uniq_reads)[show(r)] = r
    # a compound assignment target is both read and written; count it once as load
    strides, unit, nonunit, invariant, max_stride = {}, 0, 0, 0, 1
    for key, r in {**uniq_reads, **uniq_writes}.items():
        st = mem_stride(r, ivar, env, loopvars, arrays)
        strides[key] = st
        if st is None:
            nonunit += 1
        elif st == 0:
            invariant += 1
        elif abs(st) == 1:
            unit += 1
        else:
            nonunit += 1
            max_stride = max(max_stride, abs(st))

    # loop-carried dependences (flow, distance > 0)
    dep, dist = 0, 0
    for wkey, w in uniq_writes.items():
        for rkey, r in uniq_reads.items():
            if r.base != w.base:
                continue
            d = carried_distance(w, r, ivar, env, loopvars)
            if d is not None and d > 0:
                dep, dist = 1, (d if dist == 0 else min(dist, d))

    # reductions: scalar accumulator or array element invariant in inner loop
    red = 0
    for s in all_stmts(body):
        if isinstance(s, Assign) and not s.decl:
            if isinstance(s.lhs, Var):
                if s.op in ("+=", "-=", "*="):
                    red = 1
                elif s.op == "=" and any(isinstance(x, Var) and x.name == s.lhs.name
                                         for x in walk_exprs(s.rhs)):
                    red = 1
            elif isinstance(s.lhs, Index) and strides.get(show(s.lhs)) == 0:
                red = 1

    touched = {r.base for r, _ in refs}
    footprint = sum(math.prod(arrays[a]["dims"]) * arrays[a]["elem"] for a in touched)
    dtb = DT_BYTES[dtype]
    variant_refs = [k for k, st in strides.items() if st != 0]
    bytes_iter = max(len(variant_refs) * dtb, 1)
    flops = ops["+"] + ops["-"] + ops["*"] + ops["/"]
    tac = lower(body)

    feat = {
        "dtype_bytes": dtb,
        "nest_depth": len(chain),
        "inner_trip_log2": math.log2(max(trips[-1], 1)),
        "total_iters_log2": math.log2(max(math.prod(trips), 1)),
        "n_add": ops["+"] + ops["-"],
        "n_mul": ops["*"],
        "n_div": ops["/"],
        "n_flops": flops,
        "n_loads": len(uniq_reads),
        "n_stores": len(uniq_writes),
        "n_arrays": len(touched),
        "n_branches": n_branches(body),
        "n_unit_stride": unit,
        "n_nonunit_stride": nonunit,
        "n_invariant": invariant,
        "max_stride_log2": math.log2(max_stride),
        "has_reduction": red,
        "has_carried_dep": dep,
        "dep_distance": dist,
        "footprint_log2": math.log2(max(footprint, 1)),
        "bytes_per_iter": bytes_iter,
        "arith_intensity": flops / bytes_iter,
        "tac_len": body_instruction_count(body),
    }
    info = {"trips": trips, "loopvars": loopvars, "strides": strides,
            "tac_body": tac, "dtype": dtype, "footprint": footprint,
            "ntokens": len(tokenize(region))}
    return feat, info


if __name__ == "__main__":
    import json
    import sys
    f, i = analyze_source(open(sys.argv[1]).read())
    print(json.dumps(f, indent=2))
    print("\nTAC (loop nest):")
    print("\n".join(lower(parse_function(open(sys.argv[1]).read()))))
