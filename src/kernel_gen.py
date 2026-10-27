"""Generates the training corpus of C loop kernels (HPC-style loop nests).

Nine kernel families are produced by sweeping data type, problem size, number
of operands, access stride, loop order, dependence distance, ...  Every kernel
is a self-contained C file: declarations, the kernel function between the
KERNEL_BEGIN / KERNEL_END markers, and a timing harness.
"""
import os
import random
import sys

SIZES_1D = [1024, 8192, 65536, 524288]
SIZES_2D = [64, 128, 256, 512, 1024]

HARNESS = r'''
static unsigned long long rng_state = 88172645463325252ULL;
static double rnd(void){ rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7;
                         rng_state ^= rng_state << 17;
                         return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
                            return (double)t.tv_sec * 1e9 + (double)t.tv_nsec; }
#define FILL(arr) for (long q = 0; q < (long)(sizeof(arr)/sizeof(DT)); q++) ((DT*)arr)[q] = (DT)(SCALE * rnd())
#define SUM(arr)  for (long q = 0; q < (long)(sizeof(arr)/sizeof(DT)); q++) chk += (double)((DT*)arr)[q]
static void init(void){ rng_state = 88172645463325252ULL; %(fill)s }
int main(void){
    volatile DT sink = 0;
    init();
    double chk = 0;
    for (int r = 0; r < 3; r++) chk += (double)kernel(r);
    %(sum)s
    init();
    sink = sink + kernel(0);
    double t0 = now_ns();
    sink = sink + kernel(1);
    double t1 = now_ns() - t0;
    if (t1 < 1) t1 = 1;
    long reps = (long)(15e6 / t1);
    if (reps < 1) reps = 1;
    if (reps > 2000000) reps = 2000000;
    double best = 1e30;
    for (int trial = 0; trial < 5; trial++) {
        t0 = now_ns();
        for (long k = 0; k < reps; k++) sink = sink + kernel((int)k);
        double dt = (now_ns() - t0) / (double)reps;
        if (dt < best) best = dt;
    }
    printf("%%.3f %%.9e\n", best, chk);
    return 0;
}
'''


def _source(dtype, defines, arrays, body, scale=1.0, ret="0"):
    """Assemble one C file.  `arrays` maps name -> dimension string(s)."""
    decl = "static DT " + ", ".join(f"{n}{d}" for n, d in arrays.items()) + ";"
    fill = " ".join(f"FILL({n});" for n in arrays)
    sums = " ".join(f"SUM({n});" for n in arrays)
    head = ["#include <stdio.h>", "#include <stdlib.h>", "#include <time.h>",
            f"typedef {dtype} DT;", f"#define SCALE {scale}"]
    head += [f"#define {k} {v}" for k, v in defines.items()]
    head.append(decl)
    kern = ("// KERNEL_BEGIN\n"
            "__attribute__((noinline, noipa)) DT kernel(int r)\n{\n" + body +
            f"    return {ret};\n}}\n// KERNEL_END\n")
    return "\n".join(head) + "\n\n" + kern + HARNESS % {"fill": fill, "sum": sums}


def _indent(lines, n=4):
    return "\n".join(" " * n + l for l in lines) + "\n"


# --------------------------------------------------------------------- families
def axpy(dt, n, nin, deg):
    ins = [f"x{k}" for k in range(nin)]
    arrs = {a: "[N]" for a in ins + ["y"]}
    terms = " + ".join(f"{0.5 / (k + 1):.3f}*{a}[i]" for k, a in enumerate(ins))
    lines = ["for (int i = 0; i < N; i++) {"]
    if deg > 1:
        lines.append("    DT v = x0[i];")
        terms = "((0.300*v + 0.200)*v + 0.100)*v + " + terms
    lines += [f"    y[i] = {terms} + 0.250*y[i];", "}"]
    return _source(dt, {"N": n}, arrs, _indent(lines), 1.0, "y[0]")


def dot(dt, n, nin):
    ins = [f"x{k}" for k in range(nin)]
    arrs = {a: "[N]" for a in ins}
    expr = "*".join(f"{a}[i]" for a in ins) if nin > 1 else "x0[i]*x0[i]"
    lines = ["DT s = 0;", "for (int i = 0; i < N; i++) {", f"    s += {expr};", "}"]
    return _source(dt, {"N": n}, arrs, _indent(lines), 1.0, "s")


def recur(dt, n, d):
    lines = [f"for (int i = {d}; i < N; i++) {{",
             f"    a[i] = 0.5*a[i-{d}] + 0.01*b[i];", "}"]
    return _source(dt, {"N": n}, {"a": "[N]", "b": "[N]"}, _indent(lines), 1.0, "a[0]")


def stencil1d(dt, n, w):
    if w == 3:
        e, lo = "0.25*x[i-1] + 0.5*x[i] + 0.25*x[i+1]", 1
    else:
        e, lo = "0.1*x[i-2] + 0.2*x[i-1] + 0.4*x[i] + 0.2*x[i+1] + 0.1*x[i+2]", 2
    lines = [f"for (int i = {lo}; i < N - {lo}; i++) {{", f"    y[i] = {e};", "}"]
    return _source(dt, {"N": n}, {"x": "[N]", "y": "[N]"}, _indent(lines), 1.0, "y[1]")


def stencil2d(dt, m):
    lines = ["for (int i = 1; i < M - 1; i++) {", "    for (int j = 1; j < M - 1; j++) {",
             "        B[i][j] = 0.2*(A[i][j] + A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]);",
             "    }", "}"]
    return _source(dt, {"M": m}, {"A": "[M][M]", "B": "[M][M]"}, _indent(lines), 1.0, "B[1][1]")


def matmul(dt, m, order):
    idx = {"i": "i", "j": "j", "k": "k"}
    lines, ind = [], ""
    for v in order:
        lines.append(f"{ind}for (int {v} = 0; {v} < M; {v}++) {{")
        ind += "    "
    lines.append(f"{ind}C[i][j] += A[i][k]*B[k][j];")
    for _ in order:
        ind = ind[:-4]
        lines.append(f"{ind}}}")
    return _source(dt, {"M": m}, {"A": "[M][M]", "B": "[M][M]", "C": "[M][M]"},
                   _indent(lines), 0.01, "C[0][0]")


def traverse2d(dt, m, mode):
    if mode == "rowsum":
        lines = ["DT s = 0;", "for (int i = 0; i < M; i++) {", "    for (int j = 0; j < M; j++) {",
                 "        s += A[i][j];", "    }", "}"]
        arrs, ret = {"A": "[M][M]"}, "s"
    elif mode == "colsum":
        lines = ["DT s = 0;", "for (int j = 0; j < M; j++) {", "    for (int i = 0; i < M; i++) {",
                 "        s += A[i][j];", "    }", "}"]
        arrs, ret = {"A": "[M][M]"}, "s"
    elif mode == "copy":
        lines = ["for (int i = 0; i < M; i++) {", "    for (int j = 0; j < M; j++) {",
                 "        B[i][j] = 0.5*A[i][j] + 1.0;", "    }", "}"]
        arrs, ret = {"A": "[M][M]", "B": "[M][M]"}, "B[0][0]"
    else:  # transpose (write is column-strided)
        lines = ["for (int i = 0; i < M; i++) {", "    for (int j = 0; j < M; j++) {",
                 "        B[j][i] = 0.5*A[i][j] + 1.0;", "    }", "}"]
        arrs, ret = {"A": "[M][M]", "B": "[M][M]"}, "B[0][0]"
    return _source(dt, {"M": m}, arrs, _indent(lines), 1.0, ret)


def cond(dt, n, thr, heavy):
    t = "0.5*x[i] + 0.1" if not heavy else "(0.5*x[i] + 0.1)*x[i] + 0.2"
    lines = ["for (int i = 0; i < N; i++) {", f"    if (x[i] > {thr}) {{", f"        y[i] = {t};",
             "    } else {", "        y[i] = 0.25*x[i];", "    }", "}"]
    return _source(dt, {"N": n}, {"x": "[N]", "y": "[N]"}, _indent(lines), 1.0, "y[0]")


def strided(dt, n, s):
    lines = ["for (int i = 0; i < N; i++) {", f"    y[i] = 0.5*x[i*{s}] + 0.25*y[i];", "}"]
    return _source(dt, {"N": n, "XN": n * s}, {"x": "[XN]", "y": "[N]"}, _indent(lines), 1.0, "y[0]")


# --------------------------------------------------------------------- corpus
def build_corpus(seed=7):
    rnd = random.Random(seed)
    corpus = []   # (name, family, source)

    def add(fam, cfgs, fn, cap):
        rnd.shuffle(cfgs)
        for c in cfgs[:cap]:
            name = (fam + "_" + "_".join(str(x) for x in c)).replace(".", "p")
            corpus.append((name, fam, fn(*c)))

    dts = ["double", "float"]
    add("axpy", [(d, n, k, g) for d in dts for n in SIZES_1D for k in (1, 2, 3) for g in (1, 3)], axpy, 36)
    add("dot", [(d, n, k) for d in dts for n in SIZES_1D for k in (1, 2, 3)], dot, 18)
    add("recur", [(d, n, k) for d in dts for n in SIZES_1D for k in (1, 2, 4, 8)], recur, 24)
    add("stencil1d", [(d, n, w) for d in dts for n in SIZES_1D for w in (3, 5)], stencil1d, 16)
    add("stencil2d", [(d, m) for d in dts for m in SIZES_2D[:4]], stencil2d, 8)
    add("matmul", [(d, m, o) for d in dts for m in (32, 48, 64, 96, 128, 192) for o in ("ijk", "ikj", "jki")],
        matmul, 24)
    add("traverse2d", [(d, m, md) for d in dts for m in SIZES_2D[1:] for md in ("rowsum", "colsum", "copy", "transpose")],
        traverse2d, 20)
    add("cond", [(d, n, t, h) for d in dts for n in SIZES_1D for t in (0.5, 0.9) for h in (0, 1)], cond, 16)
    add("strided", [(d, n, s) for d in dts for n in SIZES_1D for s in (2, 4, 8, 16)], strided, 20)
    return corpus


def write_corpus(outdir):
    os.makedirs(outdir, exist_ok=True)
    corpus = build_corpus()
    for name, fam, src in corpus:
        with open(os.path.join(outdir, name + ".c"), "w") as f:
            f.write(src)
    return [(n, f) for n, f, _ in corpus]


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "kernels"
    items = write_corpus(out)
    print(f"wrote {len(items)} kernels to {out}")
