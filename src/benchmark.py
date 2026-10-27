"""Compiles a kernel with every candidate optimisation configuration (the
decision space of the model) and measures its execution time."""
import os
import subprocess
import tempfile
import time

# Ordered from the most conservative to the most aggressive configuration.
# The order is used for tie-breaking when labelling (see build_dataset.py).
CONFIGS = [
    ("O2",              ["-O2"]),
    ("O3",              ["-O3"]),
    ("O2_unroll4",      ["-O2", "-funroll-loops", "--param", "max-unroll-times=4"]),
    ("O2_unroll8",      ["-O2", "-funroll-loops", "--param", "max-unroll-times=8"]),
    ("O3_native",       ["-O3", "-march=native"]),
    ("O3_unroll_native", ["-O3", "-funroll-loops", "-march=native"]),
    ("O3_fast_native",  ["-O3", "-march=native", "-ffast-math"]),
]
CONFIG_NAMES = [c for c, _ in CONFIGS]
FLAGS = dict(CONFIGS)
BASELINE = "O2"


def compile_kernel(src_path, flags, exe_path, cc="gcc"):
    t0 = time.perf_counter()
    subprocess.run([cc, *flags, "-w", "-o", exe_path, src_path, "-lm"],
                   check=True, capture_output=True, timeout=180)
    return time.perf_counter() - t0


def run_kernel(exe_path):
    out = subprocess.run([exe_path], check=True, capture_output=True, text=True, timeout=120).stdout
    t, chk = out.split()
    return float(t), float(chk)


ROUNDS = 5


def benchmark_kernel(src_path, workdir=None, rounds=ROUNDS):
    """Compile every configuration once, then time them in `rounds`
    interleaved (round-robin) passes and keep the minimum per configuration.
    Interleaving spreads slow drifts of the shared machine over all
    configurations instead of biasing one of them.

    Returns ({config: {compile_s, time_ns, checksum, rounds}}, spread) where
    spread is the relative range of the baseline timings across the rounds."""
    own = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="bench_")
    res, exes = {}, {}
    for name, flags in CONFIGS:
        exe = os.path.join(workdir, f"k_{name}")
        cs = compile_kernel(src_path, flags, exe)
        res[name] = {"compile_s": cs, "rounds": [], "checksum": None}
        exes[name] = exe
    for _ in range(rounds):
        for name in CONFIG_NAMES:
            t, chk = run_kernel(exes[name])
            res[name]["rounds"].append(t)
            res[name]["checksum"] = chk
    for name in CONFIG_NAMES:
        res[name]["time_ns"] = min(res[name]["rounds"])
    for exe in exes.values():
        os.remove(exe)
    if own:
        os.rmdir(workdir)
    b = res[BASELINE]["rounds"]
    return res, (max(b) - min(b)) / min(b)


if __name__ == "__main__":
    import sys
    r, rep = benchmark_kernel(sys.argv[1])
    for k, v in r.items():
        print(f"{k:18s} {v['time_ns']:14.1f} ns  compile {v['compile_s']:.2f}s  chk {v['checksum']:.6e}")
    print("baseline spread across rounds: %.1f %%" % (100 * rep))
