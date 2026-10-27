"""Command-line optimisation advisor.

    python src/predict.py kernels/matmul_double_128_jki.c            # advise
    python src/predict.py kernels/matmul_double_128_jki.c --run      # advise + measure

Pipeline: lex -> parse -> semantic/loop analysis -> TAC -> feature vector
          -> ML decision model -> gcc flags.
"""
import argparse
import os
import sys
import tempfile

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from benchmark import BASELINE, FLAGS, compile_kernel, run_kernel   # noqa: E402
from lexer import tokenize                                          # noqa: E402
from loop_analyzer import FEATURES, SemanticError, analyze_source   # noqa: E402
from loop_parser import parse_function                              # noqa: E402
from tac import lower                                               # noqa: E402




def explain_optimization(config, flags):
    """Convert GCC optimization configurations into readable explanations."""

    explanations = {
        "O2": {
            "name": "Standard Compiler Optimization",
            "reason": "Applies GCC's general optimization techniques."
        },
        "O2_unroll4": {
            "name": "Loop Unrolling",
            "reason": (
                "Reduces loop-control overhead by allowing multiple "
                "iterations to be processed within an unrolled loop body."
            )
        },
        "O3": {
            "name": "Aggressive Compiler Optimization",
            "reason": (
                "Enables additional compiler optimizations that may "
                "improve execution speed."
            )
        },
        "O3_native": {
            "name": "CPU-Specific Optimization",
            "reason": (
                "Enables -O3 and allows GCC to use instruction features "
                "supported by the current CPU."
            )
        },
        "O3_unroll_native": {
            "name": "Loop Unrolling with CPU-Specific Optimization",
            "reason": (
                "Combines loop unrolling with aggressive optimization "
                "and CPU-specific instruction selection."
            )
        },
        "O3_fast_native": {
            "name": "Aggressive Optimization with Fast-Math",
            "reason": (
                "Combines -O3 and CPU-specific optimization with "
                "relaxed floating-point math rules."
            )
        }
    }

    details = explanations.get(config)

    if details is None:
        return (
            "Compiler Optimization",
            "The ML model selected a compiler configuration "
            "from the available trained options."
        )

    return details["name"], details["reason"]







def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="C file containing a function named 'kernel' with a loop nest")
    ap.add_argument("--model", default=os.path.join(ROOT, "results", "model.joblib"))
    ap.add_argument("--run", action="store_true", help="compile with -O2 and the advised flags and time both")
    ap.add_argument("--tokens", action="store_true", help="print the token stream")
    ap.add_argument("--tac", action="store_true", help="print the three-address code of the loop nest")
    args = ap.parse_args()

    src = open(args.source).read()
    try:
        feat, info = analyze_source(src)
    except (SyntaxError, SemanticError) as e:
        print("compile error:", e)
        sys.exit(1)

    if args.tokens:
        region = src.split("KERNEL_BEGIN", 1)[1].split("KERNEL_END", 1)[0] if "KERNEL_BEGIN" in src else src
        print("--- tokens ---")
        for t in tokenize(region):
            print(t)
    if args.tac:
        region = src.split("KERNEL_BEGIN", 1)[1].split("KERNEL_END", 1)[0] if "KERNEL_BEGIN" in src else src
        print("--- three-address code ---")
        print("\n".join(lower(parse_function(region))))

    print(f"source file      : {args.source}")
    print(f"tokens           : {info['ntokens']}")
    print(f"loop nest        : depth {int(feat['nest_depth'])}, trip counts {info['trips']}, vars {info['loopvars']}")
    print(f"memory strides   : {info['strides']}")
    print("feature vector   :")
    for f in FEATURES:
        print(f"    {f:18s} {feat[f]:.3f}")

    bundle = joblib.load(args.model)
    model = bundle["model"]
    x = np.array([[feat[f] for f in bundle["features"]]])
    pred = model.predict(x)[0]
    print(f"\nmodel            : {bundle['name']}")
    if hasattr(model, "predict_proba"):
        for c, p in sorted(zip(model.classes_, model.predict_proba(x)[0]), key=lambda z: -z[1]):
            if p > 0:
                print(f"    P({c:17s}) = {p:.2f}")
    
    flags = " ".join(FLAGS[pred])
    optimization_name, explanation = explain_optimization(
    pred, FLAGS[pred]
    )

    print("\n" + "=" * 55)
    print("       ML-BASED LOOP OPTIMIZATION SYSTEM")
    print("=" * 55)

    print(f"\nInput C program       : {args.source}")
    print(f"Machine learning model: {bundle['name']}")

    print("\nRECOMMENDED OPTIMIZATION")
    print("-" * 30)
    print(f"Optimization technique: {optimization_name}")
    print(f"Selected configuration: {pred}")
    print(f"Compiler flags        : {flags}")

    print("\nWHY THIS OPTIMIZATION?")
    print(explanation)

    print("\nCOMPILER COMMAND")
    print(f"gcc {flags} -o kernel {args.source} -lm")

    print("\n" + "=" * 55)

    if args.run:
        with tempfile.TemporaryDirectory() as d:
            res = {}
            for name in (BASELINE, pred):
                exe = os.path.join(d, name)
                compile_kernel(args.source, FLAGS[name], exe)
                res[name] = min(run_kernel(exe)[0] for _ in range(3))
            tb, tp = res[BASELINE], res[pred]

            time_saved = tb - tp
            improvement = (time_saved / tb) * 100 if tb > 0 else 0
            speedup = tb / tp if tp > 0 else 0

            print("\n" + "=" * 55)
            print("             PERFORMANCE EVALUATION")
            print("=" * 55)

            print(f"\nBaseline configuration : {BASELINE}")
            print(f"Recommended technique  : {optimization_name}")

            print(f"\nBaseline execution time  : {tb:.1f} ns")
            print(f"Optimized execution time : {tp:.1f} ns")
            print(f"Time saved               : {time_saved:.1f} ns")
            print(f"Performance improvement  : {improvement:.2f}%")
            print(f"Measured speed-up        : {speedup:.2f}x")

            if tp < tb:
                print(
                       "\nConclusion:\n"
                       "The recommended configuration was faster than the\n"
                       "baseline in this measurement."
                     )
            elif tp > tb:
                print(
                       "\nConclusion:\n"
                       "The recommended configuration was slower than the\n"
                       "baseline in this measurement."
                    )
            else:
                print(
                       "\nConclusion:\n"
                       "Both configurations had the same measured execution time."
                      )

                print("\nNote: Performance can vary between runs.")
                print("=" * 55)

                
if __name__ == "__main__":
    main()
