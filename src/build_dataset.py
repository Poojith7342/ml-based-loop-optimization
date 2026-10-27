"""Builds the training dataset.

For every kernel in kernels/:
  1. extract the static loop features with the front end (loop_analyzer.py);
  2. compile + time it under all candidate configurations (benchmark.py);
  3. label it with the best *admissible* configuration.

Labelling rule
  * a configuration is admissible only if its numerical result agrees with the
    -O2 result (relative checksum error <= 1e-3)  -> semantic safety filter;
  * the label is the most conservative admissible configuration whose time is
    within TAU of the fastest one                  -> avoids over-fitting noise.

The run is resumable: finished kernels are appended to data/raw_timings.jsonl.
"""
import glob
import json
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from benchmark import CONFIG_NAMES, BASELINE, benchmark_kernel   # noqa: E402
from loop_analyzer import analyze_source, FEATURES               # noqa: E402

TAU = 0.05          # 5 % tie tolerance (a bit above the residual measurement noise)
CHK_TOL = 1e-3      # relative checksum tolerance


def family_of(name):
    return name.split("_")[0]


def label_kernel(res):
    base = res[BASELINE]["checksum"]
    ok = {}
    for c in CONFIG_NAMES:
        err = abs(res[c]["checksum"] - base) / (abs(base) + 1e-12)
        ok[c] = err <= CHK_TOL
    best = min(res[c]["time_ns"] for c in CONFIG_NAMES if ok[c])
    for c in CONFIG_NAMES:                       # conservative -> aggressive
        if ok[c] and res[c]["time_ns"] <= best * (1 + TAU):
            return c, ok
    return BASELINE, ok


def main():
    raw_path = os.path.join(ROOT, "data", "raw_timings.jsonl")
    done = set()
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            done = {json.loads(l)["kernel"] for l in f if l.strip()}
    files = sorted(glob.glob(os.path.join(ROOT, "kernels", "*.c")))
    t_start = time.time()
    for i, path in enumerate(files, 1):
        name = os.path.basename(path)[:-2]
        if name in done:
            continue
        res, spread = benchmark_kernel(path)
        with open(raw_path, "a") as f:
            f.write(json.dumps({"kernel": name, "family": family_of(name),
                                "results": res, "baseline_spread": spread}) + "\n")
        print(f"[{i:3d}/{len(files)}] {name:40s} elapsed {time.time() - t_start:6.0f}s", flush=True)

    # ---------------------------------------------------------------- tables
    rows_f, rows_t = [], []
    with open(raw_path) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    for r in recs:
        src = open(os.path.join(ROOT, "kernels", r["kernel"] + ".c")).read()
        feat, _ = analyze_source(src)
        label, ok = label_kernel(r["results"])
        rows_f.append({"kernel": r["kernel"], "family": r["family"], **feat, "label": label})
        row = {"kernel": r["kernel"], "family": r["family"], "baseline_spread": r["baseline_spread"]}
        for c in CONFIG_NAMES:
            row[f"t_{c}"] = r["results"][c]["time_ns"]
            row[f"compile_{c}"] = r["results"][c]["compile_s"]
            row[f"ok_{c}"] = ok[c]
        rows_t.append(row)
    pd.DataFrame(rows_f).to_csv(os.path.join(ROOT, "data", "dataset.csv"), index=False)
    pd.DataFrame(rows_t).to_csv(os.path.join(ROOT, "data", "timings.csv"), index=False)
    print("dataset written:", len(rows_f), "kernels")


if __name__ == "__main__":
    main()
