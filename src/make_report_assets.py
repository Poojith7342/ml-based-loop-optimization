"""Turns the measured data (data/, results/) into LaTeX fragments in
report/generated/.  The report therefore never contains hand-typed numbers:
every table, plot and macro is regenerated from the experiment output."""
import json
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from benchmark import CONFIG_NAMES, BASELINE, FLAGS   # noqa: E402
from loop_analyzer import FEATURES                      # noqa: E402

GEN = os.path.join(ROOT, "report", "generated")
os.makedirs(GEN, exist_ok=True)

SHORT = {c: c.replace("_", "+") for c in CONFIG_NAMES}


def esc(s):
    return (str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")
            .replace("&", "\\&").replace("#", "\\#"))


def write(name, text):
    with open(os.path.join(GEN, name), "w") as f:
        f.write(text)


def table(header, rows, colspec, caption, label, small=True, wide=False):
    env = "table*" if wide else "table"
    out = [f"\\begin{{{env}}}[H]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if small:
        out.append("\\small")
    out += ["\\resizebox{\\textwidth}{!}{%" if len(colspec) >= 6 else "", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\",
            "\\midrule"]
    for r in rows:
        out.append(" & ".join(str(x) for x in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}" + ("}" if len(colspec) >= 6 else ""), f"\\end{{{env}}}"]
    return "\n".join(out) + "\n"


def main():
    ds = pd.read_csv(os.path.join(ROOT, "data", "dataset.csv"))
    tm = pd.read_csv(os.path.join(ROOT, "data", "timings.csv"))
    cmp_ = pd.read_csv(os.path.join(ROOT, "results", "model_comparison.csv"))
    fam = pd.read_csv(os.path.join(ROOT, "results", "per_family.csv"))
    cm = pd.read_csv(os.path.join(ROOT, "results", "confusion_matrix.csv"), index_col=0)
    imp = pd.read_csv(os.path.join(ROOT, "results", "feature_importance.csv"), index_col=0)["importance"]
    over = json.load(open(os.path.join(ROOT, "results", "overhead.json")))
    raw = [json.loads(l) for l in open(os.path.join(ROOT, "data", "raw_timings.jsonl")) if l.strip()]

    T = tm[[f"t_{c}" for c in CONFIG_NAMES]].to_numpy()
    OK = tm[[f"ok_{c}" for c in CONFIG_NAMES]].to_numpy().astype(bool)
    base = CONFIG_NAMES.index(BASELINE)
    t_best = np.where(OK, T, np.inf).min(axis=1)

    # ------------------------------------------------------ measurement noise
    spreads = np.array([r["baseline_spread"] for r in raw])
    odd_even = []
    for r in raw:
        rs = r["results"][BASELINE]["rounds"]
        a, b = min(rs[0::2]), min(rs[1::2])
        odd_even.append(abs(a / b - 1))
    odd_even = np.array(odd_even)

    # ------------------------------------------------------ macros / numbers
    try:
        cpu = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        cpu = platform.processor()
    gcc_v = subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    try:
        ncpu = os.cpu_count()
        mem = int([l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal")][0]) / 1048576
    except Exception:
        ncpu, mem = 0, 0
    sel = over["selected_model"]
    sel_cv = cmp_[(cmp_.model == sel) & (cmp_.protocol == "5-fold CV")].iloc[0]
    sel_lo = cmp_[(cmp_.model == sel) & (cmp_.protocol == "LOFO")].iloc[0]
    st = {r.model: r for r in cmp_[cmp_.protocol == "static"].itertuples()}
    pk = pd.read_csv(os.path.join(ROOT, "results", "portable_comparison.csv"))
    pst = {r.model: r for r in pk[pk.protocol == "static"].itertuples()}
    pcv = pk[(pk.model == "k-NN") & (pk.protocol == "5-fold CV")].iloc[0]
    plo = pk[(pk.model == "k-NN") & (pk.protocol == "LOFO")].iloc[0]
    nums = {
        "numKernels": len(ds), "numFamilies": ds.family.nunique(), "numConfigs": len(CONFIG_NAMES),
        "numFeatures": len(FEATURES), "numCompiles": len(ds) * len(CONFIG_NAMES),
        "numRuns": len(ds) * len(CONFIG_NAMES) * 5,
        "cpuModel": cpu, "gccVersion": gcc_v, "numCpus": ncpu, "memGB": f"{mem:.1f}",
        "noiseSpreadMedian": f"{100 * np.median(spreads):.1f}", "noiseSpreadMean": f"{100 * spreads.mean():.1f}",
        "noiseSplitMedian": f"{100 * np.median(odd_even):.1f}",
        "noiseSplitNinety": f"{100 * np.percentile(odd_even, 90):.1f}",
        "selModel": sel,
        "selAccCV": f"{100 * sel_cv.accuracy:.1f}", "selAccLOFO": f"{100 * sel_lo.accuracy:.1f}",
        "selFscoreCV": f"{sel_cv.macro_f1:.2f}", "selFscoreLOFO": f"{sel_lo.macro_f1:.2f}",
        "selGMCV": f"{sel_cv.gm_speedup:.3f}", "selGMLOFO": f"{sel_lo.gm_speedup:.3f}",
        "oracleGM": f"{sel_cv.oracle_gm_speedup:.3f}",
        "selFracCV": f"{100 * sel_cv.oracle_fraction:.1f}", "selFracLOFO": f"{100 * sel_lo.oracle_fraction:.1f}",
        "selSlowCV": f"{sel_cv.slowdowns:.1f}", "selSlowLOFO": f"{int(sel_lo.slowdowns)}",
        "selRegretCV": f"{sel_cv.regret_pct:.1f}", "selRegretLOFO": f"{sel_lo.regret_pct:.1f}",
        "gmOThree": f"{st['Always -O3'].gm_speedup:.3f}",
        "gmONative": f"{st['Always -O3 -march=native'].gm_speedup:.3f}",
        "gmOFast": f"{st['Always -O3 -march=native -ffast-math'].gm_speedup:.3f}",
        "fastSlow": f"{int(st['Always -O3 -march=native -ffast-math'].slowdowns)}",
        "fastRegret": f"{st['Always -O3 -march=native -ffast-math'].regret_pct:.1f}",
        "fastFrac": f"{100 * st['Always -O3 -march=native -ffast-math'].oracle_fraction:.1f}",
        "pOracle": f"{pk.iloc[0].oracle_gm_speedup:.3f}", "pOThree": f"{pst['Always -O3'].gm_speedup:.3f}",
        "pOThreeSlow": f"{int(pst['Always -O3'].slowdowns)}",
        "pCV": f"{pcv.gm_speedup:.3f}", "pLOFO": f"{plo.gm_speedup:.3f}",
        "pFracCV": f"{100 * pcv.oracle_fraction:.1f}", "pFracLOFO": f"{100 * plo.oracle_fraction:.1f}",
        "pSlowCV": f"{pcv.slowdowns:.1f}", "pSlowLOFO": f"{int(plo.slowdowns)}",
        "featMs": f"{over['feature_extraction_ms']:.1f}", "predMs": f"{over['prediction_ms']:.1f}",
        "gccCompileMs": f"{1000 * over['mean_gcc_compile_s']:.0f}",
    }
    write("numbers.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{esc(v)}}}" for k, v in nums.items()) + "\n")

    # ------------------------------------------------------ tables
    fam_desc = {
        "axpy": ("vector update $y_i=\\sum w_jx^{(j)}_i+\\dots$", "1"),
        "dot": ("dot product / norm (FP reduction)", "1"),
        "recur": ("first-order recurrence $a_i=f(a_{i-d})$", "1"),
        "stencil1d": ("3- and 5-point 1-D stencil", "1"),
        "stencil2d": ("5-point 2-D Jacobi stencil", "2"),
        "matmul": ("matrix product, loop orders ijk/ikj/jki", "3"),
        "traverse2d": ("2-D row/column sums, copy, transpose", "2"),
        "cond": ("loop with data-dependent branch", "1"),
        "strided": ("non-unit-stride gather $x[s\\,i]$", "1"),
    }
    rows = []
    for f, g in ds.groupby("family"):
        rows.append([f"\\texttt{{{f}}}", fam_desc.get(f, ("", ""))[0], fam_desc.get(f, ("", ""))[1], len(g),
                     f"$2^{{{g.footprint_log2.min():.0f}}}$--$2^{{{g.footprint_log2.max():.0f}}}$"])
    rows.append(["\\midrule", "", "", "", ""])
    rows[-1] = ["\\textbf{Total}", "", "", f"\\textbf{{{len(ds)}}}", ""]
    write("tab_corpus.tex", table(["Family", "Loop pattern", "Depth", "Kernels", "Footprint (bytes)"], rows,
                                  "llcrr", "Composition of the loop-kernel corpus.", "tab:corpus"))

    cfg_rows = [[f"\\texttt{{{esc(c)}}}", f"\\texttt{{{esc(' '.join(FLAGS[c]))}}}"] for c in CONFIG_NAMES]
    write("tab_configs.tex", table(["Class", "GCC flags"], cfg_rows, "ll",
                                   "Decision space: the seven candidate optimisation configurations.", "tab:configs"))

    # labels + static-configuration performance
    lab = ds.label.value_counts()
    rows = []
    for i, c in enumerate(CONFIG_NAMES):
        t_eff = np.where(OK[:, i], T[:, i], T[:, base])
        gm = float(np.exp(np.log(T[:, base] / t_eff).mean()))
        wins = int((np.isclose(t_eff, t_best, rtol=0.05) & OK[:, i]).sum())
        rows.append([f"\\texttt{{{esc(c)}}}", int(lab.get(c, 0)), f"{100 * lab.get(c, 0) / len(ds):.1f}",
                     f"{gm:.3f}", wins, int((~OK[:, i]).sum())])
    gm_or = float(np.exp(np.log(T[:, base] / t_best).mean()))
    rows.append(["\\textbf{Oracle}", "--", "--", f"\\textbf{{{gm_or:.3f}}}", "--", "--"])
    write("tab_labels.tex", table(
        ["Config", "Label count", "Label \\%", "GM speed-up", "Within 5\\% of best", "Rejected"], rows, "lrrrrr",
        "Label distribution and performance of each \\emph{static} configuration over all kernels (speed-up w.r.t.\\ \\texttt{O2}; "
        "a configuration that changes the numerical result is replaced by \\texttt{O2}).", "tab:labels"))

    # model comparison
    rows = []
    order = ["Always -O2", "Always -O3", "Always -O3 -march=native", "Always -O3 -march=native -ffast-math", "Majority class", "Decision Tree", "Random Forest",
             "Gradient Boosting", "k-NN"]
    for m in order:
        for proto in ("static", "5-fold CV", "LOFO"):
            r = cmp_[(cmp_.model == m) & (cmp_.protocol == proto)]
            if r.empty:
                continue
            r = r.iloc[0]
            acc = f"{100 * r.accuracy:.1f}"
            if proto == "5-fold CV" and "acc_std" in r and not pd.isna(r.acc_std):
                acc += f" $\\pm$ {100 * r.acc_std:.1f}"
            rows.append([esc(m), proto if proto != "static" else "--", acc, f"{r.macro_f1:.2f}",
                         f"{r.gm_speedup:.3f}", f"{100 * r.oracle_fraction:.0f}" if not pd.isna(r.oracle_fraction) else "--",
                         f"{r.regret_pct:.1f}", f"{r.slowdowns:.0f}"])
    write("tab_models.tex", table(
        ["Model", "Protocol", "Acc.\\ (\\%)", "Macro-F1", "GM speed-up", "Oracle gain (\\%)", "Regret (\\%)", "Slow-downs"],
        rows, "llrrrrrr",
        "Comparison of decision models and static baselines. GM = geometric mean over kernels; "
        "``Oracle gain'' is the share of the oracle's log-speed-up that is realised; "
        "``Slow-downs'' counts kernels more than 5\\% slower than \\texttt{O2}.", "tab:models"))

    # portable scenario table
    rows = []
    for r in pk.itertuples():
        rows.append([esc(r.model), r.protocol if r.protocol != "static" else "--", f"{100 * r.accuracy:.1f}",
                     f"{r.gm_speedup:.3f}", f"{100 * r.oracle_fraction:.0f}", f"{r.regret_pct:.1f}", f"{r.slowdowns:.0f}"])
    rows.append(["\\textbf{Oracle}", "--", "--", f"\\textbf{{{pk.iloc[0].oracle_gm_speedup:.3f}}}", "100", "0.0", "0"])
    write("tab_portable.tex", table(["Model", "Protocol", "Acc.\\ (\\%)", "GM speed-up", "Oracle gain (\\%)", "Regret (\\%)", "Slow-downs"],
          rows, "llrrrrr", "Portable setting: only the four IEEE-safe, non-native configurations "
          "(\\texttt{O2}, \\texttt{O3}, \\texttt{O2\\_unroll4}, \\texttt{O2\\_unroll8}) are allowed.", "tab:portable"))

    # per family
    rows = []
    fm = fam[fam.protocol == "kfold"].set_index("family")
    fl = fam[fam.protocol == "lofo"].set_index("family")
    for f in fm.index:
        rows.append([f"\\texttt{{{f}}}", int(fm.loc[f, "n"]), f"{fm.loc[f, 'oracle_gm_speedup']:.2f}",
                     f"{fm.loc[f, 'gm_speedup']:.2f}", f"{fl.loc[f, 'gm_speedup']:.2f}",
                     f"{100 * fm.loc[f, 'accuracy']:.0f}", f"{100 * fl.loc[f, 'accuracy']:.0f}"])
    write("tab_family.tex", table(
        ["Family", "$n$", "Oracle", "Model (CV)", "Model (LOFO)", "Acc.\\ CV (\\%)", "Acc.\\ LOFO (\\%)"], rows, "lrrrrrr",
        f"Per-family geometric-mean speed-up over \\texttt{{O2}} of the selected model ({esc(sel)}).", "tab:family"))

    # confusion matrix
    short = [esc(SHORT[c]) for c in CONFIG_NAMES]
    rows = [[f"\\textbf{{{esc(SHORT[r])}}}"] + [int(v) for v in cm.loc[r].values] for r in cm.index]
    write("tab_confusion.tex", table(["True $\\backslash$ Pred"] + short, rows, "l" + "r" * len(short),
                                     f"Confusion matrix of the {esc(sel)} (5-fold cross validation).",
                                     "tab:confusion", small=True, wide=False).replace("\\small", "\\scriptsize"))

    # ------------------------------------------------------ figures (pgfplots)
    def bars(name, labels, series, ylabel, caption, label, ymin=None, ymax=None, width="0.95\\textwidth", rot=30,
             ybar_interval=False):
        coords = ",".join(esc(l) for l in labels)
        plots = ""
        for (title, vals) in series:
            plots += "\\addplot coordinates {" + " ".join(f"({esc(l)},{v:.4f})" for l, v in zip(labels, vals)) + \
                     "};\n"
        leg = ",".join(esc(t) for t, _ in series)
        yb = f"ymin={ymin}," if ymin is not None else "ymin=0,"
        yx = f"ymax={ymax}," if ymax is not None else ""
        txt = f"""\\begin{{figure}}[H]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    ybar, width={width}, height=7cm,
    ylabel={{{ylabel}}},
    symbolic x coords={{{coords}}},
    xtick=data, x tick label style={{rotate={rot},anchor=east,font=\\small}},
    {yb} {yx}
    bar width={max(4, 14 - 2 * len(series))}pt,
    enlarge x limits=0.08,
    legend style={{at={{(0.5,1.03)}},anchor=south,legend columns=-1,font=\\small}},
    grid=major]
{plots}{"\\legend{" + leg + "}" if len(series) > 1 else ""}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}
"""
        write(name, txt)

    # (1) overall speed-up of each approach
    def gm_of(model, proto):
        r = cmp_[(cmp_.model == model) & (cmp_.protocol == proto)].iloc[0]
        return r.gm_speedup
    labels = ["Always O2", "Always O3", "O3 native", "O3 fast native", "Tree (CV)", "Forest (CV)", "GBoost (CV)", "kNN (CV)", "Oracle"]
    vals = [1.0, gm_of("Always -O3", "static"), gm_of("Always -O3 -march=native", "static"),
            gm_of("Always -O3 -march=native -ffast-math", "static"), gm_of("Decision Tree", "5-fold CV"), gm_of("Random Forest", "5-fold CV"),
            gm_of("Gradient Boosting", "5-fold CV"), gm_of("k-NN", "5-fold CV"), gm_or]
    bars("fig_speedup.tex", labels, [("GM speed-up", vals)], "Geometric-mean speed-up vs O2",
         "Geometric-mean speed-up over \\texttt{O2} of the static configurations, the decision models "
         "(5-fold CV) and the oracle.", "fig:speedup", ymin=0.8)

    # (2) per-family
    fams = list(fm.index)
    bars("fig_family.tex", fams,
         [("Oracle", [fm.loc[f, "oracle_gm_speedup"] for f in fams]),
          ("Model (CV)", [fm.loc[f, "gm_speedup"] for f in fams]),
          ("Model (LOFO)", [fl.loc[f, "gm_speedup"] for f in fams])],
         "GM speed-up vs O2", f"Per-family speed-up over \\texttt{{O2}} for the {esc(sel)}: oracle, random-split "
         "cross validation, and leave-one-family-out.", "fig:family", ymin=0.5, rot=35)

    # (3) feature importance
    top = imp.head(10)[::-1]
    coords = ",".join(esc(k) for k in top.index)
    pts = " ".join(f"({v:.4f},{esc(k)})" for k, v in top.items())
    write("fig_importance.tex", f"""\\begin{{figure}}[H]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    xbar, width=0.8\\textwidth, height=7.5cm,
    xlabel={{Random-forest feature importance}},
    symbolic y coords={{{coords}}},
    ytick=data, xmin=0, bar width=8pt, enlarge y limits=0.08, grid=major]
\\addplot coordinates {{{pts}}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Ten most important loop features.}}
\\label{{fig:importance}}
\\end{{figure}}
""")

    # (4) scatter-like: per-kernel best speed-up distribution as sorted curve
    sp_or = np.sort(T[:, base] / t_best)[::-1]
    pts = " ".join(f"({i},{v:.3f})" for i, v in enumerate(sp_or))
    write("fig_oracle_curve.tex", f"""\\begin{{figure}}[H]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[width=0.85\\textwidth, height=6.5cm, xlabel={{Kernel (sorted by potential gain)}},
    ylabel={{Oracle speed-up vs O2}}, ymode=log, grid=major, xmin=0, xmax={len(sp_or)}]
\\addplot[thick, blue, no marks] coordinates {{{pts}}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Best achievable speed-up over \\texttt{{O2}} for each kernel (oracle).}}
\\label{{fig:oracle}}
\\end{{figure}}
""")

    # decision-tree rules
    rules = open(os.path.join(ROOT, "results", "decision_tree_rules.txt")).read()
    write("decision_tree_rules.txt", rules)
    print("report assets written to", GEN)


if __name__ == "__main__":
    main()
