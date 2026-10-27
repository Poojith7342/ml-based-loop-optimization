"""Trains and evaluates the ML-based loop-optimisation decision model.

Task      : multi-class classification  loop features -> best compiler config
Models    : Decision Tree, Random Forest, Gradient Boosting, k-NN
Baselines : always -O2, always -O3, always -O3 -march=native, majority class
Protocols : (a) repeated 5-fold cross validation (random split)
            (b) leave-one-family-out (LOFO): the model is tested on a loop
                family that was never seen in training (hard generalisation test)
Metrics   : accuracy, macro-F1, geometric-mean speedup over -O2,
            fraction of the oracle's gain captured, regret, slowdowns.
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from benchmark import CONFIG_NAMES, BASELINE, FLAGS   # noqa: E402
from loop_analyzer import FEATURES, analyze_source    # noqa: E402

SEED = 42
REPEATS = 5


def make_models():
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=1, random_state=SEED, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
    }


def load():
    ds = pd.read_csv(os.path.join(ROOT, "data", "dataset.csv"))
    tm = pd.read_csv(os.path.join(ROOT, "data", "timings.csv"))
    assert (ds.kernel == tm.kernel).all()
    T = tm[[f"t_{c}" for c in CONFIG_NAMES]].to_numpy()
    OK = tm[[f"ok_{c}" for c in CONFIG_NAMES]].to_numpy().astype(bool)
    return ds, tm, T, OK


def realised_time(pred_idx, T, OK, base=None):
    """Time obtained when following the prediction; an inadmissible choice
    (wrong numerical result) falls back to the safe baseline."""
    base = CONFIG_NAMES.index(BASELINE) if base is None else base
    n = len(pred_idx)
    t = np.empty(n)
    unsafe = 0
    for i, p in enumerate(pred_idx):
        if OK[i, p]:
            t[i] = T[i, p]
        else:
            t[i] = T[i, base]
            unsafe += 1
    return t, unsafe


def summarize(y_true, y_pred, T, OK, names=None):
    names = names or CONFIG_NAMES
    idx = np.array([names.index(c) for c in y_pred])
    base = names.index(BASELINE)
    t_pred, unsafe = realised_time(idx, T, OK, base)
    t_base = T[:, base]
    t_best = np.where(OK, T, np.inf).min(axis=1)
    sp = t_base / t_pred
    sp_or = t_base / t_best
    gm = float(np.exp(np.log(sp).mean()))
    gm_or = float(np.exp(np.log(sp_or).mean()))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "gm_speedup": gm,
        "oracle_gm_speedup": gm_or,
        "oracle_fraction": float(np.log(gm) / np.log(gm_or)) if gm_or > 1 else float("nan"),
        "regret_pct": float(100 * (t_pred / t_best - 1).mean()),
        "slowdowns": int((t_pred > 1.05 * t_base).sum()),
        "unsafe_predictions": unsafe,
    }


def cv_predict(model_fn, X, y, groups, protocol, seed):
    pred = np.empty(len(y), dtype=object)
    if protocol == "kfold":
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed).split(X)
    else:
        splitter = LeaveOneGroupOut().split(X, y, groups)
    for tr, te in splitter:
        m = model_fn()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def main():
    ds, tm, T, OK = load()
    X = ds[FEATURES].to_numpy(dtype=float)
    y = ds["label"].to_numpy()
    groups = ds["family"].to_numpy()
    out = os.path.join(ROOT, "results")
    os.makedirs(out, exist_ok=True)

    summary = []
    # ---- baselines ---------------------------------------------------------
    for name, cfg in [("Always -O2", "O2"), ("Always -O3", "O3"), ("Always -O3 -march=native", "O3_native"),
                      ("Always -O3 -march=native -ffast-math", "O3_fast_native")]:
        summary.append({"model": name, "protocol": "static", **summarize(y, np.array([cfg] * len(y)), T, OK)})
    # majority class baseline (computed in CV)
    class Majority:
        def fit(self, X, y):
            vals, cnt = np.unique(y, return_counts=True)
            self.c = vals[cnt.argmax()]
            return self

        def predict(self, X):
            return np.array([self.c] * len(X), dtype=object)

    models = {"Majority class": Majority, **{k: (lambda k=k: make_models()[k]) for k in make_models()}}

    store = {}
    for mname, fn in models.items():
        # (a) repeated 5-fold CV
        rows = []
        for rep in range(REPEATS):
            pred = cv_predict(fn, X, y, groups, "kfold", SEED + rep)
            rows.append(summarize(y, pred, T, OK))
            if rep == 0:
                store[(mname, "kfold")] = pred
        avg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        std = {k: float(np.std([r[k] for r in rows])) for k in rows[0]}
        summary.append({"model": mname, "protocol": "5-fold CV", **avg,
                        "acc_std": std["accuracy"], "gm_std": std["gm_speedup"]})
        # (b) leave-one-family-out
        pred = cv_predict(fn, X, y, groups, "lofo", SEED)
        store[(mname, "lofo")] = pred
        summary.append({"model": mname, "protocol": "LOFO", **summarize(y, pred, T, OK)})
        print(f"{mname:18s} done", flush=True)

    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(out, "model_comparison.csv"), index=False)

    # ---- choose the deployment model by LOFO speedup + CV speedup ------------
    cand = sdf[sdf.model.isin(make_models().keys())]
    score = cand.groupby("model")["gm_speedup"].mean()
    best_name = score.idxmax()
    print("selected model:", best_name)

    # ---- per-family results of the selected model (LOFO and CV) ---------------
    fam_rows = []
    for proto in ("kfold", "lofo"):
        pred = store[(best_name, proto)]
        for fam in sorted(set(groups)):
            msk = groups == fam
            s = summarize(y[msk], pred[msk], T[msk], OK[msk])
            fam_rows.append({"family": fam, "protocol": proto, "n": int(msk.sum()), **s})
    pd.DataFrame(fam_rows).to_csv(os.path.join(out, "per_family.csv"), index=False)

    # ---- confusion matrix (selected model, 5-fold CV first repeat) -------------
    cm = confusion_matrix(y, store[(best_name, "kfold")], labels=CONFIG_NAMES)
    pd.DataFrame(cm, index=CONFIG_NAMES, columns=CONFIG_NAMES).to_csv(os.path.join(out, "confusion_matrix.csv"))

    # ---- per-kernel table -----------------------------------------------------
    pk = ds[["kernel", "family", "label"]].copy()
    pk["pred_cv"] = store[(best_name, "kfold")]
    pk["pred_lofo"] = store[(best_name, "lofo")]
    for c in CONFIG_NAMES:
        pk[f"speedup_{c}"] = T[:, CONFIG_NAMES.index(BASELINE)] / T[:, CONFIG_NAMES.index(c)]
    pk.to_csv(os.path.join(out, "per_kernel.csv"), index=False)

    # ---- final models fitted on all data ---------------------------------------
    final = make_models()[best_name].fit(X, y)
    joblib.dump({"model": final, "features": FEATURES, "classes": list(final.classes_),
                 "name": best_name}, os.path.join(out, "model.joblib"))
    imp = None
    rf = make_models()["Random Forest"].fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    imp.to_csv(os.path.join(out, "feature_importance.csv"), header=["importance"])
    dt = make_models()["Decision Tree"].fit(X, y)
    with open(os.path.join(out, "decision_tree_rules.txt"), "w") as f:
        f.write(export_text(dt, feature_names=FEATURES))
    joblib.dump({"model": dt, "features": FEATURES, "classes": list(dt.classes_), "name": "Decision Tree"},
                os.path.join(out, "model_tree.joblib"))

    # ---- portable / IEEE-safe scenario: only O2, O3, O2+unroll4, O2+unroll8 --------
    names_p = ["O2", "O3", "O2_unroll4", "O2_unroll8"]
    ip = [CONFIG_NAMES.index(c) for c in names_p]
    Tp, OKp = T[:, ip], OK[:, ip]
    best_p = np.where(OKp, Tp, np.inf).min(axis=1)
    yp = np.empty(len(Tp), dtype=object)
    for i in range(len(Tp)):
        for j, c in enumerate(names_p):
            if OKp[i, j] and Tp[i, j] <= best_p[i] * 1.05:
                yp[i] = c
                break
    prow = []
    for name, cfg in [("Always -O2", "O2"), ("Always -O3", "O3")]:
        prow.append({"model": name, "protocol": "static", **summarize(yp, np.array([cfg] * len(yp)), Tp, OKp, names_p)})
    for mname in ("Decision Tree", "Random Forest", "k-NN"):
        fn = lambda mname=mname: make_models()[mname]
        rows = [summarize(yp, cv_predict(fn, X, yp, groups, "kfold", SEED + r), Tp, OKp, names_p) for r in range(REPEATS)]
        prow.append({"model": mname, "protocol": "5-fold CV", **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]}})
        prow.append({"model": mname, "protocol": "LOFO",
                     **summarize(yp, cv_predict(fn, X, yp, groups, "lofo", SEED), Tp, OKp, names_p)})
    pd.DataFrame(prow).to_csv(os.path.join(out, "portable_comparison.csv"), index=False)
    pd.Series(yp).value_counts().to_csv(os.path.join(out, "portable_labels.csv"), header=["count"])

    # ---- overhead of the model ---------------------------------------------------
    sample = [os.path.join(ROOT, "kernels", k + ".c") for k in ds.kernel.sample(30, random_state=1)]
    t_fe, t_pr = [], []
    for p in sample:
        src = open(p).read()
        t0 = time.perf_counter()
        feat, _ = analyze_source(src)
        t1 = time.perf_counter()
        final.predict(np.array([[feat[f] for f in FEATURES]]))
        t2 = time.perf_counter()
        t_fe.append(t1 - t0)
        t_pr.append(t2 - t1)
    comp = tm[[f"compile_{c}" for c in CONFIG_NAMES]].to_numpy().mean()
    json.dump({"feature_extraction_ms": 1e3 * float(np.median(t_fe)),
               "prediction_ms": 1e3 * float(np.median(t_pr)),
               "mean_gcc_compile_s": float(comp), "selected_model": best_name},
              open(os.path.join(out, "overhead.json"), "w"), indent=2)
    print(sdf.round(3).to_string())


if __name__ == "__main__":
    main()
