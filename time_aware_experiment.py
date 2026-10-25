"""
time_aware_experiment.py - our extension of the paper (two ideas from its "future work"):

 1) TIME-AWARE SPLIT. Inside every class the flows are ordered by time; the first 80% are
    used for training and the LAST 20% for testing. Random splits mix flows from the same
    minute into train and test, which makes scores look better than they would be in
    real use. (Timestamp is used ONLY to order the rows - never as a feature.)
 2) EXFILTRATION THRESHOLD TUNING. Missing Exfiltration is the costliest error, so we
    lower the probability needed to predict it. The threshold is chosen on a VALIDATION
    block (the last 20% of the training period), never on the test block.

Run after main.py setup (sample CSV in data/, config.py set for MedSec-25):
    python time_aware_experiment.py
Output: results/paper_comparison/time_aware/
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

import config
from paper_comparison import build_models, make_cfg, metrics_for, prepare
from src import data_loader

warnings.filterwarnings("ignore")
OUT = config.RESULTS_DIR / "paper_comparison" / "time_aware"
OUT.mkdir(parents=True, exist_ok=True)


def time_split(y, train_fraction):
    """Per class: earliest `train_fraction` of rows -> train, the rest -> later block.
    Rows of y must already be in time order. Returns (train_idx, later_idx)."""
    train, later = [], []
    for k in np.unique(y):
        idx = np.where(y == k)[0]
        cut = int(round(len(idx) * train_fraction))
        train += idx[:cut].tolist()
        later += idx[cut:].tolist()
    return np.array(sorted(train)), np.array(sorted(later))


def fit(pipe, name, X, y):
    if name == "XGBoost":
        pipe.fit(X, y, classifier__sample_weight=compute_sample_weight("balanced", y))
    else:
        pipe.fit(X, y)
    return pipe


def predict_with_threshold(proba, threshold, exf):
    """Predict Exfiltration when its probability >= threshold, otherwise the best other class."""
    others = proba.copy()
    others[:, exf] = -1.0
    return np.where(proba[:, exf] >= threshold, exf, others.argmax(axis=1))


def exfil_scores(y_true, pred, exf):
    yt, yp = (y_true == exf), (pred == exf)
    return (recall_score(yt, yp, zero_division=0), precision_score(yt, yp, zero_division=0),
            float((yp & ~yt).sum() / max((~yt).sum(), 1)))


def main():
    target = config.TARGET_COLUMN
    raw = data_loader.load_csv(data_loader.find_csv_file(config.DATA_DIR, config.CSV_FILE_NAME))
    raw.columns = [str(c).strip() for c in raw.columns]
    if "Timestamp" not in raw.columns:
        raise SystemExit("This experiment needs a 'Timestamp' column to order the flows in time.")
    when = pd.to_datetime(raw["Timestamp"], dayfirst=True, utc=True, format="mixed", errors="coerce")
    raw = raw.assign(_when=when).dropna(subset=["_when"]).sort_values("_when", kind="stable")
    raw = raw.drop(columns="_when").reset_index(drop=True)

    cfg = make_cfg(config.EXCLUDE_COLUMNS)          # Timestamp stays excluded as a FEATURE
    X, y, groups, class_names, _ = prepare(raw, cfg, target)   # row order (time) is preserved
    exf = [i for i, c in enumerate(class_names) if c.lower().startswith("exfil")][0]
    print(f"{len(X)} flows, classes={class_names}, Exfiltration index={exf}")

    # ---- A. random split vs time-aware split (same models, same features)
    tr_r, te_r = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    tr_t, te_t = time_split(y, 0.8)
    rows = []
    for split_name, (tr, te) in {"Random stratified 80/20": (tr_r, te_r), "Time-aware 80/20": (tr_t, te_t)}.items():
        models = build_models(groups, cfg)
        for name, pipe in models.items():
            fit(pipe, name, X.iloc[tr], y[tr])
            rows.append({"Split": split_name, "Model": name, "Test flows": len(te),
                         **metrics_for(pipe, X.iloc[te], y[te], class_names)})
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(OUT / "random_vs_time_aware.csv", index=False)

    # ---- B. Exfiltration threshold tuning (time-aware; threshold chosen on validation block only)
    inner_rel, val_rel = time_split(y[tr_t], 0.8)
    inner, val = tr_t[inner_rel], tr_t[val_rel]
    grid = np.round(np.linspace(0.02, 0.90, 45), 3)
    tune_rows, curves = [], {}
    for name in ("Random Forest", "XGBoost"):
        pipe = fit(build_models(groups, cfg)[name], name, X.iloc[inner], y[inner])
        p_val, p_te = pipe.predict_proba(X.iloc[val]), pipe.predict_proba(X.iloc[te_t])
        f2 = []
        for t in grid:
            r, p, _ = exfil_scores(y[val], predict_with_threshold(p_val, t, exf), exf)
            f2.append(5 * p * r / (4 * p + r) if (4 * p + r) > 0 else 0.0)
        best_t = float(grid[int(np.argmax(f2))])                     # chosen on VALIDATION only
        curves[name] = [exfil_scores(y[te_t], predict_with_threshold(p_te, t, exf), exf) for t in grid]
        for label, pred in (("Default (argmax)", p_te.argmax(axis=1)),
                            (f"Tuned threshold = {best_t:.2f}", predict_with_threshold(p_te, best_t, exf))):
            r, p, fpr = exfil_scores(y[te_t], pred, exf)
            tune_rows.append({"Model": name, "Decision rule": label, "Exfil recall": r, "Exfil precision": p,
                              "False alarms per 1000 non-exfil flows": 1000 * fpr,
                              "Macro F1": f1_score(y[te_t], pred, average="macro", zero_division=0),
                              "MCC": matthews_corrcoef(y[te_t], pred)})
        best_pred = predict_with_threshold(p_te, best_t, exf)
        pd.DataFrame(confusion_matrix(y[te_t], best_pred), index=class_names, columns=class_names
                     ).to_csv(OUT / f"confusion_time_aware_tuned_{name.replace(' ', '_')}.csv")
    tune_df = pd.DataFrame(tune_rows)
    tune_df.to_csv(OUT / "exfiltration_threshold_tuning.csv", index=False)

    # ---- figures
    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = list(build_models(groups, cfg).keys())
    x = np.arange(len(names))
    for i, split in enumerate(["Random stratified 80/20", "Time-aware 80/20"]):
        vals = [cmp_df[(cmp_df["Split"] == split) & (cmp_df["Model"] == n)]["Macro F1"].iloc[0] for n in names]
        bars = ax.bar(x + (i - 0.5) * 0.4, vals, 0.4, label=split)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.1); ax.set_ylabel("Macro F1")
    ax.set_title("Random vs time-aware split"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_random_vs_time_aware.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, name in zip(axes, curves):
        rec = [c[0] for c in curves[name]]; pre = [c[1] for c in curves[name]]
        ax.plot(grid, rec, label="Exfiltration recall"); ax.plot(grid, pre, label="Exfiltration precision")
        ax.set_title(f"{name} (time-aware test block)"); ax.set_xlabel("Probability threshold for Exfiltration")
        ax.legend(loc="lower left")
    axes[0].set_ylabel("Score")
    fig.tight_layout(); fig.savefig(OUT / "fig_exfiltration_threshold.png", dpi=170); plt.close(fig)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    show = ["Split", "Model", "Test flows", "Accuracy", "Macro F1", "MCC", "AUPRC (macro)"] + \
           [c for c in cmp_df.columns if c.startswith("Recall:")]
    summary = ["RANDOM vs TIME-AWARE SPLIT", cmp_df[show].round(3).to_string(index=False), "",
               "EXFILTRATION THRESHOLD TUNING (time-aware test block)", tune_df.round(3).to_string(index=False), ""]
    (OUT / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
