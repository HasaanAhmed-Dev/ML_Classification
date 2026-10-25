"""
paper_comparison.py - extra analysis used to compare our results with the paper
"Characterization of Network Traffic Features for Intrusion Detection in IoMT
Cybersecurity" (Hananto et al., INFORMATIK 2026).

Run AFTER putting the 10,000-row MedSec-25 sample in data/ and setting
TARGET_COLUMN / EXCLUDE_COLUMNS in config.py:   python paper_comparison.py
Outputs go to results/paper_comparison/.
"""
import types
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import config
from src import data_loader, data_preprocessing as prep, train_models
from src.utils import DecisionLog

warnings.filterwarnings("ignore")
OUT = config.RESULTS_DIR / "paper_comparison"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]

# Top-20 lists READ BY HAND from Fig. 3 of the paper (paper naming -> column name in this CSV)
PAPER_MI = ["Active Mean", "RST Flag Cnt", "Down/Up Ratio", "TotLen Bwd Pkts", "Bwd IAT Mean",
            "Fwd Pkt Len Min", "Bwd Seg Size Avg", "Subflow Fwd Pkts", "Fwd IAT Mean", "FIN Flag Cnt",
            "Flow Pkts/s", "Pkt Size Avg", "Bwd Pkt Len Mean", "Bwd Pkt Len Max", "Flow IAT Mean",
            "Subflow Bwd Pkts", "Init Bwd Win Byts", "Fwd Pkt Len Std", "Flow Duration", "TotLen Fwd Pkts"]
PAPER_ANOVA = ["TotLen Fwd Pkts", "FIN Flag Cnt", "Fwd Pkt Len Std", "Bwd Seg Size Avg", "Active Mean",
               "Bwd Pkt Len Max", "URG Flag Cnt", "Idle Mean", "Flow IAT Mean", "Init Bwd Win Byts",
               "PSH Flag Cnt", "Fwd Pkt Len Max", "Flow IAT Std", "Idle Std", "Tot Fwd Pkts",
               "Fwd IAT Tot", "Subflow Bwd Pkts", "Bwd IAT Mean", "Bwd IAT Tot", "Fwd Seg Size Avg"]


def make_cfg(exclude):
    """Copy of config.py with a different EXCLUDE_COLUMNS list."""
    values = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    values["EXCLUDE_COLUMNS"] = exclude
    return types.SimpleNamespace(**values)


def prepare(raw, cfg, target):
    """Same cleaning as main.py. Returns X, y (encoded), groups, class_names, removed_df."""
    log = DecisionLog(echo=False)
    df = prep.normalize_dataframe(raw, log)
    df = prep.convert_numeric_like(df, target, log)
    df, boolean_cols = prep.convert_booleans(df, target, log)
    df, datetime_cols = prep.convert_datetimes(df, target, log)
    df, removed = prep.clean_dataset(df, target, cfg, log)
    df, _ = prep.extract_datetime_features(df, datetime_cols, removed, log)
    groups = prep.detect_feature_types(df, target, boolean_cols, cfg, log)
    labels = prep.prepare_target_labels(df[target])
    y, _enc, class_names = prep.encode_target(labels)
    return df.drop(columns=[target]), y, groups, class_names, pd.DataFrame(removed)


def class_statistics(y, class_names):
    counts = np.bincount(y)
    p = counts / counts.sum()
    return {"IR": counts.max() / counts.min(), "Gini": 1 - (p ** 2).sum(),
            "Entropy (ln)": float(-(p * np.log(p)).sum()), "counts": dict(zip(class_names, counts.tolist()))}


def relevance_scores(X, y, numeric_cols, seed=42):
    """ANOVA F-score and Mutual Information for numeric features."""
    filled = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X[numeric_cols]), columns=numeric_cols)
    nonconst = [c for c in numeric_cols if filled[c].nunique() > 1]
    f, p = f_classif(filled[nonconst], y)
    mi = mutual_info_classif(filled[nonconst], y, random_state=seed)
    return pd.DataFrame({"Feature": nonconst, "ANOVA F": f, "ANOVA p": p, "MI": mi})


def metrics_for(model, X_test, y_test, class_names):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    k = len(class_names)
    ap = [average_precision_score((y_test == c).astype(int), proba[:, c]) for c in range(k)]
    recalls = recall_score(y_test, pred, labels=list(range(k)), average=None, zero_division=0)
    out = {"Accuracy": accuracy_score(y_test, pred),
           "Macro F1": f1_score(y_test, pred, average="macro", zero_division=0),
           "Weighted F1": f1_score(y_test, pred, average="weighted", zero_division=0),
           "MCC": matthews_corrcoef(y_test, pred),
           "AUPRC (macro)": float(np.mean(ap)),
           "AUC (macro OvR)": roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=list(range(k)))}
    for name, r in zip(class_names, recalls):
        out[f"Recall: {name}"] = r
    return out


def build_models(groups, cfg, features=None):
    if features is not None:
        groups = {g: [c for c in cols if c in features] for g, cols in groups.items()}
    xgb = Pipeline([("preprocessor", train_models.build_preprocessor(groups, cfg, scale_numeric=False)),
                    ("classifier", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                                 tree_method="hist", random_state=cfg.RANDOM_STATE, n_jobs=-1))])
    return {"Logistic Regression": train_models.build_logistic_regression(groups, cfg),
            "Random Forest": train_models.build_random_forest(groups, cfg),
            "XGBoost": xgb}


def fit_and_score(models, X_tr, y_tr, X_te, y_te, class_names):
    rows = {}
    for name, pipe in models.items():
        if name == "XGBoost":   # XGBoost has no class_weight option -> balanced sample weights
            pipe.fit(X_tr, y_tr, classifier__sample_weight=compute_sample_weight("balanced", y_tr))
        else:
            pipe.fit(X_tr, y_tr)
        rows[name] = metrics_for(pipe, X_te, y_te, class_names)
    return rows


def main():
    target = config.TARGET_COLUMN
    csv_path = data_loader.find_csv_file(config.DATA_DIR, config.CSV_FILE_NAME)
    raw = data_loader.load_csv(csv_path)
    raw.columns = [str(c).strip() for c in raw.columns]
    if target not in raw.columns:
        raise SystemExit("Set TARGET_COLUMN in config.py first.")
    cfg = make_cfg(config.EXCLUDE_COLUMNS)
    X, y, groups, class_names, removed = prepare(raw, cfg, target)
    num = groups["numeric"]
    print(f"Dataset {csv_path.name}: {len(X)} rows, {len(num)} numeric features, classes={class_names}")

    # ---------- A. class statistics (paper Section II.D)
    cs = class_statistics(y, class_names)
    pd.DataFrame({"Class": class_names, "Count": [cs["counts"][c] for c in class_names],
                  "Percent": [100 * cs["counts"][c] / len(y) for c in class_names]}
                 ).to_csv(OUT / "class_stats.csv", index=False)

    # ---------- B. descriptive statistics + Kruskal-Wallis (paper III.B, III.D)
    skew = X[num].skew()
    kurt = X[num].kurt()
    kw_rows = []
    for c in num:
        groups_c = [X.loc[y == k, c].dropna().to_numpy() for k in range(len(class_names))]
        try:
            h, p = stats.kruskal(*groups_c)
        except ValueError:
            h, p = np.nan, np.nan
        kw_rows.append({"Feature": c, "KW H": h, "KW p": p, "Skewness": skew[c], "Excess kurtosis": kurt[c]})
    kw = pd.DataFrame(kw_rows)
    kw.to_csv(OUT / "descriptive_kruskal.csv", index=False)
    desc = {"n_numeric": len(num), "pct_abs_skew_gt1": 100 * (skew.abs() > 1).mean(),
            "kw_p_ge_005": int((kw["KW p"] >= 0.05).sum()), "kw_p_lt_0001": int((kw["KW p"] < 0.001).sum())}

    # ---------- C. ANOVA and MI on the whole sample (paper III.C)
    rel = relevance_scores(X, y, num)
    rel.to_csv(OUT / "anova_mi_scores.csv", index=False)
    top_mi = rel.sort_values("MI", ascending=False).head(20)["Feature"].tolist()
    top_f = rel.sort_values("ANOVA F", ascending=False).head(20)["Feature"].tolist()
    overlap = {"ours MI∩ANOVA": len(set(top_mi) & set(top_f)),
               "paper MI∩ANOVA": len(set(PAPER_MI) & set(PAPER_ANOVA)),
               "MI ours∩paper": len(set(top_mi) & set(PAPER_MI)),
               "ANOVA ours∩paper": len(set(top_f) & set(PAPER_ANOVA))}
    rank_mi = {f: i + 1 for i, f in enumerate(rel.sort_values("MI", ascending=False)["Feature"])}
    rank_f = {f: i + 1 for i, f in enumerate(rel.sort_values("ANOVA F", ascending=False)["Feature"])}
    cmp_rows = []
    for i, f in enumerate(PAPER_MI):
        cmp_rows.append({"List": "MI", "Paper rank": i + 1, "Feature": f, "Our rank": rank_mi.get(f, "n/a (removed/constant)")})
    for i, f in enumerate(PAPER_ANOVA):
        cmp_rows.append({"List": "ANOVA", "Paper rank": i + 1, "Feature": f, "Our rank": rank_f.get(f, "n/a (removed/constant)")})
    pd.DataFrame(cmp_rows).to_csv(OUT / "feature_rank_comparison.csv", index=False)
    scale = {"our max MI": rel["MI"].max(), "our max ANOVA F": rel["ANOVA F"].max(),
             "our median ANOVA F": rel["ANOVA F"].median(), "ours n(ANOVA p<0.001)": int((rel["ANOVA p"] < 0.001).sum())}

    # ---------- D. models, same split as the project (seed 42), all features
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    main_rows = fit_and_score(build_models(groups, cfg), X_tr, y_tr, X_te, y_te, class_names)
    main_df = pd.DataFrame(main_rows).T
    main_df.to_csv(OUT / "models_all_features.csv")

    # ---------- E. repeated splits (mean/std over 5 seeds)
    rep = []
    for seed in SEEDS:
        a, b, c, d = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        for name, r in fit_and_score(build_models(groups, cfg), a, c, b, d, class_names).items():
            rep.append({"Model": name, "Seed": seed, **r})
    rep_df = pd.DataFrame(rep)
    rep_df.to_csv(OUT / "models_repeated_splits_raw.csv", index=False)
    summary = rep_df.groupby("Model")[["Accuracy", "Macro F1", "MCC", "AUPRC (macro)"]].agg(["mean", "std"]).round(4)
    summary.to_csv(OUT / "models_repeated_splits_summary.csv")

    # ---------- F. paper's two-stage idea: compact MI∩ANOVA core (chosen on TRAIN only)
    rel_tr = relevance_scores(X_tr, y_tr, num)
    k = 20
    core = list(set(rel_tr.nlargest(k, "MI")["Feature"]) & set(rel_tr.nlargest(k, "ANOVA F")["Feature"]))
    if len(core) < 5:   # intersection too small -> use best average rank
        rel_tr["r"] = rel_tr["MI"].rank(ascending=False) + rel_tr["ANOVA F"].rank(ascending=False)
        core = rel_tr.nsmallest(10, "r")["Feature"].tolist()
        core_note = "MI∩ANOVA top-20 had <5 features; used the 10 best average-rank features"
    else:
        core_note = f"intersection of top-{k} MI and top-{k} ANOVA (train split only)"
    core_rows = fit_and_score(build_models(groups, cfg, features=set(core)), X_tr, y_tr, X_te, y_te, class_names)
    core_df = pd.DataFrame(core_rows).T
    core_df.to_csv(OUT / "models_core_features.csv")
    pd.Series(core, name="core_feature").to_csv(OUT / "core_features.csv", index=False)

    # ---------- G. what happens if identifier / time columns are NOT excluded (leakage demo)
    leaky_cfg = make_cfg([])
    XL, yL, gL, cnL, _ = prepare(raw, leaky_cfg, target)
    a, b, c, d = train_test_split(XL, yL, test_size=0.2, random_state=42, stratify=yL)
    leaky_rows = fit_and_score(build_models(gL, leaky_cfg), a, c, b, d, cnL)
    leaky_df = pd.DataFrame(leaky_rows).T
    leaky_df.to_csv(OUT / "models_with_identifier_columns_LEAKY.csv")

    # ---------- plots
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ["Macro F1", "MCC", "AUPRC (macro)", "Accuracy"]
    w = 0.8 / len(main_df)
    for i, (name, row) in enumerate(main_df.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(np.arange(len(metrics)) + i * w, vals, w, label=name)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(metrics)) + w)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_title("Test-set performance (10,000-flow sample, identifiers/time excluded)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_model_comparison.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax_, col, ttl in ((axes[0], "MI", "Mutual Information (top 15)"), (axes[1], "ANOVA F", "ANOVA F-score (top 15)")):
        d_ = rel.sort_values(col, ascending=False).head(15).iloc[::-1]
        ax_.barh(d_["Feature"], d_[col], color="steelblue" if col == "MI" else "darkmagenta")
        ax_.set_title(ttl)
        if col == "ANOVA F":
            ax_.set_xscale("log")
    fig.tight_layout(); fig.savefig(OUT / "fig_top_features.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = list(main_df.index)
    x = np.arange(len(names))
    ax.bar(x - 0.2, [main_df.loc[n, "Macro F1"] for n in names], 0.4, label="Identifiers/time excluded")
    ax.bar(x + 0.2, [leaky_df.loc[n, "Macro F1"] for n in names], 0.4, label="Identifiers/time INCLUDED (leaky)")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.1); ax.set_ylabel("Macro F1")
    ax.set_title("Effect of capture-time / address columns on Macro F1"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_leakage_effect.png", dpi=170); plt.close(fig)

    # ---------- text summary
    pd.set_option("display.width", 200)
    lines = ["PAPER COMPARISON SUMMARY", "=" * 70, f"Rows: {len(X)}   numeric features used: {len(num)}",
             f"Excluded columns (config): {config.EXCLUDE_COLUMNS}", "",
             f"Class counts: {cs['counts']}", f"IR={cs['IR']:.2f}  Gini={cs['Gini']:.3f}  Entropy(ln)={cs['Entropy (ln)']:.3f}", "",
             f"Descriptive: {desc}", f"Relevance scale: {scale}", f"Top-20 overlaps: {overlap}", "",
             "Our top-10 MI:   " + str(top_mi[:10]), "Our top-10 ANOVA: " + str(top_f[:10]), "",
             "MODELS - all features (seed 42)", main_df.round(4).T.to_string(), "",
             "MODELS - repeated splits (mean/std over seeds 42-46)", summary.to_string(), "",
             f"MODELS - compact core ({core_note}); n={len(core)}: {core}", core_df.round(4).T.to_string(), "",
             "MODELS - identifier/time columns INCLUDED (LEAKY)", leaky_df.round(4).T.to_string(), ""]
    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
