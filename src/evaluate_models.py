"""Metrics, ROC/AUC, feature importance, overfitting analysis, comparison and reports."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_curve)

from .utils import banner, fmt


# ------------------------------------------------------------------ metrics
def _labels_present(y_true, y_pred):
    return np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))


def compute_metrics(y_true, y_pred):
    """Accuracy + macro/weighted precision, recall, F1 (zero_division=0).
    'Precision', 'Recall' and 'F1' are the WEIGHTED averages (documented in README)."""
    labels = _labels_present(y_true, y_pred)
    out = {"Accuracy": accuracy_score(y_true, y_pred)}
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=average, zero_division=0)
        out[f"{average.capitalize()} Precision"] = p
        out[f"{average.capitalize()} Recall"] = r
        out[f"{average.capitalize()} F1"] = f
    out["Precision"], out["Recall"], out["F1"] = (
        out["Weighted Precision"], out["Weighted Recall"], out["Weighted F1"])
    return out


def compute_roc(y_true, proba, model_classes, class_names):
    """
    ROC/AUC from predict_proba.
    - 2 classes  -> one binary ROC curve (positive class = second class).
    - >2 classes -> one-vs-rest curve per class + macro-average (classes missing from the
      test set are skipped because AUC is undefined for them).
    """
    y_true = np.asarray(y_true)
    names = [class_names[int(c)] for c in model_classes]
    result = {"type": "binary" if len(model_classes) == 2 else "multiclass",
              "curves": {}, "per_class_auc": {}, "support": {},
              "macro_auc": np.nan, "macro_curve": None}

    def one_vs_rest(column, cls_label):
        positives = (y_true == cls_label).astype(int)
        result["support"][names[column]] = int(positives.sum())
        if positives.sum() == 0 or positives.sum() == len(positives):
            result["per_class_auc"][names[column]] = np.nan
            return
        fpr, tpr, _ = roc_curve(positives, proba[:, column])
        area = auc(fpr, tpr)
        result["curves"][names[column]] = (fpr, tpr, area)
        result["per_class_auc"][names[column]] = area

    if result["type"] == "binary":
        one_vs_rest(1, model_classes[1])
        result["macro_auc"] = result["per_class_auc"].get(names[1], np.nan)
        return result

    for column, cls_label in enumerate(model_classes):
        one_vs_rest(column, cls_label)
    if result["curves"]:
        grid = np.unique(np.concatenate([c[0] for c in result["curves"].values()]))
        mean_tpr = np.mean([np.interp(grid, c[0], c[1]) for c in result["curves"].values()], axis=0)
        result["macro_curve"] = (grid, mean_tpr)
        result["macro_auc"] = float(np.nanmean(list(result["per_class_auc"].values())))
    return result


def evaluate_model(display_name, key, pipeline, X_train, y_train, X_test, y_test, class_names, train_seconds):
    """Evaluate one fitted pipeline on train and test data."""
    y_pred = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)
    labels = _labels_present(y_test, y_pred)

    report = classification_report(
        y_test, y_pred, labels=labels, target_names=[class_names[int(i)] for i in labels],
        zero_division=0)
    proba = pipeline.predict_proba(X_test)
    roc = compute_roc(y_test, proba, pipeline.named_steps["classifier"].classes_, class_names)

    return {
        "name": display_name, "key": key, "pipeline": pipeline,
        "metrics": compute_metrics(y_test, y_pred),
        "train_metrics": compute_metrics(y_train, y_pred_train),
        "y_pred": y_pred, "labels": labels,
        "confusion": confusion_matrix(y_test, y_pred, labels=labels),
        "confusion_names": [class_names[int(i)] for i in labels],
        "report": report, "roc": roc, "train_seconds": train_seconds,
    }


# -------------------------------------------------------------- feature names
def get_feature_names(pipeline):
    """Recover transformed feature names from the ColumnTransformer."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return [n.split("__", 1)[1] if "__" in n else n for n in names]


def random_forest_importance(pipeline):
    """Impurity-based importance, sorted descending."""
    values = pipeline.named_steps["classifier"].feature_importances_
    df = pd.DataFrame({"Feature": get_feature_names(pipeline), "Importance": values})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


def logistic_regression_importance(pipeline):
    """
    Coefficient-based importance = mean of |coefficient| over classes
    (binary: just |coefficient|). Numeric features are standardised, so their
    coefficients are comparable with each other; one-hot columns are 0/1.
    """
    coef = np.atleast_2d(pipeline.named_steps["classifier"].coef_)
    df = pd.DataFrame({"Feature": get_feature_names(pipeline), "Importance": np.abs(coef).mean(axis=0)})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


# ------------------------------------------------------ overfitting / comparison
def interpret_gap(accuracy_gap, f1_gap, test_accuracy):
    """Careful, rule-of-thumb wording. A train score above test is normal and not overfitting by itself."""
    gap = max(accuracy_gap, f1_gap)
    if gap < 0.03:
        text = "Relatively small train/test gap"
    elif gap < 0.08:
        text = "Moderate train/test gap"
    elif gap < 0.15:
        text = "Substantial train/test gap"
    else:
        text = "Large train/test gap - possible overfitting"
    if test_accuracy < 0.6 and gap < 0.03:
        text += " (both scores are modest, so the model may be limited by the features rather than overfitting)"
    return text


def build_overfitting_table(results):
    rows = []
    for r in results:
        tr_acc, te_acc = r["train_metrics"]["Accuracy"], r["metrics"]["Accuracy"]
        tr_f1, te_f1 = r["train_metrics"]["Weighted F1"], r["metrics"]["Weighted F1"]
        rows.append({
            "Model": r["name"], "Train Accuracy": tr_acc, "Test Accuracy": te_acc,
            "Accuracy Gap": tr_acc - te_acc, "Train F1": tr_f1, "Test F1": te_f1,
            "F1 Gap": tr_f1 - te_f1,
            "Interpretation": interpret_gap(tr_acc - te_acc, tr_f1 - te_f1, te_acc),
        })
    return pd.DataFrame(rows)


def build_comparison_table(results):
    rows = []
    for r in results:
        m, t = r["metrics"], r["train_metrics"]
        rows.append({
            "Model": r["name"], "Accuracy": m["Accuracy"],
            "Macro Precision": m["Macro Precision"], "Macro Recall": m["Macro Recall"],
            "Macro F1": m["Macro F1"], "Weighted Precision": m["Weighted Precision"],
            "Weighted Recall": m["Weighted Recall"], "Weighted F1": m["Weighted F1"],
            "AUC": r["roc"]["macro_auc"],
            "Train Accuracy": t["Accuracy"], "Test Accuracy": m["Accuracy"],
            "Accuracy Gap": t["Accuracy"] - m["Accuracy"],
            "Train F1": t["Weighted F1"], "Test F1": m["Weighted F1"],
            "F1 Gap": t["Weighted F1"] - m["Weighted F1"],
        })
    return pd.DataFrame(rows)


def build_auc_table(results):
    rows = []
    for r in results:
        kind = r["roc"]["type"]
        for cls, value in r["roc"]["per_class_auc"].items():
            rows.append({"Model": r["name"], "Class": cls,
                         "AUC Type": "binary (positive class)" if kind == "binary" else "one-vs-rest",
                         "AUC": value})
        if kind == "multiclass":
            rows.append({"Model": r["name"], "Class": "MACRO AVERAGE",
                         "AUC Type": "macro-average of one-vs-rest", "AUC": r["roc"]["macro_auc"]})
    return pd.DataFrame(rows)


def build_per_class_f1(results, class_names, class_df):
    """Per-class F1 for each model, ordered by class size."""
    table = pd.DataFrame({"Class": class_names})
    for r in results:
        labels = np.arange(len(class_names))
        # y_test is not stored; recompute from the confusion matrix (rows = true labels)
        cm = np.zeros((len(class_names), len(class_names)))
        for i, a in enumerate(r["labels"]):
            for j, b in enumerate(r["labels"]):
                cm[a, b] = r["confusion"][i, j]
        tp = np.diag(cm)
        precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(len(labels)), where=cm.sum(axis=0) > 0)
        recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(len(labels)), where=cm.sum(axis=1) > 0)
        denom = precision + recall
        table[r["name"]] = np.divide(2 * precision * recall, denom, out=np.zeros(len(labels)), where=denom > 0)
    order = {c: i for i, c in enumerate(class_df["Class"].tolist())}
    table["_o"] = table["Class"].map(order)
    return table.sort_values("_o").drop(columns="_o").reset_index(drop=True)


# ------------------------------------------------------------------- reports
def write_classification_report(path, result, target):
    m = result["metrics"]
    text = [f"CLASSIFICATION REPORT - {result['name']}", "=" * 60, f"Target: {target}", "",
            result["report"], "",
            f"Accuracy           : {fmt(m['Accuracy'])}",
            f"Macro Precision    : {fmt(m['Macro Precision'])}",
            f"Macro Recall       : {fmt(m['Macro Recall'])}",
            f"Macro F1           : {fmt(m['Macro F1'])}",
            f"Weighted Precision : {fmt(m['Weighted Precision'])}",
            f"Weighted Recall    : {fmt(m['Weighted Recall'])}",
            f"Weighted F1        : {fmt(m['Weighted F1'])}",
            f"AUC ({'binary' if result['roc']['type'] == 'binary' else 'macro one-vs-rest'}): "
            f"{fmt(result['roc']['macro_auc'])}", ""]
    path.write_text("\n".join(text), encoding="utf-8")


def print_terminal_report(ctx):
    """Print the summary shown at the end of main.py."""
    lr, rf = ctx["results"]

    def block(title, r):
        m = r["metrics"]
        print(f"\n{title}\n{'-' * 40}")
        print(f"Accuracy:      {fmt(m['Accuracy'])}")
        print(f"Precision:     {fmt(m['Precision'])}  (weighted)")
        print(f"Recall:        {fmt(m['Recall'])}  (weighted)")
        print(f"Weighted F1:   {fmt(m['Weighted F1'])}")
        print(f"Macro F1:      {fmt(m['Macro F1'])}")
        print(f"AUC:           {fmt(r['roc']['macro_auc'])}")

    print("\n" + banner("MACHINE LEARNING CLASSIFICATION PROJECT", "="))
    print(f"\nDataset:\n{ctx['file_name']}")
    print(f"\nTarget:\n{ctx['target']}")
    print(f"\nOriginal Shape:\n{ctx['original_shape'][0]} rows x {ctx['original_shape'][1]} columns")
    print(f"\nCleaned Shape:\n{ctx['cleaned_shape'][0]} rows x {ctx['cleaned_shape'][1]} columns "
          f"({ctx['n_features']} features + target)")
    print(f"\nNumber of Classes:\n{ctx['n_classes']}")
    print(f"\nTraining Samples:\n{ctx['n_train']}")
    print(f"\nTesting Samples:\n{ctx['n_test']}")
    block("LOGISTIC REGRESSION", lr)
    block("RANDOM FOREST", rf)
    print(f"\nOVERFITTING ANALYSIS\n{'-' * 40}")
    ov = ctx["overfitting"].set_index("Model")
    for tag, name in (("LR", lr["name"]), ("RF", rf["name"])):
        row = ov.loc[name]
        print(f"{tag} Train Accuracy: {fmt(row['Train Accuracy'])}")
        print(f"{tag} Test Accuracy:  {fmt(row['Test Accuracy'])}")
        print(f"{tag} Gap:            {fmt(row['Accuracy Gap'])}  -> {row['Interpretation']}")
    print("\n" + banner("PROJECT COMPLETED", "="))
    print("\nGenerated files:")
    print(f"  Results folder : {ctx['results_dir']}")
    print(f"  Models folder  : {ctx['models_dir']}")
    print(f"  Summary        : {ctx['results_dir'] / 'final_results.txt'}")
    print(f"  Comparison     : {ctx['results_dir'] / 'model_comparison.csv'}")


def build_final_results_text(ctx):
    """Create the text of results/final_results.txt (all numbers computed, none hard-coded)."""
    lr, rf = ctx["results"]
    comp = ctx["comparison"].set_index("Model")
    lines = [banner("FINAL RESULTS", "="), "",
             "DATASET", "-" * 60,
             f"File               : {ctx['file_name']}",
             f"Original shape     : {ctx['original_shape'][0]} rows x {ctx['original_shape'][1]} columns",
             f"Cleaned shape      : {ctx['cleaned_shape'][0]} rows x {ctx['cleaned_shape'][1]} columns",
             f"Target column      : {ctx['target']}",
             f"Number of classes  : {ctx['n_classes']}",
             f"Number of features : {ctx['n_features']} original columns "
             f"({ctx['n_transformed']} after encoding)",
             f"Feature groups     : " + ", ".join(f"{len(v)} {k}" for k, v in ctx["groups"].items()),
             f"Training samples   : {ctx['n_train']}",
             f"Testing samples    : {ctx['n_test']}",
             f"Split              : {'stratified' if ctx['stratified'] else 'NOT stratified'} "
             f"(test_size={ctx['config'].TEST_SIZE}, random_state={ctx['config'].RANDOM_STATE})",
             f"Problem type       : {'binary' if ctx['n_classes'] == 2 else 'multiclass'} classification",
             f"Class weighting    : {'balanced' if ctx['config'].USE_CLASS_WEIGHT else 'none'}", ""]

    for r in (lr, rf):
        m = r["metrics"]
        lines += [r["name"].upper(), "-" * 60,
                  f"Accuracy           : {fmt(m['Accuracy'])}",
                  f"Precision (weighted): {fmt(m['Precision'])}",
                  f"Recall (weighted)  : {fmt(m['Recall'])}",
                  f"Macro Precision    : {fmt(m['Macro Precision'])}",
                  f"Macro Recall       : {fmt(m['Macro Recall'])}",
                  f"Macro F1           : {fmt(m['Macro F1'])}",
                  f"Weighted Precision : {fmt(m['Weighted Precision'])}",
                  f"Weighted Recall    : {fmt(m['Weighted Recall'])}",
                  f"Weighted F1        : {fmt(m['Weighted F1'])}",
                  f"AUC ({'binary' if r['roc']['type'] == 'binary' else 'macro one-vs-rest'})"
                  f"      : {fmt(r['roc']['macro_auc'])}",
                  f"Training time      : {r['train_seconds']:.1f} s", ""]

    lines += ["MODEL COMPARISON (accuracy alone is not used to rank models)", "-" * 60]
    for metric in ("Macro F1", "Weighted F1", "Macro Recall", "Accuracy", "AUC"):
        a, b = comp.loc[lr["name"], metric], comp.loc[rf["name"], metric]
        if pd.isna(a) or pd.isna(b):
            lines.append(f"{metric:<13}: not comparable (AUC unavailable for one model)")
        elif abs(a - b) < 0.01:
            lines.append(f"{metric:<13}: essentially equal (difference {abs(a - b):.4f})")
        else:
            winner = lr["name"] if a > b else rf["name"]
            lines.append(f"{metric:<13}: {winner} is higher (difference {abs(a - b):.4f})")
    lines += ["", "Primary criterion: Macro F1 (treats every class equally, so it is not dominated by the "
                  "largest class), with Weighted F1, AUC and the train/test gap as supporting evidence.", ""]

    lines += ["OVERFITTING ANALYSIS", "-" * 60, ctx["overfitting"].round(4).to_string(index=False), "",
              "Note: training scores are usually a little higher than test scores. A gap alone does not "
              "prove overfitting; judge it together with the absolute test performance.", ""]

    for r, importance in zip((lr, rf), ctx["importances"]):
        lines += [f"TOP {min(10, len(importance))} FEATURES - {r['name']}", "-" * 60,
                  importance.head(10).round(5).to_string(index=False), ""]
    lines += ["Importance caveats: Random Forest importance is impurity-based and can favour features with many "
              "distinct values; Logistic Regression importance is the mean absolute coefficient over classes.", ""]

    if len(ctx["flagged"]):
        lines += ["LEAKAGE WARNING", "-" * 60,
                  f"{ctx['flagged']['Feature'].nunique()} feature(s) flagged as possibly encoding the target: "
                  f"{ctx['flagged']['Feature'].unique().tolist()}. See data_leakage_report.txt.", ""]

    lines += ["GENERATED FILES", "-" * 60]
    for folder in (ctx["results_dir"], ctx["models_dir"]):
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.name not in (".gitkeep", "final_results.txt"):
                lines.append(str(p))
    lines.append(str(ctx["results_dir"] / "final_results.txt"))
    return "\n".join(lines) + "\n"
