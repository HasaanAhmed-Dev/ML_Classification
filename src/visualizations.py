"""All plots. Uses the non-interactive 'Agg' backend so it works anywhere."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from .utils import shorten  # noqa: E402

MAX_CLASSES_ANNOTATED = 15


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _text_figure(message, path, title):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_title(title)
    ax.axis("off")
    _save(fig, path)


def plot_class_distribution(class_df, path, max_shown=30):
    """Bar chart of class counts with percentages."""
    data = class_df.head(max_shown)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(data) + 2), 5))
    bars = ax.bar([shorten(c) for c in data["Class"]], data["Count"], color="steelblue")
    for bar, pct in zip(bars, data["Percentage"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("Target class distribution" + (f" (top {max_shown})" if len(class_df) > max_shown else ""))
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of rows")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, path)


def plot_missing_values(missing_df, path, max_shown=30):
    """Horizontal bars with the percentage of missing values (raw data)."""
    data = missing_df[missing_df["Missing Count"] > 0].head(max_shown)
    if data.empty:
        _text_figure("No missing values in the dataset", path, "Missing values")
        return
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(data) + 1.5)))
    ax.barh([shorten(f, 30) for f in data["Feature"]][::-1], data["Missing Percentage"][::-1], color="indianred")
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Missing values per feature (raw data)")
    _save(fig, path)


def plot_confusion_matrix(cm, class_names, model_name, path):
    """Counts (left) and row-normalised proportions (right)."""
    k = len(class_names)
    labels = [shorten(c) for c in class_names]
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums != 0)
    side = min(max(5, 0.55 * k + 3), 16)
    fig, axes = plt.subplots(1, 2, figsize=(2 * side, side * 0.9))
    annotate = k <= MAX_CLASSES_ANNOTATED
    sns.heatmap(cm, annot=annotate, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
                ax=axes[0], cbar=False)
    axes[0].set_title(f"{model_name} - confusion matrix (counts)")
    sns.heatmap(normalized, annot=annotate, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, ax=axes[1], cbar=False)
    axes[1].set_title(f"{model_name} - normalised by true class")
    for ax in axes:
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
    _save(fig, path)


def plot_roc_curve(roc, model_name, path, max_classes_shown=10):
    """ROC curve: single curve for binary, one-vs-rest curves + macro average for multiclass."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    drawn = False
    if roc["type"] == "binary":
        if roc["curves"]:
            name, (fpr, tpr, auc_value) = next(iter(roc["curves"].items()))
            ax.plot(fpr, tpr, linewidth=2, label=f"Positive class '{shorten(name)}' (AUC = {auc_value:.3f})")
            drawn = True
    else:
        if roc["macro_curve"] is not None:
            fpr, tpr = roc["macro_curve"]
            ax.plot(fpr, tpr, linewidth=3, color="black",
                    label=f"Macro-average OvR (AUC = {roc['macro_auc']:.3f})")
            drawn = True
        shown = sorted(roc["curves"].items(), key=lambda kv: -roc["support"].get(kv[0], 0))[:max_classes_shown]
        for name, (fpr, tpr, auc_value) in shown:
            ax.plot(fpr, tpr, linewidth=1, alpha=0.8, label=f"{shorten(name)} (AUC = {auc_value:.3f})")
            drawn = True
        if len(roc["curves"]) > max_classes_shown:
            ax.text(0.98, 0.02, f"Showing {max_classes_shown} largest of {len(roc['curves'])} classes",
                    ha="right", transform=ax.transAxes, fontsize=8)
    if not drawn:
        ax.text(0.5, 0.5, "ROC not available\n(test set lacks both outcomes)", ha="center", transform=ax.transAxes)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{model_name} - ROC curve" + (" (one-vs-rest)" if roc["type"] == "multiclass" else ""))
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def plot_feature_importance(importance_df, title, xlabel, path, top_n):
    """Horizontal bar chart of the top-N features."""
    data = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(data) + 1.5)))
    ax.barh([shorten(f, 40) for f in data["Feature"]], data["Importance"], color="teal")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    _save(fig, path)


def plot_model_comparison(comparison_df, path):
    """Left: key test metrics. Right: train vs test accuracy and weighted F1."""
    models = comparison_df["Model"].tolist()
    metrics = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "AUC"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        vals = comparison_df.loc[comparison_df["Model"] == model, metrics].iloc[0].astype(float).to_numpy()
        bars = axes[0].bar(x + i * width, np.nan_to_num(vals), width, label=model)
        for bar, v in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", fontsize=8)
    axes[0].set_xticks(x + width * (len(models) - 1) / 2)
    axes[0].set_xticklabels(metrics, rotation=20)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Test-set metrics (macro metrics matter for imbalanced data)")
    axes[0].legend()

    groups = ["Train Accuracy", "Test Accuracy", "Train F1", "Test F1"]
    x = np.arange(len(groups))
    for i, model in enumerate(models):
        vals = comparison_df.loc[comparison_df["Model"] == model, groups].iloc[0].astype(float).to_numpy()
        bars = axes[1].bar(x + i * width, vals, width, label=model)
        for bar, v in zip(bars, vals):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         f"{v:.2f}", ha="center", fontsize=8)
    axes[1].set_xticks(x + width * (len(models) - 1) / 2)
    axes[1].set_xticklabels(groups, rotation=20)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Train vs test (F1 = weighted F1)")
    axes[1].legend()
    _save(fig, path)


def plot_per_class_f1(per_class_df, path, max_shown=30):
    """Per-class F1 for both models - shows which classes are hard."""
    data = per_class_df.head(max_shown)
    models = [c for c in data.columns if c != "Class"]
    y = np.arange(len(data))
    height = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(data) + 1.5)))
    for i, model in enumerate(models):
        ax.barh(y + i * height, data[model], height, label=model)
    ax.set_yticks(y + height * (len(models) - 1) / 2)
    ax.set_yticklabels([shorten(c, 30) for c in data["Class"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-score on the test set")
    ax.set_title("Per-class F1 (classes ordered by size)")
    ax.legend(loc="lower right")
    _save(fig, path)
