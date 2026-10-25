"""Choosing the target (label) column."""
import numpy as np
import pandas as pd
from pandas.api import types as pdt

from .utils import ProjectError, split_name_tokens

# Words that often appear in the name of a label column.
TARGET_NAME_HINTS = {
    "target", "label", "class", "outcome", "status", "result", "diagnosis",
    "species", "category", "churn", "default", "survived", "response", "y",
    "grade", "level", "decision", "verdict", "output",
}
# Words that make a column a poor target candidate.
NON_TARGET_TOKENS = {"id", "uuid", "guid", "index", "idx", "date", "time", "timestamp", "datetime"}


def _normalized_entropy(counts):
    """Class balance in [0, 1]: 1 = perfectly balanced, 0 = single class."""
    probs = counts / counts.sum()
    if len(probs) < 2:
        return 0.0
    return float(-(probs * np.log(probs)).sum() / np.log(len(probs)))


def score_candidate(series, position, n_columns, max_classes):
    """
    Score one column as a possible classification target.
    Returns (score, reasons) or (None, reason_for_rejection).

    Scoring rule (documented in README):
      +1.5 text/boolean column, +1.0 integer column with few values
      +1.0 two classes, +0.7 for 3-10 classes, +0.2 for more
      +2.0 name contains a target-like word (target, label, class, ...)
      +2.0 column is the last column of the file
      +1.0 * class balance (normalized entropy)
      -0.5 if the smallest class has fewer than 2 rows
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return None, "empty column"
    n_unique = non_null.nunique()
    if n_unique < 2:
        return None, "only one value"
    if n_unique > max_classes:
        return None, f"more than {max_classes} unique values"
    if n_unique > 0.5 * len(non_null):
        return None, "too many unique values compared to rows"

    tokens = split_name_tokens(series.name)
    if NON_TARGET_TOKENS.intersection(tokens):
        return None, "name looks like an identifier or date"

    reasons = []
    score = 0.0
    if pdt.is_bool_dtype(series) or not (pdt.is_numeric_dtype(series) or pdt.is_datetime64_any_dtype(series)):
        score += 1.5
        reasons.append("text/boolean values (+1.5)")
    elif pdt.is_integer_dtype(series) or (pdt.is_float_dtype(series) and np.all(np.mod(non_null, 1) == 0)):
        score += 1.0
        reasons.append("integer-coded values (+1.0)")
    else:
        return None, "continuous numeric values"

    if n_unique == 2:
        score += 1.0
        reasons.append("binary (+1.0)")
    elif n_unique <= 10:
        score += 0.7
        reasons.append(f"{n_unique} classes (+0.7)")
    else:
        score += 0.2
        reasons.append(f"{n_unique} classes (+0.2)")

    if NON_TARGET_TOKENS.isdisjoint(tokens) and TARGET_NAME_HINTS.intersection(tokens):
        score += 2.0
        reasons.append("target-like name (+2.0)")
    if position == n_columns - 1:
        score += 2.0
        reasons.append("last column (+2.0)")

    counts = non_null.value_counts().to_numpy(dtype=float)
    balance = _normalized_entropy(counts)
    score += balance
    reasons.append(f"balance {balance:.2f} (+{balance:.2f})")
    if counts.min() < 2:
        score -= 0.5
        reasons.append("class with <2 rows (-0.5)")
    return score, reasons


def detect_target(df, configured_target, max_classes, log):
    """
    Decide which column is the target.
    1) If configured_target is set, use it.
    2) Otherwise score every column and pick the best candidate (highest score;
       ties go to the column that appears later in the file).
    """
    if configured_target:
        wanted = str(configured_target).strip()
        if wanted in df.columns:
            chosen = wanted
        else:
            lowered = {str(c).strip().lower(): c for c in df.columns}
            if wanted.lower() in lowered:
                chosen = lowered[wanted.lower()]
            else:
                raise ProjectError(
                    f"TARGET_COLUMN='{configured_target}' is not a column of the dataset.\n"
                    f"Available columns: {list(df.columns)}"
                )
        series = df[chosen].dropna()
        if pdt.is_float_dtype(series) and series.nunique() > 0.5 * max(len(series), 1) and series.nunique() > max_classes:
            raise ProjectError(
                f"Target '{chosen}' looks continuous ({series.nunique()} distinct values). "
                "This project is for classification targets."
            )
        log.add("Target", f"Using target column from config.py: '{chosen}'")
        return chosen, pd.DataFrame()

    rows = []
    n_columns = df.shape[1]
    for position, column in enumerate(df.columns):
        score, reasons = score_candidate(df[column], position, n_columns, max_classes)
        if score is not None:
            rows.append({
                "Column": column, "Score": round(score, 3), "Position": position,
                "Unique Values": int(df[column].nunique()), "Reasons": "; ".join(reasons),
            })
    if not rows:
        raise ProjectError(
            "No suitable classification target could be detected automatically.\n"
            f"Set TARGET_COLUMN in config.py. Available columns: {list(df.columns)}"
        )
    candidates = pd.DataFrame(rows).sort_values(["Score", "Position"], ascending=[False, False]).reset_index(drop=True)
    chosen = candidates.loc[0, "Column"]

    print("  Target candidates (best first):")
    for _, row in candidates.head(5).iterrows():
        print(f"    {row['Column']}  score={row['Score']}  ({row['Unique Values']} classes)")
    log.add("Target", f"Auto-detected target column: '{chosen}' "
                      f"(highest score {candidates.loc[0, 'Score']}). "
                      "If this is wrong, set TARGET_COLUMN in config.py.")
    return chosen, candidates
