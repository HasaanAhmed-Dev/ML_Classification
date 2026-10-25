"""
reproduce_paper.py - tries to reproduce the paper's sample statistics.

Needs the FULL MedSec-25 CSV (about 550,000 rows) in the folder  full_data/
(NOT in data/, because main.py expects only one CSV in data/).

Run:  python reproduce_paper.py
Output: results/paper_comparison/reproduction_attempt.csv  + a printed table.

It draws the 10,000-flow sample in several different ways and measures the numbers
the paper reports (class mix, imbalance ratio, max Mutual Information, max ANOVA F),
so you can see which sampling (if any) reproduces the paper.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "paper_comparison"
OUT.mkdir(parents=True, exist_ok=True)

LABEL = "Label"
IDENTIFIERS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]   # not features
PAPER_COUNTS = {"Reconnaissance": 5092, "Initial access": 2013, "Benign": 1457,
                "Lateral movement": 971, "Exfiltration": 467}


def find_full_csv():
    files = sorted(p for p in (ROOT / "full_data").glob("*.csv"))
    if not files:
        raise SystemExit("Put the full MedSec-25 CSV in the 'full_data' folder first.")
    return files[0]


def relevance(sample, shuffle=False):
    """(number of features, max MI, max ANOVA F, median ANOVA F)."""
    y = sample[LABEL].astype("category").cat.codes.to_numpy()
    if shuffle:                                   # break the feature-label link on purpose
        y = np.random.RandomState(0).permutation(y)
    X = sample.drop(columns=[c for c in IDENTIFIERS + [LABEL] if c in sample.columns])
    X = X.select_dtypes("number")
    X = X.loc[:, X.nunique() > 1]
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    f, _ = f_classif(X, y)
    mi = mutual_info_classif(X, y, random_state=42)
    return len(X.columns), float(mi.max()), float(np.nanmax(f)), float(np.nanmedian(f))


def draw(clean, counts, seed=42):
    parts = [clean[clean[LABEL] == k].sample(n, random_state=seed) for k, n in counts.items()]
    return pd.concat(parts)


def main():
    path = find_full_csv()
    print("Loading", path.name, "(this can take a minute) ...")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Rows: {len(df)} -> {len(clean)} after removing missing/infinite values")

    order = (clean[LABEL] != clean[LABEL].shift()).cumsum()
    print("\nLabel order inside the file (rows per block):")
    print(clean.groupby(order)[LABEL].agg(["first", "size"]).to_string(index=False))

    class_counts = clean[LABEL].value_counts()
    proportional = {k: int(round(10000 * v / len(clean))) for k, v in class_counts.items()}
    samples = {
        "A proportional stratified 10k": (draw(clean, proportional), False),
        "B plain random 10k": (clean.sample(10000, random_state=42), False),
        "C paper's exact class counts": (draw(clean, PAPER_COUNTS), False),
        "D equal per class (2000 each)": (draw(clean, {k: 2000 for k in class_counts.index}), False),
        "E plain random 10k, SHUFFLED labels": (clean.sample(10000, random_state=42), True),
    }
    rows = []
    for name, (sample, shuffled) in samples.items():
        vc = sample[LABEL].value_counts()
        n_feat, max_mi, max_f, med_f = relevance(sample, shuffled)
        rows.append({"Sampling": name, "Rows": len(sample),
                     "Imbalance ratio": round(vc.max() / vc.min(), 1),
                     **{f"% {k}": round(100 * vc.get(k, 0) / len(sample), 1) for k in PAPER_COUNTS},
                     "Features": n_feat, "Max MI": round(max_mi, 4),
                     "Max ANOVA F": round(max_f, 1), "Median ANOVA F": round(med_f, 2)})
    result = pd.DataFrame(rows)
    paper_row = {"Sampling": "PAPER (reported)", "Rows": 10000, "Imbalance ratio": 10.9,
                 **{f"% {k}": round(100 * v / 10000, 1) for k, v in PAPER_COUNTS.items()},
                 "Features": "about 40", "Max MI": "about 0.008", "Max ANOVA F": "about 1-2",
                 "Median ANOVA F": "n/a"}
    result = pd.concat([result, pd.DataFrame([paper_row])], ignore_index=True)
    result.to_csv(OUT / "reproduction_attempt.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\nREPRODUCTION ATTEMPT\n" + result.to_string(index=False))
    print("\nSaved:", OUT / "reproduction_attempt.csv")


if __name__ == "__main__":
    main()
