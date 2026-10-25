"""Loading the CSV file and producing the raw dataset inspection."""
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

from .utils import ProjectError, save_text


def find_csv_file(data_dir, file_name=None):
    """Return the CSV path to use (the only CSV in data/, or the configured one)."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ProjectError(f"Data folder not found: {data_dir}")

    if file_name:
        path = data_dir / file_name
        if not path.exists():
            raise ProjectError(f"CSV_FILE_NAME='{file_name}' was not found in {data_dir}")
        return path

    csv_files = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if not csv_files:
        raise ProjectError(
            f"No CSV file found in: {data_dir}\n"
            "Copy your dataset (.csv) into the 'data' folder and run again."
        )
    if len(csv_files) > 1:
        names = ", ".join(p.name for p in csv_files)
        raise ProjectError(
            f"Several CSV files found in {data_dir}: {names}\n"
            "Keep only one, or set CSV_FILE_NAME in config.py."
        )
    return csv_files[0]


def load_csv(path):
    """Read a CSV file, trying common encodings and delimiters."""
    path = Path(path)
    df = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
            # A single column often means a different delimiter (; or tab).
            if df.shape[1] == 1:
                try:
                    sniffed = pd.read_csv(path, sep=None, engine="python", encoding=encoding)
                    if sniffed.shape[1] > 1:
                        df = sniffed
                except Exception:
                    pass
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ProjectError(f"The CSV file is empty: {path}")
        except pd.errors.ParserError as exc:
            raise ProjectError(f"Could not parse the CSV file: {exc}")
    if df is None:
        raise ProjectError(f"Could not read {path} with any supported encoding.")
    if df.shape[0] == 0 or df.shape[1] < 2:
        raise ProjectError(
            f"The CSV needs at least 2 columns and 1 row (found shape {df.shape})."
        )
    return df


def inspect_dataset(df, file_name):
    """Collect raw dataset statistics (before any cleaning) and print a short report."""
    n_rows = len(df)
    missing_count = df.isna().sum()
    missing_df = pd.DataFrame({
        "Feature": df.columns,
        "Missing Count": missing_count.values,
        "Missing Percentage": (missing_count.values / n_rows * 100).round(3),
    }).sort_values("Missing Count", ascending=False).reset_index(drop=True)

    unique_counts = df.nunique(dropna=True)
    columns_table = pd.DataFrame({
        "Feature": df.columns,
        "Dtype": [str(t) for t in df.dtypes],
        "Unique Values": unique_counts.values,
        "Missing": missing_count.values,
        "Missing %": (missing_count.values / n_rows * 100).round(2),
    })

    numeric = df.select_dtypes(include=[np.number])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # e.g. columns holding inf
        numeric_summary = (
            numeric.describe().T.round(4).to_string() if numeric.shape[1] > 0
            else "(no numerical columns)"
        )

    info = {
        "file_name": file_name,
        "shape": df.shape,
        "duplicates": int(df.duplicated().sum()),
        "missing_df": missing_df,
        "columns_table": columns_table,
        "numeric_summary": numeric_summary,
    }

    # ---- terminal display (limited so very wide datasets stay readable) ----
    print(f"  File name      : {file_name}")
    print(f"  Shape          : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Duplicate rows : {info['duplicates']}")
    n_missing_cols = int((missing_count > 0).sum())
    print(f"  Columns with missing values: {n_missing_cols}")
    shown = columns_table.head(40)
    print("  Columns (name | dtype | unique | missing%):")
    for _, row in shown.iterrows():
        print(f"    - {row['Feature']} | {row['Dtype']} | {row['Unique Values']} | {row['Missing %']}")
    if len(columns_table) > len(shown):
        print(f"    ... and {len(columns_table) - len(shown)} more (see results/dataset_summary.txt)")
    return info


def save_dataset_summary(results_dir, info, target, class_df, extra_notes=None):
    """Write results/dataset_summary.txt and results/missing_values.csv."""
    results_dir = Path(results_dir)
    info["missing_df"].to_csv(results_dir / "missing_values.csv", index=False)

    lines = [
        "DATASET SUMMARY (raw data, before cleaning)",
        "=" * 60,
        f"File name      : {info['file_name']}",
        f"Shape          : {info['shape'][0]} rows x {info['shape'][1]} columns",
        f"Duplicate rows : {info['duplicates']}",
        "",
        "COLUMNS (name, data type, unique values, missing values)",
        "-" * 60,
        info["columns_table"].to_string(index=False),
        "",
        "MISSING VALUES (sorted, only columns with missing values)",
        "-" * 60,
    ]
    with_missing = info["missing_df"][info["missing_df"]["Missing Count"] > 0]
    lines.append(with_missing.to_string(index=False) if len(with_missing) else "No missing values.")
    lines += ["", "NUMERICAL SUMMARY", "-" * 60, info["numeric_summary"], ""]
    lines += [f"TARGET COLUMN: {target}", "-" * 60, "Class distribution (after cleaning):",
              class_df.to_string(index=False), ""]
    if extra_notes:
        lines += ["NOTES", "-" * 60] + list(extra_notes) + [""]
    save_text(results_dir / "dataset_summary.txt", "\n".join(lines))
