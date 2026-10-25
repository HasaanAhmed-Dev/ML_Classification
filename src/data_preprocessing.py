"""
Data cleaning, feature-type detection, class analysis, train/test split and
data-leakage checks.

IMPORTANT: everything in this file that LEARNS from data (imputers, scalers,
encoders) lives in train_models.py inside a scikit-learn Pipeline and is fitted
on the training split only. The steps here are row-wise or column-wise rules
that do not learn statistics from the test rows.
"""
import re
import warnings

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from .utils import ProjectError, save_text, split_name_tokens

# ---------------------------------------------------------------- constants
STRONG_ID_TOKENS = {"id", "uuid", "guid", "identifier", "idx", "index", "ref", "reference"}
WEAK_ID_TOKENS = {"no", "num", "number", "code", "key", "serial", "record", "row",
                  "order", "customer", "account", "client", "user", "member"}
ENTITY_WORDS = {"customer", "order", "account", "user", "client", "patient", "transaction",
                "member", "employee", "product", "session", "invoice", "record", "row",
                "item", "case", "ticket", "booking", "student"}
BOOLEAN_STRINGS = {"true", "false", "yes", "no", "y", "n", "t", "f"}
TRUE_STRINGS = {"true", "yes", "y", "t"}
MISSING_MARKERS = {"?"}
DATE_PATTERN = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}:\d{2}")
ID_PATTERN = re.compile(r"[A-Za-z]{0,6}[-_ ]?\d{3,}")


# ------------------------------------------------------------------ helpers
def kind_of(series):
    """Return 'bool', 'datetime', 'numeric' or 'text' for a pandas Series."""
    if pdt.is_bool_dtype(series):
        return "bool"
    if pdt.is_datetime64_any_dtype(series):
        return "datetime"
    if pdt.is_numeric_dtype(series):
        return "numeric"
    return "text"


def identifier_name_strength(name):
    """Return 'strong', 'weak' or None depending on how ID-like a column NAME is."""
    tokens = split_name_tokens(name)
    for token in tokens:
        if token in STRONG_ID_TOKENS:
            return "strong"
        if token.endswith("id") and token[:-2] in ENTITY_WORDS:
            return "strong"
    if any(token in WEAK_ID_TOKENS for token in tokens):
        return "weak"
    return None


def _clean_cell(value):
    """Strip whitespace from strings; blank strings and '?' become missing."""
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value in MISSING_MARKERS:
            return np.nan
    return value


# ------------------------------------------------- 1. normalise / convert types
def normalize_dataframe(df, log):
    """Tidy column names and text cells; turn infinite numbers into missing values."""
    df = df.copy()

    cleaned_names, seen = [], {}
    for col in df.columns:
        name = str(col).strip()
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        cleaned_names.append(name)
    df.columns = cleaned_names

    for col in df.columns:
        series = df[col]
        kind = kind_of(series)
        if kind == "text":
            df[col] = series.astype(object).map(_clean_cell)
        elif kind == "numeric":
            infinite = int(np.isinf(series.astype('float64')).sum())
            if infinite:
                df[col] = series.replace([np.inf, -np.inf], np.nan)
                log.add("Cleaning", f"Column '{col}': {infinite} infinite value(s) set to missing.")
    return df


def convert_numeric_like(df, target, log, min_fraction=0.9):
    """Text columns that are >=90% numbers become numeric (bad entries -> missing)."""
    for col in df.columns:
        if col == target or kind_of(df[col]) != "text":
            continue
        non_null = df[col].notna().sum()
        if non_null == 0:
            continue
        parsed = pd.to_numeric(df[col], errors="coerce")
        parsed = parsed.replace([np.inf, -np.inf], np.nan)
        ok = parsed.notna().sum()
        if ok / non_null >= min_fraction:
            invalid = int(non_null - ok)
            df[col] = parsed.astype(float)
            msg = f"Column '{col}' stored as text but numeric -> converted to numbers."
            if invalid:
                msg += f" {invalid} invalid entr{'y' if invalid == 1 else 'ies'} set to missing."
            log.add("Cleaning", msg)
    return df


def convert_booleans(df, target, log):
    """Convert boolean columns (real booleans or yes/no-style text) to 0/1 floats."""
    boolean_cols = []
    for col in df.columns:
        if col == target:
            continue
        kind = kind_of(df[col])
        if kind == "bool":
            df[col] = df[col].map({True: 1.0, False: 0.0}).astype(float)
            boolean_cols.append(col)
        elif kind == "text":
            values = {str(v).strip().lower() for v in df[col].dropna().unique()}
            if len(values) == 2 and values.issubset(BOOLEAN_STRINGS):
                df[col] = df[col].map(
                    lambda v: np.nan if pd.isna(v) else (1.0 if str(v).strip().lower() in TRUE_STRINGS else 0.0)
                ).astype(float)
                boolean_cols.append(col)
    for col in boolean_cols:
        log.add("Features", f"Boolean column '{col}' converted to 0/1.")
    return df, boolean_cols


def convert_datetimes(df, target, log):
    """Detect text columns that contain dates and convert them to datetime."""
    datetime_cols = []
    for col in df.columns:
        if col == target:
            continue
        kind = kind_of(df[col])
        if kind == "datetime":
            datetime_cols.append(col)
            continue
        if kind != "text":
            continue
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        sample = non_null.astype(str).head(200)
        # Cheap pre-check: values must look like dates (avoids parsing "1", "2", ...).
        if sample.map(lambda v: bool(DATE_PATTERN.search(v))).mean() < 0.95:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
            if parsed.notna().sum() / len(non_null) < 0.9:
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce", utc=True, format="mixed")
                except (TypeError, ValueError):
                    pass
        if parsed.notna().sum() / len(non_null) >= 0.9:
            df[col] = parsed
            datetime_cols.append(col)
            log.add("Features", f"Column '{col}' detected as datetime.")
    return df, datetime_cols


# ------------------------------------------------------ 2. identifier detection
def identify_identifier(series, name, id_unique_ratio):
    """
    Decide whether a column is identifier-like using MULTIPLE signals.
    Many unique values alone never causes removal.
    Returns (is_identifier, list_of_signals).
    """
    values = series.dropna()
    n = len(values)
    if n == 0:
        return False, []
    kind = kind_of(series)
    ratio = values.nunique() / n
    name_strength = identifier_name_strength(name)

    signals = []
    if name_strength == "strong":
        signals.append("ID-like column name")
    elif name_strength == "weak":
        signals.append("reference-like column name")
    if ratio >= id_unique_ratio:
        signals.append(f"unique ratio {ratio:.3f}")

    integer_valued = False
    sequential = False
    if kind == "numeric":
        arr = values.to_numpy(dtype=float)
        integer_valued = bool(np.all(np.mod(arr, 1) == 0))
        if integer_valued and ratio >= 0.99 and n > 1:
            uniq = np.sort(np.unique(arr))
            if len(uniq) > 1 and np.all(np.diff(uniq) == 1):
                sequential = True
                signals.append("sequential integers")

    pattern = False
    if kind == "text" and ratio >= 0.9:
        sample = values.astype(str).head(500)
        if sample.map(lambda v: bool(ID_PATTERN.fullmatch(v))).mean() >= 0.95:
            pattern = True
            signals.append("prefix+digits pattern")

    is_id = False
    if name_strength == "strong" and ratio >= 0.5:
        is_id = True
    elif sequential:
        is_id = True
    elif pattern:
        is_id = True
    elif kind == "text" and ratio >= id_unique_ratio and n >= 20:
        signals.append("nearly unique text values")
        is_id = True
    elif kind == "numeric" and integer_valued and ratio >= id_unique_ratio and name_strength == "weak":
        is_id = True
    return is_id, signals


# --------------------------------------------------------------- 3. cleaning
def clean_dataset(df, target, config, log):
    """
    General-purpose cleaning. Every removed column is recorded with a reason.
    Returns (clean_df, removed_records, protected_info).
    """
    removed = []

    def drop(column, reason):
        removed.append({"Feature": column, "Reason": reason})
        log.add("Cleaning", f"Removed column '{column}': {reason}")

    # --- empty columns
    for col in [c for c in df.columns if c != target and df[c].isna().all()]:
        drop(col, "Empty feature (all values missing)")
    df = df.drop(columns=[r["Feature"] for r in removed])
    if df[target].isna().all():
        raise ProjectError(f"The target column '{target}' has no values.")

    # --- empty rows
    before = len(df)
    df = df.dropna(how="all")
    if len(df) < before:
        log.add("Cleaning", f"Removed {before - len(df)} completely empty row(s).")

    # --- missing target
    before = len(df)
    df = df[df[target].notna()]
    if len(df) < before:
        log.add("Cleaning", f"Removed {before - len(df)} row(s) with a missing target value.")
    if len(df) < 10:
        raise ProjectError("Fewer than 10 usable rows remain after cleaning.")

    # --- user exclusions
    for col in config.EXCLUDE_COLUMNS:
        if col == target:
            log.add("Cleaning", f"Ignored EXCLUDE_COLUMNS entry '{col}' (it is the target).")
        elif col in df.columns:
            drop(col, "User-configured exclusion")
            df = df.drop(columns=[col])
        else:
            log.add("Cleaning", f"EXCLUDE_COLUMNS entry '{col}' not found in the data (ignored).")

    # --- constant columns
    for col in [c for c in df.columns if c != target]:
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 1:
            drop(col, "Constant feature" if n_unique == 1 else "Empty feature (no values left)")
            df = df.drop(columns=[col])

    # --- identifier-like columns
    kept_with_signals = []
    for col in [c for c in df.columns if c != target]:
        if kind_of(df[col]) == "datetime":
            continue
        is_id, signals = identify_identifier(df[col], col, config.ID_UNIQUE_RATIO)
        if is_id:
            drop(col, "Identifier-like feature (" + ", ".join(signals) + ")")
            df = df.drop(columns=[col])
        elif signals:
            kept_with_signals.append(col)

    if kept_with_signals:
        log.add("Identifiers", f"Kept {len(kept_with_signals)} column(s) that have only weak identifier "
                               f"signals (e.g. many unique values but no ID-like name/pattern): "
                               f"{kept_with_signals[:15]}{' ...' if len(kept_with_signals) > 15 else ''}")

    # --- excessive cardinality (text columns only)
    n_rows = len(df)
    for col in [c for c in df.columns if c != target and kind_of(df[c]) == "text"]:
        n_unique = df[col].nunique(dropna=True)
        ratio = n_unique / n_rows
        if n_unique > config.EXTREME_CARDINALITY_COUNT or (
            n_unique > config.HIGH_CARDINALITY_THRESHOLD and ratio > config.EXTREME_CARDINALITY_RATIO
        ):
            drop(col, f"Excessively high cardinality ({n_unique} unique values, {ratio:.1%} of rows)")
            df = df.drop(columns=[col])
        elif n_unique > config.HIGH_CARDINALITY_THRESHOLD:
            log.add("Features", f"'{col}' has {n_unique} categories -> one-hot encoding with rare "
                                f"categories grouped (max {config.MAX_ONEHOT_CATEGORIES} columns).")

    # --- duplicate rows (identifier columns are already gone, so the same record
    #     stored twice under different ids is also caught)
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        log.add("Cleaning", f"Removed {before - len(df)} duplicate row(s).")

    if df.shape[1] < 2:
        raise ProjectError("No usable feature columns remain after cleaning. Check removed_features.csv.")
    return df.reset_index(drop=True), removed


# ----------------------------------------------------------- 4. datetime features
def extract_datetime_features(df, datetime_cols, removed, log):
    """Replace datetime columns by year/month/day/day-of-week (/hour if present)."""
    new_features = []
    for col in [c for c in datetime_cols if c in df.columns]:
        dt = df[col].dt
        parts = {
            f"{col}_year": dt.year, f"{col}_month": dt.month,
            f"{col}_day": dt.day, f"{col}_dayofweek": dt.dayofweek,
        }
        if dt.hour.nunique(dropna=True) > 1:
            parts[f"{col}_hour"] = dt.hour
        kept = []
        for name, values in parts.items():
            values = values.astype("float64")
            if values.nunique(dropna=True) <= 1:
                removed.append({"Feature": name, "Reason": "Constant datetime component"})
                continue
            df[name] = values
            kept.append(name)
        df = df.drop(columns=[col])
        removed.append({"Feature": col, "Reason": "Datetime column replaced by extracted components"})
        log.add("Features", f"Datetime '{col}' -> {kept if kept else 'no varying components'}; original removed.")
        new_features += kept
    return df, new_features


# --------------------------------------------------- 5. feature type detection
def detect_feature_types(df, target, boolean_cols, config, log):
    """Group the feature columns into numeric / boolean / categorical groups."""
    groups = {"numeric": [], "boolean": [], "categorical": [], "categorical_high": []}
    for col in df.columns:
        if col == target:
            continue
        kind = kind_of(df[col])
        if col in boolean_cols or kind == "bool":
            groups["boolean"].append(col)
        elif kind == "numeric":
            groups["numeric"].append(col)
        else:
            # make categories uniform: strings only, real NaN for missing
            series = df[col].astype(object)
            df[col] = series.map(lambda v: v if pd.isna(v) else str(v))
            if df[col].nunique(dropna=True) > config.HIGH_CARDINALITY_THRESHOLD:
                groups["categorical_high"].append(col)
            else:
                groups["categorical"].append(col)
    log.add("Features", "Detected feature types: "
            + ", ".join(f"{len(v)} {k}" for k, v in groups.items()))
    if not any(groups.values()):
        raise ProjectError("No usable features were found.")
    return groups


# --------------------------------------------------------------- 6. the target
def prepare_target_labels(series):
    """Return target labels as a clean Series (integer-valued floats become ints)."""
    s = series.copy()
    if kind_of(s) == "numeric":
        arr = s.to_numpy(dtype=float)
        if np.all(np.mod(arr, 1) == 0):
            s = s.astype(int)
    elif kind_of(s) == "bool":
        s = s.astype(str)
    else:
        s = s.astype(str)
    return s


def analyze_classes(labels, rare_fraction):
    """Class distribution table, flagging rare classes and classes with <2 rows."""
    counts = labels.value_counts()
    table = pd.DataFrame({
        "Class": [str(c) for c in counts.index],
        "Count": counts.values,
        "Percentage": (counts.values / counts.sum() * 100).round(3),
    })
    table["Rare Class"] = table["Percentage"] < rare_fraction * 100
    table["Too Few For Stratified Split"] = table["Count"] < 2
    return table


def handle_small_classes(X, labels, class_df, remove_singletons, log):
    """
    A stratified split needs >=2 rows per class. Classes with 1 row are removed
    only if configured; otherwise the split falls back to non-stratified.
    Rare classes with >=2 rows are ALWAYS kept.
    Returns (X, labels, allow_stratify, removed_class_notes).
    """
    notes = []
    for _, row in class_df[class_df["Rare Class"] & ~class_df["Too Few For Stratified Split"]].iterrows():
        log.add("Classes", f"Class '{row['Class']}' is rare ({row['Count']} rows, {row['Percentage']}%) - kept.")

    singles = class_df.loc[class_df["Too Few For Stratified Split"], "Class"].tolist()
    if not singles:
        return X, labels, True, notes

    why = ("a stratified split must place at least one row of every class in both the training "
           "and testing sets, which is impossible for a class with a single row")
    if remove_singletons:
        mask = ~labels.astype(str).isin(singles)
        X, labels = X[mask].reset_index(drop=True), labels[mask].reset_index(drop=True)
        for cls in singles:
            note = f"Removed class '{cls}' (1 row): {why}."
            notes.append(note)
            log.add("Classes", note)
        if labels.nunique() < 2:
            raise ProjectError("Fewer than 2 classes remain after removing single-row classes.")
        return X, labels, True, notes

    log.add("Classes", f"Classes with 1 row kept ({singles}); stratification disabled because {why}.")
    return X, labels, False, notes


def encode_target(labels):
    """Encode labels as integers 0..k-1. Returns (y, encoder, class_names)."""
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels.astype(str) if not pdt.is_integer_dtype(labels) else labels)
    class_names = [str(c) for c in encoder.classes_]
    return y, encoder, class_names


# ----------------------------------------------------------- 7. train/test split
def split_data(X, y, test_size, random_state, allow_stratify, log):
    """80/20 split with random_state=42; stratified whenever possible."""
    if allow_stratify:
        try:
            parts = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
            log.add("Split", f"Stratified split (test_size={test_size}, random_state={random_state}).")
            return (*parts, True)
        except ValueError as exc:
            log.add("Split", f"Stratified split impossible ({exc}). Using a non-stratified split.")
    parts = train_test_split(X, y, test_size=test_size, random_state=random_state)
    log.add("Split", f"Non-stratified split (test_size={test_size}, random_state={random_state}).")
    return (*parts, False)


# ------------------------------------------------------------ 8. leakage checks
def detect_leakage_risks(X_train, y_train, groups, target, config, log, datetime_features=()):
    """
    Flag (do NOT remove) features that may directly encode the target.
    Uses TRAINING data only. Checks:
      1. feature name contains the whole target name
      2. a single feature alone almost perfectly predicts the target
      3. numeric feature almost perfectly correlated with a binary target
      4. softer warning when one feature alone is a strong predictor
      5. every feature derived from a date/time column (capture time can encode the class,
         e.g. when each class was recorded in a different session)
    """
    flagged = []
    target_tokens = set(split_name_tokens(target))
    y_series = pd.Series(np.asarray(y_train), index=X_train.index)
    n = len(X_train)
    rng = np.random.RandomState(config.RANDOM_STATE)
    sample_idx = rng.choice(n, size=min(n, 50000), replace=False)
    X_s = X_train.iloc[sample_idx]
    y_s = y_series.iloc[sample_idx]

    for col in X_train.columns:
        tokens = set(split_name_tokens(col))
        if target_tokens and target_tokens.issubset(tokens):
            flagged.append((col, "Name contains the target name", f"'{col}' vs target '{target}'"))

    discrete = groups["categorical"] + groups["categorical_high"] + groups["boolean"]
    for col in discrete:
        values = X_s[col].astype(object).where(X_s[col].notna(), "__missing__")
        if values.nunique() > len(values) / 20:
            continue  # too many categories per row: a lookup table would just memorise
        majority = pd.DataFrame({"v": values, "y": y_s}).groupby("v")["y"].agg(lambda s: s.mode().iloc[0])
        predicted = values.map(majority)
        score = balanced_accuracy_score(y_s, predicted)
        if score >= config.LEAKAGE_BALANCED_ACCURACY:
            flagged.append((col, "Single feature predicts target almost perfectly",
                            f"balanced accuracy {score:.3f} using this feature alone"))
        elif score >= config.LEAKAGE_WARNING_BALANCED_ACCURACY:
            flagged.append((col, "Warning: strong single-feature predictor",
                            f"balanced accuracy {score:.3f} using this feature alone"))

    for col in groups["numeric"]:
        values = X_s[col].astype(float)
        filled = values.fillna(values.median() if values.notna().any() else 0.0).to_numpy().reshape(-1, 1)
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(5, int(0.01 * len(filled))),
                                      random_state=config.RANDOM_STATE)
        tree.fit(filled, y_s)
        score = balanced_accuracy_score(y_s, tree.predict(filled))
        if score >= config.LEAKAGE_BALANCED_ACCURACY:
            flagged.append((col, "Single feature predicts target almost perfectly",
                            f"balanced accuracy {score:.3f} with a depth-3 tree on this feature alone"))
        elif score >= config.LEAKAGE_WARNING_BALANCED_ACCURACY:
            flagged.append((col, "Warning: strong single-feature predictor",
                            f"balanced accuracy {score:.3f} with a depth-3 tree on this feature alone"))
        if col in datetime_features:
            flagged.append((col, "Date/time-derived feature: capture time may encode the class",
                            f"single-feature balanced accuracy {score:.3f}; review how/when each class was recorded"))
        if y_s.nunique() == 2 and values.std() > 0:
            corr = np.corrcoef(filled.ravel(), y_s.to_numpy(dtype=float))[0, 1]
            if abs(corr) >= config.LEAKAGE_CORRELATION:
                flagged.append((col, "Near-perfect correlation with binary target", f"correlation {corr:.3f}"))

    table = pd.DataFrame(flagged, columns=["Feature", "Check", "Detail"]).drop_duplicates()
    if len(table):
        log.add("Leakage", f"{table['Feature'].nunique()} suspicious feature(s) FLAGGED (not removed): "
                           f"{table['Feature'].unique().tolist()}")
    else:
        log.add("Leakage", "No feature was flagged as suspicious by the automatic checks.")
    return table


def write_leakage_report(path, flagged, removed_df, target, n_train, n_test, stratified, removed_class_notes):
    """Write results/data_leakage_report.txt."""
    lines = [
        "DATA LEAKAGE REPORT", "=" * 60,
        f"Target column: {target}", "",
        "SAFEGUARDS APPLIED BY THIS PROJECT", "-" * 60,
        f"1. The data was split BEFORE any preprocessing ({n_train} training / {n_test} testing rows; "
        f"{'stratified' if stratified else 'not stratified'}).",
        "2. Imputers, scalers and encoders are inside the scikit-learn Pipeline and are fitted on the "
        "training data only.",
        "3. No hyperparameter tuning is performed, so the test set is never used for model selection.",
        "4. The target column is never used as an input feature.",
        "5. Duplicate rows were removed before splitting, so identical records cannot appear in "
        "both training and testing sets.",
        "6. Identifier-like columns were removed (see removed_features.csv).",
        "7. The dataset was not manipulated to raise accuracy.", "",
        "SUSPICIOUS FEATURES (flagged, NOT removed automatically)", "-" * 60,
    ]
    if len(flagged):
        lines.append(flagged.to_string(index=False))
        lines += ["", "Flagged features may be legitimate predictors, but a feature that alone predicts the "
                      "target almost perfectly can also be a copy, a derived value, or an outcome recorded "
                      "after the event (leakage). Please review them manually. To exclude one, add it to "
                      "EXCLUDE_COLUMNS in config.py and re-run."]
    else:
        lines.append("None flagged by the automatic checks.")
    lines += ["", "REMOVED FEATURES", "-" * 60]
    lines.append(removed_df.to_string(index=False) if len(removed_df) else "None.")
    if removed_class_notes:
        lines += ["", "REMOVED CLASSES", "-" * 60] + removed_class_notes
    lines += ["", "LIMITATIONS", "-" * 60,
              "Automatic checks cannot detect every kind of leakage (for example time-based leakage, "
              "or information that is only known after the outcome). Domain knowledge is still required.", ""]
    save_text(path, "\n".join(lines))
