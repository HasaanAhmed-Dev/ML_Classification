"""Building and training the two model pipelines (preprocessing + classifier)."""
import time
import warnings

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(groups, config, scale_numeric):
    """
    ColumnTransformer for the detected feature groups.
    All statistics (medians, most-frequent values, means, categories) are learned
    ONLY when the pipeline is fitted on the training data.
    """
    transformers = []
    if groups["numeric"]:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), groups["numeric"]))
    if groups["boolean"]:
        transformers.append(("boolean", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]), groups["boolean"]))
    if groups["categorical"]:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), groups["categorical"]))
    if groups["categorical_high"]:
        # Rare categories are grouped into one "infrequent" column (learned on train only).
        transformers.append(("categorical_grouped", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     max_categories=config.MAX_ONEHOT_CATEGORIES,
                                     min_frequency=config.RARE_CATEGORY_MIN_FREQUENCY)),
        ]), groups["categorical_high"]))
    return ColumnTransformer(transformers, remainder="drop")


def build_logistic_regression(groups, config):
    """Logistic Regression pipeline (numeric features are scaled)."""
    return Pipeline([
        ("preprocessor", build_preprocessor(groups, config, scale_numeric=True)),
        ("classifier", LogisticRegression(
            max_iter=config.LR_MAX_ITER,
            class_weight="balanced" if config.USE_CLASS_WEIGHT else None,
            random_state=config.RANDOM_STATE,
        )),
    ])


def build_random_forest(groups, config):
    """Random Forest pipeline (trees do not need scaling)."""
    return Pipeline([
        ("preprocessor", build_preprocessor(groups, config, scale_numeric=False)),
        ("classifier", RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE,
            class_weight="balanced" if config.USE_CLASS_WEIGHT else None,
            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
            max_depth=config.RF_MAX_DEPTH,
            n_jobs=-1,
        )),
    ])


def train_pipeline(name, pipeline, X_train, y_train, log):
    """Fit a pipeline on the TRAINING data only. Returns (pipeline, seconds)."""
    print(f"  Training {name} on {len(X_train)} rows ...")
    start = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(X_train, y_train)
    seconds = time.time() - start
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        log.add("Training", f"{name} did not fully converge in {pipeline.named_steps['classifier'].max_iter} "
                            "iterations. Consider raising LR_MAX_ITER in config.py.")
    log.add("Training", f"{name} trained in {seconds:.1f} seconds.")
    return pipeline, seconds


def save_model(pipeline, path, compression):
    """Save a fitted pipeline with joblib."""
    joblib.dump(pipeline, path, compress=compression)
