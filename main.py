"""
main.py - runs the complete classification workflow.

Usage:  python main.py
Put exactly one CSV file in the "data" folder first (see README.md).
"""
import sys
import time
import traceback
import warnings

import joblib
import pandas as pd

import config
from src import data_loader, data_preprocessing as prep, evaluate_models as ev
from src import target_detection, train_models, visualizations as viz
from src.utils import DecisionLog, ProjectError, banner, ensure_directories, save_text

warnings.filterwarnings("ignore", category=FutureWarning)


def run():
    # Windows consoles may not support every character: never crash on printing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    start = time.time()
    results_dir, models_dir = config.RESULTS_DIR, config.MODELS_DIR
    ensure_directories(config.DATA_DIR, models_dir, results_dir)
    log = DecisionLog(echo=True)

    # ------------------------------------------------ 1-2. load and inspect
    print(banner("STEP 1/6  LOAD AND INSPECT DATASET", "-"))
    csv_path = data_loader.find_csv_file(config.DATA_DIR, config.CSV_FILE_NAME)
    raw_df = data_loader.load_csv(csv_path)
    info = data_loader.inspect_dataset(raw_df, csv_path.name)
    viz.plot_missing_values(info["missing_df"], results_dir / "missing_values.png")

    # ------------------------------------------------ 3. target detection
    print(banner("STEP 2/6  DETECT TARGET COLUMN", "-"))
    df = prep.normalize_dataframe(raw_df, log)
    target, _candidates = target_detection.detect_target(
        df, config.TARGET_COLUMN, config.MAX_TARGET_CLASSES, log)
    print(f"\n  >>> SELECTED TARGET COLUMN: {target}\n")

    # ------------------------------------------------ 4-6. cleaning and features
    print(banner("STEP 3/6  CLEAN DATA AND DETECT FEATURES", "-"))
    df = prep.convert_numeric_like(df, target, log)
    df, boolean_cols = prep.convert_booleans(df, target, log)
    df, datetime_cols = prep.convert_datetimes(df, target, log)
    df, removed = prep.clean_dataset(df, target, config, log)
    df, new_dt_features = prep.extract_datetime_features(df, datetime_cols, removed, log)
    groups = prep.detect_feature_types(df, target, boolean_cols, config, log)

    # ------------------------------------------------ 7. class analysis
    labels = prep.prepare_target_labels(df[target])
    class_df = prep.analyze_classes(labels, config.RARE_CLASS_FRACTION)
    X = df.drop(columns=[target])
    X, labels, allow_stratify, removed_class_notes = prep.handle_small_classes(
        X, labels, class_df, config.REMOVE_SINGLETON_CLASSES, log)
    if removed_class_notes:  # classes were removed -> report the distribution actually used
        class_df = prep.analyze_classes(labels, config.RARE_CLASS_FRACTION)
        groups = {k: [c for c in v if c in X.columns] for k, v in groups.items()}
    y, label_encoder, class_names = prep.encode_target(labels)
    n_classes = len(class_names)
    print(f"\n  Number of classes: {n_classes}")
    print(class_df.to_string(index=False))
    if n_classes > 50:
        raise ProjectError(f"The target has {n_classes} classes; this looks like a regression "
                           "or identifier column. Set TARGET_COLUMN in config.py.")
    class_df.to_csv(results_dir / "class_distribution.csv", index=False)
    viz.plot_class_distribution(class_df, results_dir / "class_distribution.png")

    removed_df = pd.DataFrame(removed, columns=["Feature", "Reason"])
    removed_df.to_csv(results_dir / "removed_features.csv", index=False)
    data_loader.save_dataset_summary(results_dir, info, target, class_df, removed_class_notes)

    # ------------------------------------------------ 8. split BEFORE preprocessing
    print(banner("STEP 4/6  TRAIN/TEST SPLIT AND LEAKAGE CHECKS", "-"))
    X_train, X_test, y_train, y_test, stratified = prep.split_data(
        X, y, config.TEST_SIZE, config.RANDOM_STATE, allow_stratify, log)
    print(f"  Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    flagged = prep.detect_leakage_risks(X_train, y_train, groups, target, config, log,
                                      datetime_features=set(new_dt_features))
    prep.write_leakage_report(results_dir / "data_leakage_report.txt", flagged, removed_df, target,
                              len(X_train), len(X_test), stratified, removed_class_notes)

    # feature_summary.csv: what happened to every original column
    group_of = {c: g for g, cols in groups.items() for c in cols}
    removed_reason = dict(zip(removed_df["Feature"], removed_df["Reason"]))
    rows = []
    for col in info["columns_table"]["Feature"]:
        clean_name = str(col).strip()
        if clean_name in removed_reason:
            status = ("replaced by date components" if removed_reason[clean_name].startswith("Datetime column replaced")
                      else "removed")
        else:
            status = "target" if clean_name == target else "used"
        inferred = group_of.get(clean_name, "target" if clean_name == target else
                                ("datetime" if clean_name in datetime_cols else "removed"))
        row = info["columns_table"][info["columns_table"]["Feature"] == col].iloc[0]
        rows.append({"Feature": col, "Original Dtype": row["Dtype"], "Inferred Type": inferred,
                     "Unique Values": row["Unique Values"], "Missing": row["Missing"],
                     "Missing %": row["Missing %"], "Status": status,
                     "Reason": removed_reason.get(clean_name, "")})
    pd.DataFrame(rows).to_csv(results_dir / "feature_summary.csv", index=False)
    save_text(results_dir / "preprocessing_decisions.txt", log.to_text() + "\n")

    # ------------------------------------------------ 9-11. train
    print(banner("STEP 5/6  TRAIN MODELS", "-"))
    lr_pipe = train_models.build_logistic_regression(groups, config)
    rf_pipe = train_models.build_random_forest(groups, config)
    lr_pipe, lr_time = train_models.train_pipeline("Logistic Regression", lr_pipe, X_train, y_train, log)
    rf_pipe, rf_time = train_models.train_pipeline("Random Forest", rf_pipe, X_train, y_train, log)
    train_models.save_model(lr_pipe, models_dir / "logistic_regression_model.joblib", config.MODEL_COMPRESSION)
    train_models.save_model(rf_pipe, models_dir / "random_forest_model.joblib", config.MODEL_COMPRESSION)
    joblib.dump({"target": target, "class_names": class_names, "label_encoder": label_encoder,
                 "feature_columns": list(X.columns), "feature_groups": groups},
                models_dir / "model_metadata.joblib")

    # ------------------------------------------------ 12-17. evaluate
    print(banner("STEP 6/6  EVALUATE AND CREATE REPORTS", "-"))
    results = [
        ev.evaluate_model("Logistic Regression", "logistic_regression", lr_pipe,
                          X_train, y_train, X_test, y_test, class_names, lr_time),
        ev.evaluate_model("Random Forest", "random_forest", rf_pipe,
                          X_train, y_train, X_test, y_test, class_names, rf_time),
    ]
    for r in results:
        ev.write_classification_report(results_dir / f"classification_report_{r['key']}.txt", r, target)
        viz.plot_confusion_matrix(r["confusion"], r["confusion_names"], r["name"],
                                  results_dir / f"{r['key']}_confusion_matrix.png")
        viz.plot_roc_curve(r["roc"], r["name"], results_dir / f"{r['key']}_roc_curve.png")

    ev.build_auc_table(results).to_csv(results_dir / "auc_results.csv", index=False)

    lr_imp = ev.logistic_regression_importance(lr_pipe)
    rf_imp = ev.random_forest_importance(rf_pipe)
    lr_imp.to_csv(results_dir / "logistic_regression_feature_importance.csv", index=False)
    rf_imp.to_csv(results_dir / "random_forest_feature_importance.csv", index=False)
    viz.plot_feature_importance(
        lr_imp, f"Logistic Regression - top {config.TOP_FEATURES} features",
        "Mean |coefficient| across classes", results_dir / "logistic_regression_feature_importance.png",
        config.TOP_FEATURES)
    viz.plot_feature_importance(
        rf_imp, f"Random Forest - top {config.TOP_FEATURES} features",
        "Impurity-based importance", results_dir / "random_forest_feature_importance.png",
        config.TOP_FEATURES)

    overfitting = ev.build_overfitting_table(results)
    overfitting.to_csv(results_dir / "overfitting_analysis.csv", index=False)
    comparison = ev.build_comparison_table(results)
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    viz.plot_model_comparison(comparison, results_dir / "model_comparison.png")
    if 2 < n_classes <= 30:  # per-class view is only informative for multiclass problems
        viz.plot_per_class_f1(ev.build_per_class_f1(results, class_names, class_df),
                              results_dir / "per_class_f1.png")

    ctx = {
        "file_name": csv_path.name, "target": target, "config": config,
        "original_shape": raw_df.shape, "cleaned_shape": (len(X), X.shape[1] + 1),
        "n_features": X.shape[1], "n_classes": n_classes, "groups": groups,
        "n_transformed": len(ev.get_feature_names(lr_pipe)),
        "n_train": len(X_train), "n_test": len(X_test), "stratified": stratified,
        "results": results, "overfitting": overfitting, "comparison": comparison,
        "importances": (lr_imp, rf_imp), "flagged": flagged,
        "results_dir": results_dir, "models_dir": models_dir,
    }
    save_text(results_dir / "final_results.txt", ev.build_final_results_text(ctx))
    ev.print_terminal_report(ctx)
    print(f"\nTotal run time: {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    try:
        run()
    except ProjectError as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
    except Exception:
        print("\nUNEXPECTED ERROR - please send the text below to get help:\n")
        traceback.print_exc()
        sys.exit(2)
