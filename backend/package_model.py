"""
Stage 4 - package the trained model into ONE deployable bundle.

Run from the project root, AFTER `python main.py` has created models/*.joblib:
    python backend/package_model.py

The bundle holds the fitted pipeline (preprocessing + Random Forest), the exact
input feature names/order and the class names, so the API never has to guess them.
"""
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL = ROOT / "models" / "random_forest_model.joblib"   # or logistic_regression_model.joblib
SOURCE_META = ROOT / "models" / "model_metadata.joblib"
OUTPUT = ROOT / "backend" / "models" / "model_bundle.joblib"


def main():
    if not SOURCE_MODEL.exists() or not SOURCE_META.exists():
        raise SystemExit("Run `python main.py` first - the trained model files are missing.")
    pipeline = joblib.load(SOURCE_MODEL)          # trusted file created by this project
    meta = joblib.load(SOURCE_META)
    groups = meta["feature_groups"]
    if groups["categorical"] or groups["categorical_high"] or groups["boolean"]:
        raise SystemExit("This API/form supports numeric features only; your model has other feature types.")
    assert hasattr(pipeline, "predict_proba"), "Model must provide predict_proba"
    bundle = {
        "pipeline": pipeline,
        "features": list(meta["feature_columns"]),   # EXACT training input order
        "class_names": list(meta["class_names"]),
        "target": meta["target"],
        "task": "classification",
        "model_name": "Random Forest",
        "model_version": "1.0",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUTPUT, compress=3)
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"Saved {OUTPUT}  ({size_mb:.1f} MB)")
    print(f"{len(bundle['features'])} features, classes: {bundle['class_names']}")
    if size_mb > 90:
        print("WARNING: GitHub rejects files over 100 MB. Lower N_ESTIMATORS or set RF_MAX_DEPTH in config.py.")


if __name__ == "__main__":
    main()
