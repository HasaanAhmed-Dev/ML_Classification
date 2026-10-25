"""
Stage 5 - direct inference check (no Flask involved).
Run from the project root:   python backend/test_pipeline.py     (or: pytest backend/test_pipeline.py)
"""
import json
from pathlib import Path

import joblib
import pandas as pd

BACKEND = Path(__file__).resolve().parent


def test_direct_inference_matches_expected():
    bundle = joblib.load(BACKEND / "models" / "model_bundle.joblib")
    payload = json.loads((BACKEND / "test_payload.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame([[payload["input"][f] for f in bundle["features"]]], columns=bundle["features"])
    pipeline = bundle["pipeline"]
    proba = pipeline.predict_proba(frame)[0]
    names = [bundle["class_names"][int(c)] for c in pipeline.classes_]
    predicted = names[int(proba.argmax())]
    assert abs(proba.sum() - 1.0) < 1e-6
    assert predicted == payload["expected_prediction"]


if __name__ == "__main__":
    test_direct_inference_matches_expected()
    print("OK: direct inference matches the expected prediction")
