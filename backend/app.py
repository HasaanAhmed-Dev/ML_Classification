"""
Stage 6 - Flask REST API.

Run from the project root (Terminal 1, keep it running):
    python backend/app.py
Endpoints:  GET /health     POST /predict  (JSON object with every feature name -> number)
"""
import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE = Path(__file__).resolve().parent
BUNDLE = joblib.load(BASE / "models" / "model_bundle.joblib")
MODEL = BUNDLE["pipeline"]
FEATURES = BUNDLE["features"]
CLASS_NAMES = BUNDLE["class_names"]
TASK = BUNDLE.get("task", "classification")
DISCLAIMER = ("Research/teaching prototype only. Trained on a public dataset (MedSec-25) with a random "
              "split; not validated on real networks or on new capture sessions.")

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok", features=len(FEATURES), task=TASK, classes=CLASS_NAMES,
                   model=BUNDLE.get("model_name"), model_version=BUNDLE.get("model_version"))


def _is_real_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Expected a JSON object"), 400
    missing = [f for f in FEATURES if f not in data]
    extra = [f for f in data if f not in FEATURES]
    if missing or extra:
        return jsonify(error="Input schema mismatch", missing=missing, extra=extra), 400
    bad = [f for f in FEATURES if not _is_real_number(data[f])]
    if bad:
        return jsonify(error="Invalid input", detail="Values must be finite numbers", fields=bad), 400

    frame = pd.DataFrame([[float(data[f]) for f in FEATURES]], columns=FEATURES)
    try:
        proba = MODEL.predict_proba(frame)[0]
        names = [CLASS_NAMES[int(c)] for c in MODEL.classes_]   # map class codes -> labels
        best = int(proba.argmax())
        return jsonify(prediction=names[best], confidence=float(proba[best]), task=TASK,
                       probabilities={n: float(p) for n, p in zip(names, proba)},
                       model_version=BUNDLE.get("model_version"), disclaimer=DISCLAIMER)
    except (ValueError, TypeError) as exc:
        return jsonify(error="Invalid input", detail=str(exc)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction processing failed"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
