"""
Stage 5 - create safe sample inputs from the training sample.

Run from the project root:   python backend/make_payload.py

Creates (all contain ONLY the model's feature columns - no IP addresses, no timestamps):
  backend/test_payload.json     one valid input used by the tests
  backend/sample_inputs.csv     one example flow per class
  backend/sample_flows.json     the same examples + the model's prediction (used by the Angular page)
  backend/feature_schema.json   feature names + typical ranges (used to build the Angular form)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
BUNDLE = joblib.load(BACKEND / "models" / "model_bundle.joblib")
FEATURES = BUNDLE["features"]
CLASSES = BUNDLE["class_names"]
TARGET = BUNDLE["target"]


def find_sample_csv():
    import sys
    sys.path.insert(0, str(ROOT))
    import config
    name = config.CSV_FILE_NAME
    path = config.DATA_DIR / name if name else next(config.DATA_DIR.glob("*.csv"))
    return path


def main():
    df = pd.read_csv(find_sample_csv(), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET])
    df[TARGET] = df[TARGET].astype(str)

    # 1) Feature schema with typical ranges (1st-99th percentile) - shown as hints, not enforced,
    #    because network-flow features are heavy-tailed.
    schema = []
    for f in FEATURES:
        col = df[f].astype(float)
        schema.append({"name": f, "min": float(col.min()), "max": float(col.max()),
                       "p01": float(col.quantile(0.01)), "p99": float(col.quantile(0.99)),
                       "median": float(col.median())})
    (BACKEND / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    # 2) one example per class, chosen among rows the model classifies correctly
    pipeline = BUNDLE["pipeline"]
    rng = np.random.RandomState(42)
    flows, rows = [], []
    for cls in CLASSES:
        pool = df[df[TARGET].str.lower() == cls.lower()]
        proba = pipeline.predict_proba(pool[FEATURES])
        predicted = np.array([CLASSES[int(c)] for c in pipeline.classes_])[proba.argmax(axis=1)]
        ok = pool[predicted == cls]
        pick = (ok if len(ok) else pool).iloc[[rng.randint(len(ok if len(ok) else pool))]]
        p = pipeline.predict_proba(pick[FEATURES])[0]
        values = {f: float(pick.iloc[0][f]) for f in FEATURES}
        names = [CLASSES[int(c)] for c in pipeline.classes_]
        flows.append({"label": cls, "values": values, "prediction": names[int(p.argmax())],
                      "confidence": float(p.max()),
                      "probabilities": {n: float(v) for n, v in zip(names, p)}})
        rows.append({"true_class": cls, **values})
    (BACKEND / "sample_flows.json").write_text(json.dumps(flows, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(BACKEND / "sample_inputs.csv", index=False)

    # 3) the test payload = the Reconnaissance (or first) example + its expected result
    test = next((f for f in flows if f["label"].lower().startswith("recon")), flows[0])
    payload = {"input": test["values"], "expected_prediction": test["prediction"]}
    (BACKEND / "test_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote feature_schema.json, sample_flows.json, sample_inputs.csv, test_payload.json")
    for f in flows:
        print(f"  {f['label']:<18} -> predicted {f['prediction']:<18} ({f['confidence']:.2f})")


if __name__ == "__main__":
    main()
