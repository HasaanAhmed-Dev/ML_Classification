"""Stage 7 - four automated API tests (run from the project root: pytest backend -v)."""
import json
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
PAYLOAD = json.loads((BACKEND / "test_payload.json").read_text(encoding="utf-8"))


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok" and body["features"] > 0


def test_predict_valid_input(client):
    response = client.post("/predict", json=PAYLOAD["input"])
    assert response.status_code == 200
    body = response.get_json()
    assert body["prediction"] == PAYLOAD["expected_prediction"]
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6


def test_predict_schema_mismatch(client):
    bad = dict(PAYLOAD["input"])
    removed = next(iter(bad))
    bad.pop(removed)
    bad["not_a_feature"] = 1.0
    response = client.post("/predict", json=bad)
    assert response.status_code == 400
    body = response.get_json()
    assert removed in body["missing"] and "not_a_feature" in body["extra"]


def test_predict_rejects_non_numeric(client):
    bad = dict(PAYLOAD["input"])
    first = next(iter(bad))
    bad[first] = "abc"
    response = client.post("/predict", json=bad)
    assert response.status_code == 400
    assert first in response.get_json()["fields"]
