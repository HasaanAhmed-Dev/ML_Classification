import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


@pytest.fixture(scope="session")
def client():
    if not (BACKEND / "models" / "model_bundle.joblib").exists():
        pytest.skip("model_bundle.joblib is not available in this environment")
    from app import app
    app.config["TESTING"] = True
    return app.test_client()
