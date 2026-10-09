import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from serving.model import MODEL_PATH, FEATURES  # noqa: E402

SAMPLE = dict(age=35, gender="Female", education_years=5, hh_size=5, land_decimals=60,
              experience_years=10, farming_system="Free-range", flock_size=40,
              market_distance_km=5.0, extension_training=1, credit_access=1,
              group_member=0, n_practices=3)


def test_synthetic_data_schema():
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "synthetic" / "SYNTHETIC_duck_survey.csv")
    assert set(FEATURES) <= set(df.columns)
    assert (df["is_synthetic"] == 1).all()


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run scripts/train.py first")
def test_api_predict_and_health():
    from app.main import app
    c = TestClient(app)
    assert c.get("/").json() == {"status": "ok"}
    r = c.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    assert r.json()["prediction"] in ("Yes", "No")
    assert 0 <= r.json()["probability"] <= 1
    assert c.post("/predict", json={**SAMPLE, "gender": "X"}).status_code == 422
