"""Shared feature schema, training pipeline and prediction helpers."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "model.joblib"
TARGET = "low_mortality"  # 1 = flood-season duck mortality <= 15%

NUMERIC = ["age", "education_years", "hh_size", "land_decimals", "experience_years",
           "flock_size", "market_distance_km", "extension_training", "credit_access",
           "group_member", "n_practices"]
CATEGORICAL = ["gender", "farming_system"]
FEATURES = NUMERIC + CATEGORICAL


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def build_logreg() -> Pipeline:
    return Pipeline([("pre", build_preprocessor()), ("clf", LogisticRegression(max_iter=2000))])


def build_xgb() -> Pipeline:
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, eval_metric="logloss", random_state=42)
    return Pipeline([("pre", build_preprocessor()), ("clf", clf)])


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found - run `python scripts/train.py` first")
    return joblib.load(MODEL_PATH)


def predict(model, record: dict) -> dict:
    proba = float(model.predict_proba(pd.DataFrame([record])[FEATURES])[0, 1])
    return {"prediction": "Yes" if proba >= 0.5 else "No", "probability": round(proba, 3)}
