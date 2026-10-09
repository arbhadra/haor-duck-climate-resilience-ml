"""Train and compare models for the resilience target; log to MLflow; save the best."""
import argparse
import sys
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serving.model import FEATURES, MODEL_PATH, TARGET, build_logreg, build_xgb  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "synthetic" / "SYNTHETIC_duck_survey.csv"


def main(data_path: Path) -> None:
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("haor-duck-resilience")
    best = (None, -1.0, None)
    for name, build in [("logreg", build_logreg), ("xgboost", build_xgb)]:
        model = build()
        cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        auc, acc = roc_auc_score(y_te, proba), accuracy_score(y_te, proba >= 0.5)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "data": data_path.name, "n_rows": len(df), "target": TARGET})
            mlflow.log_metrics({"cv_auc": cv_auc, "test_auc": auc, "test_accuracy": acc})
        print(f"{name}: cv_auc={cv_auc:.3f} test_auc={auc:.3f} test_acc={acc:.3f}")
        if cv_auc > best[1]:
            best = (name, cv_auc, model)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best[2], MODEL_PATH)
    print(f"saved best model: {best[0]} -> {MODEL_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    main(ap.parse_args().data)
