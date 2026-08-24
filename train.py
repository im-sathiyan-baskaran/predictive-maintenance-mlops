"""
Train a machine-failure classifier on the AI4I 2020 Predictive Maintenance dataset.

Writes:
    artifacts/model.pkl     — fitted sklearn Pipeline (preprocessing + model)
    artifacts/metrics.json  — evaluation metrics on a held-out test split

Source: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository
(S. Matzka, 2020). ~10,000 rows, ~3.4% positive (failure) rate.
"""
import argparse
import json
import os
import pickle
from contextlib import nullcontext
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

DATA_PATH = Path("data/ai4i2020.csv")
ARTIFACTS_DIR = Path("artifacts")
DECISION_THRESHOLD = 0.30

TARGET = "Machine failure"

# TWF/HDF/PWF/OSF/RNF are sub-flags of "Machine failure" itself — if any one of
# them is 1, the target is 1. Keeping them as features would let the model
# "predict" failure by reading a column that IS the failure. Drop them.
LEAKY_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
ID_COLUMNS = ["UDI", "Product ID"]

CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    return df.drop(columns=ID_COLUMNS + LEAKY_COLUMNS)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )


# Each model handles the ~3.4% positive rate differently: RandomForest and
# GradientBoosting take it as a training-time weight, XGBoost takes it as a
# single scalar (scale_pos_weight). fit_kwargs lets main() pass the right
# sample_weight through .fit() without the model classes needing to know
# about each other.
def build_model(model_type: str, params: dict):
    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        imbalance_handling = "class_weight=balanced"
        needs_sample_weight = False
    elif model_type == "gbm":
        # GradientBoostingClassifier has no class_weight param — sklearn only
        # added that to RandomForest/tree-ensemble bagging models. The
        # equivalent lever here is sample_weight passed at .fit() time.
        clf = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
        )
        imbalance_handling = "sample_weight=balanced"
        needs_sample_weight = True
    elif model_type == "xgb":
        # XGBoost's imbalance lever is scale_pos_weight, a single ratio
        # (negatives/positives) rather than per-sample weights — computed in
        # main() once y_train is known, then passed in via params.
        clf = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            scale_pos_weight=params["scale_pos_weight"],
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
        imbalance_handling = f"scale_pos_weight={params['scale_pos_weight']:.1f}"
        needs_sample_weight = False
    else:
        raise ValueError(f"Unknown --model '{model_type}'. Choose from: rf, gbm, xgb")

    pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", clf)])
    return pipeline, imbalance_handling, needs_sample_weight


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train + MLflow-track a machine-failure classifier.")
    p.add_argument("--model", choices=["rf", "gbm", "xgb"], default="rf",
                    help="Which classifier to train (default: rf)")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=None,
                    help="Tree depth. Default: None for rf (unbounded), 3 for gbm/xgb if unset.")
    p.add_argument("--experiment", default="predictive-maintenance",
                    help="MLflow experiment name")
    p.add_argument("--run-name", default=None,
                    help="MLflow run name (default: <model>-run)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    max_depth = args.max_depth if args.max_depth is not None else (None if args.model == "rf" else 3)

    # MLflow server is in-cluster (kind + Postgres backend store) — train.py
    # only ever talks to it over this HTTP tracking URI, never to Postgres
    # directly. Set MLFLOW_TRACKING_URI to whatever exposes that service to
    # your host: `kubectl port-forward svc/<mlflow-svc> 5000:5000` gives you
    # http://localhost:5000; a NodePort gives http://localhost:<nodePort>.
    # Best-effort: your MLflow server lives on your local kind cluster, which
    # GitHub Actions' hosted runners can't reach. Locally this connects fine;
    # in CI (or anywhere else the server isn't reachable) it logs a warning
    # and mlflow_enabled flips to False, so training still runs and
    # artifacts/ still gets produced either way — CI never depends on your
    # laptop's cluster being up.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_enabled = True
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(args.experiment)
        mlflow.MlflowClient().search_experiments(max_results=1)  # cheap reachability check
    except Exception as e:
        print(f"MLflow server not reachable at {tracking_uri} ({e}); continuing without tracking.")
        mlflow_enabled = False

    df = load_data()
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # scale_pos_weight (xgb only) needs the train-set class ratio, so it's
    # computed here, before build_model(), rather than inside it.
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": max_depth,
        "scale_pos_weight": float((y_train == 0).sum() / (y_train == 1).sum()),
    }
    pipeline, imbalance_handling, needs_sample_weight = build_model(args.model, params)

    run_name = args.run_name or f"{args.model}-run"
    run_context = mlflow.start_run(run_name=run_name) if mlflow_enabled else nullcontext()
    with run_context:
        if mlflow_enabled:
            mlflow.log_param("model_type", args.model)
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("imbalance_handling", imbalance_handling)
            mlflow.log_param("decision_threshold", DECISION_THRESHOLD)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))

        if needs_sample_weight:
            sample_weight = compute_sample_weight("balanced", y_train)
            pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
        else:
            pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # The default 0.5 cutoff on predict() is a bad fit here: it optimizes
        # for being "confident", not for catching failures. 0.30 was chosen
        # by an earlier threshold sweep as the best F1 tradeoff on this split
        # — kept fixed across model types here so the comparison in MLflow is
        # apples-to-apples (same cutoff, different model).
        y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

        # Accuracy is logged for reference only — on this class balance it's
        # nearly meaningless (predicting "no failure" every time scores ~97%).
        # Recall/F1/PR-AUC are what actually tell you whether the model works.
        metrics = {
            "decision_threshold": DECISION_THRESHOLD,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "pr_auc": round(average_precision_score(y_test, y_proba), 4),
            "positive_rate_test_set": round(float(y_test.mean()), 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        run_id = None
        if mlflow_enabled:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

            # Logs the fitted pipeline itself to MLflow's artifact store (not
            # just the numbers), so any run can be reloaded or promoted from
            # the MLflow UI later — the gap in the reference tutorial's
            # train.py. Recent MLflow versions serialize sklearn models with
            # skops instead of raw pickle (safer to load later), which
            # refuses unfamiliar types by default — xgboost's classes need to
            # be explicitly trusted here; rf/gbm runs ignore this harmlessly.
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
            )
            run_id = mlflow.active_run().info.run_id

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if mlflow_enabled:
        print(f"MLflow run: {run_id} (experiment: {args.experiment})")
    print("Saved artifacts/model.pkl")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
