# Connecting `train.py` to MLflow

This covers wiring `train.py` up to an MLflow tracking server — not
deploying MLflow itself. It assumes the server is already running
(kind cluster, Postgres as the backend store), which is a separate setup
step outside this repo.

## Why this exists

`train.py` can train three model types (`--model rf|gbm|xgb`), each
handling the dataset's ~3.4% class imbalance differently. Comparing them
by eyeballing printed metrics doesn't scale past one run — MLflow's whole
value is putting multiple runs side by side. This is what actually
answers the repo's own "would GradientBoosting/XGBoost beat RandomForest"
question, with evidence instead of a guess.

## 1. Find out how the MLflow service is exposed

`train.py` never talks to Postgres directly — Postgres is only where the
MLflow *server* persists its metadata. The client (`train.py`) only needs
one thing: an HTTP URL to that server.

```bash
kubectl get svc -n <mlflow-namespace>
```

- **Port-forward** (most common for local dev):
  ```bash
  kubectl port-forward svc/<mlflow-svc> 5000:5000
  ```
  → tracking URI is `http://localhost:5000`
- **NodePort**: still needs `kubectl port-forward`, or the kind cluster's
  `extraPortMappings` set at cluster-creation time, to actually be
  reachable from the host — a NodePort alone isn't enough with kind
  specifically, since the cluster runs inside Docker.

## 2. Point `train.py` at it

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"   # match step 1

pip install -r requirements.txt   # now includes mlflow + xgboost

python train.py --model rf
python train.py --model gbm
python train.py --model xgb
```

Open the MLflow UI, `predictive-maintenance` experiment — three runs,
each with `model_type`, `imbalance_handling`, `decision_threshold` logged
as params, and `precision`/`recall`/`f1`/`roc_auc`/`pr_auc` as metrics.
Each run also has the fitted pipeline itself logged as a model artifact
(`mlflow.sklearn.log_model`), so any run can be reloaded or promoted
straight from the UI later — not just its numbers.

## Two things that will trip you up if you hit them cold

**MLflow's file-store backend (`file:./mlruns`) is deprecated in current
MLflow** — it now requires a database backend. This isn't a problem for
you: Postgres is exactly what current MLflow wants, not just what the
tutorial happened to pick.

**XGBoost models fail to log with a `skops` serialization error** unless
explicitly trusted. Recent MLflow versions serialize sklearn models with
`skops` instead of raw `pickle` (safer to load later — pickle can execute
arbitrary code on load, skops can't), and skops refuses unfamiliar types
by default. Already handled in `train.py`:

```python
mlflow.sklearn.log_model(
    pipeline,
    name="model",
    skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
)
```

## Why CI never breaks over this

Your kind cluster only exists on your machine — GitHub's hosted runners
can't reach `localhost:5000` or any NodePort on it. `train.py` treats the
MLflow connection as best-effort: it does a cheap reachability check up
front, and if the server isn't reachable, prints a warning and trains
normally without tracking. Locally, against your real cluster, you get
full tracking. In CI, it just skips MLflow — `artifacts/model.pkl` and
`metrics.json` still get produced either way, and the pipeline never
depends on your laptop's cluster being up.
