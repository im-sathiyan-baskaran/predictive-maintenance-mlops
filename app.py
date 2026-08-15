"""
Minimal Flask API serving the predictive-maintenance model.

POST /predict
{
    "type": "M",
    "air_temp": 298.1,
    "process_temp": 308.6,
    "rpm": 1551,
    "torque": 42.8,
    "tool_wear": 0
}
"""
import json
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = Path("artifacts/model.pkl")
METRICS_PATH = Path("artifacts/metrics.json")

app = Flask(__name__)
_model = None  # loaded lazily on first request
_threshold = None


def get_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def get_threshold(default: float = 0.5) -> float:
    global _threshold
    if _threshold is None:
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                _threshold = json.load(f).get("decision_threshold", default)
        else:
            _threshold = default
    return _threshold


REQUIRED_FIELDS = ["type", "air_temp", "process_temp", "rpm", "torque", "tool_wear"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    row = pd.DataFrame([{
        "Type": payload["type"],
        "Air temperature [K]": payload["air_temp"],
        "Process temperature [K]": payload["process_temp"],
        "Rotational speed [rpm]": payload["rpm"],
        "Torque [Nm]": payload["torque"],
        "Tool wear [min]": payload["tool_wear"],
    }])

    clf = get_model()
    threshold = get_threshold()
    probability = float(clf.predict_proba(row)[0][1])
    prediction = probability >= threshold

    return jsonify({
        "machine_failure": prediction,
        "failure_probability": round(probability, 4),
        "decision_threshold": threshold,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
