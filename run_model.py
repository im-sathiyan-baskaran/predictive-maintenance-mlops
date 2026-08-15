"""
CLI predictor for the machine-failure model.

Usage:
    python run_model.py --type M --air-temp 298.1 --process-temp 308.6 \
        --rpm 1551 --torque 42.8 --tool-wear 0
"""
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

MODEL_PATH = Path("artifacts/model.pkl")
METRICS_PATH = Path("artifacts/metrics.json")


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_threshold(default: float = 0.5) -> float:
    """Use the threshold train.py picked, so CLI predictions match the
    reported metrics instead of silently reverting to the naive 0.5 cutoff."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f).get("decision_threshold", default)
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict machine failure risk.")
    parser.add_argument("--type", choices=["L", "M", "H"], required=True,
                         help="Product quality variant (Low/Medium/High)")
    parser.add_argument("--air-temp", type=float, required=True, help="Air temperature [K]")
    parser.add_argument("--process-temp", type=float, required=True, help="Process temperature [K]")
    parser.add_argument("--rpm", type=float, required=True, help="Rotational speed [rpm]")
    parser.add_argument("--torque", type=float, required=True, help="Torque [Nm]")
    parser.add_argument("--tool-wear", type=float, required=True, help="Tool wear [min]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model()
    threshold = load_threshold()

    row = pd.DataFrame([{
        "Type": args.type,
        "Air temperature [K]": args.air_temp,
        "Process temperature [K]": args.process_temp,
        "Rotational speed [rpm]": args.rpm,
        "Torque [Nm]": args.torque,
        "Tool wear [min]": args.tool_wear,
    }])

    probability = float(model.predict_proba(row)[0][1])
    prediction = probability >= threshold

    print(f"Machine failure: {'YES' if prediction else 'NO'} (threshold={threshold})")
    print(f"Failure probability: {probability:.4f}")


if __name__ == "__main__":
    main()
