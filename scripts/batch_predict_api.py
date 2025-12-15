"""Batch prediction client for the Wine Quality API

Reads a CSV or JSON file with sample rows (columns matching the feature names),
posts them to the FastAPI `/predict/batch` endpoint, and writes predictions
(back into a CSV or JSON) including predicted class and confidence.

Usage example:
python scripts/batch_predict_api.py --input data/WineQT.csv --output predictions.csv --host 127.0.0.1 --port 8000 --model rf

"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests


FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid",
    "residual_sugar", "chlorides", "free_sulfur_dioxide",
    "total_sulfur_dioxide", "density", "pH",
    "sulphates", "alcohol",
]


def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() in (".csv",):
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".json", ".ndjson"):
        df = pd.read_json(path)
    else:
        raise ValueError("Unsupported input format. Use CSV or JSON.")

    # Keep only expected features and warn for missing
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")

    # Ensure column order
    return df[FEATURES].copy()


def call_batch_api(samples: list, url: str, model: str = "rf") -> dict:
    payload = {"model": model, "samples": samples}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def assemble_results(df: pd.DataFrame, api_response: dict) -> pd.DataFrame:
    # `api_response` expected to contain 'results' list where each entry has sample_index and prediction
    results = api_response.get("results", [])
    preds = []
    for r in results:
        idx = r.get("sample_index")
        if "prediction" in r:
            pred = r["prediction"]
            preds.append({
                "sample_index": idx,
                "predicted_class": pred.get("class"),
                "confidence": pred.get("confidence"),
                "probability": json.dumps(pred.get("probability", {}))
            })
        else:
            preds.append({
                "sample_index": idx,
                "predicted_class": None,
                "confidence": None,
                "probability": json.dumps({"error": r.get("error")})
            })

    pred_df = pd.DataFrame(preds).set_index("sample_index")

    # Combine input features with predictions
    df_out = df.reset_index(drop=True).join(pred_df, how="left")
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Batch predict via Wine Quality API")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV/JSON with samples")
    parser.add_argument("--output", "-o", required=True, help="Path for output CSV/JSON")
    parser.add_argument("--host", default="127.0.0.1", help="API host (default 127.0.0.1)")
    parser.add_argument("--port", default=8000, type=int, help="API port (default 8000)")
    parser.add_argument("--model", default="rf", help="Model id to request (optional)")
    parser.add_argument("--endpoint", default="/predict/batch", help="API endpoint path (default /predict/batch)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        df = load_input(input_path)
    except Exception as e:
        print(f"Failed to read input: {e}", file=sys.stderr)
        sys.exit(2)

    samples = df.to_dict(orient="records")

    url = f"http://{args.host}:{args.port}{args.endpoint}"
    print(f"Posting {len(samples)} samples to {url} ...")

    try:
        resp_json = call_batch_api(samples, url, model=args.model)
    except Exception as e:
        print(f"API request failed: {e}", file=sys.stderr)
        sys.exit(3)

    # Build output dataframe
    df_out = assemble_results(df, resp_json)

    # Write output using requested extension
    if output_path.suffix.lower() == ".csv":
        df_out.to_csv(output_path, index=False)
    elif output_path.suffix.lower() == ".json":
        df_out.to_json(output_path, orient="records", indent=2)
    else:
        print("Unsupported output format - use .csv or .json", file=sys.stderr)
        sys.exit(4)

    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
