"""SageMaker scikit-learn serving handlers for the T1 late-delivery model.

Deploy with this file as entry_point and a source_dir that also contains t1_pipeline.py
(needed to unpickle the pipeline) and requirements.txt (the final model is XGBoost).
model.tar.gz must hold <name>.joblib and <name>_meta.json.
"""
import io
import json
import os

import joblib
import pandas as pd

import t1_pipeline  # noqa: F401

MODEL_NAME = os.environ.get("T1_MODEL_NAME", "t1_final")


def model_fn(model_dir):
    with open(os.path.join(model_dir, f"{MODEL_NAME}_meta.json")) as f:
        meta = json.load(f)
    return {"pipeline": joblib.load(os.path.join(model_dir, f"{MODEL_NAME}.joblib")), "meta": meta}


def input_fn(body, content_type="application/json"):
    body = body.decode("utf-8") if isinstance(body, (bytes, bytearray)) else body
    if content_type == "application/json":
        data = json.loads(body)
        return pd.DataFrame(data["instances"] if isinstance(data, dict) else data)
    if content_type == "text/csv":
        return pd.read_csv(io.StringIO(body))
    raise ValueError(f"unsupported content type: {content_type}")


def predict_fn(df, model):
    meta = model["meta"]
    missing = sorted(set(meta["required_columns"]) - set(df.columns))
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    X = df.reindex(columns=meta["input_columns"])
    X[meta["optional_zero_columns"]] = X[meta["optional_zero_columns"]].fillna(0)
    score = model["pipeline"].predict_proba(X)[:, 1]
    ids = df["order_id"].tolist() if "order_id" in df else [None] * len(df)
    return [{"order_id": i, "risk_score": round(float(s), 6), "pred_label": int(s >= meta["threshold"]),
             "model_version": meta["model_version"]} for i, s in zip(ids, score)]


def output_fn(predictions, accept="application/json"):
    return json.dumps({"predictions": predictions})
