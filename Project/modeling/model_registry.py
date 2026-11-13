"""Saves and loads the models and their feature schemas, and saves their scores.
Nothing else in the project writes to output/models/."""

import json
import os

import joblib

from config import config


def _ensure_folder(file_path):
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def save_model(model, file_path):
    """Put a fitted model on disk, compressed, replacing any earlier file."""
    _ensure_folder(file_path)
    joblib.dump(model, file_path, compress=config.MODEL_FILE_COMPRESSION)


def save_feature_schema(feature_names, file_path):
    """Record the columns a saved model expects, in order."""
    _ensure_folder(file_path)
    with open(file_path, "w", encoding="utf-8") as schema:
        json.dump([str(name) for name in feature_names], schema, indent=config.JSON_INDENT)


def load_model(file_path):
    """Bring a saved model back into memory, ready to score."""
    return joblib.load(file_path)


def load_feature_schema(file_path):
    """Read back the columns a model expects, in their saved order."""
    with open(file_path, encoding="utf-8") as schema:
        return list(json.load(schema))


def save_metrics(metrics):
    """Leave every model's scores on disk, keys sorted."""
    _ensure_folder(config.MODEL_METRICS_FILE)
    with open(config.MODEL_METRICS_FILE, "w", encoding="utf-8") as report:
        json.dump(metrics, report, indent=config.JSON_INDENT, sort_keys=True)
