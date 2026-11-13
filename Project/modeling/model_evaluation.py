"""Scores a fitted model on the training data and on held-out data, never refitting it."""

import numpy as np
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, mean_absolute_error,
                             r2_score, roc_auc_score)

from config import config


def _round(value):
    return round(float(value), config.METRIC_DECIMALS)


def _positive_probabilities(model, features):
    classes = list(model.classes_)
    return model.predict_proba(features)[:, classes.index(1)]


def evaluate_classifier(model, train_features, train_target, held_out_features, held_out_target):
    """How well the network work model ranks unseen applications, and its calibration."""
    train_scores = _positive_probabilities(model, train_features)
    held_out_scores = _positive_probabilities(model, held_out_features)
    train_auc = roc_auc_score(train_target, train_scores)
    held_out_auc = roc_auc_score(held_out_target, held_out_scores)
    return {
        "train_roc_auc": _round(train_auc),
        "held_out_roc_auc": _round(held_out_auc),
        "brier_score": _round(brier_score_loss(held_out_target, held_out_scores)),
        "mean_predicted_probability": _round(np.mean(held_out_scores)),
        "positive_share": _round(np.mean(np.asarray(held_out_target) == 1)),
        "generalisation_gap": _round(train_auc - held_out_auc),
    }


def evaluate_regressor(model, train_features, train_target, held_out_features, held_out_target):
    """How close the duration model is on unseen connections, and whether it beats guessing."""
    train_predictions = model.predict(train_features)
    held_out_predictions = model.predict(held_out_features)
    train_r2 = r2_score(train_target, train_predictions)
    held_out_r2 = r2_score(held_out_target, held_out_predictions)
    average = float(np.mean(train_target))
    return {
        "train_r2": _round(train_r2),
        "held_out_r2": _round(held_out_r2),
        "mean_absolute_error": _round(mean_absolute_error(held_out_target, held_out_predictions)),
        "average_estimate_mean_absolute_error": _round(
            np.mean(np.abs(np.asarray(held_out_target, dtype=float) - average))),
        "lowest_prediction": _round(np.min(held_out_predictions)),
        "highest_prediction": _round(np.max(held_out_predictions)),
        "generalisation_gap": _round(train_r2 - held_out_r2),
    }


def evaluate_text_classifier(model, train_text, train_target, held_out_text, held_out_target):
    """How well the sentence model names every type of sentence, not only the common ones."""
    train_predictions = model.predict(list(train_text))
    held_out_predictions = model.predict(list(held_out_text))
    train_f1 = f1_score(train_target, train_predictions, average="macro")
    held_out_f1 = f1_score(held_out_target, held_out_predictions, average="macro")
    classes = list(model.classes_)
    per_class = f1_score(held_out_target, held_out_predictions, labels=classes, average=None,
                         zero_division=0)
    return {
        "train_macro_f1": _round(train_f1),
        "held_out_macro_f1": _round(held_out_f1),
        "accuracy": _round(accuracy_score(held_out_target, held_out_predictions)),
        "weighted_f1": _round(f1_score(held_out_target, held_out_predictions, average="weighted")),
        "generalisation_gap": _round(train_f1 - held_out_f1),
        "per_class_f1": {str(label): _round(score) for label, score in zip(classes, per_class)},
    }
