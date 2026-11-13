"""Entry point 3: trains the three models in one run, and saves them only when every
one reaches the minimum the configuration sets for it.

    python run_model_training.py
"""

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from config import config
from data_pipeline.dataset_loader import load_processed_dataset
from data_pipeline.feature_builder import (add_network_trigger_flags, add_requirement_counts,
                                           build_feature_frame)
from knowledge.pdf_extraction import extract_text_from_directory
from knowledge.standard_reading import read_connection_standard
from modeling.duration_model import train_duration_model
from modeling.model_registry import save_feature_schema, save_metrics, save_model
from modeling.network_work_model import train_network_work_model
from modeling.sentence_model import train_sentence_model


def _encode_outcomes(outcomes):
    unknown = sorted(set(outcomes.dropna()) - set(config.NETWORK_WORK_TARGET_ENCODING))
    if unknown or outcomes.isna().any():
        raise ValueError(f"Column {config.HISTORY_TARGET_COLUMN} holds outcomes the encoding does "
                         f"not know: {', '.join(map(str, unknown)) or 'a blank outcome'}")
    return outcomes.map(config.NETWORK_WORK_TARGET_ENCODING).astype(int)


def _network_work_frame(file_path, standards):
    history = load_processed_dataset(file_path)
    history = add_requirement_counts(history, standards)
    history = add_network_trigger_flags(history)
    features = build_feature_frame(history, config.NETWORK_WORK_NUMERIC_FEATURES,
                                   config.NETWORK_WORK_CATEGORICAL_FEATURES)
    return features, _encode_outcomes(history[config.HISTORY_TARGET_COLUMN])


def _duration_frame(file_path):
    logs = add_network_trigger_flags(load_processed_dataset(file_path))
    features = build_feature_frame(logs, config.DURATION_NUMERIC_FEATURES,
                                   config.DURATION_CATEGORICAL_FEATURES)
    return features, pd.to_numeric(logs[config.LOG_TARGET_COLUMN]).astype(float)


def _algorithm(model, every_step=False):
    if every_step:
        return " + ".join(type(step).__name__ for _, step in model.steps)
    return type(model.named_steps[config.MODEL_STEP_NAME]).__name__


def _record(model, train_rows, held_out_rows, scores, minimum, every_step=False):
    return {"algorithm": _algorithm(model, every_step), "training_rows": int(train_rows),
            "held_out_rows": int(held_out_rows), "scores": scores, "minimum_required": minimum,
            "scikit_learn_version": sklearn.__version__}


def main():
    """Train, score and, when every minimum is reached, save the three models."""
    print("training three models")
    documents = extract_text_from_directory(config.DOCS_DIRECTORY)
    standards = {technology: read_connection_standard(technology, documents)
                 for technology in config.CONNECTION_STANDARD_DOCUMENTS}

    network_train, network_train_target = _network_work_frame(
        config.CLEAN_APPLICATION_HISTORY_TRAIN_FILE, standards)
    network_held, network_held_target = _network_work_frame(
        config.CLEAN_APPLICATION_HISTORY_EVAL_FILE, standards)
    network_model, network_scores = train_network_work_model(
        network_train, network_train_target, network_held, network_held_target)

    duration_train, duration_train_target = _duration_frame(config.CLEAN_CONNECTION_LOGS_TRAIN_FILE)
    duration_held, duration_held_target = _duration_frame(config.CLEAN_CONNECTION_LOGS_EVAL_FILE)
    duration_model, duration_scores = train_duration_model(
        duration_train, duration_train_target, duration_held, duration_held_target)

    sentences = load_processed_dataset(config.CLEAN_STATEMENT_SENTENCES_FILE)
    train_text, held_text, train_types, held_types = train_test_split(
        sentences[config.SENTENCE_TEXT_COLUMN].astype(str), sentences[config.SENTENCE_TARGET_COLUMN],
        test_size=config.SENTENCE_HELD_OUT_SHARE, stratify=sentences[config.SENTENCE_TARGET_COLUMN],
        random_state=config.RANDOM_SEED)
    sentence_model, sentence_scores = train_sentence_model(train_text, train_types, held_text,
                                                           held_types)

    print(f"  network work    held-out ROC AUC {network_scores['held_out_roc_auc']:.4f}  "
          f"(train {network_scores['train_roc_auc']:.4f}, "
          f"gap {network_scores['generalisation_gap']:+.4f})")
    print(f"  duration        held-out R2 {duration_scores['held_out_r2']:.4f}  "
          f"MAE {duration_scores['mean_absolute_error']:.4f} working days  "
          f"(train {duration_scores['train_r2']:.4f}, gap {duration_scores['generalisation_gap']:+.4f})")
    print(f"  sentence        held-out macro F1 {sentence_scores['held_out_macro_f1']:.4f}  "
          f"(train {sentence_scores['train_macro_f1']:.4f}, "
          f"gap {sentence_scores['generalisation_gap']:+.4f})")

    shortfalls = []
    if network_scores["held_out_roc_auc"] < config.MINIMUM_NETWORK_WORK_ROC_AUC:
        shortfalls.append(f"network work model held-out ROC AUC {network_scores['held_out_roc_auc']} "
                          f"is below {config.MINIMUM_NETWORK_WORK_ROC_AUC}")
    if duration_scores["held_out_r2"] < config.MINIMUM_DURATION_R2:
        shortfalls.append(f"duration model held-out R2 {duration_scores['held_out_r2']} is below "
                          f"{config.MINIMUM_DURATION_R2}")
    if sentence_scores["held_out_macro_f1"] < config.MINIMUM_SENTENCE_MACRO_F1:
        shortfalls.append(f"sentence model held-out macro F1 {sentence_scores['held_out_macro_f1']} "
                          f"is below {config.MINIMUM_SENTENCE_MACRO_F1}")
    if shortfalls:
        raise SystemExit("No model was saved: " + "; ".join(shortfalls))

    save_model(network_model, config.NETWORK_WORK_MODEL_FILE)
    save_model(duration_model, config.DURATION_MODEL_FILE)
    save_model(sentence_model, config.SENTENCE_MODEL_FILE)
    save_feature_schema(list(network_model.feature_names_in_), config.NETWORK_WORK_SCHEMA_FILE)
    save_feature_schema(list(duration_model.feature_names_in_), config.DURATION_SCHEMA_FILE)
    save_metrics({
        "network_work": _record(network_model, len(network_train), len(network_held),
                                network_scores, config.MINIMUM_NETWORK_WORK_ROC_AUC),
        "duration": _record(duration_model, len(duration_train), len(duration_held),
                            duration_scores, config.MINIMUM_DURATION_R2),
        "sentence": _record(sentence_model, len(train_text), len(held_text), sentence_scores,
                            config.MINIMUM_SENTENCE_MACRO_F1, every_step=True),
    })
    print(f"metrics written to {config.MODEL_METRICS_FILE}")
    print("every model reaches the minimum the configuration sets for it")


if __name__ == "__main__":
    main()
