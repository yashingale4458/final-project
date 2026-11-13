"""Entry point 2: cleans the five raw extracts and writes the evidence that each one
needed cleaning.

    python run_data_pipeline.py
"""

import os

from config import config
from data_pipeline.connection_log_cleaning import clean_connection_log_dataset
from data_pipeline.dataset_loader import load_raw_dataset, save_processed_dataset
from data_pipeline.history_cleaning import clean_history_dataset
from data_pipeline.quality_profiler import (profile_dataset, profile_history_details,
                                            profile_log_details, profile_sentence_details,
                                            write_quality_report)
from data_pipeline.sentence_cleaning import clean_sentence_dataset

DATASET_JOBS = [
    (config.RAW_APPLICATION_HISTORY_TRAIN_FILE, config.CLEAN_APPLICATION_HISTORY_TRAIN_FILE,
     clean_history_dataset, config.HISTORY_TARGET_COLUMN, profile_history_details),
    (config.RAW_APPLICATION_HISTORY_EVAL_FILE, config.CLEAN_APPLICATION_HISTORY_EVAL_FILE,
     clean_history_dataset, config.HISTORY_TARGET_COLUMN, profile_history_details),
    (config.RAW_CONNECTION_LOGS_TRAIN_FILE, config.CLEAN_CONNECTION_LOGS_TRAIN_FILE,
     clean_connection_log_dataset, config.LOG_TARGET_COLUMN, profile_log_details),
    (config.RAW_CONNECTION_LOGS_EVAL_FILE, config.CLEAN_CONNECTION_LOGS_EVAL_FILE,
     clean_connection_log_dataset, config.LOG_TARGET_COLUMN, profile_log_details),
    (config.RAW_STATEMENT_SENTENCES_FILE, config.CLEAN_STATEMENT_SENTENCES_FILE,
     clean_sentence_dataset, config.SENTENCE_TARGET_COLUMN, profile_sentence_details),
]


def _dataset_name(raw_file):
    return os.path.splitext(os.path.basename(raw_file))[0]


def main():
    """Clean all five extracts first, then save them and the quality report."""
    cleaned, profiles = [], {}
    for raw_file, clean_file, clean, target_column, profile_details in DATASET_JOBS:
        raw = load_raw_dataset(raw_file)
        clean_frame = clean(raw)
        profile = profile_dataset(raw, clean_frame, target_column)
        profile.update(profile_details(raw, clean_frame))
        profiles[_dataset_name(raw_file)] = profile
        cleaned.append((clean_file, clean_frame, profile))

    for clean_file, clean_frame, _ in cleaned:
        save_processed_dataset(clean_frame, clean_file)
    write_quality_report(profiles)

    width, rows = config.DATASET_NAME_WIDTH, config.ROW_COUNT_WIDTH
    for (raw_file, *_), (_, _, profile) in zip(DATASET_JOBS, cleaned):
        print(f"{_dataset_name(raw_file):<{width}} {profile['rows_received']:>{rows}} received -> "
              f"{profile['rows_kept']:>{rows}} kept  ({profile['rows_removed']} removed)")
    print(f"quality report written to {config.DATA_QUALITY_REPORT_FILE}")


if __name__ == "__main__":
    main()
