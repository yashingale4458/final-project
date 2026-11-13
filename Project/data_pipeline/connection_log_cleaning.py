"""The connection logs: the table the duration model learns from."""

from config import config
from data_pipeline import common_cleaning as cleaning


def read_log_answers(frame):
    """Read every answer of a log as the Connection Rules read it. No row is removed."""
    logs = cleaning.normalize_category_values(frame, config.LOG_VALUE_MAPPINGS)
    logs[config.APPLICATION_REFERENCE_COLUMN] = cleaning.convert_application_references(
        logs[config.APPLICATION_REFERENCE_COLUMN])
    logs = cleaning.refuse_unreadable_identifiers(logs, config.LOG_IDENTIFIER_PATTERNS)
    logs = cleaning.convert_whole_numbers(logs, config.LOG_WHOLE_NUMBER_PATTERNS)
    logs = cleaning.convert_phases(logs)
    logs = cleaning.treat_impossible_phase_pairs_as_missing(logs)
    logs = cleaning.convert_currents(logs, config.LOG_CURRENT_PHASES)
    logs = cleaning.convert_amounts(logs, config.LOG_AMOUNT_READINGS)
    return logs


def clean_connection_log_dataset(frame):
    """One row per application with the working days its connection really took."""
    logs = cleaning.remove_duplicate_records(frame)
    logs = cleaning.refuse_missing_values(logs, [config.APPLICATION_REFERENCE_COLUMN])
    logs = read_log_answers(logs)
    logs = cleaning.refuse_repeated_references(logs, config.APPLICATION_REFERENCE_COLUMN)
    logs = cleaning.treat_values_below_minimum_as_missing(logs, config.LOG_MINIMUM_VALUES)
    logs = cleaning.treat_values_outside_sizes_as_missing(logs, config.LOG_ALLOWED_SIZES)
    return cleaning.remove_rows_missing_target(logs, config.LOG_TARGET_COLUMN)
