"""Cleaning of statement rows: past sentences and live statements alike."""

from config import config
from data_pipeline import common_cleaning as cleaning


def clean_document_records(frame, value_mappings, whole_number_patterns, id_column,
                           text_column):
    """Clean the rows of statements, whether past sentences or live statements."""
    rows = cleaning.remove_duplicate_records(frame)
    rows = cleaning.refuse_missing_values(rows, [id_column, config.APPLICATION_REFERENCE_COLUMN])
    rows = cleaning.refuse_repeated_references(rows, id_column)
    rows = cleaning.normalize_category_values(rows, value_mappings)
    rows[config.APPLICATION_REFERENCE_COLUMN] = cleaning.convert_application_references(
        rows[config.APPLICATION_REFERENCE_COLUMN])
    rows = cleaning.convert_whole_numbers(rows, whole_number_patterns)
    rows[text_column] = cleaning.clean_statement_text(rows[text_column])
    return rows
