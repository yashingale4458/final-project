"""The repairs an application's form needs, shared by the history and the live queue."""

from config import config
from data_pipeline import common_cleaning as cleaning


def read_application_answers(frame, value_mappings, whole_number_patterns):
    """Read every answer on a form as the Connection Rules read it. No row is removed."""
    forms = cleaning.normalize_category_values(frame, value_mappings)
    forms[config.APPLICATION_REFERENCE_COLUMN] = cleaning.convert_application_references(
        forms[config.APPLICATION_REFERENCE_COLUMN])
    forms = cleaning.refuse_unreadable_identifiers(forms, config.FORM_IDENTIFIER_PATTERNS)
    forms = cleaning.convert_whole_numbers(forms, whole_number_patterns)
    forms = cleaning.convert_phases(forms)
    forms = cleaning.treat_impossible_phase_pairs_as_missing(forms)
    forms = cleaning.convert_currents(forms, config.FORM_CURRENT_PHASES)
    forms = cleaning.convert_amounts(forms, config.FORM_AMOUNT_READINGS)
    return forms


def clean_application_records(frame, value_mappings, whole_number_patterns):
    """Turn raw application forms into one clean row per application."""
    forms = cleaning.remove_duplicate_records(frame)
    forms = read_application_answers(forms, value_mappings, whole_number_patterns)
    forms = cleaning.treat_values_below_minimum_as_missing(forms, config.FORM_MINIMUM_VALUES)
    forms = cleaning.treat_values_above_maximum_as_missing(forms, config.FORM_MAXIMUM_VALUES)
    forms = cleaning.treat_values_outside_sizes_as_missing(forms, config.FORM_ALLOWED_SIZES)
    return cleaning.keep_latest_revision_of_each_application(forms)
