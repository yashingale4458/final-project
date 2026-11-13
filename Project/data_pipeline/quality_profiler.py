"""Measures how dirty each raw extract was, and writes the evidence out.

Every figure is counted on the extract's first copies of each row as they
arrived, unless said otherwise.
"""

import json
import os
import re

import numpy as np
import pandas as pd

from config import config
from data_pipeline import common_cleaning as cleaning
from data_pipeline.application_cleaning import read_application_answers
from data_pipeline.connection_log_cleaning import read_log_answers


# --------------------------------------------------------------------------- helpers
def _plain(value):
    """Turn numpy and pandas scalars into plain Python values, recursively."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _first_copies(raw_frame):
    return raw_frame.drop_duplicates(keep="first").reset_index(drop=True)


def _missing_by_column(frame):
    counts = frame.isna().sum()
    return {column: int(count) for column, count in counts.items() if count > 0}


def _count_not_matching(values, pattern):
    given = values.dropna().astype(str)
    return int((~given.map(lambda value: bool(re.fullmatch(pattern, value)))).sum())


def _references_not_canonical(frame):
    return _count_not_matching(frame[config.APPLICATION_REFERENCE_COLUMN],
                               config.CANONICAL_APPLICATION_REFERENCE_PATTERN)


def _numbers_not_plain(frame, number_patterns):
    return {column: _count_not_matching(frame[column], config.PLAIN_NUMBER_PATTERN)
            for column in number_patterns if column in frame.columns}


def _phases_not_numbers(frame):
    return {column: _count_not_matching(frame[column], config.CANONICAL_PHASES_PATTERN)
            for column in config.PHASE_COLUMNS if column in frame.columns}


def _impossible_phase_pairs(frame):
    phases = cleaning.convert_phases(frame)
    supply = pd.to_numeric(phases[config.SUPPLY_PHASES_COLUMN], errors="coerce")
    equipment = pd.to_numeric(phases[config.EQUIPMENT_PHASES_COLUMN], errors="coerce")
    return int((supply.notna() & equipment.notna() & (equipment > supply)).sum())


def _currents_by_notation(frame, current_phases):
    report = {}
    for column in current_phases:
        if column not in frame.columns:
            continue
        counts = {notation: 0 for notation in config.CURRENT_NOTATIONS}
        for value in frame[column].dropna():
            notation = cleaning.current_notation(value)
            if notation is not None:
                counts[notation] += 1
        report[column] = counts
    return report


def _currents_unconvertible(raw, read, current_phases):
    return {column: int((raw[column].notna() & read[column].isna()).sum())
            for column in current_phases if column in raw.columns}


def _amounts_by_notation(frame, amount_readings):
    report = {}
    for column, reading in amount_readings.items():
        if column not in frame.columns:
            continue
        counts = {notation: 0 for notation in reading["notations"]}
        for value in frame[column].dropna():
            notation = cleaning.amount_notation(value, reading)
            if notation is not None:
                counts[notation] += 1
        report[column] = counts
    return report


def _beyond(read, limits, test):
    report = {}
    for column, limit in limits.items():
        if column in read.columns:
            numbers = pd.to_numeric(read[column], errors="coerce")
            report[column] = int((numbers.notna() & test(numbers, limit)).sum())
    return report


def _categories_not_canonical(frame, value_mappings):
    report = {}
    for column, mapping in value_mappings.items():
        if column not in frame.columns:
            continue
        given = frame[column].dropna().astype(str)
        canonical = given.str.lower().map(mapping)
        report[column] = int((canonical.astype(str) != given).sum())
    return report


def _target_shares(clean_frame, target_column):
    shares = clean_frame[target_column].value_counts(normalize=True)
    return {str(label): round(float(share), config.TARGET_SHARE_DECIMALS)
            for label, share in sorted(shares.items())}


# --------------------------------------------------------------------------- profiles
def profile_dataset(raw_frame, clean_frame, target_column):
    """Account for one dataset: what arrived, what survived, and what was missing."""
    first = _first_copies(raw_frame)
    return _plain({
        "rows_received": len(raw_frame),
        "rows_kept": len(clean_frame),
        "rows_removed": len(raw_frame) - len(clean_frame),
        "duplicate_rows": len(raw_frame) - len(first),
        "rows_missing_target": int(first[target_column].isna().sum()),
        "columns": len(clean_frame.columns),
        "missing_cells_as_received": _missing_by_column(first),
        "missing_cells_after_cleaning": _missing_by_column(clean_frame),
    })


def profile_history_details(raw_frame, clean_frame):
    """Account for the defects peculiar to the history extracts."""
    first = _first_copies(raw_frame)
    read = read_application_answers(first, config.HISTORY_VALUE_MAPPINGS,
                                    config.FORM_WHOLE_NUMBER_PATTERNS)
    reference = config.APPLICATION_REFERENCE_COLUMN
    revision = config.REVISION_NUMBER_COLUMN
    revisions = pd.to_numeric(read[revision], errors="coerce")
    highest = revisions.groupby(read[reference]).transform("max")
    earlier = revisions.notna() & (revisions < highest)
    answered_earlier = earlier & read[config.HISTORY_TARGET_COLUMN].notna()
    return _plain({
        "answered_earlier_revisions_removed": int(answered_earlier.sum()),
        "application_references_not_written_canonically": _references_not_canonical(first),
        "numbers_not_written_as_plain_numbers": _numbers_not_plain(
            first, config.FORM_WHOLE_NUMBER_PATTERNS),
        "phases_not_written_as_numbers": _phases_not_numbers(first),
        "impossible_phase_pairs_as_received": _impossible_phase_pairs(first),
        "currents_by_notation_as_written": _currents_by_notation(first, config.FORM_CURRENT_PHASES),
        "currents_unconvertible_without_phases": _currents_unconvertible(
            first, read, config.FORM_CURRENT_PHASES),
        "amounts_by_notation_as_written": _amounts_by_notation(first, config.FORM_AMOUNT_READINGS),
        "values_below_minimum_as_received": _beyond(read, config.FORM_MINIMUM_VALUES,
                                                    lambda numbers, limit: numbers < limit),
        "values_above_maximum_as_received": _beyond(read, config.FORM_MAXIMUM_VALUES,
                                                    lambda numbers, limit: numbers > limit),
        "values_outside_sizes_as_received": _beyond(read, config.FORM_ALLOWED_SIZES,
                                                    lambda numbers, sizes: ~numbers.isin(sizes)),
        "category_values_not_written_canonically": _categories_not_canonical(
            first, config.HISTORY_VALUE_MAPPINGS),
        "target_shares": _target_shares(clean_frame, config.HISTORY_TARGET_COLUMN),
    })


def profile_log_details(raw_frame, clean_frame):
    """Account for the defects peculiar to the log extracts."""
    first = _first_copies(raw_frame)
    read = read_log_answers(first)
    actual = pd.to_numeric(read[config.LOG_TARGET_COLUMN], errors="coerce")
    minimum = config.LOG_MINIMUM_VALUES[config.LOG_TARGET_COLUMN]
    return _plain({
        "rows_with_impossible_target": int((actual.notna() & (actual < minimum)).sum()),
        "values_below_minimum_as_received": _beyond(read, config.LOG_MINIMUM_VALUES,
                                                    lambda numbers, limit: numbers < limit),
        "application_references_not_written_canonically": _references_not_canonical(first),
        "numbers_not_written_as_plain_numbers": _numbers_not_plain(
            first, config.LOG_WHOLE_NUMBER_PATTERNS),
        "phases_not_written_as_numbers": _phases_not_numbers(first),
        "impossible_phase_pairs_as_received": _impossible_phase_pairs(first),
        "currents_by_notation_as_written": _currents_by_notation(first, config.LOG_CURRENT_PHASES),
        "currents_unconvertible_without_phases": _currents_unconvertible(
            first, read, config.LOG_CURRENT_PHASES),
        "amounts_by_notation_as_written": _amounts_by_notation(first, config.LOG_AMOUNT_READINGS),
        "values_outside_sizes_as_received": _beyond(read, config.LOG_ALLOWED_SIZES,
                                                    lambda numbers, sizes: ~numbers.isin(sizes)),
        "category_values_not_written_canonically": _categories_not_canonical(
            first, config.LOG_VALUE_MAPPINGS),
        "median_actual_working_days": float(
            pd.to_numeric(clean_frame[config.LOG_TARGET_COLUMN]).median()),
    })


def profile_sentence_details(raw_frame, clean_frame):
    """Account for the defects peculiar to free text."""
    first = _first_copies(raw_frame)
    raw_text = first[config.SENTENCE_TEXT_COLUMN]
    plain = cleaning.convert_markup_to_plain_text(raw_text)
    given = raw_text.notna()
    with_markup = raw_text[given].astype(str).map(
        lambda text: re.search(config.MARKUP_PATTERN, text) is not None)
    with_artifacts = raw_text[given].astype(str).map(
        lambda text: re.search(config.ENCODING_ARTIFACT_PATTERN, text) is not None)
    in_capitals = plain.dropna().map(lambda text: text.isupper())

    lengths = first[config.SENTENCE_LENGTH_COLUMN]
    disagreeing = 0
    for written, text in zip(lengths.tolist(), plain.tolist()):
        if pd.isna(written) or pd.isna(text):
            continue
        match = re.match(config.WHOLE_NUMBER_PATTERN, str(written))
        if match is None or int(match.group(1)) != len(text):
            disagreeing += 1

    word_counts = plain.map(lambda text: 0 if pd.isna(text) else len(text.split()))
    labelled = first[config.SENTENCE_TARGET_COLUMN].notna()
    too_few = labelled & (word_counts < config.MINIMUM_SENTENCE_WORDS)
    clean_words = clean_frame[config.SENTENCE_TEXT_COLUMN].map(lambda text: len(str(text).split()))
    return _plain({
        "sentences_with_markup": int(with_markup.sum()),
        "sentences_with_encoding_artifacts": int(with_artifacts.sum()),
        "sentences_in_capitals": int(in_capitals.sum()),
        "sentence_lengths_disagreeing_with_plain_text_as_received": disagreeing,
        "rows_with_too_few_words": int(too_few.sum()),
        "shortest_sentence_words": int(clean_words.min()) if len(clean_words) else 0,
        "application_references_not_written_canonically": _references_not_canonical(first),
        "category_values_not_written_canonically": _categories_not_canonical(
            first, config.SENTENCE_VALUE_MAPPINGS),
        "target_shares": _target_shares(clean_frame, config.SENTENCE_TARGET_COLUMN),
    })


def write_quality_report(profiles):
    """Write every dataset's profile to the quality report, keys sorted."""
    folder = os.path.dirname(config.DATA_QUALITY_REPORT_FILE)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(config.DATA_QUALITY_REPORT_FILE, "w", encoding="utf-8") as report:
        json.dump(_plain(profiles), report, indent=config.JSON_INDENT, sort_keys=True,
                  ensure_ascii=False)
