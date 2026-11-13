"""Reads the live queue and its statements, the connection history, and serves one
application at a time, through the same cleaning as the extracts."""

import math

import numpy as np
import pandas as pd

from config import config
from data_pipeline.application_cleaning import clean_application_records
from data_pipeline.dataset_loader import load_processed_dataset, load_raw_dataset
from data_pipeline.document_cleaning import clean_document_records


def plain_value(value):
    """One table value as a plain Python value, a missing value as None."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _record(row):
    return {column: plain_value(value) for column, value in row.items()}


def load_open_applications():
    """Read the live queue once, as the Connection Rules read it."""
    raw = load_raw_dataset(config.OPEN_APPLICATIONS_FILE)
    return clean_application_records(raw, config.FORM_VALUE_MAPPINGS,
                                     config.LIVE_WHOLE_NUMBER_PATTERNS)


def load_application_documents():
    """Read the statements attached to the open applications once."""
    raw = load_raw_dataset(config.APPLICATION_DOCUMENTS_FILE)
    return clean_document_records(raw, config.DOCUMENT_VALUE_MAPPINGS,
                                  config.DOCUMENT_WHOLE_NUMBER_PATTERNS,
                                  config.DOCUMENT_ID_COLUMN, config.DOCUMENT_TEXT_COLUMN)


def load_connection_history():
    """Every application already connected: the cleaned history, train then eval."""
    train = load_processed_dataset(config.CLEAN_APPLICATION_HISTORY_TRAIN_FILE)
    held_out = load_processed_dataset(config.CLEAN_APPLICATION_HISTORY_EVAL_FILE)
    return pd.concat([train, held_out], ignore_index=True)


def fetch_application(applications, application_reference):
    """One application's own answers, column by column, a missing value as None."""
    rows = applications[applications[config.APPLICATION_REFERENCE_COLUMN] == application_reference]
    if rows.empty:
        raise ValueError(config.UNKNOWN_APPLICATION_MESSAGE.format(
            application_reference=application_reference))
    return _record(rows.iloc[0].to_dict())


def fetch_application_documents(documents, application):
    """The statements attached to one application, in document_id order."""
    reference = application[config.APPLICATION_REFERENCE_COLUMN]
    rows = documents[documents[config.APPLICATION_REFERENCE_COLUMN] == reference]
    rows = rows.sort_values(config.DOCUMENT_ID_COLUMN, kind="mergesort")
    return [_record(row) for row in rows.to_dict(orient="records")]


def rank_application_queue(applications):
    """Heat pumps replacing failed heating first, then everything else; each by reference."""
    reference = config.APPLICATION_REFERENCE_COLUMN
    urgent = (applications[config.TECHNOLOGY_COLUMN] == config.HEAT_PUMP) & \
             (applications[config.URGENCY_COLUMN] == config.URGENCY_HEATING_FAILED)
    first = sorted(applications.loc[urgent, reference].tolist())
    rest = sorted(applications.loc[~urgent, reference].tolist())
    return [str(value) for value in first + rest]


def _describe_value(value):
    value = plain_value(value)
    if value is None:
        return config.NOT_GIVEN_TEXT
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    return str(value)


def describe_record(record):
    """A record in the words a page and a prompt show; every value as text."""
    return {key: _describe_value(value) for key, value in record.items()}
