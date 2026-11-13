"""The requirement counts, the network trigger flags, the model inputs in their saved
order, and the preparation every tabular model is fitted with."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import config


def _row_records(dataframe):
    return dataframe.to_dict(orient="records")


def add_requirement_counts(dataframe, standards):
    """Give every past application the counts its own standard's screening gives it."""
    from tools.screening_tools import screen_application

    counts = {column: [] for column in config.REQUIREMENT_COUNT_RESULTS}
    for row in _row_records(dataframe):
        technology = row.get(config.TECHNOLOGY_COLUMN)
        if technology not in standards:
            reference = row.get(config.APPLICATION_REFERENCE_COLUMN)
            raise ValueError(f"Application {reference}: " +
                             config.NO_STANDARD_CONFIGURED_MESSAGE.format(technology=technology))
        screening = screen_application(row, standards[technology])
        for column in counts:
            counts[column].append(screening[column])
    frame = dataframe.copy()
    for column, values in counts.items():
        frame[column] = pd.Series(values, index=frame.index, dtype="int64")
    return frame


def add_network_trigger_flags(dataframe):
    """Flag every network trigger each row meets, as the screening finds it."""
    from tools.screening_tools import find_network_triggers

    flags = {column: [] for column in config.NETWORK_TRIGGER_FLAG_COLUMNS.values()}
    for row in _row_records(dataframe):
        triggers = find_network_triggers(row)
        for trigger, column in config.NETWORK_TRIGGER_FLAG_COLUMNS.items():
            flags[column].append(int(trigger in triggers))
    frame = dataframe.copy()
    for column, values in flags.items():
        frame[column] = pd.Series(values, index=frame.index, dtype="int64")
    return frame


def build_feature_frame(dataframe, numeric_columns, categorical_columns):
    """Lay a table out exactly as a model is fitted and scored on it."""
    frame = pd.DataFrame(index=dataframe.index)
    for column in numeric_columns:
        frame[column] = pd.to_numeric(dataframe[column], errors="coerce").astype(float)
    for column in categorical_columns:
        values = dataframe[column].astype(object)
        frame[column] = values.map(
            lambda value: value if isinstance(value, str) else (np.nan if pd.isna(value) else str(value)))
        frame[column] = frame[column].astype(object)
    return frame


def build_model_pipeline(estimator, numeric_columns, categorical_columns):
    """Join the input preparation and the estimator into one fittable, saveable object."""
    numbers = Pipeline([
        ("fill", SimpleImputer(strategy=config.NUMBER_FILL_STRATEGY, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    words = Pipeline([
        ("fill", SimpleImputer(strategy=config.CATEGORY_FILL_STRATEGY, missing_values=np.nan,
                               keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preparation = ColumnTransformer([
        ("numbers", numbers, list(numeric_columns)),
        ("words", words, list(categorical_columns)),
    ])
    return Pipeline([(config.PREPROCESSING_STEP_NAME, preparation),
                     (config.MODEL_STEP_NAME, estimator)])
