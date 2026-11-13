"""The network work likelihood and the expected working days of one application."""

from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from config import config
from data_pipeline.feature_builder import build_feature_frame
from tools.application_data_tools import plain_value


def _refuse_other_screening(application, screening):
    reference = application.get(config.APPLICATION_REFERENCE_COLUMN)
    if screening["application_reference"] != reference:
        raise ValueError(config.SCREENING_OF_ANOTHER_APPLICATION_MESSAGE.format(
            screening_reference=screening["application_reference"],
            application_reference=reference))


def _trigger_flags(screening):
    return {column: int(trigger in screening["network_triggers"])
            for trigger, column in config.NETWORK_TRIGGER_FLAG_COLUMNS.items()}


def _inputs(application, extra, numeric_columns, categorical_columns, feature_names):
    row = {column: plain_value(value) for column, value in application.items()}
    row.update(extra)
    frame = build_feature_frame(pd.DataFrame([row]), numeric_columns, categorical_columns)
    if list(frame.columns) != list(feature_names):
        raise ValueError(f"The model inputs {list(frame.columns)} are not the saved schema "
                         f"{list(feature_names)}")
    return frame


def estimate_network_work_likelihood(network_work_model, feature_names, application, screening):
    """The probability of network work, rounded half up, and its band."""
    _refuse_other_screening(application, screening)
    extra = {column: screening[column] for column in config.REQUIREMENT_COUNT_RESULTS}
    extra.update(_trigger_flags(screening))
    frame = _inputs(application, extra, config.NETWORK_WORK_NUMERIC_FEATURES,
                    config.NETWORK_WORK_CATEGORICAL_FEATURES, feature_names)
    classes = list(network_work_model.classes_)
    probability = network_work_model.predict_proba(frame)[0][classes.index(1)]
    likelihood = Decimal(repr(float(probability))).quantize(config.LIKELIHOOD_STEP,
                                                             rounding=ROUND_HALF_UP)
    if likelihood >= config.HIGH_LIKELIHOOD_MINIMUM:
        band = config.HIGH_LIKELIHOOD_BAND
    elif likelihood >= config.MEDIUM_LIKELIHOOD_MINIMUM:
        band = config.MEDIUM_LIKELIHOOD_BAND
    else:
        band = config.LOW_LIKELIHOOD_BAND
    return {"network_work_likelihood": float(likelihood), "likelihood_band": band}


def estimate_expected_working_days(duration_model, feature_names, application, screening):
    """The working days the connection is expected to take, and whether a delay is expected."""
    _refuse_other_screening(application, screening)
    frame = _inputs(application, _trigger_flags(screening), config.DURATION_NUMERIC_FEATURES,
                    config.DURATION_CATEGORICAL_FEATURES, feature_names)
    forecast = float(duration_model.predict(frame)[0])
    expected = int(Decimal(repr(forecast)).quantize(config.WORKING_DAY_STEP, rounding=ROUND_HALF_UP))
    planned_value = plain_value(application.get(config.PLANNED_WORKING_DAYS_COLUMN))
    planned = None if planned_value is None else int(planned_value)
    delay = None if planned is None else bool(expected - planned > config.DELAY_MARGIN_WORKING_DAYS)
    return {"expected_working_days": expected, "planned_working_days": planned,
            "delay_expected": delay}


def describe_expected_working_days(duration):
    """The expected working days in words."""
    if duration["planned_working_days"] is None:
        template = config.EXPECTED_DAYS_WITHOUT_PLAN_TEMPLATE
    elif duration["delay_expected"]:
        template = config.DELAY_TEMPLATE
    else:
        template = config.NO_DELAY_TEMPLATE
    return template.format(expected_working_days=duration["expected_working_days"],
                           planned_working_days=duration["planned_working_days"])
