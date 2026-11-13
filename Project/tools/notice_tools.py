"""The details the notices and the assessment brief carry, taken by code from the
application and its screening; never worked out by a model."""

from config import config
from tools.application_data_tools import plain_value


def _refuse_other_screening(application, screening):
    reference = application.get(config.APPLICATION_REFERENCE_COLUMN)
    if screening["application_reference"] != reference:
        raise ValueError(config.SCREENING_OF_ANOTHER_APPLICATION_MESSAGE.format(
            screening_reference=screening["application_reference"],
            application_reference=reference))


def _answer(application, column):
    return plain_value(application.get(column))


def _conditions(application, technology):
    conditions = {}
    for column in config.APPROVAL_CONDITION_COLUMNS[technology]:
        value = _answer(application, column)
        if value is not None:
            conditions[column] = value
    return conditions


def build_approval_details(application, screening):
    """The details a connection approval shows, as the standards' section 7 sets out."""
    _refuse_other_screening(application, screening)
    technology = screening["technology"]
    details = {
        config.APPLICATION_REFERENCE_COLUMN: screening["application_reference"],
        "standard_document": screening["standard_document"],
    }
    for column in config.APPROVAL_DETAIL_COLUMNS:
        details[column] = _answer(application, column)
    rating_column = config.RATING_COLUMNS[technology]
    details[rating_column] = _answer(application, rating_column)
    details.update(_conditions(application, technology))
    return details


def build_household_notice_details(application, screening):
    """The only facts a household notice may rest on."""
    _refuse_other_screening(application, screening)
    technology = screening["technology"]
    rating_column = config.RATING_COLUMNS[technology]
    details = {config.TECHNOLOGY_COLUMN: technology,
               rating_column: _answer(application, rating_column)}
    details.update(_conditions(application, technology))
    return details


def build_rework_details(application, screening, documentation_to_provide, disagreements):
    """Everything a rework request must name."""
    _refuse_other_screening(application, screening)
    to_put_right = []
    for requirement in screening["requirements"]:
        if requirement["result"] == config.REQUIREMENT_MET:
            continue
        columns = config.REQUIREMENT_ANSWER_COLUMNS[requirement["requirement_kind"]]
        to_put_right.append({
            "requirement_name": requirement["requirement_name"],
            "requirement_sentence": requirement["requirement_sentence"],
            "result": requirement["result"],
            "form_answers": {column: _answer(application, column) for column in columns},
            "blank_answers": list(requirement["blank_answers"]),
        })
    technology = screening["technology"]
    return {
        config.APPLICATION_REFERENCE_COLUMN: screening["application_reference"],
        config.TECHNOLOGY_COLUMN: technology,
        "standard_document": screening["standard_document"],
        "requirements_to_put_right": to_put_right,
        "blank_rating_answer": None if screening["rating_stated"] else config.RATING_COLUMNS[technology],
        "documentation_to_provide": documentation_to_provide,
        "flagged_statements": list(disagreements),
    }


def build_assessment_brief(application, screening, likelihood, duration, disagreements):
    """The brief Network Planning receives with a referred application, from the record."""
    _refuse_other_screening(application, screening)
    brief = {
        config.APPLICATION_REFERENCE_COLUMN: screening["application_reference"],
        config.TECHNOLOGY_COLUMN: screening["technology"],
        "standard_document": screening["standard_document"],
    }
    for column in config.ASSESSMENT_BRIEF_DETAIL_COLUMNS:
        brief[column] = _answer(application, column)
    brief.update({
        "requirements": [{"requirement_name": requirement["requirement_name"],
                          "result": requirement["result"],
                          "blank_answers": list(requirement["blank_answers"])}
                         for requirement in screening["requirements"]],
        "network_triggers": list(screening["network_triggers"]),
        "flagged_statements": list(disagreements),
        "network_work_likelihood": likelihood["network_work_likelihood"],
        "likelihood_band": likelihood["likelihood_band"],
        "expected_working_days": duration["expected_working_days"],
        "planned_working_days": duration["planned_working_days"],
        "delay_expected": duration["delay_expected"],
    })
    return brief
