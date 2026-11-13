"""Screens an application's form against its own standard and the network triggers.

Every answer and figure is compared as an exact decimal, never as a binary
float, and every figure a requirement is judged against comes from the
standard handed over, never from the code.
"""

from decimal import Decimal

from config import config
from data_pipeline.common_cleaning import convert_power_to_amps_per_phase
from tools.application_data_tools import plain_value


def _answer(application, column):
    return plain_value(application.get(column))


def _decimal(application, column):
    value = _answer(application, column)
    return None if value is None else Decimal(str(value))


def _phases(application, column=config.EQUIPMENT_PHASES_COLUMN):
    value = _answer(application, column)
    return None if value is None else int(value)


def _technology(application):
    technology = _answer(application, config.TECHNOLOGY_COLUMN)
    if technology not in config.CONNECTION_STANDARD_DOCUMENTS:
        raise ValueError(config.UNKNOWN_TECHNOLOGY_MESSAGE.format(
            application_reference=_answer(application, config.APPLICATION_REFERENCE_COLUMN),
            technology=technology))
    return technology


# --------------------------------------------------------------------------- triggers
def _demand_that_counts(application):
    fitted = _answer(application, config.LOAD_LIMITER_FITTED_COLUMN)
    declared = _decimal(application, config.MAXIMUM_DEMAND_COLUMN)
    setting = _decimal(application, config.LOAD_LIMITER_SETTING_COLUMN)
    if fitted == config.BOOLEAN_YES:
        return setting
    if fitted == config.BOOLEAN_NO:
        return declared
    if declared is None or setting is None:
        return declared
    return max(declared, setting)


def _supply_capacity_reached(application):
    demand = _demand_that_counts(application)
    cut_out = _decimal(application, config.CUT_OUT_RATING_COLUMN)
    if demand is None or cut_out is None:
        return True
    return demand >= cut_out or demand > config.SUPPLY_MAXIMUM_AMPS


def _generation_above_notify_limit(application):
    inverter = _decimal(application, config.INVERTER_CAPACITY_COLUMN)
    existing = _decimal(application, config.EXISTING_GENERATION_COLUMN)
    phases = _phases(application)
    if inverter is None or existing is None or phases is None:
        return True
    capacity_amps = convert_power_to_amps_per_phase(inverter + existing, phases)
    if capacity_amps <= config.NOTIFY_GENERATION_AMPS:
        return False
    if capacity_amps > config.FAST_TRACK_GENERATION_AMPS:
        return True
    fitted = _answer(application, config.EXPORT_LIMITATION_FITTED_COLUMN)
    certified = _answer(application, config.EXPORT_SCHEME_CERTIFIED_COLUMN)
    limit = _decimal(application, config.EXPORT_LIMIT_COLUMN)
    if fitted == config.BOOLEAN_YES and certified == config.BOOLEAN_YES and limit is not None:
        if convert_power_to_amps_per_phase(limit, phases) <= config.NOTIFY_GENERATION_AMPS:
            return False
    return True


def find_network_triggers(application):
    """The names of the network triggers of the Low-Carbon Connections Policy that apply."""
    technology = _technology(application)
    triggers = []
    looped = _answer(application, config.LOOPED_SERVICE_COLUMN)
    if looped is None or looped == config.BOOLEAN_YES:
        triggers.append(config.LOOPED_SERVICE_TRIGGER)
    if _supply_capacity_reached(application):
        triggers.append(config.SUPPLY_CAPACITY_TRIGGER)
    if technology in config.GENERATING_TECHNOLOGIES and _generation_above_notify_limit(application):
        triggers.append(config.GENERATION_TRIGGER)
    return triggers


# --------------------------------------------------------------------------- requirements
def _judged(result, blank_answers=()):
    return result, list(blank_answers)


def _judge_yes_or_no(application, column):
    answer = _answer(application, column)
    if answer == config.BOOLEAN_YES:
        return _judged(config.REQUIREMENT_MET)
    if answer == config.BOOLEAN_NO:
        return _judged(config.REQUIREMENT_NOT_MET)
    return _judged(config.REQUIREMENT_NOT_DOCUMENTED, [column])


def _judge_at_most(application, column, maximum):
    value = _decimal(application, column)
    if value is None:
        return _judged(config.REQUIREMENT_NOT_DOCUMENTED, [column])
    return _judged(config.REQUIREMENT_MET if value <= maximum else config.REQUIREMENT_NOT_MET)


def _judge_power_factor(application, minimum):
    value = _decimal(application, config.POWER_FACTOR_COLUMN)
    if value is None:
        return _judged(config.REQUIREMENT_NOT_DOCUMENTED, [config.POWER_FACTOR_COLUMN])
    return _judged(config.REQUIREMENT_MET if value >= minimum else config.REQUIREMENT_NOT_MET)


def _judge_earthing(application, maximum_ohms):
    arrangement = _answer(application, config.EARTHING_ARRANGEMENT_COLUMN)
    if arrangement is None:
        return _judged(config.REQUIREMENT_NOT_DOCUMENTED, [config.EARTHING_ARRANGEMENT_COLUMN])
    if arrangement == config.EARTHING_TN_S:
        return _judged(config.REQUIREMENT_MET)
    if arrangement == config.EARTHING_TN_C_S:
        return _judge_yes_or_no(application, config.OPEN_PEN_PROTECTION_COLUMN)
    return _judge_at_most(application, config.EARTH_ELECTRODE_COLUMN, maximum_ohms)


def _judge_export_limitation(application, maximum_amps):
    fitted = _answer(application, config.EXPORT_LIMITATION_FITTED_COLUMN)
    if fitted is None:
        return _judged(config.REQUIREMENT_NOT_DOCUMENTED, [config.EXPORT_LIMITATION_FITTED_COLUMN])
    if fitted == config.BOOLEAN_NO:
        return _judged(config.REQUIREMENT_MET)
    certified = _answer(application, config.EXPORT_SCHEME_CERTIFIED_COLUMN)
    limit = _decimal(application, config.EXPORT_LIMIT_COLUMN)
    phases = _phases(application)
    limit_amps = None
    if limit is not None and phases is not None:
        limit_amps = convert_power_to_amps_per_phase(limit, phases)
    if certified == config.BOOLEAN_NO or (limit_amps is not None and limit_amps > maximum_amps):
        return _judged(config.REQUIREMENT_NOT_MET)
    blanks = []
    if certified is None:
        blanks.append(config.EXPORT_SCHEME_CERTIFIED_COLUMN)
    if limit is None:
        blanks.append(config.EXPORT_LIMIT_COLUMN)
    elif phases is None:
        blanks.append(config.EQUIPMENT_PHASES_COLUMN)
    if blanks:
        return _judged(config.REQUIREMENT_NOT_DOCUMENTED, blanks)
    return _judged(config.REQUIREMENT_MET)


def _judge_requirement(application, requirement, document_name):
    kind = requirement["requirement_kind"]
    rules = requirement["rules"]
    if kind in config.YES_OR_NO_REQUIREMENT_ANSWERS:
        return _judge_yes_or_no(application, config.YES_OR_NO_REQUIREMENT_ANSWERS[kind])
    if kind in config.CURRENT_REQUIREMENT_ANSWERS:
        return _judge_at_most(application, config.CURRENT_REQUIREMENT_ANSWERS[kind],
                              Decimal(str(rules[config.MAXIMUM_AMPS_RULE])))
    if kind == config.POWER_FACTOR_REQUIREMENT:
        return _judge_power_factor(application, Decimal(str(rules[config.MINIMUM_POWER_FACTOR_RULE])))
    if kind == config.EARTHING_PROTECTION_REQUIREMENT:
        return _judge_earthing(application, Decimal(str(rules[config.MAXIMUM_OHMS_RULE])))
    if kind == config.EXPORT_LIMITATION_REQUIREMENT:
        return _judge_export_limitation(application, Decimal(str(rules[config.MAXIMUM_AMPS_RULE])))
    raise ValueError(f"{document_name} states the requirement kind {kind}, which the screening "
                     f"does not know")


def screen_application(application, standard):
    """Screen one application against every requirement of its standard and the triggers."""
    reference = _answer(application, config.APPLICATION_REFERENCE_COLUMN)
    technology = _answer(application, config.TECHNOLOGY_COLUMN)
    if standard["technology"] != technology:
        raise ValueError(config.STANDARD_OF_ANOTHER_TECHNOLOGY_MESSAGE.format(
            application_reference=reference, technology=technology,
            document_name=standard["document_name"]))

    requirements = []
    for requirement in standard["requirements"]:
        result, blank_answers = _judge_requirement(application, requirement,
                                                   standard["document_name"])
        requirements.append({
            "requirement_kind": requirement["requirement_kind"],
            "requirement_name": requirement["requirement_name"],
            "requirement_sentence": requirement["requirement_sentence"],
            "rules": dict(requirement["rules"]),
            "result": result,
            "blank_answers": blank_answers,
        })

    counts = {column: sum(1 for requirement in requirements if requirement["result"] == result)
              for column, result in config.REQUIREMENT_COUNT_RESULTS.items()}
    triggers = find_network_triggers(application)
    rating_stated = _answer(application, config.RATING_COLUMNS[technology]) is not None
    if counts[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN] or \
            counts[config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN] or not rating_stated:
        assessment = config.DECISION_RETURN
    elif triggers:
        assessment = config.DECISION_REFER
    else:
        assessment = config.DECISION_APPROVE

    screening = {
        "application_reference": reference,
        "technology": standard["technology"],
        "standard_document": standard["document_name"],
        "requirements": requirements,
    }
    screening.update(counts)
    screening.update({
        "network_triggers": triggers,
        "rating_stated": rating_stated,
        "assessment": assessment,
        "decisions_open": list(config.DECISIONS_OPEN[assessment]),
    })
    return screening
