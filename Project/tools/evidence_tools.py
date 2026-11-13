"""Reads every sentence of an application's statements, links it to the requirements
it bears on, and flags every equipment rating that disagrees with the form."""

import re
from decimal import ROUND_HALF_UP, Decimal

from config import config
from data_pipeline.common_cleaning import convert_power_to_amps_per_phase
from tools.application_data_tools import plain_value


def read_document_evidence(sentence_model, documents):
    """Split each statement into its sentences and name each sentence's type."""
    pending = []
    for document in documents:
        text = plain_value(document.get(config.DOCUMENT_TEXT_COLUMN))
        if text is None or not str(text).strip():
            raise ValueError(config.NO_STATEMENT_TEXT_MESSAGE.format(
                document_id=document.get(config.DOCUMENT_ID_COLUMN)))
        for sentence in re.split(config.STATEMENT_SENTENCE_PATTERN, str(text)):
            if sentence:
                pending.append((document, sentence))
    if not pending:
        return []
    types = sentence_model.predict([sentence for _, sentence in pending])
    return [{
        "document_id": document.get(config.DOCUMENT_ID_COLUMN),
        "document_type": document.get(config.DOCUMENT_TYPE_COLUMN),
        "sentence": sentence,
        "sentence_type": str(sentence_type),
    } for (document, sentence), sentence_type in zip(pending, types)]


def link_evidence_to_requirements(findings, screening):
    """Record which requirements of the application's own standard each sentence bears on."""
    linked = []
    for finding in findings:
        sentence_type = finding["sentence_type"]
        if sentence_type not in config.SENTENCE_TYPE_REQUIREMENT_KINDS:
            raise ValueError(config.NO_REQUIREMENT_KINDS_MESSAGE.format(sentence_type=sentence_type))
        kinds = config.SENTENCE_TYPE_REQUIREMENT_KINDS[sentence_type]
        borne_on = [requirement["requirement_name"] for requirement in screening["requirements"]
                    if requirement["requirement_kind"] in kinds]
        linked.append({**finding, "requirements_borne_on": borne_on})
    return linked


def _form_rating_amps(application, phases):
    ratings = []
    for column in config.FORM_CURRENT_RATING_COLUMNS:
        value = plain_value(application.get(column))
        if value is not None:
            ratings.append(Decimal(str(value)))
    for column in config.FORM_POWER_RATING_COLUMNS:
        value = plain_value(application.get(column))
        if value is not None and phases is not None:
            ratings.append(convert_power_to_amps_per_phase(Decimal(str(value)), phases))
    return ratings


def _stated_amps(sentence, phases):
    stated = []
    for match in re.finditer(config.STATED_FIGURE_PATTERN, sentence):
        following = re.findall(config.FOLLOWING_WORD_PATTERN, sentence[match.end():].lower())
        if any(word in config.UNREAD_FIGURE_WORDS
               for word in following[:config.FOLLOWING_WORDS_CHECKED]):
            continue
        number = Decimal(match.group(config.STATED_NUMBER_PART).replace(config.THOUSANDS_SEPARATOR, ""))
        unit = match.group(config.STATED_UNIT_PART).lower()
        if unit in config.STATED_CURRENT_UNITS:
            stated.append(number.quantize(config.AMPS_STEP, rounding=ROUND_HALF_UP))
        elif phases is not None:
            stated.append(convert_power_to_amps_per_phase(number * config.STATED_POWER_UNITS[unit],
                                                          phases))
    return stated


def find_disagreements(findings, application):
    """Flag every equipment rating sentence whose figures are none of the form's ratings."""
    phases_value = plain_value(application.get(config.EQUIPMENT_PHASES_COLUMN))
    phases = None if phases_value is None else int(phases_value)
    form_amps = _form_rating_amps(application, phases)
    flags = []
    for finding in findings:
        if finding["sentence_type"] != config.EQUIPMENT_RATING:
            continue
        stated = _stated_amps(finding["sentence"], phases)
        if not stated:
            continue
        if not form_amps:
            flag = config.DESCRIBES_BLANK_ANSWER
        elif not any(amps in form_amps for amps in stated):
            flag = config.DISAGREES_WITH_FORM
        else:
            continue
        flags.append({
            "document_id": finding["document_id"],
            "sentence": finding["sentence"],
            "stated_amps": [float(amps) for amps in stated],
            "form_rating_amps": [float(amps) for amps in form_amps],
            "flag": flag,
        })
    return flags
