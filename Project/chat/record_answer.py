"""Answers from the records, never from a model, so that a model cannot give away what the
rules withhold."""

import re

from config import config
from persistence.models import ApplicationReview
from persistence.repository import fetch_review_records
from tools.application_data_tools import describe_record


def canonical_reference(reference):
    """A reference the pattern reads, once put in upper case, made canonical; any other as written."""
    match = re.match(config.APPLICATION_REFERENCE_PATTERN, reference.strip().upper())
    if match is None:
        return reference
    return f"{config.APPLICATION_REFERENCE_PREFIX}{int(match.group(1)):0{config.APPLICATION_REFERENCE_DIGITS}d}"


def _advisor_answer(review, technology, reference):
    status = config.REVIEW_STATUS_PHRASES[review["status"]] if review else config.CHAT_AWAITING_REVIEW_TEXT
    answer = config.CHAT_ADVISOR_STATUS_TEMPLATE.format(
        application_reference=reference,
        technology_application=config.TECHNOLOGY_APPLICATION_PHRASES[technology],
        status=status)
    if review and review["status"] == config.REVIEW_STATUS_APPROVED:
        details = review.get("notice_details") or {}
        conditions = [template.format(equipment_name=config.EQUIPMENT_NAMES[technology],
                                      value=describe_record({column: details[column]})[column])
                      for column, template in config.HOUSEHOLD_CONDITION_TEMPLATES.items()
                      if details.get(column) is not None]
        answer += " " + (" ".join(conditions) if conditions else config.CHAT_NO_CONDITIONS_TEXT)
    return answer


def _review_lines(review):
    requirements = "; ".join(f"{requirement['requirement_name']} {requirement['result']}"
                             for requirement in review["requirements"])
    counts = {}
    for finding in review["evidence"]:
        counts[finding["sentence_type"]] = counts.get(finding["sentence_type"], 0) + 1
    sentence_types = ", ".join(f"{name} {count}" for name, count in sorted(counts.items())) \
        or config.CHAT_NO_STATEMENTS_TEXT
    if review["decision"] is None:
        closing = config.CHAT_AWAITING_DECISION_TEMPLATE.format(
            recommended_decision=review["recommended_decision"])
    else:
        closing = config.CHAT_DECIDED_TEMPLATE.format(
            decided_by=review["decided_by"], decided_by_role=review["decided_by_role"],
            decision=review["decision"], decision_reason=review["decision_reason"])
    return [
        config.CHAT_REVIEW_TEMPLATE.format(
            review_id=review["review_id"], application_reference=review["application_reference"],
            technology_label=config.TECHNOLOGY_LABELS[review["technology"]],
            standard_document=review["standard_document"], coordinator_id=review["coordinator_id"],
            status=config.REVIEW_STATUS_LABELS[review["status"]]),
        config.CHAT_SCREENING_TEMPLATE.format(
            assessment=review["assessment"], requirements=requirements,
            rating_stated=config.BOOLEAN_YES if review["rating_stated"] else config.BOOLEAN_NO,
            triggers="; ".join(review["network_triggers"]) or config.CHAT_NO_TRIGGERS_TEXT,
            decisions_open=", ".join(review["decisions_open"])),
        config.CHAT_FIGURES_TEMPLATE.format(
            network_work_likelihood=f"{review['network_work_likelihood']:.{config.LIKELIHOOD_DECIMALS}f}",
            likelihood_band=review["likelihood_band"],
            expected_working_days=review["duration_text"]),
        config.CHAT_EVIDENCE_TEMPLATE.format(
            sentence_count=len(review["evidence"]), sentence_types=sentence_types,
            flag_count=len(review["disagreements"])),
        closing,
    ]


def answer_application_record(session, signed_in_user, open_applications, question_type,
                              application_reference):
    """Answer a question about one application from the review records, by the asker's role."""
    role = signed_in_user["role"]
    advisor = role == config.ROLE_CUSTOMER_SERVICE_ADVISOR
    if not advisor and role not in config.ROLES_SEEING_EVERY_REVIEW:
        raise ValueError(f"The role {role} may not hear about an application")
    if advisor and question_type == config.CHAT_TYPE_OUTCOME_PREDICTION:
        return config.OUTCOME_NOT_PREDICTED_LINE
    if not application_reference:
        return config.CHAT_APPLICATION_NOT_NAMED_MESSAGE
    reference = canonical_reference(application_reference)
    queued = open_applications[open_applications[config.APPLICATION_REFERENCE_COLUMN] == reference]
    if queued.empty:
        return config.CHAT_NO_APPLICATION_TEMPLATE.format(application_reference=reference)
    technology = queued.iloc[0][config.TECHNOLOGY_COLUMN]
    reviews = fetch_review_records(session, ApplicationReview.application_reference == reference)
    if advisor:
        return _advisor_answer(reviews[-1] if reviews else None, technology, reference)
    if not reviews:
        return config.CHAT_NO_REVIEW_TEMPLATE.format(application_reference=reference)
    lines = []
    for review in reviews:
        lines.extend(_review_lines(review))
    return "\n".join(lines)


def answer_substation_figure(connection_history, substation_reference, technology):
    """A figure about a substation, only where at least five premises make it up."""
    if not substation_reference or not technology:
        return config.CHAT_SUBSTATION_NOT_NAMED_MESSAGE
    rows = connection_history[
        (connection_history[config.SUBSTATION_REFERENCE_COLUMN] == substation_reference)
        & (connection_history[config.TECHNOLOGY_COLUMN] == technology)]
    premises = int(rows[config.PREMISES_REFERENCE_COLUMN].dropna().nunique())
    if premises < config.GROUP_SIZE_MINIMUM:
        return config.GROUP_SIZE_LINE
    return config.CHAT_SUBSTATION_FIGURE_TEMPLATE.format(
        technology_plural=config.TECHNOLOGY_PLURALS[technology], premises_count=premises,
        substation_reference=substation_reference)
