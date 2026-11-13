"""The language model client, the reply parser, and the seven agents.

Each agent is exactly one language model call. It is handed facts that the
standards, the models, the documents and the tools already established, and it
writes prose in a fixed layout of labelled fields; it calculates nothing.

Offline demo mode
-----------------
With PHASEWISE_OFFLINE_LLM=1 in .env, every agent's reply is written by a
deterministic template writer from the same facts instead of Gemini, so the whole
workflow can be demonstrated without an API key. Every reply still goes through
the same parsing and the same checks, and every call is still recorded.
"""

import os
import re
import time
from decimal import Decimal

from dotenv import load_dotenv

from config import config
from tools.application_data_tools import describe_record
from tools.prediction_tools import describe_expected_working_days

load_dotenv(config.ENV_FILE)

OFFLINE_MODE_VARIABLE = "PHASEWISE_OFFLINE_LLM"
OFFLINE_MODEL_NAME = "phasewise-offline-writer"


# =========================================================================== the client
def call_language_model(prompt_text, agent_name):
    """Make one Gemini call and measure what it cost."""
    from google import genai
    from google.genai import types

    api_key = os.getenv(config.GEMINI_API_KEY_VARIABLE)
    model_name = os.getenv(config.GEMINI_MODEL_VARIABLE)
    if not api_key or not model_name:
        raise ValueError(f"{agent_name} cannot call the language model: set both "
                         f"{config.GEMINI_API_KEY_VARIABLE} and {config.GEMINI_MODEL_VARIABLE} "
                         f"in .env")
    generation_settings = types.GenerateContentConfig(
        max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
        temperature=config.GEMINI_TEMPERATURE,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    client = genai.Client(api_key=api_key)
    try:
        started = time.perf_counter()
        response = client.models.generate_content(model=model_name, contents=prompt_text,
                                                  config=generation_settings)
        latency = int(round((time.perf_counter() - started) * config.MILLISECONDS_PER_SECOND))
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    text = response.text if response is not None else None
    if not text or not text.strip():
        raise ValueError(f"{agent_name} received an empty reply from {model_name}")
    candidates = response.candidates or []
    finish = candidates[0].finish_reason if candidates else None
    if finish != types.FinishReason.STOP:
        raise ValueError(f"{agent_name}'s reply from {model_name} did not finish normally ({finish})")
    usage = response.usage_metadata
    counts = None if usage is None else (usage.prompt_token_count, usage.candidates_token_count,
                                          usage.total_token_count)
    if counts is None or any(count is None for count in counts):
        raise ValueError(f"{agent_name}'s reply from {model_name} does not say how many tokens it used")
    return {
        "response_text": text.strip(),
        "call_record": {"agent_name": agent_name, "model_name": model_name,
                        "prompt_tokens": int(counts[0]), "response_tokens": int(counts[1]),
                        "total_tokens": int(counts[2]), "latency_milliseconds": latency},
    }


def offline_mode():
    """Whether replies are written by the offline template writer."""
    return os.getenv(OFFLINE_MODE_VARIABLE, "0").strip().lower() in {"1", "true", "yes"}


def complete(prompt_text, agent_name, offline_reply):
    """One call: Gemini, or the offline writer in demo mode. Both return the same record."""
    if not offline_mode():
        return call_language_model(prompt_text, agent_name)
    started = time.perf_counter()
    text = offline_reply().strip()
    latency = int(round((time.perf_counter() - started) * config.MILLISECONDS_PER_SECOND))
    prompt_tokens = max(1, round(len(prompt_text) / 4))
    response_tokens = max(1, round(len(text) / 4))
    return {"response_text": text,
            "call_record": {"agent_name": agent_name, "model_name": OFFLINE_MODEL_NAME,
                            "prompt_tokens": prompt_tokens, "response_tokens": response_tokens,
                            "total_tokens": prompt_tokens + response_tokens,
                            "latency_milliseconds": latency}}


# =========================================================================== reply parsing
def _label_of(head, labels):
    candidate = re.sub(config.RESPONSE_DECORATION_EDGE_PATTERN, "", head.strip())
    candidate = re.sub(r"\s+", " ", candidate.replace("-", " ")).strip().upper()
    return candidate if candidate in labels else None


def parse_labelled_response(response_text, field_labels, agent_name):
    """Read a labelled reply back into exactly the fields asked for, in order."""
    labels = {label.upper(): label for label in field_labels}
    collected = {label: [] for label in field_labels}
    current = None
    for raw_line in response_text.splitlines():
        line = raw_line.replace("**", "").strip()
        if ":" in line:
            head, rest = line.split(":", 1)
            label = _label_of(head, labels)
            if label is not None:
                current = labels[label]
                text = rest.strip(config.RESPONSE_DECORATION_CHARACTERS)
                if text:
                    collected[current].append(text)
                continue
        if current is None or not re.sub(r"[\W_]+", "", line):
            continue
        collected[current].append(line)
    fields = {label: "\n".join(collected[label]).strip() for label in field_labels}
    missing = [label for label, text in fields.items() if not text]
    if missing:
        raise ValueError(f"{agent_name} gave no answer for {', '.join(missing)}")
    return fields


def join_labelled_response(labelled_fields):
    """Write an agent's fields back as one text, a blank line between fields."""
    return "\n\n".join(f"{label}: {text}" for label, text in labelled_fields.items())


# =========================================================================== shared wording
def _preamble(role):
    return (f"You are the {role} in the home low-carbon connection review of "
            f"{config.ORGANISATION_NAME}. Use only the facts below: every result, figure and "
            f"passage was established by the connection standards, the models, the documents "
            f"and the tools. You interpret; you never calculate, never alter a result and never "
            f"introduce a figure that is not given here.\n")


def _layout(guidance):
    lines = "\n".join(f"{label}: {text}" for label, text in guidance)
    return ("\nReply in plain text with exactly these fields, in this order, each starting on a "
            f"new line with its label in capitals and a colon:\n{lines}\n")


def _technology_name(technology):
    return f"{config.TECHNOLOGY_NAMES.get(technology, technology)} ({technology})"


def _article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def _label(column):
    return config.FORM_ANSWER_LABELS.get(column, column.replace("_", " "))


def _list_or_none(items):
    return "\n".join(items) if items else "none"


def _application_facts(application_profile):
    words = describe_record(application_profile)
    technology = application_profile.get(config.TECHNOLOGY_COLUMN)
    return "\n".join([
        f"Application reference: {words.get(config.APPLICATION_REFERENCE_COLUMN)}",
        f"Portal: {words.get(config.SOURCE_PORTAL_COLUMN)}; revision {words.get(config.REVISION_NUMBER_COLUMN)}",
        f"Technology: {_technology_name(technology)}",
        f"Premises type: {words.get(config.PREMISES_TYPE_COLUMN)}; region type: "
        f"{words.get(config.REGION_TYPE_COLUMN)}; substation: {words.get(config.SUBSTATION_REFERENCE_COLUMN)}",
        f"Urgency: {words.get(config.URGENCY_COLUMN)}",
        f"Supply phases: {words.get(config.SUPPLY_PHASES_COLUMN)}; equipment phases: "
        f"{words.get(config.EQUIPMENT_PHASES_COLUMN)}",
        f"Service type: {words.get(config.SERVICE_TYPE_COLUMN)}; cut-out rating: "
        f"{words.get(config.CUT_OUT_RATING_COLUMN)} amps",
    ])


def _screening_facts(screening):
    lines = [f"Standard screened under: {screening['standard_document']}", "Requirements:"]
    for requirement in screening["requirements"]:
        lines.append(f"- {requirement['requirement_name']}: {requirement['result']}")
    lines += [
        f"Counts: Met {screening[config.REQUIREMENTS_MET_COUNT_COLUMN]}, Not met "
        f"{screening[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN]}, Not documented "
        f"{screening[config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN]}",
        f"Rating stated: {config.BOOLEAN_YES if screening['rating_stated'] else config.BOOLEAN_NO}",
        f"Network triggers: {', '.join(screening['network_triggers']) or 'none'}",
        f"Assessment: {screening['assessment']}",
        f"Decisions open: {', '.join(screening['decisions_open'])}",
    ]
    return "\n".join(lines)


def _counts_sentence(screening):
    return (f"{screening[config.REQUIREMENTS_MET_COUNT_COLUMN]} Met, "
            f"{screening[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN]} Not met and "
            f"{screening[config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN]} Not documented")


def _names_with(screening, result):
    return [requirement["requirement_name"] for requirement in screening["requirements"]
            if requirement["result"] == result]


# =========================================================================== 1. planner
PLAN_LABELS = ["PRIORITY", "CHECKS", "RATIONALE"]


def plan_review(application_profile, application_documents, screening, coordinator_request):
    """Decide what the review has to establish about this application."""
    statements = [f"- {document.get(config.DOCUMENT_ID_COLUMN)}: {document.get(config.DOCUMENT_TITLE_COLUMN)}"
                  for document in application_documents] or ["No statement is attached."]
    prompt = (_preamble("review planner") + "\nTHE APPLICATION\n" + _application_facts(application_profile)
              + "\n\nSTATEMENTS ATTACHED\n" + "\n".join(statements)
              + "\n\nTHE SCREENING\n" + _screening_facts(screening)
              + f"\n\nWHAT THE COORDINATOR ASKED\n{coordinator_request}\n"
              + "\nYou may not alter any screening result or introduce any figure beyond those above."
              + _layout([("PRIORITY", "one or two sentences on how urgent the review is and why"),
                         ("CHECKS", "the checks the review must make, one per line"),
                         ("RATIONALE", "two or three sentences explaining the plan")]))

    def offline():
        technology = application_profile.get(config.TECHNOLOGY_COLUMN)
        urgent = application_profile.get(config.URGENCY_COLUMN) == config.URGENCY_HEATING_FAILED
        checks = [f"- Confirm the {requirement['requirement_name']} answer, which is "
                  f"{requirement['result']}." for requirement in screening["requirements"]
                  if requirement["result"] != config.REQUIREMENT_MET]
        checks += [f"- Review the {trigger} trigger against the supply answers."
                   for trigger in screening["network_triggers"]]
        checks.append("- Compare every equipment rating sentence in the statements with the form.")
        where = ("replaces failed heating and sits at the front of the queue" if urgent
                 else "follows the queue in reference order")
        return (f"PRIORITY: {'High' if urgent else 'Standard'} priority. This "
                f"{config.TECHNOLOGY_NAMES[technology]} application {where}.\n"
                f"CHECKS:\n" + "\n".join(checks) + "\n"
                f"RATIONALE: The screening under {screening['standard_document']} found "
                f"{_counts_sentence(screening)}, and the assessment is {screening['assessment']}. "
                f"The review follows the coordinator's request: {coordinator_request.strip()}")

    reply = complete(prompt, config.AGENT_NAME_PLANNER, offline)
    return {"review_plan": parse_labelled_response(reply["response_text"], PLAN_LABELS,
                                                   config.AGENT_NAME_PLANNER),
            "call_record": reply["call_record"]}


# =========================================================================== 2. requirements analyst
REQUIREMENT_LABELS = ["REQUIREMENTS MET", "REQUIREMENTS NOT MET", "NOT DOCUMENTED",
                      "NETWORK TRIGGERS", "REQUIREMENTS SUMMARY"]


def analyse_requirements(screening, application_profile, review_plan):
    """Account for each requirement's result and every network trigger that applies."""
    words = describe_record(application_profile)
    lines = []
    for requirement in screening["requirements"]:
        answers = config.REQUIREMENT_ANSWER_COLUMNS[requirement["requirement_kind"]]
        answered = "; ".join(f"{_label(column)}: {words.get(column, config.NOT_GIVEN_TEXT)}"
                             for column in answers)
        blanks = ", ".join(_label(column) for column in requirement["blank_answers"]) or "none"
        lines.append(f"- {requirement['requirement_name']} ({requirement['requirement_sentence']}) "
                     f"reads {answered}. Result: {requirement['result']}. Blank answers: {blanks}.")
    trigger_answers = "\n".join(f"- {_label(column)}: {words.get(column, config.NOT_GIVEN_TEXT)}"
                                for column in config.TRIGGER_ANSWER_COLUMNS)
    prompt = (_preamble("requirements analyst") + "\nREQUIREMENTS, IN THE STANDARD'S ORDER\n"
              + "\n".join(lines)
              + f"\nCounts: {_counts_sentence(screening)}. Rating stated: "
              + f"{config.BOOLEAN_YES if screening['rating_stated'] else config.BOOLEAN_NO}.\n"
              + config.CURRENT_RULE_TEXT
              + "\n\nANSWERS THE NETWORK TRIGGERS READ\n" + trigger_answers
              + f"\nTriggers that apply: {', '.join(screening['network_triggers']) or 'none'}\n"
              + config.NETWORK_BLANK_ANSWER_RULE_TEXT
              + f"\n\nASSESSMENT: {screening['assessment']}. {config.ASSESSMENT_RULE_TEXT}\n"
              + f"Decisions open: {', '.join(screening['decisions_open'])}\n"
              + f"\nTHE PLAN'S CHECKS\n{review_plan['CHECKS']}\n"
              + "\nHold to every result exactly as the screening gave it. Never supply a blank answer "
                "from a statement, and never present the application as already decided."
              + _layout([("REQUIREMENTS MET", "each requirement Met and the answer that meets it, or none"),
                         ("REQUIREMENTS NOT MET", "each requirement Not met and why, or none"),
                         ("NOT DOCUMENTED", "each requirement Not documented and the blank answers it needs, or none"),
                         ("NETWORK TRIGGERS", "each trigger that applies and the answers behind it, or none"),
                         ("REQUIREMENTS SUMMARY", "two or three sentences summarising the screening")]))

    def offline():
        def lines_for(result, explain):
            chosen = [requirement for requirement in screening["requirements"]
                      if requirement["result"] == result]
            return _list_or_none([f"- {requirement['requirement_name']}: {explain(requirement)}"
                                  for requirement in chosen])

        return (
            "REQUIREMENTS MET:\n" + lines_for(
                config.REQUIREMENT_MET,
                lambda r: f"the form's answer satisfies \"{r['requirement_sentence']}\"") + "\n"
            "REQUIREMENTS NOT MET:\n" + lines_for(
                config.REQUIREMENT_NOT_MET,
                lambda r: f"the form's answer does not satisfy \"{r['requirement_sentence']}\"") + "\n"
            "NOT DOCUMENTED:\n" + lines_for(
                config.REQUIREMENT_NOT_DOCUMENTED,
                lambda r: "the form leaves " + ", ".join(_label(c) for c in r["blank_answers"]) + " blank") + "\n"
            "NETWORK TRIGGERS:\n" + _list_or_none([f"- The {trigger} trigger applies on the form's answers."
                                                   for trigger in screening["network_triggers"]]) + "\n"
            f"REQUIREMENTS SUMMARY: Under {screening['standard_document']} the screening found "
            f"{_counts_sentence(screening)}, with the rating "
            f"{'stated' if screening['rating_stated'] else 'left blank'}. "
            f"{len(screening['network_triggers'])} network trigger(s) apply, so the assessment is "
            f"{screening['assessment']} and the decisions open are {', '.join(screening['decisions_open'])}.")

    reply = complete(prompt, config.AGENT_NAME_REQUIREMENTS, offline)
    return {"requirements_analysis": parse_labelled_response(
        reply["response_text"], REQUIREMENT_LABELS, config.AGENT_NAME_REQUIREMENTS),
            "call_record": reply["call_record"]}


# =========================================================================== 3. evidence analyst
EVIDENCE_LABELS = ["EVIDENCE FOUND", "DISAGREEMENTS", "CHECKS TO MAKE", "EVIDENCE SUMMARY"]


def _amps(values):
    return ", ".join(f"{float(value):.{config.AMPS_DECIMALS}f}" for value in values) or "none"


def review_evidence(evidence_findings, disagreements, reviewer_checks, application_profile,
                    screening, review_plan):
    """Set the evidence in the statements beside the form, with the guides' checks."""
    words = describe_record(application_profile)
    columns = []
    for requirement in screening["requirements"]:
        for column in config.REQUIREMENT_ANSWER_COLUMNS[requirement["requirement_kind"]]:
            if column not in columns:
                columns.append(column)
    for column in config.STATEMENT_GUIDE_ANSWER_COLUMNS:
        if column not in columns:
            columns.append(column)
    form_lines = "\n".join(f"- {_label(column)}: {words.get(column, config.NOT_GIVEN_TEXT)}"
                           for column in columns)
    sentence_lines = [f"- [{finding['document_id']}] ({finding['sentence_type']}; bears on "
                      f"{', '.join(finding['requirements_borne_on']) or 'no requirement'}) {finding['sentence']}"
                      for finding in evidence_findings]
    flag_lines = [f"- [{flag['document_id']}] {flag['flag']}: \"{flag['sentence']}\" states "
                  f"{_amps(flag['stated_amps'])} amps per phase; the form's ratings are "
                  f"{_amps(flag['form_rating_amps'])}" for flag in disagreements]
    check_lines = [f"- {config.SENTENCE_TYPE_GUIDE_DOCUMENTS[sentence_type]}: {checks}"
                   for sentence_type, checks in reviewer_checks.items()]
    prompt = (_preamble("evidence analyst") + "\nTHE FORM'S ANSWERS\n" + form_lines
              + "\n\nEVERY SENTENCE READ, STATEMENT BY STATEMENT\n" + _list_or_none(sentence_lines)
              + "\n\nFLAGGED SENTENCES\n" + _list_or_none(flag_lines)
              + "\n\nTHE GUIDES' CHECKS\n" + _list_or_none(check_lines)
              + f"\n\nTHE PLAN'S CHECKS\n{review_plan['CHECKS']}\n"
              + "\nThe statements are evidence only and never a substitute for an answer on the "
                "form. A flagged disagreement is left for the Connections Engineer to settle."
              + _layout([("EVIDENCE FOUND", "what the statements say, grouped by sentence type"),
                         ("DISAGREEMENTS", "each flagged sentence beside the form, or none"),
                         ("CHECKS TO MAKE", "the guides' checks the engineer should make, one per line"),
                         ("EVIDENCE SUMMARY", "two or three sentences summarising the evidence")]))

    def offline():
        by_type = {}
        for finding in evidence_findings:
            by_type.setdefault(finding["sentence_type"], []).append(finding)
        found = [f"- {sentence_type}: {len(items)} sentence(s), for example \"{items[0]['sentence']}\""
                 for sentence_type, items in sorted(by_type.items())]
        documents = sorted({finding["document_id"] for finding in evidence_findings})
        flags = [f"- {flag['flag']} in {flag['document_id']}: the sentence states "
                 f"{_amps(flag['stated_amps'])} amps per phase against the form's "
                 f"{_amps(flag['form_rating_amps'])}." for flag in disagreements]
        checks = [f"- {text.split('. ')[0].rstrip('.')}." for text in reviewer_checks.values()]
        return ("EVIDENCE FOUND:\n" + (_list_or_none(found) if found else
                                       "No statement is attached to this application.") + "\n"
                "DISAGREEMENTS:\n" + (_list_or_none(flags) if flags else
                                      "None of the equipment rating sentences disagrees with the form.") + "\n"
                "CHECKS TO MAKE:\n" + (_list_or_none(checks) if checks else
                                       "No statement guide applies, because no sentence was read.") + "\n"
                f"EVIDENCE SUMMARY: {len(evidence_findings)} sentence(s) were read from "
                f"{len(documents)} statement(s), and {len(disagreements)} equipment rating sentence(s) "
                f"were flagged. The statements are evidence only and leave every answer on the form "
                f"as it stands.")

    reply = complete(prompt, config.AGENT_NAME_EVIDENCE, offline)
    return {"evidence_review": parse_labelled_response(reply["response_text"], EVIDENCE_LABELS,
                                                       config.AGENT_NAME_EVIDENCE),
            "call_record": reply["call_record"]}


# =========================================================================== 4. risk analyst
PREDICTION_LABELS = ["LIKELIHOOD", "EXPECTED WORKING DAYS", "PREDICTION SUMMARY"]


def analyse_predictions(likelihood, duration, application_profile, screening, review_plan):
    """Interpret the network work likelihood and the expected working days."""
    inputs = dict(application_profile)
    inputs.update({column: screening[column] for column in config.REQUIREMENT_COUNT_RESULTS})
    inputs.update({column: int(trigger in screening["network_triggers"])
                   for trigger, column in config.NETWORK_TRIGGER_FLAG_COLUMNS.items()})
    words = describe_record(inputs)

    def input_lines(columns):
        return "\n".join(f"- {_label(column)}: {words.get(column, config.NOT_GIVEN_TEXT)}"
                         for column in columns)

    network_inputs = config.NETWORK_WORK_NUMERIC_FEATURES + config.NETWORK_WORK_CATEGORICAL_FEATURES
    duration_inputs = config.DURATION_NUMERIC_FEATURES + config.DURATION_CATEGORICAL_FEATURES
    duration_text = describe_expected_working_days(duration)
    prompt = (_preamble("risk analyst")
              + f"\nNETWORK WORK LIKELIHOOD: {likelihood['network_work_likelihood']:.{config.LIKELIHOOD_DECIMALS}f}, "
              + f"band {likelihood['likelihood_band']} ({config.HIGH_LIKELIHOOD_BAND} from "
              + f"{config.HIGH_LIKELIHOOD_MINIMUM}, {config.MEDIUM_LIKELIHOOD_BAND} from "
              + f"{config.MEDIUM_LIKELIHOOD_MINIMUM}, otherwise {config.LOW_LIKELIHOOD_BAND})\n"
              + "Inputs of the network work model:\n" + input_lines(network_inputs)
              + f"\n\nEXPECTED WORKING DAYS: {duration_text} (a delay is expected when the forecast "
              + f"is more than {config.DELAY_MARGIN_WORKING_DAYS} working days above the plan)\n"
              + "Inputs of the duration model:\n" + input_lines(duration_inputs)
              + f"\n\n{config.BLANK_INPUT_RULE_TEXT}\n"
              + f"\nTHE PLAN'S CHECKS\n{review_plan['CHECKS']}\n"
              + "\nNeither figure has any bearing on a requirement, a network trigger, a decision or "
                "a notice. Never recalculate either figure."
              + _layout([("LIKELIHOOD", "what the network work likelihood and its band mean here"),
                         ("EXPECTED WORKING DAYS", "what the expected working days mean against the plan"),
                         ("PREDICTION SUMMARY", "two or three sentences, saying the figures decide nothing")]))

    def offline():
        band = likelihood["likelihood_band"]
        triggers = screening["network_triggers"]
        return (f"LIKELIHOOD: The network work likelihood is "
                f"{likelihood['network_work_likelihood']:.{config.LIKELIHOOD_DECIMALS}f}, in the {band} band. "
                f"Applications like this one, with {len(triggers)} network trigger(s) and "
                f"{_counts_sentence(screening)}, have needed network work at about this rate.\n"
                f"EXPECTED WORKING DAYS: {duration_text}. The forecast reflects similar past "
                f"connections from the go-ahead to energisation.\n"
                f"PREDICTION SUMMARY: Both figures describe connections like this one and are given "
                f"for planning only. Neither figure has any bearing on a requirement, a network "
                f"trigger, the decision or any notice.")

    reply = complete(prompt, config.AGENT_NAME_RISK, offline)
    return {"prediction_analysis": parse_labelled_response(reply["response_text"], PREDICTION_LABELS,
                                                           config.AGENT_NAME_RISK),
            "call_record": reply["call_record"]}


# =========================================================================== 5. writer
WRITER_LABELS = ["RECOMMENDATION", "RECOMMENDED DECISION"]


def write_review_recommendation(application_profile, coordinator_request, screening, likelihood,
                                duration, review_plan, requirements_analysis, evidence_review,
                                prediction_analysis):
    """Write the review a Connections Engineer decides from, recommending an open decision."""
    decisions = "\n".join(f"- {decision}: {config.DECISION_DESCRIPTIONS[decision]}. "
                          f"{config.DECISION_OPEN_CONDITIONS[decision]}" for decision in config.DECISIONS)

    def section(title, fields):
        return f"\n{title}\n" + "\n".join(f"{label}: {text}" for label, text in fields.items())

    prompt = (_preamble("review writer") + "\nTHE APPLICATION\n" + _application_facts(application_profile)
              + f"\nStandard: {screening['standard_document']}\n"
              + f"\nWHAT THE COORDINATOR ASKED\n{coordinator_request}\n"
              + f"\nTHE PLAN\nPRIORITY: {review_plan['PRIORITY']}\nCHECKS: {review_plan['CHECKS']}\n"
              + section("THE REQUIREMENTS ANALYSIS", requirements_analysis)
              + section("THE EVIDENCE REVIEW", evidence_review)
              + section("THE PREDICTION ANALYSIS", prediction_analysis)
              + "\n\nTHE SCREENING\n" + _screening_facts(screening)
              + f"\n{config.ASSESSMENT_RULE_TEXT}\n"
              + f"\nFIGURES: network work likelihood {likelihood['network_work_likelihood']:.2f} "
              + f"({likelihood['likelihood_band']}); {describe_expected_working_days(duration)}\n"
              + f"\nTHE DECISIONS\n{decisions}\nDecisions open: {', '.join(screening['decisions_open'])}\n"
              + "\nYou only recommend: a Connections Engineer decides. Recommend the assessment unless "
                "the review supports another open decision, and then give that reason. Never cite the "
                "likelihood or the expected working days as a reason."
              + _layout([("RECOMMENDATION", "several sentences of prose; the last names the open "
                                            "decision you recommend and the reason"),
                         ("RECOMMENDED DECISION", "exactly one of the decisions open, on one line")]))

    def offline():
        name = config.TECHNOLOGY_NAMES[application_profile.get(config.TECHNOLOGY_COLUMN)]
        failing = _names_with(screening, config.REQUIREMENT_NOT_MET) + \
            _names_with(screening, config.REQUIREMENT_NOT_DOCUMENTED)
        if screening["assessment"] == config.DECISION_RETURN:
            reason = ("the requirement(s) " + ", ".join(failing) + " must be put right first"
                      if failing else "the rating is blank and every application states its rating")
        elif screening["assessment"] == config.DECISION_REFER:
            reason = "the network triggers " + ", ".join(screening["network_triggers"]) + " apply"
        else:
            reason = "every requirement is Met, the rating is stated and no network trigger applies"
        return (f"RECOMMENDATION: Application {screening['application_reference']} for {_article(name)} "
                f"{name} was screened under {screening['standard_document']}, which found "
                f"{_counts_sentence(screening)}. The statements were read sentence by sentence and "
                f"every disagreement is left for the engineer to settle. Network triggers: "
                f"{', '.join(screening['network_triggers']) or 'none'}. The review recommends "
                f"{screening['assessment']}, because {reason}.\n"
                f"RECOMMENDED DECISION: {screening['assessment']}")

    reply = complete(prompt, config.AGENT_NAME_WRITER, offline)
    fields = parse_labelled_response(reply["response_text"], WRITER_LABELS, config.AGENT_NAME_WRITER)
    decision_text = fields["RECOMMENDED DECISION"]
    checked = re.sub(config.RESPONSE_DECORATION_EDGE_PATTERN, "", decision_text.strip()).upper()
    if "\n" in decision_text or checked not in screening["decisions_open"]:
        raise ValueError(f"{config.AGENT_NAME_WRITER} recommended a decision that is not open: "
                         f"{decision_text}")
    fields["RECOMMENDED DECISION"] = checked
    return {"review_recommendation": fields, "call_record": reply["call_record"]}


# =========================================================================== notice checks
def _normalised(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def _figures_in(text):
    cleaned = text.replace(config.THOUSANDS_SEPARATOR, "")
    return {Decimal(found) for found in re.findall(config.FIGURE_PATTERN, cleaned)}


def _names_technology(text, technology):
    lowered = _normalised(text)
    return config.TECHNOLOGY_NAMES[technology].lower() in lowered or \
        re.search(rf"\b{re.escape(technology.lower())}\b", lowered) is not None


def _ends_with_sign_off(notice):
    def spaced(text):
        return re.sub(config.SIGN_OFF_SPACING_PATTERN, " ", text.lower()).strip()
    return spaced(notice).endswith(spaced(config.NOTICE_SIGN_OFF))


def _check_notice_body(notice, agent_name):
    if any(character in notice for character in config.TEMPLATE_SLOT_CHARACTERS):
        raise ValueError(f"{agent_name} left a gap to fill in its notice")
    if not _ends_with_sign_off(notice):
        raise ValueError(f"{agent_name} did not end its notice with {config.NOTICE_SIGN_OFF}")


def _check_forbidden_terms(text, agent_name):
    lowered = _normalised(text)
    found = [term for term in config.NOTICE_FORBIDDEN_TERMS if term in lowered]
    if found:
        raise ValueError(f"{agent_name} wrote of {', '.join(found)} in a notice")


def _value_words(details, column):
    return describe_record({column: details.get(column)})[column]


def _phase_words(phases):
    if phases is None:
        return "on phases not given"
    return "single-phase" if int(phases) == 1 else "three-phase"


def _labelled_details(details):
    lines = []
    for column, value in details.items():
        label = config.NOTICE_DETAIL_LABELS.get(column, column.replace("_", " ").capitalize())
        words = _technology_name(value) if column == config.TECHNOLOGY_COLUMN else _value_words(details, column)
        lines.append(f"- {label}: {words}")
    return "\n".join(lines)


# =========================================================================== 6. installer notice
INSTALLER_LABELS = ["SUBJECT", "NOTICE"]


def draft_installer_notice(decision, application_profile, notice_details, notice_guidance):
    """Write the connection approval or the rework request to the installer."""
    agent = config.AGENT_NAME_INSTALLER_NOTICE
    technology = notice_details.get(config.TECHNOLOGY_COLUMN) or application_profile.get(config.TECHNOLOGY_COLUMN)
    reference = notice_details[config.APPLICATION_REFERENCE_COLUMN]
    rating_column = config.RATING_COLUMNS[technology]
    if decision == config.DECISION_APPROVE and notice_details.get(rating_column) is None:
        raise ValueError(f"{agent} cannot write a connection approval that gives no rating")
    kind = config.NOTICE_KINDS[decision]
    portal = describe_record(application_profile).get(config.SOURCE_PORTAL_COLUMN)

    if decision == config.DECISION_APPROVE:
        details_text = _labelled_details(notice_details)
    else:
        head = {column: notice_details[column] for column in
                (config.APPLICATION_REFERENCE_COLUMN, config.TECHNOLOGY_COLUMN, "standard_document")}
        parts = [_labelled_details(head), "Requirements to put right:"]
        for requirement in notice_details["requirements_to_put_right"]:
            answers = "; ".join(f"{_label(column)}: {describe_record({column: value})[column]}"
                                for column, value in requirement["form_answers"].items())
            needs = ", ".join(_label(column) for column in requirement["blank_answers"]) or "none"
            parts.append(f"- {requirement['requirement_name']} ({requirement['result']}): "
                         f"{requirement['requirement_sentence']} Form answers: {answers}. "
                         f"Answers needed: {needs}.")
        if not notice_details["requirements_to_put_right"]:
            parts.append("none")
        blank_rating = notice_details.get("blank_rating_answer")
        parts.append(f"Blank rating: {config.RATING_TERMS[blank_rating]}" if blank_rating else "Blank rating: none")
        parts.append(f"Documentation to provide: {notice_details['documentation_to_provide']}")
        flags = [f"- {flag['document_id']}: \"{flag['sentence']}\" states {_amps(flag['stated_amps'])} "
                 f"amps per phase; the form's ratings are {_amps(flag['form_rating_amps'])}"
                 for flag in notice_details["flagged_statements"]]
        parts.append("Statement sentences flagged against the form:\n" + _list_or_none(flags))
        details_text = "\n".join(parts)

    prompt = (_preamble("installer notice writer")
              + f"\nTHE DECISION: {decision}: {config.DECISION_DESCRIPTIONS[decision]}.\n"
              + f"Write the {kind} to the installer, who applied through {portal}.\n"
              + f"\nTHE DETAILS\n{details_text}\n\nTHE GUIDANCE\n{notice_guidance}\n"
              + f"\nEnd the notice with exactly: {config.NOTICE_SIGN_OFF}\n"
              + "Never give any model figure, never name anyone at the organisation, never make up a "
                "time, date or contact detail, and never leave a gap in square brackets to fill in."
              + _layout([("SUBJECT", "one line naming the application reference"),
                         ("NOTICE", "the whole notice in plain prose, ending with the sign-off")]))

    def offline():
        name = config.TECHNOLOGY_NAMES[technology]
        if decision == config.DECISION_APPROVE:
            rating = _value_words(notice_details, rating_column)
            unit = "amps per phase" if rating_column == config.RATED_CURRENT_COLUMN else "kW"
            conditions = []
            if config.EXPORT_LIMIT_COLUMN in notice_details:
                conditions.append(f"export is limited to {_value_words(notice_details, config.EXPORT_LIMIT_COLUMN)} kW")
            if config.LOAD_LIMITER_SETTING_COLUMN in notice_details:
                conditions.append("the load limiting device is set to "
                                  f"{_value_words(notice_details, config.LOAD_LIMITER_SETTING_COLUMN)} amps per phase")
            condition_text = ("The approval is made on the condition that " + " and ".join(conditions) + ". "
                              if conditions else "The approval sets no export limit and no load limiting setting. ")
            body = (f"Thank you for application {reference}, made through {portal}. The {name} "
                    f"({technology}) has been approved for connection as applied, under "
                    f"{notice_details['standard_document']}. The equipment is "
                    f"{_phase_words(notice_details.get(config.EQUIPMENT_PHASES_COLUMN))} with a "
                    f"{_value_words(notice_details, config.EARTHING_ARRANGEMENT_COLUMN)} earthing "
                    f"arrangement, and its {config.RATING_TERMS[rating_column]} is {rating} {unit}. "
                    f"{condition_text}Please keep this approval with the installation records.")
            subject = f"Connection approval for application {reference}"
        else:
            items = [f"{requirement['requirement_name']} is {requirement['result']}"
                     for requirement in notice_details["requirements_to_put_right"]]
            blank = notice_details.get("blank_rating_answer")
            if blank:
                items.append(f"the {config.RATING_TERMS[blank]} is blank and must be given")
            flags = len(notice_details["flagged_statements"])
            body = (f"Application {reference} for {_article(name)} {name} ({technology}), made through "
                    f"{portal}, has been returned for rework under {notice_details['standard_document']}. "
                    f"Please put right the following: " + "; ".join(items) + ". "
                    f"The standard sets out the documentation to provide: "
                    f"{notice_details['documentation_to_provide']} "
                    + (f"{flags} statement sentence(s) disagree with the form and should be settled. "
                       if flags else "")
                    + "Every blank answer named above must be given on the revised form.")
            subject = f"Rework request for application {reference}"
        return f"SUBJECT: {subject}\nNOTICE: {body}\n\n{config.NOTICE_SIGN_OFF}"

    reply = complete(prompt, agent, offline)
    fields = parse_labelled_response(reply["response_text"], INSTALLER_LABELS, agent)
    subject, notice = fields["SUBJECT"], fields["NOTICE"]
    if "\n" in subject or reference.lower() not in subject.lower():
        raise ValueError(f"{agent} wrote a subject that is not one line naming {reference}")
    combined = _normalised(f"{subject} {notice}")
    if decision == config.DECISION_APPROVE:
        if not _names_technology(combined, technology) or \
                Decimal(str(notice_details[rating_column])) not in _figures_in(combined):
            raise ValueError(f"{agent} did not name the technology and its rating")
    else:
        missing = [requirement["requirement_name"] for requirement in notice_details["requirements_to_put_right"]
                   if requirement["requirement_name"].lower() not in combined]
        blank = notice_details.get("blank_rating_answer")
        if blank and config.RATING_TERMS[blank].lower() not in combined:
            missing.append(config.RATING_TERMS[blank])
        if missing:
            raise ValueError(f"{agent} did not name {', '.join(missing)}")
    _check_forbidden_terms(combined, agent)
    _check_notice_body(notice, agent)
    return {"installer_notice": fields, "call_record": reply["call_record"]}


# =========================================================================== 7. household notice
HOUSEHOLD_LABELS = ["HEADLINE", "NOTICE"]


def draft_household_notice(application_profile, household_details, household_guidance):
    """Write the household notice to the people who live at the premises."""
    agent = config.AGENT_NAME_HOUSEHOLD_NOTICE
    technology = household_details[config.TECHNOLOGY_COLUMN]
    rating_column = config.RATING_COLUMNS[technology]
    if household_details.get(rating_column) is None:
        raise ValueError(f"{agent} cannot write a household notice that gives no rating")
    conditions = [column for column in config.APPROVAL_CONDITION_COLUMNS[technology]
                  if household_details.get(column) is not None]
    rating_label = config.NOTICE_DETAIL_LABELS[rating_column]
    condition_lines = [f"- {config.NOTICE_DETAIL_LABELS[column]}: {_value_words(household_details, column)}"
                       for column in conditions] or ["There is neither an export limit nor a load limiting setting."]
    prompt = (_preamble("household notice writer")
              + "\nWrite to the people who live at the premises, in plain friendly words.\n"
              + f"Technology: {config.TECHNOLOGY_NAMES[technology]}\n"
              + f"{rating_label}: {_value_words(household_details, rating_column)}\n"
              + "\n".join(condition_lines)
              + f"\n\nTHE GUIDANCE\n{household_guidance}\n"
              + f"\nEnd the notice with exactly: {config.NOTICE_SIGN_OFF}\n"
              + "Never mention the substation, the cut-out rating or the maximum demand, never give any "
                "model figure, never name anyone at the organisation, never make up a time, date or "
                "contact detail, and never leave a gap in square brackets to fill in."
              + _layout([("HEADLINE", "one friendly line"),
                         ("NOTICE", "the whole notice in plain prose, ending with the sign-off")]))

    def offline():
        name = config.TECHNOLOGY_NAMES[technology]
        equipment = config.EQUIPMENT_NAMES[technology]
        unit = "amps per phase" if rating_column == config.RATED_CURRENT_COLUMN else "kW"
        keep = [config.HOUSEHOLD_CONDITION_TEMPLATES[column].format(
            equipment_name=equipment, value=_value_words(household_details, column)) for column in conditions]
        keep_text = (" Please keep to this condition: " + " ".join(keep)) if keep else \
            " There is no export limit and no load limiting setting to keep to."
        body = (f"Good news: the {name} at your home has been approved for connection to the "
                f"electricity network. Your {equipment} is rated at "
                f"{_value_words(household_details, rating_column)} {unit}.{keep_text} Your installer "
                f"will explain how the {equipment} works once it is fitted.")
        return (f"HEADLINE: Your {name} is approved for connection\nNOTICE: {body}\n\n"
                f"{config.NOTICE_SIGN_OFF}")

    reply = complete(prompt, agent, offline)
    fields = parse_labelled_response(reply["response_text"], HOUSEHOLD_LABELS, agent)
    headline, notice = fields["HEADLINE"], fields["NOTICE"]
    if "\n" in headline:
        raise ValueError(f"{agent} wrote a headline over more than one line")
    combined = _normalised(f"{headline} {notice}")
    figures = _figures_in(combined)
    wanted = [Decimal(str(household_details[column])) for column in conditions]
    if not _names_technology(combined, technology) or any(value not in figures for value in wanted):
        raise ValueError(f"{agent} did not name the technology and every condition's figure")
    withheld = [name for name, pattern in config.HOUSEHOLD_NOTICE_WITHHELD_PATTERNS.items()
                if re.search(pattern, combined)]
    staff = [account["username"] for account in config.SEED_USER_ACCOUNTS
             if re.search(rf"\b{re.escape(account['username'].lower())}\b", combined)
             or account["display_name"].lower() in combined]
    if withheld or staff:
        raise ValueError(f"{agent} named what a household notice withholds: "
                         f"{', '.join(withheld + staff)}")
    _check_forbidden_terms(combined, agent)
    _check_notice_body(notice, agent)
    return {"household_notice": fields, "call_record": reply["call_record"]}
