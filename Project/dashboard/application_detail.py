"""The application detail page: everything established about one application."""

import streamlit as st

from config import config
from dashboard.api_client import SIGNED_IN_USER_KEY, call_service
from dashboard.application_queue import SELECTED_APPLICATION_KEY
from dashboard.ui import (answer_label, badge_html, badge_row, band_colour, callout, decision_colour,
                          delay_badge, empty_state, escape, format_amps, html_table, icon, key_values,
                          mono, muted, notice_box, page_header, result_colour, section, sentence_card,
                          spacer, stat_row, status_colour, strong)


def _option_label(row):
    return f"{row['application_reference']}  ·  {row['technology_label']}  ·  {row['premises_type']}"


def _render_screening(screening):
    rows = []
    for requirement in screening["requirements"]:
        blanks = ", ".join(answer_label(column) for column in requirement["blank_answers"])
        rows.append([
            strong(requirement["requirement_name"]),
            badge_html(requirement["result"], result_colour(requirement["result"])),
            f'<span class="pw-text">{escape(requirement["requirement_sentence"])}</span>',
            muted(blanks or "—"),
        ])
    html_table(["Requirement", "Result", "What the standard requires", "Answers left blank"], rows)
    if not screening["rating_stated"]:
        column = config.RATING_COLUMNS[screening["technology"]]
        callout(f"The {config.RATING_TERMS[column]} is blank. Every application must state its rating.",
                tone="warning", title="Rating not stated")

    section("Decisions open", "The decisions this screening leaves to a Connections Engineer.")
    badge_row([(decision, decision_colour(decision)) for decision in screening["decisions_open"]]
              or [("None", config.NEUTRAL_BADGE_COLOUR)])
    section("Network triggers", "Any one of these means the network is assessed first.")
    badge_row([(trigger, "violet") for trigger in screening["network_triggers"]]
              or [("No network trigger", config.NEUTRAL_BADGE_COLOUR)])


def _render_evidence(detail):
    documents = detail["application_documents"]
    if not documents:
        empty_state("No statements attached", "This application was submitted without a supporting statement.",
                    icon_name="file")
        return
    flags = {(flag["document_id"], flag["sentence"]): flag for flag in detail["disagreements"]}
    for document in documents:
        document_id = document[config.DOCUMENT_ID_COLUMN]
        findings = [item for item in detail["evidence_findings"] if item["document_id"] == document_id]
        with st.container(border=True):
            st.markdown(
                f'<div class="pw-doc-head">{icon("file", 16)}'
                f'<span class="pw-doc-title">{escape(document.get(config.DOCUMENT_TITLE_COLUMN))}</span>'
                f'{mono(document_id)}'
                f'<span class="pw-doc-count">{len(findings)} sentence{"s" if len(findings) != 1 else ""}</span></div>',
                unsafe_allow_html=True)
            for finding in findings:
                flag = flags.get((document_id, finding["sentence"]))
                badges = badge_html(finding["sentence_type"], config.SENTENCE_TYPE_BADGE_COLOUR)
                meta = f"Bears on {escape(', '.join(finding['requirements_borne_on']) or 'no requirement')}"
                if flag:
                    badges += badge_html(flag["flag"], "red", icon_name="flag")
                    meta += (f" · States {escape(format_amps(flag['stated_amps']))} A; the form gives "
                             f"{escape(format_amps(flag['form_rating_amps']) or 'no rating')}"
                             f"{' A' if flag['form_rating_amps'] else ''}")
                sentence_card(finding["sentence"], badges, meta, flagged=bool(flag))


def _render_form(detail):
    application = detail["application"]
    technology = detail["screening"]["technology"]
    uses = {}

    def use(column, purpose):
        purposes = uses.setdefault(column, [])
        if purpose not in purposes:
            purposes.append(purpose)

    for requirement in detail["screening"]["requirements"]:
        for column in config.REQUIREMENT_ANSWER_COLUMNS[requirement["requirement_kind"]]:
            use(column, requirement["requirement_name"])
    use(config.RATING_COLUMNS[technology], "Rating")
    trigger_columns = list(config.SUPPLY_TRIGGER_ANSWER_COLUMNS)
    if technology in config.GENERATING_TECHNOLOGIES:
        trigger_columns += config.GENERATION_TRIGGER_ANSWER_COLUMNS
    for column in trigger_columns:
        use(column, "Network triggers")

    rows = []
    for column, purposes in uses.items():
        value = application.get(column)
        answer = strong(value) if value not in (None, "") else muted(config.NOT_GIVEN_TEXT)
        rows.append([escape(answer_label(column)), answer, muted(", ".join(purposes))])
    html_table(["Question", "Answer", "Used for"], rows)


def _render_review(reference, reviews):
    user = st.session_state[SIGNED_IN_USER_KEY]
    if reviews:
        review = reviews[-1]
        badge_row([(f"Recommended: {review['recommended_decision']}",
                    decision_colour(review["recommended_decision"])),
                   (config.REVIEW_STATUS_LABELS[review["status"]], status_colour(review["status"]))],
                  meta=f"{review['review_id']} · Opened by {review['coordinator_id']}")
        spacer(4)
        notice_box(review["recommendation_text"])
        return
    if config.AREA_REVIEW not in user["areas"]:
        empty_state("No review yet", "A coordinator has not opened a review for this application.",
                    icon_name="clock")
        return

    section("Open a review", f"The agents make {config.LLM_CALLS_BEFORE_DECISION} model calls, then the "
                             "review waits for a Connections Engineer's decision.")
    with st.form(f"review_form_{reference}", border=True):
        request_text = st.text_area(
            "What should the review look at?", max_chars=config.COORDINATOR_REQUEST_MAXIMUM_CHARACTERS,
            placeholder="Check the rated current against the statements and the supply capacity.",
            height=110)
        submitted = st.form_submit_button(
            "Start review",
            type="primary",
            width="stretch",
        )
    if not submitted:
        return
    if len(request_text.strip()) < config.COORDINATOR_REQUEST_MINIMUM_CHARACTERS:
        callout(f"Write at least {config.COORDINATOR_REQUEST_MINIMUM_CHARACTERS} characters.", tone="warning")
        return
    review = call_service("POST", "/reviews", payload={"application_reference": reference,
                                                       "coordinator_request": request_text.strip()},
                          waiting_message=f"Reviewing the application · {config.LLM_CALLS_BEFORE_DECISION} "
                                          f"model calls")
    callout(f"It recommends {review['recommended_decision']} and is waiting for a Connections Engineer's "
            f"decision.", tone="success", title=f"Review {review['review_id']} is ready")
    notice_box(review["recommendation_text"])


def render_application_detail_page():
    """Show everything established about one application, and let a person open its review."""
    page_header(config.PAGE_DETAIL, "The screening, statement evidence, form answers and review of one "
                                    "application, before any model call is spent on it.")
    queue = call_service("GET", "/queue", query={"technology": config.QUEUE_TECHNOLOGY_ALL})
    if not queue:
        empty_state("The queue is empty", "There is no open application to show.")
        return
    references = [row["application_reference"] for row in queue]
    labels = {row["application_reference"]: _option_label(row) for row in queue}
    remembered = st.session_state.get(SELECTED_APPLICATION_KEY)
    index = references.index(remembered) if remembered in references else 0
    reference = st.selectbox("Application", references, index=index, format_func=labels.get)
    st.session_state[SELECTED_APPLICATION_KEY] = reference

    detail = call_service("GET", f"/queue/{reference}")
    reviews = [review for review in call_service("GET", "/reviews",
                                                 query={"status_value": config.REVIEW_STATUS_OPTIONS})
               if review["application_reference"] == reference]
    application, screening = detail["application"], detail["screening"]
    likelihood, duration = detail["likelihood"], detail["duration"]

    spacer(4)
    badges = []
    if screening["technology"] == config.HEAT_PUMP and \
            application.get(config.URGENCY_COLUMN) == config.URGENCY_HEATING_FAILED:
        badges.append((config.URGENCY_HEATING_FAILED, config.HEATING_FAILED_BADGE_COLOUR))
    badges.append((f"Assessment: {screening['assessment']}", decision_colour(screening["assessment"])))
    badges.append((f"Position {detail['queue_position']} of {len(queue)}", config.NEUTRAL_BADGE_COLOUR))
    if reviews:
        badges.append((config.REVIEW_STATUS_LABELS[reviews[-1]["status"]], status_colour(reviews[-1]["status"])))
    badge_row(badges)
    key_values([
        ("Region", escape(application.get(config.REGION_TYPE_COLUMN))),
        ("Substation", mono(application.get(config.SUBSTATION_REFERENCE_COLUMN))),
        ("Portal", escape(application.get(config.SOURCE_PORTAL_COLUMN))),
        ("Revision", escape(application.get(config.REVISION_NUMBER_COLUMN))),
        ("Screened under", escape(screening["standard_document"])),
    ])
    spacer(16)

    total = len(screening["requirements"])
    of_total = f"of {total} requirement{'s' if total != 1 else ''}"
    triggers = len(screening["network_triggers"])
    stat_row([
        {"label": "Requirements met", "value": screening[config.REQUIREMENTS_MET_COUNT_COLUMN],
         "icon": "check", "tone": "green", "hint_html": of_total},
        {"label": "Not met", "value": screening[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN],
         "icon": "x", "accent": "pw-accent-red", "hint_html": of_total},
        {"label": "Not documented", "value": screening[config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN],
         "icon": "help", "accent": "pw-accent-amber", "hint_html": of_total},
        {"label": "Network triggers", "value": triggers, "icon": "zap", "accent": "pw-accent-violet",
         "hint_html": "Network is assessed first" if triggers else "None raised"},
        {"label": "Network work likelihood", "icon": "activity",
         "value": f"{likelihood['network_work_likelihood']:.{config.LIKELIHOOD_DECIMALS}f}",
         "hint_html": badge_html(likelihood["likelihood_band"], band_colour(likelihood["likelihood_band"]))},
        {"label": "Expected working days", "value": duration["expected_working_days"], "icon": "clock",
         "hint_html": delay_badge(duration)},
    ])
    spacer(20)

    screening_tab, evidence_tab, form_tab, review_tab = st.tabs(
        ["Screening", f"Statement evidence  {len(detail['evidence_findings'])}", "Form answers", "Review"])
    with screening_tab:
        _render_screening(screening)
    with evidence_tab:
        _render_evidence(detail)
    with form_tab:
        _render_form(detail)
    with review_tab:
        _render_review(reference, reviews)
