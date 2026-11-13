"""The decision queue page: reviews waiting at the gate, decided by a Connections Engineer."""

import streamlit as st

from config import config
from dashboard.api_client import SIGNED_IN_USER_KEY, call_service
from dashboard.ui import (answer_label, badge_html, badge_row, band_colour, callout, decision_colour,
                          delay_badge, empty_state, escape, format_amps, html_table, mono, muted,
                          notice_box, page_header, result_colour, section, sentence_card, spacer,
                          stat_row, strong, subheading)

FLASH_KEY = "decision_flash"

AGENT_WRITE_UPS = [("Review planner", "review_plan_text"),
                   ("Requirements analyst", "requirements_analysis_text"),
                   ("Evidence analyst", "evidence_review_text"),
                   ("Risk analyst", "prediction_analysis_text")]


def _render_review_body(review):
    badge_row([(f"Recommended: {review['recommended_decision']}", decision_colour(review["recommended_decision"])),
               (f"Assessment: {review['assessment']}", decision_colour(review["assessment"]))],
              meta=f"{review['review_id']} · {review['standard_document']} · Opened by {review['coordinator_id']}")
    spacer(8)
    triggers = review["network_triggers"]
    stat_row([
        {"label": "Network work likelihood", "icon": "activity",
         "value": f"{review['network_work_likelihood']:.{config.LIKELIHOOD_DECIMALS}f}",
         "hint_html": badge_html(review["likelihood_band"], band_colour(review["likelihood_band"]))},
        {"label": "Expected working days", "value": review["expected_working_days"], "icon": "clock",
         "hint_html": delay_badge(review)},
        {"label": "Network triggers", "value": len(triggers), "icon": "zap", "accent": "pw-accent-violet",
         "hint_html": escape("; ".join(triggers) or "None raised")},
        {"label": "Flagged sentences", "value": len(review["disagreements"]), "icon": "flag",
         "accent": "pw-accent-amber",
         "hint_html": "Statement disagrees with the form" if review["disagreements"] else "None"},
    ])

    section("Recommendation")
    notice_box(review["recommendation_text"])

    section("Requirements")
    html_table(["Requirement", "Result", "Answers left blank"], [
        [strong(requirement["requirement_name"]),
         badge_html(requirement["result"], result_colour(requirement["result"])),
         muted(", ".join(answer_label(column) for column in requirement["blank_answers"]) or "—")]
        for requirement in review["requirements"]])
    if not review["rating_stated"]:
        column = config.RATING_COLUMNS[review["technology"]]
        callout(f"The {config.RATING_TERMS[column]} is blank.", tone="warning", title="Rating not stated")

    if review["disagreements"]:
        section("Flagged sentences", "Where a statement and the form disagree on the rated current.")
        for flag in review["disagreements"]:
            form = format_amps(flag["form_rating_amps"])
            sentence_card(
                flag["sentence"], badge_html(flag["flag"], "red", icon_name="flag"),
                f"{mono(flag['document_id'])} · States {escape(format_amps(flag['stated_amps']))} A; "
                f"the form gives {escape(form + ' A' if form else 'no rating')}",
                flagged=True)

    spacer(12)
    with st.expander("Coordinator's request and agent write-ups"):
        subheading("What the coordinator asked")
        notice_box(review["coordinator_request"])
        for title, key in AGENT_WRITE_UPS:
            subheading(title)
            notice_box(review[key])


def render_decision_queue_page():
    """Show the reviews waiting at the gate and let a Connections Engineer decide one."""
    page_header(config.PAGE_DECISION, "Reviews paused at the gate. Only a Connections Engineer who did "
                                      "not open a review may decide it.")
    flash = st.session_state.pop(FLASH_KEY, None)
    if flash:
        callout(flash, tone="success", title="Decision recorded")
    waiting = call_service("GET", "/reviews", query={"status_value": [config.REVIEW_STATUS_AWAITING]})
    if not waiting:
        empty_state("Nothing is waiting for a decision", "Reviews appear here once the agents have finished.")
        return
    labels = {review["review_id"]: f"{review['application_reference']}  ·  "
                                   f"{config.TECHNOLOGY_LABELS[review['technology']]}"
              for review in waiting}
    review_id = st.selectbox(f"Waiting for a decision ({len(waiting)})", list(labels), format_func=labels.get)
    review = next(item for item in waiting if item["review_id"] == review_id)
    spacer(4)
    _render_review_body(review)

    section("Record the decision")
    user = st.session_state[SIGNED_IN_USER_KEY]
    if review["coordinator_id"] == user["username"]:
        callout("You opened this review, so another Connections Engineer must decide it.",
                title="Decision held for a colleague")
        return
    with st.form(f"decision_form_{review_id}", border=True):
        open_decisions = review["decisions_open"]
        default = open_decisions.index(review["recommended_decision"]) \
            if review["recommended_decision"] in open_decisions else 0
        decision = st.radio("Decision", open_decisions, horizontal=True, index=default)
        reason = st.text_area("Reason, in your own words",
                              max_chars=config.DECISION_REASON_MAXIMUM_CHARACTERS, height=100)
        submitted = st.form_submit_button("Record decision", type="primary")
    if not submitted:
        return
    if len(reason.strip()) < config.DECISION_REASON_MINIMUM_CHARACTERS:
        callout(f"Write a reason of at least {config.DECISION_REASON_MINIMUM_CHARACTERS} characters.",
                tone="warning")
        return
    decided = call_service("POST", f"/reviews/{review_id}/decision",
                           payload={"decision": decision, "decision_reason": reason.strip()},
                           waiting_message="Recording the decision and writing what it calls for")
    st.session_state[FLASH_KEY] = (
        f"{decided['application_reference']} is {config.REVIEW_STATUS_PHRASES[decided['status']]}: "
        f"{config.DECISION_OUTCOME_TEXTS[decided['decision']]}.")
    st.rerun()
