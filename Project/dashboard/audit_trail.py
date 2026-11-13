"""The audit trail page: what has been reviewed and decided, and every chat turn."""

import pandas as pd
import streamlit as st

from config import config
from dashboard.api_client import call_service
from dashboard.ui import (answer_label, badge_html, badge_row, band_colour, decision_colour, empty_state,
                          escape, html_table, key_values, muted, notice_box, page_header, result_colour,
                          section, spacer, stat_row, status_colour, strong, subheading)

AGENT_WRITE_UPS = [("Review planner", "review_plan_text"),
                   ("Requirements analyst", "requirements_analysis_text"),
                   ("Evidence analyst", "evidence_review_text"),
                   ("Risk analyst", "prediction_analysis_text"),
                   ("Review writer", "recommendation_text")]


def _render_brief(brief):
    html_table(["Supply answer", "Value"], [
        [escape(answer_label(column)),
         strong(brief.get(column)) if brief.get(column) is not None else muted(config.NOT_GIVEN_TEXT)]
        for column in config.ASSESSMENT_BRIEF_DETAIL_COLUMNS])
    spacer(10)
    html_table(["Requirement", "Result", "Answers left blank"], [
        [strong(item["requirement_name"]), badge_html(item["result"], result_colour(item["result"])),
         muted(", ".join(answer_label(column) for column in item["blank_answers"]) or "—")]
        for item in brief["requirements"]])
    spacer(10)
    planned = brief["planned_working_days"]
    key_values([
        ("Network triggers", " ".join(badge_html(trigger, "violet") for trigger in brief["network_triggers"])
         or muted("None")),
        ("Flagged sentences", escape(len(brief["flagged_statements"]))),
        ("Network work likelihood",
         f"{brief['network_work_likelihood']:.{config.LIKELIHOOD_DECIMALS}f} "
         f"{badge_html(brief['likelihood_band'], band_colour(brief['likelihood_band']))}"),
        ("Expected working days", f"{escape(brief['expected_working_days'])} "
                                  f"{muted(f'against {planned} planned' if planned is not None else 'no plan given')}"),
    ])


def _render_selected(review):
    badge_row([(config.REVIEW_STATUS_LABELS[review["status"]], status_colour(review["status"])),
               (f"Assessment: {review['assessment']}", decision_colour(review["assessment"])),
               (f"Recommended: {review['recommended_decision']}", decision_colour(review["recommended_decision"]))],
              meta=f"{review['review_id']} · {review['standard_document']}")
    spacer(4)
    items = [("Expected working days", escape(review["duration_text"]))]
    if review["decision"]:
        items += [("Decision", badge_html(review["decision"], decision_colour(review["decision"]))),
                  ("Decided by", f"{escape(review['decided_by'])} {muted(review['decided_by_role'])}"),
                  ("Reason", escape(review["decision_reason"]))]
    key_values(items)

    spacer(12)
    with st.expander("Agent write-ups"):
        for title, key in AGENT_WRITE_UPS:
            subheading(title)
            notice_box(review[key])

    if review["installer_notice_text"]:
        section(f"The {config.NOTICE_KINDS[review['decision']]}")
        notice_box(review["installer_notice_text"])
    if review["household_notice_text"]:
        section("The household notice")
        notice_box(review["household_notice_text"])
    if review["assessment_brief"]:
        section("The assessment brief")
        _render_brief(review["assessment_brief"])

    budget = call_service("GET", f"/reviews/{review['review_id']}/calls")
    section("Call budget", "Every model call this review made.")
    seconds = budget["total_latency_milliseconds"] / config.MILLISECONDS_PER_SECOND
    stat_row([
        {"label": "Model calls", "value": f"{budget['call_count']} / {budget['expected_call_count']}",
         "icon": "cpu"},
        {"label": "Total tokens", "value": f"{budget['total_tokens']:,}", "icon": "hash"},
        {"label": "Total latency", "value": f"{seconds:.{config.LATENCY_SECONDS_DECIMALS}f} s", "icon": "clock"},
    ])
    spacer(10)
    st.dataframe(pd.DataFrame(budget["calls"]), hide_index=True, use_container_width=True)


def _render_reviews():
    statuses = st.multiselect("Statuses", config.REVIEW_STATUS_OPTIONS,
                              default=config.DECIDED_REVIEW_STATUSES,
                              format_func=lambda status: config.REVIEW_STATUS_LABELS[status])
    if not statuses:
        empty_state("Choose a status", "Pick at least one status to list its reviews.")
        return
    reviews = call_service("GET", "/reviews", query={"status_value": statuses})
    if not reviews:
        empty_state("No reviews yet", "No review has these statuses.")
        return
    table = pd.DataFrame([{
        "Application": review["application_reference"],
        "Technology": config.TECHNOLOGY_LABELS[review["technology"]],
        "Status": config.REVIEW_STATUS_LABELS[review["status"]],
        "Opened by": review["coordinator_id"],
        "Decided by": review["decided_by"] or "—",
        "Model calls": review["llm_call_count"],
    } for review in reviews])
    event = st.dataframe(table, hide_index=True, use_container_width=True, on_select="rerun",
                         selection_mode="single-row", key="audit_reviews")
    selected = event.selection.rows if event is not None else []
    if not selected:
        st.caption("Select a review to see its record, notices and call budget.")
        return
    spacer(8)
    with st.container(border=True):
        _render_selected(reviews[selected[0]])


def _render_chat_turns():
    turns = call_service("GET", "/audit/chat-turns", query={"turn_limit": config.CHAT_AUDIT_MAX_ROWS})
    if not turns:
        empty_state("No questions yet", "Questions asked in the connection chat are recorded here.")
        return
    st.dataframe(pd.DataFrame([{
        "Turn": turn["turn_id"], "Asked by": turn["username"], "Role": turn["user_role"],
        "Type": turn["question_type"], "Application": turn["application_reference"] or "—",
        "Substation": turn["substation_reference"] or "—", "Refused": turn["was_refused"],
        "Question": turn["question_text"], "Answer": turn["answer_text"],
        "Sources": turn["source_documents"] or "—",
    } for turn in turns]), hide_index=True, use_container_width=True,
        column_config={"Refused": st.column_config.CheckboxColumn(width="small"),
                       "Turn": st.column_config.NumberColumn(width="small"),
                       "Question": st.column_config.TextColumn(width="medium"),
                       "Answer": st.column_config.TextColumn(width="large")})


def render_audit_trail_page():
    """Show what has been reviewed and decided, and every question the chat replied to."""
    page_header(config.PAGE_AUDIT, "Every review, decision, notice and model call, and every chat turn, "
                                   "as the database recorded it.")
    reviews_tab, chat_tab = st.tabs(["Reviews", "Chat turns"])
    with reviews_tab:
        _render_reviews()
    with chat_tab:
        _render_chat_turns()
