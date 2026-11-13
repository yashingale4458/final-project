"""The application queue page."""

import pandas as pd
import streamlit as st

from config import config
from dashboard.api_client import call_service
from dashboard.ui import empty_state, escape, icon, page_header, spacer, stat_row

SELECTED_APPLICATION_KEY = "selected_application"
PAGES_KEY = "page_objects"


def _delay_text(value):
    if value is None:
        return "No plan"
    return "Delay expected" if value else "On time"


def _share(count, total):
    return f"{round(100 * count / total)}% of the queue" if total else "—"


def render_application_queue_page():
    """Show the open queue in working order, and let a person pick one application."""
    page_header(config.PAGE_QUEUE, "Every open application, screened against its own standard and "
                                   "ranked as the Low-Carbon Connections Policy sets.")
    whole_queue = call_service("GET", "/queue", query={"technology": config.QUEUE_TECHNOLOGY_ALL})
    total = len(whole_queue)
    heating_failed = sum(1 for row in whole_queue
                         if row["technology"] == config.HEAT_PUMP
                         and row["urgency"] == config.URGENCY_HEATING_FAILED)
    by_assessment = {decision: sum(1 for row in whole_queue if row["assessment"] == decision)
                     for decision in config.DECISIONS}
    approve, ret, refer = (by_assessment[config.DECISION_APPROVE], by_assessment[config.DECISION_RETURN],
                           by_assessment[config.DECISION_REFER])
    stat_row([
        {"label": "Open applications", "value": total, "icon": "layers", "hint_html": "In working order"},
        {"label": "Heating failed", "value": heating_failed, "icon": "flame", "accent": "pw-accent-red",
         "hint_html": "Moved to the front of the queue"},
        {"label": "Assessed approve", "value": approve, "icon": "check", "tone": "green",
         "hint_html": _share(approve, total)},
        {"label": "Assessed return", "value": ret, "icon": "undo", "accent": "pw-accent-amber",
         "hint_html": _share(ret, total)},
        {"label": "Assessed refer", "value": refer, "icon": "zap", "accent": "pw-accent-violet",
         "hint_html": _share(refer, total)},
    ])
    spacer(20)

    technology = st.segmented_control(
        "Technology", options=config.QUEUE_TECHNOLOGY_OPTIONS,
        format_func=lambda option: config.QUEUE_TECHNOLOGY_LABELS[option],
        default=config.QUEUE_TECHNOLOGY_ALL, key="queue_technology_filter")
    technology = technology or config.QUEUE_TECHNOLOGY_ALL
    rows = call_service("GET", "/queue", query={"technology": technology})
    if not rows:
        empty_state("No open applications", "Nothing of this technology is waiting in the queue.")
        return

    table = pd.DataFrame([{
        "Position": row["queue_position"],
        "Reference": row["application_reference"],
        "Technology": row["technology_label"],
        "Urgency": row["urgency"],
        "Assessment": row["assessment"],
        "Met": row[config.REQUIREMENTS_MET_COUNT_COLUMN],
        "Not met": row[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN],
        "Not documented": row[config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN],
        "Triggers": row["network_trigger_count"],
        "Likelihood": row["network_work_likelihood"],
        "Band": row["likelihood_band"],
        "Expected days": row["expected_working_days"],
        "Delay": _delay_text(row["delay_expected"]),
        "Flags": row["disagreement_count"],
    } for row in rows])
    st.caption(f"{len(rows)} application{'s' if len(rows) != 1 else ''} · Select a row to open it "
               f"in {config.PAGE_DETAIL.lower()}.")
    small = st.column_config.NumberColumn(width="small")
    event = st.dataframe(
        table, hide_index=True, use_container_width=True, height=min(640, 38 + 35 * len(rows)),
        on_select="rerun", selection_mode="single-row", key="queue_table",
        column_config={
            "Position": st.column_config.NumberColumn("#", width="small"),
            "Reference": st.column_config.TextColumn(width="medium"),
            "Likelihood": st.column_config.ProgressColumn(
                format=f"%.{config.LIKELIHOOD_DECIMALS}f", min_value=0.0, max_value=1.0, width="small"),
            **{name: small for name in ["Met", "Not met", "Not documented", "Triggers", "Flags",
                                        "Expected days"]},
        })

    selected = event.selection.rows if event is not None else []
    if not selected:
        return
    reference = table.iloc[selected[0]]["Reference"]
    st.session_state[SELECTED_APPLICATION_KEY] = reference
    pages = st.session_state.get(PAGES_KEY, {})
    with st.container(border=True):
        left, right = st.columns([4, 1], vertical_alignment="center")
        left.markdown(f'<div class="pw-selection">{icon("circle-check", 18)}'
                      f'<span><span class="pw-strong pw-mono">{escape(reference)}</span> selected</span></div>',
                      unsafe_allow_html=True)
        if config.PAGE_DETAIL in pages and right.button("Open application", type="primary",
                                                         use_container_width=True,
                                                         icon=":material/arrow_forward:"):
            st.switch_page(pages[config.PAGE_DETAIL])
