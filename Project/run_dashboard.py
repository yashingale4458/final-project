"""Entry point 6: the dashboard's front door — the service check, signing in, and the
navigation. Run it through Streamlit:

    python -m streamlit run run_dashboard.py
"""

import httpx
import streamlit as st

from config import config
from dashboard.api_client import SIGNED_IN_USER_KEY, call_service
from dashboard.application_detail import render_application_detail_page
from dashboard.application_queue import PAGES_KEY, render_application_queue_page
from dashboard.audit_trail import render_audit_trail_page
from dashboard.connection_chat import render_connection_chat_page
from dashboard.decision_queue import render_decision_queue_page
from dashboard.sign_in import render_sign_in_page
from dashboard.ui import escape, inject_theme

PAGE_RENDERERS = {
    config.PAGE_QUEUE: render_application_queue_page,
    config.PAGE_DETAIL: render_application_detail_page,
    config.PAGE_DECISION: render_decision_queue_page,
    config.PAGE_AUDIT: render_audit_trail_page,
    config.PAGE_CHAT: render_connection_chat_page,
}
PAGE_ICONS = {
    config.PAGE_QUEUE: ":material/format_list_numbered:",
    config.PAGE_DETAIL: ":material/description:",
    config.PAGE_DECISION: ":material/gavel:",
    config.PAGE_AUDIT: ":material/history:",
    config.PAGE_CHAT: ":material/forum:",
}
PAGE_URL_PATHS = {
    config.PAGE_QUEUE: "queue",
    config.PAGE_DETAIL: "application",
    config.PAGE_DECISION: "decisions",
    config.PAGE_AUDIT: "audit",
    config.PAGE_CHAT: "chat",
}


def _service_answers():
    try:
        response = httpx.get(f"{config.API_BASE_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except httpx.HTTPError:
        return None


def _sign_out():
    call_service("POST", "/auth/signout")
    st.session_state.clear()


def main():
    """One screen to work from, with only the pages the signed-in person's role is offered."""
    st.set_page_config(page_title=config.DASHBOARD_TITLE, page_icon="⚡", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(config.DASHBOARD_PAGE_STYLE, unsafe_allow_html=True)
    inject_theme()
    health = _service_answers()
    if not health:
        st.error(f"The PhaseWise AI service is not answering at {config.API_BASE_URL}. Start it from "
                 f"the Project folder with: python run_api_service.py", icon=":material/cloud_off:")
        return
    st.session_state["service_health"] = health

    user = st.session_state.get(SIGNED_IN_USER_KEY)
    if user is None:
        st.navigation([st.Page(render_sign_in_page, title="Sign in", icon=":material/login:",
                               url_path="signin", default=True)], position="hidden").run()
        return

    pages = {name: st.Page(PAGE_RENDERERS[name], title=name, icon=PAGE_ICONS[name],
                           url_path=PAGE_URL_PATHS[name], default=(position == 0))
             for position, name in enumerate(user["pages"])}
    st.session_state[PAGES_KEY] = pages
    navigation = st.navigation(list(pages.values()))
    with st.sidebar:
        st.markdown(f"<div style='font-size:1.25rem;font-weight:800;margin:4px 0 2px 0'>⚡ "
                    f"{escape(config.DASHBOARD_TITLE)}</div>"
                    f"<div style='opacity:.75;font-size:.82rem;margin-bottom:14px'>"
                    f"{escape(config.ORGANISATION_NAME)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding:12px 14px;border-radius:12px;background:rgba(255,255,255,.07);"
                    f"border:1px solid rgba(255,255,255,.12)'><div style='font-weight:700'>"
                    f"{escape(user['display_name'])}</div><div style='opacity:.8;font-size:.85rem'>"
                    f"{escape(user['role'])}</div></div>", unsafe_allow_html=True)
        mode = health.get("llm_mode", "unknown")
        mode_label = "Gemini live" if mode == "gemini" else "Offline demo"
        mode_colour = "#0E5E4E" if mode == "gemini" else "#9A6700"
        st.markdown(
            f"<div class='pw-runtime-status'><span style='background:{mode_colour}'></span>"
            f"{mode_label}<small>{escape(health.get('model', ''))}</small></div>",
            unsafe_allow_html=True)
        st.write("")
        if st.button("Sign out", icon=":material/logout:", use_container_width=True):
            _sign_out()
            st.rerun()
    navigation.run()


if __name__ == "__main__":
    main()
