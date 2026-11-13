"""Every request the dashboard sends to the service passes through here, so no page holds
an address, a timeout or a session token."""

import httpx
import streamlit as st

from config import config
from dashboard.ui import callout

SIGNED_IN_USER_KEY = "signed_in_user"
SESSION_ENDED_KEY = "session_ended"


def _detail(response):
    try:
        detail = response.json().get("detail")
    except ValueError:
        return f"{response.status_code}: {response.text}"
    if isinstance(detail, list):
        return "; ".join(f"{'.'.join(str(part) for part in item.get('loc', [])[1:])}: {item.get('msg')}"
                         for item in detail)
    return detail if detail else f"{response.status_code}: {response.text}"


def call_service(method, path, query=None, payload=None, waiting_message=None):
    """Make one call to the service and give the page back the reply read from JSON."""
    user = st.session_state.get(SIGNED_IN_USER_KEY)
    headers = {config.SESSION_TOKEN_HEADER: user["session_token"]} if user else {}

    def send():
        return httpx.request(method, f"{config.API_BASE_URL}{path}", params=query, json=payload,
                             headers=headers, timeout=config.API_REQUEST_TIMEOUT_SECONDS)

    try:
        if waiting_message:
            with st.spinner(waiting_message, show_time=True):
                response = send()
        else:
            response = send()
    except httpx.HTTPError as failure:
        callout(f"{config.API_BASE_URL} could not be reached. {failure}", tone="error",
                title="The service is unavailable")
        st.stop()

    if response.status_code == 401 and user:
        st.session_state.pop(SIGNED_IN_USER_KEY, None)
        st.session_state[SESSION_ENDED_KEY] = True
        st.rerun()
    if response.status_code >= 400:
        callout(_detail(response), tone="error", title="The service refused this request")
        st.stop()
    return response.json()
