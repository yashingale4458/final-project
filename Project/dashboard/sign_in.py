"""The sign-in page."""

import httpx
import streamlit as st

from config import config
from dashboard.api_client import SESSION_ENDED_KEY, SIGNED_IN_USER_KEY
from dashboard.ui import brand_mark, callout, escape, spacer


def _sign_in(username, password):
    """Return the signed-in user, or None after showing why it failed."""
    try:
        response = httpx.post(
            f"{config.API_BASE_URL}/auth/signin",
            json={
                "username": username.strip(),
                "password": password,
            },
            timeout=config.API_REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as failure:
        callout(
            f"{config.API_BASE_URL} could not be reached. {failure}",
            tone="error",
            title="The service is unavailable",
        )
        return None

    if response.status_code == 200:
        return response.json()

    try:
        message = (
            response.json().get("detail")
            or config.SIGN_IN_FAILED_MESSAGE
        )
    except ValueError:
        message = f"{response.status_code}: {response.text}"

    callout(message, tone="error")
    return None


def render_sign_in_page():
    """Let a person sign in."""

    spacer(48)
    with st.container(key="pw_signin_shell"):
        st.markdown(
            f'<div class="pw-signin-brand">'
            f'{brand_mark(42)}'
            f'<div class="pw-signin-brand-copy">'
            f'<div class="pw-signin-name">{escape(config.DASHBOARD_TITLE)}</div>'
            f'<div class="pw-signin-sub">{escape(config.DASHBOARD_SUBTITLE)}</div>'
            f'</div>'
            f'</div>'
            '<div class="pw-signin-intro">'
            '<div class="pw-form-title">Welcome back</div>'
            '<div class="pw-form-caption">Sign in to continue to your staff workspace.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.pop(SESSION_ENDED_KEY, False):
            callout(config.SESSION_ENDED_MESSAGE, tone="warning")

        with st.form("sign_in_form", border=False):
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                autocomplete="username",
            )

            password = st.text_input(
                "Password",
                type="password",
                autocomplete="current-password",
            )

            submitted = st.form_submit_button(
                "Sign in",
                type="primary",
                width="stretch",
            )

        st.markdown(
            '<div class="pw-footnote">'
            '<span class="pw-footnote-mark">✓</span>'
            '<span>Secure staff access<br>Reviews remain subject to engineer approval.</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        if submitted:
            if not username.strip() or not password:
                callout(
                    "Enter both a username and a password.",
                    tone="warning",
                )
                return

            user = _sign_in(username, password)

            if user is not None:
                st.session_state[SIGNED_IN_USER_KEY] = user
                st.rerun()
