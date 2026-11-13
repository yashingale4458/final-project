"""The connection chat page."""

import streamlit as st

from config import config
from dashboard.api_client import call_service
from dashboard.ui import badge_html, callout, chat_type_colour, escape, page_header, section

SUGGESTIONS = [
    "What does the heat pump standard require?",
    "What does the reviewer check in an earthing statement?",
    "When does a looped service mean the network is assessed first?",
]
PENDING_KEY = "chat_pending"


def _unique_turns(turns):
    """Keep one visible copy when a chat exchange has been recorded twice."""
    seen = set()
    unique = []
    for turn in turns:
        key = (turn["question_text"], turn["question_type"], turn["answer_text"],
               turn["source_documents"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(turn)
    return unique


def _render_turn(turn):
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(escape(turn["question_text"]))
    with st.chat_message("assistant", avatar=":material/electric_bolt:"):
        st.markdown(badge_html(turn["question_type"], chat_type_colour(turn["question_type"])),
                    unsafe_allow_html=True)
        st.markdown(turn["answer_text"].replace("\n", "  \n"))
        if turn["source_documents"]:
            st.caption(f":material/description: Sources: {turn['source_documents']}")


def render_connection_chat_page():
    """Let anyone signed in ask the connection assistant a question."""
    page_header(config.PAGE_CHAT, "Answers come from Merrowdale's documents and records only. "
                                  "Advisors never receive a prediction.")
    health = st.session_state.get("service_health", {})
    model_label = health.get("model", "configured model")
    mode_label = "live Gemini" if health.get("llm_mode") == "gemini" else "offline demo"
    st.caption(f"Responses are generated with {mode_label} · {model_label}. Sources are shown below each answer.")
    turns = call_service("GET", "/chat/turns", query={"turn_limit": config.CHAT_PAGE_TURN_LIMIT})
    if not turns:
        section("Start with a question", "A few things people often ask.")
        for column, suggestion in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
            if column.button(suggestion, use_container_width=True, icon=":material/arrow_outward:"):
                st.session_state[PENDING_KEY] = suggestion
    for turn in reversed(_unique_turns(turns)):
        _render_turn(turn)

    question = st.chat_input("Ask about a standard, a rule, a statement guide or an application",
                             max_chars=config.CHAT_QUESTION_MAXIMUM_CHARACTERS)
    question = question or st.session_state.pop(PENDING_KEY, None)
    if question is None:
        return
    if len(question.strip()) < config.CHAT_QUESTION_MINIMUM_CHARACTERS:
        callout(f"Ask a question of at least {config.CHAT_QUESTION_MINIMUM_CHARACTERS} characters.",
                tone="warning")
        return
    call_service("POST", "/chat/ask", payload={"question": question.strip()},
                 waiting_message="Reading the documents")
    st.rerun()
