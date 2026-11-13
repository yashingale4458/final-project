"""Recalls the last few exchanges of one person's own conversation."""

from config import config
from persistence.models import ChatTurn
from persistence.repository import fetch_chat_turns


def recall_conversation(session, username, question_text):
    """History to read and words to search with, from this person's own recent turns."""
    turns = fetch_chat_turns(session, ChatTurn.username == username, config.CHAT_HISTORY_TURN_LIMIT)
    turns = list(reversed(turns))
    if turns:
        exchanges = []
        for turn in turns:
            answer = config.CHAT_RECORD_ANSWER_IN_HISTORY \
                if turn["question_type"] in config.CHAT_RECORD_TYPES else turn["answer_text"]
            exchanges.append(f"Question: {turn['question_text']}\nAnswer: {answer}")
        history_text = "\n\n".join(exchanges)
        previous_question = turns[-1]["question_text"]
    else:
        history_text = config.CHAT_FIRST_QUESTION_TEXT
        previous_question = ""
    search_text = f"{previous_question} {question_text}".strip()
    return {"history_text": history_text, "search_text": search_text,
            "previous_question": previous_question}
