"""Every read and write against the six tables. Nothing outside this module runs a query.

Each function works inside the session its caller opened from session_factory: a
write joins the caller's transaction, which the caller commits as one unit. A read
returns plain records, never database objects.
"""

from sqlalchemy import select, update

from persistence.models import (AgentRun, ApplicationReview, ChatTurn, LlmCall, ReviewDecision,
                                UserAccount)

_CALL_FIELDS = ["agent_name", "model_name", "prompt_tokens", "response_tokens", "total_tokens",
                "latency_milliseconds"]
_RUN_FIELDS = ["thread_id", "coordinator_id", "llm_call_count"]
_DECISION_FIELDS = ["decision", "decided_by", "decided_by_role", "decision_reason"]


def _plain_record(row):
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


# --------------------------------------------------------------------------- accounts
def insert_user_account(session, account_record):
    """Create one sign-in account in the caller's transaction."""
    session.add(UserAccount(**account_record))


def fetch_user_account(session, condition):
    """One account as a plain record, or None."""
    account = session.execute(select(UserAccount).where(condition)).scalars().first()
    return None if account is None else _plain_record(account)


def update_session_token(session, username, session_token):
    """Start or end one person's session."""
    account = session.get(UserAccount, username)
    if account is None:
        raise ValueError(f"No account has the username {username}")
    account.session_token = session_token


# --------------------------------------------------------------------------- runs and calls
def insert_agent_run(session, run_record):
    """Open the record of one workflow run."""
    session.add(AgentRun(**run_record))


def update_agent_run(session, run_id, status_value, llm_call_count):
    """Move a run on, recording what it has cost."""
    run = session.get(AgentRun, run_id)
    if run is None:
        raise ValueError(f"No agent run has the identifier {run_id}")
    run.status = status_value
    run.llm_call_count = int(llm_call_count)


def insert_llm_call(session, run_id, call_record):
    """Record one language model call against its run."""
    session.add(LlmCall(run_id=run_id, **{field: call_record[field] for field in _CALL_FIELDS}))


def fetch_llm_calls(session, run_id):
    """A run's calls, in the order they were made."""
    calls = session.execute(select(LlmCall).where(LlmCall.run_id == run_id)
                            .order_by(LlmCall.call_id)).scalars().all()
    return [{field: getattr(call, field) for field in _CALL_FIELDS} for call in calls]


# --------------------------------------------------------------------------- reviews
def insert_application_review(session, review_record):
    """Store the review waiting at the gate."""
    session.add(ApplicationReview(**review_record))


def fetch_review_records(session, condition):
    """Reviews with their run's thread, opener and call count, and their decision."""
    statement = (select(ApplicationReview, AgentRun, ReviewDecision)
                 .join(AgentRun, AgentRun.run_id == ApplicationReview.run_id)
                 .outerjoin(ReviewDecision, ReviewDecision.review_id == ApplicationReview.review_id)
                 .where(condition)
                 .order_by(ApplicationReview.application_reference, ApplicationReview.review_id))
    records = []
    for review, run, decision in session.execute(statement).all():
        record = _plain_record(review)
        record.update({field: getattr(run, field) for field in _RUN_FIELDS})
        record.update({field: (getattr(decision, field) if decision is not None else None)
                       for field in _DECISION_FIELDS})
        records.append(record)
    return records


def insert_review_decision(session, decision_record):
    """Record a Connections Engineer's decision."""
    session.add(ReviewDecision(**decision_record))


def record_review_outcome(session, review_id, outcome_record):
    """Write exactly the columns a decision changes onto its review."""
    result = session.execute(update(ApplicationReview)
                             .where(ApplicationReview.review_id == review_id)
                             .values(**outcome_record))
    if result.rowcount == 0:
        raise ValueError(f"No application review has the identifier {review_id}")


# --------------------------------------------------------------------------- chat
def insert_chat_turn(session, turn_record):
    """Record one chat exchange, refusals included."""
    session.add(ChatTurn(**turn_record))


def fetch_chat_turns(session, condition, turn_limit):
    """The most recent turns meeting the condition, newest first."""
    turns = session.execute(select(ChatTurn).where(condition)
                            .order_by(ChatTurn.turn_id.desc()).limit(int(turn_limit))).scalars().all()
    return [_plain_record(turn) for turn in turns]
