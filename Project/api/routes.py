"""The REST service: the queue, the application detail, the review workflow, the decision,
the call budget, the chat and the audit trail, behind one sign-in and access guard and one
failure handler."""

import hmac
import os
import secrets
import threading
from typing import Annotated, Optional

import httpx
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from langgraph.types import Command
from sqlalchemy import true

from access.password_security import hash_password
from access.permissions import authorize_request
from agentic import nodes
from agentic.agents import offline_mode
from agentic.graph import build_workflow
from api.schemas import (ApplicationDetailResponse, CallBudgetResponse, ChatRequest, ChatResponse,
                         ChatTurnResponse, DecisionRequest, QueueRow, ReviewRecordResponse,
                         ReviewRequest, ReviewStatusValue, SignedInUserResponse, SignedOutResponse,
                         SignInRequest, TechnologyFilter)
from chat.connection_assistant import answer_from_documents
from chat.conversation_memory import recall_conversation
from chat.record_answer import answer_application_record, answer_substation_figure
from config import config
from persistence.database import session_factory
from persistence.models import ApplicationReview, ChatTurn, UserAccount
from persistence.repository import (fetch_chat_turns, fetch_llm_calls, fetch_review_records,
                                    fetch_user_account, insert_chat_turn, update_agent_run,
                                    update_session_token)
from tools.application_data_tools import describe_record, rank_application_queue
from tools.decision_tools import find_decision_refusal, find_review_conflict
from tools.prediction_tools import describe_expected_working_days

try:
    from google.genai.errors import APIError as LanguageModelError
except ImportError:  # pragma: no cover - google-genai is a listed requirement
    class LanguageModelError(Exception):
        """Stand-in when google-genai is not installed."""

SessionTokenHeader = Annotated[Optional[str], Header(alias=config.SESSION_TOKEN_HEADER)]
SignedInUser = Annotated[dict, Depends(authorize_request)]
TurnLimit = Annotated[int, Query(ge=1, le=config.CHAT_AUDIT_MAX_ROWS)]


# =========================================================================== built once, on import
def _without_rules(screening):
    keys = ["requirement_kind", "requirement_name", "requirement_sentence", "result", "blank_answers"]
    stripped = dict(screening)
    stripped["requirements"] = [{key: requirement[key] for key in keys}
                                for requirement in screening["requirements"]]
    return stripped


def _build_queue():
    details, queue = {}, []
    for position, reference in enumerate(rank_application_queue(nodes.open_applications), start=1):
        context = nodes.gather_application_context(reference)
        application = context["application_profile"]
        screening = context["screening"]
        words = describe_record(application)
        details[reference] = {
            "queue_position": position,
            "application": words,
            "application_documents": [describe_record(document)
                                      for document in context["application_documents"]],
            "screening": _without_rules(screening),
            "evidence_findings": context["evidence_findings"],
            "disagreements": context["disagreements"],
            "likelihood": context["likelihood"],
            "duration": context["duration"],
            "duration_text": describe_expected_working_days(context["duration"]),
        }
        queue.append({
            "queue_position": position,
            "application_reference": reference,
            "technology": screening["technology"],
            "technology_label": config.TECHNOLOGY_LABELS[screening["technology"]],
            "source_portal": words[config.SOURCE_PORTAL_COLUMN],
            "premises_type": words[config.PREMISES_TYPE_COLUMN],
            "urgency": words[config.URGENCY_COLUMN],
            "document_count": len(context["application_documents"]),
            config.REQUIREMENTS_MET_COUNT_COLUMN: screening[config.REQUIREMENTS_MET_COUNT_COLUMN],
            config.REQUIREMENTS_NOT_MET_COUNT_COLUMN: screening[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN],
            config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN:
                screening[config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN],
            "network_trigger_count": len(screening["network_triggers"]),
            "rating_stated": screening["rating_stated"],
            "assessment": screening["assessment"],
            "decisions_open": list(screening["decisions_open"]),
            "network_work_likelihood": context["likelihood"]["network_work_likelihood"],
            "likelihood_band": context["likelihood"]["likelihood_band"],
            "expected_working_days": context["duration"]["expected_working_days"],
            "delay_expected": context["duration"]["delay_expected"],
            "disagreement_count": len(context["disagreements"]),
        })
    return details, queue


review_workflow = build_workflow(config.CHECKPOINT_DATABASE_FILE)
application_details, application_queue = _build_queue()
application_locks = {reference: threading.Lock() for reference in application_details}

public_router = APIRouter()
guarded_router = APIRouter(dependencies=[Depends(authorize_request)])


def _thread_config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def _review_record(review_id):
    with session_factory() as session:
        records = fetch_review_records(session, ApplicationReview.review_id == review_id)
    return records[0] if records else None


@public_router.get("/health")
def health_check():
    """Small readiness response used by the dashboard and container probes."""
    return {
        "status": "ok",
        "service": config.API_TITLE,
        "llm_mode": "offline-demo" if offline_mode() else "gemini",
        "model": os.getenv(config.GEMINI_MODEL_VARIABLE) or "not configured",
    }


# =========================================================================== signing in and out
@public_router.post("/auth/signin", response_model=SignedInUserResponse)
def sign_in_user(sign_in_request: SignInRequest):
    """Exchange a username and password for a session token."""
    with session_factory() as session:
        account = fetch_user_account(session, UserAccount.username == sign_in_request.username)
        salt = account["password_salt"] if account else secrets.token_hex(config.PASSWORD_SALT_BYTES)
        offered = hash_password(sign_in_request.password, salt)
        stored = account["password_hash"] if account else secrets.token_hex(32)
        if account is None or not hmac.compare_digest(offered, stored):
            raise HTTPException(status_code=401, detail=config.SIGN_IN_FAILED_MESSAGE)
        token = secrets.token_urlsafe(config.SESSION_TOKEN_BYTES)
        update_session_token(session, account["username"], token)
        session.commit()
    role = account["role"]
    return {"username": account["username"], "display_name": account["display_name"], "role": role,
            "session_token": token, "pages": list(config.ROLE_PAGE_ACCESS[role]),
            "areas": list(config.ROLE_AREA_ACCESS[role])}


@public_router.post("/auth/signout", response_model=SignedOutResponse)
def sign_out_user(session_token: SessionTokenHeader = None):
    """End a session; the same token opens nothing afterwards."""
    if not session_token:
        raise HTTPException(status_code=401, detail=config.SIGN_IN_REQUIRED_MESSAGE)
    with session_factory() as session:
        account = fetch_user_account(session, UserAccount.session_token == session_token)
        if account is None:
            raise HTTPException(status_code=401, detail=config.SIGN_IN_REQUIRED_MESSAGE)
        update_session_token(session, account["username"], None)
        session.commit()
    return {"username": account["username"], "signed_out": True}


# =========================================================================== the queue
@guarded_router.get("/queue", response_model=list[QueueRow])
def list_application_queue(technology: TechnologyFilter):
    """The ranked list of open applications, optionally of one technology."""
    if technology == config.QUEUE_TECHNOLOGY_ALL:
        return application_queue
    return [row for row in application_queue if row["technology"] == technology]


@guarded_router.get("/queue/{application_reference}", response_model=ApplicationDetailResponse)
def get_application_detail(application_reference: str):
    """One application with everything the review would work from, before any model call."""
    if application_reference not in application_details:
        raise HTTPException(status_code=404, detail=config.UNKNOWN_APPLICATION_MESSAGE.format(
            application_reference=application_reference))
    return application_details[application_reference]


# =========================================================================== reviews
@guarded_router.post("/reviews", response_model=ReviewRecordResponse, status_code=201)
def start_application_review(review_request: ReviewRequest, signed_in_user: SignedInUser,
                             request: Request):
    """Run the five reviewing agents and leave the application waiting for an engineer."""
    reference = review_request.application_reference
    if reference not in application_details:
        raise HTTPException(status_code=404, detail=config.UNKNOWN_APPLICATION_MESSAGE.format(
            application_reference=reference))
    with application_locks[reference]:
        with session_factory() as session:
            existing = fetch_review_records(session, ApplicationReview.application_reference == reference)
        if existing:
            raise HTTPException(status_code=409, detail=config.APPLICATION_ALREADY_REVIEWED_MESSAGE.format(
                application_reference=reference))
        suffix = secrets.token_hex(config.IDENTIFIER_HEX_LENGTH).upper()[:config.IDENTIFIER_HEX_LENGTH]
        thread_id = f"{reference}-{suffix}"
        request.state.opened_thread_id = thread_id
        review_workflow.invoke({
            "application_reference": reference,
            "coordinator_request": review_request.coordinator_request,
            "coordinator_id": signed_in_user["username"],
            "thread_id": thread_id,
            "llm_call_count": 0,
        }, _thread_config(thread_id))
        review_id = review_workflow.get_state(_thread_config(thread_id)).values["review_id"]
    return _review_record(review_id)


@guarded_router.get("/reviews", response_model=list[ReviewRecordResponse])
def list_application_reviews(status_value: Annotated[list[ReviewStatusValue], Query()]):
    """Every review whose status is one of those asked for, in application-reference order."""
    with session_factory() as session:
        return fetch_review_records(session, ApplicationReview.status.in_(list(status_value)))


def _checkpoint_values(review):
    snapshot = review_workflow.get_state(_thread_config(review["thread_id"]))
    if not snapshot.values:
        raise ValueError(config.CHECKPOINT_MISSING_MESSAGE.format(
            review_id=review["review_id"], checkpoint_file=config.CHECKPOINT_DATABASE_FILE))
    return snapshot.values


@guarded_router.post("/reviews/{review_id}/decision", response_model=ReviewRecordResponse)
def submit_review_decision(review_id: str, decision_request: DecisionRequest,
                           signed_in_user: SignedInUser):
    """Check the decision, then resume the paused run with it."""
    review = _review_record(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail=config.REVIEW_NOT_FOUND_MESSAGE)
    reference = review["application_reference"]
    if reference not in application_locks:
        raise ValueError(config.REVIEW_OUTSIDE_QUEUE_MESSAGE.format(review_id=review_id,
                                                                    application_reference=reference))
    decision = {"decision": decision_request.decision, "decided_by": signed_in_user["username"],
                "decided_by_role": signed_in_user["role"],
                "decision_reason": decision_request.decision_reason}
    with application_locks[reference]:
        review = _review_record(review_id)
        values = _checkpoint_values(review)
        pending = values.get("review_decision")
        refusal = find_review_conflict(review["status"], pending, decision, reference) or \
            find_decision_refusal(decision, values["screening"], values["coordinator_id"])
        if refusal:
            raise HTTPException(status_code=409, detail=refusal)
        if pending:
            review_workflow.invoke(None, _thread_config(review["thread_id"]))
        else:
            review_workflow.invoke(Command(resume=decision), _thread_config(review["thread_id"]))
    return _review_record(review_id)


@guarded_router.get("/reviews/{review_id}/calls", response_model=CallBudgetResponse)
def get_call_budget(review_id: str):
    """Every language model call the review's run made, against the number allowed."""
    review = _review_record(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail=config.REVIEW_NOT_FOUND_MESSAGE)
    if review["decision"]:
        expected = config.LLM_CALLS_BY_DECISION[review["decision"]]
    else:
        pending = _checkpoint_values(review).get("review_decision")
        expected = config.LLM_CALLS_BY_DECISION[pending["decision"]] if pending \
            else config.LLM_CALLS_BEFORE_DECISION
    with session_factory() as session:
        calls = fetch_llm_calls(session, review["run_id"])
    return {"review_id": review_id, "run_id": review["run_id"], "call_count": len(calls),
            "expected_call_count": expected,
            "total_tokens": sum(call["total_tokens"] for call in calls),
            "total_latency_milliseconds": sum(call["latency_milliseconds"] for call in calls),
            "calls": calls}


# =========================================================================== chat and audit
@guarded_router.post("/chat/ask", response_model=ChatResponse)
def ask_connection_assistant(chat_request: ChatRequest, signed_in_user: SignedInUser):
    """Answer one question from the documents or the records, or refuse; record the exchange."""
    question = chat_request.question
    with session_factory() as session:
        conversation = recall_conversation(session, signed_in_user["username"], question)
    reply = answer_from_documents(nodes.knowledge_collection, nodes.knowledge_documents, question,
                                  conversation)
    with session_factory() as session:
        if reply["question_type"] in config.CHAT_APPLICATION_TYPES:
            reply["answer"] = answer_application_record(session, signed_in_user, nodes.open_applications,
                                                        reply["question_type"],
                                                        reply["application_reference"])
        elif reply["question_type"] == config.CHAT_TYPE_SUBSTATION_FIGURE:
            reply["answer"] = answer_substation_figure(nodes.connection_history,
                                                       reply["substation_reference"], reply["technology"])
        insert_chat_turn(session, {
            "username": signed_in_user["username"],
            "user_role": signed_in_user["role"],
            "question_text": question,
            "question_type": reply["question_type"],
            "application_reference": reply["application_reference"],
            "substation_reference": reply["substation_reference"],
            "technology": reply["technology"],
            "answer_text": reply["answer"],
            "source_documents": ", ".join(reply["source_documents"]),
            "closest_distance": reply["closest_distance"],
            "was_refused": reply["was_refused"],
        })
        session.commit()
    return {key: reply[key] for key in ChatResponse.model_fields}


@guarded_router.get("/chat/turns", response_model=list[ChatTurnResponse])
def list_own_chat_turns(signed_in_user: SignedInUser, turn_limit: TurnLimit):
    """The signed-in person's own conversation, newest first."""
    with session_factory() as session:
        return fetch_chat_turns(session, ChatTurn.username == signed_in_user["username"], turn_limit)


@guarded_router.get("/audit/chat-turns", response_model=list[ChatTurnResponse])
def list_recent_chat_turns(turn_limit: TurnLimit):
    """Every recent question, whoever asked it, newest first."""
    with session_factory() as session:
        return fetch_chat_turns(session, true(), turn_limit)


# =========================================================================== failures
def report_failed_request(request: Request, failure: Exception):
    """One 502 reply the dashboard can show; a run the request opened is marked failed."""
    thread_id = getattr(request.state, "opened_thread_id", None)
    if thread_id:
        try:
            values = review_workflow.get_state(_thread_config(thread_id)).values
            run_id = values.get("run_id")
            if run_id:
                with session_factory() as session:
                    update_agent_run(session, run_id, config.AGENT_RUN_STATUS_FAILED,
                                     len(fetch_llm_calls(session, run_id)))
                    session.commit()
        except Exception:  # the reply must still be sent when the run cannot be marked
            pass
    return JSONResponse(status_code=502, content={
        "detail": f"{request.method} {request.url.path} could not be completed: {failure}"})


service_application = FastAPI(title=config.API_TITLE)
service_application.include_router(public_router)
service_application.include_router(guarded_router)
for failure_type in (ValueError, LanguageModelError, httpx.HTTPError):
    service_application.add_exception_handler(failure_type, report_failed_request)
