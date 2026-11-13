"""The shape of every request the service accepts and every reply it returns, declared once."""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, StringConstraints

from config import config

TechnologyFilter = Literal[tuple(config.QUEUE_TECHNOLOGY_OPTIONS)]
ReviewStatusValue = Literal[tuple(config.REVIEW_STATUS_OPTIONS)]
DecisionValue = Literal[tuple(config.DECISIONS)]
CoordinatorRequestText = Annotated[str, StringConstraints(
    strip_whitespace=True, min_length=config.COORDINATOR_REQUEST_MINIMUM_CHARACTERS,
    max_length=config.COORDINATOR_REQUEST_MAXIMUM_CHARACTERS)]
DecisionReasonText = Annotated[str, StringConstraints(
    strip_whitespace=True, min_length=config.DECISION_REASON_MINIMUM_CHARACTERS,
    max_length=config.DECISION_REASON_MAXIMUM_CHARACTERS)]
QuestionText = Annotated[str, StringConstraints(
    strip_whitespace=True, min_length=config.CHAT_QUESTION_MINIMUM_CHARACTERS,
    max_length=config.CHAT_QUESTION_MAXIMUM_CHARACTERS)]


class SignInRequest(BaseModel):
    username: str
    password: str


class SignedInUserResponse(BaseModel):
    username: str
    display_name: str
    role: str
    session_token: str
    pages: list[str]
    areas: list[str]


class SignedOutResponse(BaseModel):
    username: str
    signed_out: bool


class QueueRow(BaseModel):
    queue_position: int
    application_reference: str
    technology: str
    technology_label: str
    source_portal: str
    premises_type: str
    urgency: str
    document_count: int
    requirements_met_count: int
    requirements_not_met_count: int
    requirements_not_documented_count: int
    network_trigger_count: int
    rating_stated: bool
    assessment: str
    decisions_open: list[str]
    network_work_likelihood: float
    likelihood_band: str
    expected_working_days: int
    delay_expected: Optional[bool]
    disagreement_count: int


class ApplicationDetailResponse(BaseModel):
    queue_position: int
    application: dict
    application_documents: list[dict]
    screening: dict
    evidence_findings: list[dict]
    disagreements: list[dict]
    likelihood: dict
    duration: dict
    duration_text: str


class ReviewRequest(BaseModel):
    application_reference: str
    coordinator_request: CoordinatorRequestText


class DecisionRequest(BaseModel):
    decision: DecisionValue
    decision_reason: DecisionReasonText


class ReviewRecordResponse(BaseModel):
    review_id: str
    run_id: str
    thread_id: str
    coordinator_id: str
    application_reference: str
    technology: str
    urgency: Optional[str]
    standard_document: str
    coordinator_request: str
    requirements: list[dict]
    requirements_met_count: int
    requirements_not_met_count: int
    requirements_not_documented_count: int
    network_triggers: list[str]
    rating_stated: bool
    assessment: str
    decisions_open: list[str]
    evidence: list[dict]
    disagreements: list[dict]
    network_work_likelihood: float
    likelihood_band: str
    expected_working_days: int
    planned_working_days: Optional[int]
    delay_expected: Optional[bool]
    duration_text: str
    review_plan_text: str
    requirements_analysis_text: str
    evidence_review_text: str
    prediction_analysis_text: str
    recommendation_text: str
    recommended_decision: str
    status: str
    installer_notice_text: Optional[str]
    notice_details: Optional[dict]
    household_notice_text: Optional[str]
    assessment_brief: Optional[dict]
    llm_call_count: int
    decision: Optional[str]
    decided_by: Optional[str]
    decided_by_role: Optional[str]
    decision_reason: Optional[str]


class CallBudgetResponse(BaseModel):
    review_id: str
    run_id: str
    call_count: int
    expected_call_count: int
    total_tokens: int
    total_latency_milliseconds: int
    calls: list[dict]


class ChatRequest(BaseModel):
    question: QuestionText


class ChatResponse(BaseModel):
    question_type: str
    application_reference: Optional[str]
    substation_reference: Optional[str]
    technology: Optional[str]
    answer: str
    source_documents: list[str]
    was_refused: bool


class ChatTurnResponse(BaseModel):
    turn_id: int
    username: str
    user_role: str
    question_text: str
    question_type: str
    application_reference: Optional[str]
    substation_reference: Optional[str]
    technology: Optional[str]
    answer_text: str
    source_documents: str
    closest_distance: float
    was_refused: bool
