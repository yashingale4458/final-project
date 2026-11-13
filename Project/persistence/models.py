"""The six tables. Table and column names are read by the service and the marking
suite and are reproduced exactly, in lower case."""

from typing import Optional

from sqlalchemy import JSON, Boolean, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config import config


class Base(DeclarativeBase):
    """The declarative base every table inherits from."""


class UserAccount(Base):
    __tablename__ = config.USER_ACCOUNTS_TABLE

    username: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    password_salt: Mapped[str] = mapped_column(String, nullable=False)
    session_token: Mapped[Optional[str]] = mapped_column(String, unique=True, nullable=True)


class AgentRun(Base):
    __tablename__ = config.AGENT_RUNS_TABLE

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    application_reference: Mapped[str] = mapped_column(String, nullable=False)
    coordinator_id: Mapped[str] = mapped_column(String, nullable=False)
    coordinator_request: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    llm_call_count: Mapped[int] = mapped_column(Integer, nullable=False)


class ApplicationReview(Base):
    __tablename__ = config.APPLICATION_REVIEWS_TABLE

    review_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    application_reference: Mapped[str] = mapped_column(String, nullable=False)
    technology: Mapped[str] = mapped_column(String, nullable=False)
    urgency: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    standard_document: Mapped[str] = mapped_column(String, nullable=False)
    coordinator_request: Mapped[str] = mapped_column(Text, nullable=False)
    requirements: Mapped[list] = mapped_column(JSON, nullable=False)
    requirements_met_count: Mapped[int] = mapped_column(Integer, nullable=False)
    requirements_not_met_count: Mapped[int] = mapped_column(Integer, nullable=False)
    requirements_not_documented_count: Mapped[int] = mapped_column(Integer, nullable=False)
    network_triggers: Mapped[list] = mapped_column(JSON, nullable=False)
    rating_stated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    assessment: Mapped[str] = mapped_column(String, nullable=False)
    decisions_open: Mapped[list] = mapped_column(JSON, nullable=False)
    evidence: Mapped[list] = mapped_column(JSON, nullable=False)
    disagreements: Mapped[list] = mapped_column(JSON, nullable=False)
    network_work_likelihood: Mapped[float] = mapped_column(Float, nullable=False)
    likelihood_band: Mapped[str] = mapped_column(String, nullable=False)
    expected_working_days: Mapped[int] = mapped_column(Integer, nullable=False)
    planned_working_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    delay_expected: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    duration_text: Mapped[str] = mapped_column(Text, nullable=False)
    review_plan_text: Mapped[str] = mapped_column(Text, nullable=False)
    requirements_analysis_text: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_review_text: Mapped[str] = mapped_column(Text, nullable=False)
    prediction_analysis_text: Mapped[str] = mapped_column(Text, nullable=False)
    recommendation_text: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_decision: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    installer_notice_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notice_details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    household_notice_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    assessment_brief: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class ReviewDecision(Base):
    __tablename__ = config.REVIEW_DECISIONS_TABLE

    decision_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    review_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    run_id: Mapped[str] = mapped_column(String, nullable=False)
    decision: Mapped[str] = mapped_column(String, nullable=False)
    decided_by: Mapped[str] = mapped_column(String, nullable=False)
    decided_by_role: Mapped[str] = mapped_column(String, nullable=False)
    decision_reason: Mapped[str] = mapped_column(Text, nullable=False)


class LlmCall(Base):
    __tablename__ = config.LLM_CALLS_TABLE

    call_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, nullable=False)
    agent_name: Mapped[str] = mapped_column(String, nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    response_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_milliseconds: Mapped[int] = mapped_column(Integer, nullable=False)


class ChatTurn(Base):
    __tablename__ = config.CHAT_TURNS_TABLE

    turn_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String, nullable=False)
    user_role: Mapped[str] = mapped_column(String, nullable=False)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    question_type: Mapped[str] = mapped_column(String, nullable=False)
    application_reference: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    substation_reference: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    technology: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_documents: Mapped[str] = mapped_column(Text, nullable=False)
    closest_distance: Mapped[float] = mapped_column(Float, nullable=False)
    was_refused: Mapped[bool] = mapped_column(Boolean, nullable=False)
