"""The ten steps of the review workflow, the router, and the context every node and
the service share.

Each node takes the state, does one thing and returns only the keys it wrote. A
node that calls an agent records the call in llm_calls against the run as soon as
the agent function returns, and returns llm_call_count as 1.
"""

import secrets

from langgraph.types import interrupt

from agentic import agents
from config import config
from knowledge.pdf_extraction import extract_text_from_directory
from knowledge.standard_reading import read_connection_standard
from knowledge.vector_index import open_vector_index
from modeling.model_registry import load_feature_schema, load_model
from persistence.database import session_factory
from persistence.repository import (insert_agent_run, insert_application_review, insert_llm_call,
                                    insert_review_decision, record_review_outcome, update_agent_run)
from tools.application_data_tools import (fetch_application, fetch_application_documents,
                                          load_application_documents, load_connection_history,
                                          load_open_applications)
from tools.decision_tools import find_decision_refusal
from tools.evidence_tools import (find_disagreements, link_evidence_to_requirements,
                                  read_document_evidence)
from tools.notice_tools import (build_approval_details, build_assessment_brief,
                                build_household_notice_details, build_rework_details)
from tools.prediction_tools import (describe_expected_working_days, estimate_expected_working_days,
                                    estimate_network_work_likelihood)
from tools.retrieval_tools import (read_connection_approval_guidance, read_documentation_to_provide,
                                   read_household_notice_guidance, read_reviewer_checks)
from tools.screening_tools import screen_application

# =========================================================================== module-level resources
network_work_model = load_model(config.NETWORK_WORK_MODEL_FILE)
duration_model = load_model(config.DURATION_MODEL_FILE)
sentence_model = load_model(config.SENTENCE_MODEL_FILE)
network_work_feature_names = load_feature_schema(config.NETWORK_WORK_SCHEMA_FILE)
duration_feature_names = load_feature_schema(config.DURATION_SCHEMA_FILE)
open_applications = load_open_applications()
open_application_documents = load_application_documents()
connection_history = load_connection_history()
knowledge_documents = extract_text_from_directory(config.DOCS_DIRECTORY)
knowledge_collection = open_vector_index(config.VECTOR_STORE_DIRECTORY)

_standards = {}


def connection_standard(technology):
    """A technology's standard, read once from its own document."""
    if technology not in _standards:
        _standards[technology] = read_connection_standard(technology, knowledge_documents)
    return _standards[technology]


def _checkpoint_safe(screening):
    """The screening with each rule's exact decimal written as text, so it checkpoints cleanly."""
    safe = dict(screening)
    safe["requirements"] = [{**requirement,
                             "rules": {part: str(value) for part, value in requirement["rules"].items()}}
                            for requirement in screening["requirements"]]
    return safe


def _requirements_without_rules(screening):
    keys = ["requirement_kind", "requirement_name", "requirement_sentence", "result", "blank_answers"]
    return [{key: requirement[key] for key in keys} for requirement in screening["requirements"]]


def _identifier(prefix):
    return prefix + secrets.token_hex(config.IDENTIFIER_HEX_LENGTH).upper()[:config.IDENTIFIER_HEX_LENGTH]


def _record_call(run_id, call_record):
    with session_factory() as session:
        insert_llm_call(session, run_id, call_record)
        session.commit()


# =========================================================================== context
def gather_application_context(application_reference):
    """Everything about one application that needs no language model; nothing is written."""
    application = fetch_application(open_applications, application_reference)
    documents = fetch_application_documents(open_application_documents, application)
    screening = screen_application(application, connection_standard(application[config.TECHNOLOGY_COLUMN]))
    findings = link_evidence_to_requirements(read_document_evidence(sentence_model, documents), screening)
    return {
        "application_profile": application,
        "application_documents": documents,
        "screening": screening,
        "evidence_findings": findings,
        "disagreements": find_disagreements(findings, application),
        "likelihood": estimate_network_work_likelihood(network_work_model, network_work_feature_names,
                                                       application, screening),
        "duration": estimate_expected_working_days(duration_model, duration_feature_names,
                                                   application, screening),
    }


# =========================================================================== the five reviewing steps
def load_application_context_node(state):
    """Gather the application's context and open the run."""
    context = gather_application_context(state["application_reference"])
    run_id = _identifier(config.RUN_ID_PREFIX)
    review_id = _identifier(config.REVIEW_ID_PREFIX)
    with session_factory() as session:
        insert_agent_run(session, {
            "run_id": run_id,
            "thread_id": state["thread_id"],
            "application_reference": state["application_reference"],
            "coordinator_id": state["coordinator_id"],
            "coordinator_request": state["coordinator_request"],
            "status": config.AGENT_RUN_STATUS_RUNNING,
            "llm_call_count": 0,
        })
        session.commit()
    context["screening"] = _checkpoint_safe(context["screening"])
    return {"run_id": run_id, "review_id": review_id, **context}


def plan_application_review_node(state):
    """Ask the planner what the review has to establish."""
    result = agents.plan_review(state["application_profile"], state["application_documents"],
                                state["screening"], state["coordinator_request"])
    _record_call(state["run_id"], result["call_record"])
    return {"review_plan": result["review_plan"], "llm_call_count": 1}


def assess_connection_requirements_node(state):
    """Ask the requirements analyst to explain the screening."""
    result = agents.analyse_requirements(state["screening"], state["application_profile"],
                                         state["review_plan"])
    _record_call(state["run_id"], result["call_record"])
    return {"requirements_analysis": result["requirements_analysis"], "llm_call_count": 1}


def analyse_installer_statements_node(state):
    """Ask the evidence analyst to review the statements with the guides' checks in hand."""
    found_types = {finding["sentence_type"] for finding in state["evidence_findings"]}
    reviewer_checks = {sentence_type: read_reviewer_checks(knowledge_collection, sentence_type)
                       for sentence_type in config.SENTENCE_TYPE_GUIDE_DOCUMENTS
                       if sentence_type in found_types}
    result = agents.review_evidence(state["evidence_findings"], state["disagreements"],
                                    reviewer_checks, state["application_profile"],
                                    state["screening"], state["review_plan"])
    _record_call(state["run_id"], result["call_record"])
    return {"evidence_review": result["evidence_review"], "llm_call_count": 1}


def interpret_model_predictions_node(state):
    """Ask the risk analyst to explain the two figures."""
    result = agents.analyse_predictions(state["likelihood"], state["duration"],
                                        state["application_profile"], state["screening"],
                                        state["review_plan"])
    _record_call(state["run_id"], result["call_record"])
    return {"prediction_analysis": result["prediction_analysis"], "llm_call_count": 1}


def prepare_review_recommendation_node(state):
    """Ask the writer for the recommendation, and store the review waiting at the gate."""
    result = agents.write_review_recommendation(
        state["application_profile"], state["coordinator_request"], state["screening"],
        state["likelihood"], state["duration"], state["review_plan"], state["requirements_analysis"],
        state["evidence_review"], state["prediction_analysis"])
    screening = state["screening"]
    application = state["application_profile"]
    recommendation = result["review_recommendation"]
    review_record = {
        "review_id": state["review_id"],
        "run_id": state["run_id"],
        "application_reference": state["application_reference"],
        "technology": screening["technology"],
        "urgency": application.get(config.URGENCY_COLUMN),
        "standard_document": screening["standard_document"],
        "coordinator_request": state["coordinator_request"],
        "requirements": _requirements_without_rules(screening),
        config.REQUIREMENTS_MET_COUNT_COLUMN: screening[config.REQUIREMENTS_MET_COUNT_COLUMN],
        config.REQUIREMENTS_NOT_MET_COUNT_COLUMN: screening[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN],
        config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN: screening[config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN],
        "network_triggers": list(screening["network_triggers"]),
        "rating_stated": bool(screening["rating_stated"]),
        "assessment": screening["assessment"],
        "decisions_open": list(screening["decisions_open"]),
        "evidence": state["evidence_findings"],
        "disagreements": state["disagreements"],
        "network_work_likelihood": state["likelihood"]["network_work_likelihood"],
        "likelihood_band": state["likelihood"]["likelihood_band"],
        "expected_working_days": state["duration"]["expected_working_days"],
        "planned_working_days": state["duration"]["planned_working_days"],
        "delay_expected": state["duration"]["delay_expected"],
        "duration_text": describe_expected_working_days(state["duration"]),
        "review_plan_text": agents.join_labelled_response(state["review_plan"]),
        "requirements_analysis_text": agents.join_labelled_response(state["requirements_analysis"]),
        "evidence_review_text": agents.join_labelled_response(state["evidence_review"]),
        "prediction_analysis_text": agents.join_labelled_response(state["prediction_analysis"]),
        "recommendation_text": recommendation["RECOMMENDATION"],
        "recommended_decision": recommendation["RECOMMENDED DECISION"],
        "status": config.REVIEW_STATUS_AWAITING,
        "installer_notice_text": None,
        "notice_details": None,
        "household_notice_text": None,
        "assessment_brief": None,
    }
    with session_factory() as session:
        insert_llm_call(session, state["run_id"], result["call_record"])
        insert_application_review(session, review_record)
        update_agent_run(session, state["run_id"], config.AGENT_RUN_STATUS_AWAITING_DECISION,
                         state.get("llm_call_count", 0) + 1)
        session.commit()
    return {"llm_call_count": 1}


# =========================================================================== the gate
def await_engineer_decision_node(state):
    """Stop the run until a Connections Engineer decides; accept only a recordable decision."""
    review_decision = interrupt({"review_id": state["review_id"]})
    refusal = find_decision_refusal(review_decision, state["screening"], state["coordinator_id"])
    if refusal:
        raise ValueError(refusal)
    return {"review_decision": dict(review_decision)}


def route_engineer_decision(state):
    """The path out of the gate, and out of the installer notice, is named by the decision."""
    return state["review_decision"]["decision"]


# =========================================================================== after the decision
def draft_installer_notice_node(state):
    """Write the connection approval or the rework request."""
    decision = state["review_decision"]["decision"]
    application = state["application_profile"]
    screening = state["screening"]
    technology = screening["technology"]
    if decision == config.DECISION_APPROVE:
        details = build_approval_details(application, screening)
        guidance = read_connection_approval_guidance(knowledge_collection, technology)
    else:
        documentation = read_documentation_to_provide(knowledge_collection, technology)
        details = build_rework_details(application, screening, documentation, state["disagreements"])
        guidance = config.REWORK_REQUEST_GUIDANCE
    result = agents.draft_installer_notice(decision, application, details, guidance)
    _record_call(state["run_id"], result["call_record"])
    return {"installer_notice": result["installer_notice"], "notice_details": details,
            "llm_call_count": 1}


def draft_household_notice_node(state):
    """Write the household notice, after a connection approval."""
    application = state["application_profile"]
    screening = state["screening"]
    details = build_household_notice_details(application, screening)
    guidance = read_household_notice_guidance(knowledge_collection, screening["technology"])
    result = agents.draft_household_notice(application, details, guidance)
    _record_call(state["run_id"], result["call_record"])
    return {"household_notice": result["household_notice"], "llm_call_count": 1}


def record_connection_outcome_node(state):
    """Record the decision and everything it led to, and close the run, in one transaction."""
    decision = state["review_decision"]
    choice = decision["decision"]
    outcome = {
        "status": config.REVIEW_STATUS_BY_DECISION[choice],
        "installer_notice_text": None,
        "notice_details": None,
        "household_notice_text": None,
        "assessment_brief": None,
    }
    if choice in (config.DECISION_APPROVE, config.DECISION_RETURN):
        outcome["installer_notice_text"] = agents.join_labelled_response(state["installer_notice"])
        outcome["notice_details"] = state["notice_details"]
    if choice == config.DECISION_APPROVE:
        outcome["household_notice_text"] = agents.join_labelled_response(state["household_notice"])
    if choice == config.DECISION_REFER:
        outcome["assessment_brief"] = build_assessment_brief(
            state["application_profile"], state["screening"], state["likelihood"],
            state["duration"], state["disagreements"])
    with session_factory() as session:
        insert_review_decision(session, {
            "review_id": state["review_id"],
            "run_id": state["run_id"],
            "decision": choice,
            "decided_by": decision["decided_by"],
            "decided_by_role": decision["decided_by_role"],
            "decision_reason": decision["decision_reason"],
        })
        record_review_outcome(session, state["review_id"], outcome)
        update_agent_run(session, state["run_id"], config.AGENT_RUN_STATUS_COMPLETE,
                         state.get("llm_call_count", 0))
        session.commit()
    return {}
