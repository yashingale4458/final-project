"""The order the nodes run in, the gate, the two branches and the checkpoint."""

import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agentic import nodes
from agentic.state import ApplicationReviewState
from config import config

LINEAR_NODES = [
    ("load_application_context", nodes.load_application_context_node),
    ("plan_application_review", nodes.plan_application_review_node),
    ("assess_connection_requirements", nodes.assess_connection_requirements_node),
    ("analyse_installer_statements", nodes.analyse_installer_statements_node),
    ("interpret_model_predictions", nodes.interpret_model_predictions_node),
    ("prepare_review_recommendation", nodes.prepare_review_recommendation_node),
    ("await_engineer_decision", nodes.await_engineer_decision_node),
]


def build_workflow(checkpoint_path):
    """Assemble the ten nodes into a workflow that can be run, paused and resumed."""
    builder = StateGraph(ApplicationReviewState)
    for name, node in LINEAR_NODES:
        builder.add_node(name, node)
    builder.add_node("draft_installer_notice", nodes.draft_installer_notice_node)
    builder.add_node("draft_household_notice", nodes.draft_household_notice_node)
    builder.add_node("record_connection_outcome", nodes.record_connection_outcome_node)

    builder.add_edge(START, LINEAR_NODES[0][0])
    for (earlier, _), (later, _) in zip(LINEAR_NODES, LINEAR_NODES[1:]):
        builder.add_edge(earlier, later)
    builder.add_conditional_edges("await_engineer_decision", nodes.route_engineer_decision, {
        config.DECISION_APPROVE: "draft_installer_notice",
        config.DECISION_RETURN: "draft_installer_notice",
        config.DECISION_REFER: "record_connection_outcome",
    })
    builder.add_conditional_edges("draft_installer_notice", nodes.route_engineer_decision, {
        config.DECISION_APPROVE: "draft_household_notice",
        config.DECISION_RETURN: "record_connection_outcome",
    })
    builder.add_edge("draft_household_notice", "record_connection_outcome")
    builder.add_edge("record_connection_outcome", END)

    folder = os.path.dirname(checkpoint_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    return builder.compile(checkpointer=SqliteSaver(connection))
