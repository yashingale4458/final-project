"""What the review workflow carries from one node to the next."""

import operator
from typing import Annotated, TypedDict


class ApplicationReviewState(TypedDict, total=False):
    """Every key the workflow carries; llm_call_count is added to, every other key replaced."""

    application_reference: str
    coordinator_request: str
    coordinator_id: str
    thread_id: str
    run_id: str
    review_id: str
    application_profile: dict
    application_documents: list
    screening: dict
    evidence_findings: list
    disagreements: list
    likelihood: dict
    duration: dict
    review_plan: dict
    requirements_analysis: dict
    evidence_review: dict
    prediction_analysis: dict
    review_decision: dict
    installer_notice: dict
    notice_details: dict
    household_notice: dict
    llm_call_count: Annotated[int, operator.add]
