"""Day 1 - Janavi - api/schemas.py, tools/decision_tools.py (the dashboard is checked by hand, see README)"""
import pytest
from pydantic import TypeAdapter, ValidationError
from api.schemas import CoordinatorRequestText, SignInRequest, TechnologyFilter
from tools.decision_tools import find_decision_refusal, find_review_conflict

SCREENING = {"application_reference": "LCT0052118", "assessment": "RETURN TO INSTALLER",
             "decisions_open": ["RETURN TO INSTALLER", "REFER FOR NETWORK ASSESSMENT"]}
DECISION = {"decision": "RETURN TO INSTALLER", "decided_by": "amira", "decided_by_role": "Connections Engineer",
            "decision_reason": "Earth electrode resistance is missing."}


def test_schema_types():
    assert TypeAdapter(TechnologyFilter).validate_python("EVCP") == "EVCP"
    with pytest.raises(ValidationError):
        TypeAdapter(TechnologyFilter).validate_python("XYZ")
    assert TypeAdapter(CoordinatorRequestText).validate_python("  check earthing  ") == "check earthing"
    with pytest.raises(ValidationError):
        TypeAdapter(CoordinatorRequestText).validate_python("   ab ")
    with pytest.raises(ValidationError):
        SignInRequest(username="amira")


def test_refusals_in_order():
    assert find_decision_refusal(DECISION, SCREENING, "ravi") is None
    assert "gives no decision_reason" in find_decision_refusal(dict(DECISION, decision_reason=" "), SCREENING, "ravi")
    assert "is not one of the decisions" in find_decision_refusal(dict(DECISION, decision="MAYBE"), SCREENING, "ravi")
    assert "may not decide application" in find_decision_refusal(dict(DECISION, decided_by_role="Connections Coordinator"), SCREENING, "ravi")
    assert "may not decide it" in find_decision_refusal(DECISION, SCREENING, "amira")
    refusal = find_decision_refusal(dict(DECISION, decision="APPROVE CONNECTION"), SCREENING, "ravi")
    assert refusal.endswith("a requirement is Not met or Not documented, or the rating is blank")

def test_review_conflicts():
    assert "already been decided" in find_review_conflict("APPROVED", None, DECISION, "LCT0052118")
    assert find_review_conflict("AWAITING_DECISION", DECISION, dict(DECISION), "LCT0052118") is None
    assert "still being recorded" in find_review_conflict("AWAITING_DECISION", DECISION, dict(DECISION, decided_by="lewis"), "LCT0052118")
