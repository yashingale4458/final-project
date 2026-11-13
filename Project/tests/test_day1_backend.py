"""Day 1 - Yashwanth - persistence/, access/, run_database_setup.py, agentic/state.py, sign-in routes"""
import os
import tempfile

# The engine fixes its database path when persistence/database.py is imported, so the tests import it from a
# throw-away folder: they never touch your real database/phasewise.db.
PROJECT_FOLDER, SCRATCH_FOLDER = os.getcwd(), tempfile.mkdtemp()
os.makedirs(os.path.join(SCRATCH_FOLDER, "database"))
os.chdir(SCRATCH_FOLDER)
import persistence.database  # noqa: E402
os.chdir(PROJECT_FOLDER)

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

import run_database_setup
from access.password_security import hash_password
from access.permissions import authorize_request
from agentic.state import ApplicationReviewState
from api.routes import guarded_router, service_application
from config import config
from persistence.database import session_factory
from persistence.models import ChatTurn
from persistence.repository import fetch_chat_turns, insert_chat_turn


@pytest.fixture(scope="module")
def client():
    run_database_setup.main()

    @guarded_router.get("/queue")                  # a stand-in for the Day 3 queue endpoint, to test the guard
    def queue_stand_in(signed_in_user: dict = Depends(authorize_request)):
        return {"role": signed_in_user["role"]}

    service_application.include_router(guarded_router)
    return TestClient(service_application)


def sign_in(client, username, password):
    return client.post("/auth/signin", json={"username": username, "password": password})


def test_setup_is_safe_to_run_again(client, capsys):
    run_database_setup.main()
    assert "accounts created: 0, already present: 14" in capsys.readouterr().out


def test_hash_password():
    assert hash_password("x", "salt1") == hash_password("x", "salt1") != hash_password("x", "salt2")


def test_sign_in_and_pages_by_role(client):
    reply = sign_in(client, "amira", "11111111")
    assert reply.status_code == 200 and reply.json()["pages"] == config.ROLE_PAGE_ACCESS["Connections Engineer"]
    assert sign_in(client, "zara", "33333333").json()["pages"] == ["Connection chat"]


def test_wrong_password_and_unknown_user_answered_alike(client):
    wrong, unknown = sign_in(client, "amira", "nope"), sign_in(client, "nobody", "nope")
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json() == {"detail": config.SIGN_IN_FAILED_MESSAGE}


def test_guard_401_403_200(client):
    assert client.get("/queue").status_code == 401
    advisor = sign_in(client, "joel", "33333333").json()["session_token"]
    refused = client.get("/queue", headers={config.SESSION_TOKEN_HEADER: advisor})
    assert refused.status_code == 403 and refused.json()["detail"] == "The Customer Service Advisor role does not include the queue area"
    coordinator = sign_in(client, "ravi", "22222222").json()["session_token"]
    assert client.get("/queue", params={"technology": config.QUEUE_TECHNOLOGY_ALL},
                      headers={config.SESSION_TOKEN_HEADER: coordinator}).status_code == 200


def test_one_session_per_account_and_sign_out(client):
    first = sign_in(client, "lewis", "11111111").json()["session_token"]
    second = sign_in(client, "lewis", "11111111").json()["session_token"]
    assert client.get("/queue", params={"technology": config.QUEUE_TECHNOLOGY_ALL},
                      headers={config.SESSION_TOKEN_HEADER: first}).status_code == 401
    assert client.post("/auth/signout", headers={config.SESSION_TOKEN_HEADER: second}).json() == {
        "username": "lewis", "signed_out": True}
    assert client.get("/queue", params={"technology": config.QUEUE_TECHNOLOGY_ALL},
                      headers={config.SESSION_TOKEN_HEADER: second}).status_code == 401
    assert client.post("/auth/signout").status_code == 401


def test_chat_turns_newest_first(client):
    with session_factory() as session:
        for number in range(3):
            insert_chat_turn(session, {
                "username": "zara",
                "user_role": "Customer Service Advisor",
                "question_text": f"q{number}",
                "question_type": "OUT OF SCOPE",
                "application_reference": None,
                "substation_reference": None,
                "technology": None,
                "answer_text": "a",
                "source_documents": "",
                "closest_distance": 0.0,
                "was_refused": True,
            })
        session.commit()
        turns = fetch_chat_turns(session, ChatTurn.username == "zara", 2)
    assert [turn["question_text"] for turn in turns] == ["q2", "q1"]


def test_state_counts_calls_by_adding():
    assert ApplicationReviewState.__annotations__["llm_call_count"].__metadata__[0](2, 1) == 3
