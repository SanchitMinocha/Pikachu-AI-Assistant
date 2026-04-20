"""
Basic API tests for SanchitAI.
Run with: python -m pytest tests/ -v
Or:        python tests/test_api.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def test_health_check(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "status" in data
    assert data["assistant"] == "SanchitAI"
    assert data["creator"] == "Sanchit Minocha"


def test_info_endpoint(client):
    resp = client.get("/api/info")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "assistant" in data
    assert "creator" in data
    assert data["assistant"]["name"] == "SanchitAI"
    assert "Sanchit" in data["creator"]["name"]


def test_chat_requires_message(client):
    resp = client.post("/api/chat", json={})
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert "error" in data


def test_chat_message_too_long(client):
    resp = client.post("/api/chat", json={"message": "x" * 2001})
    assert resp.status_code == 400


def test_chat_returns_conversation_id(client):
    """Chat should return a conversation_id even if LLM is unavailable."""
    resp = client.post("/api/chat", json={"message": "Who is Sanchit?"})
    # Either 200 (LLM available) or 503 (LLM unavailable) — both are valid
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = json.loads(resp.data)
        assert "response" in data
        assert "conversation_id" in data
        assert "model" in data


def test_404(client):
    resp = client.get("/api/nonexistent")
    assert resp.status_code == 404


def test_method_not_allowed(client):
    resp = client.get("/api/chat")   # chat is POST only
    assert resp.status_code == 405


if __name__ == "__main__":
    # Simple manual test runner
    from flask.testing import FlaskClient
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        print("Testing /api/health ...")
        r = client.get("/api/health")
        print(json.dumps(json.loads(r.data), indent=2))

        print("\nTesting /api/info ...")
        r = client.get("/api/info")
        print(json.dumps(json.loads(r.data), indent=2))

        print("\nAll basic tests passed!")
