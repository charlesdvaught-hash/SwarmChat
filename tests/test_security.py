import sys
import os

sys.path.insert(0, os.path.abspath("."))

import pytest
from fastapi.testclient import TestClient

from backend import security
from backend.main import app
from backend.tools import ToolManager


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("SWARMCHAT_API_TOKEN", raising=False)
    return TestClient(app, client=("127.0.0.1", 5555))


def test_loopback_client_allowed_without_token(client):
    assert client.get("/api/hardware").status_code == 200


def test_remote_client_denied_without_token(monkeypatch):
    monkeypatch.delenv("SWARMCHAT_API_TOKEN", raising=False)
    with TestClient(app, client=("203.0.113.7", 5555)) as remote:
        assert remote.get("/api/hardware").status_code == 403


def test_token_required_when_configured(monkeypatch):
    monkeypatch.setenv("SWARMCHAT_API_TOKEN", "s3cret-token")
    with TestClient(app, client=("203.0.113.7", 5555)) as c:
        assert c.get("/api/hardware").status_code == 401
        assert c.get("/api/hardware", headers={"X-SwarmChat-Token": "wrong"}).status_code == 401
        assert c.get("/api/hardware", headers={"X-SwarmChat-Token": "s3cret-token"}).status_code == 200
        assert c.get("/api/hardware", headers={"Authorization": "Bearer s3cret-token"}).status_code == 200


def test_foreign_origin_denied(client):
    assert client.get("/api/hardware", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/hardware", headers={"Origin": "http://localhost:5173"}).status_code == 200


def test_state_does_not_leak_api_keys(client):
    res = client.post(
        "/api/models/configure",
        json={
            "id": "cloud_model",
            "name": "Cloud",
            "role": "Architect",
            "provider": "claude",
            "api_key": "sk-should-not-leak",
        },
    )
    assert res.status_code == 200
    assert "sk-should-not-leak" not in res.text

    state = client.get("/api/state")
    assert state.status_code == 200
    assert "sk-should-not-leak" not in state.text
    assert res.json()["known_models"]["cloud_model"]["api_key_set"] is True


def test_invalid_request_values_rejected(client):
    assert client.post("/api/phase", json={"phase": "root_shell"}).status_code == 422
    traversal = {"id": "../../etc", "name": "x", "role": "Architect", "provider": "ollama"}
    assert client.post("/api/models/configure", json=traversal).status_code == 422
    bad_provider = {"id": "ok", "name": "x", "role": "Architect", "provider": "hackerprovider"}
    assert client.post("/api/models/configure", json=bad_provider).status_code == 422


def test_redaction_keeps_other_fields():
    redacted = security.redact_model_config({"model_id": "m", "api_key": "abc", "provider": "groq"})
    assert redacted["api_key"] == ""
    assert redacted["api_key_set"] is True
    assert redacted["provider"] == "groq"


def test_safe_id_validation():
    assert security.validate_safe_id("model_1") == "model_1"
    for bad in ["", "../etc", "a/b", "-leading", "x" * 65]:
        with pytest.raises(ValueError):
            security.validate_safe_id(bad)


def test_terminal_command_cannot_chain_extra_commands():
    tm = ToolManager(workspace_root=".test_swarmchat_security")
    res = tm.run_terminal_cmd("echo hello && echo pwned")
    assert res["success"] is True
    # The chained command is echoed as a literal argument instead of being executed as a second command.
    assert res.get("stdout", "").strip() == "hello && echo pwned"
    assert tm.run_terminal_cmd("")["success"] is False


def test_file_access_outside_workspace_denied():
    tm = ToolManager(workspace_root=".test_swarmchat_security")
    res = tm.read_file("../../../../etc/passwd")
    assert res["success"] is False
    assert "denied" in res["error"].lower()


def test_bot_id_must_be_safe():
    tm = ToolManager(workspace_root=".test_swarmchat_security")
    with pytest.raises(ValueError):
        tm.get_bot_workspace_dir("../escape")
