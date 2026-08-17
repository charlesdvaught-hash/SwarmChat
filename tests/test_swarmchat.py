import sys
import os

sys.path.insert(0, os.path.abspath("."))

import pytest
import asyncio
from backend.models import ModelManager
from backend.memory import MemoryManager
from backend.tools import ToolManager
from backend.orchestrator import Orchestrator
from backend.evaluate import EvaluateEngine
from backend.prompts import get_system_prompt

def test_hardware_sensing():
    mm = ModelManager()
    info = mm.get_hardware_info()
    assert "ram_total_gb" in info
    assert "ram_available_gb" in info
    assert isinstance(info["ram_total_gb"], float)

def test_memory_manager():
    mem = MemoryManager(storage_dir=".test_swarmchat")
    mem.set_phase("discussion")
    assert mem.get_phase() == "discussion"
    mem.add_entry("Architect", "Test decision")
    assert "Test decision" in mem.get_memory_summary()
    mem.set_phase("execution")
    assert mem.get_phase() == "execution"

def test_thinking_disabled_prompts():
    prompt_disc = get_system_prompt("Architect", "discussion", is_moderator=False)
    assert "THINKING / REASONING MODE: DISABLED BY DEFAULT" in prompt_disc
    
    prompt_mod = get_system_prompt("Architect", "discussion", is_moderator=True)
    assert "MODERATOR REASONING MODE: BRIEF & FOCUSED" in prompt_mod

def test_tool_risk_classification():
    tm = ToolManager()
    assert tm.classify_tool_risk("read_file") == "low"
    assert tm.classify_tool_risk("write_file") == "consequential"
    assert tm.classify_tool_risk("run_terminal_cmd") == "high"

def test_orchestrator_multi_model_turn():
    mm = ModelManager()
    mem = MemoryManager(storage_dir=".test_swarmchat")
    tm = ToolManager()
    orch = Orchestrator(mm, mem, tm)

    orch.add_chat_message("Admin", "Admin", "Let's begin requirements discussion", is_admin=True)
    speaker = orch.get_next_speaker()
    assert speaker == "model_architect"

    res = asyncio.run(orch.step_model_turn(speaker))
    assert res["sender"] == "Architect"
    assert len(orch.chat_history) == 2

def test_tool_voting_and_admin_override():
    mm = ModelManager()
    mem = MemoryManager(storage_dir=".test_swarmchat")
    tm = ToolManager()
    orch = Orchestrator(mm, mem, tm)

    vote_req = orch.propose_tool_call("model_coder", "write_file", {"filepath": "test.txt", "content": "hello world"})
    assert vote_req["risk_level"] == "consequential"
    assert len(orch.pending_tool_votes) == 1

    vote_id = vote_req["id"]
    override_res = orch.admin_override_vote(vote_id, "approve")
    assert override_res["success"] is True
    assert override_res["executed"] is True

def test_evaluate_engine():
    mm = ModelManager()
    ee = EvaluateEngine(mm)
    candidates = [
        {"id": "cand_1", "name": "Llama 1B", "role": "Architect", "provider": "ollama"},
        {"id": "cand_2", "name": "Bonsai 1.7B", "role": "Coder", "provider": "gguf_local"}
    ]
    res = asyncio.run(ee.run_candidate_evaluation(candidates, "Build a file parser"))
    assert "rankings" in res
    assert len(res["rankings"]) == 2

def test_tiny_gguf_models_interaction_and_memory():
    # Verify that two tiny GGUF models can be configured, interact, and write self-journals to shared memory
    mm = ModelManager()
    mem = MemoryManager(storage_dir=".test_swarmchat_gguf")
    tm = ToolManager()
    orch = Orchestrator(mm, mem, tm)

    # Configure two authentic tiny GGUF models in known models and active room
    tiny_gguf_1 = {
        "id": "gguf_qwen_0_5b",
        "name": "Qwen 0.5B Architect",
        "role": "Architect",
        "provider": "gguf_local",
        "model_name": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "gguf_path": "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "enabled": True,
        "is_moderator": True,
        "status": "active",
        "max_context_tokens": 2048
    }

    tiny_gguf_2 = {
        "id": "gguf_qwen_0_8b",
        "name": "Qwen 0.8B Coder",
        "role": "Coder",
        "provider": "gguf_local",
        "model_name": "qwen2.5-0.8b-instruct-q4_k_m.gguf",
        "gguf_path": "/models/qwen2.5-0.8b-instruct-q4_k_m.gguf",
        "enabled": True,
        "is_moderator": False,
        "status": "active",
        "max_context_tokens": 2048
    }

    orch.add_or_update_known_model(tiny_gguf_1)
    orch.add_or_update_known_model(tiny_gguf_2)

    assert "gguf_qwen_0_5b" in orch.models
    assert "gguf_qwen_0_8b" in orch.models

    # Step turns for both models and check response & memory journaling
    msg1 = asyncio.run(orch.step_model_turn("gguf_qwen_0_5b"))
    assert msg1["sender"] == "Qwen 0.5B Architect"

    msg2 = asyncio.run(orch.step_model_turn("gguf_qwen_0_8b"))
    assert msg2["sender"] == "Qwen 0.8B Coder"

    # Test self-journaling and napping
    mem.record_model_nap("gguf_qwen_0_5b", "Qwen 0.5B Architect self-journal summary: Architecture layout approved.")
    latest_journal = mem.get_model_latest_journal("gguf_qwen_0_5b")
    assert "Architecture layout approved" in latest_journal

    # Test kicking a model and re-adding from known library
    kick_res = orch.kick_model_from_room("gguf_qwen_0_5b")
    assert kick_res["was_moderator"] is True
    assert "gguf_qwen_0_5b" not in orch.models
    assert "gguf_qwen_0_5b" in orch.known_models

    readd_res = orch.readd_model_to_room("gguf_qwen_0_5b")
    assert readd_res["success"] is True
    assert "gguf_qwen_0_5b" in orch.models
