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
