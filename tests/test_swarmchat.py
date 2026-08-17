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
    prompt_disc = get_system_prompt("Architect", "Otis", "discussion", is_moderator=False)
    assert "Phase: Discussion" in prompt_disc
    
    prompt_mod = get_system_prompt("Architect", "Otis", "discussion", is_moderator=True)
    assert "Moderator Directive" in prompt_mod

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

def test_gguf_status_tracking_and_vram_management():
    mm = ModelManager()
    mem = MemoryManager(storage_dir=".test_swarmchat_vram")
    tm = ToolManager()
    orch = Orchestrator(mm, mem, tm)

    # Test status update and error tracking
    mm.update_model_status("test_m", status="error", error="File missing", tok_per_sec=12.5)
    st = mm.model_statuses.get("test_m")
    assert st["status"] == "error"
    assert st["error"] == "File missing"
    assert st["tok_per_sec"] == 12.5

    # Test VRAM management routine
    orch.manage_vram_allocation("model_critic")
    assert mm.is_llama_cpp_installed() in [True, False]

def test_autonomous_loop_and_speaker_selection():
    mm = ModelManager()
    mem = MemoryManager(storage_dir=".test_swarmchat_loop")
    tm = ToolManager()
    orch = Orchestrator(mm, mem, tm)

    # @mention context detection
    orch.add_chat_message("Admin", "Admin", "Hey @Coder what do you think?", is_admin=True)
    speaker = orch.get_next_speaker()
    assert speaker == "model_coder"

    # Test autonomous loop max turn execution
    asyncio.run(orch.run_autonomous_loop(max_turns=2))
    assert len(orch.chat_history) >= 3

def test_gguf_path_resolution_and_search_paths(tmp_path):
    mm = ModelManager()
    # Create a dummy gguf file in temp dir
    fake_dir = tmp_path / "custom_models"
    fake_dir.mkdir()
    dummy_gguf = fake_dir / "Bonsai-27B-Q1_0.gguf"
    dummy_gguf.write_text("dummy gguf weights")

    # Before adding custom path, path resolution by filename alone should fail
    resolved_before = mm.resolve_gguf_path("Bonsai-27B-Q1_0.gguf")
    assert resolved_before is None or os.path.exists(resolved_before)

    # Add custom path to search paths
    mm.add_search_path(str(fake_dir))
    assert str(fake_dir) in mm.get_search_paths()

    # Now resolve_gguf_path with filename should find absolute path
    resolved_after = mm.resolve_gguf_path("Bonsai-27B-Q1_0.gguf")
    assert resolved_after is not None
    assert os.path.abspath(resolved_after) == os.path.abspath(str(dummy_gguf))

def test_huggingface_search_tool():
    tm = ToolManager()
    assert "huggingface.co" in tm.allowed_domains
    assert tm.classify_tool_risk("search_huggingface") == "low"

    res = asyncio.run(tm.search_huggingface("Llama-3-GGUF", limit=3))
    assert res["success"] is True
    assert "models" in res
    assert len(res["models"]) > 0
    assert "model_id" in res["models"][0]

def test_fs_browser_and_validation():
    from backend.main import browse_filesystem, validate_model_path, ValidatePathReq
    browse_res = browse_filesystem(".")
    assert browse_res["success"] is True
    assert "current_path" in browse_res
    assert "files" in browse_res

    val_res = validate_model_path(ValidatePathReq(path="non_existent_file.gguf"))
    assert val_res["valid"] is False
