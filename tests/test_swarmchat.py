import sys
import os

sys.path.insert(0, os.path.abspath("."))

import asyncio
import httpx
import pytest
from backend.errors import MemoryPersistenceError, ModelInvocationError
from backend.models import ModelManager
from backend.memory import MemoryManager
from backend.tools import ToolManager
from backend.orchestrator import Orchestrator
from backend.evaluate import EvaluateEngine
from backend.prompts import PromptTemplateManager, get_system_prompt
from fastapi import HTTPException


def stub_generation(model_manager: ModelManager, text: str = "Stubbed model turn."):
    """Replaces model generation with a deterministic response (no backend required)."""
    async def _generate(model_config, system_prompt, messages):
        return text
    model_manager.generate_response = _generate


def fail_generation(model_manager: ModelManager, message: str = "backend unavailable"):
    async def _generate(model_config, system_prompt, messages):
        raise ModelInvocationError(message, model_id=model_config.get("id", ""))
    model_manager.generate_response = _generate

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
    
    # The "MODERATOR / CHIEF PROJECT MANAGER DIRECTIVE" block was deliberately removed: it
    # told the moderator to assign participant turns, which it cannot do (speaker selection
    # is entirely orchestrator-side) and which made the Architect emit turn-order chatter
    # instead of planning. is_moderator must not change the prompt's mandate any more.
    prompt_mod = get_system_prompt("Architect", "Otis", "discussion", is_moderator=True)
    assert "CHIEF PROJECT MANAGER" not in prompt_mod
    assert "Phase: Discussion" in prompt_mod

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

    stub_generation(mm)

    orch.add_chat_message("Admin", "Admin", "Let's begin requirements discussion", is_admin=True)
    speaker = orch.get_next_speaker()
    assert speaker == "model_architect"

    res = asyncio.run(orch.step_model_turn(speaker))
    assert res["sender"] == "Architect"
    assert len(orch.chat_history) == 2

def test_model_failure_is_reported_in_chat():
    """A failing model backend must be reported, never replaced by plausible filler text."""
    mm = ModelManager()
    mem = MemoryManager(storage_dir=".test_swarmchat")
    orch = Orchestrator(mm, mem, ToolManager())
    fail_generation(mm, "Ollama is not reachable at http://localhost:11434")

    msg = asyncio.run(orch.step_model_turn("model_architect"))
    assert "[MODEL ERROR]" in msg["content"]
    assert "Ollama is not reachable" in msg["content"]
    assert msg["role"] == "System"

def test_step_turn_rejects_unknown_and_disabled_models():
    mm = ModelManager()
    mem = MemoryManager(storage_dir=".test_swarmchat")
    orch = Orchestrator(mm, mem, ToolManager())

    with pytest.raises(KeyError):
        asyncio.run(orch.step_model_turn("does_not_exist"))

    orch.models["model_coder"]["enabled"] = False
    with pytest.raises(ValueError):
        asyncio.run(orch.step_model_turn("model_coder"))

def test_malformed_directive_is_announced():
    mm = ModelManager()
    mem = MemoryManager(storage_dir=".test_swarmchat")
    orch = Orchestrator(mm, mem, ToolManager())
    stub_generation(mm, "Tuning myself. [UPDATE_CONFIG: temperature=very-hot]")

    asyncio.run(orch.step_model_turn("model_architect"))
    system_msgs = [m for m in orch.chat_history if "DIRECTIVE IGNORED" in m["content"]]
    assert len(system_msgs) == 1
    assert "temperature=very-hot" in system_msgs[0]["content"]
    assert orch.models["model_architect"]["temperature"] == 0.7

def test_valid_config_directive_is_applied():
    mm = ModelManager()
    mem = MemoryManager(storage_dir=".test_swarmchat")
    orch = Orchestrator(mm, mem, ToolManager())
    stub_generation(mm, "Adjusting. [UPDATE_CONFIG: temperature=0.35, top_k=20]")

    asyncio.run(orch.step_model_turn("model_architect"))
    assert orch.models["model_architect"]["temperature"] == 0.35
    assert orch.models["model_architect"]["top_k"] == 20
    assert not [m for m in orch.chat_history if "DIRECTIVE IGNORED" in m["content"]]

def test_autonomous_loop_reports_turn_failure():
    mm = ModelManager()
    mem = MemoryManager(storage_dir=".test_swarmchat")
    orch = Orchestrator(mm, mem, ToolManager())

    async def _boom(model_config, system_prompt, messages):
        raise RuntimeError("unexpected orchestration bug")
    mm.generate_response = _boom

    asyncio.run(orch.run_autonomous_loop(max_turns=2))
    assert any("LOOP HALTED" in m["content"] for m in orch.chat_history)
    assert orch.loop_active is False

def test_invalid_phase_and_turn_mode_are_rejected():
    mem = MemoryManager(storage_dir=".test_swarmchat")
    mem.set_phase("discussion")
    with pytest.raises(ValueError):
        mem.set_phase("nap-time")
    assert mem.get_phase() == "discussion"

    orch = Orchestrator(ModelManager(), mem, ToolManager())
    with pytest.raises(ValueError):
        orch.set_turn_mode("whatever_mode")
    assert orch.turn_mode == "round_robin"

def test_corrupt_memory_file_is_quarantined_and_reported(tmp_path):
    project_dir = tmp_path / "mem" / "projects" / "default_project"
    project_dir.mkdir(parents=True)
    (project_dir / "shared_memory.json").write_text("{not valid json")

    mem = MemoryManager(storage_dir=str(tmp_path / "mem"))
    assert mem.last_load_error is not None
    quarantined = list(project_dir.glob("shared_memory.json.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text() == "{not valid json"

def test_memory_save_failure_propagates(tmp_path):
    storage = tmp_path / "mem2"
    storage.mkdir()
    mem = MemoryManager(storage_dir=str(storage))
    mem.json_path = str(storage / "missing-dir" / "shared_memory.json")
    with pytest.raises(MemoryPersistenceError):
        mem.save_memory()

def test_prompt_template_save_failure_propagates(tmp_path):
    mgr = PromptTemplateManager(storage_path=str(tmp_path / "cfg" / "prompt_templates.json"))
    mgr.storage_path = str(tmp_path / "cfg" / "a-file" / "prompt_templates.json")
    (tmp_path / "cfg").mkdir(exist_ok=True)
    (tmp_path / "cfg" / "a-file").write_text("i am a file, not a directory")
    with pytest.raises(MemoryPersistenceError):
        mgr.save_templates()

def test_corrupt_prompt_templates_keep_defaults(tmp_path):
    path = tmp_path / "prompt_templates.json"
    path.write_text("[1, 2, 3]")
    mgr = PromptTemplateManager(storage_path=str(path))
    assert mgr.last_load_error is not None
    assert "Phase: Discussion" in mgr.templates["start_prompt"]

def test_tool_voting_and_admin_override():
    mm = ModelManager()
    mem = MemoryManager(storage_dir=".test_swarmchat")
    tm = ToolManager()
    orch = Orchestrator(mm, mem, tm)

    vote_req = orch.propose_tool_call("model_coder", "write_file", {"filepath": "test.txt", "content": "hello world"})
    assert vote_req["risk_level"] == "consequential"
    assert len(orch.pending_tool_votes) == 1

    vote_id = vote_req["id"]

    # Write tools are locked in discussion phase: the override must report that, not claim success.
    locked_res = orch.admin_override_vote(vote_id, "approve")
    assert locked_res["success"] is False
    assert "locked" in locked_res["error"].lower()

    mem.set_phase("execution")
    override_res = orch.admin_override_vote(vote_id, "approve")
    assert override_res["success"] is True
    assert override_res["executed"] is True

def test_evaluate_engine():
    mm = ModelManager()
    stub_generation(mm, "Candidate evaluation answer.")
    ee = EvaluateEngine(mm)
    candidates = [
        {"id": "cand_1", "name": "Llama 1B", "role": "Architect", "provider": "ollama"},
        {"id": "cand_2", "name": "Bonsai 1.7B", "role": "Coder", "provider": "gguf_local"}
    ]
    res = asyncio.run(ee.run_candidate_evaluation(candidates, "Build a file parser"))
    assert res["success"] is True
    assert "rankings" in res
    assert len(res["rankings"]) == 2

def test_evaluate_engine_reports_candidate_failures():
    """An unusable candidate is ranked last with its error, not scored as if it answered."""
    mm = ModelManager()
    fail_generation(mm, "llama-cpp-python engine not installed")
    ee = EvaluateEngine(mm)
    res = asyncio.run(ee.run_candidate_evaluation(
        [{"id": "cand_1", "name": "Broken", "role": "Coder", "provider": "gguf_local"}],
        "Build a file parser"
    ))
    assert res["success"] is False
    assert res["top_recommendation"] == "None"
    assert res["rankings"][0]["overall_score"] == 0.0
    assert "llama-cpp-python" in res["rankings"][0]["error"]

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

    stub_generation(mm)

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
    stub_generation(mm)

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

class _FailingClient:
    """Minimal httpx.AsyncClient stand-in whose requests always fail."""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, *args, **kwargs):
        raise httpx.ConnectError("network unreachable")

def test_huggingface_search_failure_is_not_fabricated(monkeypatch):
    """A failed HF lookup must not be reported as a successful hit on an invented model."""
    monkeypatch.setattr(httpx, "AsyncClient", _FailingClient)
    res = asyncio.run(ToolManager().search_huggingface("Llama-3-GGUF"))
    assert res["success"] is False
    assert res["models"] == []
    assert "network unreachable" in res["error"]

def test_internet_search_failure_is_not_fabricated(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _FailingClient)
    res = asyncio.run(ToolManager().internet_search("latest python release"))
    assert res["success"] is False
    assert "network unreachable" in res["error"]
    assert "results" not in res

def test_workspace_read_and_list_report_missing_paths():
    tm = ToolManager()
    read_res = tm.read_file("definitely_missing_file.txt")
    assert read_res["success"] is False
    assert read_res["error"]

    list_res = tm.list_files("no_such_directory_here")
    assert list_res["success"] is False

def test_terminal_command_reports_nonzero_exit_and_timeout():
    tm = ToolManager()
    failed = tm.run_terminal_cmd("python -c 'import sys; sys.exit(3)'")
    assert failed["success"] is False
    assert failed["returncode"] == 3

    timed_out = tm.run_terminal_cmd("python -c 'import time; time.sleep(5)'", timeout=1)
    assert timed_out["success"] is False
    assert timed_out["timed_out"] is True

def test_fs_browser_and_validation():
    from backend.main import browse_filesystem, validate_model_path, ValidatePathReq
    browse_res = browse_filesystem(".")
    assert browse_res["success"] is True
    assert "current_path" in browse_res
    assert "files" in browse_res

    val_res = validate_model_path(ValidatePathReq(path="non_existent_file.gguf"))
    assert val_res["valid"] is False

def test_action_tag_parsing_and_context_reset():
    mm = ModelManager()
    stub_generation(mm, "Stubbed turn after context refresh.")
    mem = MemoryManager(storage_dir=".test_swarmchat_tags")
    tm = ToolManager()
    orch = Orchestrator(mm, mem, tm)

    # Test UPDATE_TASK tag parsing
    mem.add_itinerary_task("Initial Task", "Task description")

    # Exceed model token limit to trigger context refresh
    mem.state.setdefault("tokens_used", {})["model_architect"] = 4000
    asyncio.run(orch.step_model_turn("model_architect"))

    # Token counter should reset to 0 after turn
    assert mem.state["tokens_used"]["model_architect"] < 4000
    assert len(mem.state.get("model_journals", {}).get("model_architect", [])) > 0

def test_note_chunking_and_workspace_auto_save(tmp_path):
    mm = ModelManager()
    mem = MemoryManager(storage_dir=str(tmp_path / "mem_notes"))
    tm = ToolManager()
    orch = Orchestrator(mm, mem, tm)

    # Test SAVE_NOTE directive
    res_notes = mem.add_note_chunk("model_coder", "This is an indexed note chunk detailing architecture specs and design decisions.", title="Spec Note")
    assert res_notes["added_count"] == 1
    searched = mem.search_note_chunks("model_coder", "architecture")
    assert len(searched) == 1
    assert "architecture specs" in searched[0]["content"]

    # Test auto-save of generated markdown code block
    stub_generation(mm, "Here is the code implementation:\n```python\n# filename: calculate.py\ndef add(a, b):\n    return a + b\n```")
    asyncio.run(orch.step_model_turn("model_coder"))

    # Check that file was auto-saved in model workspace
    bot_dir = tm.get_bot_workspace_dir("model_coder")
    code_path = os.path.join(bot_dir, "calculate.py")
    assert os.path.exists(code_path)
    with open(code_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "def add(a, b):" in content

def test_api_returns_error_status_codes():
    """Endpoints must answer with real HTTP errors instead of 200s carrying success: false."""
    from backend import main

    with pytest.raises(HTTPException) as invalid_phase:
        main.set_phase(main.PhaseSwitchReq(phase="siesta"))
    assert invalid_phase.value.status_code == 400

    with pytest.raises(HTTPException) as unknown_task:
        main.update_itinerary_task(main.ItineraryTaskUpdateReq(task_id="task_missing", status="completed"))
    assert unknown_task.value.status_code == 404

    # set_moderator is retired: Architect and Moderator are one seat, named by `role`.
    # The endpoint stays as an explicit 410 so a stale UI bundle gets a readable reason.
    with pytest.raises(HTTPException) as retired_moderator:
        main.set_moderator("model_does_not_exist")
    assert retired_moderator.value.status_code == 410

    with pytest.raises(HTTPException) as missing_file:
        main.get_workspace_file_content("definitely_missing_file.txt")
    assert missing_file.value.status_code == 404

    with pytest.raises(HTTPException) as unknown_vote:
        main.override_vote(main.VoteOverrideReq(vote_id="vote_missing", action="approve"))
    assert unknown_vote.value.status_code == 404

def test_search_endpoint_reports_upstream_failure(monkeypatch):
    from backend import main

    monkeypatch.setattr(httpx, "AsyncClient", _FailingClient)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.search_huggingface("Llama-3-GGUF"))
    assert exc.value.status_code == 502
