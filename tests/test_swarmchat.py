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


def capture_generation(model_manager: ModelManager, text: str = "Stubbed model turn."):
    """Like stub_generation, but returns a list that collects each system prompt used.

    A turn can call generate_response more than once (the empty-generation retry), so the
    caller should assert against the whole list or its first entry, never assume one call.
    """
    prompts = []

    async def _generate(model_config, system_prompt, messages):
        prompts.append(system_prompt)
        return text

    model_manager.generate_response = _generate
    return prompts


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
    # Its own storage dir on purpose. Chat history is PERSISTED per storage root and
    # reloaded by Orchestrator.__init__, so a test asserting the ABSENCE of a message
    # inherits every "DIRECTIVE IGNORED" line the shared-root tests wrote before it -
    # which is exactly how this test failed depending on run order, with nothing wrong
    # in the code under test.
    mem = MemoryManager(storage_dir=".test_swarmchat_config_directive")
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

def test_narration_is_admin_only_and_never_reaches_a_model(tmp_path):
    """Chat narration goes in `content`; models read `model_visible`.

    Discussion turns feed chat_history[-3:] back into the next speaker, so anything written
    into `content` is model input by default. The moment presentation text becomes model
    input, models start responding to the narration instead of to the work. This pins the
    one-way boundary.
    """
    mm = ModelManager()
    mem = MemoryManager(storage_dir=str(tmp_path / "mem_narrate"))
    tm = ToolManager()
    orch = Orchestrator(mm, mem, tm)

    msg = orch.add_chat_message(
        sender="Coder",
        role="Coder",
        content="\U0001f527 shipped `word_count.py` (+4/-1).",
        model_visible="Updated `word_count.py` (+4/-1).",
    )
    assert "shipped" in msg["content"]
    assert "shipped" not in msg["model_visible"]

    # Callers that pass no narration must be unaffected - model_visible mirrors content.
    plain = orch.add_chat_message(sender="Admin", role="Admin", content="build a parser", is_admin=True)
    assert plain["model_visible"] == "build a parser"

    # And the discussion-phase context builder must read the un-narrated body.
    prompts = capture_generation(mm, "Acknowledged.")
    asyncio.run(orch.step_model_turn("model_architect"))
    assert prompts, "the turn should have called generate_response"


def test_status_headline_is_built_from_facts_only():
    """Every headline traces to a diff, a verdict or a directive - never to model prose."""
    orch_cls = Orchestrator
    updates = [{"filename": "word_count.py", "added": 4, "removed": 1, "is_new": False}]

    coder = orch_cls._status_headline(orch_cls.SEAT_CODER, updates, "here is the code")
    assert "word_count.py" in coder and "+4/-1" in coder

    created = orch_cls._status_headline(
        orch_cls.SEAT_CODER, [{"filename": "new.py", "added": 12, "removed": 0, "is_new": True}], ""
    )
    assert "+12" in created

    # The Critic's headline follows the verdict, not the prose around it.
    assert "back" in orch_cls._status_headline(orch_cls.SEAT_CRITIC, updates, "Looks fine to me.\nREJECT")
    assert "approved" in orch_cls._status_headline(orch_cls.SEAT_CRITIC, updates, "Some notes.\nAPPROVE")
    # No verdict means no claim about the review either way.
    assert orch_cls._status_headline(orch_cls.SEAT_CRITIC, updates, "I am still reading it.") == ""

    # A turn with no recorded fact gets no headline rather than an invented one.
    assert orch_cls._status_headline(orch_cls.SEAT_CODER, [], "I thought about it a lot.") == ""

    # The Architect never gets one: its deliverable is prose, so a headline would either
    # duplicate the paragraph or restate the directive sentence the body already carries.
    assert orch_cls._status_headline(
        orch_cls.SEAT_ARCHITECT, [], "[UPDATE_TASK: title=Add parser, status=in_progress]"
    ) == ""


def test_narration_does_not_duplicate_the_fallback_summary():
    """A turn with unusable prose must not read its own diff out twice."""
    mm = ModelManager()
    mem = MemoryManager(storage_dir=".test_swarmchat_narrate2")
    tm = ToolManager()
    orch = Orchestrator(mm, mem, tm)

    updates = [{"filename": "word_count.py", "added": 4, "removed": 1, "is_new": False}]
    cfg = {"name": "Coder", "role": "Coder"}
    display = orch._build_chat_display_text("{}", cfg, updates)   # garbage prose -> synthesized summary
    narrated = orch._narrate_turn(Orchestrator.SEAT_CODER, updates, "{}", display)

    assert narrated.count("word_count.py") == 1
    assert "shipped" in narrated


def test_context_trim_measures_the_prompt_not_words_spoken():
    """The context trim must gate on the PROMPT size, not on how much the model has talked.

    It used to compare `tokens_used` - a running total of words the model had SPOKEN - against
    its context WINDOW. Those are unrelated quantities: the prompt is rebuilt from scratch each
    turn and every component of it is capped, so measured prompts stay flat regardless of how
    long the room runs. The trim therefore fired on whichever model had been most productive,
    at a moment with no relationship to how full anything was, and it dropped the task details
    while keeping the project-wide digest - backwards for a small model.
    """
    mem = MemoryManager(storage_dir=".test_swarmchat_tags")
    tm = ToolManager()
    mem.add_itinerary_task("Initial Task", "Task description")

    # --- A big "words spoken" total must NOT trim anything ---
    mm = ModelManager()
    prompts = capture_generation(mm, "Stubbed turn.")
    orch = Orchestrator(mm, mem, tm)
    mem.state.setdefault("tokens_used", {})["model_architect"] = 4000
    mm.last_prompt_tokens["model_architect"] = 120  # the real prompt is small
    asyncio.run(orch.step_model_turn("model_architect"))

    assert prompts, "the turn should have called generate_response"
    assert "CONTEXT TRIMMED" not in prompts[0]
    assert "SHARED MEMORY SUMMARY" in prompts[0]
    # The counter is a usage statistic now, not a gate - it accumulates instead of being zeroed.
    assert mem.state["tokens_used"]["model_architect"] > 4000

    # --- A genuinely large measured prompt DOES trim ---
    mm2 = ModelManager()
    prompts2 = capture_generation(mm2, "Stubbed turn after trim.")
    orch2 = Orchestrator(mm2, mem, tm)
    mm2.last_prompt_tokens["model_architect"] = 99999
    asyncio.run(orch2.step_model_turn("model_architect"))

    assert prompts2, "the trimmed turn should have called generate_response"
    trimmed = prompts2[0]
    assert "CONTEXT TRIMMED" in trimmed
    # Specific context survives, diffuse context does not.
    assert "ACTIVE ITINERARY ITEM" in trimmed
    assert "SHARED MEMORY SUMMARY" not in trimmed
    assert "RECENT EPISODE CHECKPOINTS" not in trimmed
    # A trim is still journalled so the event is traceable.
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


# --- QUESTION-DRIVEN DISCUSSION ---
# Full coverage lives in verify_discussion_phase.py; these are the load-bearing few, so a
# plain `pytest` run catches a regression in the parser and in the gate's new shape.

def _planning_room(tmp_path, same_weights=False):
    """A three-seat room sitting at the start of the planning gate."""
    mm = ModelManager()
    mem = MemoryManager(storage_dir=str(tmp_path / "mem"))
    orch = Orchestrator(mm, mem, ToolManager(workspace_root=str(tmp_path)),
                        storage_root=str(tmp_path / "root"))
    seats = {
        "arch": ("Otis", "Architect"),
        "crit": ("Vera", "Critic"),
        "code": ("Bill", "Coder"),
    }
    orch.models = {
        k: {
            "id": k, "name": n, "role": r, "provider": "gguf_local",
            "gguf_path": ("shared.gguf" if same_weights else k + ".gguf"), "enabled": True,
        }
        for k, (n, r) in seats.items()
    }
    orch.known_models = dict(orch.models)
    orch.chat_history = []
    orch.save_chat_history = lambda *a, **k: None
    mem.set_phase("discussion")
    return mem, orch


def _gate_turn(orch, model_id, text):
    cfg = orch.models[model_id]
    orch.add_chat_message(cfg["name"], "Assistant", text, model_id=model_id)
    orch._advance_plan_gate(model_id, cfg, text)
    return orch.memory_manager.get_plan_stage()


def test_directive_parser_keeps_commas_inside_a_signature():
    """The bug the question board would otherwise have inherited: a naive comma split
    turned `title=Implement char_count(text, strip)` into the task `Implement char_count(text`."""
    from backend.directives import parse_fields

    fields = parse_fields(
        "title=Implement char_count(text, strip), description=Add to counter.py, handling empties",
        ("id", "title", "description", "priority", "status", "assigned_model"),
    )
    assert fields["title"] == "Implement char_count(text, strip)"
    assert fields["description"] == "Add to counter.py, handling empties"


def test_question_cap_is_enforced_in_code(tmp_path):
    mem, orch = _planning_room(tmp_path)
    assert mem.get_plan_stage() == "awaiting_questions"
    stage = _gate_turn(orch, "arch", "\n".join(
        "[QUESTION: ask=Question %d, what behaviour do we want?, for=team]" % i for i in range(9)
    ))
    assert len(mem.get_plan_questions()) == Orchestrator.MAX_PLAN_QUESTIONS
    assert stage == "resolving_questions"


def test_one_question_is_settled_per_turn(tmp_path):
    mem, orch = _planning_room(tmp_path)
    _gate_turn(orch, "arch", "[QUESTION: ask=Should punctuation count as part of a word?, for=team]\n"
                             "[QUESTION: ask=Should numbers count as words?, for=team]")
    _gate_turn(orch, "arch", "[ANSWER: Punctuation is stripped before counting.]")
    assert len(mem.questions_by_status("resolved")) == 1
    assert mem.get_plan_stage() == "resolving_questions"


def test_reject_becomes_a_question_not_a_rewrite(tmp_path):
    mem, orch = _planning_room(tmp_path)
    mem.set_plan_stage("awaiting_plan")
    _gate_turn(orch, "arch", "Goal: a word counter. Create counter.py with count_words(text) "
                             "returning an int, and test_counter.py covering the empty string.")
    stage = _gate_turn(orch, "crit", "count_words says nothing about hyphenated words.\nREJECT")
    assert stage == "resolving_questions"
    assert "hyphenated" in mem.get_plan_questions()[0]["text_internal"]
    _gate_turn(orch, "arch", "[ANSWER: A hyphenated word counts as one word.]")
    assert mem.get_plan_stage() == "awaiting_plan"
    # And the decision reaches the next plan prompt, which is the whole point.
    assert "counts as one word" in orch._build_questions_context()


def test_admin_question_parks_without_blocking_the_room(tmp_path):
    mem, orch = _planning_room(tmp_path)
    _gate_turn(orch, "arch",
               "[QUESTION: ask=How do you want to see the answer?, "
               "options=On the screen: it prints when you run it | "
               "In a file: it writes it out for you to open, "
               "recommended=On the screen, why=You can try it immediately, for=admin]\n"
               "[QUESTION: ask=Should punctuation count as part of a word?, for=team]")
    admin_q = next(q for q in mem.get_plan_questions() if q["resolvable_by"] == "admin")
    assert admin_q["status"] == "parked"
    # The room moved on to the team question rather than waiting.
    assert mem.next_open_question()["resolvable_by"] == "model"
    assert orch.answer_plan_question(admin_q["id"], "On the screen")["success"] is True
    assert mem.get_plan_question(admin_q["id"])["decided_by"] == "admin"


def test_admin_question_written_in_jargon_is_never_asked(tmp_path):
    mem, orch = _planning_room(tmp_path)
    _gate_turn(orch, "arch", "[QUESTION: ask=Do you want a CLI or a library?, options=CLI | Library, "
                             "recommended=CLI, why=Faster to try, for=admin]")
    # Admin is not a coder. A question whose options are implementation nouns is a defect in
    # the question, so the room settles it instead of stalling on an unanswerable prompt.
    assert mem.get_plan_questions()[0]["resolvable_by"] == "model"


def test_a_vote_needs_two_distinct_models_not_two_seats(tmp_path):
    mem, orch = _planning_room(tmp_path, same_weights=True)
    assert len(orch._distinct_model_paths()) == 1
    orch.loop_active = True  # Auto Mode: nobody is watching for the question to appear.
    _gate_turn(orch, "arch",
               "[QUESTION: ask=How do you want to see the answer?, "
               "options=On the screen: it prints when you run it | "
               "In a file: it writes it out for you to open, "
               "recommended=On the screen, why=You can try it immediately, for=admin]")
    q = mem.get_plan_questions()[0]
    assert q["decided_by"] == "default", "one set of weights cannot outvote itself"
    assert q["answer"] == "On the screen"


def test_switching_project_clears_the_admin_facing_room(tmp_path):
    """A project owns its chat, its question board and its pending votes.

    The admin-facing chat is where parked "[YOUR CALL]" question cards live, and those are
    clickable - a question belonging to a project you just left must not still be sitting in
    the room, or answering it would record a decision against the wrong project."""
    mm = ModelManager()
    mem = MemoryManager(storage_dir=str(tmp_path / "mem"))
    orch = Orchestrator(mm, mem, ToolManager(workspace_root=str(tmp_path)),
                        storage_root=str(tmp_path / "root"))
    orch.models = {"arch": {"id": "arch", "name": "Otis", "role": "Architect",
                            "provider": "gguf_local", "gguf_path": "a.gguf", "enabled": True}}
    orch.known_models = dict(orch.models)

    first = mem.get_project_id()
    orch.add_chat_message("Otis", "Assistant", "Plan for the first project", model_id="arch")
    orch.add_chat_message("System / Plan Questions", "System",
                          "[YOUR CALL] How do you want to see the answer?", is_admin=True)
    mem.add_plan_question("How should output appear?", question_admin="How do you want to see the answer?",
                          options=[{"label": "On screen", "means": "it prints when you run it"}],
                          recommended="On screen", resolvable_by="admin")
    orch.pending_tool_votes.append({"id": "vote_1", "status": "pending", "tool_name": "x",
                                    "args": {}, "model_id": "arch", "model_name": "Otis",
                                    "risk_level": "high", "votes": {}, "created_at": 0})

    orch.set_project("second_project")
    assert orch.chat_history == [], "the previous project's admin chat followed the switch"
    assert mem.get_plan_questions() == [], "a parked question followed the switch"
    assert orch.pending_tool_votes == []

    # And switching back restores that project's own room rather than losing it.
    orch.set_project(first)
    assert [m["content"] for m in orch.chat_history] == [
        "Plan for the first project",
        "[YOUR CALL] How do you want to see the answer?",
    ]
    assert len(mem.get_plan_questions()) == 1
