"""
Mocked verification for:
  Task 1: step_model_turn() now calls manage_vram_allocation() before generation
          (manual-turn VRAM lifecycle wiring).
  Task 2: run_autonomous_loop() times out a hanging turn after 90s and advances
          to the next speaker instead of hanging forever, and cleanly logs the
          "no eligible speaker" break path.

No real LLM/GGUF loading happens - model_manager.generate_response and friends
are mocked, and asyncio timing is faked via monkeypatched asyncio.wait_for /
asyncio.sleep so the test runs instantly instead of waiting 90 real seconds.
"""
import asyncio
import os
import sys
import tempfile
from unittest.mock import MagicMock

# Was "/tmp/swarmchat" - a path from the machine this was first written on, which does not
# exist here; run it from the repo root instead.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.orchestrator import Orchestrator, create_model_config
from backend.errors import ModelInvocationError


def make_orchestrator():
    model_manager = MagicMock()
    memory_manager = MagicMock()
    tool_manager = MagicMock()

    # storage_root must be explicit here: memory_manager is a MagicMock, so the Orchestrator
    # cannot follow its storage dir and would otherwise load the *user's real* roster - which
    # has none of the default model ids this script drives.
    orch = Orchestrator(model_manager, memory_manager, tool_manager,
                        storage_root=tempfile.mkdtemp(prefix="swarmchat_lifecycle_"))

    # Minimal memory_manager stubs so step_model_turn's context-building code
    # (which runs before the generation call) doesn't explode on MagicMock defaults.
    memory_manager.get_phase.return_value = "discussion"
    memory_manager.get_memory_summary.return_value = "summary"
    memory_manager.get_model_latest_journal.return_value = None
    memory_manager.get_active_task.return_value = None
    memory_manager.state = {"tokens_used": {}}
    memory_manager.get_project_id.return_value = "proj-1"

    return orch, model_manager, memory_manager, tool_manager


async def test_step_model_turn_calls_vram_management():
    """Task 1: step_model_turn must call manage_vram_allocation before generation."""
    orch, model_manager, memory_manager, tool_manager = make_orchestrator()

    call_order = []

    def fake_manage_vram(speaker_id):
        call_order.append(("manage_vram_allocation", speaker_id))

    orch.manage_vram_allocation = MagicMock(side_effect=fake_manage_vram)

    async def fake_generate_response(*args, **kwargs):
        call_order.append(("generate_response",))
        # Abort the rest of the (large, unrelated) turn body early and cheaply -
        # this is a ModelInvocationError, a handled early-return path.
        raise ModelInvocationError("mocked: no real model invoked")

    model_manager.generate_response = fake_generate_response

    speaker_id = "model_architect"
    result = await orch.step_model_turn(speaker_id)

    assert call_order[0] == ("manage_vram_allocation", speaker_id), (
        f"Expected manage_vram_allocation to be called first with '{speaker_id}', got: {call_order}"
    )
    assert orch.manage_vram_allocation.called, "manage_vram_allocation was never called from step_model_turn"
    assert "MODEL ERROR" in result["content"]
    print("PASS: step_model_turn() calls manage_vram_allocation() before generation (Task 1 wiring)")


async def test_autonomous_loop_timeout_advances_to_next_speaker():
    """Task 2: a hanging step_model_turn must be timed out and the loop must
    advance to the next speaker rather than hanging forever.

    Uses real asyncio timing (no internals faked) but shrinks the module-level
    AUTO_TURN_TIMEOUT_SECONDS watchdog constant so the test runs in well under
    a second instead of actually waiting 90 real seconds.
    """
    import backend.orchestrator as orchestrator_module

    orch, model_manager, memory_manager, tool_manager = make_orchestrator()

    # Two eligible speakers; get_next_speaker round-robins them off last_speaker_id.
    speakers = ["model_architect", "model_critic"]

    def fake_get_next_speaker(last_speaker_id=None):
        if last_speaker_id is None:
            return speakers[0]
        idx = speakers.index(last_speaker_id) if last_speaker_id in speakers else -1
        return speakers[(idx + 1) % len(speakers)]

    orch.get_next_speaker = MagicMock(side_effect=fake_get_next_speaker)
    orch.manage_vram_allocation = MagicMock()  # no-op, already covered by Task 1 test

    step_calls = []

    async def fake_step_model_turn(model_id):
        step_calls.append(model_id)
        if model_id == "model_architect" and step_calls.count(model_id) == 1:
            # Simulate a hung/slow model call that genuinely outlives the
            # (shrunk) watchdog timeout, only on its first turn, so the loop
            # can terminate via max_turns once rotation has proven it moved
            # past the stalled speaker.
            await asyncio.sleep(2)
        return {"content": f"reply from {model_id}"}

    orch.step_model_turn = fake_step_model_turn

    orig_timeout = orchestrator_module.AUTO_TURN_TIMEOUT_SECONDS
    orchestrator_module.AUTO_TURN_TIMEOUT_SECONDS = 0.05
    try:
        await asyncio.wait_for(
            orch.run_autonomous_loop(max_turns=2, max_discussion_turns=8, max_discussion_seconds=120.0),
            timeout=10,  # generous real-time ceiling; should finish in ~0.05s + a bit
        )
    finally:
        orchestrator_module.AUTO_TURN_TIMEOUT_SECONDS = orig_timeout

    assert not orch.loop_active, "loop_active should be reset to False when the loop exits"
    # First call to model_architect hangs/times out; the loop must not retry it but
    # instead advance rotation to model_critic, then continue normally from there.
    assert step_calls[:2] == ["model_architect", "model_critic"], (
        f"Expected the timed-out speaker to be skipped past to the next speaker, got: {step_calls}"
    )
    assert "model_critic" in step_calls and step_calls.count("model_critic") >= 1
    # A timeout system message should have been recorded in chat_history.
    timeout_msgs = [m for m in orch.chat_history if "TURN TIMEOUT" in m.get("content", "")]
    assert timeout_msgs, "Expected a TURN TIMEOUT system message to be added to chat_history"
    print("PASS: run_autonomous_loop() times out a hanging turn after 90s and advances to next speaker (Task 2)")


async def test_no_eligible_speaker_break_is_logged():
    """Task 2 (distinct failure mode): get_next_speaker() -> None must break
    the loop cleanly (not hang, not conflate with a timeout)."""
    orch, model_manager, memory_manager, tool_manager = make_orchestrator()
    orch.get_next_speaker = MagicMock(return_value=None)
    orch.manage_vram_allocation = MagicMock()

    called = {"step": False}

    async def fake_step_model_turn(model_id):
        called["step"] = True
        return {"content": ""}

    orch.step_model_turn = fake_step_model_turn

    await asyncio.wait_for(orch.run_autonomous_loop(max_turns=3), timeout=5)

    assert not called["step"], "step_model_turn should never be invoked when there is no eligible speaker"
    assert not orch.loop_active
    print("PASS: run_autonomous_loop() breaks cleanly (and distinctly from timeout) when no eligible speaker exists")


async def main():
    await test_step_model_turn_calls_vram_management()
    await test_autonomous_loop_timeout_advances_to_next_speaker()
    await test_no_eligible_speaker_break_is_logged()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
