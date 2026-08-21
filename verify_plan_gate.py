"""Verifies the pre-execution planning gate.

Discussion used to be shapeless: the round-robin roster picked whoever was next, any model
could emit [READY_FOR_EXECUTION] and flip the room, and the autonomous loop force-flipped to
execution after a turn/time cap regardless. A plan could therefore reach the Coder with
nobody having read it.

The gate now runs questions -> resolution -> Architect -> Critic -> Programmer ->
Architect-opens-execution, skipping review seats the roster does not contain. No LLM is
invoked.

This file covers the PLAN half of the gate, so `build()` starts each case at awaiting_plan
with the question board already clear - that is a legitimate real state (the Architect had
nothing undecided to raise). The question half has its own file, verify_discussion_phase.py.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath("."))

from backend.models import ModelManager
from backend.memory import MemoryManager
from backend.tools import ToolManager
from backend.orchestrator import Orchestrator

failures = []
_case = [0]

PLAN = (
    "Goal: ship a CSV exporter. Create exporter.py with write_rows(path, rows) writing a "
    "UTF-8 file with a header row, then test_exporter.py covering the empty-rows case. "
    "Coder writes both; Tester runs them."
)


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail and not ok else ""))
    if not ok:
        failures.append(name)


def build(tmp, seats=("arch", "crit", "code"), phase="discussion"):
    _case[0] += 1
    mm, tm = ModelManager(), ToolManager(workspace_root=tmp)
    mem = MemoryManager(storage_dir=os.path.join(tmp, "mem_case%d" % _case[0]))
    orch = Orchestrator(mm, mem, tm, storage_root=os.path.join(tmp, "root_case%d" % _case[0]))
    catalog = {
        "arch": {"id": "arch", "name": "Otis", "role": "Architect"},
        "crit": {"id": "crit", "name": "Vera", "role": "Critic"},
        "code": {"id": "code", "name": "Bill", "role": "Coder"},
        "test": {"id": "test", "name": "Tess", "role": "Tester/Debugger"},
    }
    orch.models = {
        k: dict(v, provider="gguf_local", gguf_path=k + ".gguf", enabled=True)
        for k, v in catalog.items() if k in seats
    }
    orch.known_models = dict(orch.models)
    # These synthetic messages must never land in a real project log.
    orch.chat_history = []
    orch.save_chat_history = lambda *a, **k: None
    mem.set_phase(phase)
    # Start past the question stages: this file is about the plan/review half of the gate.
    # A room whose Architect raised no open questions legitimately lands here on turn one.
    mem.set_plan_stage("awaiting_plan")
    mem.state["plan_revision"] = 0
    return mem, orch


def turn(orch, model_id, text):
    """The gate half of step_model_turn, without invoking a model."""
    cfg = orch.models[model_id]
    orch.add_chat_message(cfg["name"], "Assistant", text, model_id=model_id)
    if "[READY_FOR_EXECUTION]" in text:
        orch._request_execution_phase(model_id, cfg)
    if orch.memory_manager.get_phase() == "discussion":
        orch._advance_plan_gate(model_id, cfg, text)
    return orch.memory_manager.get_plan_stage()


def main():
    tmp = tempfile.mkdtemp(prefix="swarmchat_gate_")

    # 1. A fresh room starts at the Architect, not at whoever the roster happens to list.
    mem, orch = build(tmp)
    orch.turn_schedule = ["code", "crit", "arch"]
    check("Planning opens on the Architect", orch.get_next_speaker() == "arch",
          "got %r" % orch.get_next_speaker())
    check("Gate starts at awaiting_plan once questions are clear",
          mem.get_plan_stage() == "awaiting_plan", "got %r" % mem.get_plan_stage())

    # 2. The full happy path: plan -> critic -> programmer -> approved -> execution.
    mem, orch = build(tmp)
    check("Architect's plan hands off to the Critic", turn(orch, "arch", PLAN) == "critic_review")
    check("Critic is next to speak", orch.get_next_speaker() == "crit",
          "got %r" % orch.get_next_speaker())
    check("Critic APPROVE hands off to the Programmer",
          turn(orch, "crit", "Steps are consistent and the files are named.\nAPPROVE") == "programmer_review")
    check("Programmer is next to speak", orch.get_next_speaker() == "code",
          "got %r" % orch.get_next_speaker())
    check("Programmer APPROVE closes the gate",
          turn(orch, "code", "I can build this as written.\nAPPROVE") == "approved")
    check("Architect speaks again once approved", orch.get_next_speaker() == "arch",
          "got %r" % orch.get_next_speaker())
    turn(orch, "arch", "Bill takes exporter.py first.\n[READY_FOR_EXECUTION]")
    check("Architect opens Execution once approved", mem.get_phase() == "execution",
          "phase is %r" % mem.get_phase())

    # 3. A REJECT at either review sends the plan back for a rehash - it does not stall,
    #    and it does not slide forward.
    mem, orch = build(tmp)
    turn(orch, "arch", PLAN)
    # A REJECT no longer triggers a blind rewrite: the objection is recorded as a question,
    # settled, and the room lands back on awaiting_plan carrying that decision.
    check("Critic REJECT returns to the Architect via a recorded objection",
          turn(orch, "crit", "Step 2 contradicts step 1 on the header row.\nREJECT") == "resolving_questions")
    check("The Critic's objection became a question",
          len(mem.get_plan_questions()) == 1 and mem.get_plan_questions()[0]["status"] == "open",
          "got %r" % mem.get_plan_questions())
    turn(orch, "arch", "[ANSWER: The header row is written once, inside write_rows.]")
    check("Settling the objection returns the room to the plan",
          mem.get_plan_stage() == "awaiting_plan", "got %r" % mem.get_plan_stage())
    check("Architect owns the rehash turn", orch.get_next_speaker() == "arch",
          "got %r" % orch.get_next_speaker())
    turn(orch, "arch", PLAN + " Revised: the header row is written once, in write_rows.")
    check("Rehashed plan goes back to the Critic", mem.get_plan_stage() == "critic_review",
          "got %r" % mem.get_plan_stage())
    turn(orch, "crit", "Fixed.\nAPPROVE")
    check("Programmer REJECT also becomes a question",
          turn(orch, "code", "rows has no defined type, I cannot build this.\nREJECT") == "resolving_questions")
    check("Plan revision counter tracks the rehashes", mem.get_plan_revision() == 2,
          "got %r" % mem.get_plan_revision())

    # 4. The two ways execution used to open by accident.
    mem, orch = build(tmp)
    turn(orch, "arch", PLAN + "\n[READY_FOR_EXECUTION]")
    check("Architect cannot open Execution before review", mem.get_phase() == "discussion",
          "phase is %r" % mem.get_phase())

    mem, orch = build(tmp)
    turn(orch, "arch", PLAN)
    turn(orch, "crit", "APPROVE")
    turn(orch, "code", "APPROVE")
    check("Gate reached approved", mem.get_plan_stage() == "approved")
    turn(orch, "code", "Starting now.\n[READY_FOR_EXECUTION]")
    check("A non-Architect cannot open Execution even when approved",
          mem.get_phase() == "discussion", "phase is %r" % mem.get_phase())

    # 5. A review turn with no verdict holds the stage. Nothing downstream would catch a bad
    #    plan, so silence must not read as consent (unlike the execution-phase Critic, where
    #    an actual test run follows).
    mem, orch = build(tmp)
    turn(orch, "arch", PLAN)
    check("No verdict holds the plan in critic review",
          turn(orch, "crit", "Here are some thoughts on the general shape of it.") == "critic_review")

    # 6. A too-thin Architect turn is not a plan.
    mem, orch = build(tmp)
    check("A one-word Architect turn does not count as a plan",
          turn(orch, "arch", "Okay.") == "awaiting_plan")

    # 7. Missing seats are skipped, not blocked on - Admin's rule.
    mem, orch = build(tmp, seats=("arch", "code"))
    check("No Critic in the room skips straight to programmer review",
          turn(orch, "arch", PLAN) == "programmer_review")
    mem, orch = build(tmp, seats=("arch",))
    check("An Architect-only room reaches approved directly",
          turn(orch, "arch", PLAN) == "approved")
    turn(orch, "arch", "Starting.\n[READY_FOR_EXECUTION]")
    check("Architect-only room can then open Execution", mem.get_phase() == "execution",
          "phase is %r" % mem.get_phase())

    # 8. The VRAM prefetcher must load the gate's running order, not the round-robin queue -
    #    the same mismatch that was fixed for the execution phase.
    mem, orch = build(tmp)
    orch.turn_schedule = ["test", "test", "test"]
    check("Planning 'up next' is the gate order",
          orch.upcoming_speakers(3) == ["arch", "crit", "code"],
          "got %r" % orch.upcoming_speakers(3))

    # 9. An @mention still pre-empts the gate, but speaking out of turn must not move it.
    mem, orch = build(tmp)
    turn(orch, "arch", PLAN)
    orch.add_chat_message("Admin", "Admin", "@Bill what do you think?", is_admin=True)
    check("@mention pre-empts the gate", orch.get_next_speaker() == "code",
          "got %r" % orch.get_next_speaker())
    turn(orch, "code", "Looks buildable to me.\nAPPROVE")
    check("Speaking out of turn does not advance the gate",
          mem.get_plan_stage() == "critic_review", "got %r" % mem.get_plan_stage())

    # 10. Returning to discussion restarts planning, so the Architect cannot immediately
    #     re-open execution on a plan that was just sent back.
    mem, orch = build(tmp)
    turn(orch, "arch", PLAN)
    turn(orch, "crit", "APPROVE")
    turn(orch, "code", "APPROVE")
    mem.set_phase("discussion")
    mem.reset_plan_gate()
    check("Back to discussion restarts the gate at the question stage",
          mem.get_plan_stage() == "awaiting_questions", "got %r" % mem.get_plan_stage())
    check("Restarting the gate clears the question board",
          mem.get_plan_questions() == [], "got %r" % mem.get_plan_questions())

    print()
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("All plan-gate checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
