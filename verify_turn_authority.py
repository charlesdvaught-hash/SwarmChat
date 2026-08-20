"""Verifies there is exactly ONE authority on who speaks next.

Before this, two systems answered that question: the residency-major task router
(_select_execution_speaker) drove actual turns, while `turn_schedule` - a round-robin queue
that is never consumed during execution - still drove the VRAM prefetcher and the UI's
"up next" list. @mentions were also unreachable in execution phase.
No LLM is invoked.
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


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail and not ok else ""))
    if not ok:
        failures.append(name)


def build(tmp, phase="execution", resident=()):
    _case[0] += 1
    mm, tm = ModelManager(), ToolManager(workspace_root=tmp)
    mem = MemoryManager(storage_dir=os.path.join(tmp, "mem_case%d" % _case[0]))
    orch = Orchestrator(mm, mem, tm)
    orch.models = {
        "arch": {"id": "arch", "name": "Otis", "role": "Architect", "provider": "gguf_local",
                 "gguf_path": "a.gguf", "enabled": True, "is_moderator": True},
        "code": {"id": "code", "name": "Bill", "role": "Coder", "provider": "gguf_local",
                 "gguf_path": "c.gguf", "enabled": True},
        "crit": {"id": "crit", "name": "Vera", "role": "Critic", "provider": "gguf_local",
                 "gguf_path": "r.gguf", "enabled": True},
        "test": {"id": "test", "name": "Tess", "role": "Tester/Debugger", "provider": "gguf_local",
                 "gguf_path": "t.gguf", "enabled": True},
    }
    orch.known_models = dict(orch.models)
    # No moderator assignment: "arch" holds the Architect role, which IS the supervisor seat.
    # The orchestrator restores - and persists - the project chat log under ./.swarmchat.
    # Start each case from an empty log and never write back, or these synthetic messages
    # land in a real project and the next verifier run reads them as live conversation.
    orch.chat_history = []
    orch.save_chat_history = lambda *a, **k: None
    mem.set_phase(phase)
    for m in resident:
        mm.gguf_instances[m] = object()
    return mem, orch


def add(mem, title, status):
    t = mem.add_itinerary_task(title=title, description="do " + title, priority="high")
    mem.update_itinerary_task(t["id"], {"status": status})
    return t["id"]


def main():
    tmp = tempfile.mkdtemp(prefix="swarmchat_turn_")

    # 1. @mention must pre-empt the task router during execution (it never could before).
    mem, orch = build(tmp, resident=("code",))
    add(mem, "write parser", "in_progress")   # router would pick the Coder
    orch.add_chat_message("Admin", "Admin", "hold on @Vera - look at this first", is_admin=True)
    check("@mention pre-empts the execution router", orch.get_next_speaker() == "crit",
          "got %r" % orch.get_next_speaker())

    # 2. The orchestrator's own system notices contain "@<name>" and must NOT count as
    #    mentions, or a skipped/timed-out model re-selects itself forever.
    mem, orch = build(tmp, resident=("code",))
    add(mem, "write parser", "in_progress")
    orch.add_chat_message("System / Conversation Loop", "System",
                          "[TURN TIMEOUT] @Vera did not respond - forcing rotation.", is_admin=True)
    check("System notices are not treated as @mentions", orch.get_next_speaker() == "code",
          "got %r" % orch.get_next_speaker())

    # 3. "Up next" during execution is task-driven, not the stale round-robin queue.
    mem, orch = build(tmp, resident=("code",))
    add(mem, "review the parser", "needs_review")   # owner: Critic
    add(mem, "write the parser", "in_progress")     # owner: Coder
    orch.turn_schedule = ["arch", "test", "arch"]   # stale legacy queue, never consumed here
    up = orch.upcoming_speakers(3)
    check("Execution 'up next' comes from outstanding work", up[:2] == ["crit", "code"],
          "got %r" % up)
    check("Execution 'up next' ignores the stale roster queue", "test" not in up, "got %r" % up)

    # 4. Outside execution, the planning gate is the authority - the roster queue is only a
    #    fallback. (This check used to assert the raw queue; the gate replaced it. See
    #    verify_plan_gate.py.) The gate order here is Architect -> Critic -> Coder.
    mem, orch = build(tmp, phase="discussion")
    orch.turn_schedule = ["arch", "test", "crit"]
    check("Discussion 'up next' comes from the planning gate",
          orch.upcoming_speakers(3) == ["arch", "crit", "code"],
          "got %r" % orch.upcoming_speakers(3))

    # 5. The VRAM prefetcher asks that same authority instead of reading turn_schedule.
    mem, orch = build(tmp, resident=("code",))
    add(mem, "review the parser", "needs_review")
    orch.turn_schedule = ["test", "test", "test"]
    asked = {}
    orch.upcoming_speakers = lambda limit=3, _a=asked: _a.setdefault("limit", limit) and [] or []
    orch.model_manager.get_hardware_info = lambda: {"gpu_name": "fake", "vram_total_gb": 24.0}
    orch.manage_vram_allocation("code")
    check("manage_vram_allocation routes through upcoming_speakers()", "limit" in asked,
          "prefetcher never called upcoming_speakers")

    # 6. A mention is honoured once, then the router takes back over - otherwise the mention
    #    stays the newest non-system message and re-selects the same model forever.
    mem, orch = build(tmp, resident=("code",))
    add(mem, "write parser", "in_progress")
    orch.add_chat_message("Admin", "Admin", "@Vera take a look", is_admin=True)
    first = orch.get_next_speaker()
    orch.add_chat_message("System / Conversation Loop", "System", "[TURN TIMEOUT] @Vera stalled.",
                          is_admin=True)
    second = orch.get_next_speaker()
    check("A mention is served once, then the router resumes",
          first == "crit" and second == "code", "got %r then %r" % (first, second))

    print()
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("All turn-authority checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
