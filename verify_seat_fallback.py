"""Verifies the OTHER half of the vacant-seat fallback: the contract, not just the routing.

The hackrawler run (2026-08-21, roster = qwen3-4b-thinking as Architect + Qwen3.5-4B as
Critic, no Coder at all) produced five consecutive Architect turns between 00:01:27 and
00:03:47 that each re-created the same "define core game loop" task under a slightly
different filename, and not one line of code. Two halves of one fallback had drifted apart:

  * `_outstanding_work` DID fall back - an in_progress task with no Coder-role model in the
    room was handed to the Architect, exactly as intended.
  * `_seat_for_turn` did NOT. "Architect" does not claim SEAT_CODER, so it fell through to
    `_default_seat_from_role`, which handed that same turn the *Architect* contract. The
    Architect was asked to write code and told to assign work, so it assigned work. Forever.

The seat-keyed supervisor write guard then discarded any code it did emit anyway, so the
room looked busy and produced nothing - the exact signature the collapsed-roster work was
supposed to have killed off, reappearing on a roster that is split rather than collapsed.

These checks are all mocked - no LLM is invoked.
"""
import asyncio
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath("."))

from backend.models import ModelManager
from backend.memory import MemoryManager
from backend.tools import ToolManager
from backend.orchestrator import Orchestrator

CODE_TURN = (
    "Implementing it now:\n"
    "```python\n"
    "# filename: game_loop.py\n"
    "def run():\n"
    "    return 'tick'\n"
    "```\n"
)

failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail and not ok else ""))
    if not ok:
        failures.append(name)


def stub(mm, text):
    async def _generate(model_config=None, system_prompt="", messages=None, **kw):
        _generate.prompts.append(system_prompt)
        return text
    _generate.prompts = []
    mm.generate_response = _generate
    return _generate


_seq = [0]


def build(tmp, roles):
    """roles: {model_id: role_string}. Each case gets its own storage dir so one case's
    itinerary never leaks into the next."""
    _seq[0] += 1
    tmp = os.path.join(tmp, f"case{_seq[0]}")
    os.makedirs(tmp, exist_ok=True)
    mm, tm = ModelManager(), ToolManager(workspace_root=tmp)
    mem = MemoryManager(storage_dir=os.path.join(tmp, "mem"))
    orch = Orchestrator(mm, mem, tm)
    orch.chat_history = []
    orch.save_chat_history = lambda *a, **k: None
    orch.models = {
        mid: {"id": mid, "name": mid.title(), "role": role, "provider": "ollama",
              "model_name": "stub", "enabled": True}
        for mid, role in roles.items()
    }
    orch.known_models = dict(orch.models)
    return mm, mem, tm, orch


# The exact hackrawler roster: an Architect and a Critic, and nobody else.
HACKRAWLER = {"arch": "Architect", "crit": "Critic"}


def files_in(tm, bot_id):
    d = tm.get_bot_workspace_dir(bot_id)
    return [f for f in os.listdir(d)] if os.path.isdir(d) else []


def main():
    tmp = tempfile.mkdtemp(prefix="swarmchat_seatfallback_")
    try:
        # --- 1. The routing half (already worked) --------------------------------------
        _, mem, _, orch = build(tmp, HACKRAWLER)
        mem.set_phase("execution")
        mem.add_itinerary_task(title="Define core game loop",
                               description="Create game_loop.py", priority="high")
        task = mem.get_task_itinerary()[0]
        mem.update_itinerary_task(task["id"], {"status": "in_progress"})
        work = orch._outstanding_work(["arch", "crit"])
        check("in_progress task with no Coder in the room is routed to the Architect",
              any(w[0]["id"] == task["id"] and w[1] == "arch" for w in work), str(work))

        # --- 2. The contract half (the bug) --------------------------------------------
        arch_cfg = orch.models["arch"]
        seat = orch._seat_for_turn(arch_cfg, "execution", {"status": "in_progress"})
        check("Architect covering a vacant Coder seat GETS the Coder seat",
              seat == orch.SEAT_CODER, str(seat))
        check("...and therefore the Coder contract, not the Architect's",
              orch._role_output_contract(arch_cfg, "execution", seat=seat)
              == orch.ROLE_OUTPUT_CONTRACTS["coder"])
        check("A failed task routes the same way",
              orch._seat_for_turn(arch_cfg, "execution", {"status": "failed"}) == orch.SEAT_CODER)

        # No Tester either: needs_test falls to the Critic, who must get the Tester seat.
        crit_cfg = orch.models["crit"]
        check("Critic covering a vacant Tester seat gets the Tester seat",
              orch._seat_for_turn(crit_cfg, "execution", {"status": "needs_test"})
              == orch.SEAT_TESTER)

        # --- 3. A filled seat is NEVER reassigned --------------------------------------
        _, mem, _, orch = build(tmp, {"arch": "Architect", "coder": "Coder", "crit": "Critic"})
        arch_cfg = orch.models["arch"]
        got = orch._seat_for_turn(arch_cfg, "execution", {"status": "in_progress"})
        check("With a real Coder present the Architect does NOT take the Coder seat",
              got == orch.SEAT_ARCHITECT, str(got))
        check("The real Coder still takes it",
              orch._seat_for_turn(orch.models["coder"], "execution", {"status": "in_progress"})
              == orch.SEAT_CODER)
        check("No pinned task still means the Architect seat",
              orch._seat_for_turn(arch_cfg, "execution", None) == orch.SEAT_ARCHITECT)

        # --- 4. End to end: the room actually produces a file now -----------------------
        mm, mem, tm, orch = build(tmp, HACKRAWLER)
        gen = stub(mm, CODE_TURN)
        mem.set_phase("execution")
        mem.add_itinerary_task(title="Define core game loop",
                               description="Create game_loop.py", priority="high")
        task = mem.get_task_itinerary()[0]
        mem.update_itinerary_task(task["id"], {"status": "in_progress"})

        asyncio.run(orch.step_model_turn("arch"))

        wrote = [f for f in files_in(tm, "arch") if f.endswith(".py")]
        check("Architect-as-Coder turn WRITES a file (hackrawler regression)",
              bool(wrote), str(wrote))
        check("...and its prompt names the Coder seat",
              any("SPEAKING AS: CODER" in p for p in gen.prompts),
              str([p[:60] for p in gen.prompts]))
        guard = [m for m in orch.chat_history if "SUPERVISOR WROTE CODE" in m.get("content", "")]
        check("...and the supervisor write guard does NOT fire on that turn", not guard)

        fresh = next(t for t in mem.get_task_itinerary() if t["id"] == task["id"])
        check("...and the task advances to needs_review instead of churning",
              fresh.get("status") == "needs_review", str(fresh.get("status")))

        # --- 5. The Architect seat is still write-locked when it IS the Architect -------
        mm, mem, tm, orch = build(tmp, HACKRAWLER)
        stub(mm, CODE_TURN)
        mem.set_phase("execution")  # no task at all -> genuine Architect seat
        asyncio.run(orch.step_model_turn("arch"))
        wrote = [f for f in files_in(tm, "arch") if f.endswith(".py")]
        check("Genuine Architect-seat turn still writes nothing", not wrote, str(wrote))

        print()
        if failures:
            print(f"{len(failures)} CHECK(S) FAILED: " + ", ".join(failures))
            return 1
        print("All seat-fallback checks passed.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
