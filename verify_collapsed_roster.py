"""Verifies the seat model: one model holding every seat still runs the pipeline.

A 12 GB card cannot hold four 3-4B models plus their KV caches, so the roster collapses to
one resident "brain" whose role is "Architect/Coder/Critic/Tester". Everything in the
orchestrator used to branch on that role *string*, which broke in ways that all look like
"the pipeline is busy but never produces a file":

  1. The supervisor file-write guard keyed off "architect" in role, so it discarded EVERY
     code block the single model ever emitted.
  2. _role_output_contract picked by a fixed if/elif precedence, so a Critic turn on a
     needs_review task was handed the Coder contract.
  3. The two-call/grammar split was disabled for anything Architect-ish, so the Coder and
     Tester seats silently lost their action schema.
  4. needs_review excluded the task's author from reviewing it. On a collapsed roster the
     author is the only model, so the task stalled on needs_review forever.

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
    "# filename: exporter.py\n"
    "def write_rows(path, rows):\n"
    "    return len(rows)\n"
    "```\n"
)

failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail and not ok else ""))
    if not ok:
        failures.append(name)


def stub(mm, text):
    # Collect EVERY system prompt, not just the last one. An execution turn makes two calls
    # (prose, then the grammar-constrained action emission), so keeping only the last one
    # captures the action schema and never the role prompt we want to assert on.
    async def _generate(model_config=None, system_prompt="", messages=None, **kw):
        _generate.prompts.append(system_prompt)
        return text
    _generate.prompts = []
    mm.generate_response = _generate
    return _generate


_build_seq = [0]


def build(tmp, roles):
    """roles: {model_id: role_string}"""
    # Each section gets its own storage dir. Sharing one leaked the previous section's task
    # itinerary into the next, so the "no active task -> Architect seat" case silently ran
    # with a leftover needs_review task and resolved to the Critic seat instead.
    _build_seq[0] += 1
    tmp = os.path.join(tmp, f"case{_build_seq[0]}")
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


BRAIN = "Architect/Coder/Critic/Tester"


def files_in(tm, bot_id):
    d = tm.get_bot_workspace_dir(bot_id)
    return [f for f in os.listdir(d)] if os.path.isdir(d) else []


def main():
    tmp = tempfile.mkdtemp(prefix="swarmchat_collapsed_")
    try:
        # --- 1. Seat derivation --------------------------------------------------------
        _, mem, _, orch = build(tmp, {"brain": BRAIN})
        cfg = orch.models["brain"]

        check("Multi-seat role claims all four seats",
              set(orch._claimed_seats(cfg)) == {"architect", "coder", "critic", "tester"},
              str(orch._claimed_seats(cfg)))

        seats = {
            "pending": orch.SEAT_ARCHITECT,
            "in_progress": orch.SEAT_CODER,
            "needs_review": orch.SEAT_CRITIC,
            "needs_test": orch.SEAT_TESTER,
            "failed": orch.SEAT_CODER,
        }
        got = {
            st: orch._seat_for_turn(cfg, "execution", {"status": st})
            for st in seats
        }
        check("Task status picks the seat, not the role string", got == seats, str(got))

        # A single-seat model must resolve exactly as it did before the change.
        coder_only = {"id": "c", "name": "C", "role": "Coder"}
        check("Single-seat Coder still resolves to the Coder seat",
              orch._seat_for_turn(coder_only, "execution", {"status": "in_progress"}) == orch.SEAT_CODER)
        check("Single-seat Coder handed a needs_review task falls back, not crashes",
              orch._seat_for_turn(coder_only, "execution", {"status": "needs_review"}) == orch.SEAT_CODER)

        # --- 2. Contracts follow the seat ----------------------------------------------
        crit = orch._role_output_contract(cfg, "execution", seat=orch.SEAT_CRITIC)
        code = orch._role_output_contract(cfg, "execution", seat=orch.SEAT_CODER)
        check("Critic seat gets the Critic contract, not the Coder's",
              crit == orch.ROLE_OUTPUT_CONTRACTS["critic"] and crit != code)
        check("Coder seat still gets the Coder contract",
              code == orch.ROLE_OUTPUT_CONTRACTS["coder"])

        # --- 3. The write guard is per-turn, not per-model ------------------------------
        mm, mem, tm, orch = build(tmp, {"brain": BRAIN})
        gen = stub(mm, CODE_TURN)
        mem.set_phase("execution")
        mem.add_itinerary_task(title="Add CSV export", description="Create exporter.py",
                               priority="high")
        task = mem.get_task_itinerary()[0]
        mem.update_itinerary_task(task["id"], {"status": "in_progress"})

        asyncio.run(orch.step_model_turn("brain"))
        wrote = [f for f in files_in(tm, "brain") if f.endswith(".py")]
        check("Multi-seat model DOES write a file on a Coder-seat turn", bool(wrote), str(wrote))
        check("Coder-seat prompt names the seat",
              any("SPEAKING AS: CODER" in p for p in gen.prompts),
              str([p[:60] for p in gen.prompts]))

        fresh = next(t for t in mem.get_task_itinerary() if t["id"] == task["id"])
        check("Coder-seat turn hands the task to review",
              fresh.get("status") == "needs_review", str(fresh.get("status")))

        # --- 4. Same model, Architect seat: code discarded ------------------------------
        mm, mem, tm, orch = build(tmp, {"brain": BRAIN})
        gen = stub(mm, CODE_TURN)
        mem.set_phase("execution")   # no task at all -> Architect seat
        asyncio.run(orch.step_model_turn("brain"))
        wrote = [f for f in files_in(tm, "brain") if f.endswith(".py")]
        check("Same model writes NOTHING on an Architect-seat turn", not wrote, str(wrote))
        guard = [m for m in orch.chat_history if "SUPERVISOR WROTE CODE" in m.get("content", "")]
        check("Architect-seat turn still posts the Role Guard notice", bool(guard))

        # --- 5. needs_review does not deadlock on a collapsed roster --------------------
        mm, mem, tm, orch = build(tmp, {"brain": BRAIN})
        mem.set_phase("execution")
        mem.add_itinerary_task(title="Review me", description="x", priority="high")
        t = mem.get_task_itinerary()[0]
        mem.update_itinerary_task(t["id"], {"status": "needs_review", "author_bot_id": "brain"})
        work = orch._outstanding_work(["brain"])
        check("Self-authored needs_review task still has an owner (no deadlock)",
              any(w[0]["id"] == t["id"] and w[1] == "brain" for w in work), str(work))

        # ...but a real second reviewer is still preferred when one exists.
        mm, mem, tm, orch = build(tmp, {"brain": BRAIN, "crit": "Critic"})
        mem.set_phase("execution")
        mem.add_itinerary_task(title="Review me", description="x", priority="high")
        t = mem.get_task_itinerary()[0]
        mem.update_itinerary_task(t["id"], {"status": "needs_review", "author_bot_id": "brain"})
        work = orch._outstanding_work(["brain", "crit"])
        check("A separate Critic is still preferred over self-review",
              any(w[0]["id"] == t["id"] and w[1] == "crit" for w in work), str(work))

        print()
        if failures:
            print(f"{len(failures)} CHECK(S) FAILED: " + ", ".join(failures))
            return 1
        print("All collapsed-roster checks passed.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
