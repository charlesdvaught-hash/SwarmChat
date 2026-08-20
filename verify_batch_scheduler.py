"""Verifies residency-major scheduling, the batch cap, and task isolation. No LLM needed.

Simulates a roster where only some models are 'loaded' and checks which speaker the
scheduler picks, counting how many times it would have to swap a model in.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath("."))

from backend.models import ModelManager
from backend.memory import MemoryManager
from backend.tools import ToolManager
from backend.orchestrator import Orchestrator

failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail and not ok else ""))
    if not ok:
        failures.append(name)


_case = [0]


def build(tmp, resident):
    """resident = set of model ids to pretend are loaded in VRAM.

    Each call gets its own storage dir - MemoryManager persists the itinerary, so sharing a
    directory between cases let an earlier case's tasks leak into a later one."""
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
    mem.set_phase("execution")
    # Pretend these models are live llama.cpp instances.
    for m in resident:
        mm.gguf_instances[m] = object()
    return mem, orch


def add(mem, title, status, filename=None):
    t = mem.add_itinerary_task(title=title, description="do " + title, priority="high")
    upd = {"status": status}
    if filename:
        upd["filename"] = filename
    mem.update_itinerary_task(t["id"], upd)
    return t["id"]


def main():
    tmp = tempfile.mkdtemp(prefix="swarmchat_sched_")
    try:
        # --- 1. Residency: only the Coder is loaded, and work exists for Coder AND Critic.
        # Precedence says needs_review (Critic) outranks in_progress (Coder), but the Critic
        # is not resident - the loaded Coder should be used instead of swapping.
        mem, orch = build(tmp, {"code"})
        add(mem, "review-me", "needs_review", "a.py")
        add(mem, "code-me", "in_progress", "b.py")
        spk = orch.get_next_speaker()
        check("Prefers the loaded Coder over a higher-precedence unloaded Critic", spk == "code",
              "picked %r" % spk)

        # --- 2. Batch cap: give the Coder 5 tasks, nothing else resident.
        mem, orch = build(tmp, {"code"})
        for i in range(5):
            add(mem, "task%d" % i, "in_progress", "f%d.py" % i)
        picks = []
        for _ in range(5):
            picks.append(orch.get_next_speaker())
        check("Cap does not stall the room when nothing else is resident",
              all(p == "code" for p in picks), "picks=%s" % picks)

        # --- 3. Batch cap with an alternative resident model: after EXECUTION_BATCH_CAP
        # coder turns, the resident Critic should get a turn.
        mem, orch = build(tmp, {"code", "crit"})
        for i in range(5):
            add(mem, "c%d" % i, "in_progress", "f%d.py" % i)
        add(mem, "review", "needs_review", "r.py")
        picks = [orch.get_next_speaker() for _ in range(4)]
        check("Batch cap hands over to another resident model",
              picks[:1] != [] and "crit" in picks, "picks=%s" % picks)
        check("Cap respected: no more than EXECUTION_BATCH_CAP coder turns in a row",
              picks.count("code") <= orch.EXECUTION_BATCH_CAP, "picks=%s" % picks)

        # --- 4. Swap only when drained: nothing resident can work.
        mem, orch = build(tmp, {"code"})
        add(mem, "review-only", "needs_review", "a.py")
        spk = orch.get_next_speaker()
        check("Swaps in a model only when no resident model has work", spk == "crit",
              "picked %r" % spk)

        # --- 5. Task isolation: the pinned task drives context, not 'most urgent'.
        mem, orch = build(tmp, {"code"})
        t1 = add(mem, "first", "in_progress", "one.py")
        t2 = add(mem, "second", "in_progress", "two.py")
        orch.get_next_speaker()
        pinned = orch._current_task()
        check("Scheduler pins exactly one task for the turn", orch.pinned_task_id in (t1, t2),
              "pinned=%r" % orch.pinned_task_id)
        ctx = orch._build_task_context(pinned)
        other_title = "second" if pinned["title"] == "first" else "first"
        check("Task context contains only the pinned task",
              pinned["title"] in ctx and other_title not in ctx, ctx[:160])

        # --- 6. Episode isolation: checkpoints from a sibling task's file are excluded.
        mem, orch = build(tmp, {"code"})
        tid = add(mem, "mine", "in_progress", "mine.py")
        mem.record_episode(author="Bill", summary="worked on someone else's file",
                           action="Handoff", modified_files=["other.py"])
        mem.record_episode(author="Bill", summary="worked on my file",
                           action="Handoff", modified_files=["mine.py"])
        task = [t for t in mem.get_task_itinerary() if t["id"] == tid][0]
        ep = orch._build_episode_context(task)
        check("Episode context keeps this task's file", "my file" in ep, ep[:200])
        check("Episode context drops a sibling task's file", "someone else's file" not in ep, ep[:200])

        # --- 7. Swap accounting: how many model loads a 3-task project would need.
        # Task-major (old) = 3 stages x 3 tasks = 9 switches. Residency-major should be far fewer.
        mem, orch = build(tmp, {"code"})
        for i in range(3):
            add(mem, "t%d" % i, "in_progress", "f%d.py" % i)
        switches, prev = 0, None
        for _ in range(9):
            s = orch.get_next_speaker()
            if s != prev:
                switches += 1
                prev = s
            # advance the task the scheduler just pinned to its next stage
            cur = orch._current_task()
            if cur:
                nxt = {"in_progress": "needs_review", "needs_review": "needs_test",
                       "needs_test": "completed"}.get(cur.get("status"), "completed")
                mem.update_itinerary_task(cur["id"], {"status": nxt})
                if nxt in ("needs_review", "needs_test"):
                    orch.model_manager.gguf_instances.setdefault(
                        "crit" if nxt == "needs_review" else "test", object())
        check("Residency-major needs fewer role switches than task-major's 9", switches < 9,
              "switches=%d" % switches)
        print("      (role switches for a 3-task project: %d; old task-major design: 9)" % switches)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print("%d CHECK(S) FAILED: %s" % (len(failures), ", ".join(failures)))
        sys.exit(1)
    print("All batch-scheduler checks passed.")


if __name__ == "__main__":
    main()
