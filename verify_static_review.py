"""Verifies the deterministic Critic pre-pass (ruff) without needing an LLM.

Why this exists: asking a 3-4B model to "review this code" produces agreeable prose, while
ruff produces `game.py:14:5: F821 Undefined name 'board'` in ~20ms. The static gate runs
BEFORE the Critic model, so:
  - real defects (undefined names, syntax errors) bounce the task straight back to the Coder
    with no model load, no generation, and no chance of the Critic saying APPROVE anyway;
  - clean files still go to the Critic, but with the linter's verdict in the prompt so it
    doesn't invent syntax problems ruff would have caught.

A missing ruff must read as "no opinion", never as "the code is clean".
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

# F821: `total` is never defined. The classic 3-4B failure - it referenced a variable it
# described in prose but never wrote.
BROKEN = """# filename: broken.py
def add_rows(rows):
    for r in rows:
        total += r
    return total
"""

# Clean: parses, no undefined names, has an entry point.
CLEAN = """# filename: clean.py
def add_rows(rows):
    total = 0
    for r in rows:
        total += r
    return total


def main():
    print(add_rows([1, 2, 3]))


if __name__ == "__main__":
    main()
"""

# Advisory only - unused import. Real, but not worth failing a task over.
SMELLY = """# filename: smelly.py
import json


def main():
    print("hi")


if __name__ == "__main__":
    main()
"""

failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail and not ok else ""))
    if not ok:
        failures.append(name)


_seq = [0]


def build(tmp):
    _seq[0] += 1
    tmp = os.path.join(tmp, f"case{_seq[0]}")
    os.makedirs(tmp, exist_ok=True)
    mm, tm = ModelManager(), ToolManager(workspace_root=tmp)
    mem = MemoryManager(storage_dir=os.path.join(tmp, "mem"))
    orch = Orchestrator(mm, mem, tm)
    orch.chat_history = []
    orch.save_chat_history = lambda *a, **k: None
    orch.models = {
        "coder": {"id": "coder", "name": "Coder", "role": "Coder", "provider": "ollama",
                  "model_name": "stub", "enabled": True},
        "crit": {"id": "crit", "name": "Crit", "role": "Critic", "provider": "ollama",
                 "model_name": "stub", "enabled": True},
    }
    orch.known_models = dict(orch.models)
    mem.set_phase("execution")
    return mm, mem, tm, orch


def seed(tm, mem, source, filename):
    """Write a file into the Coder's sandbox and stage a needs_review task for it."""
    tm.bot_workspace_write("coder", filename, source)
    mem.add_itinerary_task(title=f"Build {filename}", description="x", priority="high")
    t = mem.get_task_itinerary()[-1]
    mem.update_itinerary_task(t["id"], {
        "status": "needs_review", "filename": filename, "author_bot_id": "coder"
    })
    return next(x for x in mem.get_task_itinerary() if x["id"] == t["id"])


def main():
    tmp = tempfile.mkdtemp(prefix="swarmchat_static_")
    try:
        # --- 0. Is ruff even here? -----------------------------------------------------
        _, mem0, tm0, orch0 = build(tmp)
        t0 = seed(tm0, mem0, BROKEN, "broken.py")
        probe = tm0.lint_file("broken.py", bot_id="coder")
        if not probe.get("available", True):
            print("SKIP  ruff is not installed - install with: python -m pip install ruff")
            return 0
        check("lint_file runs and reports success", probe.get("success"), str(probe))

        # --- 1. Real defect is detected ------------------------------------------------
        check("Undefined name is a BLOCKING finding",
              any("F821" in ln for ln in probe.get("blocking", [])), str(probe.get("blocking")))
        check("Findings name the file, not an absolute path",
              all("C:\\" not in ln and "/tmp/" not in ln for ln in probe.get("blocking", [])),
              str(probe.get("blocking")))

        # --- 2. The gate bounces it back to the Coder, no model turn -------------------
        handled = asyncio.run(orch0._critic_static_gate(t0))
        check("Static gate reports handled for a broken file", handled)
        fresh = next(x for x in mem0.get_task_itinerary() if x["id"] == t0["id"])
        check("Broken file is sent back as failed",
              fresh.get("status") == "failed", str(fresh.get("status")))
        check("Failure reason carries the actual lint line",
              "F821" in (fresh.get("blocked_reason") or ""), str(fresh.get("blocked_reason"))[:120])
        check("Attempt count incremented", fresh.get("attempt_count", 0) >= 1)
        check("A Static Review notice is posted to chat",
              any("LINT FAILED" in m.get("content", "") for m in orch0.chat_history))

        # --- 3. The Critic's turn is actually skipped ----------------------------------
        mm3, mem3, tm3, orch3 = build(tmp)
        called = {"n": 0}

        async def _never(*a, **kw):
            called["n"] += 1
            return "APPROVE"
        mm3.generate_response = _never
        seed(tm3, mem3, BROKEN, "broken.py")
        res = asyncio.run(orch3.step_model_turn("crit"))
        check("Critic turn is skipped entirely", res.get("skipped") is True, str(res)[:160])
        check("No generation happened", called["n"] == 0, f"generate called {called['n']}x")

        # --- 4. A clean file is NOT blocked, and the Critic still runs -----------------
        mm4, mem4, tm4, orch4 = build(tmp)
        t4 = seed(tm4, mem4, CLEAN, "clean.py")
        check("Clean file produces no blocking findings",
              not tm4.lint_file("clean.py", bot_id="coder").get("blocking"),
              str(tm4.lint_file("clean.py", bot_id="coder")))
        check("Static gate does NOT handle a clean file",
              asyncio.run(orch4._critic_static_gate(t4)) is False)

        prompts = []

        async def _capture(model_config=None, system_prompt="", messages=None, **kw):
            prompts.append(system_prompt)
            return "Looks correct. APPROVE"
        mm4.generate_response = _capture
        res4 = asyncio.run(orch4.step_model_turn("crit"))
        check("Clean file: the Critic actually gets a turn", not res4.get("skipped"), str(res4)[:120])
        check("Critic prompt states ruff found nothing",
              any("no defects" in p for p in prompts),
              str([p[-160:] for p in prompts])[:200])

        # --- 5. Advisory-only findings do not fail the task ---------------------------
        mm5, mem5, tm5, orch5 = build(tmp)
        t5 = seed(tm5, mem5, SMELLY, "smelly.py")
        rep = tm5.lint_file("smelly.py", bot_id="coder")
        check("Unused import is ADVISORY, not blocking",
              any("F401" in ln for ln in rep.get("advisory", [])) and not rep.get("blocking"),
              str(rep))
        check("Advisory-only file is not sent back",
              asyncio.run(orch5._critic_static_gate(t5)) is False)
        ctx = orch5._static_review_context(t5)
        check("Advisory findings still reach the Critic's prompt", "F401" in ctx, ctx[:160])

        # --- 6. Missing ruff means 'no opinion', never 'clean' -------------------------
        mm6, mem6, tm6, orch6 = build(tmp)
        t6 = seed(tm6, mem6, BROKEN, "broken.py")
        tm6.lint_file = lambda *a, **kw: {"success": False, "available": False,
                                          "error": "ruff is not installed."}
        check("ruff missing: gate does not fail the task",
              asyncio.run(orch6._critic_static_gate(t6)) is False)
        fresh6 = next(x for x in mem6.get_task_itinerary() if x["id"] == t6["id"])
        check("ruff missing: task stays on needs_review for the model",
              fresh6.get("status") == "needs_review", str(fresh6.get("status")))
        check("ruff missing: no false 'code is clean' claim in the prompt",
              orch6._static_review_context(t6) == "", orch6._static_review_context(t6)[:120])

        print()
        if failures:
            print(f"{len(failures)} CHECK(S) FAILED: " + ", ".join(failures))
            return 1
        print("All static-review checks passed.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
