"""Verifies the supervisory-Architect change without needing an LLM.

Checks:
  1. An Architect turn containing a fenced code block writes NO file and posts a Role Guard notice.
  2. The same code block from a Coder DOES write a file (the guard is role-scoped, not a global ban).
  3. is_moderator no longer injects the 'Chief Project Manager' text into any prompt.
  4. An exhausted turn schedule refills itself instead of posting a request to a model.
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
from backend.prompts import get_system_prompt

CODE_TURN = (
    "Here is the exporter we need:\n"
    "```python\n"
    "# filename: exporter.py\n"
    "def write_rows(path, rows):\n"
    "    return len(rows)\n"
    "```\n"
    "[UPDATE_TASK: title=Add CSV export, description=Create exporter.py, priority=high, status=in_progress]"
)

failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail and not ok else ""))
    if not ok:
        failures.append(name)


def stub(mm, text):
    async def _generate(model_config, system_prompt, messages, **kw):
        return text
    mm.generate_response = _generate


def build(tmp):
    mm, tm = ModelManager(), ToolManager(workspace_root=tmp)
    mem = MemoryManager(storage_dir=os.path.join(tmp, "mem"))
    orch = Orchestrator(mm, mem, tm)
    # Don't read or write the real ./.swarmchat chat log: residue from a previous run was
    # being restored here and shifting the message-order assertions below (false failures).
    orch.chat_history = []
    orch.save_chat_history = lambda *a, **k: None
    orch.models = {
        "sup": {"id": "sup", "name": "Otis", "role": "Architect", "provider": "ollama",
                "model_name": "stub", "enabled": True, "is_moderator": True},
        "cod": {"id": "cod", "name": "Bill", "role": "Coder", "provider": "ollama",
                "model_name": "stub", "enabled": True, "is_moderator": False},
    }
    orch.known_models = dict(orch.models)
    # No moderator assignment: "sup" holds the Architect role, which IS the supervisor seat.
    mem.set_phase("execution")
    return mm, mem, tm, orch


def files_in(tm, bot_id):
    d = tm.get_bot_workspace_dir(bot_id)
    return [f for f in os.listdir(d)] if os.path.isdir(d) else []


def main():
    tmp = tempfile.mkdtemp(prefix="swarmchat_verify_")
    try:
        # 1 + 2: same code block, two roles
        mm, mem, tm, orch = build(tmp)
        stub(mm, CODE_TURN)

        asyncio.run(orch.step_model_turn("sup"))
        sup_files = files_in(tm, "sup")
        check("Architect code block is NOT written to disk", sup_files == [],
              "found %s" % sup_files)
        guard = [m for m in orch.chat_history if "SUPERVISOR WROTE CODE" in m.get("content", "")]
        check("Architect gets a Role Guard notice in chat", len(guard) == 1,
              "%d notices" % len(guard))
        tasks = mem.get_itinerary() if hasattr(mem, "get_itinerary") else None
        check("Architect's UPDATE_TASK still processed (directive survives the guard)",
              tasks is None or len(tasks) >= 1, "itinerary=%s" % tasks)

        asyncio.run(orch.step_model_turn("cod"))
        cod_files = files_in(tm, "cod")
        check("Coder code block IS written to disk", "exporter.py" in cod_files,
              "found %s" % cod_files)

        # 3: moderator flag no longer changes the prompt
        p_mod = get_system_prompt("Architect", "Otis", phase="execution", is_moderator=True)
        p_plain = get_system_prompt("Architect", "Otis", phase="execution", is_moderator=False)
        check("is_moderator no longer injects Chief Project Manager text",
              "Chief Project Manager" not in p_mod and p_mod == p_plain)

        # 4: exhausted schedule refills itself silently
        mm2, mem2, tm2, orch2 = build(tmp)
        mem2.set_phase("discussion")
        orch2.turn_schedule = []
        before = len(orch2.chat_history)
        spk = orch2.get_next_speaker()
        nag = [m for m in orch2.chat_history[before:] if "ROSTER QUEUE EXHAUSTED" in m.get("content", "")]
        check("Exhausted schedule returns a speaker", spk in orch2.models, "got %r" % spk)
        check("Exhausted schedule posts no request-to-model message", not nag)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print("%d CHECK(S) FAILED: %s" % (len(failures), ", ".join(failures)))
        sys.exit(1)
    print("All supervisor-role checks passed.")


if __name__ == "__main__":
    main()
