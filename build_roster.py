"""Builds the 3-model, all-<=5B roster and backs up whatever was there before.

Sizing (Q4/Q5 on disk, + ~0.25GB KV each at n_ctx 8192 with q8 KV):
    Architect  NVIDIA-Nemotron3-Nano-4B-Q4_K_M      2.64 GB
    Coder      qwen3.5-4B-super-coder.Q4_0          2.43 GB
    Critic     qwen2.5-coder-3b-instruct-q4_k_m     1.96 GB
    Tester     -- no model, run_python/pytest --    0.00 GB
                                          weights   7.03 GB
                                          + KV q8   ~0.75 GB
                                          total     ~7.78 GB of ~10.5 GB usable

Two things this fixes in the previous roster, beyond the size:
  1. Nemotron-4B was active TWICE, as Architect *and* Critic. Same weights, two seats,
     so the "second opinion" was the same model agreeing with itself - and it paid for
     the load twice.
  2. Every seat ran at temperature 0.7, including the Coder. Code generation at 0.7 is
     why the same task produced a different (and differently broken) file each attempt.
     Coder drops to 0.15, Critic 0.3, Architect 0.4.

The Tester seat is deliberately unfilled: _execute_task_test_run() already parses the
file, checks it has an entry point, and runs it, setting completed/failed without asking
a model. Adding a Tester model would cost ~2 GB to produce an opinion about a result the
pipeline already knows for certain.

Run from the repo root:  python build_roster.py
"""
import json
import os
import shutil
import sys
import time

DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
ROSTER = os.path.join(".swarmchat", "roster.json")

SEATS = [
    {
        "id": "seat_architect",
        "name": "Architect (Nemotron3-Nano-4B)",
        "role": "Architect",
        "gguf": "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
        # Best tool discipline of the <=5B field in the v2 benchmark (25/36), which is what
        # the Architect seat actually needs - it emits [UPDATE_TASK: ...] directives, not code.
        "temperature": 0.4,
        "repeat_penalty": 1.1,
    },
    {
        "id": "seat_coder",
        "name": "Coder (Qwen3.5-4B-super-coder)",
        "role": "Coder",
        "gguf": "qwen3.5-4B-super-coder.Q4_0.gguf",
        # Low temperature: the Coder's contract is "emit the COMPLETE file". Sampling
        # variety is not a feature here.
        # CAVEAT: this file is Q4_0, and the model card warns Q4_0 degrades its codegen.
        # A K-quant requant is the single cheapest quality win available on this roster.
        "temperature": 0.15,
        "repeat_penalty": 1.05,
    },
    {
        "id": "seat_critic",
        "name": "Critic (Qwen2.5-Coder-3B)",
        "role": "Critic",
        "gguf": "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        # Scored 6/6 on planted bugs in the v2 benchmark - tied with VibeThinker-3B on the
        # one metric this seat exists for, while being 1.1 GB smaller. VibeThinker is the
        # upgrade if you ever want it: same role, +1.1 GB, and it tag-spams at large budgets.
        "temperature": 0.3,
        "repeat_penalty": 1.1,
    },
]

N_CTX = 8192


def build_entry(seat: dict) -> dict:
    path = os.path.join(DOWNLOADS, seat["gguf"])
    return {
        "id": seat["id"],
        "name": seat["name"],
        "role": seat["role"],
        "provider": "gguf_local",
        "model_name": path,
        "gguf_path": path,
        "mmproj_path": "",
        "escalation_model_path": "",
        "max_context_tokens": N_CTX,
        "temperature": seat["temperature"],
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": seat["repeat_penalty"],
        "api_key": "",
        "enabled": True,
        "custom_start_prompt": None,
        "custom_execution_prompt": None,
        "live_status": "Idle / Live in Chat",
    }


def main() -> int:
    if not os.path.isdir(".swarmchat"):
        print("ERROR: run this from the repo root (no .swarmchat directory here).")
        return 1

    missing = [s["gguf"] for s in SEATS if not os.path.exists(os.path.join(DOWNLOADS, s["gguf"]))]
    if missing:
        print("ERROR: these GGUFs are not in ~/Downloads:")
        for m in missing:
            print(f"  - {m}")
        return 1

    existing = {}
    if os.path.exists(ROSTER):
        backup = f"{ROSTER}.bak.{int(time.time())}"
        shutil.copy2(ROSTER, backup)
        print(f"Backed up previous roster -> {backup}")
        try:
            with open(ROSTER, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARNING: could not parse the old roster ({e}); starting clean.")
            existing = {}

    # Keep the model library - it is the user's collection and costs nothing to retain.
    # Only the ACTIVE set changes.
    known = dict(existing.get("known_models") or {})
    prev_active = list(existing.get("active_model_ids") or [])

    total_gb = 0.0
    for seat in SEATS:
        known[seat["id"]] = build_entry(seat)
        total_gb += os.path.getsize(os.path.join(DOWNLOADS, seat["gguf"])) / (1024 ** 3)

    roster = {
        "moderator_model_id": "seat_architect",
        "known_models": known,
        "active_model_ids": [s["id"] for s in SEATS],
    }
    with open(ROSTER, "w", encoding="utf-8") as f:
        json.dump(roster, f, indent=2)

    kv_gb = 0.25 * len(SEATS)
    print(f"\nWrote {ROSTER}")
    print(f"  previously active: {len(prev_active)} model(s)")
    print(f"  now active:        {len(SEATS)} model(s) + deterministic Tester\n")
    for seat in SEATS:
        gb = os.path.getsize(os.path.join(DOWNLOADS, seat["gguf"])) / (1024 ** 3)
        print(f"  {seat['role']:<10} {seat['gguf']:<40} {gb:5.2f} GB  temp={seat['temperature']}")
    print(f"  {'Tester':<10} {'(run_python + pytest, no model)':<40} {0.0:5.2f} GB")
    print(f"\n  weights {total_gb:.2f} GB + KV q8 ~{kv_gb:.2f} GB = ~{total_gb + kv_gb:.2f} GB")
    print("  budget  ~10.5 GB usable on a 12 GB card after desktop + UI")
    print(f"  headroom ~{10.5 - total_gb - kv_gb:.2f} GB\n")
    print("NOTE: the Coder GGUF is Q4_0 and its model card warns that hurts codegen.")
    print("      Requanting it to Q4_K_M/Q5_K_M is the cheapest quality win on this roster.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
