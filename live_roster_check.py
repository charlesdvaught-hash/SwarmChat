"""Loads every active roster model for real and measures actual VRAM. No mocks.

Everything up to this point has been arithmetic on file sizes. This is the first thing
that answers the only question that matters: do all three fit on the card AT THE SAME
TIME, on this machine, with whatever else is already running.

Reports per-model VRAM delta, whether the q8_0 KV cache path was actually taken by this
llama-cpp-python build, and the headroom left at the end.

Run from the repo root:  python live_roster_check.py
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from backend.models import (
    ModelManager, DEFAULT_N_CTX, KV_CACHE_QUANTIZED, KV_CACHE_QUANT,
)

ROSTER = os.path.join(".swarmchat", "roster.json")


def vram_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20
        ).stdout.strip().splitlines()[0]
        used, free, total = (int(x.strip()) for x in out.split(","))
        return used, free, total
    except Exception as e:
        print(f"  (nvidia-smi unavailable: {e})")
        return None, None, None


def main() -> int:
    if not os.path.exists(ROSTER):
        print("ERROR: no .swarmchat/roster.json - run build_roster.py first.")
        return 1
    with open(ROSTER, "r", encoding="utf-8") as f:
        roster = json.load(f)

    active = roster.get("active_model_ids") or []
    known = roster.get("known_models") or {}
    seats = [known[m] for m in active if m in known]
    if not seats:
        print("ERROR: roster has no active models.")
        return 1

    print(f"KV cache quantization: {KV_CACHE_QUANT!r} (quantized={KV_CACHE_QUANTIZED})")
    print(f"DEFAULT_N_CTX: {DEFAULT_N_CTX}\n")

    base_used, base_free, total = vram_used_mib()
    if base_used is None:
        print("Cannot measure VRAM on this machine; aborting.")
        return 1
    print(f"GPU total {total} MiB | already in use {base_used} MiB | free {base_free} MiB")
    print("(that baseline is desktop + browser + anything else already running)\n")

    mm = ModelManager()
    prev_used = base_used
    results = []
    t_start = time.time()

    for cfg in seats:
        name = cfg.get("name", cfg["id"])
        path = cfg.get("gguf_path") or cfg.get("model_name")
        n_ctx = cfg.get("max_context_tokens") or DEFAULT_N_CTX
        print(f"--- loading {cfg.get('role','?'):<10} {name}")
        t0 = time.time()
        try:
            mm.load_gguf_model(cfg["id"], path, n_ctx)
        except Exception as e:
            print(f"    LOAD FAILED: {e}\n")
            results.append((cfg.get("role"), name, None, None, f"FAILED: {e}"))
            continue
        dt = time.time() - t0
        used, free, _ = vram_used_mib()
        delta = used - prev_used
        prev_used = used
        loaded = cfg["id"] in mm.gguf_instances
        print(f"    +{delta} MiB VRAM | {dt:.1f}s | free now {free} MiB | resident={loaded}\n")
        results.append((cfg.get("role"), name, delta, free, "ok"))

    used, free, _ = vram_used_mib()
    resident = [m for m in active if m in mm.gguf_instances]

    print("=" * 68)
    print(f"Resident simultaneously: {len(resident)} / {len(seats)}")
    for role, name, delta, freeat, status in results:
        d = f"+{delta} MiB" if delta is not None else "  --  "
        print(f"  {str(role):<10} {name[:38]:<38} {d:>10}  {status}")
    print(f"\nVRAM used by the roster: {used - base_used} MiB")
    print(f"Free remaining:          {free} MiB")
    print(f"Total wall time:         {time.time() - t_start:.1f}s")

    ok = len(resident) == len(seats)
    if not ok:
        print("\nRESULT: NOT all models stayed resident. The capacity gate evicted at least")
        print("one, so the roster is too big for this card with the current baseline.")
    elif free < 500:
        print(f"\nRESULT: all {len(seats)} resident, but only {free} MiB free - too tight.")
        print("Any spike (a browser tab, a game, a second app) will force an eviction.")
        ok = False
    else:
        print(f"\nRESULT: all {len(seats)} models resident with {free} MiB to spare.")

    print("\nUnloading...")
    for m in list(mm.gguf_instances.keys()):
        mm.unload_gguf_model(m)
    _, free_after, _ = vram_used_mib()
    print(f"Free after unload: {free_after} MiB (baseline was {base_free} MiB)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
