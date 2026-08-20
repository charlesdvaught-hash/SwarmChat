"""Prints hardware budget vs the VRAM the active roster actually needs. Read-only."""
import json
import os

from backend.models import ModelManager

mm = ModelManager()
hw = mm.get_hardware_info()
print("HW", {k: hw.get(k) for k in ("gpu_name", "vram_total_gb", "vram_free_gb", "ram_total_gb")})

d = json.load(open(".swarmchat/roster.json", encoding="utf-8"))
active = d.get("active_model_ids", [])
total = 0.0
for mid, c in d["known_models"].items():
    if mid not in active:
        continue
    p = mm.resolve_gguf_path(c.get("gguf_path") or c.get("model_name", ""))
    size = round(os.path.getsize(p) / (1024 ** 3), 2) if p and os.path.exists(p) else None
    if size:
        total += size
    print("%-20s %-34s %-11s %-8s ctx=%s" % (
        c.get("role"), (c.get("name") or "")[:34], c.get("provider"), size, c.get("max_context_tokens")))
print("TOTAL roster GB:", round(total, 2))
