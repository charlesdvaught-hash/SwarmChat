#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
import psutil

def check_system_resources():
    print("=== Checking Hardware Resources ===")
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)
    print(f"RAM Total: {total_gb:.2f} GB | Available: {available_gb:.2f} GB")

    vram_gb = 0
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,nounits,noheader"],
                text=True
            )
            total_v, free_v = res.strip().split("\n")[0].split(",")
            vram_gb = float(free_v.strip()) / 1024.0
            print(f"NVIDIA GPU Detected! Total VRAM: {float(total_v)/1024:.2f} GB | Free VRAM: {vram_gb:.2f} GB")
        except Exception as e:
            print(f"NVIDIA GPU check failed: {e}")
    else:
        print("No nvidia-smi detected. Running on CPU / integrated memory mode.")

    return {
        "ram_total_gb": total_gb,
        "ram_available_gb": available_gb,
        "vram_free_gb": vram_gb
    }

def check_dependencies():
    print("\n=== Checking System Dependencies ===")
    deps = {
        "python": sys.version.split()[0],
        "ollama": shutil.which("ollama") is not None,
        "git": shutil.which("git") is not None,
        "node": shutil.which("node") is not None,
        "npm": shutil.which("npm") is not None
    }

    for dep, status in deps.items():
        if isinstance(status, bool):
            symbol = "✓" if status else "✗ (Optional/Missing)"
            print(f"  [{symbol}] {dep}")
        else:
            print(f"  [✓] {dep}: {status}")

    return deps

def main():
    print("SwarmChat Diagnostic & Setup Utility")
    print("------------------------------------\n")
    check_system_resources()
    check_dependencies()
    print("\nSetup diagnostics complete.")

if __name__ == "__main__":
    main()
