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
    
    cuda_available = False
    vram_gb = 0
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,nounits,noheader"],
                text=True
            )
            parts = res.strip().split("\n")[0].split(",")
            gpu_name = parts[0].strip()
            total_v, free_v = parts[1].strip(), parts[2].strip()
            vram_gb = float(free_v) / 1024.0
            cuda_available = True
            print(f"⚡ NVIDIA GPU Detected: {gpu_name}")
            print(f"   Total VRAM: {float(total_v)/1024:.2f} GB | Free VRAM: {vram_gb:.2f} GB")
        except Exception as e:
            print(f"NVIDIA GPU check note: {e}")
    else:
        print("No nvidia-smi detected. SwarmChat will run in pure CPU mode.")

    return {
        "ram_total_gb": total_gb,
        "ram_available_gb": available_gb,
        "vram_free_gb": vram_gb,
        "cuda_available": cuda_available
    }

def check_dependencies():
    print("\n=== Checking System Runtimes & GPU Accelerators ===")
    deps = {
        "python": sys.version.split()[0],
        "ollama": shutil.which("ollama") is not None,
        "nvcc (CUDA Toolkit)": shutil.which("nvcc") is not None,
        "git": shutil.which("git") is not None,
        "node": shutil.which("node") is not None,
        "npm": shutil.which("npm") is not None
    }
    
    for dep, status in deps.items():
        if isinstance(status, bool):
            symbol = "✓" if status else "✗ (Optional)"
            print(f"  [{symbol}] {dep}")
        else:
            print(f"  [✓] {dep}: {status}")

    if deps.get("nvcc (CUDA Toolkit)") or shutil.which("nvidia-smi"):
        print("\n💡 Optional CUDA Acceleration for llama-cpp-python:")
        print("   - PowerShell:")
        print('     $env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir')
        print("   - Command Prompt (cmd.exe):")
        print('     set CMAKE_ARGS="-DGGML_CUDA=on" && pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir')
        print("   - Linux/macOS (Bash):")
        print('     CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir')
        print("   Note: Ollama automatically uses your GPU out of the box without any extra steps!")

    return deps

def main():
    print("SwarmChat Diagnostic & Setup Utility")
    print("------------------------------------\n")
    check_system_resources()
    check_dependencies()
    print("\nSetup diagnostics complete.")

if __name__ == "__main__":
    main()
