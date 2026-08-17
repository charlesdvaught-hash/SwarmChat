# SwarmChat — Multi-Model Agentic Chat Room (Local GGUF + Cloud Hybrid) for budget to high end hardware.

SwarmChat is a user-friendly desktop application that loads multiple local GGUF models (via Ollama or custom runners) and optional cloud models (Claude, Groq, Gemini) into a shared, interactive chat room. Models act as specialized participants (Architect, Critic, Solver, Coder, Tester/Debugger) and collaborate under an optional Moderator.

## 🌟 Key Capabilities
- **Two-Phase Prompting:**
  - **💬 Discussion Phase (Default):** Models converse with role awareness and philosophical depth to build mutual understanding before writing files.
  - **⚡ Execution Phase:** Models switch to specialized task execution with risk-based tool authorization.
- **Thinking Disabled by Default:** Models respond concisely without chain-of-thought bloat; brief reasoning is enabled only for Moderator decisions.
- **Hardware-Aware Memory Engine:** Automatically senses available system RAM and NVIDIA VRAM, applying safety headroom checks.
- **GPU & CUDA Acceleration Ready:** Seamlessly utilizes CUDA/ROCm GPU acceleration if present (via Ollama or CUDA-enabled llama.cpp), while maintaining pure CPU core logic fallback.
- **Risk-Based Tool Voting & Admin Override:** Read-only research tools run automatically; file modifications require voting/Moderator decision; terminal commands require explicit Admin approval.
- **Continuous Shared Memory Archive:** Generates `shared_memory.json` and markdown summaries while supporting context nap refreshes.
- **Zero-Dependency 1-Click Launchers:** Pre-built web interface included so users do NOT need Node or npm installed! Double-click `run.bat` (Windows) or `./run.sh` (Linux/macOS).

## 🚀 Quick Start (1-Click)

### Windows
Double click `run.bat` or run in terminal:
```cmd
run.bat
```

### Linux / macOS
```bash
chmod +x run.sh
./run.sh
```

The script will automatically verify Python dependencies, run setup diagnostics, serve the pre-built web interface, and open `http://localhost:8000` in your browser.

## ⚡ CUDA / GPU Acceleration (Optional)
If your machine has an NVIDIA GPU and CUDA installed:
1. **Ollama:** Ollama automatically detects and utilizes CUDA/ROCm GPUs out of the box without any extra configuration.
2. **Custom GGUF / llama-cpp-python:**
   - **PowerShell:**
     ```powershell
     $env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
     ```
   - **Command Prompt (cmd.exe):**
     ```cmd
     set CMAKE_ARGS="-DGGML_CUDA=on" && pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
     ```
   - **Linux / macOS (Bash):**
     ```bash
     CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
     ```

## 🧪 Testing
Run backend unit and integration tests:
```bash
pytest tests/
```
