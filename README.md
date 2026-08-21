# SwarmChat

A desktop app that puts several local GGUF models — plus optional cloud models (Claude, Groq, Gemini) — into one shared chat room and has them build software together. Each model takes on a role (Architect, Coder, Critic, Tester/Debugger) and the room runs on a real state machine: discuss, plan, get the plan reviewed, build, get the build reviewed, ship. Scales from a single laptop GPU running one collapsed model up to a full multi-model rig.

## What it does

- **Discussion, then execution.** The room talks through the problem first — asking real questions instead of guessing, no tools available yet — then moves into execution once there's an actual plan, at which point file writes and shell commands come with risk-tiered permission checks.
- **Thinking off by default.** Models answer directly instead of padding every turn with visible chain-of-thought. The Architect gets a little extra reasoning room for planning calls; nobody else does.
- **Sized to your hardware, automatically.** SwarmChat reads your available system RAM and VRAM and won't load a roster it can't fit.
- **GPU acceleration where you have it.** CUDA (and ROCm through Ollama) work out of the box; falls back to CPU cleanly if you don't have a GPU.
- **Risk-tiered permissions.** Read-only tools (search, file read) just run. File writes go through a vote or the Moderator. Shell commands always need a human to click yes.
- **A real project memory.** Shared state and decisions get written to `shared_memory.json` plus markdown summaries, so a long build survives a context reset instead of forgetting what it already decided.
- **No build step to get started.** The web UI ships pre-built — run `run.bat` or `run.sh` and you're in the browser in a few seconds, no Node or npm required.

## Recent changes

- **Any model can cover a missing seat.** If nothing in your roster is labeled Coder, the Architect steps in rather than the task just sitting there untouched. Same idea for Tester: falls back to Critic, then Coder, then Architect. This mirrors the fallback chain the sandbox's auto-repair loop already used, so it's consistent behavior rather than a special case.
- **Role matching fixed for "developer" / "engineer" / "reviewer" labels.** These used to work fine once a model was already picked for a seat, but could get passed over when the app decided *who* to hand a task to in the first place. One shared list now, used everywhere.
- **Discussion phase only asks what actually matters.** The Architect skips anything it can reasonably decide on its own, phrases user-facing questions in plain language, and can park an admin decision without stalling the room. A vote needs two different models to actually agree — one model can't out-vote itself across three seats.
- **A lint pass runs before the Critic does.** [ruff](https://docs.astral.sh/ruff/) checks the file first; real syntax errors and undefined names go straight back to the Coder without spending a Critic turn on them. A clean file gets an explicit "ruff found nothing" note so the Critic isn't inventing problems that aren't there.
- **A plan has to survive review before execution opens.** The Architect can't just declare a plan ready — Critic review and a buildability check both have to clear first (or the step is skipped cleanly if that seat isn't filled).
- **Execution batches work by what's already loaded**, instead of swapping models in and out every single turn.
- **One queue decides who speaks next.** The discussion-phase queue and the execution work-queue used to be able to disagree about that; now there's one source of truth, and `@mention` still jumps the line.
- **A hung turn no longer hangs the room.** 90 seconds and it moves on to the next model.

## Quick start

**Windows** — double-click `run.bat`, or from a terminal:
```cmd
run.bat
```

**Linux / macOS**
```bash
chmod +x run.sh
./run.sh
```

Either script checks your Python dependencies, runs a quick diagnostic, and opens `http://localhost:8000` in your browser.

## GPU acceleration

**Ollama** picks up CUDA or ROCm automatically — nothing to configure. See [Ollama's GPU docs](https://github.com/ollama/ollama/blob/main/docs/gpu.md) if it's not detecting your card.

**Local GGUF via llama-cpp-python** needs to be built with CUDA support:

```powershell
# PowerShell
$env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```
```cmd
:: Command Prompt
set CMAKE_ARGS="-DGGML_CUDA=on" && pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```
```bash
# Linux / macOS
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

Full options (ROCm, Metal, Vulkan) are in the [llama-cpp-python docs](https://llama-cpp-python.readthedocs.io/en/latest/).

## Which models to run — RTX 5070 (12GB)

Everything below came out of running the app's actual directive parser and a real Architect → Coder → Critic → execution loop against each model — graded against real test execution, not a static checklist. Not vibes, not vendor benchmarks.

**Budget for about 8GB, not 12.** With a desktop and a browser open, measured free VRAM on a 12GB [RTX 5070](https://en.wikipedia.org/wiki/GeForce_RTX_50_series) runs around 8.1GB, and that number moves by more than a gigabyte depending on what else is running. Go past it and the driver quietly spills into system RAM, which is a lot slower than staying in VRAM — so leave yourself headroom rather than sizing to the full 12GB card.

**One model (~2.2GB), simplest setup:**
Run **Qwen3.8-4B-Q8_0** as your Architect and let it cover Coder and Critic too. It's the one 4B-class model in our testing that reviews its own code honestly — zero false rejections of correct work across two separate rounds. Tester is covered by the deterministic pytest backstop, no model needed.

**Two models (~7GB), the strongest pairing we've tested:**
**Qwen3-4B-Thinking-2507** (Q8_0, 4.3GB) as Architect + Coder, and **Qwen3.5-text-4B** as a separate Critic. Qwen3-4B-Thinking-2507 is the first model in our testing to post a perfect 18/18 on the tool-discipline suite, and it ties the best critic-quality score too — but it hasn't been cleared for reviewing its own code yet (it sometimes lands no verdict at all rather than a wrong one), so keep it paired with a separate Critic rather than collapsing roles onto it.

**Two models (~5GB), the lighter pairing:**
Qwen3.8-4B-Q8_0 as Architect + Coder, and **Qwen3.5-text-4B** as a separate Critic. Qwen3.5-text-4B is the strongest standalone Critic we tested (6/6 on a critic-quality retest), but don't have it review its own code — it hallucinates rejections of work it wrote itself about a third of the time. Keep these two roles on separate weights.

Tester is optional either way — skip it unless you specifically want one, in which case **Granite-4.0-h-micro** or **Qwen3-4B-Instruct-2507** are solid picks.

**More VRAM to spend?** **DeltaCoder-9B-DPO** is a reasonable escalator model if you can spare ~5-6GB for one weight. A fully separate Architect + Coder + Critic roster (three distinct 4B-class models loaded together) doesn't comfortably fit an 8GB budget once you account for KV cache overhead on top of the raw file sizes — that's why the two-model roster above shares a weight between Architect and Coder instead of running three apart.

**Models we tried and don't recommend:** qwen2.5-coder-3b as Critic (inconsistent verdicts, not a one-off), Qwen3.8-9B (fails role boundaries and approves code with real bugs in it), Qwen3.5-9B-Defiant as Critic (real reasoning, but rarely lands on an actual verdict), Phi-4-mini-instruct for Architect/discussion duty, and PrismML's [Bonsai-27B](https://docs.prismml.com/models/bonsai-27b) at 1-bit quantization (worse code accuracy than the picks above, and it never once produced a bare APPROVE/REJECT across any of our tests).

## Testing

```bash
pytest tests/
```

Full suite plus the `verify_*.py` scripts at the repo root cover the seat/role system, the plan gate, the batch scheduler, and the static-review pre-pass — see [pytest's docs](https://docs.pytest.org/) if you're adding new tests.
