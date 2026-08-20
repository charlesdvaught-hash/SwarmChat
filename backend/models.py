import logging
import asyncio
import gc
import os
import sys
import time
import psutil
import shutil
import subprocess
import httpx
from typing import Dict, Any, List, Optional

from backend.errors import ModelInvocationError, ModelLoadError, ProviderNotConfiguredError

logger = logging.getLogger(__name__)

# Default llama.cpp context window for a locally loaded GGUF when a model config doesn't
# pin one. Deliberately generous: the role prompts + reviewed file routinely exceed 1k
# tokens, and every model in the recommended 3-8B lineup handles 8k fine.
DEFAULT_N_CTX = 8192

# Ceiling on tokens generated in a single turn. Must comfortably fit a whole source file,
# since the Coder's contract is "emit the COMPLETE file in one fenced block".
MAX_GENERATION_TOKENS = 3072

# Hard ceiling on a single GGUF turn. A model that fits in VRAM finishes a 3k-token reply in
# well under this; one that has spilled to CPU will not, and the turn should fail loudly
# instead of pinning the room forever.
GGUF_GENERATION_TIMEOUT_SECONDS = float(os.environ.get("SWARMCHAT_GGUF_TIMEOUT", "300"))

# Turn-terminating tokens. Several GGUFs in the wild ship a ChatML chat template but declare
# a generic EOS (e.g. <|endoftext|>), so llama.cpp never stops at <|im_end|> and the model
# runs until it hits max_tokens - which reads as "the model rambled and never answered".
# Passing them as stop strings costs nothing on models that already stop correctly.
GGUF_STOP_STRINGS = ["<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>"]

# Prompt format for a GGUF that ships no chat template of its own.
#
# Guessing a *chat* format from the architecture turns out to be actively wrong. A file with no
# template was almost never chat-tuned in the first place - it is a base model or a completion
# -style fine-tune - so dressing it in its family's chat wrapper produces confident nonsense.
# Measured on the one template-less GGUF here (Coder-2B.Q5_K_M, arch gemma2): with the "gemma"
# chat format it scored 0/16 on the code-repair suite and answered Python questions with SQL;
# with "alpaca" it scored 10/16. Alpaca is also the format most such fine-tunes are trained on.
#
# So: default template-less models to Alpaca, and keep the architecture table only as the
# vocabulary for an explicit per-model {"chat_format": ...} override. n=1 - if a template-less
# model ever behaves worse under Alpaca, set its format explicitly rather than re-guessing here.
TEMPLATELESS_CHAT_FORMAT = "alpaca"

CHAT_FORMAT_BY_ARCH = {
    "gemma": "gemma", "gemma2": "gemma", "gemma3": "gemma", "gemma4": "gemma",
    "llama": "chatml", "qwen2": "chatml", "qwen3": "chatml", "qwen35": "chatml",
    "phi3": "zephyr", "mistral": "mistral-instruct",
}

# Reasoning models think by default and burn the whole token budget before answering.
# A model config can override with {"no_think": false} or a custom "think_control" string.
# Measured on this machine's GGUF set (swarmchat_bench, Aug 2026): /no_think is a clear win
# on SmolLM3 - its template actually parses the flag, and it went 16 -> 20 with both runaway
# <think> blocks gone. On Qwen3 / Qwen3.5 / Nemotron the effect was inside noise (+3, +1, -1,
# -1 across four models) because their templates take enable_thinking as a jinja kwarg, not a
# prompt token. So only SmolLM3 gets it by default; anything else opts in per model config
# with {"think_control": "/no_think"}.
NO_THINK_CONTROL_BY_ARCH = {
    "smollm3": "/no_think",       # SmolLM3 / TwIL-LM3 - template defaults enable_thinking=true
}


class ModelManager:
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.model_statuses: Dict[str, Dict[str, Any]] = {}
        self.gguf_instances: Dict[str, Any] = {}
        # Resolved path + last-use timestamp per loaded GGUF, so the loader can size an
        # eviction and pick a least-recently-used victim instead of guessing.
        self.gguf_paths: Dict[str, str] = {}
        self.model_last_used: Dict[str, float] = {}
        self.custom_search_paths: List[str] = []
        self._cached_search_paths: Optional[List[str]] = None

    def add_search_path(self, path: str):
        if path:
            abs_p = os.path.abspath(path)
            if abs_p not in self.custom_search_paths:
                self.custom_search_paths.append(abs_p)
                self._cached_search_paths = None

    def get_search_paths(self) -> List[str]:
        if self._cached_search_paths is not None:
            return self._cached_search_paths

        paths = [
            os.path.abspath("."),
            os.path.abspath("models"),
            os.path.expanduser("~/Downloads"),
            os.path.expanduser("~/models"),
            os.path.expanduser("~/.cache/lm-studio/models"),
        ]
        if os.name == "nt":
            paths.append("C:\\models")
        else:
            paths.append("/models")

        for cp in self.custom_search_paths:
            if cp not in paths:
                paths.append(cp)
        self._cached_search_paths = [p for p in paths if os.path.exists(p)]
        return self._cached_search_paths

    def resolve_gguf_path(self, raw_path: Optional[str]) -> Optional[str]:
        if not raw_path or not raw_path.strip():
            return None

        raw_path = raw_path.strip()

        # 1. Direct path check
        if os.path.exists(raw_path) and os.path.isfile(raw_path):
            return os.path.abspath(raw_path)

        basename = os.path.basename(raw_path)
        search_dirs = self.get_search_paths()

        # Helper for candidate evaluation
        def _check_candidates() -> Optional[str]:
            for sdir in search_dirs:
                c1 = os.path.join(sdir, raw_path)
                if os.path.exists(c1) and os.path.isfile(c1):
                    return os.path.abspath(c1)

                c2 = os.path.join(sdir, basename)
                if os.path.exists(c2) and os.path.isfile(c2):
                    return os.path.abspath(c2)
            return None

        found = _check_candidates()
        if found:
            return found

        # Helper for case-insensitive search
        def _case_insensitive_search() -> Optional[str]:
            for sdir in search_dirs:
                try:
                    entries = os.listdir(sdir)
                except OSError as e:
                    logger.warning("Skipping unreadable model search directory %s: %s", sdir, e)
                    continue
                for entry in entries:
                    if entry.lower() == basename.lower():
                        full = os.path.join(sdir, entry)
                        if os.path.isfile(full):
                            return os.path.abspath(full)
            return None

        return _case_insensitive_search()

    def is_llama_cpp_installed(self) -> bool:
        try:
            import llama_cpp
            return True
        except ImportError:
            return False

    def _run_install_cmd(self, cmd: List[str], env: Dict[str, str], timeout: int = 180) -> subprocess.CompletedProcess:
        """Executes a subprocess pip command for engine installation."""
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)

    def install_llama_cpp(self, use_cuda_wheels: bool = True) -> Dict[str, Any]:
        """Attempts 1-click installation of llama-cpp-python, preferring official pre-built CUDA wheels if GPU is present."""
        hw = self.get_hardware_info()
        env = os.environ.copy()

        # If NVIDIA GPU is detected and pre-built CUDA wheel installation requested
        if hw.get("gpu_name") and use_cuda_wheels:
            cmd = [
                sys.executable, "-m", "pip", "install", "llama-cpp-python",
                "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/wheels/cu121",
                "--no-cache-dir", "--force-reinstall"
            ]
        else:
            cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python"]
            if hw.get("gpu_name"):
                env["CMAKE_ARGS"] = "-DGGML_CUDA=on"

        try:
            res = self._run_install_cmd(cmd, env)
            if res.returncode == 0:
                return {"success": True, "message": "Successfully installed llama-cpp-python engine (CUDA wheel / build)."}

            # Fallback to standard pip install if wheel url failed
            fallback_cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python"]
            res2 = self._run_install_cmd(fallback_cmd, env)
            if res2.returncode == 0:
                return {"success": True, "message": "Successfully installed standard llama-cpp-python engine."}

            logger.error(
                "llama-cpp-python installation failed (primary and fallback attempts). primary=%s fallback=%s",
                res.stderr or res.stdout,
                res2.stderr or res2.stdout,
            )
            return {
                "success": False,
                "error": f"Installation failed: {res2.stderr or res2.stdout}",
                "primary_attempt_error": res.stderr or res.stdout,
            }
        except subprocess.TimeoutExpired as e:
            logger.exception("llama-cpp-python installation timed out")
            return {"success": False, "error": f"Engine installation timed out after {e.timeout}s.", "timed_out": True}
        except OSError as e:
            logger.exception("llama-cpp-python installation could not be started")
            return {"success": False, "error": f"Error during engine installation: {e}"}

    def update_model_status(self, model_id: str, status: str, error: Optional[str] = None, tok_per_sec: Optional[float] = None, vram_used_gb: float = 0.0, location: str = "RAM"):
        current = self.model_statuses.get(model_id, {})
        self.model_statuses[model_id] = {
            "status": status,  # "online", "offline", "error"
            "error": error,
            "tok_per_sec": tok_per_sec if tok_per_sec is not None else current.get("tok_per_sec", 0.0),
            "vram_used_gb": vram_used_gb,
            "location": location  # "VRAM", "RAM", or "Cloud"
        }

    def unload_gguf_model(self, model_id: str) -> bool:
        """Drops a loaded GGUF instance and actually releases its VRAM.

        Popping the dict alone is not enough: llama_cpp.Llama holds the context until it is
        closed, and any lingering reference (a traceback frame, a local in a caller) keeps
        several GB of VRAM pinned. That is what made 'unloaded' models keep occupying the GPU
        until the process exited."""
        llm = self.gguf_instances.pop(model_id, None)
        if llm is None:
            return False
        for closer in ("close", "_sampler_close", "__del__"):
            fn = getattr(llm, closer, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception as e:
                    logger.debug("Closing GGUF instance %s via %s failed: %s", model_id, closer, e)
        del llm
        gc.collect()
        self.model_last_used.pop(model_id, None)
        return True

    def loaded_gguf_size_gb(self) -> float:
        """Total on-disk size of every currently loaded GGUF - the practical VRAM/RAM footprint."""
        total = 0.0
        for m_id in list(self.gguf_instances.keys()):
            p = self.gguf_paths.get(m_id)
            if p and os.path.exists(p):
                total += os.path.getsize(p) / (1024 ** 3)
        return round(total, 2)

    def free_gguf_capacity(self, needed_gb: float, keep_ids: Optional[List[str]] = None) -> List[str]:
        """Unloads least-recently-used GGUFs until `needed_gb` fits. Returns the ids evicted.

        Without this the app happily accepts a roster larger than the GPU, loads models
        eagerly until llama.cpp starts thrashing or OOMs, and the next turn appears to hang."""
        keep = set(keep_ids or [])
        evicted: List[str] = []
        candidates = sorted(
            (m for m in self.gguf_instances if m not in keep),
            key=lambda m: self.model_last_used.get(m, 0.0)
        )
        for m_id in candidates:
            fit = self.can_load_model(needed_gb)
            if fit.get("allowed"):
                break
            if self.unload_gguf_model(m_id):
                evicted.append(m_id)
                self.update_model_status(m_id, status="online", error=None, vram_used_gb=0.0, location="Unloaded")
                logger.info("Evicted GGUF %s to make room for %.2f GB", m_id, needed_gb)
        return evicted

    def load_gguf_model(self, model_id: str, gguf_path: str, max_tokens: int = 2048, mmproj_path: Optional[str] = None, force_device: Optional[str] = None, chat_format: Optional[str] = None) -> Optional[Any]:
        if not self.is_llama_cpp_installed():
            self._fail_load(
                model_id,
                "llama-cpp-python engine not installed. Use 1-click installer in Settings."
            )

        resolved_gguf = self.resolve_gguf_path(gguf_path)
        if not resolved_gguf:
            searched = ", ".join(self.get_search_paths())
            self._fail_load(
                model_id,
                f"GGUF file not found at: '{gguf_path}'. Searched directories: [{searched}]"
            )

        if model_id in self.gguf_instances:
            # A cached instance keeps whatever n_ctx it was first loaded with. If an earlier
            # caller (the VRAM preloader used to pass 2048) built it smaller than this caller
            # needs, the generate path silently loses most of its token budget - which is how
            # reasoning models ended up never finishing a <think> block. Rebuild instead.
            cached = self.gguf_instances[model_id]
            try:
                cached_ctx = int(cached.n_ctx())
            except Exception:
                cached_ctx = max_tokens
            if cached_ctx >= max_tokens:
                self.model_last_used[model_id] = time.time()
                return cached
            logger.info(
                "Reloading %s: cached context %d is smaller than the requested %d",
                model_id, cached_ctx, max_tokens
            )
            self.unload_gguf_model(model_id)

        import llama_cpp

        # Capacity gate. Previously any number of models could be configured and every one of
        # them was loaded eagerly and never released, so the first turn that landed on a model
        # with no room left either OOM'd or silently fell back to CPU and appeared to hang.
        size_gb = round(os.path.getsize(resolved_gguf) / (1024 ** 3), 2)
        fit = self.can_load_model(size_gb, n_ctx=max_tokens)
        if not fit.get("allowed"):
            evicted = self.free_gguf_capacity(size_gb, keep_ids=[model_id])
            fit = self.can_load_model(size_gb, n_ctx=max_tokens)
            if evicted:
                logger.info("Unloaded %s to make room for %s (%.2f GB)", ", ".join(evicted), model_id, size_gb)
        if not fit.get("allowed"):
            self._fail_load(
                model_id,
                f"Not enough memory to load '{os.path.basename(resolved_gguf)}' ({size_gb} GB). "
                f"{fit.get('message', '')} Remove a model from the roster or pick a smaller quant."
            )

        hw = self.get_hardware_info()
        # Assume VRAM preferred by default if GPU present
        has_gpu = bool(hw.get("gpu_name") or hw.get("vram_total_gb", 0) > 0)
        if force_device == "gpu":
            # "gpu" is a preference, not a promise: a full offload that doesn't fit in free
            # VRAM is exactly the case that used to OOM or crawl. Fall back to RAM instead.
            fits_vram = hw.get("vram_free_gb", 0) >= (size_gb + 0.7)
            n_gpu_layers = -1 if (has_gpu and fits_vram) else 0
            location = "VRAM" if (has_gpu and fits_vram) else "RAM"
            if has_gpu and not fits_vram:
                logger.warning(
                    "Requested GPU load for %s (%.2f GB) but only %.2f GB VRAM free - loading on CPU instead",
                    model_id, size_gb, hw.get("vram_free_gb", 0)
                )
        elif force_device == "cpu":
            n_gpu_layers = 0
            location = "RAM"
        else:
            # Prefer VRAM if free VRAM exists and GPU is detected
            if has_gpu and hw.get("vram_free_gb", 0) >= (size_gb + 0.7):
                n_gpu_layers = -1
                location = "VRAM"
            else:
                n_gpu_layers = 0
                location = "RAM"

        # Resolve mmproj (clip/vision projector) if provided
        chat_handler = None
        resolved_mmproj = self.resolve_gguf_path(mmproj_path) if mmproj_path else None
        if resolved_mmproj:
            chat_handler = self._build_vision_chat_handler(model_id, resolved_mmproj)

        try:
            kwargs = {
                "model_path": resolved_gguf,
                "n_ctx": max_tokens,
                "n_gpu_layers": n_gpu_layers,
                "verbose": False
            }
            if chat_handler:
                kwargs["chat_handler"] = chat_handler

            # Flash attention shrinks the KV cache and is what the Qwen3.5 cards ask for
            # (-fa). Not every llama-cpp-python build accepts the argument, so fall back
            # rather than failing the load over it.
            try:
                llm = llama_cpp.Llama(flash_attn=True, **kwargs)
            except (TypeError, ValueError) as e:
                logger.debug("flash_attn unavailable for %s (%s); loading without it", model_id, e)
                llm = llama_cpp.Llama(**kwargs)

            # Some GGUFs (base-model conversions, sloppy requants) ship with no chat template
            # at all. llama-cpp-python then silently falls back to a Llama-2 prompt format,
            # and the model answers with garbage or nothing. Pick a format from the
            # architecture instead so the roles at least get coherent turns.
            try:
                meta = getattr(llm, "metadata", {}) or {}
                if chat_format:
                    # Explicit per-model override always wins. Needed for GGUFs trained on a
                    # non-chat format - e.g. Coder-2B.Q5_K_M, which is an Alpaca-style
                    # completion model and answers correctly with chat_format="alpaca" while
                    # producing unrelated SQL under any chat template.
                    llm.chat_format = chat_format
                    logger.info("Using configured chat_format=%r for %s", chat_format, model_id)
                elif not meta.get("tokenizer.chat_template"):
                    arch = str(meta.get("general.architecture", "")).lower()
                    llm.chat_format = TEMPLATELESS_CHAT_FORMAT
                    logger.warning(
                        "%s (arch %r) has no embedded chat template - it is probably a base or "
                        "completion-style model. Using chat_format=%r; set a per-model "
                        "\"chat_format\" if that is wrong.",
                        os.path.basename(resolved_gguf), arch, TEMPLATELESS_CHAT_FORMAT
                    )
            except Exception as e:
                logger.debug("Chat-format fallback probe failed for %s: %s", model_id, e)

            self.gguf_instances[model_id] = llm
            self.gguf_paths[model_id] = resolved_gguf
            self.model_last_used[model_id] = time.time()
            file_size_gb = size_gb
            self.update_model_status(
                model_id,
                status="online",
                error=None,
                vram_used_gb=file_size_gb if location == "VRAM" else 0.0,
                location=location
            )
            return llm
        except Exception as e:
            self._fail_load(model_id, f"Failed to load GGUF model: {e}", cause=e)

    def _fail_load(self, model_id: str, message: str, cause: Optional[BaseException] = None) -> None:
        """Records a load failure on the model status board and raises it to the caller."""
        logger.error("GGUF load failure for %s: %s", model_id, message)
        self.update_model_status(model_id, status="error", error=message)
        raise ModelLoadError(message, model_id=model_id, provider="gguf_local") from cause

    def _build_vision_chat_handler(self, model_id: str, resolved_mmproj: str) -> Optional[Any]:
        """Builds a vision projector chat handler, recording (not hiding) why it could not be built."""
        errors: List[str] = []
        for handler_name in ("Llava15ChatHandler", "NanoLlavaChatHandler"):
            try:
                from llama_cpp import llama_chat_handler
                return getattr(llama_chat_handler, handler_name)(clip_model_path=resolved_mmproj)
            except Exception as e:
                errors.append(f"{handler_name}: {e}")
        logger.warning(
            "No vision chat handler could be built for %s from mmproj '%s' (%s); continuing text-only.",
            model_id, resolved_mmproj, "; ".join(errors)
        )
        return None

    def get_hardware_info(self) -> Dict[str, Any]:
        mem = psutil.virtual_memory()
        total_ram_gb = round(mem.total / (1024 ** 3), 2)
        avail_ram_gb = round(mem.available / (1024 ** 3), 2)
        ram_percent = mem.percent

        vram_free_gb = 0.0
        vram_total_gb = 0.0
        gpu_name = None
        gpu_probe_error: Optional[str] = None

        nvidia_smi_cmd = shutil.which("nvidia-smi")
        if not nvidia_smi_cmd and os.name == "nt":
            # Check standard Windows nvidia-smi installation paths
            candidate_paths = [
                r"C:\Windows\System32\nvidia-smi.exe",
                r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
            ]
            for cp in candidate_paths:
                if os.path.exists(cp):
                    nvidia_smi_cmd = cp
                    break

        if nvidia_smi_cmd:
            try:
                res = subprocess.check_output(
                    [nvidia_smi_cmd, "--query-gpu=name,memory.total,memory.free", "--format=csv,nounits,noheader"],
                    text=True
                )
                parts = res.strip().split("\n")[0].split(",")
                gpu_name = parts[0].strip()
                vram_total_gb = round(float(parts[1].strip()) / 1024.0, 2)
                vram_free_gb = round(float(parts[2].strip()) / 1024.0, 2)
            except (subprocess.SubprocessError, OSError, IndexError, ValueError) as e:
                gpu_probe_error = f"nvidia-smi probe failed: {e}"
                logger.warning(gpu_probe_error)

        return {
            "ram_total_gb": total_ram_gb,
            "ram_available_gb": avail_ram_gb,
            "ram_percent": ram_percent,
            "gpu_name": gpu_name,
            "gpu_probe_error": gpu_probe_error,
            "vram_total_gb": vram_total_gb,
            "vram_free_gb": vram_free_gb,
            "ollama_available": self.check_ollama_status()
        }

    def check_ollama_status(self) -> bool:
        try:
            resp = httpx.get(f"{self.ollama_host}/api/version", timeout=1.5)
            return resp.status_code == 200
        except httpx.HTTPError as e:
            logger.debug("Ollama unreachable at %s: %s", self.ollama_host, e)
            return False

    def list_ollama_models(self) -> List[str]:
        if not self.check_ollama_status():
            return []
        try:
            resp = httpx.get(f"{self.ollama_host}/api/tags", timeout=2.0)
        except httpx.HTTPError as e:
            logger.warning("Failed to list Ollama models from %s: %s", self.ollama_host, e)
            return []
        if resp.status_code != 200:
            logger.warning("Ollama tag listing returned HTTP %s: %s", resp.status_code, resp.text[:200])
            return []
        try:
            return [m["name"] for m in resp.json().get("models", [])]
        except (ValueError, KeyError, TypeError) as e:
            logger.warning("Unexpected Ollama tag listing payload: %s", e)
            return []

    def can_load_model(self, estimated_size_gb: float, n_ctx: int = 4096, quant_type: Optional[str] = None) -> Dict[str, Any]:
        hw = self.get_hardware_info()
        avail_ram = hw["ram_available_gb"]
        avail_vram = hw["vram_free_gb"]
        headroom_gb = 1.5
        effective_avail = max(avail_ram - headroom_gb, 0.0) + avail_vram

        # Account for KV cache + compute buffer overhead (n_ctx * n_layer * n_kv_heads * head_dim * 2 * 2 bytes)
        # Approximate average LLM layer config: ~32 layers, ~8 KV heads, ~128 head dim for 1-3B models (~0.2 - 0.5 GB at 4k ctx)
        kv_cache_bytes = n_ctx * 32 * 8 * 128 * 2 * 2
        compute_buffer_bytes = 256 * 1024 * 1024  # ~256MB compute overhead
        overhead_gb = round((kv_cache_bytes + compute_buffer_bytes) / (1024 ** 3), 2)
        total_needed_gb = estimated_size_gb + overhead_gb

        quant_warning = ""
        # Surface quantization warning for small models (<=3B / ~3.5GB file) at <=Q4
        if estimated_size_gb <= 3.5 and quant_type:
            q_lower = quant_type.lower()
            if any(q in q_lower for q in ["q2", "q3", "q4", "q4_k_m", "q4_k_s", "q4_0", "q4_1"]):
                quant_warning = (
                    f" ⚠️ Quantization Warning: Model is ≤3B and loaded at {quant_type.upper()}. "
                    "Small models suffer severe instruction-following accuracy drops at Q4 or lower. "
                    "Q5_K_M minimum or Q6_K/Q8_0 is recommended for models ≤5B."
                )

        if total_needed_gb > effective_avail:
            return {
                "allowed": False,
                "warning": True,
                "quant_warning": quant_warning,
                "message": f"Total memory needed ({total_needed_gb:.2f} GB = {estimated_size_gb:.1f} GB model + {overhead_gb:.2f} GB KV/compute) exceeds safe available memory ({effective_avail:.1f} GB with headroom).{quant_warning}"
            }
        elif total_needed_gb > (effective_avail * 0.8):
            return {
                "allowed": True,
                "warning": True,
                "quant_warning": quant_warning,
                "message": f"High memory utilization warning: Loading model ({total_needed_gb:.2f} GB total) leaves tight RAM/VRAM margin.{quant_warning}"
            }
        return {
            "allowed": True,
            "warning": bool(quant_warning),
            "quant_warning": quant_warning,
            "message": f"Sufficient memory headroom available ({total_needed_gb:.2f} GB needed).{quant_warning}"
        }

    @staticmethod
    def get_action_json_schema() -> Dict[str, Any]:
        """Returns the JSON schema definition for SwarmChat action emissions."""
        return {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["log_memory", "update_task", "update_spec", "journal", "ready_for_execution", "request_discussion", "search_hf", "run_tests", "run_python", "vote_tool"]
                            },
                            "payload": {"type": "string"},
                            "task_id": {"type": "string"},
                            "status": {"type": "string"},
                            "title": {"type": "string"},
                            "tool_name": {"type": "string"},
                            "args": {"type": "object"}
                        },
                        "required": ["type"]
                    }
                }
            },
            "required": ["message", "actions"]
        }

    async def generate_response(
        self,
        model_config: Dict[str, Any],
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        use_gbnf: bool = False
    ) -> str:
        provider = model_config.get("provider", "ollama")
        model_name = model_config.get("model_name", "llama3.2:1b")
        api_key = model_config.get("api_key", "")

        # Role-based sampling parameter defaults
        role = model_config.get("role", "Participant")
        role_lower = role.lower()

        # Temperature by role: Tool/tag emission & refiner run at 0.0-0.2; ideation/brainstorming at 0.6-0.8
        if temperature is not None:
            temp = float(temperature)
        elif any(r in role_lower for r in ["coder", "tester", "debugger", "refiner"]):
            temp = 0.1
        elif any(r in role_lower for r in ["architect", "critic"]):
            temp = 0.7
        else:
            temp = float(model_config.get("temperature", 0.2 if "temperature" not in model_config else model_config["temperature"]))

        top_p = float(model_config.get("top_p", 0.9))
        top_k = int(model_config.get("top_k", 40))
        min_p = float(model_config.get("min_p", 0.05))

        # Repeat penalty: drop to 1.0-1.05 for Coder role specifically so legitimate repeated identifiers/indentation aren't penalized
        if "repeat_penalty" in model_config:
            repeat_penalty = float(model_config["repeat_penalty"])
        else:
            repeat_penalty = 1.02 if "coder" in role_lower else 1.1

        from backend.sanitizer import normalize_messages_for_gguf, sanitize_message_content
        role = model_config.get("role", "Participant")
        full_messages = normalize_messages_for_gguf(system_prompt, messages, role=role)

        model_id = model_config.get("id", f"{provider}_model")

        if provider == "ollama":
            if not self.check_ollama_status():
                self._fail_generation(
                    model_id, provider,
                    f"Ollama is not reachable at {self.ollama_host}. Start Ollama or switch the model provider."
                )
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    options = {
                        "temperature": temp,
                        "top_p": top_p,
                        "top_k": top_k,
                        "min_p": min_p,
                        "repeat_penalty": repeat_penalty
                    }
                    payload = {
                        "model": model_name,
                        "messages": full_messages,
                        "stream": False,
                        "options": options
                    }
                    if response_schema:
                        payload["format"] = response_schema

                    resp = await client.post(
                        f"{self.ollama_host}/api/chat",
                        json=payload
                    )
            except httpx.HTTPError as e:
                self._fail_generation(model_id, provider, f"Ollama request failed: {e}", cause=e)

            if resp.status_code != 200:
                self._fail_generation(
                    model_id, provider,
                    f"Ollama API error {resp.status_code} for model '{model_name}': {resp.text[:300]}"
                )
            try:
                raw_res = resp.json()["message"]["content"]
            except (ValueError, KeyError, TypeError) as e:
                self._fail_generation(model_id, provider, f"Unexpected Ollama response payload: {e}", cause=e)

            self.update_model_status(model_id, status="online", error=None)
            return sanitize_message_content(raw_res)

        elif provider in ["claude", "groq", "gemini"]:
            env_key_map = {
                "claude": "ANTHROPIC_API_KEY",
                "groq": "GROQ_API_KEY",
                "gemini": "GEMINI_API_KEY"
            }
            effective_key = api_key or os.environ.get(env_key_map.get(provider, ""), "")
            if not effective_key:
                self._fail_generation(
                    model_id, provider,
                    f"API key missing for cloud provider '{provider}'. Configure key in model settings or set {env_key_map.get(provider)}.",
                    error_cls=ProviderNotConfiguredError
                )

            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    if provider == "groq":
                        # Groq API (OpenAI-compatible)
                        resp = await client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {effective_key}", "Content-Type": "application/json"},
                            json={
                                "model": model_name or "llama-3.3-70b-versatile",
                                "messages": full_messages,
                                "temperature": temp,
                                "top_p": top_p
                            }
                        )
                        if resp.status_code != 200:
                            self._fail_generation(model_id, provider, f"Groq API error {resp.status_code}: {resp.text[:300]}")
                        raw_res = resp.json()["choices"][0]["message"]["content"]

                    elif provider == "claude":
                        # Anthropic Claude API
                        # Format system message separately for Claude
                        user_assistant_msgs = [m for m in full_messages if m["role"] != "system"]
                        resp = await client.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={
                                "x-api-key": effective_key,
                                "anthropic-version": "2023-06-01",
                                "content-type": "application/json"
                            },
                            json={
                                "model": model_name or "claude-3-5-sonnet-20241022",
                                "system": system_prompt,
                                "messages": user_assistant_msgs,
                                "max_tokens": 1024,
                                "temperature": temp
                            }
                        )
                        if resp.status_code != 200:
                            self._fail_generation(model_id, provider, f"Claude API error {resp.status_code}: {resp.text[:300]}")
                        raw_res = resp.json()["content"][0]["text"]

                    elif provider == "gemini":
                        # Google Gemini API
                        contents = []
                        for m in full_messages:
                            role = "user" if m["role"] in ["user", "system"] else "model"
                            contents.append({"role": role, "parts": [{"text": m["content"]}]})

                        gemini_model = model_name or "gemini-1.5-flash"
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={effective_key}"
                        resp = await client.post(
                            url,
                            json={"contents": contents, "generationConfig": {"temperature": temp}}
                        )
                        if resp.status_code != 200:
                            self._fail_generation(model_id, provider, f"Gemini API error {resp.status_code}: {resp.text[:300]}")
                        raw_res = resp.json()["candidates"][0]["content"]["parts"][0]["text"]

            except httpx.HTTPError as e:
                self._fail_generation(model_id, provider, f"Cloud API request failed for '{provider}': {e}", cause=e)

            self.update_model_status(model_id, status="online", error=None, location="Cloud")
            return sanitize_message_content(raw_res)

        elif provider == "gguf_local":
            gguf_path = model_config.get("gguf_path") or model_config.get("model_name", "")
            mmproj_path = model_config.get("mmproj_path") or model_config.get("clip_model_path", "")
            # 2048 was far too tight for this app's actual job. An execution-phase prompt
            # (system prompt + task + spec + memory summary + the file under review) can run
            # 1000+ tokens on its own, leaving almost nothing to generate with - which is why
            # Coders emitted files that stopped mid-function and Critics ran out of room
            # before reaching their verdict.
            n_ctx = model_config.get("max_context_tokens") or DEFAULT_N_CTX

            # Raises ModelLoadError (a ModelInvocationError) if the model cannot be loaded.
            # Loading a multi-GB GGUF takes seconds to minutes and is pure blocking C code;
            # running it on the event loop froze every other request (including the UI's
            # status polling), which is what made a slow load look like a dead app.
            llm = await asyncio.to_thread(
                self.load_gguf_model, model_id, gguf_path, n_ctx, mmproj_path,
                None, model_config.get("chat_format")
            )

            # Reasoning-model control. llama-cpp-python cannot pass `enable_thinking=False`
            # through to the GGUF's jinja template, so the documented prompt-level switch is
            # the only lever available. Without it SmolLM3/Qwen3/Nemotron burn the entire
            # token budget on a <think> block and the sanitizer hands the room an empty turn.
            think_control = model_config.get("think_control")
            if think_control is None and model_config.get("no_think", True):
                arch = ""
                try:
                    arch = (getattr(llm, "metadata", {}) or {}).get("general.architecture", "")
                except Exception:
                    arch = ""
                think_control = NO_THINK_CONTROL_BY_ARCH.get(str(arch).lower())
            if think_control:
                full_messages = [dict(m) for m in full_messages]
                for m in full_messages:
                    if m.get("role") == "system":
                        m["content"] = f"{think_control}\n{m.get('content', '')}"
                        break
                else:
                    full_messages.insert(0, {"role": "system", "content": think_control})

            try:
                # Token counting & derived max_tokens: n_ctx - prompt_tokens with a floor
                prompt_tokens_est = 0
                for m in full_messages:
                    content_str = m.get("content", "")
                    try:
                        prompt_tokens_est += len(llm.tokenize(content_str.encode("utf-8")))
                    except Exception:
                        prompt_tokens_est += int(len(content_str.split()) * 1.3)

                # Reserve max_tokens from remaining context headroom. The old 1024 ceiling
                # truncated any generated source file longer than ~60 lines.
                gen_max_tokens = max(256, min(MAX_GENERATION_TOKENS, n_ctx - prompt_tokens_est - 32))

                start_time = time.time()
                kwargs = {
                    "messages": full_messages,
                    "temperature": temp,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": min_p,
                    "repeat_penalty": repeat_penalty,
                    "max_tokens": gen_max_tokens,
                    "stop": list(model_config.get("stop") or GGUF_STOP_STRINGS)
                }

                if response_schema:
                    try:
                        from llama_cpp.llama_grammar import LlamaGrammar
                        # Attempt to construct LlamaGrammar from schema if available
                        grammar = LlamaGrammar.from_json_schema(json.dumps(response_schema))
                        kwargs["grammar"] = grammar
                    except Exception as ge:
                        # Fallback to response_format dict if llama_cpp supports it natively
                        kwargs["response_format"] = {"type": "json_object", "schema": response_schema}
                        logger.debug("LlamaGrammar schema construction fallback for %s: %s", model_id, ge)

                # Same reasoning as the load above: llama.cpp generation is blocking C code.
                # Off the event loop it can be waited on with a timeout, so one wedged model
                # no longer stalls the room forever - the turn fails and rotation continues.
                self.model_last_used[model_id] = time.time()
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(llm.create_chat_completion, **kwargs),
                        timeout=GGUF_GENERATION_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    self._fail_generation(
                        model_id, provider,
                        f"GGUF generation exceeded {GGUF_GENERATION_TIMEOUT_SECONDS}s and was abandoned. "
                        f"The model is probably running on CPU because it does not fit in VRAM."
                    )
                elapsed = time.time() - start_time

                finish_reason = response["choices"][0].get("finish_reason", "")
                if finish_reason == "length":
                    logger.warning(
                        "Generation for %s hit the max_tokens limit (%d tokens, n_ctx=%d, prompt~%d) - "
                        "output is truncated",
                        model_id, gen_max_tokens, n_ctx, prompt_tokens_est
                    )
                elapsed = time.time() - start_time
                content = response["choices"][0]["message"]["content"]

                # Calculate speed (tok/sec)
                completion_tokens = response.get("usage", {}).get("completion_tokens", len(content.split()))
                tok_per_sec = round(completion_tokens / max(elapsed, 0.001), 1)

                self.update_model_status(
                    model_id,
                    status="online",
                    error=None,
                    tok_per_sec=tok_per_sec
                )
                return sanitize_message_content(content)
            except (KeyError, IndexError, TypeError, ValueError) as e:
                self._fail_generation(model_id, provider, f"Unexpected GGUF completion payload: {e}", cause=e)
            except Exception as e:
                self._fail_generation(model_id, provider, f"GGUF inference error: {e}", cause=e)

        else:
            self._fail_generation(
                model_id, provider,
                f"Unknown provider '{provider}'. Supported: ollama, gguf_local.",
                error_cls=ProviderNotConfiguredError
            )

    def _fail_generation(
        self,
        model_id: str,
        provider: str,
        message: str,
        cause: Optional[BaseException] = None,
        error_cls: type = ModelInvocationError
    ) -> None:
        """Records a generation failure on the status board and raises it instead of returning filler text."""
        logger.error("Generation failure for %s (%s): %s", model_id, provider, message)
        self.update_model_status(model_id, status="error", error=message)
        raise error_cls(message, model_id=model_id, provider=provider) from cause
