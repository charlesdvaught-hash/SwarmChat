import logging
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

class ModelManager:
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.model_statuses: Dict[str, Dict[str, Any]] = {}
        self.gguf_instances: Dict[str, Any] = {}
        self.custom_search_paths: List[str] = []

    def add_search_path(self, path: str):
        if path and path not in self.custom_search_paths:
            self.custom_search_paths.append(os.path.abspath(path))

    def get_search_paths(self) -> List[str]:
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
        return [p for p in paths if os.path.exists(p)]

    def resolve_gguf_path(self, raw_path: Optional[str]) -> Optional[str]:
        if not raw_path or not raw_path.strip():
            return None

        raw_path = raw_path.strip()

        # 1. Direct path check
        if os.path.exists(raw_path) and os.path.isfile(raw_path):
            return os.path.abspath(raw_path)

        basename = os.path.basename(raw_path)
        search_dirs = self.get_search_paths()

        # 2. Search in search_dirs for direct join or basename join
        for sdir in search_dirs:
            candidate1 = os.path.join(sdir, raw_path)
            if os.path.exists(candidate1) and os.path.isfile(candidate1):
                return os.path.abspath(candidate1)

            candidate2 = os.path.join(sdir, basename)
            if os.path.exists(candidate2) and os.path.isfile(candidate2):
                return os.path.abspath(candidate2)

        # 3. Case-insensitive basename search in search_dirs
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

    def is_llama_cpp_installed(self) -> bool:
        try:
            import llama_cpp
            return True
        except ImportError:
            return False

    def install_llama_cpp(self, use_cuda_wheels: bool = True) -> Dict[str, Any]:
        """Attempts 1-click installation of llama-cpp-python, preferring official pre-built CUDA wheels if GPU is present."""
        hw = self.get_hardware_info()
        env = os.environ.copy()

        # If NVIDIA GPU is detected and pre-built CUDA wheel installation requested
        if hw.get("gpu_name") and use_cuda_wheels:
            # Install pre-compiled wheel index for CUDA support
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
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
            if res.returncode == 0:
                return {"success": True, "message": "Successfully installed llama-cpp-python engine (CUDA wheel / build)." }
            else:
                # Fallback to standard pip install if wheel url failed
                fallback_cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python"]
                res2 = subprocess.run(fallback_cmd, env=env, capture_output=True, text=True, timeout=180)
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
        """Drops a loaded GGUF instance. Returns True when a model was actually unloaded."""
        return self.gguf_instances.pop(model_id, None) is not None

    def load_gguf_model(self, model_id: str, gguf_path: str, max_tokens: int = 2048, mmproj_path: Optional[str] = None, force_device: Optional[str] = None) -> Optional[Any]:
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
            return self.gguf_instances[model_id]

        import llama_cpp
        hw = self.get_hardware_info()
        # Assume VRAM preferred by default if GPU present
        has_gpu = bool(hw.get("gpu_name") or hw.get("vram_total_gb", 0) > 0)
        if force_device == "gpu":
            n_gpu_layers = -1 if has_gpu else 0
            location = "VRAM" if has_gpu else "RAM"
        elif force_device == "cpu":
            n_gpu_layers = 0
            location = "RAM"
        else:
            # Prefer VRAM if free VRAM exists and GPU is detected
            if has_gpu and hw.get("vram_free_gb", 0) > 0.5:
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

            llm = llama_cpp.Llama(**kwargs)
            self.gguf_instances[model_id] = llm
            file_size_gb = round(os.path.getsize(resolved_gguf) / (1024 ** 3), 2)
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
            n_ctx = model_config.get("max_context_tokens", 2048)

            # Raises ModelLoadError (a ModelInvocationError) if the model cannot be loaded.
            llm = self.load_gguf_model(model_id, gguf_path, n_ctx, mmproj_path=mmproj_path)

            try:
                # Token counting & derived max_tokens: n_ctx - prompt_tokens with a floor
                prompt_tokens_est = 0
                for m in full_messages:
                    content_str = m.get("content", "")
                    try:
                        prompt_tokens_est += len(llm.tokenize(content_str.encode("utf-8")))
                    except Exception:
                        prompt_tokens_est += int(len(content_str.split()) * 1.3)

                # Reserve max_tokens based on available context headroom, minimum 256, max 1024
                gen_max_tokens = max(256, min(1024, n_ctx - prompt_tokens_est - 32))

                start_time = time.time()
                kwargs = {
                    "messages": full_messages,
                    "temperature": temp,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": min_p,
                    "repeat_penalty": repeat_penalty,
                    "max_tokens": gen_max_tokens
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

                response = llm.create_chat_completion(**kwargs)
                elapsed = time.time() - start_time

                finish_reason = response["choices"][0].get("finish_reason", "")
                if finish_reason == "length":
                    logger.warning("Generation for %s reached max_tokens length limit (%d tokens)", model_id, gen_max_tokens)
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
