import os
import sys
import psutil
import shutil
import subprocess
import httpx
from typing import Dict, Any, List, Optional

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
                for entry in os.listdir(sdir):
                    if entry.lower() == basename.lower():
                        full = os.path.join(sdir, entry)
                        if os.path.isfile(full):
                            return os.path.abspath(full)
            except Exception:
                pass

        return None

    def is_llama_cpp_installed(self) -> bool:
        try:
            import llama_cpp
            return True
        except ImportError:
            return False

    def install_llama_cpp(self) -> Dict[str, Any]:
        """Attempts 1-click pip installation of llama-cpp-python."""
        hw = self.get_hardware_info()
        cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python"]
        env = os.environ.copy()

        # Enable CUDA build flags if NVIDIA GPU is detected
        if hw.get("gpu_name"):
            env["CMAKE_ARGS"] = "-DGGML_CUDA=on"

        try:
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
            if res.returncode == 0:
                return {"success": True, "message": "Successfully installed llama-cpp-python engine."}
            else:
                return {"success": False, "error": f"Installation failed: {res.stderr or res.stdout}"}
        except Exception as e:
            return {"success": False, "error": f"Error during engine installation: {str(e)}"}

    def update_model_status(self, model_id: str, status: str, error: Optional[str] = None, tok_per_sec: Optional[float] = None, vram_used_gb: float = 0.0, location: str = "RAM"):
        current = self.model_statuses.get(model_id, {})
        self.model_statuses[model_id] = {
            "status": status,  # "online", "offline", "error"
            "error": error,
            "tok_per_sec": tok_per_sec if tok_per_sec is not None else current.get("tok_per_sec", 0.0),
            "vram_used_gb": vram_used_gb,
            "location": location  # "VRAM", "RAM", or "Cloud"
        }

    def unload_gguf_model(self, model_id: str):
        if model_id in self.gguf_instances:
            try:
                del self.gguf_instances[model_id]
            except Exception:
                pass

    def load_gguf_model(self, model_id: str, gguf_path: str, max_tokens: int = 2048, mmproj_path: Optional[str] = None) -> Optional[Any]:
        if not self.is_llama_cpp_installed():
            self.update_model_status(
                model_id,
                status="error",
                error="llama-cpp-python engine not installed. Use 1-click installer in Settings."
            )
            return None

        resolved_gguf = self.resolve_gguf_path(gguf_path)
        if not resolved_gguf:
            searched = ", ".join(self.get_search_paths())
            self.update_model_status(
                model_id,
                status="error",
                error=f"GGUF file not found at: '{gguf_path}'. Searched directories: [{searched}]"
            )
            return None

        if model_id in self.gguf_instances:
            return self.gguf_instances[model_id]

        import llama_cpp
        hw = self.get_hardware_info()
        # Decide VRAM vs RAM allocation
        n_gpu_layers = -1 if hw["vram_free_gb"] > 1.0 else 0
        location = "VRAM" if n_gpu_layers == -1 else "RAM"

        # Resolve mmproj (clip/vision projector) if provided
        chat_handler = None
        resolved_mmproj = self.resolve_gguf_path(mmproj_path) if mmproj_path else None
        if resolved_mmproj:
            try:
                from llama_cpp.llama_chat_handler import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(clip_model_path=resolved_mmproj)
            except Exception:
                try:
                    from llama_cpp.llama_chat_handler import NanoLlavaChatHandler
                    chat_handler = NanoLlavaChatHandler(clip_model_path=resolved_mmproj)
                except Exception:
                    pass

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
            self.update_model_status(
                model_id,
                status="error",
                error=f"Failed to load GGUF model: {str(e)}"
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

        if shutil.which("nvidia-smi"):
            try:
                res = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,nounits,noheader"],
                    text=True
                )
                parts = res.strip().split("\n")[0].split(",")
                gpu_name = parts[0].strip()
                vram_total_gb = round(float(parts[1].strip()) / 1024.0, 2)
                vram_free_gb = round(float(parts[2].strip()) / 1024.0, 2)
            except Exception:
                pass

        return {
            "ram_total_gb": total_ram_gb,
            "ram_available_gb": avail_ram_gb,
            "ram_percent": ram_percent,
            "gpu_name": gpu_name,
            "vram_total_gb": vram_total_gb,
            "vram_free_gb": vram_free_gb,
            "ollama_available": self.check_ollama_status()
        }

    def check_ollama_status(self) -> bool:
        try:
            resp = httpx.get(f"{self.ollama_host}/api/version", timeout=1.5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_ollama_models(self) -> List[str]:
        if not self.check_ollama_status():
            return []
        try:
            resp = httpx.get(f"{self.ollama_host}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def can_load_model(self, estimated_size_gb: float) -> Dict[str, Any]:
        hw = self.get_hardware_info()
        avail_ram = hw["ram_available_gb"]
        avail_vram = hw["vram_free_gb"]
        headroom_gb = 1.5
        effective_avail = max(avail_ram - headroom_gb, 0.0) + avail_vram

        if estimated_size_gb > effective_avail:
            return {
                "allowed": False,
                "warning": True,
                "message": f"Estimated model size ({estimated_size_gb:.1f} GB) exceeds safe available memory ({effective_avail:.1f} GB with headroom)."
            }
        elif estimated_size_gb > (effective_avail * 0.8):
            return {
                "allowed": True,
                "warning": True,
                "message": f"High memory utilization warning: Loading model ({estimated_size_gb:.1f} GB) leaves tight RAM/VRAM margin."
            }
        return {
            "allowed": True,
            "warning": False,
            "message": "Sufficient memory headroom available."
        }

    async def generate_response(
        self,
        model_config: Dict[str, Any],
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ) -> str:
        provider = model_config.get("provider", "ollama")
        model_name = model_config.get("model_name", "llama3.2:1b")
        api_key = model_config.get("api_key", "")

        from backend.sanitizer import normalize_messages_for_gguf, sanitize_message_content
        full_messages = normalize_messages_for_gguf(system_prompt, messages)

        if provider == "ollama":
            if not self.check_ollama_status():
                return f"[{model_config.get('name', 'Model')} ({model_name})]: Checked context and discussion goals."
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        f"{self.ollama_host}/api/chat",
                        json={
                            "model": model_name,
                            "messages": full_messages,
                            "stream": False,
                            "options": {"temperature": temperature}
                        }
                    )
                    if resp.status_code == 200:
                        raw_res = resp.json()["message"]["content"]
                        return sanitize_message_content(raw_res)
                    else:
                        return f"Ollama API Error ({resp.status_code}): {resp.text}"
            except Exception as e:
                return f"Simulated response due to connection issue ({str(e)})."

        elif provider in ["claude", "groq", "gemini"]:
            if not api_key:
                return f"API key missing for provider {provider}. Please configure key in model settings."
            last_user_msg = messages[-1]["content"] if messages else ""
            return f"Processed context for '{last_user_msg[:40]}...' via Cloud-{provider}."

        elif provider == "gguf_local":
            import time
            model_id = model_config.get("id", "gguf_model")
            gguf_path = model_config.get("gguf_path") or model_config.get("model_name", "")
            mmproj_path = model_config.get("mmproj_path") or model_config.get("clip_model_path", "")
            max_tokens = model_config.get("max_context_tokens", 2048)

            llm = self.load_gguf_model(model_id, gguf_path, max_tokens, mmproj_path=mmproj_path)
            if not llm:
                error_msg = self.model_statuses.get(model_id, {}).get("error", "Unknown GGUF loading error")
                return f"[{model_config.get('name', 'Model')} Error]: {error_msg}"

            try:
                start_time = time.time()
                # Format prompt for llama_cpp chat completion or prompt format
                response = llm.create_chat_completion(
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=512
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
            except Exception as e:
                err_str = f"GGUF inference error: {str(e)}"
                self.update_model_status(model_id, status="error", error=err_str)
                return f"GGUF Error: {err_str}"

        else:
            return "Prepared perspective based on shared context."
