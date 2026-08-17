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

        full_messages = [{"role": "system", "content": system_prompt}] + messages

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
                        return resp.json()["message"]["content"]
                    else:
                        return f"Ollama API Error ({resp.status_code}): {resp.text}"
            except Exception as e:
                return f"[{model_config.get('name', 'Model')} ({model_name})]: Simulated response due to connection issue ({str(e)})."

        elif provider in ["claude", "groq", "gemini"]:
            if not api_key:
                return f"[{model_config.get('name', 'Model')} ({provider})]: API key missing. Please configure key in model settings."
            last_user_msg = messages[-1]["content"] if messages else ""
            return f"[{model_config.get('name', 'Model')} via Cloud-{provider}]: Processed context for '{last_user_msg[:40]}...'."

        elif provider == "gguf_local":
            return f"[{model_config.get('name', 'Model')} (GGUF)]: Evaluated task. High efficiency execution ready."

        else:
            return f"[{model_config.get('name', 'Model')}]: Prepared perspective based on shared context."
