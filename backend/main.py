import os
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from backend.models import ModelManager
from backend.memory import MemoryManager
from backend.tools import ToolManager
from backend.orchestrator import Orchestrator
from backend.evaluate import EvaluateEngine

app = FastAPI(title="SwarmChat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_mgr = ModelManager()
memory_mgr = MemoryManager()
tool_mgr = ToolManager()
orchestrator = Orchestrator(model_mgr, memory_mgr, tool_mgr)
evaluate_engine = EvaluateEngine(model_mgr)

class PhaseSwitchReq(BaseModel):
    phase: str

class ChatMsgReq(BaseModel):
    sender: str = "Admin"
    content: str
    is_admin: bool = True

class ModelConfigReq(BaseModel):
    id: str
    name: str
    role: str
    provider: str = "ollama"
    model_name: str = "llama3.2:1b"
    gguf_path: Optional[str] = ""
    mmproj_path: Optional[str] = ""
    api_key: Optional[str] = ""
    enabled: bool = True
    is_moderator: bool = False

class ValidatePathReq(BaseModel):
    path: str
    mmproj_path: Optional[str] = None

class SearchPathReq(BaseModel):
    path: str

class VoteOverrideReq(BaseModel):
    vote_id: str
    action: str
    modified_args: Optional[Dict[str, Any]] = None

class EvaluateReq(BaseModel):
    candidates: List[Dict[str, Any]]
    task_context: str

class ItineraryTaskReq(BaseModel):
    title: str
    description: str
    priority: str = "medium"
    assigned_model: Optional[str] = None

class ItineraryTaskUpdateReq(BaseModel):
    task_id: str
    status: Optional[str] = None
    priority: Optional[str] = None
    assigned_model: Optional[str] = None

@app.get("/api/fs/browse")
def browse_filesystem(path: Optional[str] = None):
    target = path.strip() if path and path.strip() else os.path.expanduser("~")
    if not os.path.exists(target):
        target = os.path.abspath(".")

    abs_target = os.path.abspath(target)
    if not os.path.isdir(abs_target):
        abs_target = os.path.dirname(abs_target)

    parent_path = os.path.dirname(abs_target) if abs_target != os.path.dirname(abs_target) else None

    directories = []
    files = []

    try:
        for entry in os.listdir(abs_target):
            full_p = os.path.join(abs_target, entry)
            if os.path.isdir(full_p):
                directories.append({
                    "name": entry,
                    "path": full_p
                })
            elif os.path.isfile(full_p):
                lower = entry.lower()
                is_gguf = lower.endswith(".gguf") or lower.endswith(".bin")
                is_mmproj = "mmproj" in lower or "clip" in lower
                try:
                    size_mb = round(os.path.getsize(full_p) / (1024 * 1024), 2)
                except Exception:
                    size_mb = 0.0

                files.append({
                    "name": entry,
                    "path": full_p,
                    "size_mb": size_mb,
                    "is_gguf": is_gguf,
                    "is_mmproj": is_mmproj
                })
    except Exception as e:
        return {"success": False, "error": str(e), "current_path": abs_target}

    directories.sort(key=lambda x: x["name"].lower())
    files.sort(key=lambda x: x["name"].lower())

    return {
        "success": True,
        "current_path": abs_target,
        "parent_path": parent_path,
        "directories": directories,
        "files": files
    }

@app.post("/api/fs/validate")
def validate_model_path(req: ValidatePathReq):
    resolved_gguf = model_mgr.resolve_gguf_path(req.path)
    resolved_mmproj = model_mgr.resolve_gguf_path(req.mmproj_path) if req.mmproj_path else None

    valid = resolved_gguf is not None
    size_gb = 0.0
    if resolved_gguf and os.path.exists(resolved_gguf):
        size_gb = round(os.path.getsize(resolved_gguf) / (1024 ** 3), 2)

    msg = f"Path valid. Absolute location: '{resolved_gguf}' ({size_gb} GB)" if valid else f"File not found in path or search directories. Path: '{req.path}'"
    if req.mmproj_path:
        mm_valid = resolved_mmproj is not None
        msg += f" | mmproj: {'Valid (' + str(resolved_mmproj) + ')' if mm_valid else 'Not found (' + req.mmproj_path + ')'}"

    return {
        "valid": valid,
        "resolved_path": resolved_gguf,
        "file_size_gb": size_gb,
        "mmproj_valid": resolved_mmproj is not None if req.mmproj_path else True,
        "resolved_mmproj_path": resolved_mmproj,
        "message": msg
    }

@app.get("/api/models/search_paths")
def get_search_paths():
    return {
        "search_paths": model_mgr.get_search_paths(),
        "custom_paths": model_mgr.custom_search_paths
    }

@app.post("/api/models/search_paths")
def add_search_path(req: SearchPathReq):
    if req.path:
        model_mgr.add_search_path(req.path)
    return {
        "success": True,
        "search_paths": model_mgr.get_search_paths(),
        "custom_paths": model_mgr.custom_search_paths
    }

@app.get("/api/hardware")
def get_hardware_info():
    return model_mgr.get_hardware_info()

@app.get("/api/dependencies")
def get_dependencies():
    return {
        "ollama": model_mgr.check_ollama_status(),
        "ollama_models": model_mgr.list_ollama_models(),
        "llama_cpp_installed": model_mgr.is_llama_cpp_installed(),
        "memory_status": "healthy"
    }

@app.post("/api/engine/install")
def install_engine():
    res = model_mgr.install_llama_cpp()
    return res

@app.get("/api/state")
def get_full_state():
    return {
        "phase": memory_mgr.get_phase(),
        "turn_mode": orchestrator.turn_mode,
        "moderator_model_id": orchestrator.moderator_model_id,
        "models": orchestrator.models,
        "known_models": orchestrator.known_models,
        "model_statuses": model_mgr.model_statuses,
        "pending_votes": orchestrator.pending_tool_votes,
        "chat_history": orchestrator.chat_history,
        "shared_memory": memory_mgr.state.get("shared_entries", []),
        "model_journals": memory_mgr.state.get("model_journals", {}),
        "tokens_used": memory_mgr.state.get("tokens_used", {}),
        "allowed_domains": tool_mgr.allowed_domains,
        "episodes": memory_mgr.state.get("episodes", []),
        "task_itinerary": memory_mgr.get_task_itinerary(),
        "active_task": memory_mgr.get_active_task(),
        "file_audit_log": memory_mgr.get_file_audit_log(),
        "active_file_locks": memory_mgr.state.get("active_file_locks", {})
    }

@app.post("/api/itinerary/task")
def add_itinerary_task(req: ItineraryTaskReq):
    task = memory_mgr.add_itinerary_task(req.title, req.description, req.priority, req.assigned_model)
    return {"success": True, "task": task}

@app.post("/api/itinerary/update")
def update_itinerary_task(req: ItineraryTaskUpdateReq):
    updates = {}
    if req.status is not None:
        updates["status"] = req.status
    if req.priority is not None:
        updates["priority"] = req.priority
    if req.assigned_model is not None:
        updates["assigned_model"] = req.assigned_model
    updated = memory_mgr.update_itinerary_task(req.task_id, updates)
    return {"success": updated is not None, "task": updated}

@app.get("/api/workspace/file")
def get_workspace_file_content(filepath: str):
    return tool_mgr.read_file(filepath)

@app.get("/api/workspace/audit")
def get_workspace_file_audit(filepath: Optional[str] = None):
    return {
        "success": True,
        "audit_log": memory_mgr.get_file_audit_log(filepath),
        "active_file_locks": memory_mgr.state.get("active_file_locks", {})
    }

@app.post("/api/phase")
def set_phase(req: PhaseSwitchReq):
    new_p = memory_mgr.set_phase(req.phase)
    return {"success": True, "phase": new_p}

@app.post("/api/chat/message")
async def send_chat_message(req: ChatMsgReq):
    msg = orchestrator.add_chat_message(
        sender=req.sender,
        role="Admin" if req.is_admin else "User",
        content=req.content,
        is_admin=req.is_admin
    )
    # Automatically launch conversation loop in background
    asyncio.create_task(orchestrator.run_autonomous_loop(max_turns=5))
    return {"success": True, "message": msg}

@app.post("/api/chat/step")
async def step_turn(model_id: Optional[str] = None):
    target_id = model_id or orchestrator.get_next_speaker()
    if not target_id:
        return {"success": False, "message": "No active model available for turn."}
    msg = await orchestrator.step_model_turn(target_id)
    return {"success": True, "speaker_id": target_id, "message": msg}

@app.post("/api/models/configure")
def configure_model(req: ModelConfigReq):
    m_dict = req.dict()
    orchestrator.add_or_update_known_model(m_dict)
    return {"success": True, "models": orchestrator.models, "known_models": orchestrator.known_models}

@app.post("/api/models/kick")
def kick_model(model_id: str):
    res = orchestrator.kick_model_from_room(model_id)
    return res

@app.post("/api/models/readd")
def readd_model(model_id: str):
    res = orchestrator.readd_model_to_room(model_id)
    return res

@app.post("/api/models/set_moderator")
def set_moderator(model_id: str):
    orchestrator.set_moderator(model_id)
    return {"success": True, "moderator_model_id": orchestrator.moderator_model_id}

@app.get("/api/tools/search_hf")
async def search_huggingface(query: str, limit: int = 5):
    res = await tool_mgr.search_huggingface(query, limit=limit)
    return res

@app.get("/api/tools/internet_search")
async def internet_search(query: str, domain_filter: Optional[str] = None):
    res = await tool_mgr.internet_search(query, domain_filter=domain_filter)
    return res

@app.post("/api/votes/override")
def override_vote(req: VoteOverrideReq):
    res = orchestrator.admin_override_vote(req.vote_id, req.action, req.modified_args)
    return res

@app.post("/api/evaluate")
async def run_evaluate(req: EvaluateReq):
    res = await evaluate_engine.run_candidate_evaluation(req.candidates, req.task_context)
    return res

@app.get("/api/evaluate/health")
def get_room_health():
    return evaluate_engine.evaluate_room_health(
        active_models=orchestrator.models,
        chat_history=orchestrator.chat_history,
        tokens_used=memory_mgr.state.get("tokens_used", {})
    )

@app.get("/api/workspace/files")
def list_workspace_files(rel_dir: str = "."):
    return tool_mgr.list_files(rel_dir)

frontend_dist = os.path.join(os.path.dirname(__file__), "../frontend/dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
