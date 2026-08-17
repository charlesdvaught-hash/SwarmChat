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
    api_key: Optional[str] = ""
    enabled: bool = True
    is_moderator: bool = False

class VoteOverrideReq(BaseModel):
    vote_id: str
    action: str
    modified_args: Optional[Dict[str, Any]] = None

class EvaluateReq(BaseModel):
    candidates: List[Dict[str, Any]]
    task_context: str

@app.get("/api/hardware")
def get_hardware_info():
    return model_mgr.get_hardware_info()

@app.get("/api/dependencies")
def get_dependencies():
    return {
        "ollama": model_mgr.check_ollama_status(),
        "ollama_models": model_mgr.list_ollama_models(),
        "memory_status": "healthy"
    }

@app.get("/api/state")
def get_full_state():
    return {
        "phase": memory_mgr.get_phase(),
        "turn_mode": orchestrator.turn_mode,
        "moderator_model_id": orchestrator.moderator_model_id,
        "models": orchestrator.models,
        "known_models": orchestrator.known_models,
        "pending_votes": orchestrator.pending_tool_votes,
        "chat_history": orchestrator.chat_history,
        "shared_memory": memory_mgr.state.get("shared_entries", []),
        "model_journals": memory_mgr.state.get("model_journals", {}),
        "tokens_used": memory_mgr.state.get("tokens_used", {}),
        "allowed_domains": tool_mgr.allowed_domains
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

@app.post("/api/votes/override")
def override_vote(req: VoteOverrideReq):
    res = orchestrator.admin_override_vote(req.vote_id, req.action, req.modified_args)
    return res

@app.post("/api/evaluate")
async def run_evaluate(req: EvaluateReq):
    res = await evaluate_engine.run_candidate_evaluation(req.candidates, req.task_context)
    return res

@app.get("/api/workspace/files")
def list_workspace_files(rel_dir: str = "."):
    return tool_mgr.list_files(rel_dir)

frontend_dist = os.path.join(os.path.dirname(__file__), "../frontend/dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
