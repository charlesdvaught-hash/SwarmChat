import asyncio
import json
import logging
import os
import shutil
import subprocess

# Must be set BEFORE llama_cpp (imported via backend.models) initialises the CUDA backend.
# This app loads and unloads GGUF models repeatedly while running (VRAM lifecycle
# management), and llama.cpp's CUDA virtual-memory pool asserts and hard-aborts the whole
# process when buffers are released out of the order it expects:
#   ggml-cuda.cu:680: GGML_ASSERT(ptr == (void *)((char *)pool_addr + pool_used)) failed
# That abort kills the server mid-turn with no Python traceback, which is what the
# long-standing "auto mode just goes silent" symptom actually was. Disabling the VMM pool
# uses plain cudaMalloc instead - marginally slower allocation, no reordering assert.
# Set SWARMCHAT_CUDA_VMM=1 to opt back in.
if os.environ.get("SWARMCHAT_CUDA_VMM") != "1":
    os.environ.setdefault("GGML_CUDA_NO_VMM", "1")

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from backend.errors import (
    MemoryPersistenceError,
    ModelInvocationError,
    SwarmChatError,
)
from backend.models import ModelManager
from backend.memory import MemoryManager
from backend.tools import ToolManager
from backend.orchestrator import Orchestrator
from backend.evaluate import EvaluateEngine

# The application configured no logging handlers at all, so every logger.info /
# logger.warning in the backend (turn failures, empty generations, VRAM offloads,
# refinement decisions, dropped directives) was silently discarded - uvicorn only
# configures its own loggers. That made every failure mode in this app invisible
# from the terminal AND the UI at the same time. Configure the root logger once,
# at import, honouring SWARMCHAT_LOG_LEVEL for quick debugging.
logging.basicConfig(
    level=os.environ.get("SWARMCHAT_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="SwarmChat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ACTIVE_PROJECT_FILE = os.path.join(".swarmchat", "active_project.json")


def _load_active_project_id() -> str:
    """The project the user was last working in. Without this a restart silently
    dropped them back into default_project while their work sat in another one."""
    try:
        with open(ACTIVE_PROJECT_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("project_id") or "default_project"
    except (OSError, ValueError):
        return "default_project"


def _save_active_project_id(project_id: str):
    try:
        os.makedirs(os.path.dirname(ACTIVE_PROJECT_FILE), exist_ok=True)
        with open(ACTIVE_PROJECT_FILE, "w", encoding="utf-8") as f:
            json.dump({"project_id": project_id}, f)
    except OSError as e:
        logger.warning("Could not persist active project: %s", e)


_active_project = _load_active_project_id()

model_mgr = ModelManager()
memory_mgr = MemoryManager(project_id=_active_project)
tool_mgr = ToolManager(project_id=_active_project)
orchestrator = Orchestrator(model_mgr, memory_mgr, tool_mgr)
evaluate_engine = EvaluateEngine(model_mgr)

# Startup tidy: files older than a week and workspaces belonging to models that are
# no longer in the room get moved to .swarmchat/trash so old broken output does not
# accumulate in the sandboxes the swarm reads from.
try:
    tool_mgr.clean_workspaces(
        active_bot_ids=orchestrator.get_active_model_ids(),
        max_age_days=float(os.environ.get("SWARMCHAT_WORKSPACE_MAX_AGE_DAYS", "7")),
        # Automatic runs only sweep orphans that are ALSO stale - a workspace touched
        # today belongs to work in progress even if its model id changed.
        orphans_must_be_stale=True
    )
except OSError as e:
    logger.warning("Startup workspace clean failed: %s", e)

# --- FRONTEND FRESHNESS ---
# The UI is served from frontend/dist, which is a build artifact. Editing
# frontend/src changes nothing until someone runs the build, and a stale bundle
# looks exactly like a feature that "didn't work" - so the app checks for itself.
FRONTEND_DIR = "frontend"
FRONTEND_SRC_DIR = os.path.join(FRONTEND_DIR, "src")
FRONTEND_DIST_DIR = os.path.join(FRONTEND_DIR, "dist")
FRONTEND_WATCHED_FILES = ("index.html", "vite.config.ts", "tailwind.config.js", "postcss.config.js", "package.json")

_frontend_rebuild_log: List[str] = []


def _newest_source_mtime() -> float:
    """Most recent edit across the frontend sources that feed the bundle."""
    newest = 0.0
    for dirpath, dirnames, filenames in os.walk(FRONTEND_SRC_DIR):
        dirnames[:] = [d for d in dirnames if d != "node_modules"]
        for fname in filenames:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(dirpath, fname)))
            except OSError:
                continue
    for fname in FRONTEND_WATCHED_FILES:
        try:
            newest = max(newest, os.path.getmtime(os.path.join(FRONTEND_DIR, fname)))
        except OSError:
            continue
    return newest


def _bundle_mtime() -> float:
    """Newest built asset. Only the assets the current index.html references matter,
    but taking the newest built file is close enough and far cheaper."""
    newest = 0.0
    for dirpath, dirnames, filenames in os.walk(FRONTEND_DIST_DIR):
        # Superseded bundles are parked here; they must not make dist look fresh.
        dirnames[:] = [d for d in dirnames if d != "_old_bundles"]
        for fname in filenames:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(dirpath, fname)))
            except OSError:
                continue
    return newest


def _npm_executable() -> Optional[str]:
    """npm is npm.cmd on Windows; shutil.which finds neither reliably without both."""
    for candidate in ("npm.cmd", "npm"):
        found = shutil.which(candidate)
        if found:
            return found
    # A backend started before Node was installed inherits a PATH without it, so
    # fall back to the standard install locations rather than reporting "no Node".
    for fallback in (
        r"C:\Program Files\nodejs\npm.cmd",
        r"C:\Program Files (x86)\nodejs\npm.cmd",
        os.path.expanduser(r"~\AppData\Roaming\npm\npm.cmd"),
        "/usr/local/bin/npm",
        "/usr/bin/npm",
    ):
        if os.path.exists(fallback):
            return fallback
    return None


def get_frontend_status() -> Dict[str, Any]:
    src = _newest_source_mtime()
    built = _bundle_mtime()
    npm = _npm_executable()
    return {
        "dist_exists": os.path.isdir(FRONTEND_DIST_DIR) and built > 0,
        "source_mtime": src,
        "bundle_mtime": built,
        # A one second slack absorbs filesystem timestamp granularity.
        "is_stale": bool(src and built and src > built + 1),
        "node_available": npm is not None,
        "deps_installed": os.path.isdir(os.path.join(FRONTEND_DIR, "node_modules")),
        "last_rebuild_log": _frontend_rebuild_log[-40:]
    }


def _run_npm(args: List[str], timeout: int) -> Dict[str, Any]:
    npm = _npm_executable()
    if not npm:
        return {"success": False, "error": "npm was not found on PATH. Install Node.js to let the app rebuild its own UI."}
    # Package postinstall scripts (esbuild's, for one) shell out to plain `node`, so
    # node's directory has to be on PATH for the child - a backend started before Node
    # was installed would otherwise fail deep inside npm install with "'node' is not
    # recognized".
    env = dict(os.environ)
    node_dir = os.path.dirname(npm)
    if node_dir and node_dir not in env.get("PATH", ""):
        env["PATH"] = node_dir + os.pathsep + env.get("PATH", "")
    try:
        proc = subprocess.run(
            [npm, *args],
            cwd=os.path.abspath(FRONTEND_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            env=env
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"success": False, "error": f"npm {' '.join(args)} failed: {e}"}
    output = (proc.stdout or "") + (proc.stderr or "")
    return {"success": proc.returncode == 0, "returncode": proc.returncode, "output": output.splitlines()[-40:]}


# Background loop tasks are retained so their failures are logged instead of being
# discarded when the task object is garbage collected.
background_tasks: set = set()
last_background_error: Optional[str] = None


@app.exception_handler(SwarmChatError)
async def swarmchat_error_handler(request: Request, exc: SwarmChatError):
    """Turns backend failures into real error responses instead of 200s that look successful."""
    logger.error("%s %s failed: %s", request.method, request.url.path, exc)
    status = 503 if isinstance(exc, ModelInvocationError) else 500
    return JSONResponse(status_code=status, content={"success": False, "error": str(exc)})


def _track_background_task(task: asyncio.Task, description: str):
    background_tasks.add(task)

    def _on_done(finished: asyncio.Task):
        background_tasks.discard(finished)
        if finished.cancelled():
            return
        exc = finished.exception()
        if exc is not None:
            global last_background_error
            last_background_error = f"{description}: {exc}"
            logger.error("Background task failed (%s)", last_background_error, exc_info=exc)

    task.add_done_callback(_on_done)

class PhaseSwitchReq(BaseModel):
    phase: str

class ChatMsgReq(BaseModel):
    sender: str = "Admin"
    content: str
    is_admin: bool = True

class PromptTemplateUpdateReq(BaseModel):
    start_prompt: Optional[str] = None
    execution_prompt: Optional[str] = None

class ModelConfigReq(BaseModel):
    id: str
    name: str
    role: str
    provider: str = "ollama"
    model_name: str = "llama3.2:1b"
    gguf_path: Optional[str] = ""
    mmproj_path: Optional[str] = ""
    # Optional heavier GGUF model swapped in only for the Architect's one-off
    # escalation turn (task failed 3x in execution phase) - see orchestrator.step_model_turn.
    escalation_model_path: Optional[str] = ""
    # Was never settable through the API, so every model added from the UI silently fell
    # back to a tiny context and produced truncated turns.
    max_context_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    api_key: Optional[str] = ""
    enabled: bool = True
    # Accepted so an older UI bundle still validates, then discarded by the orchestrator:
    # the supervisor seat is the Architect role, not a separate flag.
    is_moderator: bool = False
    custom_start_prompt: Optional[str] = None
    custom_execution_prompt: Optional[str] = None

class ValidatePathReq(BaseModel):
    path: str
    mmproj_path: Optional[str] = None

class SearchPathReq(BaseModel):
    path: str

class VoteOverrideReq(BaseModel):
    vote_id: str
    action: str
    modified_args: Optional[Dict[str, Any]] = None

class PlanQuestionAnswerReq(BaseModel):
    question_id: str
    answer: str

class HireVoteReq(BaseModel):
    model_id: str
    model_name: str
    gguf_url_or_tag: str
    votes_for: List[str]
    notes: Optional[str] = ""

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

class RosterUpdateReq(BaseModel):
    schedule: List[str]

class ItineraryTaskDeleteReq(BaseModel):
    task_id: str
    # Files the task produced are moved to .swarmchat/trash/ by default, not
    # hard-deleted, so a mistaken delete is recoverable.
    trash_artifacts: bool = True

class ItineraryTaskMoveReq(BaseModel):
    task_id: str
    target_project_id: str

class ProjectReq(BaseModel):
    project_id: str

class WorkspaceCleanReq(BaseModel):
    max_age_days: float = 7.0
    prune_orphans: bool = True

@app.get("/api/prompts/templates")
def get_prompt_templates():
    from backend.prompts import prompt_template_mgr
    return prompt_template_mgr.templates

@app.post("/api/prompts/templates")
def update_prompt_templates(req: PromptTemplateUpdateReq):
    from backend.prompts import prompt_template_mgr
    try:
        prompt_template_mgr.update_templates(
            start_prompt=req.start_prompt,
            execution_prompt=req.execution_prompt
        )
    except MemoryPersistenceError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"success": True, "templates": prompt_template_mgr.templates}

last_loaded_model_dir: Optional[str] = None

@app.get("/api/fs/browse")
def browse_filesystem(path: Optional[str] = None):
    global last_loaded_model_dir
    target = path.strip() if path and path.strip() else (last_loaded_model_dir or os.path.expanduser("~"))
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

                # In Server Filesystem Explorer, show ONLY model files (GGUF/bin/mmproj) and folders
                if is_gguf or is_mmproj:
                    try:
                        size_mb = round(os.path.getsize(full_p) / (1024 * 1024), 2)
                    except OSError as e:
                        logger.warning("Could not stat %s: %s", full_p, e)
                        size_mb = 0.0

                    files.append({
                        "name": entry,
                        "path": full_p,
                        "size_mb": size_mb,
                        "is_gguf": is_gguf,
                        "is_mmproj": is_mmproj
                    })
    except OSError as e:
        logger.warning("Could not browse %s: %s", abs_target, e)
        raise HTTPException(status_code=400, detail=f"Cannot list '{abs_target}': {e}") from e

    directories.sort(key=lambda x: x["name"].lower())
    files.sort(key=lambda x: x["name"].lower())

    return {
        "success": True,
        "current_path": abs_target,
        "parent_path": parent_path,
        "directories": directories,
        "files": files,
        "last_loaded_dir": last_loaded_model_dir
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
    if not req.path or not req.path.strip():
        raise HTTPException(status_code=400, detail="A non-empty search path is required.")
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
        "memory_status": "degraded" if memory_mgr.last_load_error else "healthy",
        "memory_error": memory_mgr.last_load_error,
        "last_background_error": last_background_error
    }

@app.post("/api/engine/install")
def install_engine():
    res = model_mgr.install_llama_cpp()
    if not res.get("success"):
        raise HTTPException(status_code=500, detail=res.get("error", "llama-cpp-python installation failed."))
    return res

@app.get("/api/state")
def get_full_state():
    return {
        "phase": memory_mgr.get_phase(),
        # Where the room is inside the pre-execution planning gate, and who owes the next
        # move. Without this the UI showed "DISCUSSION" for the whole planning sequence and
        # gave no way to tell "waiting on the Critic" from "stuck".
        **orchestrator.plan_gate_status(),
        "turn_mode": orchestrator.turn_mode,
        "moderator_model_id": orchestrator.moderator_model_id,
        "models": orchestrator.models,
        "known_models": orchestrator.known_models,
        "model_statuses": model_mgr.model_statuses,
        "pending_votes": orchestrator.pending_tool_votes,
        # The planning question board. A question routed to the Admin is PARKED, not
        # blocking - the room carries on - so the UI has to surface it independently of
        # whose turn it is, the same way pending tool votes are surfaced.
        "plan_questions": orchestrator.plan_questions(),
        "chat_history": orchestrator.chat_history,
        # What the UI labels "up next". During execution the roster queue is not what drives
        # turns (the task router is), so showing the raw queue there was fiction.
        "turn_schedule": orchestrator.upcoming_speakers(8),
        "roster_queue": orchestrator.turn_schedule,
        "shared_memory": memory_mgr.state.get("shared_entries", []),
        "model_journals": memory_mgr.state.get("model_journals", {}),
        "tokens_used": memory_mgr.state.get("tokens_used", {}),
        "allowed_domains": tool_mgr.allowed_domains,
        "episodes": memory_mgr.state.get("episodes", []),
        "task_itinerary": memory_mgr.get_task_itinerary(),
        "active_task": memory_mgr.get_active_task(),
        "file_audit_log": memory_mgr.get_file_audit_log(),
        "active_file_locks": memory_mgr.state.get("active_file_locks", {}),
        "project_id": memory_mgr.get_project_id(),
        "projects": memory_mgr.list_projects(),
        "frontend_status": get_frontend_status(),
        "memory_error": memory_mgr.last_load_error,
        "last_background_error": last_background_error,
        # Whether a server-side conversation loop is mid-flight. The Auto toggle needs
        # this: sending a chat message already starts a loop, and the UI must not drive
        # `/api/chat/step` on top of one that is already taking turns.
        "loop_active": orchestrator.loop_active,
        # What is ACTUALLY resident, as opposed to what is on the roster. These two lists
        # diverging is the symptom of a leak, and until now nothing surfaced it.
        "loaded_model_ids": list(model_mgr.gguf_instances.keys()),
        "hardware": model_mgr.get_hardware_info()
    }

@app.post("/api/roster/update")
def update_roster(req: RosterUpdateReq):
    orchestrator.turn_schedule = req.schedule
    return {"success": True, "turn_schedule": orchestrator.turn_schedule}

@app.post("/api/roster/refresh")
def refresh_roster():
    sched = orchestrator.generate_turn_schedule()
    return {"success": True, "turn_schedule": sched}

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
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Itinerary task '{req.task_id}' not found.")
    return {"success": True, "task": updated}

@app.post("/api/itinerary/delete")
def delete_itinerary_task(req: ItineraryTaskDeleteReq):
    """Deletes a task outright and, by default, sweeps the files it produced out of
    the bot workspaces into trash so abandoned work stops polluting later turns."""
    removed = memory_mgr.delete_itinerary_task(req.task_id)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"Itinerary task '{req.task_id}' not found.")
    trashed: List[str] = []
    if req.trash_artifacts and removed.get("filename"):
        author = removed.get("author_bot_id")
        trashed = tool_mgr.trash_task_artifacts(
            [removed["filename"]],
            bot_ids=[author] if author else None
        )
    # A deleted task must not stay pinned, or the scheduler keeps trying to work it.
    if orchestrator.pinned_task_id == req.task_id:
        orchestrator.pinned_task_id = None
    return {"success": True, "deleted_task": removed, "trashed_files": trashed}

@app.post("/api/itinerary/move")
def move_itinerary_task(req: ItineraryTaskMoveReq):
    """Sorts a task into another project's itinerary."""
    moved = memory_mgr.move_task_to_project(req.task_id, req.target_project_id)
    if moved is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not move task '{req.task_id}' to project '{req.target_project_id}'."
        )
    if orchestrator.pinned_task_id == req.task_id:
        orchestrator.pinned_task_id = None
    return {"success": True, "task": moved, "target_project_id": req.target_project_id}

@app.get("/api/projects")
def list_projects():
    return {
        "success": True,
        "active_project_id": memory_mgr.get_project_id(),
        "projects": memory_mgr.list_projects()
    }

@app.post("/api/projects/create")
def create_project(req: ProjectReq):
    res = memory_mgr.create_project(req.project_id)
    return {"success": True, **res, "projects": memory_mgr.list_projects()}

@app.post("/api/projects/switch")
def switch_project(req: ProjectReq):
    """Full context swap - tasks, shared memory, chat history and bot workspaces all
    follow the project."""
    memory_mgr.create_project(req.project_id)
    new_pid = orchestrator.set_project(req.project_id)
    _save_active_project_id(new_pid)
    return {
        "success": True,
        "active_project_id": new_pid,
        "projects": memory_mgr.list_projects()
    }

@app.post("/api/projects/delete")
def delete_project(req: ProjectReq):
    """Archives a project (memory + workspaces) into trash.

    Deleting the ACTIVE project used to be refused outright ("switch away first"). The UI
    only ever offers the active project for deletion - selecting a project in the switcher
    IS switching to it - so that refusal made the Delete button impossible to satisfy: the
    project you had selected was always the one you could not delete. Auto-switch away
    instead, which is what "switch away first" was asking the user to do by hand anyway.

    The one case still refused is the last project standing - there is nowhere to switch to,
    and a room with no project has no memory archive, no itinerary and no workspace root.
    """
    from backend.tools import slugify_project_id
    pid = slugify_project_id(req.project_id)
    pdir = os.path.join(".swarmchat", "projects", pid)
    if not os.path.isdir(pdir):
        raise HTTPException(status_code=404, detail=f"Project '{pid}' not found.")

    switched_to = None
    if pid == memory_mgr.get_project_id():
        others = [
            p["project_id"] for p in memory_mgr.list_projects()
            if p.get("project_id") and p["project_id"] != pid
        ]
        if not others:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{pid}' is the only project. Create another project first, then delete "
                    f"this one."
                )
            )
        # A turn in flight would keep writing into the project we are about to archive.
        orchestrator.loop_active = False
        switched_to = orchestrator.set_project(others[0])
        _save_active_project_id(switched_to)

    dest = tool_mgr.move_to_trash(pdir, label=f"project_{pid}")
    return {
        "success": True,
        "trashed_to": dest,
        "deleted_project_id": pid,
        # Non-null when deleting the active project forced a switch, so the UI can say where
        # the room ended up instead of silently changing under the user.
        "switched_to": switched_to,
        "active_project_id": memory_mgr.get_project_id(),
        "projects": memory_mgr.list_projects()
    }

@app.get("/api/frontend/status")
def frontend_status():
    return {"success": True, **get_frontend_status()}

@app.post("/api/frontend/rebuild")
def rebuild_frontend():
    """Rebuilds the UI bundle from frontend/src so source edits actually reach the
    browser. Installs dependencies first if node_modules is missing."""
    global _frontend_rebuild_log
    status = get_frontend_status()
    if not status["node_available"]:
        raise HTTPException(
            status_code=503,
            detail="npm was not found on PATH. Install Node.js (LTS) and restart the backend to rebuild the UI here."
        )
    log: List[str] = []
    if not status["deps_installed"]:
        log.append("$ npm install")
        # --include=dev is explicit because a global `omit=dev` npm config would
        # otherwise skip vite and the build would fail with "'vite' is not recognized".
        install = _run_npm(["install", "--no-audit", "--no-fund", "--include=dev"], timeout=900)
        log.extend(install.get("output", []) or [install.get("error", "")])
        if not install["success"]:
            _frontend_rebuild_log = log
            raise HTTPException(status_code=500, detail="npm install failed. See last_rebuild_log in /api/frontend/status.")
    log.append("$ npm run build")
    build = _run_npm(["run", "build"], timeout=900)
    log.extend(build.get("output", []) or [build.get("error", "")])
    _frontend_rebuild_log = log
    if not build["success"]:
        raise HTTPException(status_code=500, detail="npm run build failed. See last_rebuild_log in /api/frontend/status.")
    logger.info("Frontend bundle rebuilt")
    return {"success": True, "log": log[-40:], **get_frontend_status()}

@app.post("/api/workspace/clean")
def clean_workspaces(req: WorkspaceCleanReq):
    """Tidies the active project's bot workspaces: orphaned model dirs and files older
    than max_age_days go to trash, __pycache__ is removed."""
    active_ids = orchestrator.get_active_model_ids() if req.prune_orphans else None
    return tool_mgr.clean_workspaces(
        active_bot_ids=active_ids,
        max_age_days=req.max_age_days
    )

@app.get("/api/workspace/file")
def get_workspace_file_content(filepath: str):
    res = tool_mgr.read_file(filepath)
    if not res.get("success"):
        raise HTTPException(status_code=404, detail=res.get("error", f"Could not read '{filepath}'."))
    return res

@app.get("/api/workspace/audit")
def get_workspace_file_audit(filepath: Optional[str] = None):
    return {
        "success": True,
        "audit_log": memory_mgr.get_file_audit_log(filepath),
        "active_file_locks": memory_mgr.state.get("active_file_locks", {})
    }

@app.post("/api/chat/stop")
def trigger_emergency_stop():
    res = orchestrator.emergency_stop()
    return res

@app.post("/api/phase")
def set_phase(req: PhaseSwitchReq):
    """Admin's manual override. Models cannot reach this - the Architect opens Execution
    through the plan gate - but the Admin can still force either direction."""
    try:
        new_p = memory_mgr.set_phase(req.phase)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if new_p == "discussion":
        # Coming back to discussion restarts planning. Leaving the gate on `approved` would
        # let the Architect flip straight back to execution on its very next turn without
        # anyone re-reading the plan that just got sent back.
        memory_mgr.reset_plan_gate()
    return {"success": True, "phase": new_p, **orchestrator.plan_gate_status()}

@app.post("/api/chat/message")
async def send_chat_message(req: ChatMsgReq):
    msg = orchestrator.add_chat_message(
        sender=req.sender,
        role="Admin" if req.is_admin else "User",
        content=req.content,
        is_admin=req.is_admin
    )
    # Automatically launch conversation loop in background
    _track_background_task(
        asyncio.create_task(orchestrator.run_autonomous_loop(max_turns=5)),
        "autonomous conversation loop"
    )
    return {"success": True, "message": msg}

@app.post("/api/chat/step")
async def step_turn(model_id: Optional[str] = None):
    target_id = model_id or orchestrator.get_next_speaker()
    if not target_id:
        raise HTTPException(status_code=409, detail="No active model available for turn.")
    try:
        msg = await orchestrator.step_model_turn(target_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'")) from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return {"success": True, "speaker_id": target_id, "message": msg}

@app.post("/api/models/configure")
def configure_model(req: ModelConfigReq):
    global last_loaded_model_dir
    m_dict = req.dict()

    # Store persistent last loaded model directory
    gguf_p = req.gguf_path or req.model_name
    resolved = model_mgr.resolve_gguf_path(gguf_p)
    if resolved and os.path.exists(resolved):
        last_loaded_model_dir = os.path.dirname(resolved)

    orchestrator.add_or_update_known_model(m_dict)

    # Roster capacity check. Nothing used to stop a user from adding more GGUFs than the box
    # can hold; the app only found out mid-conversation, when the turn landed on a model with
    # no room and the room appeared to hang. Report it at configure time instead.
    capacity_warning = None
    try:
        hw = model_mgr.get_hardware_info()
        budget_gb = max(hw.get("vram_total_gb", 0.0), 0.0) or max(hw.get("ram_total_gb", 0.0) - 4.0, 0.0)
        roster_gb, counted = 0.0, 0
        for cfg in orchestrator.models.values():
            if cfg.get("provider") != "gguf_local":
                continue
            p = model_mgr.resolve_gguf_path(cfg.get("gguf_path") or cfg.get("model_name", ""))
            if p and os.path.exists(p):
                roster_gb += os.path.getsize(p) / (1024 ** 3)
                counted += 1
        roster_gb = round(roster_gb, 2)
        if budget_gb and roster_gb > budget_gb:
            capacity_warning = (
                f"Roster of {counted} local models totals {roster_gb} GB, more than the "
                f"{budget_gb:.1f} GB available. They cannot all stay resident: SwarmChat will "
                f"unload the least recently used model on each turn, so expect a reload pause "
                f"whenever the turn rotates. Drop a model or pick smaller quants to avoid it."
            )
    except Exception as e:
        logger.debug("Roster capacity check skipped: %s", e)

    return {
        "success": True,
        "models": orchestrator.models,
        "known_models": orchestrator.known_models,
        "last_loaded_dir": last_loaded_model_dir,
        "capacity_warning": capacity_warning
    }

@app.post("/api/models/kick")
def kick_model(model_id: str):
    if model_id not in orchestrator.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' is not in the chat room.")
    return orchestrator.kick_model_from_room(model_id)

@app.post("/api/models/readd")
def readd_model(model_id: str):
    res = orchestrator.readd_model_to_room(model_id)
    if not res.get("success"):
        raise HTTPException(status_code=404, detail=res.get("error", f"Model '{model_id}' not found."))
    return res

@app.post("/api/models/set_moderator", deprecated=True)
def set_moderator(model_id: str):
    """Retired. Architect and Moderator are one seat, named by a model's `role`.

    Kept as a 410 rather than deleted so an older cached UI bundle gets a readable reason
    instead of a 404 that looks like a broken build.
    """
    raise HTTPException(
        status_code=410,
        detail=("The moderator flag has been removed - the supervisor seat is whichever model "
                "holds the Architect role. Change that model's role instead.")
    )

@app.get("/api/tools/search_hf")
async def search_huggingface(query: str, limit: int = 5):
    res = await tool_mgr.search_huggingface(query, limit=limit)
    if not res.get("success"):
        # 502: the upstream provider failed, and the caller must not treat this as "no results".
        raise HTTPException(status_code=502, detail=res.get("error", "HuggingFace search failed."))
    return res

@app.get("/api/tools/internet_search")
async def internet_search(query: str, domain_filter: Optional[str] = None):
    res = await tool_mgr.internet_search(query, domain_filter=domain_filter)
    if not res.get("success"):
        raise HTTPException(status_code=502, detail=res.get("error", "Internet search failed."))
    return res

@app.post("/api/votes/override")
def override_vote(req: VoteOverrideReq):
    res = orchestrator.admin_override_vote(req.vote_id, req.action, req.modified_args)
    if not res.get("success"):
        status = 404 if "not found" in str(res.get("error", "")).lower() else 400
        raise HTTPException(status_code=status, detail=res.get("error", "Vote override failed."))
    return res

@app.post("/api/plan/questions/answer")
def answer_plan_question(req: PlanQuestionAnswerReq):
    """The Admin's answer to a parked planning question.

    The room never waited for this - the question was checkpointed and the next item was
    worked - so answering RESUMES the gate rather than unblocking it, and nothing that
    happened in the meantime is lost."""
    res = orchestrator.answer_plan_question(req.question_id, req.answer)
    if not res.get("success"):
        status = 404 if "not found" in str(res.get("error", "")).lower() else 400
        raise HTTPException(status_code=status, detail=res.get("error", "Answering the question failed."))
    return res

@app.post("/api/hiring/vote")
def submit_hire_vote(req: HireVoteReq):
    """Handles a successful bot vote on a HuggingFace candidate and notifies the Admin via private notification."""
    admin_msg = orchestrator.add_chat_message(
        sender="System / Hiring Pipeline",
        role="System",
        content=f"📬 [PRIVATE NOTIFICATION TO ADMIN]: The room models voted unanimously ({len(req.votes_for)} votes) to hire candidate '{req.model_name}' ({req.model_id}). Notes: {req.notes or 'None'}. GGUF Tag: {req.gguf_url_or_tag}",
        is_admin=True
    )
    return {"success": True, "notification_sent": True, "message": admin_msg}

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
    res = tool_mgr.list_files(rel_dir)
    if not res.get("success"):
        raise HTTPException(status_code=404, detail=res.get("error", f"Could not list '{rel_dir}'."))
    return res

frontend_dist = os.path.join(os.path.dirname(__file__), "../frontend/dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
