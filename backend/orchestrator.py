import difflib
import json
import logging
import os
import re
import time
import asyncio
from typing import Callable, Dict, Any, List, Optional
from backend.errors import DirectiveParseError, ModelInvocationError, SwarmChatError
from backend.models import ModelManager, DEFAULT_N_CTX
from backend.prompts import get_system_prompt
from backend.memory import MemoryManager
from backend.tools import ToolManager

logger = logging.getLogger(__name__)

# Auto-mode stall watchdog: max seconds a single turn may take before
# run_autonomous_loop forces rotation to the next speaker instead of hanging.
# Module-level so tests can shrink it instead of actually waiting real minutes.
AUTO_TURN_TIMEOUT_SECONDS = 90

# How many consecutive turns one model may take while draining its queue of outstanding
# tasks before the room moves to another role. Batching is what stops the load/evict thrash,
# but an unbounded batch means a Coder could write five files before any Critic sees one and
# repeat the same mistake five times. Capping it also leaves downstream roles a backlog of
# several items, so a failed review doesn't leave the Critic idle waiting for one task to
# come back around.
EXECUTION_BATCH_CAP = 3


def create_model_config(
    model_id: str,
    name: str,
    role: str,
    model_name: str,
    provider: str = "ollama",
    enabled: bool = True,
    is_moderator: bool = False,
    status: str = "active",
    max_context_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    escalation_model_path: Optional[str] = None,
    **extra
) -> Dict[str, Any]:
    """Factory helper to create a model configuration dictionary with canonical defaults.

    `is_moderator` is accepted for call-compatibility and DELIBERATELY DISCARDED. Architect
    and Moderator are one seat, named by `role`; a separate flag could only agree with the
    role (redundant) or disagree with it (a bug - it used to hand Architect duties to
    whichever model had the flag, regardless of its actual role).
    """
    cfg = {
        "id": model_id,
        "name": name,
        "role": role,
        "provider": provider,
        "model_name": model_name,
        "enabled": enabled,
        "status": status,
        "max_context_tokens": max_context_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        # Optional heavier GGUF model swapped in only for the Architect's escalation
        # turn (task failed 3x in execution phase).
        "escalation_model_path": escalation_model_path,
    }
    extra.pop("is_moderator", None)  # never let it back in through **extra
    cfg.update(extra)
    return cfg


class Orchestrator:
    def __init__(self, model_manager: ModelManager, memory_manager: MemoryManager,
                 tool_manager: ToolManager, storage_root: Optional[str] = None):
        self.model_manager = model_manager
        self.memory_manager = memory_manager
        self.tool_manager = tool_manager

        # Where the roster and per-project chat logs live. This used to be a hardcoded
        # "./.swarmchat" relative to the process CWD, which meant a test passing
        # MemoryManager(storage_dir=<tmp>) still loaded the *user's real* roster and chat
        # log - so tests referring to the default model ids failed, and synthetic messages
        # from one verifier script were restored as live conversation by the next. Follow
        # the MemoryManager's storage dir so isolating memory isolates everything.
        self.storage_root = self._resolve_storage_root(memory_manager, storage_root)
        self.ROSTER_FILE = os.path.join(self.storage_root, "roster.json")
        self.RAW_TURN_LOG = os.path.join(self.storage_root, "raw_turns.log")

        self.turn_mode = "round_robin"
        self.voting_threshold = "majority"
        self.respond_immediately_to_at = True

        self.loop_active = False
        self.tie_counters: Dict[str, int] = {}
        self.last_speaker_id: Optional[str] = None
        # Raw (pre-display-formatting) text of each model's last generation, used for
        # repetition detection - see the dedup guard in step_model_turn.
        self._last_raw_response: Dict[str, str] = {}

        # Known models library with specialized default model recommendations per role
        self.known_models: Dict[str, Dict[str, Any]] = {
            "model_architect": create_model_config("model_architect", "Architect", "Architect", "llama3.2:1b", is_moderator=True, temperature=0.7, repeat_penalty=1.1),
            "model_critic": create_model_config("model_critic", "Critic", "Critic", "qwen2.5-coder:3b", is_moderator=False, temperature=0.7, repeat_penalty=1.1),
            "model_coder": create_model_config("model_coder", "Coder", "Coder", "qwen2.5-coder:1.5b", is_moderator=False, temperature=0.1, repeat_penalty=1.02),
            # "Refiner" and "Tester/Debugger" were the same job under two different names
            # (test the code, find what's wrong, fix it) - unified onto the single
            # "Tester/Debugger" role, which already has full prompt/UI wiring. model_id kept
            # as "model_refiner" for backward compat with any saved configs referencing it.
            "model_refiner": create_model_config("model_refiner", "Tester/Debugger", "Tester/Debugger", "qwen2.5-coder:3b", is_moderator=False, temperature=0.1, repeat_penalty=1.05),
        }

        # Active chatroom models (subset of known models currently in the room)
        # Initial status set to "Connecting..." until confirmed connected/online on turn step or validation
        self.models: Dict[str, Dict[str, Any]] = {
            m_id: {**cfg, "live_status": "Connecting..."} for m_id, cfg in self.known_models.items()
        }

        self.pending_tool_votes: List[Dict[str, Any]] = []
        self.chat_history: List[Dict[str, Any]] = []
        self.turn_schedule: List[str] = []  # 5-10 turn scheduled roster queue

        # Residency-major execution scheduling (see _select_execution_speaker).
        self.pinned_task_id: Optional[str] = None   # the one task this turn is about
        self._served_mention_id: Optional[str] = None  # chat msg whose @mention was honoured
        self._cohort_model_id: Optional[str] = None  # model currently draining its queue
        self._cohort_count: int = 0                  # turns taken in the current batch
        self.autorun_enabled: bool = False
        self.last_speech_time: float = time.time()
        self.spoken_models: set = set()  # Tracks models that have spoken at least once

        # Restore the room the user actually built. Roster/known-model state lived only in
        # memory, so every backend restart silently reset the room to the placeholder
        # Ollama models - and those point at models that were never pulled, so the room
        # came back broken and every configured GGUF had to be re-added by hand.
        self.load_roster()
        # Conversation for the active project, restored across restarts/project switches.
        self.load_chat_history()

    DEFAULT_STORAGE_ROOT = ".swarmchat"

    # Kept as a class-level default so anything still reading Orchestrator.ROSTER_FILE
    # resolves; instances override it from self.storage_root in __init__.
    ROSTER_FILE = os.path.join(DEFAULT_STORAGE_ROOT, "roster.json")

    @staticmethod
    def _resolve_storage_root(memory_manager: Any, override: Optional[str]) -> str:
        """Explicit override wins; otherwise follow the MemoryManager's storage dir.

        Falls back to the app default when the memory manager has no usable dir (a test
        double, for instance) - never guess a path that could be the user's real one."""
        if isinstance(override, str) and override:
            return override
        base = getattr(memory_manager, "base_storage_dir", None)
        if isinstance(base, str) and base:
            return base
        return Orchestrator.DEFAULT_STORAGE_ROOT

    _ROSTER_TRANSIENT_KEYS = ("live_status",)

    def save_roster(self):
        """Persists the active room + known-model library so a restart resumes the same setup."""
        try:
            os.makedirs(os.path.dirname(self.ROSTER_FILE), exist_ok=True)
            payload = {
                # Written for readability/back-compat only - load_roster ignores it, because
                # the supervisor seat is derived from the Architect role.
                "moderator_model_id": self.moderator_model_id,
                "known_models": self.known_models,
                "active_model_ids": list(self.models.keys()),
            }
            tmp = self.ROSTER_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self.ROSTER_FILE)
        except (OSError, TypeError, ValueError) as e:
            # Never let a persistence failure break a live room operation.
            logger.warning("Could not persist roster to %s: %s", self.ROSTER_FILE, e)

    def load_roster(self):
        """Restores a previously saved roster, falling back to defaults on any problem."""
        if not os.path.exists(self.ROSTER_FILE):
            return
        try:
            with open(self.ROSTER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"expected object, got {type(data).__name__}")
            known = data.get("known_models")
            if not isinstance(known, dict) or not known:
                raise ValueError("no known_models in saved roster")
            # Migration: drop the retired is_moderator flag from anything saved earlier. A
            # roster written before the Architect/Moderator merge can carry the flag on a
            # non-Architect model (e.g. the Coder), which used to silently hand that model
            # the Architect's planning, escalation and VRAM priority.
            for _cfg in known.values():
                self._strip_moderator_flag(_cfg)
            self.known_models = known
            active_ids = [m for m in data.get("active_model_ids", []) if m in known]
            if not active_ids:
                active_ids = list(known.keys())
            self.models = {
                m_id: {**known[m_id], "live_status": "Connecting..."} for m_id in active_ids
            }
            # data["moderator_model_id"] is deliberately NOT read - see the property.
            logger.info("Restored roster from %s: %d active model(s); supervisor=%s",
                        self.ROSTER_FILE, len(self.models), self.moderator_model_id)
        except (OSError, ValueError, json.JSONDecodeError, KeyError) as e:
            logger.warning("Could not restore roster from %s (%s); using defaults", self.ROSTER_FILE, e)

    # --- PROJECT SWITCHING ---
    def _chat_history_path(self, project_id: Optional[str] = None) -> str:
        pid = project_id or self.memory_manager.get_project_id()
        return os.path.join(self.storage_root, "projects", pid, "chat_history.json")

    def save_chat_history(self, project_id: Optional[str] = None):
        """Chat history used to be in-memory only, so a project switch (or a restart)
        threw the conversation away. Persisted per project so switching back resumes."""
        path = self._chat_history_path(project_id)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.chat_history[-500:], f, indent=2)
            os.replace(tmp, path)
        except (OSError, TypeError, ValueError) as e:
            logger.warning("Could not persist chat history to %s: %s", path, e)

    def load_chat_history(self, project_id: Optional[str] = None):
        path = self._chat_history_path(project_id)
        if not os.path.exists(path):
            self.chat_history = []
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chat_history = data if isinstance(data, list) else []
        except (OSError, ValueError, json.JSONDecodeError) as e:
            logger.warning("Could not load chat history from %s: %s", path, e)
            self.chat_history = []

    def set_project(self, project_id: str) -> str:
        """Full context swap: memory archive, bot workspaces and chat history all move
        to the named project, and per-turn execution state is reset so the scheduler
        does not keep chasing a task that belongs to the project we just left."""
        current = self.memory_manager.get_project_id()
        if project_id == current:
            return current
        self.save_chat_history(current)
        self.memory_manager.set_project_id(project_id)
        new_pid = self.memory_manager.get_project_id()
        self.tool_manager.set_project_id(new_pid)
        # Blank the in-memory log *before* loading the new project's. Every consumer that
        # scans chat_history (the @mention pre-empt, the "recent messages" prompt window,
        # the 15-message fairness rule) reads a plain list with no project tag on it, so a
        # failed or partial load used to leave the previous project's conversation sitting
        # in the room - the Architect planning project A inside project B's chat.
        self.chat_history = []
        self._served_mention_id = None
        self.load_chat_history(new_pid)
        self.pinned_task_id = None
        self._cohort_model_id = None
        self._cohort_count = 0
        self.last_speaker_id = None
        self._last_raw_response = {}
        self.spoken_models = set()
        self.pending_tool_votes = []
        self.turn_schedule = []
        logger.info("Switched project '%s' -> '%s'", current, new_pid)
        return new_pid

    VALID_TURN_MODES = ("admin_controlled", "moderator_controlled", "round_robin")

    def get_active_models(self) -> List[Dict[str, Any]]:
        """Returns list of model configuration dicts for active/enabled models in the chat room."""
        return [m for m in self.models.values() if m.get("enabled", True)]

    def get_active_model_ids(self) -> List[str]:
        """Returns list of IDs for active/enabled models in the chat room."""
        return [m_id for m_id, m in self.models.items() if m.get("enabled", True)]

    def _get_model_meta(self, model_id: str) -> Dict[str, Any]:
        """Returns normalized model metadata safely."""
        m_cfg = self.models.get(model_id, {})
        return {
            "id": model_id,
            "name": m_cfg.get("name", "Unknown"),
            "role": m_cfg.get("role", "Participant"),
            # Derived from role now, not a stored flag.
            "is_moderator": self._is_supervisor(m_cfg),
            "enabled": m_cfg.get("enabled", True),
            "live_status": m_cfg.get("live_status", "Unknown")
        }

    def set_turn_mode(self, mode: str):
        if mode not in self.VALID_TURN_MODES:
            raise ValueError(f"Unknown turn mode '{mode}'. Valid modes: {', '.join(self.VALID_TURN_MODES)}.")
        self.turn_mode = mode

    # set_moderator() was removed. The supervisor seat is whichever model holds the
    # Architect/planner role, so there is nothing separate to assign - to change who
    # supervises, change that model's role. `moderator_model_id` is now a derived property.

    @staticmethod
    def _strip_moderator_flag(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Drops the retired `is_moderator` key from a config loaded off disk or the API."""
        cfg.pop("is_moderator", None)
        return cfg

    def emergency_stop(self) -> Dict[str, Any]:
        """Halts all active conversation loops, background tasks, and pending tool calls without deleting evidence or history."""
        self.loop_active = False
        self.turn_schedule = []
        stopped_votes_count = len(self.pending_tool_votes)
        for v in self.pending_tool_votes:
            if v.get("status") == "pending":
                v["status"] = "stopped"

        self.add_chat_message(
            sender="System / Emergency Stop",
            role="System",
            content="🛑 [EMERGENCY STOP TRIGGERED]: All active conversation loops, tool executions, and turn schedules have been forcefully halted. Historical logs and evidence remain preserved.",
            is_admin=True
        )
        return {
            "success": True,
            "message": "Emergency stop executed successfully.",
            "stopped_votes_count": stopped_votes_count
        }

    def add_or_update_known_model(self, model_cfg: Dict[str, Any]):
        m_id = model_cfg["id"]
        if "live_status" not in model_cfg:
            model_cfg["live_status"] = "Connecting..."
        self._strip_moderator_flag(model_cfg)
        self.known_models[m_id] = model_cfg
        self.models[m_id] = dict(model_cfg)
        self.save_roster()

    def set_model_live_status(self, model_id: str, status: str):
        if model_id in self.models:
            self.models[model_id]["live_status"] = status
        if model_id in self.known_models:
            self.known_models[model_id]["live_status"] = status

    def kick_model_from_room(self, model_id: str) -> Dict[str, Any]:
        was_moderator = False
        if model_id in self.models:
            was_moderator = self._is_supervisor(self.models[model_id])
            del self.models[model_id]

        # No auto-reassignment any more: the supervisor seat follows the Architect role, so
        # kicking the Architect leaves the room genuinely without one until another
        # Architect-role model is added. Silently promoting an arbitrary survivor (the old
        # behaviour) handed Architect duties to whatever happened to be first in the dict.
        new_mod_id = self._architect_id(self.get_active_model_ids())
        if was_moderator and not new_mod_id:
            self.add_chat_message(
                sender="System / Role Manager",
                role="System",
                content=("⚠️ [NO ARCHITECT] The model holding the Architect seat was removed. "
                         "Planning, task creation and escalation are unavailable until a model "
                         "with the Architect role is added to the room."),
                is_admin=True
            )

        self.save_roster()
        return {
            "success": True,
            "kicked_id": model_id,
            "was_moderator": was_moderator,
            "auto_assigned_moderator": new_mod_id,
            "active_models": self.models
        }

    def readd_model_to_room(self, model_id: str) -> Dict[str, Any]:
        if model_id in self.known_models:
            m_cfg = dict(self.known_models[model_id])
            m_cfg["live_status"] = "Connecting..."
            self.models[model_id] = self._strip_moderator_flag(m_cfg)
            self.save_roster()
            return {"success": True, "model": self.models[model_id]}
        return {"success": False, "error": "Model not found in known models library"}

    def add_chat_message(self, sender: str, role: str, content: str, is_admin: bool = False, model_id: Optional[str] = None) -> Dict[str, Any]:
        msg = {
            "id": f"msg_{int(time.time()*1000)}",
            "timestamp": time.time(),
            "sender": sender,
            "role": role,
            "content": content,
            "is_admin": is_admin,
            "model_id": model_id,
            "phase": self.memory_manager.get_phase()
        }
        self.chat_history.append(msg)
        self.save_chat_history()
        return msg

    def generate_turn_schedule(self, length: int = 8) -> List[str]:
        """Generates a 5-10 turn scheduled roster based on available user roles, prioritizing Architect first."""
        active_models = self.get_active_model_ids()
        if not active_models:
            return []

        schedule: List[str] = []

        # 1. Ensure Architect is first if present and not spoke very recently
        architect_id = self._architect_id(active_models)

        if architect_id:
            schedule.append(architect_id)

        # 2. Add other active roles sequentially, ensuring fair coverage without default bias
        remaining = [m_id for m_id in active_models if m_id not in schedule]
        while len(schedule) < length:
            if not remaining:
                remaining = list(active_models)
            next_m = remaining.pop(0)
            schedule.append(next_m)

        self.turn_schedule = schedule
        return self.turn_schedule

    # Execution-phase task status -> which role should take the next turn.
    _EXECUTION_STAGE_ROLES = {
        "in_progress": ("coder",),
        "needs_review": ("critic",),
        "needs_test": ("tester", "debugger"),
        "failed": ("coder",),
    }

    # Order in which outstanding work is considered. Same precedence the old single-task
    # picker used: unblock things that are waiting on someone before starting new work.
    _STATUS_PRECEDENCE = ("failed", "needs_review", "needs_test", "in_progress", "pending")

    # Instance-level alias so a test (or a user with a bigger GPU) can tune the batch size
    # without patching the module constant.
    EXECUTION_BATCH_CAP = EXECUTION_BATCH_CAP

    def _resident_model_ids(self) -> set:
        """Model ids that can take a turn right now without loading anything.

        Cloud/Ollama models cost no VRAM, so they always count as resident; a local GGUF
        counts only while its llama.cpp instance is actually live."""
        resident = set()
        for m_id, cfg in self.models.items():
            if not cfg.get("enabled", True):
                continue
            if cfg.get("provider") != "gguf_local" or m_id in self.model_manager.gguf_instances:
                resident.add(m_id)
        return resident

    def _outstanding_work(self, active_models: List[str]) -> List[tuple]:
        """[(task, model_id)] for every task some configured role could advance, in
        precedence order. One entry per task - the model that owns its current stage."""
        itinerary = self.memory_manager.get_task_itinerary()
        by_status = {s: [] for s in self._STATUS_PRECEDENCE}
        for t in itinerary:
            st = t.get("status", "pending")
            if st in by_status:
                by_status[st].append(t)

        work = []
        for status in self._STATUS_PRECEDENCE:
            for t in by_status[status]:
                if status == "pending":
                    # Untouched work still needs the Architect to turn it into a real task.
                    owner = self._architect_id(active_models)
                elif status == "failed" and t.get("attempt_count", 0) >= 3:
                    owner = self._architect_id(active_models)
                else:
                    roles = self._EXECUTION_STAGE_ROLES.get(status)
                    exclude = t.get("author_bot_id") if status == "needs_review" else None
                    owner = self._find_model_by_role(active_models, roles, exclude=exclude) if roles else None
                if owner:
                    work.append((t, owner))
        return work

    def _architect_id(self, active_models: List[str]) -> Optional[str]:
        """The model holding the Architect/supervisor seat, by role alone."""
        for m_id in active_models:
            if self._is_supervisor(self.models[m_id]):
                return m_id
        return None

    @property
    def moderator_model_id(self) -> Optional[str]:
        """Derived, not stored: the supervisor IS the Architect.

        Kept as a name because the API/UI and the VRAM prefetcher ask for it, but it can no
        longer be set to something that contradicts the roster."""
        return self._architect_id(self.get_active_model_ids())

    def _find_model_by_role(self, active_models: List[str], role_substrings: tuple, exclude: Optional[str] = None) -> Optional[str]:
        for m_id in active_models:
            if exclude and m_id == exclude:
                continue
            role_lower = self.models[m_id].get("role", "").lower()
            if any(s in role_lower for s in role_substrings):
                return m_id
        return None

    # --- PRE-EXECUTION PLANNING GATE -------------------------------------------------
    # Discussion used to be shapeless: the round-robin roster picked whoever was next and
    # the autonomous loop force-flipped to execution after N turns or M seconds. A plan
    # could therefore reach the Coder with nobody having read it. The gate below gives the
    # planning phase the same kind of state machine execution already has.
    #
    #   awaiting_plan     Architect proposes (or rehashes) the build plan
    #   critic_review     Critic hunts for weak / contradictory parts -> APPROVE or REJECT
    #   programmer_review Coder confirms it is actually buildable    -> APPROVE or REJECT
    #   approved          Architect - and only the Architect - calls [READY_FOR_EXECUTION]
    #
    # A REJECT at either review returns the room to awaiting_plan for a rehash.
    _CRITIC_ROLES = ("critic", "reviewer")
    _PROGRAMMER_ROLES = ("coder", "programmer", "developer", "engineer")

    # stage -> the roles that own the turn at that stage.
    _PLAN_STAGE_ROLES = {
        "awaiting_plan": "architect",
        "critic_review": "critic",
        "programmer_review": "programmer",
        "approved": "architect",
    }

    def _critic_id(self, active_models: List[str]) -> Optional[str]:
        return self._find_model_by_role(active_models, self._CRITIC_ROLES)

    def _programmer_id(self, active_models: List[str]) -> Optional[str]:
        return self._find_model_by_role(active_models, self._PROGRAMMER_ROLES)

    def _plan_stage_owner(self, stage: str, active_models: Optional[List[str]] = None) -> Optional[str]:
        """Who speaks at this stage, or None when that seat is not in the room."""
        active_models = self.get_active_model_ids() if active_models is None else active_models
        role = self._PLAN_STAGE_ROLES.get(stage, "architect")
        if role == "critic":
            return self._critic_id(active_models)
        if role == "programmer":
            return self._programmer_id(active_models)
        return self._architect_id(active_models)

    def plan_gate_status(self) -> Dict[str, Any]:
        """Public read of the gate for /api/state - stage, revision, and who owes the move."""
        stage = self.memory_manager.get_plan_stage()
        owner = self._plan_gate_speaker() if self.memory_manager.get_phase() == "discussion" else None
        return {
            "plan_stage": stage,
            "plan_revision": self.memory_manager.get_plan_revision(),
            "plan_stage_owner": owner,
            "plan_stage_owner_name": self.models.get(owner, {}).get("name") if owner else None,
        }

    def _next_plan_stage(self, stage: str) -> str:
        """The stage after this one, skipping review seats the roster does not contain.

        Admin's answer to 'what if there is no Critic?' was: skip that step. A gate that
        blocks on an absent seat is just a deadlock with better manners."""
        order = list(MemoryManager.PLAN_STAGES)
        try:
            idx = order.index(stage)
        except ValueError:
            return "awaiting_plan"
        active_models = self.get_active_model_ids()
        for nxt in order[idx + 1:]:
            if nxt == "approved" or self._plan_stage_owner(nxt, active_models):
                return nxt
        return "approved"

    def _advance_plan_stage(self, stage: str, reason: str) -> str:
        """Move the gate forward and say so in chat, so the room's state is legible."""
        new_stage = self.memory_manager.set_plan_stage(stage)
        labels = {
            "awaiting_plan": "📝 PLAN REVISION",
            "critic_review": "🔍 CRITIC REVIEW",
            "programmer_review": "🔧 PROGRAMMER REVIEW",
            "approved": "✅ PLAN APPROVED",
        }
        owner = self._plan_stage_owner(new_stage)
        owner_name = self.models.get(owner, {}).get("name") if owner else None
        who = f" Next up: @{owner_name}." if owner_name else ""
        self.add_chat_message(
            sender="System / Plan Gate",
            role="System",
            content=f"{labels.get(new_stage, new_stage.upper())} — {reason}{who}",
            is_admin=True
        )
        return new_stage

    def _plan_gate_speaker(self) -> Optional[str]:
        """Whose turn it is during discussion, according to the gate.

        Returns None only when the gate's seat is empty *and* no Architect exists, in which
        case get_next_speaker falls back to the legacy roster so the room is never silent."""
        active_models = self.get_active_model_ids()
        if not active_models:
            return None
        stage = self.memory_manager.get_plan_stage()
        owner = self._plan_stage_owner(stage, active_models)
        if owner:
            return owner
        # The seat this stage needs is not in the room. Skip ahead rather than stall - the
        # same rule _next_plan_stage applies, but read-only: don't mutate state from a picker.
        for nxt in self._forward_stages(stage):
            owner = self._plan_stage_owner(nxt, active_models)
            if owner:
                return owner
        return self._architect_id(active_models)

    @staticmethod
    def _forward_stages(stage: str) -> List[str]:
        order = list(MemoryManager.PLAN_STAGES)
        try:
            return order[order.index(stage) + 1:]
        except ValueError:
            return order

    def _upcoming_planning_models(self, limit: int = 3) -> List[str]:
        """The gate's remaining running order, so the VRAM prefetcher loads the Critic and
        Programmer *alongside* the Architect instead of discovering them one eviction at a
        time. This is the 'load the models that will be needed first' half of the gate."""
        active_models = self.get_active_model_ids()
        if not active_models:
            return []
        stage = self.memory_manager.get_plan_stage()
        upcoming: List[str] = []
        for st in [stage] + self._forward_stages(stage):
            owner = self._plan_stage_owner(st, active_models)
            if owner and owner not in upcoming:
                upcoming.append(owner)
            if len(upcoming) >= limit:
                break
        # Pad with the roster queue so a two-model room still prefetches something sensible.
        for m_id in self.turn_schedule:
            if len(upcoming) >= limit:
                break
            if m_id in active_models and m_id not in upcoming:
                upcoming.append(m_id)
        return upcoming[:limit]

    def _select_execution_speaker(self) -> Optional[str]:
        """Residency-major scheduling: advance whatever the *currently loaded* models can
        advance, and only swap models when the loaded set genuinely cannot make progress.

        The previous version was task-major - it picked one task, then whichever role owned
        that task's stage. Because consecutive stages belong to different roles
        (in_progress -> Coder, needs_review -> Critic, needs_test -> Tester), every single
        task dragged the room through three model swaps. On a roster larger than VRAM that
        is maximal thrash: load, one turn, evict, repeat.

        Now the room drains what it can. A resident Coder takes every task waiting on code
        (up to EXECUTION_BATCH_CAP in a row), then hands off; a model is only loaded when no
        resident model has anything left to do. The batch cap exists so downstream roles get
        a queue of several items to fall back on - if one of them fails review, the Critic
        still has other work rather than idling until the Coder comes back around."""
        active_models = self.get_active_model_ids()
        if not active_models:
            return None

        architect_id = self._architect_id(active_models)
        work = self._outstanding_work(active_models)

        if not work:
            # Nothing outstanding - the Architect decomposes the project into new itinerary
            # items, or wraps up if everything is genuinely complete.
            self._pin_task(None)
            self._cohort_model_id, self._cohort_count = None, 0
            return architect_id

        resident = self._resident_model_ids()
        resident_work = [(t, m) for t, m in work if m in resident]

        # 1. Keep draining the current cohort while it has work and hasn't hit the cap.
        if self._cohort_model_id and self._cohort_count < self.EXECUTION_BATCH_CAP:
            for t, m in resident_work:
                if m == self._cohort_model_id:
                    return self._begin_turn(t, m)

        # 2. Cohort is capped, finished, or none is running. Prefer another model that is
        #    already loaded - switching between two resident models costs nothing.
        for t, m in resident_work:
            if m != self._cohort_model_id:
                return self._begin_turn(t, m, new_cohort=True)

        # 3. The capped cohort is the only resident model with work. Let it continue rather
        #    than evicting it to load someone with nothing queued.
        if self._cohort_model_id:
            for t, m in resident_work:
                if m == self._cohort_model_id:
                    logger.debug("Batch cap reached for %s but nothing else is resident; continuing", m)
                    return self._begin_turn(t, m, new_cohort=True)

        # 4. Nothing resident can proceed. This is the only point where a model swap is
        #    justified - the loaded team is genuinely out of work.
        t, m = work[0]
        logger.info("Loaded models are drained; switching cohort to %s for task %r", m, t.get("title"))
        return self._begin_turn(t, m, new_cohort=True)

    def _begin_turn(self, task: Dict[str, Any], model_id: str, new_cohort: bool = False) -> str:
        """Pins the task this turn is about and maintains the batch counter."""
        if new_cohort or model_id != self._cohort_model_id:
            self._cohort_model_id = model_id
            self._cohort_count = 0
        self._cohort_count += 1
        self._pin_task(task.get("id"))
        return model_id

    def _pin_task(self, task_id: Optional[str]) -> None:
        """Scopes the upcoming turn to exactly one task.

        Task isolation: a turn must see its own task and nothing about the siblings queued
        behind it in the same batch, or a model working through several tasks in a row
        starts blending their requirements."""
        self.pinned_task_id = task_id

    def _current_task(self) -> Optional[Dict[str, Any]]:
        """The task the current turn is about - the pinned one when the scheduler set it,
        otherwise whatever the itinerary says is most urgent."""
        if self.pinned_task_id:
            for t in self.memory_manager.get_task_itinerary():
                if t.get("id") == self.pinned_task_id:
                    return t
            self.pinned_task_id = None
        return self.memory_manager.get_active_task()

    def _mentioned_model(self, active_models: List[str]) -> Optional[str]:
        """The model a *person* explicitly @mentioned since the last model turn, if any.

        Three deliberate restrictions, each one a loop this caused in practice:
          - System notices are skipped. This class posts its own "@<name>" notices (turn
            timeout, dedup skip); honouring those re-selects the model that just failed.
          - Model messages end the scan. A model addressing another model must not seize
            turn order, and a restored chat log must not re-fire an ancient mention.
          - A mention is honoured once (`_served_mention_id`), so system notices posted
            after it can't leave it standing as the newest human message forever."""
        for msg in reversed(self.chat_history):
            if msg.get("role") == "System" or (msg.get("sender") or "").startswith("System"):
                continue
            if msg.get("model_id"):
                return None  # a model has spoken since; any earlier mention is stale
            if msg.get("id") and msg["id"] == self._served_mention_id:
                return None
            last_msg = (msg.get("content") or "").lower()
            for m_id in active_models:
                m_cfg = self.models[m_id]
                if f"@{m_cfg['name'].lower()}" in last_msg or f"@{m_cfg['role'].lower()}" in last_msg:
                    self._served_mention_id = msg.get("id")
                    return m_id
            return None  # only the most recent human message counts
        return None

    def _upcoming_execution_models(self, limit: int = 3) -> List[str]:
        """Who the task router will most likely call next, in order.

        The VRAM prefetcher used to read `turn_schedule` for this - but that queue is only
        consumed on the legacy (non-execution) path, so during execution the room was taking
        turns residency-major while prefetching round-robin. Two systems, one disagreeing
        about who speaks next. Derive the prefetch list from the same outstanding work the
        scheduler uses instead."""
        active_models = self.get_active_model_ids()
        if not active_models:
            return []
        upcoming: List[str] = []
        for _t, m_id in self._outstanding_work(active_models):
            if m_id not in upcoming:
                upcoming.append(m_id)
            if len(upcoming) >= limit:
                break
        if not upcoming:
            arch = self._architect_id(active_models)
            if arch:
                upcoming.append(arch)
        return upcoming

    def upcoming_speakers(self, limit: int = 3) -> List[str]:
        """Single public answer to 'who is up next', for the API/UI and the VRAM prefetcher.
        Execution phase is task-driven; discussion is gate-driven; the roster queue is now
        only a fallback. Prefetching off the raw queue during discussion loaded whoever the
        round-robin happened to list, not the Critic and Programmer the gate is about to
        call - the same class of bug as the execution-phase prefetch mismatch."""
        if self.memory_manager.get_phase() == "execution":
            return self._upcoming_execution_models(limit)
        planning = self._upcoming_planning_models(limit)
        return planning or list(self.turn_schedule[:limit])

    def get_next_speaker(self, last_speaker_id: Optional[str] = None) -> Optional[str]:
        active_models = self.get_active_model_ids()
        if not active_models:
            return None

        # @mention pre-empt. An explicit address outranks every scheduler, in every phase.
        # This used to sit *below* the execution branch, which meant mentions were silently
        # dead once execution started - the task router always answered first.
        mentioned = self._mentioned_model(active_models)
        if mentioned:
            return mentioned

        # Execution phase: route off task state instead of the fixed rotation. Falls
        # through to the rotation below only if no task-driven speaker exists at all
        # (e.g. an empty room), so the room never goes silent - but that fallback swaps
        # scheduling models mid-run, so say so loudly instead of doing it silently.
        if self.memory_manager.get_phase() == "execution":
            exec_speaker = self._select_execution_speaker()
            if exec_speaker and exec_speaker in active_models:
                return exec_speaker
            logger.warning(
                "Execution router produced no eligible speaker (returned %r); falling back to "
                "the legacy roster - this turn is NOT task-driven.", exec_speaker
            )
        else:
            # Discussion phase: the planning gate owns turn order. Architect proposes,
            # Critic reviews, Programmer signs off, Architect closes. Falling through to the
            # round-robin roster here is what let an unreviewed plan reach execution.
            gate_speaker = self._plan_gate_speaker()
            if gate_speaker and gate_speaker in active_models:
                return gate_speaker
            logger.warning(
                "Plan gate produced no eligible speaker at stage %r; falling back to the "
                "legacy roster - this turn is NOT gate-driven.",
                self.memory_manager.get_plan_stage()
            )

        # Check safety rule: Every 15 messages, ensure any model that hasn't spoken gets a turn
        if len(self.chat_history) >= 15:
            last_15 = self.chat_history[-15:]
            spoken_ids = {m.get("model_id") for m in last_15 if m.get("model_id")}
            unspoken = [m_id for m_id in active_models if m_id not in spoken_ids]
            if unspoken:
                return unspoken[0]

        # Use scheduled queue if available
        if not self.turn_schedule:
            self.generate_turn_schedule()

        if self.turn_schedule:
            next_spk = self.turn_schedule.pop(0)
            if next_spk in active_models:
                return next_spk

        # Roster queue exhausted. This used to post a chat message asking the "Chief Project
        # Manager" to refill the queue - a request no model could act on, since the schedule
        # is a Python list only this class writes. Refill it here and carry on; the room
        # never needed a model's permission to keep taking turns.
        self.generate_turn_schedule()
        if self.turn_schedule:
            next_spk = self.turn_schedule.pop(0)
            if next_spk in active_models:
                logger.debug("Turn schedule was exhausted; regenerated and resumed at %s", next_spk)
                return next_spk

        # Fallback to round-robin
        effective_last = last_speaker_id or self.last_speaker_id
        if effective_last and effective_last in active_models and len(active_models) > 1:
            idx = active_models.index(effective_last)
            return active_models[(idx + 1) % len(active_models)]

        return active_models[0]

    @staticmethod
    def _resolve_verdict(response_text: str) -> Optional[str]:
        """'approve' / 'reject' / None from a reviewer's prose.

        Shared by the execution-phase Critic gate and the pre-execution plan gate, so a
        Critic that says the same thing gets the same reading in both. Order matters:
        the explicit verdict token the output contract asks for wins; the soft keyword
        list only runs when there is no token at all."""
        resp_lower = (response_text or "").lower()
        if not resp_lower.strip():
            return None

        # 1. Explicit verdict token, as the reviewer's output contract asks for. Search the
        #    tail first so a closing verdict beats the word "approve" appearing incidentally
        #    earlier in the review prose.
        for chunk in (resp_lower[-400:], resp_lower):
            has_rej = re.search(r"\breject(ed|s)?\b", chunk)
            has_app = re.search(r"\bapprove(d|s|s it|d it)?\b", chunk)
            if has_rej and not has_app:
                return "reject"
            if has_app and not has_rej:
                return "approve"
            if has_rej and has_app:
                return "reject" if has_rej.start() > has_app.start() else "approve"

        # 2. Softer natural-language cues, widened - the original list missed the words
        #    small models actually use ("problem", "issue", "missing", "wrong").
        reject_kw = (
            "not approved", "does not pass", "doesn't pass", "flaw", "bug", "fails",
            "incorrect", "unsafe", "vulnerable", "problem", "issue", "missing",
            "wrong", "broken", "crash", "does not work", "doesn't work",
            "contradict", "unclear", "ambiguous", "weak"
        )
        approve_kw = (
            "looks good", "lgtm", "no issues", "no problems", "passes review",
            "ready for test", "meets the requirements", "satisfies the requirements",
            "no concerns", "no objections", "ready to build", "sound plan"
        )
        if any(k in resp_lower for k in approve_kw):
            return "approve"
        if any(k in resp_lower for k in reject_kw):
            return "reject"
        return None

    # A plan has to be substantive before it counts as a proposal - otherwise a one-word
    # Architect turn ("Okay.") would push the room straight into review.
    MIN_PLAN_CHARS = 60

    def _request_execution_phase(self, model_id: str, model_cfg: Dict[str, Any]) -> bool:
        """The single door into execution. Two conditions, both refusable in chat.

        Admin's rule: the Architect is the one who calls the models and switches phases.
        So a Coder that emits [READY_FOR_EXECUTION] mid-sentence no longer flips the room,
        and neither does an Architect whose plan nobody has reviewed yet. Previously this
        was an unconditional `set_phase("execution")` reachable from any model's turn.
        """
        self.memory_manager.add_entry(model_cfg["name"], "Declared readiness for Execution Phase.")
        if self.memory_manager.get_phase() != "discussion":
            return False

        name = model_cfg.get("name", model_id)

        if not self._is_supervisor(model_cfg):
            arch_id = self._architect_id(self.get_active_model_ids())
            arch_name = self.models.get(arch_id, {}).get("name", "the Architect") if arch_id else "the Architect"
            self.add_chat_message(
                sender="System / Plan Gate",
                role="System",
                content=(
                    f"🛑 @{name} called for execution, but only the Architect starts the build. "
                    f"Noted for @{arch_name}."
                ),
                is_admin=True
            )
            return False

        stage = self.memory_manager.get_plan_stage()
        if stage != "approved":
            pending = self._plan_stage_owner(stage)
            pending_name = self.models.get(pending, {}).get("name") if pending else None
            waiting = f" Waiting on @{pending_name}." if pending_name else ""
            self.add_chat_message(
                sender="System / Plan Gate",
                role="System",
                content=(
                    f"🛑 @{name} called for execution while the plan is still at "
                    f"`{stage}`. The plan needs review sign-off first.{waiting}"
                ),
                is_admin=True
            )
            return False

        self.memory_manager.set_phase("execution")
        if not self.memory_manager.get_active_task():
            self.memory_manager.add_itinerary_task(
                title="Execute Project Requirements",
                description="Implement codebase updates based on the approved build plan.",
                priority="high",
                assigned_model=model_id
            )
        self.add_chat_message(
            sender="System / Plan Gate",
            role="System",
            content=(
                f"🚀 @{name} opened Execution. Plan revision "
                f"{self.memory_manager.get_plan_revision()} cleared review; file-writing tools are unlocked."
            ),
            is_admin=True
        )
        return True

    def _advance_plan_gate(self, model_id: str, model_cfg: Dict[str, Any], response_text: str) -> None:
        """Walk the pre-execution gate on the back of the turn that just happened.

        Called only in discussion phase, and only for the model whose stage it is - a model
        speaking out of turn (an @mention, or the 15-message fairness rule) must not be able
        to advance or reset the gate."""
        stage = self.memory_manager.get_plan_stage()
        active_models = self.get_active_model_ids()
        if self._plan_stage_owner(stage, active_models) != model_id:
            return

        name = model_cfg.get("name", model_id)
        prose = self._strip_directive_tags(response_text)
        prose = re.sub(r"```[\s\S]*?```", "", prose).strip()

        if stage == "awaiting_plan":
            if (
                len(prose) < self.MIN_PLAN_CHARS
                or self._looks_like_garbage(prose)
                or self._is_prompt_echo(prose)
            ):
                logger.info("Architect %s produced no usable plan this turn; staying on awaiting_plan", model_id)
                return
            rev = self.memory_manager.get_plan_revision() + 1
            self._advance_plan_stage(
                self._next_plan_stage("awaiting_plan"),
                f"{name} put up build plan revision {rev}."
            )
            return

        if stage in ("critic_review", "programmer_review"):
            verdict = self._resolve_verdict(response_text)
            seat = "Critic" if stage == "critic_review" else "Programmer"
            if verdict == "reject":
                self._advance_plan_stage(
                    "awaiting_plan",
                    f"{name} ({seat}) rejected the plan: {prose[:160]}"
                )
            elif verdict == "approve":
                self._advance_plan_stage(
                    self._next_plan_stage(stage),
                    f"{name} ({seat}) found no blocking issues."
                )
            else:
                # No verdict is not consent. Unlike the execution-phase Critic - where an
                # actual test run is better evidence than an unfinished opinion - there is
                # nothing downstream here that would catch a bad plan, so hold the stage and
                # let the same seat try again rather than advancing on silence.
                logger.info("%s %s produced no verdict on the plan; holding at %s", seat, model_id, stage)
                self.add_chat_message(
                    sender="System / Plan Gate",
                    role="System",
                    content=(
                        f"⏸️ @{name} did not end with APPROVE or REJECT, so the plan stays in "
                        f"{seat.lower()} review. Reply with one verdict word on its own final line."
                    ),
                    is_admin=True
                )

    @staticmethod
    def _calculate_similarity(str1: str, str2: str) -> float:
        """Calculates character sequence similarity ratio between two strings."""
        if not str1 or not str2:
            return 0.0
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1, str2).ratio()

    def _build_system_prompt(self, model_id: str, model_cfg: Dict[str, Any], current_phase: str, active_task: Optional[Dict[str, Any]], is_first_turn: bool) -> str:
        # Previously every turn after a model's first collapsed to a one-line
        # "continue contributing concisely" prompt, throwing away the entire role/phase
        # directive block. Small local models then had nothing telling them to emit code,
        # action tags, or fenced blocks - so an execution-phase Coder just restated the
        # task in prose forever and the pipeline never produced a file. Always send the
        # full phase prompt; only the "you already introduced yourself" framing differs.
        base = get_system_prompt(
            role=model_cfg["role"],
            name=model_cfg["name"],
            phase=current_phase,
            project_info=self.memory_manager.get_project_id(),
            current_task=active_task["title"] if active_task else "General Discussion / Alignment",
            custom_template=model_cfg.get("custom_start_prompt") if current_phase == "discussion" else model_cfg.get("custom_execution_prompt"),
            model_id=model_id
        )
        if not is_first_turn:
            base += "\n\nYou have already introduced yourself. Do not re-introduce or restate the task - produce the next concrete increment of work."
        return base + self._role_output_contract(model_cfg, current_phase)

    ROLE_OUTPUT_CONTRACTS = {
        "coder": (
            "\n\nOUTPUT CONTRACT (Coder, execution phase) - your turn is only useful if it contains code:\n"
            "1. Reply with ONE fenced code block containing the COMPLETE file, not a snippet or a description of it.\n"
            "2. The FIRST line inside the fence must be a comment naming the file, e.g. `# filename: game.py` "
            "(or `// filename: game.js` for JS).\n"
            "3. Keep prose to at most one short sentence before the fence. Never reply with prose alone.\n"
            "4. Never restate the task back - implement it.\n"
            "5. EXCEPTION: if the last message is a direct question about existing code, answer it "
            "in one or two plain sentences and emit no code block and no action tags. Re-emitting "
            "the file to satisfy rule 1 is a wasted turn."
        ),
        "critic": (
            "\n\nOUTPUT CONTRACT (Critic, execution phase):\n"
            "1. Review the file shown to you above. Cite concrete evidence - a line, a function name, "
            "a traceback, or a failing input.\n"
            "2. End your reply with exactly one verdict word on its own line: APPROVE or REJECT.\n"
            "3. Use REJECT only with a specific, fixable reason. Do not rewrite the file yourself.\n"
            "4. Action tags are not a review. A turn made only of [SAVE_NOTE: ...] / [LOG_TO_MEMORY: ...] "
            "brackets is discarded - the findings must be written out as readable text."
        ),
        "tester": (
            "\n\nOUTPUT CONTRACT (Tester/Debugger, execution phase):\n"
            "1. State in one line what you are running and what you expect.\n"
            "2. The harness executes the task's file for you and reports the real result - do not invent output.\n"
            "3. If you need a test file, emit it as ONE fenced block whose first line is `# filename: test_<name>.py`."
        ),
        # NOTE: the Architect contract deliberately does NOT end on a bracket-directive
        # template. When it did, small models simply auto-completed the template's tail
        # (observed raw output: ", priority=high, status=in_progress]") instead of doing the
        # planning, which surfaced as an apparently "empty" or dead Architect across every
        # model tried, including an 8B. Keep the format spec in the middle and finish on a
        # concrete instruction so there is no dangling pattern to continue.
        "architect": (
            "\n\nOUTPUT CONTRACT (Architect, execution phase):\n"
            "1. Write one short paragraph naming the single next piece of work, then on its own "
            "final line emit the task directive.\n"
            "2. The directive line looks exactly like this worked example:\n"
            "   [UPDATE_TASK: title=Add CSV export, description=Create exporter.py with a "
            "write_rows(path, rows) function that writes a UTF-8 CSV with a header row, "
            "priority=high, status=in_progress]\n"
            "   Use the same four keys with your own values.\n"
            "3. The description must be implementable by a Coder who has never seen this "
            "conversation - name the file and what it must contain.\n"
            "4. If a task keeps failing, read its failure reason and describe a DIFFERENT approach, "
            "or split it smaller.\n"
            "5. Use status=in_progress so a Coder picks the task up immediately, not pending.\n"
            "6. You supervise; you do not implement. You have NO file-write permission - a fenced "
            "code block in your turn is discarded, not saved. Delegate it in the description instead.\n"
            "7. If a teammate has been idle while work is outstanding, name them in your paragraph "
            "and aim the task at their role. Never plan more than one task per turn."
        ),
        # Discussion phase: the Architect is the only role that gets a contract here, because
        # it is the one that has to decide when discussion is over. Without this it either
        # talks indefinitely or starts writing code it has no permission to save.
        "architect_discussion": (
            "\n\nOUTPUT CONTRACT (Architect, planning — you are writing the BUILD PLAN):\n"
            "1. Write the plan, not a discussion. State in order: the goal in one sentence; "
            "the files to be created or changed and what each one does; the sequence of steps; "
            "and which teammate role does each step.\n"
            "2. Keep it concrete enough that a Coder who never saw this conversation could "
            "start. Name files. Name functions.\n"
            "3. If the Critic or Programmer rejected the last revision, address their objection "
            "explicitly and say what you changed — do not re-post the same plan.\n"
            "4. You supervise; you do not implement. Fenced code from you is discarded in this "
            "phase, and file-writing tools are locked for everyone until execution begins.\n"
            "5. Do NOT declare readiness yet. The plan goes to review first; you will be asked "
            "again once it clears."
        ),
        "architect_approved": (
            "\n\nOUTPUT CONTRACT (Architect, plan approved — you are opening EXECUTION):\n"
            "1. The Critic and Programmer have signed off. In one short paragraph, restate the "
            "first piece of work and who takes it.\n"
            "2. Then emit [READY_FOR_EXECUTION] on its own final line. You are the only role "
            "that can open Execution.\n"
            "3. Emit no code."
        ),
        "critic_discussion": (
            "\n\nOUTPUT CONTRACT (Critic, plan review):\n"
            "1. You are reviewing the Architect's BUILD PLAN, not code. Hunt for the weak and "
            "contradictory parts: steps that conflict, missing files, undefined inputs, an "
            "ordering that cannot work, scope the team cannot finish.\n"
            "2. Quote the specific part of the plan you are objecting to. A vague objection "
            "cannot be fixed.\n"
            "3. End your reply with exactly one verdict word on its own line: APPROVE or REJECT.\n"
            "4. REJECT only with a specific, fixable reason. If you find nothing blocking, "
            "APPROVE — you are not required to find a problem.\n"
            "5. Do not rewrite the plan yourself and emit no code."
        ),
        "programmer_discussion": (
            "\n\nOUTPUT CONTRACT (Programmer, buildability check):\n"
            "1. The Critic has cleared this plan. Your question is narrower: can this actually "
            "be built as written? Look for missing dependencies, undefined interfaces, steps "
            "with no stated input or output, and work that is far larger than it looks.\n"
            "2. Name what you would need that the plan does not give you.\n"
            "3. End your reply with exactly one verdict word on its own line: APPROVE or REJECT.\n"
            "4. Do not start implementing. File-writing tools are locked until the Architect "
            "opens Execution, and code in this turn is wasted."
        ),
    }

    @staticmethod
    def _is_supervisor(model_cfg: Dict[str, Any]) -> bool:
        """True for the model holding the Architect seat - the supervisory role.

        Architect and Moderator are one seat, named by `role`. The old `is_moderator` flag
        was OR-ed in here, which meant a model whose role was Coder could hold Architect
        duties (planning, escalation, VRAM priority) purely because a checkbox was ticked.
        The flag is gone; role is the only source of truth."""
        role = (model_cfg.get("role") or "").lower()
        return "architect" in role or "planner" in role

    def _role_output_contract(self, model_cfg: Dict[str, Any], current_phase: str) -> str:
        """Role-specific, phase-specific instruction on what a *usable* turn looks like.

        The generic execution prompt only mentions that fenced code is auto-saved; it never
        tells a 3-4B local model that emitting code is the point. These contracts are the
        difference between the pipeline producing a file and producing chat."""
        role = (model_cfg.get("role") or "").lower()
        if current_phase != "execution":
            # Discussion is now the planning gate, so all three gate seats carry a contract
            # and the Architect's depends on where the gate is. Previously only the Architect
            # had one and it told it to declare readiness on every single turn - which is why
            # plans went to execution without the Critic ever having read them.
            stage = self.memory_manager.get_plan_stage()
            if self._is_supervisor(model_cfg):
                key = "architect_approved" if stage == "approved" else "architect_discussion"
                return self.ROLE_OUTPUT_CONTRACTS[key]
            if stage == "critic_review" and any(s in role for s in self._CRITIC_ROLES):
                return self.ROLE_OUTPUT_CONTRACTS["critic_discussion"]
            if stage == "programmer_review" and any(s in role for s in self._PROGRAMMER_ROLES):
                return self.ROLE_OUTPUT_CONTRACTS["programmer_discussion"]
            return ""
        if "coder" in role:
            key = "coder"
        elif "critic" in role:
            key = "critic"
        elif "tester" in role or "debugger" in role or "refiner" in role:
            key = "tester"
        elif self._is_supervisor(model_cfg):
            key = "architect"
        else:
            return ""
        return self.ROLE_OUTPUT_CONTRACTS[key]

    def _build_episode_context(self, active_task: Optional[Dict[str, Any]] = None) -> str:
        """Handoff checkpoints for THIS task, not the room's last three of anything.

        Task isolation: while a model drains a batch of tasks, unfiltered checkpoints are
        summaries of its *sibling* tasks, which is exactly the context that gets blended
        into the current one. When the task names a file, keep only the checkpoints that
        touched that file; episodes carrying no file attribution stay, since they are
        usually project-level handoffs rather than another task's work."""
        episodes = self.memory_manager.get_latest_episodes(limit=12)
        if not episodes:
            return ""

        focus_file = (active_task or {}).get("filename")
        if focus_file:
            scoped = [
                e for e in episodes
                if not e.get("modified_files") or focus_file in (e.get("modified_files") or [])
            ]
            episodes = scoped or episodes[-1:]
        episodes = episodes[-3:]

        header = (
            f"\n\n### EPISODE CHECKPOINTS FOR `{focus_file}`:\n" if focus_file
            else "\n\n### RECENT EPISODE CHECKPOINTS (HANDOFFS):\n"
        )
        ep_lines = [f"- [{e['author']}] Task ({e['action']}): {e['summary']}" for e in episodes]
        return header + "\n".join(ep_lines)

    def _build_task_context(self, active_task: Optional[Dict[str, Any]]) -> str:
        if not active_task:
            return ""
        ctx = f"\n\n### 🎯 ACTIVE ITINERARY ITEM / MEETING AGENDA:\nTitle: {active_task['title']}\nDescription: {active_task['description']}\nPriority: {active_task['priority'].upper()}\nStatus: {active_task['status'].upper()}"
        if active_task.get("blocked_reason"):
            ctx += f"\nLast Feedback / Failure Reason: {active_task['blocked_reason']}"
        if active_task.get("attempt_count"):
            ctx += f"\nAttempt Count: {active_task['attempt_count']}"
        return ctx

    def _build_journal_context(self, latest_journal: Optional[str]) -> str:
        if not latest_journal:
            return ""
        truncated = latest_journal[:300] + "... [truncated]" if len(latest_journal) > 300 else latest_journal
        return f"\n\n### YOUR LATEST TIMESTAMPED SELF-JOURNAL (PRE-NAP PERSPECTIVE):\n{truncated}"

    def _build_spec_context(self, model_id: str) -> str:
        own_spec = self.memory_manager.get_spec_file(model_id)
        if own_spec and len(own_spec) > 500:
            own_spec = own_spec[:500] + "... [truncated]"
        return f"\n\n### YOUR PERSONAL SPEC NOTEBOOK:\n{own_spec if own_spec else '(Empty)'}"

    async def _execute_task_test_run(self, tester_cfg: Dict[str, Any], task: Dict[str, Any]) -> None:
        """Deterministically runs the active task's file in its author's sandbox and
        advances the task to completed/failed - the Tester's own turn doesn't need to
        successfully emit a run_tests action for this to happen."""
        filename = task.get("filename")
        author_id = task.get("author_bot_id")
        if not filename or not author_id:
            return

        if filename.endswith(".py"):
            # Completeness gate before the run. "python file.py exits 0" is a weak pass: a
            # truncated file whose `if __name__ == "__main__":` block never made it out of
            # the model imports cleanly, does nothing, exits 0, and used to be marked
            # completed - a silent false pass on a half-written program.
            read_res = self.tool_manager.read_file(filename, bot_id=author_id)
            source = read_res.get("content", "") if read_res.get("success") else ""
            if not self._source_parses(source):
                self.memory_manager.update_itinerary_task(task["id"], {
                    "status": "failed",
                    "blocked_reason": f"`{filename}` does not parse - the file looks incomplete or truncated.",
                    "attempt_count": task.get("attempt_count", 0) + 1
                })
                self.memory_manager.add_entry(tester_cfg["name"], f"`{filename}` failed to parse; not a valid Python file.")
                return
            if "__main__" not in source and "def main" not in source:
                self.memory_manager.update_itinerary_task(task["id"], {
                    "status": "failed",
                    "blocked_reason": (
                        f"`{filename}` parses but has no entry point (no `main()` and no "
                        "`if __name__ == \"__main__\":` block) - it exits without doing anything. "
                        "The file is most likely truncated."
                    ),
                    "attempt_count": task.get("attempt_count", 0) + 1
                })
                self.memory_manager.add_entry(tester_cfg["name"], f"`{filename}` has no entry point; treating as incomplete.")
                return
            exec_res = self.tool_manager.run_python(filepath=filename, bot_id=author_id)
        else:
            exec_res = {"success": True, "stderr": ""}

        if exec_res.get("success"):
            self.memory_manager.update_itinerary_task(task["id"], {"status": "completed"})
            self.memory_manager.add_entry(tester_cfg["name"], f"Test run passed for `{filename}`.")
        else:
            stderr = (exec_res.get("stderr") or "unknown error")[:200]
            self.memory_manager.update_itinerary_task(task["id"], {
                "status": "failed",
                "blocked_reason": f"Test run failed: {stderr}",
                "attempt_count": task.get("attempt_count", 0) + 1
            })
            self.memory_manager.add_entry(tester_cfg["name"], f"Test run failed for `{filename}`: {stderr}")

    async def step_model_turn(self, model_id: str) -> Dict[str, Any]:
        model_cfg = self.models.get(model_id)
        if not model_cfg:
            raise KeyError(f"Model '{model_id}' is not in the chat room.")
        if not model_cfg["enabled"]:
            raise ValueError(f"Model '{model_cfg['name']}' is disabled and cannot take a turn.")

        # Manual per-turn usage (e.g. admin-driven single turns, or any caller that isn't
        # run_autonomous_loop) previously never ran VRAM lifecycle management, so GGUF
        # models only ever lazy-loaded on first use and were never offloaded, exhausting
        # VRAM over a long manual session. Mirror the Auto-mode call site here. Wrapped
        # defensively - VRAM management is a resource-optimization side effect and must
        # never prevent a turn from completing on single-model/no-GPU setups.
        try:
            self.manage_vram_allocation(model_id)
        except Exception as e:
            logger.warning("VRAM allocation management skipped for %s: %s", model_id, e)

        current_phase = self.memory_manager.get_phase()

        # Architect escalation-model swap: when execution-phase routing has escalated to
        # the Architect because a task has failed repeatedly (status=failed,
        # attempt_count>=3 - see _select_execution_speaker), and the Architect has a
        # heavier "escalation_model_path" configured, temporarily swap that model into
        # VRAM for just this turn so the Architect reasons with more capacity, then
        # unload it afterward so the normal 4 role-models stay resident.
        escalation_active = False
        is_architect = self._is_supervisor(model_cfg)
        escalation_path = model_cfg.get("escalation_model_path")
        if is_architect and escalation_path:
            active_task_for_escalation = self._current_task()
            if (
                current_phase == "execution"
                and active_task_for_escalation
                and active_task_for_escalation.get("status") == "failed"
                and active_task_for_escalation.get("attempt_count", 0) >= 3
            ):
                try:
                    self.model_manager.load_gguf_model(
                        model_id,
                        escalation_path,
                        max_tokens=model_cfg.get("max_context_tokens") or DEFAULT_N_CTX,
                        mmproj_path=model_cfg.get("mmproj_path"),
                        force_device="gpu"
                    )
                    escalation_active = True
                    logger.info("Escalation model swapped in for Architect %s: %s", model_id, escalation_path)
                except Exception as e:
                    logger.warning("Escalation model swap for %s skipped: %s", model_id, e)

        # Evidence-Gated Critique enforcement: If Coder receives a critique, require tool evidence reference in previous messages.
        # An unsubstantiated critique (no traceback/test output/line reference) must not steer the Coder - it's swapped
        # out of the Coder's own context below instead of being silently accepted.
        role_lower = model_cfg.get("role", "").lower()
        unsupported_critique_id: Optional[str] = None
        if "coder" in role_lower and self.chat_history:
            last_msg = self.chat_history[-1]
            if "critic" in last_msg.get("role", "").lower():
                has_evidence = any(ev in last_msg["content"].lower() for ev in ["traceback", "pytest", "test", "file:", "line", "error:", "def ", "class "])
                if not has_evidence:
                    unsupported_critique_id = last_msg.get("id")
                    logger.info("Ignoring unsupported critique for Coder %s (no tool evidence cited)", model_id)

        memory_summary = self.memory_manager.get_memory_summary()
        latest_journal = self.memory_manager.get_model_latest_journal(model_id)

        # The scheduler pins exactly which task this turn is about. Without the pin a model
        # draining a batch would rebuild its context from "whatever is most urgent" each
        # turn and could answer about a sibling task instead of the one it was handed.
        active_task = self._current_task()
        is_first_turn = model_id not in self.spoken_models
        self.spoken_models.add(model_id)

        sys_prompt = self._build_system_prompt(model_id, model_cfg, current_phase, active_task, is_first_turn)
        ep_summary = self._build_episode_context(active_task)
        task_context = self._build_task_context(active_task)
        journal_context = self._build_journal_context(latest_journal)
        spec_context = self._build_spec_context(model_id)

        # Prefix Cache Optimization: Put stable content FIRST (system prompt, task details, personal spec)
        # and volatile context LAST (shared memory summary, episodes, journal)
        context_prompt = f"{sys_prompt}{task_context}{spec_context}\n\n### SHARED MEMORY SUMMARY:\n{memory_summary}{ep_summary}{journal_context}"

        # Check model token usage against limits - enforce context reset if exceeded to prevent degradation
        tokens_used = self.memory_manager.state.get("tokens_used", {}).get(model_id, 0)
        max_tokens = model_cfg.get("max_context_tokens", 4096)
        context_was_reset = False
        if tokens_used > (max_tokens * 0.75):
            # Record automatic LLM-driven/rule self-journal and reset token counter for model
            auto_journal = f"Context refresh checkpoint for {model_cfg['name']} ({model_cfg['role']}): Active project '{self.memory_manager.get_project_id()}'. Current task: '{active_task['title'] if active_task else 'General Discussion'}'. Key contributions logged to shared memory."
            self.memory_manager.record_model_nap(model_id, auto_journal)
            self.memory_manager.state["tokens_used"][model_id] = 0
            self.memory_manager.save_memory()
            context_was_reset = True

            # Construct ~50-token high-level project summary
            proj_summary_50_tokens = (
                f"PROJECT INDEX ({self.memory_manager.get_project_id()}): "
                f"Phase: {current_phase.upper()}. Task: {active_task['title'] if active_task else 'General Alignment'}. "
                f"Goal: Execute requirements with full audit trail, indexed memory logging, and zero context bloat."
            )
            context_prompt = f"{sys_prompt}\n\n### 📌 HIGH-LEVEL PROJECT INDEX (~50 TOKENS):\n{proj_summary_50_tokens}\n\n### SHARED INDEXED MEMORY SUMMARY:\n{memory_summary}{ep_summary}{task_context}\n\n### 🔄 REFRESHED CONTEXT SAVE FILE:\n{auto_journal}\nPlease continue your concise contribution."

        if context_was_reset or current_phase == "execution":
            # Clean context refresh mode: minimal message context to prevent token overload.
            # In execution phase, Critic/Tester need to see the actual FILE THE TASK
            # PRODUCED, which was written into the author's own sandboxed workspace, not
            # their own (empty) one - bot_workspace_write is per-bot-id, so listing/reading
            # with the current speaker's own model_id here only ever showed their own files.
            review_target_id = model_id
            focus_filename = None
            if current_phase == "execution" and active_task and active_task.get("author_bot_id"):
                review_target_id = active_task["author_bot_id"]
                focus_filename = active_task.get("filename")

            file_manifest = self.tool_manager.list_files(".", bot_id=review_target_id)
            manifest_str = ", ".join(file_manifest.get("files", [])[:15]) if isinstance(file_manifest, dict) else ""

            file_content_block = ""
            if focus_filename:
                read_res = self.tool_manager.read_file(focus_filename, bot_id=review_target_id)
                if read_res.get("success"):
                    contents = read_res.get("content", "")
                    if len(contents) > 3000:
                        contents = contents[:3000] + "\n... [truncated]"
                    file_content_block = f"\n\nContents of `{focus_filename}` (written by {review_target_id}):\n```\n{contents}\n```"

            # A generic "provide your next concise contribution" order is why small models
            # answered execution turns with a one-line restatement of the task. Give the
            # speaker the actual imperative for its stage of the pipeline instead.
            task_title = active_task['title'] if active_task else 'Execute or discuss requirements'
            task_desc = (active_task.get('description') or '').strip() if active_task else ''
            role_l = (model_cfg.get("role") or "").lower()
            blocked = (active_task or {}).get("blocked_reason") or ""
            attempts = (active_task or {}).get("attempt_count", 0) if active_task else 0
            if "coder" in role_l:
                order = (
                    f"Write the complete working code for: {task_title}."
                    " Reply with one fenced code block whose first line is `# filename: <name>`."
                    " Prose alone is not an acceptable answer."
                )
                # A rejected task used to come back to the Coder with the identical prompt it
                # had already answered, so a low-temperature model regenerated its previous
                # file byte-for-byte and the dedup guard skipped the turn - the pipeline
                # looked busy and went nowhere. Put the rejection in front of it as the job.
                if blocked:
                    order = (
                        "Your previous version of this file was REJECTED. Fix it.\n"
                        f"Reason given: {blocked}\n\n"
                        "Re-emit the COMPLETE corrected file in one fenced code block whose first line is "
                        "`# filename: <name>`. The new version must differ from the one above - address the "
                        "stated reason specifically. Do not reply with prose, an apology, or an unchanged file."
                    )
            elif "critic" in role_l:
                order = (
                    "Review the file contents shown above against the task. Cite specific evidence,"
                    " then end with APPROVE or REJECT on its own line."
                )
            elif "tester" in role_l or "debugger" in role_l or "refiner" in role_l:
                order = (
                    "State in one line what you are verifying about the file above."
                    " The harness will run it and report the real result."
                )
            else:
                # Ends on the concrete question, not on a template the model can complete.
                order = (
                    "Decide the single next piece of work for this project and plan it. "
                    "Write one short paragraph explaining the choice, then finish with the task "
                    "directive line described in your output contract. "
                    "What is the one thing the Coder should build next?"
                )
            # The execution-phase context replaced chat history wholesale, which meant the
            # Admin's actual project brief never reached the model. For the Coder/Critic/
            # Tester that was survivable - their prompt still carried a task description and
            # the file under review. For the Architect with no open task it was fatal: task
            # context empty, workspace manifest empty, brief discarded, so the single richest
            # thing in the room (what the user asked for) was the one thing it couldn't see.
            # It generated nothing because it was given nothing. Carry the standing brief.
            brief_block = ""
            admin_msgs = [m for m in self.chat_history if m.get("is_admin") and not str(m.get("sender", "")).startswith("System")]
            if admin_msgs:
                brief = admin_msgs[-1]["content"].strip()
                if len(brief) > 1200:
                    brief = brief[:1200] + "... [truncated]"
                brief_block = f"\n\nStanding instruction from the Admin:\n{brief}"

            recent_msgs = [
                {
                    "role": "user",
                    "content": (
                        f"[EXECUTION TURN]\nTask: {task_title}"
                        + (f"\nDetails: {task_desc}" if task_desc else "")
                        + f"\nWorkspace Files: {manifest_str}{file_content_block}"
                        + brief_block
                        + f"\n\n{order}"
                    )
                }
            ]
        else:
            # Discussion phase: Truncate messages to prevent prompt overflow & context degradation
            # Limit each message to max 300 characters and include at most the last 3 turns to prevent echo loops
            from backend.sanitizer import sanitize_message_content
            recent_msgs = []
            for m in self.chat_history[-3:]:
                if unsupported_critique_id and m.get("id") == unsupported_critique_id:
                    recent_msgs.append({
                        "role": "user",
                        "content": "[A critique was received but cited no verifiable evidence - no traceback, test output, or line reference. Disregard it and continue with your prior plan unless a follow-up critique cites concrete evidence.]"
                    })
                    continue
                clean_c = sanitize_message_content(m["content"])
                if len(clean_c) > 300:
                    clean_c = clean_c[:300] + "... [truncated]"
                role = "user" if m["is_admin"] else "assistant"
                recent_msgs.append({
                    "role": role,
                    "content": clean_c
                })
            if not recent_msgs:
                recent_msgs = [{"role": "user", "content": "Please introduce your perspective on the current project."}]

        self.set_model_live_status(model_id, f"Formulating response as {model_cfg.get('role', 'Participant')}")

        # Determine if two-call split should be used: call 1 for unconstrained prose, call 2 for grammar-constrained action emission
        is_execution_phase = (current_phase == "execution")
        role_lower = model_cfg.get("role", "").lower()
        use_two_call = is_execution_phase or any(r in role_lower for r in ["coder", "tester", "debugger", "refiner"])
        # The Architect's output is a planning directive, not code. Showing it the
        # grammar-constrained action schema on a second call made small models mirror that
        # JSON back as their actual answer (and it then got saved as a source file). It
        # plans in prose with [UPDATE_TASK: ...] brackets instead, which they emit reliably.
        if "architect" in role_lower or "planner" in role_lower:
            use_two_call = False

        response_text = ""
        actions_list = []

        # Escalate sampling temperature with repeated failed attempts. At the low
        # temperatures the Coder/Tester run at, an unchanged prompt yields an unchanged
        # generation - so a retry that must produce something *different* needs headroom.
        if current_phase == "execution" and active_task and active_task.get("attempt_count", 0) > 0:
            bump = min(0.3, 0.1 * active_task["attempt_count"])
            model_cfg = {**model_cfg, "temperature": min(0.9, model_cfg.get("temperature", 0.7) + bump)}

        try:
            if use_two_call:
                # Call 1: Unconstrained prose contribution
                prose_response = await self.model_manager.generate_response(
                    model_config=model_cfg,
                    system_prompt=context_prompt,
                    messages=recent_msgs
                )
                response_text = prose_response

                # Call 2: Low-temperature grammar-constrained action emission for what was just said
                action_sys_prompt = (
                    f"You are the action parsing module for {model_cfg['name']}. "
                    "Analyze the response just emitted and extract all action tags or tool executions into structured JSON format."
                )
                action_msgs = [
                    {"role": "assistant", "content": prose_response},
                    {"role": "user", "content": "Emit the JSON action object corresponding to your message."}
                ]
                schema = self.model_manager.get_action_json_schema()
                try:
                    action_json_str = await self.model_manager.generate_response(
                        model_config=model_cfg,
                        system_prompt=action_sys_prompt,
                        messages=action_msgs,
                        temperature=0.1,
                        response_schema=schema
                    )
                    import json
                    parsed = json.loads(action_json_str)
                    actions_list = parsed.get("actions", [])
                except Exception as json_err:
                    logger.debug("Grammar-constrained action call parsing failed or empty for %s: %s", model_id, json_err)
            else:
                # Single call for general discussion
                response_text = await self.model_manager.generate_response(
                    model_config=model_cfg,
                    system_prompt=context_prompt,
                    messages=recent_msgs
                )

            # Some GGUFs (reasoning-tuned ones especially) intermittently return an entirely
            # empty completion for a given prompt shape while answering a different one fine.
            # A silently empty turn is indistinguishable from a dead model and stalls the
            # room, so retry once with a plain, minimal prompt before giving up on the turn.
            if not response_text.strip():
                logger.info("Empty generation from %s; retrying once with a simplified prompt", model_id)
                retry_task = active_task["title"] if active_task else "the current project"
                try:
                    response_text = await self.model_manager.generate_response(
                        model_config={**model_cfg, "temperature": max(0.6, model_cfg.get("temperature", 0.7))},
                        system_prompt=f"You are {model_cfg['name']}, the {model_cfg['role']} on a software team.",
                        messages=[{
                            "role": "user",
                            "content": (
                                f"In two or three sentences, what is your next concrete step on {retry_task}? "
                                "Answer in plain prose."
                            )
                        }]
                    )
                except Exception as retry_err:
                    logger.warning("Empty-generation retry for %s also failed: %s", model_id, retry_err)
        except ModelInvocationError as e:
            logger.error("Turn failed for %s: %s", model_id, e)
            self.set_model_live_status(model_id, f"Error: {e}")
            self.last_speaker_id = model_id
            if escalation_active:
                # Swap-out on the error path too - never leave the heavier escalation
                # model resident once its one-off turn is over.
                try:
                    self.model_manager.unload_gguf_model(model_id)
                except Exception as unload_err:
                    logger.warning("Escalation model unload for %s skipped: %s", model_id, unload_err)
            return self.add_chat_message(
                sender=f"System / Model Error ({model_cfg['name']})",
                role="System",
                content=f"⚠️ [MODEL ERROR] @{model_cfg['name']} ({model_cfg['role']}) could not respond: {e}",
                is_admin=True,
                model_id=model_id
            )
        else:
            self.set_model_live_status(model_id, "Idle / Live in Chat")
            if escalation_active:
                # Turn is done - unload the heavier escalation model so the normal
                # 4 role-models can stay resident in VRAM again.
                try:
                    self.model_manager.unload_gguf_model(model_id)
                except Exception as unload_err:
                    logger.warning("Escalation model unload for %s skipped: %s", model_id, unload_err)

        # Deduplication: If new message is >=90% similar to model's previous message, skip appending turn
        # Compare raw generation to raw generation. Comparing the new raw text against the
        # previously *displayed* message was an apples-to-oranges match, and it also meant two
        # consecutive unusable turns (both rendered as the same synthesized "still working on
        # it" line) were not caught while two fine turns could be.
        last_prev = self._last_raw_response.get(model_id)
        self._last_raw_response[model_id] = response_text
        if last_prev:
            sim = self._calculate_similarity(response_text, last_prev)
            if sim >= 0.90:
                logger.info("Skipping turn for %s due to high message repetition (similarity: %.2f)", model_id, sim)
                # A dedup skip used to leave task state completely untouched, so the
                # execution router re-selected the same stuck model on the very next turn
                # forever - the room looked alive but the pipeline was deadlocked. Count the
                # repeat against the task so it escalates to the Architect after 3.
                if current_phase == "execution" and active_task:
                    try:
                        fresh = next(
                            (t for t in self.memory_manager.get_task_itinerary() if t["id"] == active_task["id"]),
                            active_task
                        )
                        self.memory_manager.update_itinerary_task(fresh["id"], {
                            "status": "failed",
                            "blocked_reason": f"{model_cfg['name']} ({model_cfg['role']}) repeated itself without producing usable output.",
                            "attempt_count": fresh.get("attempt_count", 0) + 1
                        })
                    except Exception as e:
                        logger.warning("Could not record dedup stall against task: %s", e)
                # Persist the skip (not just return it) and advance last_speaker_id - otherwise the
                # room goes silent: the frontend never sees this turn, and round-robin fallback can
                # get stuck re-selecting the same repeating model forever.
                self.last_speaker_id = model_id
                return self.add_chat_message(
                    sender=f"System / Dedup ({model_cfg['name']})",
                    role="System",
                    content=f"ℹ️ @{model_cfg['name']} had nothing new to add this turn (repeated its last message) — skipping.",
                    is_admin=True,
                    model_id=model_id
                )

        self.memory_manager.update_token_usage(model_id, len(response_text.split()))

        # Process structured JSON actions first
        for act in actions_list:
            act_type = act.get("type")
            if act_type == "ready_for_execution":
                self._request_execution_phase(model_id, model_cfg)
            elif act_type == "request_discussion":
                self.memory_manager.add_entry(model_cfg["name"], "Requested return to Discussion Phase due to ambiguity.")
            elif act_type == "log_memory":
                payload = act.get("payload", "")
                if payload:
                    self.memory_manager.add_entry(model_cfg["name"], payload)
            elif act_type == "update_spec":
                payload = act.get("payload", "")
                if payload:
                    self.memory_manager.update_spec_file(model_id, payload)
            elif act_type == "journal":
                payload = act.get("payload", "")
                if payload:
                    self.memory_manager.record_model_nap(model_id, payload)
            elif act_type == "update_task":
                task_id = act.get("task_id")
                status = act.get("status", "pending")
                title = (act.get("title") or "").strip()
                if task_id:
                    updates = {"status": status}
                    if title:
                        updates["title"] = title
                    self.memory_manager.update_itinerary_task(task_id, updates)
                elif len(title) >= 8 and title.lower() != "updated task":
                    # A model that emits update_task with no usable title used to mint a new
                    # itinerary entry called "Updated Task" every single turn, burying the real
                    # backlog under dozens of empty placeholders. Require a real title.
                    self.memory_manager.add_itinerary_task(
                        title=title,
                        description=(act.get("payload") or act.get("description") or title),
                        assigned_model=model_id
                    )
                else:
                    logger.info("Ignoring update_task action from %s with no usable title", model_id)
            elif act_type in ("run_tests", "run_python"):
                # Previously declared in the action schema but never handled - a model could
                # ask for a test run and nothing would happen. Delegate to the same
                # deterministic execution the Tester's turn triggers automatically below,
                # so this is a no-op if that's already covered the current task this turn.
                focus_task = self._current_task()
                if focus_task and focus_task.get("status") == "needs_test":
                    await self._execute_task_test_run(model_cfg, focus_task)

        # Legacy Bracket Tag Fallback Processing
        if "[READY_FOR_EXECUTION]" in response_text:
            self._request_execution_phase(model_id, model_cfg)
        elif "[REQUEST_DISCUSSION]" in response_text:
            self.memory_manager.add_entry(model_cfg["name"], "Requested return to Discussion Phase due to ambiguity.")
            if self.memory_manager.get_phase() == "execution":
                self.memory_manager.set_phase("discussion")
                self.memory_manager.reset_plan_gate()
                self.add_chat_message(
                    sender="System / Plan Gate",
                    role="System",
                    content=(
                        f"↩️ @{model_cfg['name']} sent the room back to discussion. The planning "
                        f"gate restarts at the Architect's plan."
                    ),
                    is_admin=True
                )

        # Walk the pre-execution planning gate. Runs after the readiness directives above so
        # an Architect that legitimately opened Execution this turn isn't then re-read as a
        # plan proposal for a phase the room already left.
        if self.memory_manager.get_phase() == "discussion":
            self._advance_plan_gate(model_id, model_cfg, response_text)

        if "[UPDATE_CONFIG:" in response_text:
            self._run_directive(
                "UPDATE_CONFIG", model_cfg,
                lambda payload: self._apply_config_directive(model_id, model_cfg, payload),
                self._directive_payload(response_text, "[UPDATE_CONFIG:")
            )

        if "[UPDATE_SPEC:" in response_text:
            self._run_directive(
                "UPDATE_SPEC", model_cfg,
                lambda payload: self.memory_manager.update_spec_file(model_id, payload),
                self._directive_payload(response_text, "[UPDATE_SPEC:")
            )

        if "[UPDATE_TASK:" in response_text:
            self._run_directive(
                "UPDATE_TASK", model_cfg,
                lambda payload: self._apply_task_directive(model_id, model_cfg, payload),
                self._directive_payload(response_text, "[UPDATE_TASK:")
            )

        if "[SEARCH_HF:" in response_text:
            query = self._directive_payload(response_text, "[SEARCH_HF:")
            if query is None:
                self._report_directive_failure("SEARCH_HF", model_cfg, "missing closing ']'")
            else:
                hf_res = await self.tool_manager.search_huggingface(query)
                if hf_res.get("success"):
                    res_summary = ", ".join([m["model_id"] for m in hf_res.get("models", [])[:3]])
                    self.memory_manager.add_entry(
                        model_cfg["name"],
                        f"HuggingFace search for '{query}' returned candidate models: {res_summary}"
                    )
                else:
                    self._report_directive_failure(
                        "SEARCH_HF", model_cfg,
                        f"HuggingFace search for '{query}' failed: {hf_res.get('error', 'unknown error')}"
                    )

        if "[JOURNAL:" in response_text:
            self._run_directive(
                "JOURNAL", model_cfg,
                lambda payload: self.memory_manager.record_model_nap(model_id, payload),
                self._directive_payload(response_text, "[JOURNAL:")
            )

        if "[LOG_TO_MEMORY:" in response_text:
            self._run_directive(
                "LOG_TO_MEMORY", model_cfg,
                lambda payload: self.memory_manager.add_entry(model_cfg["name"], payload),
                self._directive_payload(response_text, "[LOG_TO_MEMORY:")
            )

        if "[SAVE_NOTE:" in response_text:
            self._run_directive(
                "SAVE_NOTE", model_cfg,
                lambda payload: self.memory_manager.add_note_chunk(model_id, payload, title=f"Note by {model_cfg['name']}"),
                self._directive_payload(response_text, "[SAVE_NOTE:")
            )

        if "[REQUEST_NAP]" in response_text and "[JOURNAL:" not in response_text:
            self.memory_manager.record_model_nap(model_id, f"{model_cfg['name']} completed a context nap.")

        # Architect backstop. The Critic and Tester already have deterministic fallbacks for
        # when a small model won't emit the structured action; the Architect had none, so a
        # turn where it correctly described the next piece of work in prose but omitted the
        # [UPDATE_TASK: ...] line left the itinerary empty and the room with nothing to do.
        # Promote its own plan into the task rather than discarding a good answer over syntax.
        if (
            current_phase == "execution"
            and not active_task
            and ("architect" in role_lower or "planner" in role_lower)
            and "[UPDATE_TASK" not in response_text.upper()
        ):
            plan = self._strip_directive_tags(response_text).strip()
            plan = re.sub(r"```[\s\S]*?```", "", plan).strip()
            if len(plan) >= 40 and not self._looks_like_garbage(plan) and not self._is_prompt_echo(plan):
                first_sentence = re.split(r"(?<=[.!?])\s+", plan)[0].strip()
                title = (first_sentence[:80].rstrip(" .") or "Next implementation step")
                self.memory_manager.add_itinerary_task(
                    title=title,
                    description=plan[:1500],
                    priority="high",
                    assigned_model=None
                )
                created = self.memory_manager.get_active_task()
                if created:
                    self.memory_manager.update_itinerary_task(created["id"], {"status": "in_progress"})
                logger.info("Architect %s omitted [UPDATE_TASK]; promoted its prose plan to a task", model_id)

        # Deterministic task-state backstop: small models often don't reliably emit the
        # structured update_task/run_tests actions, so infer the pipeline transition from
        # simple keyword cues (Critic) or just run the tests ourselves (Tester) instead of
        # letting the state machine stall waiting on an action call that never comes.
        if current_phase == "execution" and active_task:
            fresh_task = next(
                (t for t in self.memory_manager.get_task_itinerary() if t["id"] == active_task["id"]),
                active_task
            )
            resp_lower = response_text.lower()

            if "critic" in role_lower and fresh_task.get("status") == "needs_review":
                # Verdict resolution, most-trustworthy signal first. A small Critic that
                # rambles past its token budget used to leave the task on needs_review
                # forever, and the execution router re-picked the Critic every turn - the
                # room stayed busy while the pipeline was frozen. Every path below now
                # moves the task off needs_review.
                verdict = self._resolve_verdict(response_text)

                if verdict == "reject":
                    self.memory_manager.update_itinerary_task(fresh_task["id"], {
                        "status": "failed",
                        "blocked_reason": f"Critic rejected: {response_text[:200]}",
                        "attempt_count": fresh_task.get("attempt_count", 0) + 1
                    })
                else:
                    # 3. No verdict at all (ran out of tokens mid-review, or answered with
                    #    pure prose). Advance to the Tester rather than deadlocking: an
                    #    actual test run is stronger evidence than an unfinished opinion.
                    if verdict is None:
                        logger.info("Critic %s produced no verdict; advancing task to needs_test", model_id)
                    self.memory_manager.update_itinerary_task(fresh_task["id"], {"status": "needs_test"})

            elif ("tester" in role_lower or "debugger" in role_lower) and fresh_task.get("status") == "needs_test":
                await self._execute_task_test_run(model_cfg, fresh_task)

        # Auto-extract markdown code blocks and save into model workspace, tracking a diff summary
        # for each file so the chat can describe the change instead of dumping raw code as prose.
        code_file_updates: List[Dict[str, Any]] = []

        # The supervisor has no file-write permission. This used to be prompt-only ("Do not
        # write the code yourself") while the extractor below was role-blind, so an Architect
        # that ignored the instruction silently got a file written into its own sandbox -
        # work no Coder owned and no Critic reviewed. Enforce it here instead of asking.
        if self._is_supervisor(model_cfg) and "```" in response_text:
            logger.info(
                "Discarding %d code block(s) from supervisor %s - Architect turns do not write files",
                response_text.count("```") // 2, model_id
            )
            self.add_chat_message(
                sender="System / Role Guard",
                role="System",
                content=(
                    f"🛑 [SUPERVISOR WROTE CODE] @{model_cfg['name']} emitted code in an Architect "
                    f"turn. It was discarded, not saved. The Architect assigns work with "
                    f"[UPDATE_TASK: ...]; a Coder implements it."
                ),
                is_admin=True
            )
        elif "```" in response_text:
            parts = response_text.split("```")
            for i in range(1, len(parts), 2):
                block = parts[i]
                lines = block.splitlines()
                if not lines:
                    continue
                first_line = lines[0].strip()
                # Check for comment filename header like `# filename: example.py` or `// filename: index.ts`
                fn = f"generated_code_{int(time.time()*1000)}_{i//2}.py"
                code_body = block
                if len(lines) > 1 and ("filename:" in lines[0] or "filename:" in lines[1]):
                    for line in lines[:2]:
                        if "filename:" in line:
                            fn = line.split("filename:")[-1].strip().strip("`*#/")
                    code_body = "\n".join(lines[1:])
                elif first_line and not any(char in first_line for char in " =():[]{}#"):
                    # First line is language identifier (e.g., python, ts)
                    code_body = "\n".join(lines[1:])
                    fl_lower = first_line.lower()
                    if "py" in fl_lower:
                        ext = ".py"
                    elif any(tag in fl_lower for tag in ("js", "ts", "javascript", "typescript", "jsx", "tsx")):
                        ext = ".js"
                    elif any(tag in fl_lower for tag in ("html", "htm")):
                        ext = ".html"
                    elif "css" in fl_lower:
                        ext = ".css"
                    elif "json" in fl_lower:
                        ext = ".json"
                    else:
                        ext = ".txt"
                    fn = f"generated_code_{int(time.time()*1000)}{ext}"

                # A model that answers the prose call with the action-schema JSON (small
                # models mirror whatever structure they were last shown) used to get that
                # JSON written into its workspace as a source file - real-looking artifacts
                # that are actually just the model echoing the tool schema back.
                if self._is_action_schema_echo(code_body):
                    logger.info("Discarding action-schema JSON echoed as a code block by %s", model_id)
                    continue

                if code_body.strip():
                    prev_res = self.tool_manager.read_file(fn, bot_id=model_id)
                    prev_content = prev_res.get("content", "") if prev_res.get("success") else ""
                    write_res = self.tool_manager.bot_workspace_write(bot_id=model_id, filepath=fn, content=code_body)
                    diff_lines = list(difflib.unified_diff(prev_content.splitlines(), code_body.splitlines(), lineterm=""))
                    added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
                    removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
                    code_file_updates.append({
                        "filename": fn,
                        "added": added,
                        "removed": removed,
                        "is_new": not prev_content.strip()
                    })
                    # The Coder writes code as chat-fenced blocks (per its prompt), not via the
                    # explicit bot_workspace_write tool call - so without this, the sandbox
                    # refinement loop (Critic/fixer iteration, test-and-repair) never ran for the
                    # normal Coder workflow, only for the rarely-used explicit tool path. Gate on
                    # the write having happened at all (not on "success", which here means
                    # "syntax already valid" - gating on that would skip refinement for exactly
                    # the broken files it exists to repair).
                    if "bytes_written" in write_res and fn.endswith(".py"):
                        asyncio.create_task(self.trigger_sandbox_refinement_loop(bot_id=model_id, filepath=fn))

                    # In execution phase, a Coder producing a file for the active task hands
                    # it to the Critic next - stamp attribution on the task itself so the
                    # state-machine router (and the next speaker's own context) knows which
                    # file and whose workspace to look at.
                    if current_phase == "execution" and active_task and "coder" in role_lower:
                        self.memory_manager.update_itinerary_task(active_task["id"], {
                            "status": "needs_review",
                            "filename": fn,
                            "author_bot_id": model_id
                        })

        self._log_raw_turn(model_cfg, current_phase, response_text)
        display_content = self._build_chat_display_text(response_text, model_cfg, code_file_updates)

        msg = self.add_chat_message(
            sender=model_cfg["name"],
            role=model_cfg["role"],
            content=display_content,
            is_admin=False,
            model_id=model_id
        )

        self.last_speaker_id = model_id
        return msg

    @staticmethod
    def _source_parses(source: str) -> bool:
        """True when `source` is syntactically valid Python (used as a hard gate on repairs)."""
        if not source or not source.strip():
            return False
        try:
            import ast
            ast.parse(source)
            return True
        except (SyntaxError, ValueError):
            return False

    def _classify_error(self, stderr: str, syntax_error: Optional[str] = None) -> str:
        err_str = (syntax_error or "") + "\n" + (stderr or "")
        if "SyntaxError" in err_str:
            return "SyntaxError"
        elif "NameError" in err_str:
            return "NameError"
        elif "ImportError" in err_str or "ModuleNotFoundError" in err_str:
            return "ImportError"
        elif "AssertionError" in err_str:
            return "AssertionError"
        elif err_str.strip():
            return "RuntimeError"
        return "None"

    async def trigger_sandbox_refinement_loop(self, bot_id: str, filepath: str) -> Dict[str, Any]:
        """Out-of-chat self-refinement feedback loop.

        Runs Python/pytest on newly written workspace file. On SyntaxError/NameError/ImportError,
        runs isolated repair turns with best-of retention (early stopping).
        Escalates AssertionError or persistent failure to cloud model/human alert.
        """
        bot_dir = self.tool_manager.get_bot_workspace_dir(bot_id)
        full_file_path = os.path.abspath(os.path.join(bot_dir, filepath))

        # Check initial syntax first
        syntax_check = self.tool_manager.validate_file_syntax(full_file_path)
        if syntax_check.get("valid"):
            exec_res = self.tool_manager.run_python(filepath=filepath, bot_id=bot_id)
            if exec_res.get("success"):
                self.memory_manager.add_entry(
                    author=f"Execution Feedback ({bot_id})",
                    content=f"✅ Sandbox execution succeeded for `{filepath}`."
                )
                return {"success": True, "refinement_needed": False}
            else:
                stderr = exec_res.get("stderr", "")
                error_cls = self._classify_error(stderr)
        else:
            stderr = syntax_check.get("error", "Syntax error")
            error_cls = "SyntaxError"
            exec_res = {"success": False, "stderr": stderr}

        stderr = exec_res.get("stderr", "")
        if not syntax_check.get("valid"):
            syntax_err = syntax_check.get("error", "")
            error_cls = "SyntaxError"
            stderr = f"{syntax_err}\n{stderr}"
        else:
            error_cls = self._classify_error(stderr)

        self.memory_manager.add_entry(
            author=f"Execution Feedback ({bot_id})",
            content=f"❌ Sandbox execution failed for `{filepath}` [Error Class: `{error_cls}`].\nTraceback snippet:\n```{stderr[:400]}```"
        )

        # Classify & Route
        # A file that PARSES but fails at runtime must not be auto-rewritten. The sandbox has
        # no TTY, no display and no network, so an interactive program (curses/pygame), a
        # server, or anything awaiting input "fails" here for purely environmental reasons.
        # The repair loop used to treat that as a broken file and overwrite a perfectly good
        # implementation with a truncated repair - which is how a complete snake.py ended up
        # cut off mid-line at `from _curses import`. Report the traceback as evidence and let
        # the Critic/Tester judge it instead.
        if syntax_check.get("valid") and error_cls not in ("SyntaxError",):
            self.memory_manager.add_entry(
                author=f"Execution Feedback ({bot_id})",
                content=f"`{filepath}` parses cleanly but exited non-zero in the sandbox [{error_cls}]. Not auto-repairing; recorded as review evidence."
            )
            self.add_chat_message(
                sender="System / Refinement Router",
                role="System",
                content=(
                    f"ℹ️ [SANDBOX RUN] `{filepath}` is syntactically valid but did not run to completion "
                    f"in the sandbox ({error_cls}). This is often environmental (no terminal/display). "
                    f"File left untouched.\nTraceback snippet:\n```{stderr[:400]}```"
                ),
                is_admin=True
            )
            return {"success": True, "refinement_needed": False, "error_class": error_cls, "auto_repair_skipped": True}

        if error_cls in ["AssertionError", "LogicError"]:
            # Logic error carries low traceback signal -> do not loop; escalate to human/cloud
            self.add_chat_message(
                sender="System / Refinement Router",
                role="System",
                content=f"🚨 [LOGIC ERROR ESCALATION] Execution of `{filepath}` failed with `{error_cls}`. Logic errors require architectural or spec clarification; auto-refinement loop bypassed. Please review or escalate to cloud model.",
                is_admin=True
            )
            return {"success": False, "error_class": error_cls, "escalated": True}

        # Auto-refine for SyntaxError / NameError / ImportError
        refiner_id = None
        # Locate Tester/Debugger model (the unified test+fix role - see model_refiner
        # default config), falling back to Critic then Coder if none is configured.
        for m_id, m in self.models.items():
            role_lower = m.get("role", "").lower()
            if "tester" in role_lower or "debugger" in role_lower:
                refiner_id = m_id
                break
        if not refiner_id:
            for m_id, m in self.models.items():
                if m.get("role", "").lower() in ["critic", "coder"]:
                    refiner_id = m_id
                    break
        if not refiner_id:
            refiner_id = bot_id

        refiner_cfg = self.models.get(refiner_id, self.known_models.get(refiner_id, {}))

        # Best-of retention
        read_res = self.tool_manager.read_file(filepath, bot_id=bot_id)
        best_artifact = read_res.get("content", "")
        best_error_count = len(stderr.splitlines()) if stderr else 100

        refined_successfully = False
        active_task = self.memory_manager.get_active_task()
        task_desc = (
            f"{active_task['title']}: {active_task['description']}" if active_task
            else f"Re-implement `{filepath}` to satisfy the project requirements."
        )

        # Set 1: 3 patch-tries against the existing artifact. If 3 consecutive repairs don't
        # work, don't keep patching a file that's fundamentally off - set 2 starts with a fresh
        # regeneration from the original requirements (not the broken code), then patches that.
        for set_num in range(1, 3):
            top_k_val = 40 if set_num == 1 else 20
            for try_num in range(1, 4):
                if set_num == 2 and try_num == 1:
                    prompt = (
                        f"A previous attempt at `{filepath}` failed 3 consecutive repair tries "
                        f"(last error class: {error_cls}). Don't patch it further - write a fresh "
                        f"implementation from the requirements below.\n\n"
                        f"Requirements: {task_desc}\n\n"
                        "Provide ONLY the full new code file without explanations or markdown wrappers."
                    )
                else:
                    prompt = (
                        f"Fix ONLY the error in `{filepath}`.\n"
                        f"Raw Traceback:\n{stderr[:1000]}\n\n"
                        f"Current File Content:\n{best_artifact}\n\n"
                        "Provide ONLY the full updated code file without explanations or markdown wrappers."
                    )

                repair_cfg = dict(refiner_cfg)
                repair_cfg["top_k"] = top_k_val
                repair_cfg["temperature"] = 0.1

                try:
                    repaired_code = await self.model_manager.generate_response(
                        model_config=repair_cfg,
                        system_prompt="You are an automated code repair agent. Fix ONLY the runtime/syntax error. Output clean code only.",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1
                    )
                except Exception as e:
                    logger.warning("Repair turn generation failed: %s", e)
                    continue

                # Strip markdown code blocks if present
                clean_repaired = repaired_code.strip()
                if clean_repaired.startswith("```"):
                    lines = clean_repaired.splitlines()
                    if len(lines) > 2:
                        clean_repaired = "\n".join(lines[1:-1])

                # Reject obviously-degenerate repairs BEFORE they touch the file. A repair turn
                # that runs out of tokens returns a truncated fragment; writing it and scoring
                # it by "fewer stderr lines" let that fragment win, permanently destroying a
                # working artifact.
                if not clean_repaired.strip():
                    logger.info("Discarding empty repair candidate for %s", filepath)
                    continue
                if best_artifact.strip() and len(clean_repaired) < 0.6 * len(best_artifact):
                    logger.info(
                        "Discarding truncated repair candidate for %s (%d chars vs best %d)",
                        filepath, len(clean_repaired), len(best_artifact)
                    )
                    continue

                # Write to bot sandbox
                self.tool_manager.bot_workspace_write(bot_id=bot_id, filepath=filepath, content=clean_repaired)
                bot_dir = self.tool_manager.get_bot_workspace_dir(bot_id)
                full_repair_path = os.path.abspath(os.path.join(bot_dir, filepath))
                eval_syntax = self.tool_manager.validate_file_syntax(full_repair_path)
                eval_res = self.tool_manager.run_python(filepath=filepath, bot_id=bot_id)

                if eval_syntax.get("valid") and (eval_res.get("success") or eval_res.get("returncode") in [0, None]):
                    refined_successfully = True
                    best_artifact = clean_repaired
                    self.memory_manager.add_entry(
                        author=f"Refinement Loop ({refiner_cfg.get('name', 'Refiner')})",
                        content=f"🎉 [REFINEMENT SUCCESS] Auto-repaired `{filepath}` on Set {set_num} Try {try_num}!"
                    )
                    break
                else:
                    new_stderr = eval_res.get("stderr", "") or eval_syntax.get("error", "")
                    new_error_count = len(new_stderr.splitlines()) if new_stderr else 999
                    # Early stopping / best-of retention. Syntax validity is a HARD gate, not
                    # part of the score: a shorter error message from a file that no longer
                    # parses is not an improvement, and scoring purely on stderr line count
                    # made exactly that swap look like progress.
                    best_parses = self._source_parses(best_artifact)
                    if best_parses and not eval_syntax.get("valid"):
                        self.tool_manager.bot_workspace_write(bot_id=bot_id, filepath=filepath, content=best_artifact)
                        logger.info("Rejected repair for %s: candidate no longer parses", filepath)
                    elif new_error_count < best_error_count:
                        best_artifact = clean_repaired
                        best_error_count = new_error_count
                        stderr = new_stderr
                    else:
                        # Revert to last best artifact
                        self.tool_manager.bot_workspace_write(bot_id=bot_id, filepath=filepath, content=best_artifact)

            if refined_successfully:
                break

        if not refined_successfully:
            # Revert to original passing or best artifact
            self.tool_manager.bot_workspace_write(bot_id=bot_id, filepath=filepath, content=best_artifact)
            self.add_chat_message(
                sender="System / Refinement Router",
                role="System",
                content=f"⚠️ [REFINEMENT EXHAUSTED] Auto-refinement loop could not resolve error for `{filepath}` after 2 sets of attempts. Escalate to cloud model or Admin.",
                is_admin=True
            )

        # Log per-turn outcome to shared memory
        outcome_log = {
            "timestamp": time.time(),
            "filepath": filepath,
            "error_class": error_cls,
            "success": refined_successfully,
            "bot_id": bot_id
        }
        self.memory_manager.state.setdefault("refinement_outcomes", []).append(outcome_log)
        self.memory_manager.save_memory()

        return {"success": refined_successfully, "error_class": error_cls}

    # Internal action-tag names that must never leak into what the user/admin reads as chat.
    _DIRECTIVE_TAG_RE = re.compile(
        r"\[(READY_FOR_EXECUTION|REQUEST_DISCUSSION|REQUEST_NAP|LOG_TO_MEMORY|JOURNAL|"
        r"UPDATE_CONFIG|UPDATE_SPEC|UPDATE_TASK|SEARCH_HF|SAVE_NOTE)(:[^\]]*)?\]",
        re.IGNORECASE
    )
    # Leftover JSON-fragment / punctuation-only garbage a small model sometimes emits
    # instead of prose (e.g. `, status:"in_progress"}]`).
    _GARBAGE_ONLY_RE = re.compile(r'^[\s,\{\}\[\]":\'\-_.]*$')

    # Section headers we inject into the model's own prompt (task context, spec notebook,
    # shared memory summary, context-refresh blocks). Small models sometimes just echo the
    # prompt back verbatim instead of answering it - if any of these show up in the model's
    # *output*, treat it as an echo, not a real update.
    _PROMPT_ECHO_MARKERS = (
        "ACTIVE ITINERARY ITEM", "YOUR PERSONAL SPEC NOTEBOOK", "SHARED MEMORY SUMMARY",
        "SYSTEM REFRESH", "HIGH-LEVEL PROJECT INDEX", "RECENT EPISODE CHECKPOINTS",
        "REFRESHED CONTEXT SAVE FILE", "YOUR LATEST TIMESTAMPED SELF-JOURNAL",
        "INDEXED MEMORY MODE", "EXECUTION TURN", "OUTPUT CONTRACT", "WORKSPACE FILES:"
    )

    @classmethod
    def _strip_directive_tags(cls, text: str) -> str:
        """Removes internal pipeline action tags so chat reads like a human update, not a control channel."""
        return cls._DIRECTIVE_TAG_RE.sub("", text).strip()

    @classmethod
    def _looks_like_garbage(cls, text: str) -> bool:
        """Flags empty / punctuation-only / stray-JSON-fragment output that small models occasionally emit."""
        stripped = text.strip()
        if not stripped:
            return True
        if cls._GARBAGE_ONLY_RE.match(stripped):
            return True
        if re.match(r'^[,\}\]]', stripped):
            return True
        return sum(c.isalpha() for c in stripped) < 3

    @classmethod
    def _is_prompt_echo(cls, text: str) -> bool:
        """Flags a response that is the injected prompt reflected back, not one that mentions it.

        Matching a marker anywhere in the text was too blunt: a model that legitimately says
        "per the output contract, here's the task" was discarded as garbage and replaced with
        a "nothing solid to share yet" placeholder, hiding real work. A genuine echo starts
        by replaying a prompt block, so only the opening of the response is checked."""
        head = text.strip()[:160].upper()
        return any(marker in head for marker in cls._PROMPT_ECHO_MARKERS)

    def _build_chat_display_text(
        self,
        raw_response: str,
        model_cfg: Dict[str, Any],
        code_file_updates: List[Dict[str, Any]]
    ) -> str:
        """Turns a model's raw turn output into what actually shows up in chat.

        Chat is a reflection of the pipeline, not a raw terminal: internal action tags are
        stripped, and code fences stay as a collapsible block, but the accompanying prose is
        checked for coherence. When a small model produces no usable prose (empty, or a stray
        JSON/punctuation fragment), a short coworker-style status line is synthesized instead
        of posting the garbage or a "here's my code" dump verbatim.
        """
        display_text = self._strip_directive_tags(raw_response)
        code_fences = re.findall(r"```[\s\S]*?```", display_text)
        prose_only = re.sub(r"```[\s\S]*?```", "", display_text).strip()

        if self._looks_like_garbage(prose_only) or self._is_prompt_echo(prose_only):
            if code_file_updates:
                bits = []
                for u in code_file_updates:
                    verb = "Created" if u["is_new"] else "Updated"
                    stat = f"+{u['added']}/-{u['removed']}" if not u["is_new"] else f"+{u['added']}"
                    bits.append(f"{verb} `{u['filename']}` ({stat})")
                prose_only = " ".join(bits) + "."
            else:
                # A turn whose entire content was action tags (very common for the Architect,
                # whose job is to emit [UPDATE_TASK: ...]) strips down to an empty string and
                # used to be reported as "nothing solid to share yet" - actively misleading,
                # since the pipeline did move. Describe what the directives actually did.
                did = self._describe_directives(raw_response)
                if did:
                    prose_only = did
                elif not raw_response.strip():
                    # An empty generation is a model/loader fault (bad chat template, immediate
                    # EOS), not a model "thinking". Reporting it as "still working through this
                    # one" made a completely dead model look like a busy one for a whole session.
                    prose_only = (
                        f"⚠️ @{model_cfg.get('name', 'this model')} returned an empty response. "
                        "This usually means the GGUF's chat template isn't being applied correctly — "
                        "try a different model for this role."
                    )
                else:
                    prose_only = f"{model_cfg.get('name', 'This teammate')} is still working through this one — nothing solid to share yet."

        if code_fences:
            return (prose_only + "\n\n" + "\n\n".join(code_fences)).strip()
        return prose_only

    # Keys that only ever appear in the grammar-constrained action object. A fenced block
    # containing these is the model reflecting the action schema, not code it wrote.
    _ACTION_SCHEMA_KEYS = frozenset({
        "actions", "task_updates", "execution_status", "memory_entries",
        "notes_summary", "log_memory", "update_spec", "ready_for_execution",
    })

    @classmethod
    def _is_action_schema_echo(cls, block: str) -> bool:
        """True when a fenced block is just the action-call JSON schema echoed back."""
        text = block.strip()
        if not text.startswith("{") and not text.lstrip().startswith("{"):
            # Allow a leading language tag line that the caller already stripped.
            brace = text.find("{")
            if brace == -1:
                return False
            text = text[brace:]
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return False
        if not isinstance(parsed, dict):
            return False
        return bool(cls._ACTION_SCHEMA_KEYS.intersection(parsed.keys()))

    # Class-level default; instances override it from self.storage_root in __init__ so a
    # test run doesn't append to the user's real turn log.
    RAW_TURN_LOG = os.path.join(DEFAULT_STORAGE_ROOT, "raw_turns.log")

    def _log_raw_turn(self, model_cfg: Dict[str, Any], phase: str, response_text: str) -> None:
        """Appends each model's unmodified generation to a rolling log.

        Chat deliberately shows a cleaned-up reflection of a turn, which makes a bad turn
        ("nothing solid to share yet") impossible to diagnose from the UI alone. This keeps
        the raw text on disk so the actual failure - empty output, prompt echo, a thinking
        block, a truncated file - is inspectable after the fact."""
        try:
            os.makedirs(os.path.dirname(self.RAW_TURN_LOG), exist_ok=True)
            if os.path.exists(self.RAW_TURN_LOG) and os.path.getsize(self.RAW_TURN_LOG) > 2_000_000:
                os.replace(self.RAW_TURN_LOG, self.RAW_TURN_LOG + ".1")
            with open(self.RAW_TURN_LOG, "a", encoding="utf-8") as f:
                f.write(
                    f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} | {model_cfg.get('name')} "
                    f"({model_cfg.get('role')}) | phase={phase} | {len(response_text)} chars =====\n"
                )
                f.write(response_text + "\n")
        except OSError as e:
            logger.debug("Could not write raw turn log: %s", e)

    _DIRECTIVE_DESCRIPTIONS = {
        "UPDATE_TASK": "updated the task itinerary",
        "UPDATE_SPEC": "updated their spec notebook",
        "UPDATE_CONFIG": "adjusted their model configuration",
        "LOG_TO_MEMORY": "logged a note to shared memory",
        "SAVE_NOTE": "saved an indexed note",
        "JOURNAL": "wrote a journal checkpoint",
        "SEARCH_HF": "searched Hugging Face",
        "READY_FOR_EXECUTION": "declared the team ready for execution",
        "REQUEST_DISCUSSION": "asked to return to discussion",
        "REQUEST_NAP": "took a context nap",
    }

    @classmethod
    def _describe_directives(cls, raw_response: str) -> str:
        """One-line, human-readable summary of the action tags a turn emitted.

        Used when a turn's visible prose is empty because it consisted only of directives -
        so chat reports the real effect rather than implying the model did nothing."""
        found = []
        for m in cls._DIRECTIVE_TAG_RE.finditer(raw_response or ""):
            desc = cls._DIRECTIVE_DESCRIPTIONS.get(m.group(1).upper())
            if desc and desc not in found:
                found.append(desc)
        if not found:
            return ""
        if len(found) == 1:
            body = found[0]
        else:
            body = ", ".join(found[:-1]) + " and " + found[-1]
        return f"Took no chat action this turn beyond the pipeline: {body}."

    @staticmethod
    def _directive_payload(response_text: str, opening_token: str) -> Optional[str]:
        """Extracts an inline directive's payload, or None when the directive is unterminated."""
        start = response_text.find(opening_token) + len(opening_token)
        end = response_text.find("]", start)
        if end == -1:
            return None
        return response_text[start:end].strip()

    def _report_directive_failure(self, directive: str, model_cfg: Dict[str, Any], reason: str):
        """Announces a dropped directive so an unapplied instruction is never invisible."""
        logger.warning("Directive [%s] from %s failed: %s", directive, model_cfg.get("name"), reason)
        self.add_chat_message(
            sender="System / Directive Parser",
            role="System",
            content=f"⚠️ [DIRECTIVE IGNORED] @{model_cfg.get('name')}'s [{directive}] directive was not applied: {reason}",
            is_admin=True
        )

    def _run_directive(
        self,
        directive: str,
        model_cfg: Dict[str, Any],
        handler: Callable[[str], Any],
        payload: Optional[str]
    ):
        if payload is None:
            self._report_directive_failure(directive, model_cfg, "missing closing ']'")
            return
        try:
            handler(payload)
        except (SwarmChatError, ValueError, KeyError, TypeError, OSError) as e:
            self._report_directive_failure(directive, model_cfg, str(e))

    def _apply_config_directive(self, model_id: str, model_cfg: Dict[str, Any], payload: str):
        """Applies an [UPDATE_CONFIG: key=value, ...] directive, rejecting malformed payloads loudly."""
        numeric_keys = {"top_p": float, "temperature": float, "repeat_penalty": float, "top_k": int}
        updates: Dict[str, Any] = {}
        target_id = model_id
        unknown_keys: List[str] = []

        for part in [p.strip() for p in payload.split(",") if p.strip()]:
            if "=" not in part:
                raise DirectiveParseError(f"segment '{part}' is not a key=value pair")
            k, v = (s.strip() for s in part.split("=", 1))
            if k == "model_id":
                target_id = v
            elif k in numeric_keys:
                try:
                    updates[k] = numeric_keys[k](v)
                except ValueError as e:
                    raise DirectiveParseError(f"'{k}={v}' is not a valid {numeric_keys[k].__name__}") from e
            else:
                unknown_keys.append(k)

        if unknown_keys:
            raise DirectiveParseError(f"unsupported setting(s): {', '.join(unknown_keys)}")
        if not updates:
            raise DirectiveParseError("no supported sampling settings were provided")
        if target_id not in self.models:
            raise DirectiveParseError(f"target model '{target_id}' is not in the chat room")

        self.models[target_id].update(updates)
        if target_id in self.known_models:
            self.known_models[target_id].update(updates)
        self.memory_manager.add_entry(
            author=model_cfg["name"],
            content=f"Updated sampling settings for `{target_id}` based on Hugging Face / performance research: {updates}"
        )

    def _apply_task_directive(self, model_id: str, model_cfg: Dict[str, Any], payload: str):
        """Applies an [UPDATE_TASK: key=value, ...] directive, updating an existing task or creating one."""
        # Split on ", key=" boundaries rather than on every comma. Naive comma-splitting
        # made the directive unusable for its single most important field: any description
        # written as real English ("...a flag to include whitespace, handling empty strings")
        # contains commas, so the whole directive was rejected and the Architect's plan
        # thrown away - the model was doing its job and the parser was discarding it.
        known_keys = ("id", "title", "description", "priority", "status", "assigned_model")
        key_pattern = r"(?:^|,)\s*(" + "|".join(known_keys) + r")\s*="
        matches = list(re.finditer(key_pattern, payload, flags=re.IGNORECASE))

        kwargs: Dict[str, str] = {}
        if matches:
            for idx, m in enumerate(matches):
                key = m.group(1).strip().lower()
                start = m.end()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(payload)
                kwargs[key] = payload[start:end].strip().strip(",").strip()
        else:
            # No recognised key at all - fall back to the original strict parse so a
            # genuinely malformed directive still reports a clear error.
            for part in [p.strip() for p in payload.split(",") if p.strip()]:
                if "=" not in part:
                    raise DirectiveParseError(f"segment '{part}' is not a key=value pair")
                k, v = (t.strip() for t in part.split("=", 1))
                kwargs[k] = v

        if not kwargs:
            raise DirectiveParseError("no task fields were provided")

        task_id = kwargs.get("id")
        if task_id and any(t["id"] == task_id for t in self.memory_manager.get_task_itinerary()):
            updates = {k: v for k, v in kwargs.items() if k != "id"}
            self.memory_manager.update_itinerary_task(task_id, updates)
            self.memory_manager.add_entry(
                model_cfg["name"],
                f"Updated itinerary task '{task_id}': {updates}"
            )
            return

        if task_id:
            raise DirectiveParseError(f"itinerary task '{task_id}' does not exist")

        title = kwargs.get("title", kwargs.get("description", "New Task"))
        status = kwargs.get("status", "pending")
        created = self.memory_manager.add_itinerary_task(
            title=title,
            description=kwargs.get("description", title),
            priority=kwargs.get("priority", "medium"),
            assigned_model=kwargs.get("assigned_model", model_id)
        )
        if status != "pending":
            self.memory_manager.update_itinerary_task(created["id"], {"status": status})
        self.memory_manager.add_entry(
            model_cfg["name"],
            f"Created new itinerary task '{created['id']}': {title} (Status: {status})"
        )

    async def run_autonomous_loop(self, max_turns: int = 5, max_discussion_turns: int = 8, max_discussion_seconds: float = 120.0):
        if self.loop_active:
            return
        self.loop_active = True
        start_time = time.time()
        discussion_turns = 0

        try:
            turns = 0
            while turns < max_turns and self.loop_active:
                current_phase = self.memory_manager.get_phase()

                # Discussion hard cap (turns and wall clock). This used to force-flip the
                # room into Execution, which is exactly how an unreviewed plan reached the
                # Coder: whatever the Architect happened to have said when the timer expired
                # became the build. The cap now only stops the loop - the Architect opens
                # Execution once the plan gate reaches `approved`, nobody else and nothing else.
                if current_phase == "discussion":
                    discussion_turns += 1
                    elapsed_disc = time.time() - start_time
                    if discussion_turns >= max_discussion_turns or elapsed_disc >= max_discussion_seconds:
                        stage = self.memory_manager.get_plan_stage()
                        pending = self._plan_stage_owner(stage)
                        pending_name = self.models.get(pending, {}).get("name") if pending else None
                        waiting = f" Waiting on @{pending_name}." if pending_name else ""
                        self.add_chat_message(
                            sender="System / Plan Gate",
                            role="System",
                            content=(
                                f"⏹️ [PLANNING CAP REACHED] Stopping after {discussion_turns} turns / "
                                f"{int(elapsed_disc)}s with the plan still at `{stage}`.{waiting} "
                                f"Step the room on, or send a note to redirect the plan."
                            ),
                            is_admin=True
                        )
                        break

                next_speaker = self.get_next_speaker(self.last_speaker_id)
                if not next_speaker:
                    # Distinct failure mode from a timed-out turn below: routing/turn-schedule
                    # state produced no eligible speaker at all (e.g. empty room, or every
                    # candidate filtered out). Log clearly before the silent break so this
                    # exit path is debuggable instead of the loop just quietly stopping.
                    logger.warning(
                        "Autonomous loop stopping: get_next_speaker() returned no eligible speaker "
                        "(phase=%s, last_speaker=%s, turn_schedule=%s)",
                        self.memory_manager.get_phase(), self.last_speaker_id, self.turn_schedule
                    )
                    break

                # Manage VRAM allocation prior to turn (ensure Moderator & Refiner get priority).
                # Never let a VRAM-management failure hang or abort turn advancement.
                try:
                    self.manage_vram_allocation(next_speaker)
                except Exception as e:
                    logger.warning("VRAM allocation management skipped for %s: %s", next_speaker, e)

                try:
                    res = await asyncio.wait_for(self.step_model_turn(next_speaker), timeout=AUTO_TURN_TIMEOUT_SECONDS)
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    # A hung/slow turn must not stall the whole room indefinitely - log it,
                    # treat the stalled speaker as having "gone" so rotation moves past it,
                    # and continue the loop with the next eligible speaker instead of
                    # retrying the same one or breaking out entirely.
                    logger.warning("Turn for %s timed out after %ss, forcing next speaker", next_speaker, AUTO_TURN_TIMEOUT_SECONDS)
                    self.add_chat_message(
                        sender="System / Conversation Loop",
                        role="System",
                        content=f"⏱️ [TURN TIMEOUT] @{next_speaker} did not respond within {AUTO_TURN_TIMEOUT_SECONDS}s - forcing rotation to the next speaker.",
                        is_admin=True
                    )
                    self.last_speaker_id = next_speaker
                    await asyncio.sleep(0.5)
                    continue
                except Exception as e:
                    logger.exception("Autonomous loop aborted while stepping %s", next_speaker)
                    self.add_chat_message(
                        sender="System / Conversation Loop",
                        role="System",
                        content=f"🛑 [LOOP HALTED] The conversation loop stopped while @{next_speaker} was taking a turn: {e}",
                        is_admin=True
                    )
                    break
                turns += 1

                msg_content = res.get("content", "")
                if "[READY_FOR_EXECUTION]" in msg_content or "CONSENSUS_REACHED" in msg_content:
                    break
                await asyncio.sleep(0.5)
        finally:
            self.loop_active = False

    def manage_vram_allocation(self, active_speaker_id: str):
        """
        Dynamic Resource Management:
        Ensures active speaker, upcoming roster models, and Moderator are prioritized in VRAM based on who is expected to be needed next.
        Offloads remaining models to RAM if VRAM headroom is tight.
        """
        hw = self.model_manager.get_hardware_info()
        has_gpu = bool(hw.get("gpu_name") or hw.get("vram_total_gb", 0) > 0)
        if not has_gpu:
            return

        # The supervisor (Architect) is worth keeping resident: it plans, creates tasks and
        # handles escalation. Derived from role now - it used to come from a settable flag
        # that could point at a non-Architect model.
        mod_id = self.moderator_model_id

        # Priority 1: Active speaker & Architect
        # Priority 2: Upcoming speakers in turn roster
        # Ask the same authority the turn loop asks. Reading turn_schedule directly here was
        # the second turn-order system: during execution that queue is never consumed, so the
        # prefetcher was loading round-robin models while the router ran residency-major.
        upcoming_roster = self.upcoming_speakers(3)
        # Ordered, not a set: the active speaker must be loaded first so that if the roster
        # still doesn't fit, the model whose turn it actually is keeps the VRAM.
        priority_ids: List[str] = []
        for _pid in [active_speaker_id, mod_id] + upcoming_roster:
            if _pid and _pid not in priority_ids:
                priority_ids.append(_pid)

        # Deliberately NO speculative eviction here.
        #
        # This used to unload every non-priority model on every turn. Combined with the old
        # task-major router - which changed role, and therefore model, after literally every
        # turn - it meant the room evicted and reloaded models continuously and spent most of
        # its wall-clock time on model I/O rather than work.
        #
        # The scheduler now keeps a loaded team working until it runs out of things to do
        # (see _select_execution_speaker), so residency is the thing worth preserving. When a
        # model genuinely does not fit, load_gguf_model's capacity gate evicts the
        # least-recently-used model on demand and only as far as it must. Eviction is a last
        # resort, not a per-turn routine.

        # Preload priority GGUF models into VRAM if room exists. Only the active speaker
        # (index 0) may evict to get its slot; speculative preloads of upcoming speakers must
        # never push the current speaker out.
        for _idx, pid in enumerate(priority_ids):
            if _idx > 0 and pid in self.models and self.models[pid].get("provider") == "gguf_local":
                _p = self.model_manager.resolve_gguf_path(
                    self.models[pid].get("gguf_path") or self.models[pid].get("model_name", "")
                )
                if not _p or not os.path.exists(_p):
                    continue
                _size = round(os.path.getsize(_p) / (1024 ** 3), 2)
                if not self.model_manager.can_load_model(_size).get("allowed"):
                    logger.debug("Skipping speculative preload of %s - no headroom", pid)
                    continue
            if pid in self.models and self.models[pid].get("provider") == "gguf_local":
                pcfg = self.models[pid]
                ppath = pcfg.get("gguf_path") or pcfg.get("model_name", "")
                if ppath and pid not in self.model_manager.gguf_instances:
                    try:
                        self.model_manager.load_gguf_model(
                            pid,
                            ppath,
                            # Must match the generate path's default. When this preload used
                            # 2048, the cached instance kept that context for the rest of the
                            # session and every turn afterwards was silently starved: a
                            # reasoning model got ~800 generation tokens and never finished
                            # its <think> block, so the room saw an empty turn.
                            max_tokens=pcfg.get("max_context_tokens") or DEFAULT_N_CTX,
                            mmproj_path=pcfg.get("mmproj_path"),
                            force_device="gpu"
                        )
                    except Exception as e:
                        logger.warning("Priority VRAM preload for %s skipped: %s", pid, e)

        # The old "if VRAM is tight, offload everything non-priority" sweep lived here. It is
        # gone for the same reason as the eviction pass above: tight VRAM is the normal
        # steady state for a roster this size, so the sweep fired constantly and undid the
        # residency the scheduler is trying to build. The capacity gate in load_gguf_model
        # frees exactly as much as the next model needs, when it needs it.

    def propose_tool_call(self, model_id: str, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        risk = self.tool_manager.classify_tool_risk(tool_name)
        vote_req = {
            "id": f"vote_{int(time.time()*1000)}",
            "model_id": model_id,
            "model_name": self.models.get(model_id, {}).get("name", "Unknown"),
            "tool_name": tool_name,
            "args": args,
            "risk_level": risk,
            "votes": {},
            "status": "pending",
            "created_at": time.time()
        }

        if risk == "low":
            exec_res = self._execute_tool_sync(tool_name, args, caller_id=model_id)
            vote_req["status"] = "executed" if exec_res.get("success") else "failed"
            vote_req["result"] = exec_res
            if not exec_res.get("success"):
                vote_req["error"] = exec_res.get("error", "Tool execution failed.")
            return vote_req

        self.pending_tool_votes.append(vote_req)
        return vote_req

    def _execute_tool_sync(self, tool_name: str, args: Dict[str, Any], caller_id: str = "Admin") -> Dict[str, Any]:
        """Runs a tool from synchronous code.

        Loop discovery is separated from tool execution so a failing tool is never silently
        retried - and therefore executed twice - by a catch-all fallback.
        """
        try:
            running_loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is None:
            return asyncio.run(self._execute_tool_async(tool_name, args, caller_id))

        try:
            import nest_asyncio
        except ImportError as e:
            logger.error("Cannot run tool '%s' inside a running event loop: %s", tool_name, e)
            return {
                "success": False,
                "error": (
                    f"Tool '{tool_name}' was invoked synchronously from a running event loop, "
                    "which requires the 'nest_asyncio' package."
                )
            }

        nest_asyncio.apply(running_loop)
        return running_loop.run_until_complete(self._execute_tool_async(tool_name, args, caller_id))

    async def _execute_tool_async(self, tool_name: str, args: Dict[str, Any], caller_id: str = "Admin") -> Dict[str, Any]:
        # Lock write tools during discussion phase
        current_phase = self.memory_manager.get_phase()
        if current_phase == "discussion" and tool_name in ["write_file", "patch_file", "bot_workspace_write", "bot_workspace_merge"]:
            return {
                "success": False,
                "error": (
                    "File write tools are locked during the planning phase. The plan is at "
                    f"`{self.memory_manager.get_plan_stage()}`; it must clear Critic and "
                    "Programmer review, then the Architect opens Execution with [READY_FOR_EXECUTION]."
                )
            }

        try:
            if tool_name == "read_file":
                filepath = args.get("filepath", "")
                self.set_model_live_status(caller_id, f"Perusing {filepath or 'files'}")
                return self.tool_manager.read_file(filepath, bot_id=caller_id)
            elif tool_name == "list_files":
                self.set_model_live_status(caller_id, "Listing workspace directory files")
                return self.tool_manager.list_files(args.get("dir", "."), bot_id=caller_id)
            elif tool_name == "search_workspace":
                query = args.get("query", "")
                self.set_model_live_status(caller_id, f"Searching workspace for '{query}'")
                return self.tool_manager.search_workspace(query)
            elif tool_name == "internet_search":
                query = args.get("query", "")
                self.set_model_live_status(caller_id, f"Searching web for '{query}'")
                return await self.tool_manager.internet_search(query, domain_filter=args.get("domain_filter"))
            elif tool_name == "search_huggingface":
                query = args.get("query", "")
                self.set_model_live_status(caller_id, f"Researching HuggingFace for '{query}'")
                return await self.tool_manager.search_huggingface(query, limit=args.get("limit", 5))
            elif tool_name == "copy_file":
                src = args.get("src", "")
                dest = args.get("dest", "")
                self.set_model_live_status(caller_id, f"Cloning / copying {src} to {dest}")
                res = self.tool_manager.copy_file(src, dest, bot_id=caller_id)
                if res.get("success"):
                    self.memory_manager.log_file_edit(
                        filepath=dest,
                        author=caller_id,
                        action="copy",
                        diff_snippet=f"Copied from {src}"
                    )
                return res
            elif tool_name == "write_file":
                filepath = args.get("filepath", "")
                self.set_model_live_status(caller_id, f"Editing file {filepath}")
                content = args.get("content", "")
                res = self.tool_manager.write_file(filepath, content, bot_id=caller_id)
                if res.get("success"):
                    self.memory_manager.log_file_edit(
                        filepath=filepath,
                        author=caller_id,
                        action="write",
                        diff_snippet=f"Written {res.get('bytes_written', 0)} bytes"
                    )
                return res
            elif tool_name == "run_python":
                filepath = args.get("filepath", "")
                self.set_model_live_status(caller_id, f"Executing python sandbox script {filepath}")
                return self.tool_manager.run_python(filepath=filepath, bot_id=caller_id)
            elif tool_name == "run_tests":
                test_path = args.get("test_path")
                self.set_model_live_status(caller_id, "Running sandbox pytest suite")
                return self.tool_manager.run_tests(bot_id=caller_id, test_path=test_path)
            elif tool_name == "bot_workspace_write":
                filepath = args.get("filepath", "")
                self.set_model_live_status(caller_id, f"Writing workspace sandbox {filepath}")
                res = self.tool_manager.bot_workspace_write(bot_id=caller_id, filepath=filepath, content=args.get("content", ""))
                # bot_workspace_write's "success" means "syntax valid", not "file written" - the
                # write itself always lands on disk first. Gating the refinement loop on
                # write-success meant it could only ever fire for files that were ALREADY valid,
                # never for the syntax errors it exists to repair. Gate on the write having
                # actually happened (no OSError) instead.
                if "bytes_written" in res and filepath.endswith(".py"):
                    asyncio.create_task(self.trigger_sandbox_refinement_loop(bot_id=caller_id, filepath=filepath))
                return res
            elif tool_name == "bot_workspace_merge":
                filepath = args.get("filepath", "")
                self.set_model_live_status(caller_id, f"Merging sandbox edits for {filepath}")
                res = self.tool_manager.bot_workspace_merge_to_main(bot_id=caller_id, filepath=filepath)
                if res.get("success"):
                    self.memory_manager.log_file_edit(
                        filepath=filepath,
                        author=caller_id,
                        action="merge",
                        diff_snippet=f"Merged from workspace {caller_id}"
                    )
                return res
            elif tool_name == "run_terminal_cmd":
                cmd = args.get("command", "")
                self.set_model_live_status(caller_id, f"Executing command '{cmd[:20]}...'")
                return self.tool_manager.run_terminal_cmd(cmd)
            elif tool_name == "git_status":
                self.set_model_live_status(caller_id, "Checking git repository status")
                return self.tool_manager.git_status()
            elif tool_name == "git_diff":
                self.set_model_live_status(caller_id, "Reviewing git diff")
                return self.tool_manager.git_diff()
            elif tool_name == "condense_workspace_code":
                target_fn = args.get("target_filename", "main.py")
                self.set_model_live_status(caller_id, f"Condensing generated workspace code into {target_fn}")
                return self.tool_manager.condense_workspace_code(bot_id=caller_id, target_filename=target_fn)
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.exception("Tool '%s' raised while running for %s", tool_name, caller_id)
            return {"success": False, "error": f"Tool '{tool_name}' failed: {e}"}
        finally:
            self.set_model_live_status(caller_id, "Idle / Live in Chat")

    def admin_override_vote(self, vote_id: str, action: str, modified_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        for req in self.pending_tool_votes:
            if req["id"] == vote_id:
                if action == "approve":
                    req["status"] = "approved"
                    args = modified_args or req["args"]
                    exec_res = self._execute_tool_sync(req["tool_name"], args, caller_id=req.get("model_id", "Admin"))
                    tool_ok = bool(exec_res.get("success"))
                    req["status"] = "executed" if tool_ok else "failed"
                    req["result"] = exec_res
                    return {
                        "success": tool_ok,
                        "executed": True,
                        "result": exec_res,
                        "error": None if tool_ok else exec_res.get("error", "Tool execution failed.")
                    }
                elif action == "reject":
                    req["status"] = "rejected"
                    return {"success": True, "status": "rejected"}
                return {"success": False, "error": f"Unknown vote action '{action}'. Use 'approve' or 'reject'."}
        return {"success": False, "error": f"Vote ID '{vote_id}' not found"}
