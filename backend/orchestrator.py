import time
import random
import asyncio
from typing import Dict, Any, List, Optional
from backend.models import ModelManager
from backend.prompts import get_system_prompt
from backend.memory import MemoryManager
from backend.tools import ToolManager

class Orchestrator:
    def __init__(self, model_manager: ModelManager, memory_manager: MemoryManager, tool_manager: ToolManager):
        self.model_manager = model_manager
        self.memory_manager = memory_manager
        self.tool_manager = tool_manager

        self.turn_mode = "round_robin"
        self.voting_threshold = "majority"
        self.respond_immediately_to_at = True
        self.moderator_model_id = "model_architect"

        self.loop_active = False
        self.tie_counters: Dict[str, int] = {}
        self.last_speaker_id: Optional[str] = None

        # Known models library (persisted across room additions/removals)
        self.known_models: Dict[str, Dict[str, Any]] = {
            "model_architect": {
                "id": "model_architect",
                "name": "Architect",
                "role": "Architect",
                "provider": "ollama",
                "model_name": "llama3.2:1b",
                "enabled": True,
                "is_moderator": True,
                "status": "active",
                "max_context_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1
            },
            "model_critic": {
                "id": "model_critic",
                "name": "Critic",
                "role": "Critic",
                "provider": "ollama",
                "model_name": "llama3.2:1b",
                "enabled": True,
                "is_moderator": False,
                "status": "active",
                "max_context_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1
            },
            "model_coder": {
                "id": "model_coder",
                "name": "Coder",
                "role": "Coder",
                "provider": "ollama",
                "model_name": "llama3.2:1b",
                "enabled": True,
                "is_moderator": False,
                "status": "active",
                "max_context_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1
            }
        }

        # Active chatroom models (subset of known models currently in the room)
        self.models: Dict[str, Dict[str, Any]] = {
            m_id: {**cfg, "live_status": "Idle / Live in Chat"} for m_id, cfg in self.known_models.items()
        }

        self.pending_tool_votes: List[Dict[str, Any]] = []
        self.chat_history: List[Dict[str, Any]] = []
        self.turn_schedule: List[str] = []  # 5-10 turn scheduled roster queue
        self.autorun_enabled: bool = False
        self.last_speech_time: float = time.time()
        self.spoken_models: set = set()  # Tracks models that have spoken at least once

    def set_turn_mode(self, mode: str):
        if mode in ["admin_controlled", "moderator_controlled", "round_robin"]:
            self.turn_mode = mode

    def set_moderator(self, model_id: str):
        old_mod_id = self.moderator_model_id
        if model_id in self.models:
            for m_id, m_cfg in self.models.items():
                was_mod = m_cfg.get("is_moderator", False)
                should_be_mod = (m_id == model_id)
                m_cfg["is_moderator"] = should_be_mod
                if was_mod and not should_be_mod and m_id != model_id:
                    # Notify demoted model
                    self.add_chat_message(
                        sender="System / Role Manager",
                        role="System",
                        content=f"📢 [SYSTEM NOTIFICATION TO @{m_cfg['name']}]: You have been demoted from Moderator / Chief Project Manager status. @{self.models[model_id]['name']} is now the Chief Project Manager.",
                        is_admin=True
                    )
            for m_id, m_cfg in self.known_models.items():
                m_cfg["is_moderator"] = (m_id == model_id)
            self.moderator_model_id = model_id

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
            model_cfg["live_status"] = "Idle / Live in Chat"
        self.known_models[m_id] = model_cfg
        self.models[m_id] = dict(model_cfg)
        if model_cfg.get("is_moderator"):
            self.set_moderator(m_id)

    def set_model_live_status(self, model_id: str, status: str):
        if model_id in self.models:
            self.models[model_id]["live_status"] = status
        if model_id in self.known_models:
            self.known_models[model_id]["live_status"] = status

    def kick_model_from_room(self, model_id: str) -> Dict[str, Any]:
        was_moderator = False
        if model_id in self.models:
            was_moderator = self.models[model_id].get("is_moderator", False)
            del self.models[model_id]

        new_mod_id = None
        if was_moderator:
            # Select replacement that wasn't former moderator
            candidates = [m_id for m_id in self.models if m_id != model_id]
            if candidates:
                new_mod_id = candidates[0]
                self.set_moderator(new_mod_id)
            else:
                self.moderator_model_id = None

        return {
            "success": True,
            "kicked_id": model_id,
            "was_moderator": was_moderator,
            "auto_assigned_moderator": new_mod_id,
            "active_models": self.models
        }

    def readd_model_to_room(self, model_id: str) -> Dict[str, Any]:
        if model_id in self.known_models:
            self.models[model_id] = dict(self.known_models[model_id])
            if self.known_models[model_id].get("is_moderator"):
                self.set_moderator(model_id)
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
        return msg

    def generate_turn_schedule(self, length: int = 8) -> List[str]:
        """Generates a 5-10 turn scheduled roster based on available user roles, prioritizing Architect first."""
        active_models = [m_id for m_id, m in self.models.items() if m.get("enabled", True)]
        if not active_models:
            return []

        schedule: List[str] = []

        # 1. Ensure Architect is first if present and not spoke very recently
        architect_id = None
        for m_id in active_models:
            if self.models[m_id].get("role", "").lower() in ["architect", "planner"] or self.models[m_id].get("is_moderator"):
                architect_id = m_id
                break

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

    def get_next_speaker(self, last_speaker_id: Optional[str] = None) -> Optional[str]:
        active_models = [m_id for m_id, m in self.models.items() if m.get("enabled", True)]
        if not active_models:
            return None

        # Check safety rule: Every 15 messages, ensure any model that hasn't spoken gets a turn
        if len(self.chat_history) >= 15:
            last_15 = self.chat_history[-15:]
            spoken_ids = {m.get("model_id") for m in last_15 if m.get("model_id")}
            unspoken = [m_id for m_id in active_models if m_id not in spoken_ids]
            if unspoken:
                return unspoken[0]

        # Context-based selection: check if last message @mentions a specific model name/role
        if self.chat_history:
            last_msg = self.chat_history[-1]["content"].lower()
            for m_id in active_models:
                m_cfg = self.models[m_id]
                if f"@{m_cfg['name'].lower()}" in last_msg or f"@{m_cfg['role'].lower()}" in last_msg:
                    return m_id

        # Use scheduled queue if available
        if not self.turn_schedule:
            self.generate_turn_schedule()

        if self.turn_schedule:
            next_spk = self.turn_schedule.pop(0)
            if next_spk in active_models:
                return next_spk

        # If roster queue runs out, inform Moderator explicitly about loaded models vs available models
        loaded_info = [f"@{m['name']} ({m['role']})" for m_id, m in self.models.items() if m.get("enabled", True)]
        avail_info = [f"@{m['name']} ({m['role']})" for m_id, m in self.known_models.items() if m_id not in self.models]

        mod_msg = (
            f"⚠️ [ROSTER QUEUE EXHAUSTED]: The turn schedule queue has run out! "
            f"Chief Project Manager (@{self.models.get(self.moderator_model_id, {}).get('name', 'Moderator')}), please refill the roster queue. "
            f"\n- Currently Loaded Active Models: {', '.join(loaded_info) if loaded_info else 'None'}"
            f"\n- Available Models in Library: {', '.join(avail_info) if avail_info else 'None'}"
        )
        self.add_chat_message(
            sender="System / Roster Manager",
            role="System",
            content=mod_msg,
            is_admin=True
        )

        # Fallback to round-robin
        effective_last = last_speaker_id or self.last_speaker_id
        if effective_last and effective_last in active_models and len(active_models) > 1:
            idx = active_models.index(effective_last)
            return active_models[(idx + 1) % len(active_models)]

        return active_models[0]

    async def step_model_turn(self, model_id: str) -> Dict[str, Any]:
        model_cfg = self.models.get(model_id)
        if not model_cfg or not model_cfg["enabled"]:
            return {"error": "Model not available"}

        current_phase = self.memory_manager.get_phase()

        memory_summary = self.memory_manager.get_memory_summary()
        latest_journal = self.memory_manager.get_model_latest_journal(model_id)

        # Retrieve recent episodes for NAC-style thread weaving
        episodes = self.memory_manager.get_latest_episodes(limit=3)
        ep_summary = ""
        if episodes:
            ep_lines = [f"- [{e['author']}] Task ({e['action']}): {e['summary']}" for e in episodes]
            ep_summary = f"\n\n### RECENT EPISODE CHECKPOINTS (HANDOFFS):\n" + "\n".join(ep_lines)

        # Retrieve Active Task / Itinerary Item for Meetings
        active_task = self.memory_manager.get_active_task()

        is_first_turn = model_id not in self.spoken_models
        self.spoken_models.add(model_id)

        sys_prompt = get_system_prompt(
            role=model_cfg["role"],
            name=model_cfg["name"],
            phase=current_phase,
            project_info=self.memory_manager.get_project_id(),
            current_task=active_task["title"] if active_task else "General Discussion / Alignment",
            is_moderator=model_cfg.get("is_moderator", False),
            custom_template=model_cfg.get("custom_start_prompt") if current_phase == "discussion" else model_cfg.get("custom_execution_prompt"),
            model_id=model_id
        )

        if not is_first_turn:
            # Subsequent turn reprompting: Keep context window fresh by providing concise instructions and only the last few messages
            sys_prompt = f"You are {model_cfg['name']} ({model_cfg['role']}). Continue contributing concisely to the task: {active_task['title'] if active_task else 'General Discussion'}."

        task_context = ""
        if active_task:
            task_context = f"\n\n### 🎯 ACTIVE ITINERARY ITEM / MEETING AGENDA:\nTitle: {active_task['title']}\nDescription: {active_task['description']}\nPriority: {active_task['priority'].upper()}\nStatus: {active_task['status'].upper()}"

        journal_context = ""
        if latest_journal:
            journal_context = f"\n\n### YOUR LATEST TIMESTAMPED SELF-JOURNAL (PRE-NAP PERSPECTIVE):\n{latest_journal}"

        own_spec = self.memory_manager.get_spec_file(model_id)
        spec_context = f"\n\n### YOUR PERSONAL SPEC NOTEBOOK:\n{own_spec if own_spec else '(Empty - use [UPDATE_SPEC: <content>] to record research or notes)'}"

        context_prompt = f"{sys_prompt}\n\n### SHARED MEMORY SUMMARY:\n{memory_summary}{ep_summary}{task_context}{journal_context}{spec_context}"

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
            # Clean context refresh mode: minimal message context to prevent token overload
            file_manifest = self.tool_manager.list_files(".", bot_id=model_id)
            manifest_str = ", ".join(file_manifest.get("files", [])[:15]) if isinstance(file_manifest, dict) else ""

            recent_msgs = [
                {
                    "role": "user",
                    "content": f"[SYSTEM REFRESH - INDEXED MEMORY MODE]\nTask: {active_task['title'] if active_task else 'Execute or discuss requirements'}\nWorkspace Files: {manifest_str}\nProvide your next concise contribution or action tag based on indexed memory."
                }
            ]
        else:
            # Discussion phase: Truncate messages to prevent prompt overflow & context degradation
            # Limit each message to max 300 characters and include at most the last 3 turns to prevent echo loops
            from backend.sanitizer import sanitize_message_content
            recent_msgs = []
            for m in self.chat_history[-3:]:
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
        try:
            response_text = await self.model_manager.generate_response(
                model_config=model_cfg,
                system_prompt=context_prompt,
                messages=recent_msgs
            )
        finally:
            self.set_model_live_status(model_id, "Idle / Live in Chat")

        self.memory_manager.update_token_usage(model_id, len(response_text.split()))

        if "[READY_FOR_EXECUTION]" in response_text:
            self.memory_manager.add_entry(model_cfg["name"], "Declared readiness for Execution Phase.")
            if self.memory_manager.get_phase() == "discussion":
                self.memory_manager.set_phase("execution")
                # Auto-select or create first itinerary task if none in progress
                active_t = self.memory_manager.get_active_task()
                if not active_t:
                    self.memory_manager.add_itinerary_task(
                        title="Execute Project Requirements",
                        description="Implement codebase updates based on discussion phase consensus.",
                        priority="high",
                        assigned_model=model_id
                    )
        elif "[REQUEST_DISCUSSION]" in response_text:
            self.memory_manager.add_entry(model_cfg["name"], "Requested return to Discussion Phase due to ambiguity.")

        if "[UPDATE_CONFIG:" in response_text:
            try:
                start = response_text.find("[UPDATE_CONFIG:") + len("[UPDATE_CONFIG:")
                end = response_text.find("]", start)
                if end != -1:
                    raw_cfg = response_text[start:end].strip()
                    parts = [p.strip() for p in raw_cfg.split(",")]
                    updates = {}
                    target_id = model_id
                    for p in parts:
                        if "=" in p:
                            k, v = p.split("=", 1)
                            k, v = k.strip(), v.strip()
                            if k == "model_id":
                                target_id = v
                            elif k in ["top_p", "temperature", "repeat_penalty"]:
                                updates[k] = float(v)
                            elif k == "top_k":
                                updates[k] = int(v)
                    if updates and target_id in self.models:
                        self.models[target_id].update(updates)
                        if target_id in self.known_models:
                            self.known_models[target_id].update(updates)
                        self.memory_manager.add_entry(
                            author=model_cfg["name"],
                            content=f"Updated sampling settings for `{target_id}` based on Hugging Face / performance research: {updates}"
                        )
            except Exception:
                pass

        if "[UPDATE_SPEC:" in response_text:
            try:
                start = response_text.find("[UPDATE_SPEC:") + len("[UPDATE_SPEC:")
                end = response_text.find("]", start)
                if end != -1:
                    spec_content = response_text[start:end].strip()
                    self.memory_manager.update_spec_file(model_id, spec_content)
            except Exception:
                pass

        if "[UPDATE_TASK:" in response_text:
            try:
                start = response_text.find("[UPDATE_TASK:") + len("[UPDATE_TASK:")
                end = response_text.find("]", start)
                if end != -1:
                    raw_task = response_text[start:end].strip()
                    parts = [p.strip() for p in raw_task.split(",")]
                    kwargs = {}
                    for p in parts:
                        if "=" in p:
                            k, v = p.split("=", 1)
                            kwargs[k.strip()] = v.strip()

                    task_id = kwargs.get("id")
                    if task_id and any(t["id"] == task_id for t in self.memory_manager.get_task_itinerary()):
                        updates = {k: v for k, v in kwargs.items() if k != "id"}
                        self.memory_manager.update_itinerary_task(task_id, updates)
                        self.memory_manager.add_entry(
                            model_cfg["name"],
                            f"Updated itinerary task '{task_id}': {updates}"
                        )
                    else:
                        title = kwargs.get("title", kwargs.get("description", "New Task"))
                        desc = kwargs.get("description", title)
                        priority = kwargs.get("priority", "medium")
                        status = kwargs.get("status", "pending")
                        created = self.memory_manager.add_itinerary_task(
                            title=title,
                            description=desc,
                            priority=priority,
                            assigned_model=kwargs.get("assigned_model", model_id)
                        )
                        if status != "pending":
                            self.memory_manager.update_itinerary_task(created["id"], {"status": status})
                        self.memory_manager.add_entry(
                            model_cfg["name"],
                            f"Created new itinerary task '{created['id']}': {title} (Status: {status})"
                        )
            except Exception as e:
                print(f"Error parsing UPDATE_TASK tag: {e}")

        if "[SEARCH_HF:" in response_text:
            try:
                start = response_text.find("[SEARCH_HF:") + len("[SEARCH_HF:")
                end = response_text.find("]", start)
                if end != -1:
                    query = response_text[start:end].strip()
                    hf_res = await self.tool_manager.search_huggingface(query)
                    m_list = hf_res.get("models", [])
                    res_summary = ", ".join([m["model_id"] for m in m_list[:3]])
                    self.memory_manager.add_entry(
                        model_cfg["name"],
                        f"HuggingFace search for '{query}' returned candidate models: {res_summary}"
                    )
            except Exception:
                pass

        if "[JOURNAL:" in response_text:
            try:
                start = response_text.find("[JOURNAL:") + len("[JOURNAL:")
                end = response_text.find("]", start)
                if end != -1:
                    journal_content = response_text[start:end].strip()
                    self.memory_manager.record_model_nap(model_id, journal_content)
            except Exception:
                pass

        if "[LOG_TO_MEMORY:" in response_text:
            try:
                start = response_text.find("[LOG_TO_MEMORY:") + len("[LOG_TO_MEMORY:")
                end = response_text.find("]", start)
                if end != -1:
                    mem_content = response_text[start:end].strip()
                    self.memory_manager.add_entry(model_cfg["name"], mem_content)
            except Exception:
                pass

        if "[REQUEST_NAP]" in response_text and "[JOURNAL:" not in response_text:
            self.memory_manager.record_model_nap(model_id, f"{model_cfg['name']} completed a context nap.")

        msg = self.add_chat_message(
            sender=model_cfg["name"],
            role=model_cfg["role"],
            content=response_text,
            is_admin=False,
            model_id=model_id
        )

        self.last_speaker_id = model_id
        return msg

    async def run_autonomous_loop(self, max_turns: int = 5):
        if self.loop_active:
            return
        self.loop_active = True
        try:
            turns = 0
            while turns < max_turns and self.loop_active:
                next_speaker = self.get_next_speaker(self.last_speaker_id)
                if not next_speaker:
                    break

                # Manage VRAM allocation prior to turn (ensure Moderator gets priority if VRAM is tight)
                self.manage_vram_allocation(next_speaker)

                res = await self.step_model_turn(next_speaker)
                turns += 1

                # If model indicated consensus / execution readiness or requested pause, pause loop
                msg_content = res.get("content", "")
                if "[READY_FOR_EXECUTION]" in msg_content or "CONSENSUS_REACHED" in msg_content:
                    break
                await asyncio.sleep(0.5)
        finally:
            self.loop_active = False

    def manage_vram_allocation(self, active_speaker_id: str):
        """
        Dynamic Moderator Resource Management:
        Ensures Architect / Moderator loads in VRAM first.
        If VRAM headroom is tight, unloads inactive models or offloads smaller models to CPU/RAM,
        while ensuring models can be dropped and reloaded on demand.
        """
        hw = self.model_manager.get_hardware_info()
        mod_id = self.moderator_model_id or "model_architect"

        # Ensure Moderator/Architect is prioritized in VRAM
        if mod_id in self.models and self.models[mod_id].get("provider") == "gguf_local":
            mod_cfg = self.models[mod_id]
            mod_path = mod_cfg.get("gguf_path") or mod_cfg.get("model_name", "")
            if mod_path and mod_id not in self.model_manager.gguf_instances:
                self.model_manager.load_gguf_model(
                    mod_id,
                    mod_path,
                    max_tokens=mod_cfg.get("max_context_tokens", 2048),
                    mmproj_path=mod_cfg.get("mmproj_path"),
                    force_device="gpu"
                )

        # If VRAM headroom is tight (< 1.5 GB), offload non-active models to RAM or unload them
        if hw.get("vram_free_gb", 0) < 1.5:
            for m_id, m_cfg in list(self.models.items()):
                if m_cfg.get("provider") == "gguf_local":
                    if m_id != mod_id and m_id != active_speaker_id:
                        self.model_manager.unload_gguf_model(m_id)
                        self.model_manager.update_model_status(
                            m_id,
                            status="online",
                            error=None,
                            vram_used_gb=0.0,
                            location="RAM"
                        )

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
            vote_req["status"] = "executed"
            vote_req["result"] = exec_res
            return vote_req

        self.pending_tool_votes.append(vote_req)
        return vote_req

    def _execute_tool_sync(self, tool_name: str, args: Dict[str, Any], caller_id: str = "Admin") -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Run in existing event loop
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self._execute_tool_async(tool_name, args, caller_id))
            else:
                return loop.run_until_complete(self._execute_tool_async(tool_name, args, caller_id))
        except Exception:
            return asyncio.run(self._execute_tool_async(tool_name, args, caller_id))

    async def _execute_tool_async(self, tool_name: str, args: Dict[str, Any], caller_id: str = "Admin") -> Dict[str, Any]:
        # Lock write tools during discussion phase
        current_phase = self.memory_manager.get_phase()
        if current_phase == "discussion" and tool_name in ["write_file", "patch_file", "bot_workspace_write", "bot_workspace_merge"]:
            return {
                "success": False,
                "error": "File write tools are locked during Discussion Phase. Declare readiness with [READY_FOR_EXECUTION] first."
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
            elif tool_name == "bot_workspace_write":
                filepath = args.get("filepath", "")
                self.set_model_live_status(caller_id, f"Writing workspace sandbox {filepath}")
                return self.tool_manager.bot_workspace_write(bot_id=caller_id, filepath=filepath, content=args.get("content", ""))
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
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        finally:
            self.set_model_live_status(caller_id, "Idle / Live in Chat")

    def admin_override_vote(self, vote_id: str, action: str, modified_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        for req in self.pending_tool_votes:
            if req["id"] == vote_id:
                if action == "approve":
                    req["status"] = "approved"
                    args = modified_args or req["args"]
                    exec_res = self._execute_tool_sync(req["tool_name"], args, caller_id=req.get("model_id", "Admin"))
                    req["status"] = "executed"
                    req["result"] = exec_res
                    return {"success": True, "executed": True, "result": exec_res}
                elif action == "reject":
                    req["status"] = "rejected"
                    return {"success": True, "status": "rejected"}
        return {"success": False, "error": "Vote ID not found"}
