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
                "max_context_tokens": 4096
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
                "max_context_tokens": 4096
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
                "max_context_tokens": 4096
            }
        }

        # Active chatroom models (subset of known models currently in the room)
        self.models: Dict[str, Dict[str, Any]] = {
            m_id: dict(cfg) for m_id, cfg in self.known_models.items()
        }

        self.pending_tool_votes: List[Dict[str, Any]] = []
        self.chat_history: List[Dict[str, Any]] = []

    def set_turn_mode(self, mode: str):
        if mode in ["admin_controlled", "moderator_controlled", "round_robin"]:
            self.turn_mode = mode

    def set_moderator(self, model_id: str):
        if model_id in self.models:
            for m_id, m_cfg in self.models.items():
                m_cfg["is_moderator"] = (m_id == model_id)
            for m_id, m_cfg in self.known_models.items():
                m_cfg["is_moderator"] = (m_id == model_id)
            self.moderator_model_id = model_id

    def add_or_update_known_model(self, model_cfg: Dict[str, Any]):
        m_id = model_cfg["id"]
        self.known_models[m_id] = model_cfg
        self.models[m_id] = dict(model_cfg)
        if model_cfg.get("is_moderator"):
            self.set_moderator(m_id)

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

        # Fallback to round-robin or stalling resolution (pick random eligible speaker who didn't speak last)
        effective_last = last_speaker_id or self.last_speaker_id
        if effective_last and effective_last in active_models and len(active_models) > 1:
            candidates = [m_id for m_id in active_models if m_id != effective_last]
            if self.turn_mode == "round_robin":
                idx = active_models.index(effective_last)
                return active_models[(idx + 1) % len(active_models)]
            return random.choice(candidates)

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

        sys_prompt = get_system_prompt(
            role=model_cfg["role"],
            name=model_cfg["name"],
            phase=current_phase,
            project_info=self.memory_manager.get_project_id(),
            current_task=active_task["title"] if active_task else "General Discussion / Alignment",
            is_moderator=model_cfg.get("is_moderator", False),
            custom_template=model_cfg.get("custom_start_prompt") if current_phase == "discussion" else model_cfg.get("custom_execution_prompt")
        )
        task_context = ""
        if active_task:
            task_context = f"\n\n### 🎯 ACTIVE ITINERARY ITEM / MEETING AGENDA:\nTitle: {active_task['title']}\nDescription: {active_task['description']}\nPriority: {active_task['priority'].upper()}\nStatus: {active_task['status'].upper()}"

        journal_context = ""
        if latest_journal:
            journal_context = f"\n\n### YOUR LATEST TIMESTAMPED SELF-JOURNAL (PRE-NAP PERSPECTIVE):\n{latest_journal}"

        context_prompt = f"{sys_prompt}\n\n### SHARED MEMORY SUMMARY:\n{memory_summary}{ep_summary}{task_context}{journal_context}"

        # Check model token usage against limits
        tokens_used = self.memory_manager.state.get("tokens_used", {}).get(model_id, 0)
        max_tokens = model_cfg.get("max_context_tokens", 4096)
        if tokens_used > (max_tokens * 0.8):
            context_prompt += "\n\n⚠️ WARNING: Your context usage is high! Please write a 200-300 token self-journal using `[JOURNAL: <summary>]` and request a nap `[REQUEST_NAP]`."

        recent_msgs = [
            {"role": "user" if m["is_admin"] else "assistant", "content": f"[{m['sender']} ({m['role']})]: {m['content']}"}
            for m in self.chat_history[-6:]
        ]
        if not recent_msgs:
            recent_msgs = [{"role": "user", "content": "Please introduce your perspective on the current project."}]

        response_text = await self.model_manager.generate_response(
            model_config=model_cfg,
            system_prompt=context_prompt,
            messages=recent_msgs
        )

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
        elif "[LOG_TO_MEMORY:" in response_text:
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
        """Prioritizes VRAM for the Moderator model and active speaker, offloading other GGUF models to RAM if tight."""
        hw = self.model_manager.get_hardware_info()
        if hw.get("vram_free_gb", 0) < 1.0:
            # Unload any GGUF instances that are not the Moderator and not the active speaker
            for m_id, m_cfg in list(self.models.items()):
                if m_cfg.get("provider") == "gguf_local":
                    if m_id != self.moderator_model_id and m_id != active_speaker_id:
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

        if tool_name == "read_file":
            return self.tool_manager.read_file(args.get("filepath", ""), bot_id=caller_id)
        elif tool_name == "list_files":
            return self.tool_manager.list_files(args.get("dir", "."), bot_id=caller_id)
        elif tool_name == "search_workspace":
            return self.tool_manager.search_workspace(args.get("query", ""))
        elif tool_name == "internet_search":
            return await self.tool_manager.internet_search(args.get("query", ""), domain_filter=args.get("domain_filter"))
        elif tool_name == "search_huggingface":
            return await self.tool_manager.search_huggingface(args.get("query", ""), limit=args.get("limit", 5))
        elif tool_name == "write_file":
            filepath = args.get("filepath", "")
            content = args.get("content", "")
            res = self.tool_manager.write_file(filepath, content, bot_id=caller_id)
            if res.get("success"):
                # Track file modification attribution
                self.memory_manager.log_file_edit(
                    filepath=filepath,
                    author=caller_id,
                    action="write",
                    diff_snippet=f"Written {res.get('bytes_written', 0)} bytes"
                )
            return res
        elif tool_name == "bot_workspace_write":
            return self.tool_manager.bot_workspace_write(bot_id=caller_id, filepath=args.get("filepath", ""), content=args.get("content", ""))
        elif tool_name == "bot_workspace_merge":
            res = self.tool_manager.bot_workspace_merge_to_main(bot_id=caller_id, filepath=args.get("filepath", ""))
            if res.get("success"):
                self.memory_manager.log_file_edit(
                    filepath=args.get("filepath", ""),
                    author=caller_id,
                    action="merge",
                    diff_snippet=f"Merged from workspace {caller_id}"
                )
            return res
        elif tool_name == "run_terminal_cmd":
            return self.tool_manager.run_terminal_cmd(args.get("command", ""))
        elif tool_name == "git_status":
            return self.tool_manager.git_status()
        elif tool_name == "git_diff":
            return self.tool_manager.git_diff()
        return {"success": False, "error": f"Unknown tool: {tool_name}"}

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
