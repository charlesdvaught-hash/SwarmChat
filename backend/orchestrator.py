import time
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

        self.models: Dict[str, Dict[str, Any]] = {
            "model_architect": {
                "id": "model_architect",
                "name": "Architect",
                "role": "Architect",
                "provider": "ollama",
                "model_name": "llama3.2:1b",
                "enabled": True,
                "is_moderator": True,
                "status": "active"
            },
            "model_critic": {
                "id": "model_critic",
                "name": "Critic",
                "role": "Critic",
                "provider": "ollama",
                "model_name": "llama3.2:1b",
                "enabled": True,
                "is_moderator": False,
                "status": "active"
            },
            "model_coder": {
                "id": "model_coder",
                "name": "Coder",
                "role": "Coder",
                "provider": "ollama",
                "model_name": "llama3.2:1b",
                "enabled": True,
                "is_moderator": False,
                "status": "active"
            }
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
            self.moderator_model_id = model_id

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
        active_models = [m_id for m_id, m in self.models.items() if m["enabled"] and m["status"] == "active"]
        if not active_models:
            return None

        if self.turn_mode == "round_robin":
            if not last_speaker_id or last_speaker_id not in active_models:
                return active_models[0]
            idx = active_models.index(last_speaker_id)
            return active_models[(idx + 1) % len(active_models)]

        elif self.turn_mode == "moderator_controlled":
            return self.moderator_model_id if self.moderator_model_id in active_models else active_models[0]

        return None

    async def step_model_turn(self, model_id: str) -> Dict[str, Any]:
        model_cfg = self.models.get(model_id)
        if not model_cfg or not model_cfg["enabled"]:
            return {"error": "Model not available"}

        current_phase = self.memory_manager.get_phase()
        sys_prompt = get_system_prompt(
            role=model_cfg["role"],
            phase=current_phase,
            is_moderator=model_cfg.get("is_moderator", False)
        )

        memory_summary = self.memory_manager.get_memory_summary()
        context_prompt = f"{sys_prompt}\n\n### SHARED MEMORY SUMMARY:\n{memory_summary}"

        recent_msgs = [
            {"role": "user" if m["is_admin"] else "assistant", "content": f"[{m['sender']} ({m['role']})]: {m['content']}"}
            for m in self.chat_history[-8:]
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
        elif "[REQUEST_DISCUSSION]" in response_text:
            self.memory_manager.add_entry(model_cfg["name"], "Requested return to Discussion Phase due to ambiguity.")
        elif "[LOG_TO_MEMORY:" in response_text:
            try:
                start = response_text.find("[LOG_TO_MEMORY:") + len("[LOG_TO_MEMORY:")
                end = response_text.find("]", start)
                if end != -1:
                    mem_content = response_text[start:end].strip()
                    self.memory_manager.add_entry(model_cfg["name"], mem_content)
            except Exception:
                pass

        if "[REQUEST_NAP]" in response_text:
            self.memory_manager.record_model_nap(model_id, f"{model_cfg['name']} completed a context nap.")

        msg = self.add_chat_message(
            sender=model_cfg["name"],
            role=model_cfg["role"],
            content=response_text,
            is_admin=False,
            model_id=model_id
        )

        return msg

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
            exec_res = self._execute_tool(tool_name, args)
            vote_req["status"] = "executed"
            vote_req["result"] = exec_res
            return vote_req

        self.pending_tool_votes.append(vote_req)
        return vote_req

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "read_file":
            return self.tool_manager.read_file(args.get("filepath", ""))
        elif tool_name == "list_files":
            return self.tool_manager.list_files(args.get("dir", "."))
        elif tool_name == "search_workspace":
            return self.tool_manager.search_workspace(args.get("query", ""))
        elif tool_name == "write_file":
            return self.tool_manager.write_file(args.get("filepath", ""), args.get("content", ""))
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
                    exec_res = self._execute_tool(req["tool_name"], args)
                    req["status"] = "executed"
                    req["result"] = exec_res
                    return {"success": True, "executed": True, "result": exec_res}
                elif action == "reject":
                    req["status"] = "rejected"
                    return {"success": True, "status": "rejected"}
        return {"success": False, "error": "Vote ID not found"}
