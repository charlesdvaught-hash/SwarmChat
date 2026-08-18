import os
import json
import time
from typing import Dict, Any, List, Optional

from backend.utils import format_clock, format_datetime, timestamped_id, write_json_file, write_text_file

class MemoryManager:
    def __init__(self, storage_dir: str = ".swarmchat", project_id: str = "default_project"):
        self.base_storage_dir = storage_dir
        self.project_id = project_id
        self.project_dir = os.path.join(self.base_storage_dir, "projects", self.project_id)
        os.makedirs(self.project_dir, exist_ok=True)

        self.json_path = os.path.join(self.project_dir, "shared_memory.json")
        self.md_path = os.path.join(self.project_dir, "shared_memory.md")

        self.state: Dict[str, Any] = {
            "project_id": self.project_id,
            "phase": "discussion",
            "phase_last_changed": time.time(),
            "shared_entries": [],
            "model_journals": {},
            "model_spec_files": {},  # Per-model spec files/notebooks (model_id -> str content)
            "episodes": [],  # Structured NAC-style episodes
            "task_itinerary": [],  # Meeting itinerary tasks
            "file_audit_log": [],  # File edits and user/model attribution
            "active_file_locks": {},  # Currently edited files
            "tokens_used": {},
            "session_id": "default_session"
        }
        self.load_memory()

    def get_project_id(self) -> str:
        return self.project_id

    def set_project_id(self, project_id: str):
        if project_id and project_id != self.project_id:
            self.save_memory()
            self.project_id = project_id
            self.project_dir = os.path.join(self.base_storage_dir, "projects", self.project_id)
            os.makedirs(self.project_dir, exist_ok=True)
            self.json_path = os.path.join(self.project_dir, "shared_memory.json")
            self.md_path = os.path.join(self.project_dir, "shared_memory.md")
            self.load_memory()

    def load_memory(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    self.state.update(saved)
            except Exception as e:
                print(f"Error loading shared memory: {e}")

    def save_memory(self):
        try:
            write_json_file(self.json_path, self.state)
            self._render_markdown_archive()
        except Exception as e:
            print(f"Error saving shared memory: {e}")

    def _render_markdown_archive(self):
        try:
            lines = [
                f"# 🧠 SwarmChat Continuous Shared Memory Archive — Project: `{self.project_id}`",
                f"**Session ID:** `{self.state.get('session_id', 'default_session')}`",
                f"**Current Phase:** `{self.state.get('phase', 'discussion').upper()}`",
                f"**Last Phase Switch:** {format_datetime(self.state.get('phase_last_changed'))}",
                "\n---",
                "## 📌 Key Decisions & Shared Memory Entries\n"
            ]

            entries = self.state.get("shared_entries", [])
            if not entries:
                lines.append("*No shared entries logged yet.*")
            else:
                for entry in entries:
                    timestamp = format_clock(entry.get("timestamp"))
                    author = entry.get("author", "Unknown")
                    content = entry.get("content", "")
                    lines.append(f"- **[{timestamp}] {author}:** {content}")

            lines.append("\n---\n## 📦 Episodes & Thread Checkpoints (NAC Architecture)\n")
            episodes = self.state.get("episodes", [])
            if not episodes:
                lines.append("*No episodes recorded yet.*")
            else:
                for ep in episodes:
                    t_str = format_clock(ep.get("timestamp"))
                    lines.append(f"### Episode `{ep.get('id')}` — [{ep.get('author')}] ({ep.get('thread_name', 'main')})")
                    lines.append(f"- **Time:** {t_str}")
                    lines.append(f"- **Action / Task:** {ep.get('action', '')}")
                    lines.append(f"- **Accomplishments:** {ep.get('summary', '')}")
                    if ep.get("modified_files"):
                        lines.append(f"- **Modified Files:** {', '.join(ep.get('modified_files'))}")

            lines.append("\n---\n## 💤 Model Context Journals & Naps\n")
            journals = self.state.get("model_journals", {})
            if not journals:
                lines.append("*No model naps or context refreshes logged yet.*")
            else:
                for model_id, logs in journals.items():
                    lines.append(f"### 🤖 Model: `{model_id}`")
                    for log in logs:
                        t_str = format_clock(log.get("timestamp"))
                        lines.append(f"- **[{t_str}] Journal Summary:** {log.get('summary', '')}")

            write_text_file(self.md_path, "\n".join(lines))
        except Exception as e:
            print(f"Error rendering markdown archive: {e}")

    def set_phase(self, new_phase: str) -> str:
        if new_phase.lower() in ["discussion", "execution"]:
            old_phase = self.state["phase"]
            self.state["phase"] = new_phase.lower()
            self.state["phase_last_changed"] = time.time()
            self.add_entry(
                author="System State Machine",
                content=f"Phase switched from '{old_phase.upper()}' to '{new_phase.upper()}'."
            )
            self.save_memory()
            return self.state["phase"]
        return self.state["phase"]

    def get_phase(self) -> str:
        return self.state.get("phase", "discussion")

    def add_entry(self, author: str, content: str):
        entry = {
            "id": timestamped_id("mem"),
            "timestamp": time.time(),
            "author": author,
            "content": content
        }
        self.state.setdefault("shared_entries", []).append(entry)
        self.save_memory()

    def record_episode(
        self,
        author: str,
        summary: str,
        action: str = "General Task Handoff",
        thread_name: str = "main",
        modified_files: Optional[List[str]] = None,
        source_threads: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        ep_id = timestamped_id("ep")
        episode = {
            "id": ep_id,
            "timestamp": time.time(),
            "author": author,
            "action": action,
            "summary": summary,
            "thread_name": thread_name,
            "modified_files": modified_files or [],
            "source_threads": source_threads or []
        }
        self.state.setdefault("episodes", []).append(episode)
        self.add_entry(
            author=author,
            content=f"📦 [EPISODE CHECKPOINT #{ep_id}] {action}: {summary}"
        )
        self.save_memory()
        return episode

    def record_model_nap(self, model_id: str, summary: str, action: str = "Context Nap Handoff", modified_files: Optional[List[str]] = None):
        log = {
            "timestamp": time.time(),
            "summary": summary
        }
        self.state.setdefault("model_journals", {}).setdefault(model_id, []).append(log)
        
        # Save model-isolated journal in project models folder
        model_dir = os.path.join(self.project_dir, "models", model_id)
        os.makedirs(model_dir, exist_ok=True)
        journal_path = os.path.join(model_dir, "journal.json")
        try:
            write_json_file(journal_path, self.state["model_journals"][model_id])
        except Exception:
            pass

        # Automatically generate a NAC-style Episode checkpoint on Nap
        self.record_episode(
            author=f"Model ({model_id})",
            summary=summary,
            action=action,
            thread_name=model_id,
            modified_files=modified_files
        )
        
        self.add_entry(
            author=f"Moderator System ({model_id})",
            content=f"Model `{model_id}` took a nap / refreshed context. Shared summary: {summary}"
        )
        self.state.setdefault("tokens_used", {})[model_id] = 0
        self.save_memory()

    def get_latest_episodes(self, limit: int = 5) -> List[Dict[str, Any]]:
        episodes = self.state.get("episodes", [])
        return episodes[-limit:] if episodes else []

    # --- TASK ITINERARY & MEETINGS ---
    def add_itinerary_task(self, title: str, description: str, priority: str = "medium", assigned_model: Optional[str] = None) -> Dict[str, Any]:
        task = {
            "id": timestamped_id("task"),
            "title": title,
            "description": description,
            "priority": priority,  # high, medium, low
            "status": "pending",  # pending, in_progress, completed
            "assigned_model": assigned_model,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        self.state.setdefault("task_itinerary", []).append(task)
        self.save_memory()
        return task

    def update_itinerary_task(self, task_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for task in self.state.get("task_itinerary", []):
            if task["id"] == task_id:
                task.update(updates)
                task["updated_at"] = time.time()
                self.save_memory()
                return task
        return None

    def get_task_itinerary(self) -> List[Dict[str, Any]]:
        return self.state.get("task_itinerary", [])

    def get_active_task(self) -> Optional[Dict[str, Any]]:
        tasks = self.state.get("task_itinerary", [])
        for t in tasks:
            if t["status"] == "in_progress":
                return t
        for t in tasks:
            if t["status"] == "pending":
                return t
        return None

    # --- FILE AUDIT LOG & ATTRIBUTION ---
    def log_file_edit(self, filepath: str, author: str, action: str, diff_snippet: Optional[str] = None):
        entry = {
            "id": timestamped_id("audit"),
            "timestamp": time.time(),
            "filepath": filepath,
            "author": author,
            "action": action,  # write, create, patch
            "diff_snippet": diff_snippet
        }
        self.state.setdefault("file_audit_log", []).append(entry)
        self.state.setdefault("active_file_locks", {})[filepath] = {
            "last_edited_by": author,
            "last_edited_at": time.time(),
            "status": "edited"
        }
        self.save_memory()
        return entry

    def get_file_audit_log(self, filepath: Optional[str] = None) -> List[Dict[str, Any]]:
        log = self.state.get("file_audit_log", [])
        if filepath:
            return [e for e in log if e["filepath"] == filepath]
        return log

    def update_token_usage(self, model_id: str, added_tokens: int) -> int:
        current = self.state.setdefault("tokens_used", {}).get(model_id, 0)
        new_total = current + added_tokens
        self.state["tokens_used"][model_id] = new_total
        self.save_memory()
        return new_total

    def get_memory_summary(self) -> str:
        entries = self.state.get("shared_entries", [])
        if not entries:
            return "No shared memory entries recorded yet."
        summary_lines = [f"- [{e.get('author')}]: {e.get('content')}" for e in entries[-10:]]
        return "\n".join(summary_lines)

    def get_model_latest_journal(self, model_id: str) -> Optional[str]:
        journals = self.state.get("model_journals", {}).get(model_id, [])
        if journals:
            return journals[-1].get("summary", "")
        return None

    # --- SPEC FILES / NOTEBOOKS ---
    def get_spec_file(self, model_id: str) -> str:
        """Returns the model's dedicated spec file content."""
        return self.state.setdefault("model_spec_files", {}).get(model_id, "")

    def update_spec_file(self, model_id: str, content: str) -> str:
        """Updates the model's dedicated spec file."""
        self.state.setdefault("model_spec_files", {})[model_id] = content

        # Save to file
        model_dir = os.path.join(self.project_dir, "models", model_id)
        os.makedirs(model_dir, exist_ok=True)
        spec_path = os.path.join(model_dir, "spec_file.md")
        try:
            write_text_file(spec_path, content)
        except Exception:
            pass

        self.save_memory()
        return content

    def get_all_spec_files(self) -> Dict[str, str]:
        """Returns all models' spec files for selective reading."""
        return self.state.get("model_spec_files", {})
