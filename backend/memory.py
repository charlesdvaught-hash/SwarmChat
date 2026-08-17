import os
import json
import time
from typing import Dict, Any, List, Optional

class MemoryManager:
    def __init__(self, storage_dir: str = ".swarmchat"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.json_path = os.path.join(self.storage_dir, "shared_memory.json")
        self.md_path = os.path.join(self.storage_dir, "shared_memory.md")

        self.state: Dict[str, Any] = {
            "phase": "discussion",
            "phase_last_changed": time.time(),
            "shared_entries": [],
            "model_journals": {},
            "tokens_used": {},
            "session_id": "default_session"
        }
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
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
            self._render_markdown_archive()
        except Exception as e:
            print(f"Error saving shared memory: {e}")

    def _render_markdown_archive(self):
        try:
            lines = [
                "# 🧠 SwarmChat Continuous Shared Memory Archive",
                f"**Session ID:** `{self.state.get('session_id', 'default_session')}`",
                f"**Current Phase:** `{self.state.get('phase', 'discussion').upper()}`",
                f"**Last Phase Switch:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.state.get('phase_last_changed', time.time())))}",
                "\n---",
                "## 📌 Key Decisions & Shared Memory Entries\n"
            ]

            entries = self.state.get("shared_entries", [])
            if not entries:
                lines.append("*No shared entries logged yet.*")
            else:
                for entry in entries:
                    timestamp = time.strftime('%H:%M:%S', time.localtime(entry.get("timestamp", time.time())))
                    author = entry.get("author", "Unknown")
                    content = entry.get("content", "")
                    lines.append(f"- **[{timestamp}] {author}:** {content}")

            lines.append("\n---\n## 💤 Model Context Journals & Naps\n")
            journals = self.state.get("model_journals", {})
            if not journals:
                lines.append("*No model naps or context refreshes logged yet.*")
            else:
                for model_id, logs in journals.items():
                    lines.append(f"### 🤖 Model: `{model_id}`")
                    for log in logs:
                        t_str = time.strftime('%H:%M:%S', time.localtime(log.get("timestamp", time.time())))
                        lines.append(f"- **[{t_str}] Journal Summary:** {log.get('summary', '')}")

            with open(self.md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
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
            "id": f"mem_{int(time.time() * 1000)}",
            "timestamp": time.time(),
            "author": author,
            "content": content
        }
        self.state.setdefault("shared_entries", []).append(entry)
        self.save_memory()

    def record_model_nap(self, model_id: str, summary: str):
        log = {
            "timestamp": time.time(),
            "summary": summary
        }
        self.state.setdefault("model_journals", {}).setdefault(model_id, []).append(log)
        self.add_entry(
            author=f"Moderator System ({model_id})",
            content=f"Model `{model_id}` took a nap / refreshed context. Shared summary: {summary}"
        )
        self.state.setdefault("tokens_used", {})[model_id] = 0
        self.save_memory()

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
