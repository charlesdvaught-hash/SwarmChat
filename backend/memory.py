import os
import json
import logging
import time
from typing import Dict, Any, List, Optional

from backend.errors import MemoryPersistenceError
from backend.tools import slugify_project_id

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, storage_dir: str = ".swarmchat", project_id: str = "default_project"):
        self.base_storage_dir = storage_dir
        self.project_id = slugify_project_id(project_id)
        self.project_dir = os.path.join(self.base_storage_dir, "projects", self.project_id)
        os.makedirs(self.project_dir, exist_ok=True)

        self.json_path = os.path.join(self.project_dir, "shared_memory.json")
        self.md_path = os.path.join(self.project_dir, "shared_memory.md")

        self.state: Dict[str, Any] = self._fresh_state()
        # Startup must not hard-fail on a damaged archive, but the failure has to stay visible:
        # load_memory() quarantines the bad file and the reason is exposed via last_load_error.
        self.last_load_error: Optional[str] = None
        try:
            self.load_memory()
        except MemoryPersistenceError as e:
            self.last_load_error = str(e)
            logger.error("Shared memory could not be loaded: %s", e)

    def _fresh_state(self) -> Dict[str, Any]:
        """A blank archive for the current project. load_memory() layers a saved
        archive on top of this, so it must be rebuilt on every project switch -
        otherwise a project with no file on disk inherits the previous project's
        tasks and conversation."""
        return {
            "project_id": self.project_id,
            "phase": "discussion",
            "phase_last_changed": time.time(),
            # Where the room is inside the pre-execution planning gate. Discussion used to be
            # unstructured - whoever the roster happened to pick talked until a turn/time cap
            # force-flipped the phase, so a plan could reach execution with nobody having
            # reviewed it. See MemoryManager.PLAN_STAGES.
            "plan_stage": "awaiting_plan",
            "plan_revision": 0,
            "shared_entries": [],
            "model_journals": {},
            "model_spec_files": {},  # Per-model spec files/notebooks (model_id -> str content)
            "model_notes": {},  # Per-model indexed note chunks (model_id -> List[Dict])
            "episodes": [],  # Structured NAC-style episodes
            "task_itinerary": [],  # Meeting itinerary tasks
            "file_audit_log": [],  # File edits and user/model attribution
            "active_file_locks": {},  # Currently edited files
            "tokens_used": {},
            "session_id": "default_session"
        }

    def get_project_id(self) -> str:
        return self.project_id

    def set_project_id(self, project_id: str):
        project_id = slugify_project_id(project_id)
        if project_id and project_id != self.project_id:
            self.save_memory()
            self.project_id = project_id
            self.project_dir = os.path.join(self.base_storage_dir, "projects", self.project_id)
            os.makedirs(self.project_dir, exist_ok=True)
            self.json_path = os.path.join(self.project_dir, "shared_memory.json")
            self.md_path = os.path.join(self.project_dir, "shared_memory.md")
            # Start from a blank archive so nothing leaks across the project boundary.
            self.state = self._fresh_state()
            self.last_load_error = None
            try:
                self.load_memory()
            except MemoryPersistenceError as e:
                self.last_load_error = str(e)
                logger.error("Shared memory for project '%s' could not be loaded: %s", project_id, e)
            self.state["project_id"] = self.project_id

    def load_memory(self):
        """Loads persisted state, quarantining an unreadable archive instead of overwriting it."""
        if not os.path.exists(self.json_path):
            return
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            quarantine_path = f"{self.json_path}.corrupt-{int(time.time())}"
            try:
                os.replace(self.json_path, quarantine_path)
            except OSError:
                logger.exception("Could not quarantine unreadable memory archive %s", self.json_path)
                raise MemoryPersistenceError(
                    f"Shared memory at '{self.json_path}' is unreadable and could not be quarantined: {e}"
                ) from e
            raise MemoryPersistenceError(
                f"Shared memory at '{self.json_path}' was unreadable ({e}); "
                f"it has been preserved at '{quarantine_path}' and a fresh archive will be started."
            ) from e

        if not isinstance(saved, dict):
            raise MemoryPersistenceError(
                f"Shared memory at '{self.json_path}' has unexpected type {type(saved).__name__}, expected object."
            )
        self.state.update(saved)

    def save_memory(self):
        """Persists state to disk. Raises MemoryPersistenceError so callers never assume a silent success."""
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except (OSError, TypeError, ValueError) as e:
            logger.exception("Failed to persist shared memory to %s", self.json_path)
            raise MemoryPersistenceError(f"Failed to persist shared memory to '{self.json_path}': {e}") from e
        self._render_markdown_archive()

    def _render_markdown_archive(self):
        try:
            lines = [
                f"# 🧠 SwarmChat Continuous Shared Memory Archive — Project: `{self.project_id}`",
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

            lines.append("\n---\n## 📦 Episodes & Thread Checkpoints (NAC Architecture)\n")
            episodes = self.state.get("episodes", [])
            if not episodes:
                lines.append("*No episodes recorded yet.*")
            else:
                for ep in episodes:
                    t_str = time.strftime('%H:%M:%S', time.localtime(ep.get("timestamp", time.time())))
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
                        t_str = time.strftime('%H:%M:%S', time.localtime(log.get("timestamp", time.time())))
                        lines.append(f"- **[{t_str}] Journal Summary:** {log.get('summary', '')}")

            with open(self.md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except (OSError, TypeError, ValueError) as e:
            logger.exception("Failed to render markdown archive to %s", self.md_path)
            raise MemoryPersistenceError(f"Failed to render markdown archive '{self.md_path}': {e}") from e

    VALID_PHASES = ("discussion", "execution")

    def set_phase(self, new_phase: str) -> str:
        """Switches phase. Raises ValueError on an unknown phase instead of silently keeping the old one."""
        normalized = (new_phase or "").strip().lower()
        if normalized not in self.VALID_PHASES:
            raise ValueError(
                f"Unknown phase '{new_phase}'. Valid phases: {', '.join(self.VALID_PHASES)}."
            )
        old_phase = self.state["phase"]
        self.state["phase"] = normalized
        self.state["phase_last_changed"] = time.time()
        self.add_entry(
            author="System State Machine",
            content=f"Phase switched from '{old_phase.upper()}' to '{normalized.upper()}'."
        )
        self.save_memory()
        return self.state["phase"]

    def get_phase(self) -> str:
        return self.state.get("phase", "discussion")

    # --- PRE-EXECUTION PLANNING GATE ---
    # Discussion is no longer a free-for-all. The room walks a fixed sequence before any
    # file can be written, and only the Architect closes it out:
    #   awaiting_plan     -> Architect proposes / rehashes the build plan
    #   critic_review     -> Critic hunts for weak or contradictory parts
    #   programmer_review -> Coder confirms the plan is actually buildable
    #   approved          -> Architect may call [READY_FOR_EXECUTION]
    # A rejection at either review step returns to awaiting_plan, so the Architect rehashes
    # rather than the room drifting into execution on an unreviewed plan.
    PLAN_STAGES = ("awaiting_plan", "critic_review", "programmer_review", "approved")

    def set_plan_stage(self, stage: str) -> str:
        normalized = (stage or "").strip().lower()
        if normalized not in self.PLAN_STAGES:
            raise ValueError(
                f"Unknown plan stage '{stage}'. Valid stages: {', '.join(self.PLAN_STAGES)}."
            )
        old = self.get_plan_stage()
        if old == normalized:
            return normalized
        self.state["plan_stage"] = normalized
        if old == "awaiting_plan" and normalized != "awaiting_plan":
            self.state["plan_revision"] = self.get_plan_revision() + 1
        self.save_memory()
        return normalized

    def get_plan_stage(self) -> str:
        return self.state.get("plan_stage", "awaiting_plan")

    def get_plan_revision(self) -> int:
        try:
            return int(self.state.get("plan_revision", 0))
        except (TypeError, ValueError):
            return 0

    def reset_plan_gate(self) -> str:
        """Back to square one - a new project, or a return to discussion, starts planning over."""
        self.state["plan_stage"] = "awaiting_plan"
        self.state["plan_revision"] = 0
        self.save_memory()
        return "awaiting_plan"

    def add_entry(self, author: str, content: str):
        entry = {
            "id": f"mem_{int(time.time() * 1000)}",
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
        ep_id = f"ep_{int(time.time() * 1000)}"
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
            with open(journal_path, "w", encoding="utf-8") as f:
                json.dump(self.state["model_journals"][model_id], f, indent=2)
        except (OSError, TypeError, ValueError) as e:
            logger.exception("Failed to write per-model journal for %s", model_id)
            raise MemoryPersistenceError(f"Failed to write journal for model '{model_id}': {e}") from e

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
            "id": f"task_{int(time.time() * 1000)}",
            "title": title,
            "description": description,
            "priority": priority,  # high, medium, low
            # pending, in_progress, needs_review, needs_test, failed, completed
            "status": "pending",
            "assigned_model": assigned_model,
            # Execution-pipeline bookkeeping: which file the task produced, who
            # actually wrote it (so Critic/Tester can look at the right
            # workspace), why it was last kicked back, and how many times.
            "filename": None,
            "author_bot_id": None,
            "blocked_reason": None,
            "attempt_count": 0,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        self.state.setdefault("task_itinerary", []).append(task)
        self.save_memory()
        return task

    def update_itinerary_task(self, task_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Applies updates to a task, or returns None when the task id is unknown."""
        for task in self.state.get("task_itinerary", []):
            if task["id"] == task_id:
                # Backfill fields for tasks created before this schema existed.
                task.setdefault("filename", None)
                task.setdefault("author_bot_id", None)
                task.setdefault("blocked_reason", None)
                task.setdefault("attempt_count", 0)
                task.update(updates)
                task["updated_at"] = time.time()
                self.save_memory()
                return task
        return None

    def delete_itinerary_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Removes a task outright. Returns the deleted task (so the caller can trash
        the files it produced), or None if the id was unknown."""
        tasks = self.state.get("task_itinerary", [])
        for i, task in enumerate(tasks):
            if task.get("id") == task_id:
                removed = tasks.pop(i)
                self.save_memory()
                logger.info("Deleted itinerary task %s ('%s')", task_id, removed.get("title"))
                return removed
        return None

    # --- PROJECTS ---
    def _projects_root(self) -> str:
        root = os.path.join(self.base_storage_dir, "projects")
        os.makedirs(root, exist_ok=True)
        return root

    def list_projects(self) -> List[Dict[str, Any]]:
        """Every project on disk, with a task count so the UI can show what is in each."""
        root = self._projects_root()
        projects: List[Dict[str, Any]] = []
        for name in sorted(os.listdir(root)):
            pdir = os.path.join(root, name)
            if not os.path.isdir(pdir):
                continue
            task_count = 0
            open_count = 0
            if name == self.project_id:
                tasks = self.state.get("task_itinerary", [])
            else:
                tasks = self._read_project_state(name).get("task_itinerary", [])
            task_count = len(tasks)
            open_count = len([t for t in tasks if t.get("status") != "completed"])
            projects.append({
                "project_id": name,
                "is_active": name == self.project_id,
                "task_count": task_count,
                "open_task_count": open_count,
                "updated_at": os.path.getmtime(pdir)
            })
        return projects

    def _project_json_path(self, project_id: str) -> str:
        return os.path.join(self._projects_root(), slugify_project_id(project_id), "shared_memory.json")

    def _read_project_state(self, project_id: str) -> Dict[str, Any]:
        """Reads another project's archive off disk without switching to it."""
        path = self._project_json_path(project_id)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read project '%s': %s", project_id, e)
            return {}

    def _write_project_state(self, project_id: str, state: Dict[str, Any]):
        path = self._project_json_path(project_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, path)
        except OSError as e:
            raise MemoryPersistenceError(f"Could not write project '{project_id}': {e}") from e

    def create_project(self, project_id: str) -> Dict[str, Any]:
        pid = slugify_project_id(project_id)
        pdir = os.path.join(self._projects_root(), pid)
        existed = os.path.isdir(pdir)
        os.makedirs(pdir, exist_ok=True)
        return {"project_id": pid, "created": not existed}

    def move_task_to_project(self, task_id: str, target_project_id: str) -> Optional[Dict[str, Any]]:
        """Moves a task from the active project into another project's itinerary.
        The task's produced files stay in the source project's workspaces - the
        target project's models start it fresh, which is the point of moving it."""
        target = slugify_project_id(target_project_id)
        if target == self.project_id:
            return None
        task = None
        for t in self.state.get("task_itinerary", []):
            if t.get("id") == task_id:
                task = t
                break
        if task is None:
            return None

        target_state = self._read_project_state(target)
        if not target_state:
            target_state = {"project_id": target, "task_itinerary": []}
        moved = dict(task)
        moved["moved_from_project"] = self.project_id
        # Files live in the source project's sandbox, so the new owner re-derives them.
        moved["filename"] = None
        moved["author_bot_id"] = None
        moved["updated_at"] = time.time()
        target_state.setdefault("task_itinerary", []).append(moved)
        self._write_project_state(target, target_state)

        self.state["task_itinerary"] = [t for t in self.state.get("task_itinerary", []) if t.get("id") != task_id]
        self.save_memory()
        logger.info("Moved task %s to project '%s'", task_id, target)
        return moved

    def get_task_itinerary(self) -> List[Dict[str, Any]]:
        return self.state.get("task_itinerary", [])

    def get_active_task(self) -> Optional[Dict[str, Any]]:
        """Picks the single task the execution pipeline should focus on right
        now. Precedence favors tasks that need someone's attention to move
        forward (failed repairs, pending review/test) over ones already being
        worked, and both of those over untouched pending work."""
        tasks = self.state.get("task_itinerary", [])
        for status in ("failed", "needs_review", "needs_test", "in_progress", "pending"):
            for t in tasks:
                if t.get("status") == status:
                    return t
        return None

    # --- FILE AUDIT LOG & ATTRIBUTION ---
    def log_file_edit(self, filepath: str, author: str, action: str, diff_snippet: Optional[str] = None):
        entry = {
            "id": f"audit_{int(time.time() * 1000)}",
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
            with open(spec_path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            logger.exception("Failed to write spec notebook for %s", model_id)
            raise MemoryPersistenceError(f"Failed to write spec file for model '{model_id}': {e}") from e

        self.save_memory()
        return content

    def get_all_spec_files(self) -> Dict[str, str]:
        """Returns all models' spec files for selective reading."""
        return self.state.get("model_spec_files", {})

    # --- INDEXED NOTE CHUNKS (100-300 TOKENS) ---
    def add_note_chunk(self, model_id: str, content: str, title: str = "Note") -> Dict[str, Any]:
        """Saves an internal note segment, chunking long content into 100-300 token segments (~150-400 words)."""
        words = content.strip().split()
        chunks = []
        chunk_size = 200  # Target ~200 words (~150-250 tokens)

        if len(words) <= chunk_size:
            chunks.append(" ".join(words))
        else:
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunks.append(" ".join(chunk_words))

        added_entries = []
        model_notes = self.state.setdefault("model_notes", {}).setdefault(model_id, [])

        for idx, chunk_text in enumerate(chunks):
            idx_title = f"{title} (Part {idx + 1}/{len(chunks)})" if len(chunks) > 1 else title
            entry = {
                "id": f"note_{model_id}_{int(time.time() * 1000)}_{idx}",
                "title": idx_title,
                "content": chunk_text,
                "timestamp": time.time(),
                "est_tokens": int(len(chunk_text.split()) * 1.3)
            }
            model_notes.append(entry)
            added_entries.append(entry)

        # Persist to model directory
        model_dir = os.path.join(self.project_dir, "models", model_id)
        os.makedirs(model_dir, exist_ok=True)
        notes_path = os.path.join(model_dir, "notes.json")
        try:
            with open(notes_path, "w", encoding="utf-8") as f:
                json.dump(model_notes, f, indent=2)
        except OSError as e:
            logger.exception("Failed to write notes store for %s", model_id)
            raise MemoryPersistenceError(f"Failed to write notes store for model '{model_id}': {e}") from e

        self.save_memory()
        return {"added_count": len(added_entries), "entries": added_entries}

    def search_note_chunks(self, model_id: str, query: str = "", limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieves and searches indexed note chunks for a model."""
        notes = self.state.get("model_notes", {}).get(model_id, [])
        if not query or not query.strip():
            return notes[-limit:]

        q_lower = query.lower()
        matched = [
            n for n in notes
            if q_lower in n.get("title", "").lower() or q_lower in n.get("content", "").lower()
        ]
        return matched[-limit:]
