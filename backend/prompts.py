"""
Prompt Template Engine for SwarmChat.
Supports placeholders:
  %r - Base role name (e.g. Architect, Coder)
  %a - Apex career title (e.g. Head Systems Architect, Senior Lead Programmer)
  %n - Model name (e.g. Otis, Bill)
  %p - Project path and goals summary
  %t - Current active task/agenda item
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List

from backend.errors import MemoryPersistenceError

logger = logging.getLogger(__name__)

SETTINGS_FILE = os.path.join(".swarmchat", "prompt_templates.json")

# Mapping of standard roles to apex-career titles
APEX_TITLES = {
    "Architect": "Head Systems Architect & Technical Director",
    "Coder": "Senior Lead Software Engineer & Principal Developer",
    "Critic": "Chief QA & Code Security Auditor",
    "Solver": "Principal Algorithms Specialist & Systems Analyst",
    "Tester/Debugger": "Lead Quality Assurance & Automated Test Engineer"
}

# Default templates designed to be under 300 tokens, strict and clear for unaligned/Dolphin models.
DEFAULT_START_PROMPT = """You are %n (%r) in project %p. Task: %t. Phase: Discussion (planning gate).

The room is walking a fixed sequence before any code is written: the Architect lists what is
genuinely undecided, the room settles those questions ONE PER TURN, the Architect writes the
build plan against those decisions, the Critic reviews it, the Programmer confirms it is
buildable, then the Architect - and only the Architect - opens Execution. Speak to the step
you are on. A settled question stays settled - do not re-open it. File-writing tools are
locked until Execution opens.

DIRECTIVES:
1. NO THINKING BLOCKS or name headers.
2. Save notes and specs to indexed memory across context resets:
   - `[SAVE_NOTE: <100-300 token chunk>]` save indexed note chunk.
   - `[LOG_TO_MEMORY: <decision>]` save shared memory entry.
   - `[UPDATE_SPEC: <spec>]` update spec file.
   - `[UPDATE_TASK: title=<title>, status=<pending|in_progress>]` update task itinerary.
   - `[READY_FOR_EXECUTION]` Architect only, and only once the plan has cleared review.
3. Reference peer code with `@bot_name:filename` or `workspace://<bot_id>/<file>`."""

DEFAULT_EXECUTION_PROMPT = """You are %n (%r) in project %p. Task: %t. Phase: Execution.

DIRECTIVES:
1. NO THINKING BLOCKS or name headers.
2. Code generated in chat blocks is auto-saved to your workspace sandbox (`.swarmchat/workspaces/%n/`).
3. Reference peer code with `@bot_name:filename` or `workspace://<bot_id>/<file>`.
4. ACTION TAGS:
   - `[SAVE_NOTE: <chunk>]` save indexed note chunk.
   - `[LOG_TO_MEMORY: <log>]` log completion or test result.
   - `[UPDATE_TASK: id=<task_id>, status=<completed>]` mark finished.
   - `[REQUEST_DISCUSSION]` if ambiguous."""

ROLE_DEFINITIONS = {
    "Architect": {
        "description": "Design high-level architecture, component boundaries, and overall project blueprints.",
        "icon": "🏗️",
        "apex_title": APEX_TITLES["Architect"]
    },
    "Critic": {
        "description": "Red-team solutions, identify edge cases, vulnerabilities, performance risks, and missing requirements.",
        "icon": "🧐",
        "apex_title": APEX_TITLES["Critic"]
    },
    "Solver": {
        "description": "Analyze core algorithms, math logic, and step-by-step problem breakdown.",
        "icon": "💡",
        "apex_title": APEX_TITLES["Solver"]
    },
    "Coder": {
        "description": "Write clean, modular, production-ready code files and patches.",
        "icon": "💻",
        "apex_title": APEX_TITLES["Coder"]
    },
    "Tester/Debugger": {
        "description": "Write test suites, reproduce bugs, verify edge cases, and validate functionality.",
        "icon": "🧪",
        "apex_title": APEX_TITLES["Tester/Debugger"]
    }
}

class PromptTemplateManager:
    def __init__(self, storage_path: str = SETTINGS_FILE):
        self.storage_path = storage_path
        self.templates = {
            "start_prompt": DEFAULT_START_PROMPT,
            "execution_prompt": DEFAULT_EXECUTION_PROMPT,
            "custom_role_prompts": {},
            "per_model_prompts": {}  # model_id -> {"start_prompt": str, "execution_prompt": str}
        }
        self.last_load_error: Optional[str] = None
        self.load_templates()

    def load_templates(self):
        """Loads saved templates, keeping defaults and recording the reason when the file is unusable."""
        self.last_load_error = None
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.last_load_error = f"Could not load prompt templates from '{self.storage_path}': {e}"
            logger.error(self.last_load_error)
            return
        if not isinstance(data, dict):
            self.last_load_error = (
                f"Prompt template file '{self.storage_path}' contains {type(data).__name__}, expected object."
            )
            logger.error(self.last_load_error)
            return
        self.templates.update(data)

    def save_templates(self):
        """Persists templates, raising so an API caller learns the edit was not stored."""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.templates, f, indent=2)
        except (OSError, TypeError, ValueError) as e:
            logger.exception("Failed to persist prompt templates to %s", self.storage_path)
            raise MemoryPersistenceError(
                f"Failed to persist prompt templates to '{self.storage_path}': {e}"
            ) from e

    def update_templates(
        self,
        start_prompt: Optional[str] = None,
        execution_prompt: Optional[str] = None,
        custom_role_prompts: Optional[Dict[str, Any]] = None,
        per_model_prompts: Optional[Dict[str, Any]] = None
    ):
        if start_prompt is not None:
            self.templates["start_prompt"] = start_prompt
        if execution_prompt is not None:
            self.templates["execution_prompt"] = execution_prompt
        if custom_role_prompts is not None:
            self.templates["custom_role_prompts"] = custom_role_prompts
        if per_model_prompts is not None:
            self.templates["per_model_prompts"] = per_model_prompts
        self.save_templates()

    def apply_batch_prompt(self, model_ids: List[str], start_prompt: Optional[str] = None, execution_prompt: Optional[str] = None):
        """Applies a custom start/execution prompt across multiple selected models."""
        per_model = self.templates.get("per_model_prompts", {})
        for m_id in model_ids:
            if m_id not in per_model:
                per_model[m_id] = {}
            if start_prompt is not None:
                per_model[m_id]["start_prompt"] = start_prompt
            if execution_prompt is not None:
                per_model[m_id]["execution_prompt"] = execution_prompt
        self.templates["per_model_prompts"] = per_model
        self.save_templates()

    def get_apex_title(self, role: str) -> str:
        return APEX_TITLES.get(role, f"Senior {role} Specialist")

    def format_prompt(
        self,
        phase: str,
        role: str,
        name: str,
        project_info: str = "Default Workspace",
        current_task: str = "General Discussion / Alignment",
        custom_template: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> str:
        apex_title = self.get_apex_title(role)

        # Precedence: 1) explicit custom_template arg, 2) per-model custom prompt, 3) global default
        template = custom_template
        if not template and model_id and model_id in self.templates.get("per_model_prompts", {}):
            m_prompts = self.templates["per_model_prompts"][model_id]
            key = "execution_prompt" if phase.lower() == "execution" else "start_prompt"
            if m_prompts.get(key):
                template = m_prompts[key]

        if not template or not template.strip():
            if phase.lower() == "execution":
                template = self.templates.get("execution_prompt", DEFAULT_EXECUTION_PROMPT)
            else:
                template = self.templates.get("start_prompt", DEFAULT_START_PROMPT)

        formatted = (
            template
            .replace("%a", apex_title)
            .replace("%r", role)
            .replace("%n", name)
            .replace("%p", project_info)
            .replace("%t", current_task)
        )
        return formatted

prompt_template_mgr = PromptTemplateManager()

def get_system_prompt(
    role: str,
    name: str = "Bot",
    phase: str = "discussion",
    project_info: str = "Default Project Context",
    current_task: str = "No active task assigned",
    is_moderator: bool = False,
    custom_template: Optional[str] = None,
    model_id: Optional[str] = None
) -> str:
    base_prompt = prompt_template_mgr.format_prompt(
        phase=phase,
        role=role,
        name=name,
        project_info=project_info,
        current_task=current_task,
        custom_template=custom_template,
        model_id=model_id
    )

    # NOTE: `is_moderator` deliberately no longer changes the prompt.
    #
    # It used to append a "CHIEF PROJECT MANAGER" mandate telling the model to "coordinate
    # participant turns" - a job it does not have and cannot do. Speaker selection is entirely
    # deterministic Python (Orchestrator.get_next_speaker / _select_execution_speaker route off
    # task status, @mentions and the turn schedule); no model is ever asked who speaks next.
    # The rest of that block - break the project into small testable tasks, revise a task that
    # keeps failing - simply restated the Architect output contract in different words, so a
    # 3-4B Architect was carrying two overlapping mandates plus one fictional one in a single
    # context window. Both now live in exactly one place: ROLE_OUTPUT_CONTRACTS["architect"]
    # in orchestrator.py.
    #
    # The flag itself has since been REMOVED from model configs entirely (Architect and
    # Moderator are one seat, named by `role`). This parameter is retained only so older
    # callers/tests keep working; it is ignored.

    return base_prompt
