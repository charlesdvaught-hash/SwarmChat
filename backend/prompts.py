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
import os
from typing import Dict, Any, Optional, List

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
DEFAULT_START_PROMPT = """You are %n, serving as %a (%r) in project %p.
Current Task: %t
Phase: Discussion (Planning & Alignment)

Guidelines:
1. Act with the expertise of a %a. Express ideas concisely and naturally as a senior peer.
2. DO NOT write code files or perform file writes in this phase.
3. Call out unverified assumptions directly: "@Model, verify that requirement first."
4. Maintain safety and alignment with project goals. Keep responses short and conversational.
5. Do not prefix your message with your name or role header."""

DEFAULT_EXECUTION_PROMPT = """You are %n, serving as %a (%r) in project %p.
Current Task: %t
Phase: Execution (Implementation & Testing)

Guidelines:
1. Execute %a responsibilities strictly. Work incrementally in small code edits from a stateless perspective.
2. Leverage indexed memories, prior self-journals, and file manifests to complete your task.
3. Use available tools to write files, inspect diffs, and run tests.
4. Validate code before proposing merges. Check syntax and diffs carefully.
5. Maintain safety and code quality. Keep chat responses focused and concise.
6. Do not prefix your message with your name or role header."""

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
        self.load_templates()

    def load_templates(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.templates.update(data)
            except Exception as e:
                print(f"Error loading prompt templates: {e}")

    def save_templates(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.templates, f, indent=2)
        except Exception as e:
            print(f"Error saving prompt templates: {e}")

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

    if is_moderator:
        base_prompt += "\n\nModerator Directive: Coordinate turns, keep discussion focused, and assign unassigned itinerary tasks to idle bots."

    return base_prompt
