"""
Prompt Template Engine for SwarmChat.
Supports placeholders:
  %r - Role name (e.g. Architect, Coder)
  %n - Model name (e.g. Otis, Bonsai)
  %p - Project path and goals summary
  %t - Current active task/agenda item
"""

import json
import os
from typing import Dict, Any, Optional

SETTINGS_FILE = os.path.join(".swarmchat", "prompt_templates.json")

# Default templates designed to be under 300 tokens, strict and clear for unaligned/Dolphin models.
DEFAULT_START_PROMPT = """You are %n, serving as %r in project %p.
Current Task: %t
Phase: Discussion (Planning & Alignment)

Guidelines:
1. Act purely as %r. Express ideas concise and natural, like a peer coworker.
2. DO NOT write code files or perform file writes in this phase.
3. Call out unverified assumptions directly: "@Model, verify that requirement first."
4. Maintain safety and alignment with project goals. Keep responses short and conversational.
5. Do not prefix your message with your name or role header."""

DEFAULT_EXECUTION_PROMPT = """You are %n, serving as %r in project %p.
Current Task: %t
Phase: Execution (Implementation & Testing)

Guidelines:
1. Execute %r responsibilities strictly. Work incrementally in small code edits.
2. Use available tools to write files, inspect diffs, and run tests.
3. Validate code before proposing merges. Check syntax and diffs carefully.
4. Maintain safety and code quality. Keep chat responses focused and concise.
5. Do not prefix your message with your name or role header."""

ROLE_DEFINITIONS = {
    "Architect": {
        "description": "Design high-level architecture, component boundaries, and overall project blueprints.",
        "icon": "🏗️"
    },
    "Critic": {
        "description": "Red-team solutions, identify edge cases, vulnerabilities, performance risks, and missing requirements.",
        "icon": "🧐"
    },
    "Solver": {
        "description": "Analyze core algorithms, math logic, and step-by-step problem breakdown.",
        "icon": "💡"
    },
    "Coder": {
        "description": "Write clean, modular, production-ready code files and patches.",
        "icon": "💻"
    },
    "Tester/Debugger": {
        "description": "Write test suites, reproduce bugs, verify edge cases, and validate functionality.",
        "icon": "🧪"
    }
}

class PromptTemplateManager:
    def __init__(self, storage_path: str = SETTINGS_FILE):
        self.storage_path = storage_path
        self.templates = {
            "start_prompt": DEFAULT_START_PROMPT,
            "execution_prompt": DEFAULT_EXECUTION_PROMPT,
            "custom_role_prompts": {}
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

    def update_templates(self, start_prompt: Optional[str] = None, execution_prompt: Optional[str] = None, custom_role_prompts: Optional[Dict[str, Any]] = None):
        if start_prompt is not None:
            self.templates["start_prompt"] = start_prompt
        if execution_prompt is not None:
            self.templates["execution_prompt"] = execution_prompt
        if custom_role_prompts is not None:
            self.templates["custom_role_prompts"] = custom_role_prompts
        self.save_templates()

    def format_prompt(
        self,
        phase: str,
        role: str,
        name: str,
        project_info: str = "Default Workspace",
        current_task: str = "General Discussion / Alignment",
        custom_template: Optional[str] = None
    ) -> str:
        if custom_template and custom_template.strip():
            template = custom_template
        elif phase.lower() == "execution":
            template = self.templates.get("execution_prompt", DEFAULT_EXECUTION_PROMPT)
        else:
            template = self.templates.get("start_prompt", DEFAULT_START_PROMPT)

        formatted = (
            template
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
    custom_template: Optional[str] = None
) -> str:
    base_prompt = prompt_template_mgr.format_prompt(
        phase=phase,
        role=role,
        name=name,
        project_info=project_info,
        current_task=current_task,
        custom_template=custom_template
    )

    if is_moderator:
        base_prompt += "\n\nModerator Directive: Coordinate turns, keep discussion focused, and assign unassigned itinerary tasks to idle bots."

    return base_prompt
