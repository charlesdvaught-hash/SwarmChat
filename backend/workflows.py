import logging
import os
import time
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger(__name__)

# --- WORKFLOW TEMPLATES ---

WORKFLOW_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "CHAT_BRAINSTORM": {
        "id": "CHAT_BRAINSTORM",
        "name": "Chat Brainstorm",
        "description": "Collaborative ideation and discussion without file modification.",
        "allowed_phases": ["discussion"],
        "initial_moderator_responsibility": "Facilitate open discussion and summarize key concepts.",
        "research_allowed": True,
        "file_modification_allowed": False,
        "testing_mandatory": False,
        "repair_policy": {"max_repair_attempts": 0, "alternate_coder_attempts": 0, "solver_escalation": False},
        "escalation_policy": "None",
        "completion_criteria": "Consensus reached or user ends session.",
        "steps": ["1. Open discussion", "2. Explore ideas", "3. Synthesize summary"]
    },
    "APPLICATION_DESIGN": {
        "id": "APPLICATION_DESIGN",
        "name": "Application Design",
        "description": "Architecture planning, system requirements, and task graph generation.",
        "allowed_phases": ["discussion", "execution"],
        "initial_moderator_responsibility": "Analyze system requirements, evaluate constraints, and produce structured task graph.",
        "research_allowed": True,
        "file_modification_allowed": True,
        "testing_mandatory": False,
        "repair_policy": {"max_repair_attempts": 1, "alternate_coder_attempts": 0, "solver_escalation": False},
        "escalation_policy": "Escalate architectural ambiguity to Human.",
        "completion_criteria": "Structured task graph and architecture specs written to workspace.",
        "steps": ["1. Requirements analysis", "2. Tech stack evaluation", "3. Task graph generation"]
    },
    "APPLICATION_BUILD": {
        "id": "APPLICATION_BUILD",
        "name": "Application Build",
        "description": "End-to-end multi-agent feature or application construction.",
        "allowed_phases": ["discussion", "execution"],
        "initial_moderator_responsibility": "Formulate execution task graph, assign coding roles, and supervise execution.",
        "research_allowed": True,
        "file_modification_allowed": True,
        "testing_mandatory": True,
        "repair_policy": {"max_repair_attempts": 3, "alternate_coder_attempts": 1, "solver_escalation": True},
        "escalation_policy": "3 Repairs -> Alternate Coder -> Solver Escalation -> Human Review.",
        "completion_criteria": "All tasks executed, tests passing, and changes merged into main workspace.",
        "steps": [
            "1. Understand requirements",
            "2. Inspect existing project",
            "3. Research unknowns",
            "4. Establish architecture",
            "5. Create structured task graph",
            "6. Execute tasks in dependency order",
            "7. Test each implementation",
            "8. Repair failures",
            "9. Try alternate implementation if necessary",
            "10. Escalate persistent failures",
            "11. Integrate successful work",
            "12. Run project-wide tests",
            "13. Final review"
        ]
    },
    "APPLICATION_DEBUG_REPAIR": {
        "id": "APPLICATION_DEBUG_REPAIR",
        "name": "Application Debug & Repair",
        "description": "Targeted bug fixing, test diagnostic, and code repair.",
        "allowed_phases": ["discussion", "execution"],
        "initial_moderator_responsibility": "Identify reproducing test cases, create diagnostic tasks, and assign to Coder/Refiner.",
        "research_allowed": True,
        "file_modification_allowed": True,
        "testing_mandatory": True,
        "repair_policy": {"max_repair_attempts": 3, "alternate_coder_attempts": 1, "solver_escalation": True},
        "escalation_policy": "3 Repairs -> Alternate Coder -> Solver Escalation -> Human Review.",
        "completion_criteria": "Failing test cases pass without regressions.",
        "steps": [
            "1. Reproduce failure with test case",
            "2. Isolate root cause",
            "3. Create repair task",
            "4. Apply targeted fix",
            "5. Run test suite",
            "6. Repair / escalate if needed",
            "7. Merge fix"
        ]
    },
    "REFACTOR": {
        "id": "REFACTOR",
        "name": "Refactor Codebase",
        "description": "Code quality improvements, performance optimization, and modularization without changing behavior.",
        "allowed_phases": ["discussion", "execution"],
        "initial_moderator_responsibility": "Analyze codebase smell/bottlenecks and produce refactoring task graph.",
        "research_allowed": True,
        "file_modification_allowed": True,
        "testing_mandatory": True,
        "repair_policy": {"max_repair_attempts": 3, "alternate_coder_attempts": 1, "solver_escalation": True},
        "escalation_policy": "Revert to pre-refactor state if tests fail after Solver escalation.",
        "completion_criteria": "Refactored code passes all unit and regression tests.",
        "steps": ["1. Audit codebase", "2. Map refactoring tasks", "3. Execute refactoring", "4. Validate tests", "5. Merge"]
    },
    "RESEARCH": {
        "id": "RESEARCH",
        "name": "External Research",
        "description": "Deep web research, documentation retrieval, and technical benchmarking.",
        "allowed_phases": ["discussion"],
        "initial_moderator_responsibility": "Break research topic into investigation queries and assign Researcher models.",
        "research_allowed": True,
        "file_modification_allowed": False,
        "testing_mandatory": False,
        "repair_policy": {"max_repair_attempts": 0, "alternate_coder_attempts": 0, "solver_escalation": False},
        "escalation_policy": "None",
        "completion_criteria": "Structured research report saved to personal spec / shared memory.",
        "steps": ["1. Formulate search queries", "2. Execute internet & web fetch", "3. Synthesize findings"]
    },
    "CODE_REVIEW": {
        "id": "CODE_REVIEW",
        "name": "Code Review",
        "description": "Static code review, security audit, and test coverage assessment.",
        "allowed_phases": ["discussion"],
        "initial_moderator_responsibility": "Assign Reviewer/Critic models to inspect git diffs and codebase files.",
        "research_allowed": True,
        "file_modification_allowed": False,
        "testing_mandatory": True,
        "repair_policy": {"max_repair_attempts": 0, "alternate_coder_attempts": 0, "solver_escalation": False},
        "escalation_policy": "None",
        "completion_criteria": "Review report generated with actionable feedback.",
        "steps": ["1. Review git diff", "2. Run test coverage", "3. Produce review notes"]
    }
}

VALID_TASK_STATUSES = {
    "pending",
    "researching",
    "planned",
    "implementing",
    "testing",
    "repairing",
    "alternate_attempt",
    "solver_review",
    "passed",
    "failed",
    "blocked",
    "needs_human",
    "merged"
}


# --- TASK & TASK GRAPH MODEL ---

class Task:
    def __init__(
        self,
        task_id: str,
        title: str,
        description: str,
        requirements: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        priority: str = "medium",
        assigned_model: Optional[str] = None,
        assigned_role: Optional[str] = None,
        allowed_files: Optional[List[str]] = None,
        test_command: Optional[str] = None,
        status: str = "pending",
        attempt_count: int = 0,
        repair_count: int = 0,
        workspace: Optional[str] = None,
        failure_history: Optional[List[Dict[str, Any]]] = None,
        candidate_results: Optional[List[Dict[str, Any]]] = None,
        diffs: Optional[List[Dict[str, Any]]] = None,
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None
    ):
        self.id = task_id
        self.title = title
        self.description = description
        self.requirements = requirements or []
        self.dependencies = dependencies or []
        self.priority = priority.lower() if priority else "medium"
        self.assigned_model = assigned_model
        self.assigned_role = assigned_role
        self.allowed_files = allowed_files or []
        self.test_command = test_command
        self.status = status if status in VALID_TASK_STATUSES else "pending"
        self.attempt_count = attempt_count
        self.repair_count = repair_count
        self.workspace = workspace
        self.failure_history = failure_history or []
        self.candidate_results = candidate_results or []
        self.diffs = diffs or []
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "requirements": self.requirements,
            "dependencies": self.dependencies,
            "priority": self.priority,
            "assigned_model": self.assigned_model,
            "assigned_role": self.assigned_role,
            "allowed_files": self.allowed_files,
            "test_command": self.test_command,
            "status": self.status,
            "attempt_count": self.attempt_count,
            "repair_count": self.repair_count,
            "workspace": self.workspace,
            "failure_history": self.failure_history,
            "candidate_results": self.candidate_results,
            "diffs": self.diffs,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(
            task_id=data.get("id", f"TASK-{int(time.time()*1000)}"),
            title=data.get("title", "Untitled Task"),
            description=data.get("description", ""),
            requirements=data.get("requirements", []),
            dependencies=data.get("dependencies", []),
            priority=data.get("priority", "medium"),
            assigned_model=data.get("assigned_model"),
            assigned_role=data.get("assigned_role"),
            allowed_files=data.get("allowed_files", []),
            test_command=data.get("test_command"),
            status=data.get("status", "pending"),
            attempt_count=data.get("attempt_count", 0),
            repair_count=data.get("repair_count", 0),
            workspace=data.get("workspace"),
            failure_history=data.get("failure_history", []),
            candidate_results=data.get("candidate_results", []),
            diffs=data.get("diffs", []),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at")
        )


class TaskGraph:
    def __init__(self, tasks: Optional[List[Task]] = None):
        self.tasks: Dict[str, Task] = {}
        if tasks:
            for t in tasks:
                self.add_task(t)

    def add_task(self, task: Task):
        self.tasks[task.id] = task

    def update_task(self, task_id: str, updates: Dict[str, Any]) -> Optional[Task]:
        task = self.tasks.get(task_id)
        if not task:
            return None
        for k, v in updates.items():
            if hasattr(task, k):
                if k == "status" and v not in VALID_TASK_STATUSES:
                    continue
                setattr(task, k, v)
        task.updated_at = time.time()
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def get_executable_tasks(self) -> List[Task]:
        """Returns pending tasks whose dependencies have all passed or merged."""
        ready = []
        for task in self.tasks.values():
            if task.status not in ["pending", "planned"]:
                continue
            deps_satisfied = True
            for dep_id in task.dependencies:
                dep_task = self.tasks.get(dep_id)
                if not dep_task or dep_task.status not in ["passed", "merged"]:
                    deps_satisfied = False
                    break
            if deps_satisfied:
                ready.append(task)
        return ready

    def is_complete(self) -> bool:
        """Returns True if all tasks in graph are passed, merged, or terminal failed/needs_human."""
        if not self.tasks:
            return True
        for task in self.tasks.values():
            if task.status not in ["passed", "merged", "failed", "needs_human"]:
                return False
        return True

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.tasks.values()]

    @classmethod
    def from_dict_list(cls, data_list: List[Dict[str, Any]]) -> "TaskGraph":
        tasks = [Task.from_dict(d) for d in data_list]
        return cls(tasks)


# --- MODEL CAPABILITY METADATA & GGUF DISCOVERY ---

class ModelCapability:
    def __init__(
        self,
        parameter_count: Optional[str] = None,
        native_context: int = 4096,
        operational_context: int = 4096,
        hardware_context_limit: int = 4096,
        quantization: Optional[str] = None,
        architecture: Optional[str] = None,
        supports_tool_calling: bool = True,
        supports_reasoning: bool = True,
        recommended_temperature: float = 0.2,
        recommended_top_p: float = 0.9,
        recommended_top_k: int = 40,
        recommended_repeat_penalty: float = 1.1,
        coding_strength: float = 0.5,
        reasoning_strength: float = 0.5,
        tool_use_strength: float = 0.5
    ):
        self.parameter_count = parameter_count
        self.native_context = native_context
        self.operational_context = operational_context
        self.hardware_context_limit = hardware_context_limit
        self.quantization = quantization
        self.architecture = architecture
        self.supports_tool_calling = supports_tool_calling
        self.supports_reasoning = supports_reasoning
        self.recommended_temperature = recommended_temperature
        self.recommended_top_p = recommended_top_p
        self.recommended_top_k = recommended_top_k
        self.recommended_repeat_penalty = recommended_repeat_penalty
        self.coding_strength = coding_strength
        self.reasoning_strength = reasoning_strength
        self.tool_use_strength = tool_use_strength

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_count": self.parameter_count,
            "native_context": self.native_context,
            "operational_context": self.operational_context,
            "hardware_context_limit": self.hardware_context_limit,
            "quantization": self.quantization,
            "architecture": self.architecture,
            "supports_tool_calling": self.supports_tool_calling,
            "supports_reasoning": self.supports_reasoning,
            "recommended_temperature": self.recommended_temperature,
            "recommended_top_p": self.recommended_top_p,
            "recommended_top_k": self.recommended_top_k,
            "recommended_repeat_penalty": self.recommended_repeat_penalty,
            "coding_strength": self.coding_strength,
            "reasoning_strength": self.reasoning_strength,
            "tool_use_strength": self.tool_use_strength
        }


def infer_model_capabilities(model_name: str, gguf_path: Optional[str] = None) -> ModelCapability:
    """Infers model capability metadata from filename, model tag, or GGUF metadata."""
    param_cnt = None
    quant = None
    arch = "llama"
    native_ctx = 4096
    coding_s = 0.5
    reasoning_s = 0.5
    tool_s = 0.5

    lower = (model_name + " " + (gguf_path or "")).lower()

    # Parameter count inference
    if "0.5b" in lower or "500m" in lower:
        param_cnt = "0.5B"
    elif "1.5b" in lower or "1_5b" in lower:
        param_cnt = "1.5B"
    elif "3b" in lower:
        param_cnt = "3B"
    elif "7b" in lower or "8b" in lower:
        param_cnt = "7B-8B"
    elif "14b" in lower:
        param_cnt = "14B"
    elif "32b" in lower or "33b" in lower:
        param_cnt = "32B"
    elif "70b" in lower:
        param_cnt = "70B"

    # Architecture / Specialization
    if "coder" in lower or "code" in lower:
        coding_s = 0.9
        tool_s = 0.8
        arch = "qwen2.5-coder" if "qwen" in lower else "coder"
    if "qwen" in lower:
        native_ctx = 32768 if "32k" in lower else 8192
    elif "llama3" in lower or "llama-3" in lower:
        native_ctx = 8192
    elif "mistral" in lower or "mixtral" in lower:
        native_ctx = 8192

    # Quantization inference
    for q in ["q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "q4_0", "q3_k_m", "q2_k", "f16", "bf16"]:
        if q in lower:
            quant = q.upper()
            break

    # If GGUF file exists, try reading GGUF header
    if gguf_path and os.path.exists(gguf_path):
        try:
            import struct
            with open(gguf_path, "rb") as f:
                magic = f.read(4)
                if magic == b"GGUF":
                    version = struct.unpack("<I", f.read(4))[0]
                    logger.debug("Inspected GGUF header for %s: version %d", gguf_path, version)
        except Exception as e:
            logger.debug("GGUF header inspection skipped for %s: %s", gguf_path, e)

    return ModelCapability(
        parameter_count=param_cnt,
        native_context=native_ctx,
        operational_context=min(native_ctx, 4096),
        hardware_context_limit=4096,
        quantization=quant,
        architecture=arch,
        supports_tool_calling=True,
        supports_reasoning=True,
        recommended_temperature=0.1 if coding_s > 0.7 else 0.7,
        recommended_top_p=0.9,
        recommended_top_k=40,
        recommended_repeat_penalty=1.02 if coding_s > 0.7 else 1.1,
        coding_strength=coding_s,
        reasoning_strength=reasoning_s,
        tool_use_strength=tool_s
    )
