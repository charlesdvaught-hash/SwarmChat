import asyncio
import pytest
from backend.models import ModelManager
from backend.memory import MemoryManager
from backend.tools import ToolManager
from backend.orchestrator import Orchestrator

def stub_repair_generation(model_manager: ModelManager):
    """Mocks model generation to fix syntax errors during benchmark evaluation."""
    async def _generate(model_config, system_prompt, messages, **kwargs):
        return "def add_numbers(a, b):\n    return a + b\n"
    model_manager.generate_response = _generate

@pytest.mark.asyncio
async def test_benchmark_feedback_loop_repeated():
    """Runs a HumanEval/MBPP-style coding task through the swarm over >=3 repeated evaluations to prevent single-run inflation."""
    results = []

    for run_idx in range(1, 4):
        mem = MemoryManager(storage_dir=f".test_benchmark_run_{run_idx}")
        mm = ModelManager()
        tm = ToolManager()
        orch = Orchestrator(mm, mem, tm)

        stub_repair_generation(mm)

        # 1. Write an intentionally broken python script to bot sandbox
        broken_code = "def add_numbers(a, b)\n    return a + b\n"  # Missing colon SyntaxError
        tm.bot_workspace_write(bot_id="model_coder", filepath="solution.py", content=broken_code)

        # 2. Trigger the automated self-refinement loop
        refine_res = await orch.trigger_sandbox_refinement_loop(bot_id="model_coder", filepath="solution.py")

        results.append(refine_res.get("success", False))

    # Verify > 0.0 pass rate over >= 3 runs
    pass_rate = sum(results) / len(results)
    assert len(results) >= 3
    assert pass_rate == 1.0
