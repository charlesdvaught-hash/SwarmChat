"""Test isolation for SwarmChat.

Two things used to leak into tests, and both produced failures that had nothing to do
with the code under test:

1. The Orchestrator persisted its roster and per-project chat log under a hardcoded
   "./.swarmchat", so a test that passed MemoryManager(storage_dir=<tmp>) still loaded the
   *user's real* room - tests naming `model_architect` / `model_coder` failed because the
   saved roster had the user's own model ids. Fixed in Orchestrator: it now follows the
   MemoryManager's storage dir.

2. The `.test_swarmchat*` storage dirs are relative to the CWD and survive between runs,
   so state accumulated across invocations - a suite that passed once would fail the second
   time (chat messages from the previous run were restored and counted).

This fixture removes those scratch dirs once per session, so every run starts clean.
"""
import os
import shutil

import pytest

# Scratch storage dirs the tests create in the repo root (relative to the CWD pytest runs in).
_SCRATCH_PREFIXES = (".test_swarmchat", ".test_benchmark_run_", ".test_dbg")


@pytest.fixture(scope="session", autouse=True)
def _clean_scratch_storage():
    """Delete leftover test storage dirs before (and after) the session."""
    def _sweep():
        for name in os.listdir("."):
            if name.startswith(_SCRATCH_PREFIXES):
                shutil.rmtree(name, ignore_errors=True)

    _sweep()
    yield
    _sweep()


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Run every test in its own empty directory.

    The tests build storage and workspace paths relative to the CWD (".test_swarmchat",
    ToolManager(workspace_root=".")). Sharing one CWD meant one test's chat log and bot
    workspaces were restored by the next - e.g. a "DIRECTIVE IGNORED" message written by
    the malformed-directive test showed up in the valid-directive test's history and
    failed it. Per-test CWD makes each test start from nothing, and keeps test artifacts
    out of the user's real ./.swarmchat.
    """
    monkeypatch.chdir(tmp_path)
