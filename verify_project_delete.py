"""Verifies you can actually delete a project.

The bug: the UI's project switcher makes *selecting* a project the same act as *switching*
to it, so the only project the Delete button could ever target was the active one - and the
backend refused exactly that with "switch to another project first". The button was
impossible to satisfy from the UI.

Deleting the active project now switches the room to another project first. Only the last
project standing is still refused. No LLM is invoked; this drives the real HTTP endpoints.
"""
import os
import sys
import shutil
import tempfile

# main.py builds its MemoryManager/ToolManager at import time against "./.swarmchat", so the
# CWD has to be a throwaway before the import - otherwise this script archives real projects.
_REPO = os.path.abspath(os.path.dirname(__file__))
_TMP = tempfile.mkdtemp(prefix="swarmchat_projdel_")
os.chdir(_TMP)
sys.path.insert(0, _REPO)

from fastapi.testclient import TestClient  # noqa: E402
from backend.main import app  # noqa: E402

client = TestClient(app)
failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail and not ok else ""))
    if not ok:
        failures.append(name)


def projects():
    return [p["project_id"] for p in client.get("/api/projects").json()["projects"]]


def active():
    return client.get("/api/projects").json()["active_project_id"]


def main():
    client.post("/api/projects/create", json={"project_id": "alpha"})
    client.post("/api/projects/create", json={"project_id": "beta"})
    client.post("/api/projects/switch", json={"project_id": "alpha"})
    check("Setup: alpha is active", active() == "alpha", "active is %r" % active())

    # 1. The bug itself: deleting the project you have selected.
    r = client.post("/api/projects/delete", json={"project_id": "alpha"})
    check("Deleting the ACTIVE project succeeds", r.status_code == 200,
          "HTTP %s %s" % (r.status_code, r.text[:160]))
    if r.status_code == 200:
        body = r.json()
        check("Delete reports where the room switched to", bool(body.get("switched_to")),
              "switched_to=%r" % body.get("switched_to"))
        check("The room is no longer in the deleted project", active() != "alpha",
              "active is %r" % active())
        check("The deleted project is gone from the list", "alpha" not in projects(),
              "projects=%r" % projects())
        check("The project directory was archived, not left behind",
              not os.path.isdir(os.path.join(".swarmchat", "projects", "alpha")))
        check("Something was moved to trash", bool(body.get("trashed_to")),
              "trashed_to=%r" % body.get("trashed_to"))

    # 2. A non-active project still deletes without touching the active one.
    client.post("/api/projects/create", json={"project_id": "gamma"})
    before = active()
    r = client.post("/api/projects/delete", json={"project_id": "gamma"})
    check("Deleting a non-active project still works", r.status_code == 200,
          "HTTP %s %s" % (r.status_code, r.text[:160]))
    check("Deleting a non-active project does not switch the room", active() == before,
          "active moved %r -> %r" % (before, active()))
    check("Non-active delete reports no switch", r.json().get("switched_to") is None,
          "switched_to=%r" % r.json().get("switched_to"))

    # 3. The last project standing is refused - a room with no project has no memory
    #    archive, no itinerary and no workspace root.
    for pid in projects():
        if pid != active():
            client.post("/api/projects/delete", json={"project_id": pid})
    check("Setup: exactly one project left", len(projects()) == 1, "projects=%r" % projects())
    r = client.post("/api/projects/delete", json={"project_id": active()})
    check("Deleting the last project is refused", r.status_code == 400,
          "HTTP %s %s" % (r.status_code, r.text[:160]))
    check("The last project survives the refusal", len(projects()) == 1,
          "projects=%r" % projects())

    # 4. A project that does not exist is a 404, not a 500.
    r = client.post("/api/projects/delete", json={"project_id": "no_such_project"})
    check("Unknown project is a 404", r.status_code == 404,
          "HTTP %s %s" % (r.status_code, r.text[:160]))

    print()
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("All project-delete checks passed.")
    return 0


if __name__ == "__main__":
    code = main()
    os.chdir(_REPO)
    shutil.rmtree(_TMP, ignore_errors=True)
    sys.exit(code)
