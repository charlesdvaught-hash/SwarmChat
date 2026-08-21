"""Verifies the question-driven discussion phase. No LLM is invoked.

The old gate looped. A REJECT dumped the room back to awaiting_plan for a full rewrite, and
the objection survived only as one 300-character chat message that fell out of the
three-message discussion window within two turns - so revision 2 did not reliably address
revision 1's complaint. Observed live: "Critic rejected the plan" -> "build plan revision 2"
-> rejected again, with nothing accumulating.

What is checked here is the fix: discussion now produces STATE, not opinions.
  * the Architect's questions are recorded, capped in code at five,
  * exactly ONE question is pinned per turn,
  * a REJECT becomes a new question instead of a rewrite,
  * an Admin question is PARKED - the room never blocks on it,
  * an Admin question written in jargon is not asked at all,
  * a vote needs two DISTINCT MODELS, not two seats,
  * every answer records its provenance (admin / vote / default).
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath("."))

from backend.directives import find_payloads, parse_fields, split_options, split_title
from backend.models import ModelManager
from backend.memory import MemoryManager
from backend.tools import ToolManager
from backend.orchestrator import Orchestrator

failures = []
_case = [0]

PLAN = (
    "Goal: ship a word counter. Create counter.py with count_words(text) returning an int, "
    "then test_counter.py covering the empty-string case. Coder writes both; Tester runs them."
)


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail and not ok else ""))
    if not ok:
        failures.append(name)


def build(tmp, seats=("arch", "crit", "code"), same_weights=False):
    """A room at the start of planning.

    `same_weights` collapses every seat onto ONE gguf path, which is the roster that makes a
    vote meaningless - the case the voting rule exists for."""
    _case[0] += 1
    mm, tm = ModelManager(), ToolManager(workspace_root=tmp)
    mem = MemoryManager(storage_dir=os.path.join(tmp, "mem_case%d" % _case[0]))
    orch = Orchestrator(mm, mem, tm, storage_root=os.path.join(tmp, "root_case%d" % _case[0]))
    catalog = {
        "arch": {"id": "arch", "name": "Otis", "role": "Architect"},
        "crit": {"id": "crit", "name": "Vera", "role": "Critic"},
        "code": {"id": "code", "name": "Bill", "role": "Coder"},
    }
    orch.models = {
        k: dict(
            v,
            provider="gguf_local",
            gguf_path=("shared.gguf" if same_weights else k + ".gguf"),
            enabled=True,
        )
        for k, v in catalog.items() if k in seats
    }
    orch.known_models = dict(orch.models)
    orch.chat_history = []
    orch.save_chat_history = lambda *a, **k: None
    mem.set_phase("discussion")
    return mem, orch


def turn(orch, model_id, text):
    """The gate half of step_model_turn, without invoking a model."""
    cfg = orch.models[model_id]
    orch.add_chat_message(cfg["name"], "Assistant", text, model_id=model_id)
    orch._advance_plan_gate(model_id, cfg, text)
    return orch.memory_manager.get_plan_stage()


def main():
    tmp = tempfile.mkdtemp(prefix="swarmchat_disc_")

    # --- 1. The parser this feature is built on (the stated dependency) ---
    fields = parse_fields(
        "title=Implement char_count(text, strip), description=Add it to counter.py, "
        "handling empty strings, priority=high",
        ("id", "title", "description", "priority", "status", "assigned_model"),
    )
    check("A comma inside a signature does not split the title",
          fields.get("title") == "Implement char_count(text, strip)", "got %r" % fields.get("title"))
    check("A comma inside real English does not truncate the description",
          fields.get("description") == "Add it to counter.py, handling empty strings",
          "got %r" % fields.get("description"))
    check("The trailing field still parses", fields.get("priority") == "high")

    long_title = (
        "Build the word counter. It reads a file, counts the words in it, and prints the "
        "total to the screen so the user can see it."
    )
    t, d = split_title(long_title, "")
    check("A runaway title is trimmed to its first sentence", t == "Build the word counter",
          "got %r" % t)
    check("The rest of a runaway title is kept as the description", "reads a file" in d,
          "got %r" % d)

    check("A payload containing brackets is not cut short",
          find_payloads("[ANSWER: use the list [a, b] form]", "[ANSWER:") == ["use the list [a, b] form"],
          "got %r" % find_payloads("[ANSWER: use the list [a, b] form]", "[ANSWER:"))
    check("An unterminated directive reports as unterminated",
          find_payloads("[ANSWER: no closing bracket", "[ANSWER:") == [None])

    opts = split_options("Type a command: you see the answer right away | A background piece: other programs call it")
    check("Options split into label and what it means in practice",
          len(opts) == 2 and opts[0]["label"] == "Type a command"
          and opts[1]["means"] == "other programs call it", "got %r" % opts)

    # --- 2. Questions are collected, and the cap is enforced in CODE ---
    mem, orch = build(tmp)
    check("A fresh room opens on the question stage", mem.get_plan_stage() == "awaiting_questions",
          "got %r" % mem.get_plan_stage())
    stage = turn(orch, "arch", "\n".join(
        "[QUESTION: ask=Question number %d, which behaviour do we want here?, for=team]" % i
        for i in range(1, 9)
    ))
    check("The question cap is enforced in code, not asked for in the prompt",
          len(mem.get_plan_questions()) == Orchestrator.MAX_PLAN_QUESTIONS,
          "got %d" % len(mem.get_plan_questions()))
    check("Questions move the room to the resolving stage", stage == "resolving_questions",
          "got %r" % stage)

    # --- 3. Exactly ONE question is settled per turn ---
    pinned = mem.next_open_question()
    turn(orch, "arch", "It should ignore blank lines. [ANSWER: Blank lines are skipped.]")
    check("One turn settles exactly one question",
          len(mem.questions_by_status("resolved")) == 1,
          "got %d" % len(mem.questions_by_status("resolved")))
    check("The settled one is the question that was pinned",
          mem.get_plan_question(pinned["id"])["status"] == "resolved")
    check("The recorded answer is the model's, not a placeholder",
          mem.get_plan_question(pinned["id"])["answer"] == "Blank lines are skipped.",
          "got %r" % mem.get_plan_question(pinned["id"])["answer"])
    check("A model-settled answer is recorded as a default, not as a real decision",
          mem.get_plan_question(pinned["id"])["decided_by"] == "default")
    check("The room stays on the question stage while questions remain",
          mem.get_plan_stage() == "resolving_questions", "got %r" % mem.get_plan_stage())

    # --- 4. Zero open questions is the exit condition, and it is countable ---
    for _ in range(6):
        if mem.get_plan_stage() != "resolving_questions":
            break
        turn(orch, "arch", "[ANSWER: Decided.]")
    check("An empty question board opens the plan stage",
          mem.get_plan_stage() == "awaiting_plan", "got %r" % mem.get_plan_stage())
    check("Every question is settled once the stage moves on",
          not mem.questions_by_status("open"), "got %r" % mem.questions_by_status("open"))

    # --- 5. A turn with no questions is allowed to skip straight to the plan ---
    mem, orch = build(tmp)
    stage = turn(orch, "arch", "Nothing here is genuinely undecided - the job is a plain word count "
                               "of a text file, so I will go straight to the build plan.")
    check("No questions means straight to the plan", stage == "awaiting_plan", "got %r" % stage)

    # A garbled turn must NOT be read as "no questions".
    mem, orch = build(tmp)
    stage = turn(orch, "arch", ", status:\"in_progress\"}]")
    check("A garbled turn holds the question stage", stage == "awaiting_questions", "got %r" % stage)

    # --- 6. Plain prose questions are not thrown away over syntax ---
    mem, orch = build(tmp)
    turn(orch, "arch", "A couple of things are unclear:\n"
                       "1. Should punctuation count as part of a word?\n"
                       "2. What should an empty file report?\n")
    check("Numbered prose questions are harvested when brackets are missing",
          len(mem.get_plan_questions()) == 2, "got %r" % mem.get_plan_questions())

    # --- 7. A REJECT becomes a question, not a rewrite ---
    mem, orch = build(tmp)
    mem.set_plan_stage("awaiting_plan")
    turn(orch, "arch", PLAN)
    stage = turn(orch, "crit", "count_words is never told what to do with hyphenated words.\nREJECT")
    check("A REJECT does not send the plan back for a blind rewrite", stage != "awaiting_plan",
          "got %r" % stage)
    check("A REJECT puts the objection on the question board",
          len(mem.get_plan_questions()) == 1, "got %r" % mem.get_plan_questions())
    check("The objection is recorded in full, not as a 300-char chat message",
          "hyphenated" in mem.get_plan_questions()[0]["text_internal"])
    turn(orch, "arch", "[ANSWER: Hyphenated words count as one word.]")
    check("Settling the objection returns the room to the plan",
          mem.get_plan_stage() == "awaiting_plan", "got %r" % mem.get_plan_stage())
    check("The decision is carried into the next plan prompt",
          "Hyphenated words count as one word." in orch._build_questions_context(),
          "got %r" % orch._build_questions_context())

    # --- 8. An Admin question PARKS. It never blocks the room. ---
    mem, orch = build(tmp)
    turn(orch, "arch",
         "[QUESTION: ask=How do you want to see the answer?, "
         "options=On the screen: you run it and the count appears | "
         "In a file: it writes the count into a file you open later, "
         "recommended=On the screen, why=You can try it immediately, for=admin]\n"
         "[QUESTION: ask=Should punctuation count as part of a word?, for=team]")
    admin_qs = [q for q in mem.get_plan_questions() if q["resolvable_by"] == "admin"]
    check("An admin-routed question is parked, not left open",
          len(admin_qs) == 1 and admin_qs[0]["status"] == "parked", "got %r" % admin_qs)
    check("Parking an admin question does not stall the room - the team question is pinned",
          mem.next_open_question() is not None and mem.next_open_question()["resolvable_by"] == "model")
    check("The parked question is visible to the Admin in chat",
          any("YOUR CALL" in m["content"] for m in orch.chat_history))

    # Answering it later resumes the gate, and provenance says it was really the Admin.
    res = orch.answer_plan_question(admin_qs[0]["id"], "On the screen")
    check("The Admin's answer is accepted", res.get("success") is True, "got %r" % res)
    answered = mem.get_plan_question(admin_qs[0]["id"])
    check("An Admin answer is recorded as decided by the Admin",
          answered["decided_by"] == "admin" and answered["status"] == "resolved",
          "got %r" % answered)
    check("Answering an unknown question is refused, not silently ignored",
          orch.answer_plan_question("q_nope", "x").get("success") is False)

    # --- 9. Jargon never reaches the Admin ---
    mem, orch = build(tmp)
    turn(orch, "arch", "[QUESTION: ask=Do you want a CLI or a library?, options=CLI | Library, "
                       "recommended=CLI, why=Faster to try, for=admin]")
    q = mem.get_plan_questions()[0]
    check("A jargon question is not put to the Admin at all",
          q["resolvable_by"] == "model", "got %r" % q)
    check("A jargon question is still settled by the room rather than dropped",
          q["status"] in ("open", "resolved"), "got %r" % q["status"])

    # The same question, written by what the person actually experiences, IS asked.
    mem, orch = build(tmp)
    turn(orch, "arch",
         "[QUESTION: ask=How do you want to use this?, "
         "options=Type a command: you type it in a terminal and the answer appears | "
         "Library: a piece other programs can call, with no screen of its own, "
         "recommended=Type a command, why=You can try it immediately, for=admin]")
    q = mem.get_plan_questions()[0]
    check("A jargon label WITH a plain-English meaning is allowed through",
          q["resolvable_by"] == "admin" and q["status"] == "parked", "got %r" % q)

    # An admin question with only one option is a defect, not a choice.
    mem, orch = build(tmp)
    turn(orch, "arch", "[QUESTION: ask=Where should the total be shown to you?, "
                       "options=On the screen: it prints when you run it, for=admin]")
    check("An admin question with no real choice is handed back to the room",
          mem.get_plan_questions()[0]["resolvable_by"] == "model")

    # --- 10. The voting trap: a vote needs two DISTINCT MODELS, not two seats ---
    mem, orch = build(tmp, same_weights=True)
    check("Three seats on one set of weights count as ONE model",
          len(orch._distinct_model_paths()) == 1, "got %r" % orch._distinct_model_paths())
    orch.loop_active = True  # Auto Mode: the room must not wait for an absent Admin.
    turn(orch, "arch",
         "[QUESTION: ask=How do you want to see the answer?, "
         "options=On the screen: it prints when you run it | "
         "In a file: it writes it out for you to open, "
         "recommended=On the screen, why=You can try it immediately, for=admin]")
    q = mem.get_plan_questions()[0]
    check("With one model behind every seat there is no vote, only a recorded default",
          q["status"] == "resolved" and q["decided_by"] == "default", "got %r" % q)
    check("The default taken is the Architect's own recommendation",
          q["answer"] == "On the screen", "got %r" % q["answer"])

    # With genuinely distinct models, ballots decide it and provenance says so.
    mem, orch = build(tmp)
    check("Three seats on three sets of weights count as three models",
          len(orch._distinct_model_paths()) == 3, "got %r" % orch._distinct_model_paths())
    turn(orch, "arch",
         "[QUESTION: ask=How do you want to see the answer?, "
         "options=On the screen: it prints when you run it | "
         "In a file: it writes it out for you to open, "
         "recommended=On the screen, why=You can try it immediately, for=admin]")
    q = mem.get_plan_questions()[0]
    check("With the Admin present the question simply waits", q["status"] == "parked",
          "got %r" % q["status"])
    orch.loop_active = True
    turn(orch, "crit", "[VOTE: q=1, choice=In a file]")
    turn(orch, "code", "[VOTE: q=1, choice=In a file]")
    q = mem.get_plan_question(q["id"])
    check("Ballots from other models are counted", len(q.get("votes") or {}) == 2,
          "got %r" % q.get("votes"))
    orch._settle_questions("Every open question is settled.")
    q = mem.get_plan_question(q["id"])
    check("A real vote decides the question and is recorded as a vote",
          q["status"] == "resolved" and q["decided_by"] == "vote" and q["answer"] == "In a file",
          "got %r" % q)

    # And the Admin can still override a closed ballot.
    orch.answer_plan_question(q["id"], "On the screen")
    q = mem.get_plan_question(q["id"])
    check("The Admin overrides a vote the room took while they were away",
          q["decided_by"] == "admin" and q["answer"] == "On the screen", "got %r" % q)

    # --- 11. Provenance reaches the plan prompt as an explicit assumption ---
    mem, orch = build(tmp)
    turn(orch, "arch",
         "[QUESTION: ask=How do you want to see the answer?, "
         "options=On the screen: it prints when you run it | "
         "In a file: it writes it out for you to open, "
         "recommended=On the screen, why=You can try it immediately, for=admin]")
    ctx = orch._build_questions_context()
    check("A question still with the Admin is stated in the plan prompt as an assumption",
          "assumed: On the screen" in ctx, "got %r" % ctx)

    # --- 12. Exactly one question is exposed to the model during resolution ---
    mem, orch = build(tmp)
    turn(orch, "arch", "[QUESTION: ask=Should punctuation count as part of a word?, for=team]\n"
                       "[QUESTION: ask=Should numbers count as words?, for=team]")
    ctx = orch._build_questions_context()
    check("The resolving prompt shows one question and only one",
          ctx.count("Q1:") + ctx.count("Q2:") == 1, "got %r" % ctx)
    check("The resolving prompt tells the model not to re-open the plan",
          "nothing else" in ctx.lower())

    print()
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("All discussion-phase checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
