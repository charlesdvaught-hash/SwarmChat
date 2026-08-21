"""Shared parsing for bracket directives (`[UPDATE_TASK: ...]`, `[QUESTION: ...]`, ...).

This module exists because the directive parser was, for a long time, the single
most damaging piece of code in the pipeline: it silently mangled the Architect's
output. Two observed failures, both of which this module fixes for *every*
directive rather than for one of them:

1. A naive `payload.split(",")` split inside a function signature, so
   `title=Implement char_count(text, strip)` became the task
   `Implement char_count(text`. Real English descriptions contain commas too, so
   the whole directive was frequently rejected and a good plan thrown away.
2. When the model emitted a title but no `description=` key, the title swallowed
   an entire description block and the itinerary grew a 900-character "title".

The rules here are deliberately boring and testable:
  * a key only starts a new field when it is at bracket/quote depth zero,
  * `,`, `;` and newlines all separate fields,
  * an unterminated directive is an error, never a silent truncation.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Longest title we will ever store. Beyond this the text is prose, not a name,
# and prose belongs in the description.
MAX_TITLE_CHARS = 90

_OPENERS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = {")": "(", "]": "[", "}": "{"}
_QUOTES = ("'", '"', "`")


def _depths(text: str) -> List[int]:
    """Bracket/quote nesting depth at each character position.

    Depth > 0 means "inside something", and inside something a comma is part of
    the content - never a field separator.
    """
    depths: List[int] = []
    stack: List[str] = []
    quote: Optional[str] = None
    for ch in text:
        if quote:
            depths.append(len(stack) + 1)
            if ch == quote:
                quote = None
            continue
        if ch in _QUOTES:
            quote = ch
            depths.append(len(stack) + 1)
            continue
        if ch in _OPENERS:
            depths.append(len(stack))
            stack.append(ch)
            continue
        if ch in _CLOSERS:
            if stack and stack[-1] == _CLOSERS[ch]:
                stack.pop()
            depths.append(len(stack))
            continue
        depths.append(len(stack))
    return depths


def find_payloads(text: str, opening_token: str) -> List[Optional[str]]:
    """Every payload for `opening_token` in `text`, in order.

    A `None` entry marks a directive that was opened and never closed - the caller
    reports that as a parse failure rather than guessing where it ended. Nested
    brackets inside the payload are matched, so a directive whose text contains
    `[` / `]` is not cut short at the first `]`.
    """
    payloads: List[Optional[str]] = []
    idx = 0
    token_len = len(opening_token)
    while True:
        start = text.find(opening_token, idx)
        if start == -1:
            return payloads
        body_start = start + token_len
        depth = 0
        end = -1
        for pos in range(body_start, len(text)):
            ch = text[pos]
            if ch == "[":
                depth += 1
            elif ch == "]":
                if depth == 0:
                    end = pos
                    break
                depth -= 1
        if end == -1:
            payloads.append(None)
            return payloads
        payloads.append(text[body_start:end].strip())
        idx = end + 1


def find_payload(text: str, opening_token: str) -> Optional[str]:
    """The first payload for `opening_token`, or None when it is unterminated."""
    found = find_payloads(text, opening_token)
    return found[0] if found else None


def parse_fields(payload: str, known_keys: Tuple[str, ...]) -> Dict[str, str]:
    """`key=value` pairs out of a directive payload.

    Only a recognised key at depth zero, preceded by the start of the payload or
    by a separator, opens a new field. Everything up to the next such key is the
    value - commas, colons, parentheses and all.

    Returns `{}` when no recognised key is present; the caller decides whether
    that is an error (a malformed directive) or a cue to fall back.
    """
    if not payload:
        return {}
    depths = _depths(payload)
    key_pattern = r"(?:^|[,;\n])\s*(" + "|".join(re.escape(k) for k in known_keys) + r")\s*="
    matches = [
        m for m in re.finditer(key_pattern, payload, flags=re.IGNORECASE)
        # The '=' must be at depth zero. `f(a=1, b=2)` inside a description is
        # content, not two more directive fields.
        if depths[m.end() - 1] == 0
    ]
    if not matches:
        return {}

    fields: Dict[str, str] = {}
    for i, m in enumerate(matches):
        key = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(payload)
        value = payload[start:end].strip().strip(",;").strip()
        # A repeated key is the model restating itself; keep the first, which is
        # the one the rest of the directive was written around.
        fields.setdefault(key, value)
    return fields


def parse_strict_pairs(payload: str) -> Dict[str, str]:
    """Last-resort `a=1, b=2` parse for a payload with no recognised key.

    Kept so a genuinely malformed directive still raises a clear, specific error
    instead of being quietly dropped.
    """
    out: Dict[str, str] = {}
    for part in [p.strip() for p in re.split(r"[,;\n]", payload) if p.strip()]:
        if "=" not in part:
            raise ValueError(f"segment '{part}' is not a key=value pair")
        k, v = (t.strip() for t in part.split("=", 1))
        out[k.lower()] = v
    return out


def split_title(title: str, description: str = "") -> Tuple[str, str]:
    """Keep a title a title.

    When a model writes the whole plan into `title=`, the first sentence becomes
    the title and the remainder is pushed into the description rather than being
    thrown away. A task called `Implement the word counter. It should read a file
    and, for each line...` is unusable in every list that renders it.
    """
    title = (title or "").strip()
    description = (description or "").strip()
    if len(title) <= MAX_TITLE_CHARS:
        return title, description

    first = re.split(r"(?<=[.!?])\s+", title)[0].strip()
    if len(first) > MAX_TITLE_CHARS:
        # No sentence break either - cut on the last word boundary that fits.
        cut = title[:MAX_TITLE_CHARS].rsplit(" ", 1)[0] or title[:MAX_TITLE_CHARS]
        first = cut.strip()
    remainder = title[len(first):].strip(" .\t\n")
    if remainder:
        description = f"{remainder}\n\n{description}".strip() if description else remainder
    return first.rstrip(" .") or title[:MAX_TITLE_CHARS], description


def split_options(raw: str) -> List[Dict[str, str]]:
    """`A: what it means | B: what it means` -> [{label, means}, ...].

    Accepts `|`, ` / ` or a numbered/lettered list as the separator, because a
    <=5B model will use whichever it feels like. An option with no explanation
    keeps its label as its own meaning rather than being dropped - an option the
    Admin can still read is better than one that vanished.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    if "|" in raw:
        chunks = raw.split("|")
    else:
        chunks = re.split(r"(?:^|\s)\(?[A-Da-d1-4][\).]\s+", raw)
    options: List[Dict[str, str]] = []
    for chunk in chunks:
        chunk = chunk.strip().strip("-•").strip()
        if not chunk:
            continue
        if ":" in chunk:
            label, means = chunk.split(":", 1)
            label, means = label.strip(), means.strip()
        elif " - " in chunk:
            label, means = chunk.split(" - ", 1)
            label, means = label.strip(), means.strip()
        else:
            label, means = chunk, chunk
        if not label:
            continue
        options.append({"label": label[:80], "means": (means or label)[:300]})
    return options[:4]
