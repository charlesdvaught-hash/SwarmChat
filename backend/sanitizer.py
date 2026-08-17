import re
from typing import List, Dict, Any

def sanitize_message_content(content: str) -> str:
    """Strips recursive self-echoing headers like [Otis (Architect)]: or [Bonsai 27B Q1 0 (Architect)]:, <think> tags, and 'Thinking Process:' outputs."""
    if not content:
        return ""

    cleaned = content.strip()

    # Strip <think>...</think> blocks or unclosed <think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)

    # Strip "Thinking Process: ..." or "Here's a thinking process: ..." blocks
    cleaned = re.sub(r"(?:Thinking Process|Here's a thinking process|Thinking):.*?(?:\n\n|\n(?=[A-Z0-9]))", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^(?:Thinking Process|Here's a thinking process|Thinking):.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Repeatedly strip starting headers like [Name (Role)]: or [Name]:
    while True:
        new_cleaned = re.sub(r"^(?:\[[^\]]+\]:\s*)+", "", cleaned.strip())
        new_cleaned = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", new_cleaned.strip())
        if new_cleaned == cleaned:
            break
        cleaned = new_cleaned

    return cleaned.strip()

def normalize_messages_for_gguf(system_prompt: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Normalizes message sequence so GGUF / llama-cpp-python never throws 'No user query found in messages' error."""
    full_msgs = [{"role": "system", "content": system_prompt}]

    user_msgs_count = 0
    for m in messages:
        r = m.get("role", "user")
        c = m.get("content", "")
        if r == "user" and c.strip():
            user_msgs_count += 1
        full_msgs.append({"role": r, "content": c})

    if user_msgs_count == 0 or full_msgs[-1]["role"] == "assistant":
        # Inject explicit user turn prompt to avoid GGUF 'No user query found' error
        full_msgs.append({
            "role": "user",
            "content": "[System Direct]: It is your turn to respond to the ongoing discussion/task."
        })

    return full_msgs
