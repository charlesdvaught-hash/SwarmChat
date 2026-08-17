import re
from typing import List, Dict, Any

def sanitize_message_content(content: str) -> str:
    """Strips recursive self-echoing headers like [Otis (Architect)]: or [Bonsai 27B Q1 0 (Architect)]:"""
    if not content:
        return ""

    # Pattern to match starting headers like [Name (Role)]: or [Name]: repeatedly
    pattern = r"^(?:\[[^\]]+\]:\s*)+"
    cleaned = re.sub(pattern, "", content.strip())

    # Also clean if header is on a newline at the beginning
    cleaned = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", cleaned)
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
