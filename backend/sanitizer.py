import re
from typing import List, Dict, Any

def sanitize_message_content(content: str) -> str:
    """Strips recursive self-echoing headers like [Otis (Architect)]:, <think> tags, orphan </think> blocks, thinking process logs, and meta-commentary."""
    if not content:
        return ""

    cleaned = content.strip()

    # Aggressively strip thinking tags <think>...</think>, <thought>...</thought>, <reasoning>...</reasoning> for Qwen, Llama, Dolphin
    cleaned = re.sub(r"<(?:think|thought|reasoning)>.*?</(?:think|thought|reasoning)>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<(?:think|thought|reasoning)>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Handle orphan closing tags </think>, </thought>, </reasoning>
    for tag in ["</think>", "</thought>", "</reasoning>"]:
        if tag in cleaned.lower():
            idx = cleaned.lower().rfind(tag)
            cleaned = cleaned[idx + len(tag):].strip()

    # Strip "Thinking Process: ...", "Chain of thought: ...", or "Thinking: ..." blocks
    cleaned = re.sub(r"(?:Thinking Process|Here's a thinking process|Thinking|Chain of Thought|Thought):.*?(?:\n\n|\n(?=[A-Z0-9]))", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^(?:Thinking Process|Here's a thinking process|Thinking|Chain of Thought|Thought):.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Strip meta-commentary role-confusion paragraphs (e.g. "Apologies for the confusion...", "The user is telling me it's my turn...", "Let me read the context again...")
    meta_patterns = [
        r"^(?:Apologies|Sorry) for the confusion (?:earlier|previously)[^\.\n]*[\.\n]?",
        r"^(?:Let me|I need to) read the context again[^\.\n]*[\.\n]?",
        r"^(?:The user is|The system is) telling me it's my turn[^\.\n]*[\.\n]?",
        r"^Okay, the user wants [^\.\n]*[\.\n]?",
        r"^This is a stateless execution mode[^\.\n]*[\.\n]?"
    ]
    for pattern in meta_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Repeatedly strip starting headers like [Name (Role)]: or [Name]:
    while True:
        new_cleaned = re.sub(r"^(?:\[[^\]]+\]:\s*)+", "", cleaned.strip())
        new_cleaned = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", new_cleaned.strip())
        if new_cleaned == cleaned:
            break
        cleaned = new_cleaned

    return cleaned.strip()

def normalize_messages_for_gguf(system_prompt: str, messages: List[Dict[str, str]], role: str = "Participant") -> List[Dict[str, str]]:
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
        # Inject unambiguous, role-affirming user turn prompt to avoid GGUF 'No user query found' error
        full_msgs.append({
            "role": "user",
            "content": f"Please contribute your next input to the room in your assigned role as {role}."
        })

    return full_msgs
