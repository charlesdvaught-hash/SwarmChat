"""Shared helpers used across the SwarmChat backend modules."""

import functools
import json
import os
import time
from typing import Any, Callable, Dict, Optional

BYTES_PER_GB = 1024 ** 3
BYTES_PER_MB = 1024 ** 2


def timestamped_id(prefix: str) -> str:
    """Millisecond-resolution identifier, e.g. `msg_1723946400123`."""
    return f"{prefix}_{int(time.time() * 1000)}"


def format_clock(timestamp: Optional[float] = None) -> str:
    return time.strftime("%H:%M:%S", time.localtime(timestamp if timestamp is not None else time.time()))


def format_datetime(timestamp: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp if timestamp is not None else time.time()))


def bytes_to_gb(num_bytes: float, precision: int = 2) -> float:
    return round(num_bytes / BYTES_PER_GB, precision)


def bytes_to_mb(num_bytes: float, precision: int = 2) -> float:
    return round(num_bytes / BYTES_PER_MB, precision)


def file_size_gb(path: str, precision: int = 2) -> float:
    try:
        return bytes_to_gb(os.path.getsize(path), precision)
    except OSError:
        return 0.0


def file_size_mb(path: str, precision: int = 2) -> float:
    try:
        return bytes_to_mb(os.path.getsize(path), precision)
    except OSError:
        return 0.0


def is_existing_file(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)


def read_text_file(path: str, errors: str = "ignore") -> str:
    with open(path, "r", encoding="utf-8", errors=errors) as f:
        return f.read()


def write_text_file(path: str, content: str, make_parents: bool = True) -> int:
    if make_parents:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return len(content)


def write_json_file(path: str, data: Any, make_parents: bool = True, indent: int = 2) -> None:
    write_text_file(path, json.dumps(data, indent=indent), make_parents=make_parents)


def failure(error: str) -> Dict[str, Any]:
    return {"success": False, "error": error}


def guarded(func: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Converts unexpected exceptions into the `{"success": False, "error": ...}` result shape."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            return failure(str(exc))

    return wrapper


def extract_directive(text: str, tag: str) -> Optional[str]:
    """Extracts the payload of an inline model directive such as `[JOURNAL: ...]`.

    Returns the stripped payload, or None when the tag is absent or unterminated.
    """
    marker = f"[{tag}:"
    start = text.find(marker)
    if start == -1:
        return None
    start += len(marker)
    end = text.find("]", start)
    if end == -1:
        return None
    return text[start:end].strip()
