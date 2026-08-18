import ipaddress
import os
import re
import secrets
from typing import Any, Dict, List, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

TOKEN_HEADER = "X-SwarmChat-Token"

#: Identifiers that are safe to use as a single filesystem path component.
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

_LOOPBACK_HOSTNAMES = ("localhost", "localhost.localdomain")
_LOCAL_ORIGIN_HOSTS = ("localhost", "127.0.0.1", "[::1]")
_LOCAL_ORIGIN_PORTS = (8000, 5173, 4173, 3000)

SENSITIVE_MODEL_FIELDS = ("api_key",)


def get_api_token() -> str:
    return os.environ.get("SWARMCHAT_API_TOKEN", "").strip()


def _split_env_list(name: str) -> List[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def get_allowed_origins() -> List[str]:
    origins = [o.rstrip("/") for o in _split_env_list("SWARMCHAT_ALLOWED_ORIGINS")]
    for host in _LOCAL_ORIGIN_HOSTS:
        for port in _LOCAL_ORIGIN_PORTS:
            origins.append(f"http://{host}:{port}")
    deduped: List[str] = []
    for origin in origins:
        if origin not in deduped:
            deduped.append(origin)
    return deduped


def get_allowed_hosts() -> List[str]:
    configured = _split_env_list("SWARMCHAT_ALLOWED_HOSTS")
    if configured:
        return configured
    return ["localhost", "localhost.localdomain", "127.0.0.1", "::1", "testserver"]


def is_loopback_client(host: Optional[str]) -> bool:
    if not host:
        return False
    candidate = host.strip().strip("[]")
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return candidate.lower() in _LOOPBACK_HOSTNAMES


def _presented_token(request: Request) -> str:
    header_token = request.headers.get(TOKEN_HEADER, "").strip()
    if header_token:
        return header_token
    authorization = request.headers.get("authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[len("bearer "):].strip()
    return ""


def _deny(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"success": False, "error": detail})


def check_api_access(request: Request) -> Optional[JSONResponse]:
    """Authorizes an /api request. Returns a denial response, or None when the request is allowed.

    Access requires either a matching SWARMCHAT_API_TOKEN or a loopback client. Requests carrying a
    foreign Origin are refused so a web page cannot drive the local API through the browser.
    """
    token = get_api_token()
    if token:
        presented = _presented_token(request)
        if not presented or not secrets.compare_digest(presented, token):
            return _deny(401, f"Missing or invalid API token. Send it in the {TOKEN_HEADER} header.")
    elif not is_loopback_client(request.client.host if request.client else None):
        return _deny(
            403,
            "Remote access to the SwarmChat API is disabled. Set SWARMCHAT_API_TOKEN on the server "
            "and send the token with each request to allow non-local clients.",
        )

    origin = request.headers.get("origin", "").rstrip("/")
    if origin and origin not in get_allowed_origins():
        return _deny(403, f"Origin '{origin}' is not allowed. Configure SWARMCHAT_ALLOWED_ORIGINS to permit it.")

    return None


def validate_safe_id(value: str, field_name: str = "id") -> str:
    """Rejects identifiers that could escape a directory or confuse path handling."""
    if not SAFE_ID_PATTERN.match(value or ""):
        raise ValueError(
            f"{field_name} must be 1-64 characters of letters, digits, '_', '.' or '-' and start alphanumerically."
        )
    return value


def redact_model_config(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Copies a model config with provider credentials removed, keeping a flag for the UI."""
    redacted = dict(model_cfg)
    for field in SENSITIVE_MODEL_FIELDS:
        if field in redacted:
            redacted[f"{field}_set"] = bool(redacted.get(field))
            redacted[field] = ""
    return redacted


def redact_model_configs(models: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {m_id: redact_model_config(cfg) for m_id, cfg in models.items()}
