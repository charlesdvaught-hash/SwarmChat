"""Exception hierarchy for SwarmChat backend failures."""


class SwarmChatError(Exception):
    """Base class for all SwarmChat backend errors."""


class MemoryPersistenceError(SwarmChatError):
    """Raised when shared memory cannot be read from or written to disk."""


class ModelInvocationError(SwarmChatError):
    """Raised when a model backend fails to produce a response.

    Carries the model identity so callers can attribute the failure.
    """

    def __init__(self, message: str, model_id: str = "", provider: str = ""):
        super().__init__(message)
        self.model_id = model_id
        self.provider = provider


class ProviderNotConfiguredError(ModelInvocationError):
    """Raised when a provider is selected but cannot be used (missing key, unsupported)."""


class ModelLoadError(ModelInvocationError):
    """Raised when a local GGUF model cannot be loaded."""


class ToolExecutionError(SwarmChatError):
    """Raised when a tool cannot be dispatched or fails unexpectedly."""


class DirectiveParseError(SwarmChatError):
    """Raised when an inline model directive (e.g. [UPDATE_CONFIG: ...]) is malformed."""
