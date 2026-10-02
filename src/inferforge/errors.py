"""Explicit failure categories shared by InferForge components.

Raise these errors with a descriptive message and preserve underlying failures
with ``raise ... from cause``. HTTP status mapping belongs to the serving layer.
"""


class InferForgeError(Exception):
    """Base exception for expected project failures."""


class ConfigurationError(InferForgeError):
    """Configuration is invalid or contains an unsupported combination."""


class DeviceUnavailableError(InferForgeError):
    """The requested execution device is unavailable."""


class BackendUnavailableError(InferForgeError):
    """The requested backend cannot run with available runtime dependencies."""


class ModelLoadError(InferForgeError):
    """A model or checkpoint cannot be loaded."""


class InvalidInputError(InferForgeError):
    """Input fails the workload's validation requirements."""


class QueueFullError(InferForgeError):
    """The bounded request queue cannot accept more work."""


class InferenceTimeoutError(InferForgeError):
    """An inference request exceeds its allowed wait or execution time."""


class ArtifactCompatibilityError(InferForgeError):
    """An exported model or engine is incompatible with the requested runtime."""
