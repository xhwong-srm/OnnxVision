class VisionWorkflowError(Exception):
    """Base error for expected application failures."""


class ConfigurationError(VisionWorkflowError):
    """A request is incomplete or contains incompatible options."""


class DatasetFormatError(VisionWorkflowError):
    """A dataset cannot be read or written under its declared contract."""


class ValidationFailedError(VisionWorkflowError):
    """A requested validation gate did not pass."""


class BackendUnavailableError(VisionWorkflowError):
    """A selected backend is not installed or cannot run on this host."""


class ArtifactError(VisionWorkflowError):
    """An artifact could not be safely created, loaded, or finalized."""
