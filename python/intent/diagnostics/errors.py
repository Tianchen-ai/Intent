class IntentError(Exception):
    """Base class for user-facing Intent errors."""


class LanguageUseError(IntentError):
    """Raised when a DSL construct is used outside frontend capture."""


class DefinitionError(IntentError):
    """Raised when a kernel or helper definition is malformed."""
