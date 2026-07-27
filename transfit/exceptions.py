"""Shared exceptions for model states that are outside their physical domain."""


class NonPhysicalModelError(ValueError):
    """Raised when valid API inputs describe a state the selected model cannot represent."""
