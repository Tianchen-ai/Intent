from __future__ import annotations

from dataclasses import dataclass

from intent.diagnostics import IntentError

from .locations import Location


@dataclass(frozen=True, slots=True)
class FrontendDiagnostic:
    message: str
    location: Location

    def format(self) -> str:
        return f"{self.location.format()}: error: {self.message}"


class FrontendError(IntentError):
    def __init__(self, message: str, location: Location) -> None:
        self.diagnostic = FrontendDiagnostic(message, location)
        super().__init__(self.diagnostic.format())


__all__ = [
    "FrontendDiagnostic",
    "FrontendError",
]
