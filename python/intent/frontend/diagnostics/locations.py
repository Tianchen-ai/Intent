from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceSpan:
    filename: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    def __post_init__(self) -> None:
        if not isinstance(self.filename, str) or not self.filename:
            raise ValueError("source filename must not be empty")
        for name, value in (
            ("start_line", self.start_line),
            ("start_column", self.start_column),
            ("end_line", self.end_line),
            ("end_column", self.end_column),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.start_line <= 0 or self.end_line <= 0:
            raise ValueError("source lines are one-based and must be positive")
        if self.start_column < 0 or self.end_column < 0:
            raise ValueError("source columns must be non-negative")
        if (self.end_line, self.end_column) < (
            self.start_line,
            self.start_column,
        ):
            raise ValueError("source span end precedes its start")

    def format(self) -> str:
        return f"{self.filename}:{self.start_line}:{self.start_column + 1}"


@dataclass(frozen=True, slots=True)
class Location:
    primary: SourceSpan
    expansion_stack: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.primary, SourceSpan):
            raise TypeError("location primary must be a SourceSpan")
        object.__setattr__(self, "expansion_stack", tuple(self.expansion_stack))
        if any(not isinstance(span, SourceSpan) for span in self.expansion_stack):
            raise TypeError("location expansion stack must contain SourceSpan values")

    def format(self) -> str:
        return self.primary.format()


def file_location(filename: str, line: int = 1, column: int = 0) -> Location:
    return Location(SourceSpan(filename, line, column, line, column))
