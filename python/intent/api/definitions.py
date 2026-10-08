from __future__ import annotations

import functools
import inspect
import textwrap
from dataclasses import dataclass
from enum import Enum
from types import FunctionType
from typing import Any
from typing import Callable
from typing import Generic
from typing import ParamSpec
from typing import TypeVar

from intent.diagnostics import DefinitionError
from intent.diagnostics import LanguageUseError


P = ParamSpec("P")
R = TypeVar("R")


class DefinitionKind(Enum):
    KERNEL = "kernel"
    HELPER = "helper"


@dataclass(frozen=True, slots=True)
class DefinitionSource:
    filename: str
    first_line: int


class Definition(Generic[P, R]):
    def __init__(self, function: Callable[P, R], kind: DefinitionKind) -> None:
        if not isinstance(function, FunctionType):
            raise DefinitionError("Intent definitions must decorate a Python function")
        self.python_function = function
        self.kind = kind
        self.source = DefinitionSource(
            filename=inspect.getsourcefile(function) or function.__code__.co_filename,
            first_line=function.__code__.co_firstlineno,
        )
        functools.update_wrapper(self, function)

    @property
    def signature(self) -> inspect.Signature:
        return inspect.signature(self.python_function, eval_str=True)

    def source_text(self) -> str:
        return textwrap.dedent(inspect.getsource(self.python_function))

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if self.kind is DefinitionKind.KERNEL:
            raise LanguageUseError(
                f"kernel {self.__name__} must be compiled before runtime invocation"
            )
        raise LanguageUseError(
            f"helper {self.__name__} is only callable inside an @intent.kernel body"
        )

    def __repr__(self) -> str:
        return f"<intent {self.kind.value} {self.__module__}.{self.__qualname__}>"


class KernelDefinition(Definition[P, R]):
    def __init__(self, function: Callable[P, R]) -> None:
        super().__init__(function, DefinitionKind.KERNEL)


class HelperDefinition(Definition[P, R]):
    def __init__(self, function: Callable[P, R]) -> None:
        super().__init__(function, DefinitionKind.HELPER)


def kernel(function: Callable[P, R] | None = None) -> Any:
    if function is None:
        return KernelDefinition
    return KernelDefinition(function)


def fn(function: Callable[P, R] | None = None) -> Any:
    if function is None:
        return HelperDefinition
    return HelperDefinition(function)
