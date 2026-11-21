"""Public author API. Kernel bodies are captured, never executed as Python."""

from dataclasses import dataclass
from inspect import getsource
from textwrap import dedent


class _Direction:
    def __class_getitem__(cls, item):
        return cls, item


class In(_Direction):
    pass


class Out(_Direction):
    pass


f16 = "f16"
f32 = "f32"


@dataclass(frozen=True)
class Definition:
    name: str
    source: bytes


def kernel(fn):
    return Definition(fn.__name__, dedent(getsource(fn)).encode())


def jit(definition, *, backend, library=None, configuration=None):
    import os
    import sys
    from pathlib import Path
    from compilerinfra import Compiler

    from .compiler import Kernel

    installed = Path(__file__).resolve().parents[3] / "libintentdsl.so"
    path = (
        library
        or os.environ.get("INTENTDSL_LIBRARY")
        or (
            installed
            if installed.is_file()
            else Path(sys.prefix) / "lib/libintentdsl.so"
        )
    )
    compiler = Compiler(path, configuration)
    if backend not in compiler.describe()["exits"]:
        raise ValueError(f"unknown exit: {backend}")
    return Kernel(compiler, definition, backend)
