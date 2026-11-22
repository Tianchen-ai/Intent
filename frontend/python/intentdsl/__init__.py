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


class InOut(_Direction):
    pass


bf16 = "bf16"
i8 = "i8"
u8 = "u8"
f8e4m3fn = "f8e4m3fn"
i32 = "i32"
i64 = "i64"
u64 = "u64"
f16 = "f16"
f32 = "f32"


@dataclass(frozen=True)
class Definition:
    name: str
    source: bytes


def kernel(fn):
    return Definition(fn.__name__, dedent(getsource(fn)).encode())

def program(fn):
    """Capture the author's ordered host operations without running their Python."""
    from .composition import capture_program

    return capture_program(fn)


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
    from .composition import Program, ProgramKernel
    if isinstance(definition, Program):
        return ProgramKernel(compiler, definition, backend)
    return Kernel(compiler, definition, backend)
