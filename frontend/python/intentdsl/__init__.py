"""Public author API. Kernel bodies are captured, never executed as Python."""

from dataclasses import dataclass
from inspect import getsource
from textwrap import dedent


from .completion import Completion

__version__ = "2.0.0"


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


_DENORMAL_MODES = ("ieee", "preserve-sign", "positive-zero", "dynamic")
_FPMATH_MODES = ("strict", "relaxed", "bf16", "fast")


def kernel(fn=None, *, numerics="source", fast_math=False, denormal_fp_math="ieee"):
    if numerics not in ("source", "relaxed_normalization"):
        raise ValueError(f"unsupported kernel numerics: {numerics}")
    if type(fast_math) is not bool:
        raise ValueError("fast_math must be a boolean")
    if denormal_fp_math not in _DENORMAL_MODES:
        raise ValueError(f"unsupported denormal_fp_math: {denormal_fp_math}")

    def wrap(f):
        return Definition(f.__name__, dedent(getsource(f)).encode())

    return wrap if fn is None else wrap(fn)

def program(fn):
    """Capture the author's ordered host operations without running their Python."""
    from .composition import capture_program

    return capture_program(fn)


def library_path(library=None):
    """Resolve the installed or bundled libintentdsl.so path."""
    import os
    import sys
    from pathlib import Path

    if library:
        return Path(library).resolve(strict=True)
    env = os.environ.get("INTENTDSL_LIBRARY", "").strip()
    if env:
        return Path(env).resolve(strict=True)
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "_bundled" / "lib" / "libintentdsl.so",
        pkg_dir.parents[2] / "libintentdsl.so",
        Path(sys.prefix) / "lib" / "libintentdsl.so",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve(strict=True)
    raise FileNotFoundError(
        "libintentdsl.so not found in installed package or sys.prefix; "
        "reinstall intentdsl wheel or set INTENTDSL_LIBRARY"
    )


def layout(*, library=None, configuration=None):
    """Return the compiled product tree, units, exits, and active configuration."""
    from compilerinfra import Compiler

    with Compiler(library_path(library), configuration) as compiler:
        return compiler.describe()


def doctor(*, backend=None, library=None):
    """Probe the installed compiler, external toolchains, optional libraries, and devices."""
    from .diagnostics import run_doctor

    return run_doctor(backend=backend, library=library)


def setup(*, backend=None, library=None):
    """Verify installation assets and return setup status and actionable environment hints."""
    from .diagnostics import run_setup

    return run_setup(backend=backend, library=library)


def manual(topic=None, *, lang="zh"):
    """Query the packaged offline manual ('zh' or 'en') without network access."""
    from .diagnostics import read_manual

    return read_manual(topic=topic, lang=lang)


def skills(*, library=None):
    """Discover and read manifest-verified product skills from the installed package."""
    from .diagnostics import read_skills

    return read_skills(library=library)


def jit(
    definition,
    *,
    backend,
    library=None,
    configuration=None,
    fpmath_mode="strict",
    fast_math=False,
    denormal_fp_math=None,
    numerics=None,
):
    from compilerinfra import Compiler

    from .compiler import Kernel

    if fpmath_mode not in _FPMATH_MODES:
        raise ValueError(f"unsupported fpmath_mode: {fpmath_mode}")
    if type(fast_math) is not bool:
        raise ValueError("fast_math must be a boolean")
    if denormal_fp_math is not None and denormal_fp_math not in _DENORMAL_MODES:
        raise ValueError(f"unsupported denormal_fp_math: {denormal_fp_math}")
    math_options = {"fpmath_mode": fpmath_mode, "fast_math": fast_math}
    if numerics is not None:
        if numerics not in ("source", "relaxed_normalization"):
            raise ValueError(f"unsupported normalization policy: {numerics}")
        math_options["numerics"] = numerics
    if denormal_fp_math is not None:
        math_options["denormal_fp_math"] = denormal_fp_math
    path = library_path(library)
    compiler = Compiler(path, configuration)
    if backend not in compiler.describe()["exits"]:
        raise ValueError(f"unknown exit: {backend}")
    from .composition import Program, ProgramKernel
    if isinstance(definition, Program):
        return ProgramKernel(compiler, definition, backend, math_options=math_options)
    return Kernel(compiler, definition, backend, math_options=math_options)
