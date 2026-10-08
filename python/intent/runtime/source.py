from __future__ import annotations

import builtins as python_builtins
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
import linecache
from pathlib import Path

from .artifact import BackendIRCollector
from .artifact import CompiledArtifact


_MATERIALIZATION_IDS = count()


@dataclass(frozen=True, slots=True)
class AuthorSourceProgram:
    run: Callable[..., object]
    launch: Callable[..., object]

    @property
    def interface(self):
        raise NotImplementedError("author-supplied source has no compiler invocation interface")


def load_python_source(*, target_name: str, source: str, entry_name: str,
                       source_path: Path | None = None) -> dict[str, object]:
    materialization_id = next(_MATERIALIZATION_IDS)
    module_name = (
        f"intent.generated.{target_name}.{entry_name}.{materialization_id}"
    )
    if source_path is None:
        filename = f"<{module_name}>"
    else:
        path = source_path.expanduser().resolve(strict=True)
        if path.read_text(encoding="utf-8") != source:
            raise ValueError(f"materialized Python source differs from its source file: {path}")
        filename = str(path)
    source_lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, source_lines, filename)
    namespace: dict[str, object] = {
        "__name__": module_name,
    }
    if source_path is not None:
        namespace["__file__"] = filename
    code = python_builtins.compile(source, filename, "exec", dont_inherit=True)
    exec(code, namespace)
    return namespace


def materialize_python_source(
    *,
    target_name: str,
    source: str,
    module_text: str,
    entry_name: str,
    device: int,
    backend_ir_collector: BackendIRCollector | None,
    source_path: Path | None = None,
) -> CompiledArtifact:
    """Load an author-supplied provider program with explicit host callables."""
    namespace = load_python_source(target_name=target_name, source=source, entry_name=entry_name,
                                   source_path=source_path)
    launcher = namespace.get("launch")
    runner = namespace.get("run")
    if not callable(launcher) or not callable(runner):
        raise RuntimeError(
            f"author {target_name} source must define launch() and run()"
        )
    return CompiledArtifact(
        source=source,
        mlir=module_text,
        device=device,
        runtime=AuthorSourceProgram(run=runner, launch=launcher),
        _backend_ir_collector=backend_ir_collector,
    )
