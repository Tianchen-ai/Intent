from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING
from .options import CompileOptions

if TYPE_CHECKING:
    from intent.runtime import CompiledArtifact, PublicInterface
    from intent.runtime.contract import ProgramContract
    from intent.targets.base import ResolvedTarget, Target
    from intent.targets.specification import CompilationTarget


@dataclass(frozen=True, slots=True)
class CompiledIR:
    """Verified KIR or shared physical IR, without provider source or a runtime."""

    stage: str
    ir: str
    cache_directory: Path


@dataclass(frozen=True, slots=True)
class OptimizedIR:
    """IR produced by the requested standard pass pipeline, without execution.

    cache_directory archives this invocation's input, command, output and
    diagnostics. Optimization results are not reused across requests: a textual
    pipeline may read external resources whose dependencies belong to its passes.
    """

    ir: str
    pipeline: str
    cache_directory: Path


@dataclass(frozen=True, slots=True, init=False)
class GeneratedProgram:
    """Provider source, IR and metadata, with an optional process-local binding.

    load() restores the compiler contract only. materialize() binds a local
    runtime; native compilation and loading follow that provider's lifecycle.
    """

    source: str
    ir: str
    cache_directory: Path
    entry_name: str
    _binding: ResolvedTarget | None = field(default=None, repr=False, compare=False)
    _contract: ProgramContract = field(repr=False, compare=False)

    def __init__(self, source: str, ir: str, metadata: dict[str, object],
                 cache_directory: Path, entry_name: str,
                 _binding: ResolvedTarget | None = None) -> None:
        from intent.runtime.contract import ProgramContract

        contract = ProgramContract.read(source, metadata)
        if not isinstance(ir, str) or not ir.strip():
            raise ValueError("generated program IR must be nonempty text")
        if not isinstance(entry_name, str) or not entry_name.strip():
            raise ValueError("generated program requires a nonempty diagnostic name")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "ir", ir)
        object.__setattr__(self, "cache_directory", Path(cache_directory))
        object.__setattr__(self, "entry_name", entry_name)
        object.__setattr__(self, "_binding", _binding)
        object.__setattr__(self, "_contract", contract)

    @property
    def metadata(self) -> dict[str, object]:
        """An inspection copy of the validated compiler metadata.

        Editing the returned object does not change this program or a runtime
        already materialized from it. The saved compiler contract remains fixed.
        """
        return self._contract.metadata

    @property
    def compile_options(self) -> CompileOptions:
        """Read the saved numerical permissions and optimization controls.

        Loading or materializing preserves this policy; it does not replace it
        with the current compiler's defaults or a runtime fast-math setting.
        """
        return self._contract.options

    @property
    def target(self) -> CompilationTarget:
        """Compiler target facts from this program's authoritative metadata, without runtime probing."""
        return self._contract.target

    @property
    def interface(self) -> PublicInterface:
        """Read the compiler's public parameter contract without a device or SDK.

        This also validates the serialized native slots or GPU host dependencies.
        The returned declaration excludes native slots and compiler-private
        resources; materialization consumes the same authoritative metadata.
        """
        return self._contract.interface

    def save(self, directory: str | Path) -> Path:
        """Save source, final IR and compiler metadata into a new directory.

        Runtime devices, SDK paths and native handles are not serialized. The
        files use the compiler's existing artifact layout and can be loaded on
        another host without a provider SDK.
        """
        from .toolchain import CompilationStageError

        path = Path(directory).expanduser().absolute()
        try:
            if path.exists():
                raise FileExistsError(f"generated program destination already exists: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as temporary:
                staging = Path(temporary) / "program"
                staging.mkdir()
                (staging / "kernel.source").write_text(self.source, encoding="utf-8")
                (staging / "kernel.mlir").write_text(self.ir, encoding="utf-8")
                (staging / "artifact.json").write_text(
                    json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                staging.rename(path)
        except (OSError, UnicodeError, TypeError, KeyError, ValueError, NotImplementedError) as error:
            raise CompilationStageError("generated_program_export", str(error),
                                        cache_directory=self.cache_directory) from error
        return path

    @classmethod
    def load(cls, directory: str | Path, *, name: str | None = None) -> GeneratedProgram:
        """Load an exported program or compiler artifact without loading native code.

        A restored program has no implicit runtime device. Supply a target to
        materialize(). name optionally overrides its diagnostic identifier; it
        does not select an entry from the compiled module.
        """
        from .toolchain import CompilationStageError

        path = Path(directory).expanduser().absolute()
        try:
            source = (path / "kernel.source").read_text(encoding="utf-8")
            ir = (path / "kernel.mlir").read_text(encoding="utf-8")
            metadata = json.loads((path / "artifact.json").read_text(encoding="utf-8"))
            if not source.strip() or not ir.strip() or not isinstance(metadata, dict):
                raise ValueError("generated program requires nonempty source, IR and object metadata")
            diagnostic_name = metadata["entry_name"] if name is None else name
            if not isinstance(diagnostic_name, str) or not diagnostic_name.strip():
                raise ValueError("generated program requires a nonempty diagnostic name")
            return cls(source, ir, metadata, path, diagnostic_name)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, NotImplementedError) as error:
            detail = str(error)
            if isinstance(error, (ValueError, KeyError, TypeError, NotImplementedError)):
                detail += "\nRegenerate an incompatible artifact from its original Intent definition or KIR; generated source is not the original algorithm."
            raise CompilationStageError("generated_program_loading", detail, cache_directory=path) from error

    def materialize(self, *, target: Target | None = None) -> CompiledArtifact:
        """Bind this program to a matching runtime target, without recompiling KIR.

        An explicit target chooses the local device or SDK. Programs generated
        with a local target may reuse that process-local binding; programs loaded
        from disk or generated from explicit compiler facts require a target.
        Providers may defer native compilation and loading until invocation.
        A prepared call's compile() requests compilation without tuning or
        execution. Materialization never launches or retargets the program.
        """
        from intent.targets.base import ResolvedTarget
        from .toolchain import CompilationStageError

        try:
            binding = target.resolve() if target is not None else self._binding
            if binding is None:
                raise ValueError("materialize requires an explicit runtime target for this generated program")
            if not isinstance(binding, ResolvedTarget):
                raise NotImplementedError("the selected target supplies compiler facts only; use a runtime target or its AOT toolchain")
            artifact = binding.materialize(self)
        except CompilationStageError:
            raise
        except Exception as error:
            raise CompilationStageError("generated_source_materialization", str(error),
                                        cache_directory=self.cache_directory) from error
        artifact.cache_directory = self.cache_directory
        return artifact
