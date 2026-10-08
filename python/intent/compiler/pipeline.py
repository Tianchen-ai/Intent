from __future__ import annotations

from pathlib import Path

from intent.api import KernelDefinition
from intent.frontend import lower_to_mlir
from intent.runtime import CompiledArtifact
from intent.targets.base import Target, SourceTarget, ResolvedTarget
from intent.targets.specification import CompilationTarget, require_matching_target

from .artifact import CompiledIR, GeneratedProgram, OptimizedIR
from .options import CompileOptions, compilation_arguments
from .toolchain import CompilationStageError
from .toolchain import CompilerStage
from .toolchain import run_compiler
from .toolchain import run_optimizer


def compile(
    definition: KernelDefinition[object, object],
    *,
    target: Target,
    compiler: str | Path | None = None,
    constexprs: dict[str, object] | None = None,
    tuning_config: str | Path | None = None,
    options: CompileOptions | None = None,
) -> CompiledArtifact:
    kernel_mlir = _capture(definition, constexprs)
    resolved, binding = _resolve(target)
    if binding is None:
        raise NotImplementedError("This target only generates source; use intent.generate, not intent.compile")
    program = _generate_source(kernel_mlir, resolved, compiler, tuning_config, definition.__name__,
                               binding=binding, compile_options=options)
    return program.materialize()


def generate(
    definition: KernelDefinition[object, object],
    *,
    target: SourceTarget,
    compiler: str | Path | None = None,
    constexprs: dict[str, object] | None = None,
    tuning_config: str | Path | None = None,
    options: CompileOptions | None = None,
) -> GeneratedProgram:
    kernel_mlir = _capture(definition, constexprs)
    resolved, binding = _resolve(target)
    return _generate_source(kernel_mlir, resolved, compiler, tuning_config, definition.__name__,
                            binding=binding, compile_options=options)


def generate_from_ir(
    ir: str,
    *,
    input_stage: str = "shared",
    name: str,
    target: SourceTarget,
    compiler: str | Path | None = None,
) -> GeneratedProgram:
    """Generate provider source from existing KIR or shared physical IR.

    The caller supplies the IR stage, a diagnostic name and target; none is
    inferred from the text. This compiles the whole module: callable entries and
    candidates come from its IR and metadata, and name does not select a kernel.
    Shared input retains its existing physical
    program, configuration and compile options. Compile options are not reselected
    by this API. The native compiler checks that its capabilities
    agree with the selected target, without reconstructing or retuning that IR.
    The returned program uses the usual materialize() path and does not launch.
    """
    if not isinstance(ir, str) or not ir.strip():
        raise ValueError("ir must contain nonempty MLIR text")
    if input_stage not in ("kir", "shared"):
        raise ValueError("input_stage must be 'kir' or 'shared'")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must provide a nonempty diagnostic identifier for the generated program")
    resolved, binding = _resolve(target)
    return _generate_source(ir, resolved, compiler, None, name, input_stage=input_stage, binding=binding)


def compile_ir(
    definition: KernelDefinition[object, object],
    *,
    stage: str = "kir",
    target: SourceTarget | None = None,
    compiler: str | Path | None = None,
    constexprs: dict[str, object] | None = None,
    tuning_config: str | Path | None = None,
    options: CompileOptions | None = None,
) -> CompiledIR:
    """Compile a definition to verified KIR or shared physical IR.

    ``stage='kir'`` normalizes and verifies target-independent Kernel IR using
    the Intent compiler. It accepts neither a target nor a tuning configuration,
    and does not resolve a device or import a provider SDK. ``stage='shared'``
    requires a target and stops after that execution family's shared pipeline.
    Both return IR and its cache directory without materialization or launch;
    use ``generate`` for provider source and its invocation metadata.
    """
    policy_arguments = compilation_arguments(options)
    selected = CompilerStage(stage)
    if selected is CompilerStage.PROVIDER:
        raise ValueError("compile_ir supports 'kir' and 'shared'; use generate for provider source")
    if selected is CompilerStage.KIR:
        if target is not None or tuning_config is not None:
            raise ValueError("KIR compilation does not accept a target or tuning configuration")
    elif target is None:
        raise ValueError("shared IR compilation requires a target")
    kernel_mlir = _capture(definition, constexprs)
    if selected is CompilerStage.KIR:
        arguments, role = policy_arguments, "Intent compiler"
    else:
        resolved, _ = _resolve(target)
        arguments, role = _options(resolved, tuning_config) + policy_arguments, resolved.compiler_role
    output = run_compiler(compiler, kernel_mlir, arguments, role, stage=selected)
    return CompiledIR(selected.value, output.ir, output.directory)


def _capture(definition, constexprs) -> str:
    try:
        return lower_to_mlir(definition, constexprs=constexprs)
    except Exception as error:
        raise CompilationStageError("frontend_kir", str(error)) from error


def _resolve(target: SourceTarget) -> tuple[CompilationTarget, ResolvedTarget | None]:
    try:
        resolved = target.resolve()
        if isinstance(resolved, ResolvedTarget):
            return resolved.compilation, resolved
        if isinstance(resolved, CompilationTarget):
            return resolved, None
        raise TypeError("target resolution must provide compiler facts or a runtime binding")
    except Exception as error:
        raise CompilationStageError("target_resolution", str(error)) from error


def _options(resolved: CompilationTarget, tuning_config: str | Path | None) -> tuple[str, ...]:
    return resolved.compiler_options + (
        (f"--tuning-config={Path(tuning_config).resolve()}",)
        if tuning_config is not None else ()
    )


def _generate_source(kernel_mlir, resolved, compiler, tuning_config, entry_name,
                     *, input_stage: str | None = None, binding: ResolvedTarget | None = None,
                     compile_options: CompileOptions | None = None) -> GeneratedProgram:
    options = _options(resolved, tuning_config) + compilation_arguments(compile_options)
    if input_stage is not None:
        options += (f"--input-stage={input_stage}",)
    output = run_compiler(
        compiler,
        kernel_mlir,
        options,
        resolved.compiler_role,
    )
    try:
        program = GeneratedProgram(output.source, output.ir, output.metadata, output.directory, entry_name, binding)
        require_matching_target(program.target, resolved)
        if compile_options is not None and program.compile_options != compile_options:
            raise ValueError("compiler output does not preserve the requested compile options")
    except (KeyError, TypeError, ValueError, NotImplementedError) as error:
        raise CompilationStageError("compiler_output", str(error), cache_directory=output.directory) from error
    return program


def compile_shared_gpu(
    definition: KernelDefinition[object, object],
    *,
    target: Target,
    compiler: str | Path | None = None,
    constexprs: dict[str, object] | None = None,
    tuning_config: str | Path | None = None,
    options: CompileOptions | None = None,
) -> str:
    return compile_ir(
        definition, stage="shared", target=target, compiler=compiler,
        constexprs=constexprs, tuning_config=tuning_config, options=options,
    ).ir


def optimize_ir(
    ir: str,
    *,
    pipeline: str,
    optimizer: str | Path | None = None,
) -> OptimizedIR:
    """Run a standard MLIR pass pipeline on existing IR with intent-opt.

    ir is MLIR text, and pipeline uses standard MLIR pipeline syntax, such as
    ``builtin.module(canonicalize,cse)``. Pass prerequisites must be present in
    the input IR or established by earlier passes in the same pipeline. Every
    request executes the optimizer and archives its input, command, output and
    diagnostics. This neither generates a runtime nor launches a kernel.
    """
    if not isinstance(ir, str) or not ir.strip():
        raise ValueError("ir must contain nonempty MLIR text")
    if not isinstance(pipeline, str) or not pipeline.strip():
        raise ValueError("pipeline must contain a standard MLIR pass pipeline")
    return run_optimizer(optimizer, ir, pipeline)
