"""Shared implementation for the public command line and compile MCP."""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import MISSING, asdict, fields, is_dataclass
import importlib
import inspect
import io
from pathlib import Path

from .backends import BACKENDS, backend, make_target
from intent.targets.provider import provider as get_provider


def _files(directory: Path | None) -> dict[str, str]:
    if directory is None or not directory.is_dir():
        return {}
    names = {"input.mlir", "kernel.mlir", "kernel.source", "artifact.json", "request.json",
             "compiler.log", "kernel.mojo", "fp_environment.c", "native.json", "link.d"}
    suffixes = {".command", ".stdout", ".stderr", ".status", ".seconds"}
    return {path.name: str(path) for path in sorted(directory.iterdir())
            if path.is_file() and (path.name in names or path.suffix in suffixes)}


def _failure(result: dict, error: BaseException, stage: str) -> None:
    diagnostic = {"type": type(error).__name__, "message": str(error)}
    causes = []
    directories = []
    cause = error
    while cause is not None:
        detail = {"type": type(cause).__name__, "message": str(cause)}
        if getattr(cause, "stage", None) is not None:
            detail["stage"] = cause.stage
        if getattr(cause, "candidate", None) is not None:
            detail["candidate"] = cause.candidate
        directory = getattr(cause, "cache_directory", None)
        if directory is not None:
            detail["cache_directory"] = str(directory)
            if str(directory) not in directories:
                directories.append(str(directory))
        artifacts = getattr(cause, "artifacts", {})
        if artifacts:
            detail["files"] = {name: str(path) for name, path in artifacts.items()}
        causes.append(detail)
        for name in ("stdout", "stderr"):
            value = getattr(cause, name, None)
            if isinstance(value, str) and value:
                diagnostic.setdefault(name, value)
        frontend = getattr(cause, "diagnostic", None)
        if frontend is not None and is_dataclass(frontend):
            diagnostic["frontend"] = asdict(frontend)
        native = getattr(cause, "observation", None)
        if native is not None:
            result.setdefault("native_observation", asdict(native))
        cause = cause.__cause__
    diagnostic["causes"] = causes
    structured = next((item for item in causes if "stage" in item), None)
    if directories:
        # Keep earlier compiler output reachable when native materialization has
        # a different directory. A log directory is evidence, not a new cache.
        previous = result.get("cache_directory")
        if previous is not None and previous not in directories:
            directories.append(previous)
        result["artifact_directories"] = [
            {"path": path, "files": _files(Path(path))} for path in directories]
        result.update(cache_directory=directories[0], files=_files(Path(directories[0])))
    if causes[0].get("files"):
        result.setdefault("files", {}).update(causes[0]["files"])
    if structured is not None and "candidate" in structured:
        diagnostic["candidate"] = structured["candidate"]
    result.update(status="error", stage=structured["stage"] if structured is not None else stage,
                  diagnostic=diagnostic, tool_invoked_kernel=False)


def read_artifact(path: str, *, offset: int = 0, limit: int = 16000) -> dict:
    """Read an explicitly selected UTF-8 source, IR, metadata or log file.

    Offset and limit count Unicode characters. Continue at next_offset until eof;
    no module is imported and no compiler or kernel is invoked.
    """
    result = {"path": path, "tool_invoked_kernel": False}
    try:
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer character offset")
        if type(limit) is not int or not 1 <= limit <= 64000:
            raise ValueError("limit must be an integer from 1 through 64000 characters")
        selected = Path(path).expanduser().resolve(strict=True)
        if not selected.is_file():
            raise ValueError("path must name an existing text file")
        with selected.open(encoding="utf-8", newline="") as stream:
            remaining, line = offset, 1
            while remaining:
                skipped = stream.read(min(remaining, 65536))
                if not skipped:
                    raise ValueError("offset exceeds the file's character length")
                remaining -= len(skipped)
                line += skipped.count("\n")
            page = stream.read(limit + 1)
        text = page[:limit]
        result.update(status="read", path=str(selected), offset=offset, first_line=line,
                      text=text, next_offset=offset + len(text), eof=len(page) <= limit)
    except Exception as error:
        _failure(result, error, "artifact_reading")
    return result


def _declaration(value) -> dict:
    declaration = {"name": f"{value.__module__}.{value.__qualname__}",
                   "signature": str(inspect.signature(value)), "description": inspect.getdoc(value)}
    if is_dataclass(value):
        declaration["fields"] = []
        for field in fields(value):
            if not field.init:
                continue
            item = {"name": field.name, "type": str(field.type), "required": field.default is MISSING
                    and field.default_factory is MISSING}
            if field.default is not MISSING:
                item["default"] = field.default
            elif field.default_factory is not MISSING:
                item["default_factory"] = field.default_factory.__qualname__
            declaration["fields"].append(item)
    return declaration


def describe(target: str | None = None) -> dict:
    """Discover public call signatures and target fields without probing a device or SDK."""
    import intent

    result = {"target": target, "tool_invoked_kernel": False}
    try:
        names = (target,) if target is not None else tuple(BACKENDS)
        result["backends"] = {name: {**backend(name),
            "target": _declaration(get_provider(name).target)} for name in names}
        result["api"] = {name: _declaration(getattr(intent, name)) for name in (
            "compile", "generate", "compile_ir", "generate_from_ir", "optimize_ir", "CompileOptions",
            "GPUCompilationTarget", "GPUCapabilities", "CPUCompilationTarget", "DSACompilationTarget",
            "NativeObservation", "NativeResource", "CandidateObservation", "InvocationArgument")}
        result["tools"] = {name: _declaration(function) for name, function in (
            ("compile", compile_request), ("generate_from_ir", generate_ir_request),
            ("materialize", materialize_request), ("optimize", optimize_request),
            ("environment", doctor), ("read_artifact", read_artifact))}
        result.update(status="described", scope="Declarations only; compiler availability and device support are not checked")
    except Exception as error:
        _failure(result, error, "interface_discovery")
    return result


def _ir_text(ir_file: str) -> str:
    path = Path(ir_file).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("ir_file must name an existing MLIR file")
    return path.read_text(encoding="utf-8")


def _generated_result(result: dict, generated, materialize: bool, *,
                      runtime_target=None, export_directory: str | None = None) -> None:
    result.update(cache_directory=str(generated.cache_directory),
                  files=_files(generated.cache_directory), metadata=generated.metadata)
    if export_directory is not None:
        directory = generated.save(export_directory)
        result.update(export_directory=str(directory), exported_files=_files(directory))
    if materialize:
        generated.materialize(target=runtime_target)
    result.update(status="materialized" if materialize else "generated")


def compile_request(program: str | Path, kernel: str, target: str | None = None, *,
                    target_options: dict | None = None, constexprs: dict | None = None,
                    compiler: str | None = None, tuning_config: str | None = None,
                    materialize: bool = False, stage: str = "provider",
                    target_facts: dict | None = None, export_directory: str | None = None,
                    options: dict | None = None) -> dict:
    """Compile an existing definition without invoking its kernel.

    Loading the Python module executes its ordinary top-level host code. Native
    JIT/tuning may remain deferred after materialization, depending on provider.
    """
    from .requests import request
    selected_program = str(program.expanduser().absolute()) if isinstance(program, Path) else program
    return request("compile", dict(program=selected_program, kernel=kernel, target=target,
        target_options=target_options, constexprs=constexprs, compiler=compiler,
        tuning_config=tuning_config, materialize=materialize, stage=stage,
        target_facts=target_facts, export_directory=export_directory, options=options))


def _compile_request(program: str, kernel: str, target: str | None = None, *,
                     target_options: dict | None = None, constexprs: dict | None = None,
                     compiler: str | None = None, tuning_config: str | None = None,
                     materialize: bool = False, stage: str = "provider",
                     target_facts: dict | None = None, export_directory: str | None = None,
                     options: dict | None = None) -> dict:
    from .worker import load_definition
    import intent
    from intent.compiler.toolchain import CompilerStage

    current_stage = "request"
    transcript = io.StringIO()
    result = {"program": str(program), "kernel": kernel, "target": target, "requested_stage": stage}
    try:
        compile_options = None
        if options is not None:
            if not isinstance(options, dict):
                raise TypeError("options must be a compile options object")
            compile_options = intent.CompileOptions(**options)
            result["requested_compile_options"] = asdict(compile_options)
        selected_stage = CompilerStage(stage)
        if selected_stage is CompilerStage.KIR:
            if target is not None or target_options or target_facts is not None or tuning_config is not None or materialize:
                raise ValueError("KIR compilation does not accept target, target options, tuning or materialization")
        elif target is None:
            raise ValueError("shared/provider compilation requires a target")
        if materialize and selected_stage is not CompilerStage.PROVIDER:
            raise ValueError("materialization requires provider compilation")
        if export_directory is not None and selected_stage is not CompilerStage.PROVIDER:
            raise ValueError("export requires provider source and metadata")
        if target_facts is not None and target_options and not materialize:
            raise ValueError("runtime target options require materialization when explicit compiler facts are supplied")
        current_stage = "program_loading"
        # User module prints must not corrupt JSON output or the MCP transport.
        with redirect_stdout(transcript):
            definition = load_definition(program, kernel)
            result["definition"] = asdict(definition.source)
            selected = None
            if target is not None:
                current_stage = "target_options"
                selected = make_target(target, {} if target_facts is not None else dict(target_options or {}), facts=target_facts)
            current_stage = "compilation"
            if selected_stage is CompilerStage.PROVIDER:
                generated = intent.generate(definition, target=selected, compiler=compiler,
                                            constexprs=constexprs, tuning_config=tuning_config,
                                            options=compile_options)
                if materialize:
                    current_stage = "generated_source_materialization"
                runtime_target = make_target(target, dict(target_options or {})) if materialize and target_facts is not None else None
                _generated_result(result, generated, materialize,
                                  runtime_target=runtime_target, export_directory=export_directory)
            else:
                lowered = intent.compile_ir(definition, stage=selected_stage.value,
                                            target=selected, compiler=compiler,
                                            constexprs=constexprs, tuning_config=tuning_config,
                                            options=compile_options)
                result.update(status="lowered", cache_directory=str(lowered.cache_directory),
                              files=_files(lowered.cache_directory))
            result["tool_invoked_kernel"] = False
    except (Exception, SystemExit) as error:
        _failure(result, error, current_stage)
    if transcript.getvalue():
        result["program_stdout"] = transcript.getvalue()
    return result


def generate_ir_request(ir_file: str, name: str, target: str, *,
                        input_stage: str = "shared", target_options: dict | None = None,
                        compiler: str | None = None, materialize: bool = False,
                        target_facts: dict | None = None, export_directory: str | None = None) -> dict:
    """Resume compilation from explicit existing IR through the public API."""
    import intent

    result = {"ir_file": ir_file, "name": name, "target": target,
              "input_stage": input_stage, "tool_invoked_kernel": False}
    current_stage = "ir_loading"
    transcript = io.StringIO()
    try:
        text = _ir_text(ir_file)
        with redirect_stdout(transcript):
            current_stage = "target_options"
            if target_facts is not None and target_options and not materialize:
                raise ValueError("runtime target options require materialization when explicit compiler facts are supplied")
            selected = make_target(target, {} if target_facts is not None else dict(target_options or {}), facts=target_facts)
            current_stage = "compilation"
            generated = intent.generate_from_ir(text, input_stage=input_stage,
                name=name, target=selected, compiler=compiler)
            if materialize:
                current_stage = "generated_source_materialization"
            runtime_target = make_target(target, dict(target_options or {})) if materialize and target_facts is not None else None
            _generated_result(result, generated, materialize,
                              runtime_target=runtime_target, export_directory=export_directory)
    except (Exception, SystemExit) as error:
        _failure(result, error, current_stage)
    if transcript.getvalue():
        result["program_stdout"] = transcript.getvalue()
    return result


def materialize_request(directory: str, target: str, *, target_options: dict | None = None) -> dict:
    """Load a generated program and bind an explicitly selected local runtime; never launch."""
    from intent.compiler.artifact import GeneratedProgram

    result = {"program_directory": directory, "target": target, "tool_invoked_kernel": False}
    transcript = io.StringIO()
    current_stage = "generated_program_loading"
    try:
        program = GeneratedProgram.load(directory)
        current_stage = "target_options"
        selected = make_target(target, dict(target_options or {}))
        with redirect_stdout(transcript):
            current_stage = "generated_source_materialization"
            _generated_result(result, program, True, runtime_target=selected)
    except (Exception, SystemExit) as error:
        _failure(result, error, current_stage)
    if transcript.getvalue():
        result["program_stdout"] = transcript.getvalue()
    return result


def optimize_request(ir_file: str, pipeline: str, *, optimizer: str | None = None) -> dict:
    """Apply a standard pass pipeline to an explicitly selected existing IR file."""
    import intent

    result = {"ir_file": ir_file, "pipeline": pipeline, "tool_invoked_kernel": False}
    current_stage = "ir_loading"
    try:
        text = _ir_text(ir_file)
        current_stage = "ir_optimization"
        optimized = intent.optimize_ir(text, pipeline=pipeline, optimizer=optimizer)
        result.update(status="optimized", cache_directory=str(optimized.cache_directory),
                      files=_files(optimized.cache_directory))
    except (Exception, SystemExit) as error:
        _failure(result, error, current_stage)
    return result


def doctor(target: str | None = None, *, target_options: dict | None = None, compiler: str | None = None,
           target_facts: dict | None = None) -> dict:
    """Check the compiler alone, or one selected provider's dependencies and target.

    No target checks base compiler startup and the KIR stage. Explicit target
    facts check offline generation prerequisites; local targets resolve their
    actual SDK/device requirements. No kernel is compiled or executed.
    """
    from intent.compiler.toolchain import compiler_info
    from intent.targets.base import ResolvedTarget

    checks = []
    result = {"target": target, "scope": "Dependency and target resolution only; no kernel compilation or execution",
              "checks": checks, "tool_invoked_kernel": False}

    def check(name, category, action):
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                value = action()
        except Exception as error:
            failed = {"name": name, "category": category}
            _failure(failed, error, category)
            if output.getvalue():
                failed["stdout"] = output.getvalue()
            checks.append(failed)
            return None
        checks.append({"name": name, "category": category, "status": "available", "detail": value})
        if output.getvalue():
            checks[-1]["stdout"] = output.getvalue()
        return value

    def python_module(name):
        module = importlib.import_module(name)
        return {"path": getattr(module, "__file__", None), "version": str(getattr(module, "__version__", "not declared"))}

    def inspect_compiler():
        facts = compiler_info(compiler)
        if "kir" not in facts["stages"] or "ir" not in facts["outputs"]["kir"]:
            raise NotImplementedError("The selected compiler does not provide verified KIR output")
        return facts

    information = check("intent compiler", "compiler", inspect_compiler)
    if target is None:
        if target_options or target_facts is not None:
            def require_target():
                raise ValueError("target_options and target_facts require an explicitly selected target")
            check("target options", "target_options", require_target)
        result["scope"] = "Base compiler startup and KIR availability only; provider packages, SDKs and devices were not checked"
        result["status"] = "available" if all(item["status"] == "available" for item in checks) else "unavailable"
        return result
    description = check("backend selection", "target_options", lambda: backend(target))
    if description is None:
        result["status"] = "unavailable"
        return result
    result["toolchain"] = description["toolchain"]

    def inspect_provider():
        provider = information["providers"].get(target)
        if provider is None or not provider["available"]:
            available = [name for name, entry in information["providers"].items() if entry["available"]]
            raise NotImplementedError(
                f"The selected Intent compiler does not contain {target!r}; "
                f"built providers: {', '.join(available)}"
            )
        return {"name": target, **provider}

    if information is not None:
        check("compiled provider", "compiler_provider", inspect_provider)
    provider = information["providers"].get(target) if information is not None else None
    if provider is not None and provider["available"]:
        def profiles():
            paths = [Path(path) for path in provider["profiles"]]
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(f"Compiler profile is missing: {path}")
            return [str(path) for path in paths]
        check("compiler profiles", "compiler_resources", profiles)
    if target_facts is None:
        for name in description["modules"]:
            check(name, "python_package", lambda name=name: python_module(name))
    else:
        result["scope"] = "Compiler and explicit compiler facts only; provider SDK, runtime device and execution were not checked"
    resolved = None
    def resolve():
        nonlocal resolved
        resolved = make_target(target, dict(target_options or {}), facts=target_facts).resolve()
        result["callable_materialization"] = isinstance(resolved, ResolvedTarget)
        compilation = resolved.compilation if isinstance(resolved, ResolvedTarget) else resolved
        return {"compiler_options": list(compilation.compiler_options),
                "facts": asdict(resolved) if is_dataclass(resolved) else {}}

    check("target", "target_resolution", resolve)
    adapter = get_provider(target)
    if target_facts is None:
        if adapter.environment_scope is not None:
            result["scope"] = adapter.environment_scope
        for item in adapter.environment_checks(resolved):
            check(item.name, item.category, item.action)
    result["status"] = "available" if all(item["status"] == "available" for item in checks) else "unavailable"
    return result
