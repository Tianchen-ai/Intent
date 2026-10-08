from pathlib import Path
from types import SimpleNamespace

from ..artifact import CompiledArtifact
from ..gpu.expressions import evaluate_shape
from ..gpu.program import LaunchResult, materialize_gpu_program
from ..tuning import TuningState
from ..diagnostics import CacheObservation, CandidateObservation, bindings, observation, unavailable_resources
from intent.compiler.toolchain import CompilationStageError


def bind_array_view(view, group_ends: tuple[int, ...]):
    shape, strides = [], []
    begin = 0
    for end in group_ends:
        extent = 1
        for axis in range(begin, end):
            if view.shape[axis] <= 0:
                return view, False
            if axis > begin and (view.stride(axis) <= 0 or
                                 view.stride(axis - 1) != view.shape[axis] * view.stride(axis)):
                return view, False
            extent *= view.shape[axis]
        shape.append(extent)
        strides.append(view.stride(end - 1))
        begin = end
    return view.as_strided(shape, strides), True


def array_index_kernels(function, view_names: tuple[str, ...]):
    from types import FunctionType
    from typing import Annotated, get_args
    import cuda.tile as ct

    narrow = FunctionType(function.__code__, function.__globals__, function.__name__,
                          function.__defaults__, function.__closure__)
    narrow.__qualname__ = function.__qualname__
    narrow.__annotations__ = dict(function.__annotations__)
    for name in view_names:
        array = get_args(function.__annotations__[name])[1]
        narrow.__annotations__[name] = Annotated[
            ct.Array, ct.ArrayAnnotation(static_shape_dims=array.static_shape_dims,
                                        static_stride_dims=array.static_stride_dims)
        ]
    return ct.kernel(narrow), ct.kernel(function)


def can_use_i32_array_indices(views: tuple, tile_bounds: tuple) -> bool:
    limit = (1 << 31) - 1
    for view, tiles in zip(views, tile_bounds, strict=True):
        span = 0
        padded_elements = 1
        for extent, stride, tile in zip(view.shape, view.stride(), tiles, strict=True):
            if extent <= 0 or tile <= 0 or stride < 0 or stride > limit:
                return False
            # Native access origins are in the view; include the physical tail.
            padded_extent = extent + tile - 1
            if padded_extent > limit:
                return False
            span += (padded_extent - 1) * stride
            padded_elements *= padded_extent
        if (span + 1) * view.element_size() > limit or padded_elements > limit:
            return False
    return True


def _search(*arguments, **keywords):
    from cuda.tile.tune import exhaustive_search

    return exhaustive_search(*arguments, **keywords)


class CuTileProgram:
    def __init__(self, interface, namespace: dict, facts, target: dict, *, source: str) -> None:
        from .compilation import CuTileCompilation

        self.interface = interface
        self.configurations = interface.configuration_space
        self.facts = facts
        self.target = bindings(target)
        self.compilation = CuTileCompilation(source, {
            name: namespace[name] for name in (facts.kernel, facts.narrow_kernel) if name is not None})
        self.kernel = self.compilation.kernel(facts.kernel)
        self.narrow_kernel = None if facts.narrow_kernel is None else self.compilation.kernel(facts.narrow_kernel)
        self.tile_bounds = facts.index_tile_bounds
        self.search = _search
        self.observe_tuning = None
        # Use the SDK worker to bound each candidate's first warmup probe.
        self.tuning_options: dict = {"single_run_timeout_sec": 30.0}
        self._winners: dict = {}

    def _array_values(self, values: dict) -> None:
        for entry in self.facts.array_views:
            values[entry.name], values[entry.eligible] = bind_array_view(
                values[entry.base], entry.group_ends)

    def _hints(self, config) -> dict:
        hints = {hint: getattr(config, parameter) for hint, parameter in self.facts.compiler_hints}
        if hints.get("num_worker_warps") == self.facts.inferred_worker_warps:
            hints["num_worker_warps"] = None
        return hints

    def _arguments(self, values: dict, config) -> tuple:
        bindings = {**values, **vars(config)}
        return tuple(self.interface.native_value(name, bindings) for name in self.facts.kernel_arguments)

    def _grid(self, values: dict, config) -> tuple:
        return (*evaluate_shape(self.interface.grid, {**values, **vars(config)}), 1, 1)

    def _key(self, invocation, values: dict) -> tuple:
        return (tuple((tuple(view.shape), tuple(view.stride()), view.dtype, view.device)
                      for view in invocation.views),
                tuple(self.interface.native_value(name, values) for name in self.facts.tuning_key_scalars),
                tuple(values[name] for name in self.configurations.coverage_names),
                tuple(values[entry.eligible] for entry in self.facts.array_views),
                tuple(values[entry.name] for entry in self.interface.overlaps))

    def _configurations(self, values):
        kernel = self._kernel(values)
        selected, keys = [], []
        for row in self.configurations.candidates(values):
            config = SimpleNamespace(**self.configurations.bound_configuration(values, row))
            variant = kernel.replace_hints(**self._hints(config))
            key = (self.compilation.request_key(variant, self._arguments(values, config)),
                   self._grid(values, config))
            if key not in keys:
                keys.append(key)
                selected.append(config)
        return tuple(selected)

    def _kernel(self, values):
        if self.tile_bounds is not None:
            arrays, bounds = [], []
            for entry in self.tile_bounds:
                if entry.eligible is not None and not values[entry.eligible]:
                    continue
                arrays.append(self.interface.native_value(entry.array, values))
                bounds.append(evaluate_shape(entry.bounds, values))
            if can_use_i32_array_indices(tuple(arrays), tuple(bounds)):
                return self.narrow_kernel
        return self.kernel

    def compile(self, invocation):
        import cuda.tile as ct

        values = dict(invocation.values)
        self._array_values(values)
        records = ()

        def details(stage):
            return observation("cutile", self.target, invocation.description, None, _native_resources(),
                               records, stage=stage)

        try:
            configurations = self._configurations(values)
            kernel = self._kernel(values)
            failures = self.compilation.compile(tuple(
                (kernel.replace_hints(**self._hints(config)), self._arguments(values, config))
                for config in configurations))
            records = tuple(CandidateObservation(
                bindings(self.configurations.bound_configuration(invocation.values, vars(config))),
                "compiled" if error is None else "failed", "provider_native_compilation",
                None if error is None else error[0], None if error is None else error[1])
                for config, error in zip(configurations, failures, strict=True))
            if all(error is not None for error in failures):
                kind, message = failures[0]
                cause = ct.TileError(f"{kind}: {message}")
                raise CompilationStageError("provider_native_compilation",
                    f"No cuTile candidate compiled. Configuration: {vars(configurations[0])}\n{cause}",
                    observation=details("failed")) from cause
        except CompilationStageError as error:
            if error.observation is None:
                error.observation = details("failed")
            raise
        except Exception as error:
            raise CompilationStageError("provider_native_compilation", str(error),
                                        observation=details("failed")) from error
        return details("compiled")

    def launch(self, invocation) -> LaunchResult:
        import torch
        import cuda.tile as ct

        values = dict(invocation.values)
        self._array_values(values)
        key = self._key(invocation, values)
        cached = self._winners.get(key)
        reused = cached is not None
        candidates = ()
        if cached is None:
            self.compile(invocation)
            try:
                configurations = self._configurations(values)
            except CompilationStageError as error:
                error.observation = observation("cutile", self.target, invocation.description, None, _native_resources(),
                                                stage="failed", history_unavailable="Candidate binding failed before native tuning")
                raise
            kernel = self._kernel(values)
            state = TuningState(invocation.public_views,
                                tuple(view.writable for view in self.interface.public_views))
            trial_values = dict(values)
            trial_values.update((view.id, trial) for view, trial in
                                zip(self.interface.public_views, state.views, strict=True))
            self._array_values(trial_values)
            hints = (self._hints,) if self.facts.compiler_hints else ()
            try:
                result = self.search(configurations, torch.cuda.current_stream(),
                                      lambda config: self._grid(values, config), kernel,
                                      lambda config: state.arguments(self._arguments(trial_values, config)),
                                      *hints, quiet=True, **self.tuning_options)
            except Exception as error:
                details = observation("cutile", self.target, invocation.description, None, _native_resources(),
                    stage="failed", history_unavailable=
                    "cuTile did not return a TuningResult; its exception is preserved, but no complete candidate history is available")
                raise CompilationStageError("provider_tuning", str(error), observation=details) from error
            if self.observe_tuning is not None:
                self.observe_tuning(configurations, result)
            config = result.best.config
            selected = kernel.replace_hints(**self._hints(config)) if hints else kernel
            candidates = tuple(
                [CandidateObservation(bindings(self.configurations.bound_configuration(invocation.values, vars(item.config))),
                                      "trial_completed", "provider_tuning",
                                      elapsed_ms=item.mean_us * 1e-3)
                 for item in result.successes] +
                [CandidateObservation(bindings(self.configurations.bound_configuration(invocation.values, vars(config))),
                                      "failed", "provider_tuning",
                                      error if isinstance(error, str) else error.__name__, message)
                 for config, error, message in result.failures])
            cached = config, selected
            self._winners[key] = cached
        config, selected = cached
        grid = self._grid(values, config)
        arguments = self._arguments(values, config)

        def invoke():
            try:
                return ct.launch(torch.cuda.current_stream(), grid, selected, arguments)
            except Exception as error:
                details = observation("cutile", self.target, invocation.description,
                                      bindings(self.configurations.bound_configuration(invocation.values, vars(config))), _native_resources(),
                                      candidates, stage="failed")
                raise CompilationStageError("provider_invocation", str(error), observation=details) from error

        invoke()
        details = observation(
            "cutile", self.target, invocation.description,
            bindings(self.configurations.bound_configuration(invocation.values, vars(config))), _native_resources(), candidates,
            caches=(CacheObservation("tuning", "runtime_instance", reused, "CuTileProgram._winners", "selection"),
                    CacheObservation("native_compilation", "sdk", None, "cuTile dispatcher",
                                     "native_dispatch",
                                     "The dispatcher does not report native compilation cache hits")),
            history_unavailable="The existing tuning winner was reused; no candidates were retried in this invocation" if reused else None)
        return LaunchResult(invoke, selected, details)

    def tuning_configurations(self, invocation):
        return self.configurations.enumerate(invocation.values,
                                             self.configurations.select(self.inspect_configurations(invocation)))

    def inspect_configurations(self, invocation):
        return self.configurations.inspect(invocation.values)

def _native_resources():
    return unavailable_resources("cuda.tile.CompilationResult",
        "The selected cuTile dispatcher exposes no native register/shared-memory resource fields")


def materialize_cutile_artifact(
    source: str,
    module_text: str,
    entry_name: str,
    device: int,
    contract,
    *,
    source_path: Path | None = None,
) -> CompiledArtifact:
    return materialize_gpu_program(
        provider_name="cutile", provider_type=lambda *args: CuTileProgram(*args, source=source), contract=contract,
        source=source,
        source_path=source_path,
        module_text=module_text,
        entry_name=entry_name,
        device=device,
        backend_ir_collector=None,
    )
