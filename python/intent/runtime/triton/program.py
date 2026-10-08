from contextvars import copy_context
from dataclasses import replace
from math import isfinite
from pathlib import Path

from ..artifact import CompiledArtifact
from ..gpu.expressions import evaluate_shape
from ..gpu.program import LaunchResult, materialize_gpu_program
from ..tuning import TuningState
from ..diagnostics import CacheObservation, CandidateObservation, CandidateRecorder, bindings, observation, resource, unavailable_resources
from intent.compiler.toolchain import CompilationStageError


class TuningHooks:
    def __init__(self, names: tuple[str, ...], writable: tuple[bool, ...],
                 readable: tuple[bool, ...]):
        self.names = names
        self.writable = writable
        self.readable = readable
        self.state = None
        self.observe_trial = None
        self._observed_arguments = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        try:
            self.finish()
        finally:
            self._observed_arguments = None

    def finish(self) -> None:
        if self.state is None:
            return
        from torch.utils._python_dispatch import _disable_current_modes

        # Cleanup must also run when measurement fails outside Triton's hooks.
        try:
            with _disable_current_modes():
                self.state.restore()
        finally:
            self.state = None

    def before(self, arguments: dict, reset_only: bool = False) -> None:
        if reset_only:
            self.finish()
            return
        if self.state is None:
            self.state = TuningState(tuple(arguments[name] for name in self.names),
                                     self.writable)
        self.state.restore(self.readable)

    def after(self, arguments: dict, exception: Exception | None) -> None:
        try:
            # Triton reuses full_nargs through the warmup/measurement calls of a
            # candidate. Capture the first completed trial and any failure, not
            # one diagnostic record allocation on every timed callback.
            if self.observe_trial is not None and (
                exception is not None or arguments is not self._observed_arguments
            ):
                self._observed_arguments = arguments
                self.observe_trial(arguments, exception)
        finally:
            if exception is not None:
                self.finish()
            else:
                self.state.restore(self.readable)


def _descriptor_allocator(size, alignment, stream):
    import torch

    buffer = torch.empty(size, dtype=torch.int8, device="cuda")
    if buffer.data_ptr() % alignment:
        raise RuntimeError("Triton descriptor allocator returned a misaligned buffer")
    return buffer


class TritonProgram:
    def __init__(self, interface, namespace: dict, facts, target: dict) -> None:
        import triton

        self.interface = interface
        self.configurations = interface.configuration_space
        self.facts = facts
        self.target = bindings(target)
        self.descriptors = facts.descriptors
        self.hooks = TuningHooks(tuple(view.kernel_name for view in interface.public_views),
                                 tuple(view.writable for view in interface.public_views),
                                 tuple(view.access != 1 for view in interface.public_views))
        kernel_parameters = tuple(name for name in facts.kernel_parameters
                                  if name not in self.configurations.coverage_names)
        native_options = dict(facts.native_options)
        configs = [triton.Config(
            {name: row[name] for name in kernel_parameters},
            num_warps=row[native_options["num_warps"]],
            num_stages=row[native_options["num_stages"]],
            num_ctas=row[native_options["num_ctas"]],
        ) for row in self.configurations.rows]
        self._config_rows = {id(config): row for config, row in
                             zip(configs, self.configurations.rows, strict=True)}
        kernel = namespace[facts.kernel]
        self._native_signature = kernel.signature
        # Native JIT invocation also covers the autotuner's one-config and
        # cached-winner paths, which do not call early_config_prune.
        kernel.add_pre_run_hook(self._validate_native_configuration)
        if self.descriptors:
            kernel = triton.heuristics({entry.name: self._descriptor_hook(entry)
                                        for entry in self.descriptors})(kernel)
        options = {}
        if facts.descriptor_choice is not None or self.configurations.requirements:
            options["prune_configs_by"] = {"early_config_prune": self._prune}
        kernel = triton.autotune(configs=configs, key=facts.autotune_key,
                                pre_hook=self.hooks.before, post_hook=self.hooks.after,
                                **options)(kernel)
        self.kernel = kernel

    def _trial_configuration(self, arguments):
        row = {name: arguments[name] for name in self.facts.kernel_parameters
               if name not in self.configurations.coverage_names}
        row.update((parameter, arguments[option]) for option, parameter in self.facts.native_options)
        return row

    def _context(self, arguments: dict) -> dict:
        values = self.interface.callback_values(arguments)
        packed = self.facts.metadata_argument
        if packed is not None:
            for entry, value in zip(self.interface.metadata, values[packed], strict=True):
                values[entry.id] = value
        return values

    def _descriptor_hook(self, entry):
        def bind(arguments):
            return self._descriptor(entry, self._context(arguments))
        return bind

    def _descriptor(self, entry, values):
        from triton.tools.tensor_descriptor import TensorDescriptor

        if not values[self.facts.descriptor_choice.config]:
            return values[entry.base]
        return TensorDescriptor(values[entry.base],
                                shape=list(evaluate_shape(entry.shape, values)),
                                strides=list(evaluate_shape(entry.strides, values)),
                                block_shape=list(evaluate_shape(entry.block_shape, values)),
                                padding=entry.padding)

    @staticmethod
    def _eligible(entry, values) -> bool:
        tensor = values[entry.base]
        shape = evaluate_shape(entry.shape, values)
        strides = evaluate_shape(entry.strides, values)
        alignment = entry.alignment
        return (tensor.ndim == entry.rank and tensor.data_ptr() % alignment == 0
                and (not entry.require_positive_shape or all(extent > 0 for extent in shape))
                and all(extent <= entry.maximum_shape_extent for extent in shape)
                and (not entry.require_positive_strides or all(stride > 0 for stride in strides))
                and strides[-1] == 1
                and all(tensor.stride(axis) == 1 for axis in entry.unit_stride_axes)
                and all(tensor.stride(axis) * tensor.element_size() % alignment == 0
                        for axis in entry.aligned_stride_axes))

    def _assess_rows(self, values, rows=None):
        assessed = self.configurations.inspect(values, rows=rows)
        choice = self.facts.descriptor_choice
        if choice is not None and not values[choice.eligibility]:
            assessed = tuple(replace(entry, provider_reason=
                "Triton descriptors require the declared view alignment, shape and strides")
                if dict(entry.configuration)[choice.config] else entry for entry in assessed)
        return assessed

    def _eligible_rows(self, values, rows=None):
        return self.configurations.select(self._assess_rows(values, rows))

    def _prune(self, configs, named_args, **kwargs):
        values = self._context({**named_args, **kwargs})
        rows = tuple(self._config_rows[id(config)] for config in configs)
        retained = self._eligible_rows(values, rows)
        return [config for config, row in zip(configs, rows, strict=True) if row in retained]

    def _validate_native_configuration(self, *arguments, **keywords):
        named = dict(zip(self.facts.kernel_arguments, arguments, strict=True))
        named.update(keywords)
        row = self._trial_configuration(named)
        self._eligible_rows(self._context(named), (row,))

    def _arguments(self, invocation):
        values = dict(invocation.values)
        packed = self.facts.metadata_argument
        if packed is not None:
            values[packed] = tuple(values[entry.id] for entry in self.interface.metadata)
        packed = self.facts.overlap_argument
        if packed is not None:
            values[packed] = tuple(values[entry.name] for entry in self.interface.overlaps)
        choice = self.facts.descriptor_choice
        if choice is not None:
            values[choice.eligibility] = all(self._eligible(entry, values) for entry in self.descriptors)
        arguments = tuple(self.interface.native_value(name, values) for name in self.facts.kernel_arguments)
        coverage = {name: values[name] for name in self.configurations.coverage_names}
        return values, arguments, coverage

    def _bound_launch(self, compiled, arguments, coverage, configuration, grid):
        named = dict(zip(self.facts.kernel_arguments, arguments, strict=True))
        named.update(coverage)
        named.update(configuration.all_kwargs())
        if self.descriptors:
            context = self._context(named)
            for entry in self.descriptors:
                named[entry.name] = self._descriptor(entry, context)
        bound = self._native_signature.bind(**{
            name: named[name] for name in self._native_signature.parameters if name in named
        })
        bound.apply_defaults()
        native_arguments = tuple(bound.arguments.values())
        selected_grid = tuple(grid(bound.arguments))
        native_launch = compiled[selected_grid + (1,) * (3 - len(selected_grid))]

        def execute():
            native_launch(*native_arguments)

        return execute

    def compile(self, invocation):
        import triton
        from triton.compiler.errors import CompileTimeAssertionFailure
        from triton.runtime.errors import OutOfResources, PTXASError

        values, arguments, coverage = self._arguments(invocation)
        records = []
        last_error = None

        def details(stage):
            return observation("triton", self.target, invocation.description, None,
                unavailable_resources("triton", "Compilation did not select or load an execution winner"),
                records, stage=stage,
                caches=(CacheObservation("native_compilation", "sdk", None, "Triton JIT warmup",
                                         "native_compilation", "The SDK does not expose this cache decision"),))

        try:
            rows = tuple(self._trial_configuration(config.all_kwargs()) for config in self.kernel.configs)
            eligible = self._eligible_rows(values, rows)
            for config, row in zip(self.kernel.configs, rows, strict=True):
                if row not in eligible:
                    continue
                configuration = bindings(self.configurations.bound_configuration(invocation.values, row))

                def warmup():
                    if self.facts.allocator:
                        triton.set_allocator(_descriptor_allocator)
                    return self.kernel.fn.warmup(
                        *arguments, grid=evaluate_shape(self.interface.grid, {**values, **row}),
                        **coverage, **config.all_kwargs(), enable_fp_fusion=True, enable_reflect_ftz=False)

                try:
                    copy_context().run(warmup) if self.facts.allocator else warmup()
                except (OutOfResources, CompileTimeAssertionFailure, PTXASError) as error:
                    # Match the SDK autotuner's candidate rejection boundary.
                    records.append(CandidateObservation(configuration, "failed", "provider_native_compilation",
                                                        type(error).__name__, str(error)))
                    last_error = error
                    continue
                except Exception as error:
                    records.append(CandidateObservation(configuration, "failed", "provider_native_compilation",
                                                        type(error).__name__, str(error)))
                    raise
                records.append(CandidateObservation(configuration, "compiled", "provider_native_compilation"))
            if not any(record.status == "compiled" for record in records):
                raise CompilationStageError("provider_native_compilation",
                    f"No Triton candidate compiled. Configuration: {dict(records[-1].configuration)}\n{last_error}",
                    observation=details("failed")) from last_error
        except CompilationStageError as error:
            if error.observation is None:
                error.observation = details("failed")
            raise
        except Exception as error:
            raise CompilationStageError("provider_native_compilation", str(error),
                                        observation=details("failed")) from error
        return details("compiled")

    def launch(self, invocation) -> LaunchResult:
        import triton

        values, arguments, coverage = self._arguments(invocation)
        recorder = CandidateRecorder()

        def trial(arguments, error):
            recorder.record(self._trial_configuration(arguments),
                            "failed" if error is not None else "trial_completed", "provider_tuning", error)

        def candidate_history():
            return tuple(replace(item, configuration=bindings(self.configurations.bound_configuration(
                invocation.values, dict(item.configuration)))) for item in recorder.snapshot())

        def grid(config):
            return evaluate_shape(self.interface.grid, {**values, **config})

        def execute():
            if self.facts.allocator:
                triton.set_allocator(_descriptor_allocator)
            with self.hooks:
                return self.kernel[grid](*arguments, **coverage,
                                         enable_fp_fusion=True, enable_reflect_ftz=False)

        def invoke():
            previous = self.hooks.observe_trial
            self.hooks.observe_trial = trial
            try:
                # Triton's public allocator setter binds a ContextVar. Keep the
                # descriptor's allocator local to this invocation, including
                # its native JIT/tuning, without replacing the caller's binding.
                return copy_context().run(execute) if self.facts.allocator else execute()
            except CompilationStageError as error:
                if error.observation is None:
                    error.observation = observation("triton", self.target, invocation.description, None,
                        unavailable_resources("triton", "No loaded kernel was returned by the failed invocation"),
                        candidate_history(), stage="failed")
                raise
            except Exception as error:
                details = observation("triton", self.target, invocation.description, None,
                    unavailable_resources("triton", "The invocation did not return a loaded native kernel"),
                    candidate_history(), stage="failed")
                raise CompilationStageError("provider_invocation", str(error), observation=details) from error
            finally:
                self.hooks.observe_trial = previous

        compiled = invoke()
        # The native pre-run hook has checked this complete binding before
        # execution, including winners reconstructed by Triton's disk cache.
        selected = self.kernel.best_config
        config = self._trial_configuration(selected.all_kwargs())
        history = recorder.snapshot()
        if history:
            # Triton's autotuner returns the median in milliseconds at index 0
            # for its requested (0.5, 0.2, 0.8) quantiles. Do not read an older
            # search's timings when this invocation did not run trial callbacks.
            for candidate, samples in self.kernel.configs_timings.items():
                elapsed = samples[0] if isinstance(samples, (tuple, list)) else samples
                if isinstance(elapsed, (int, float)) and isfinite(elapsed):
                    recorder.timing(self._trial_configuration(candidate.all_kwargs()), float(elapsed))
        details = observation("triton", self.target, invocation.description,
                              bindings(self.configurations.bound_configuration(invocation.values, config)), _native_resources(compiled),
                              candidate_history(), history_unavailable=None if history else
                              "No tuning trial callbacks occurred in this invocation; only the selected kernel was observed",
                              caches=(CacheObservation("tuning", "provider", False if history else None,
                                  "Triton autotuner trial callbacks", "selection",
                                  None if history else "No trial callbacks does not distinguish a cached winner from a single configuration"),
                                      CacheObservation("native_compilation", "sdk", None, "Triton JIT",
                                                       "native_dispatch",
                                                       "The returned compiled kernel does not identify a native compilation cache hit")))
        # A prepared invocation retains its arguments and metadata. Bind the
        # selected SDK kernel once; fresh invocations still pass through tuning
        # and the native hook, including single-config and cached-winner paths.
        try:
            native_launch = self._bound_launch(
                compiled, arguments, coverage, selected, grid)
        except Exception as error:
            raise CompilationStageError("provider_invocation", str(error),
                observation=replace(details, stage="failed")) from error

        def execute_bound():
            if self.facts.allocator:
                triton.set_allocator(_descriptor_allocator)
            native_launch()

        def replay():
            try:
                if self.facts.allocator:
                    copy_context().run(execute_bound)
                else:
                    execute_bound()
            except CompilationStageError as error:
                if error.observation is None:
                    error.observation = replace(details, stage="failed")
                raise
            except Exception as error:
                raise CompilationStageError("provider_invocation", str(error),
                    observation=replace(details, stage="failed")) from error

        return LaunchResult(replay, compiled, details)

    def tuning_configurations(self, invocation):
        return self.configurations.enumerate(invocation.values,
                                             self.configurations.select(self.inspect_configurations(invocation)))

    def inspect_configurations(self, invocation):
        values = dict(invocation.values)
        choice = self.facts.descriptor_choice
        if choice is not None:
            values[choice.eligibility] = all(self._eligible(entry, values) for entry in self.descriptors)
        return self._assess_rows(values)

def _collect_triton_ir(compiled_kernel: object) -> dict[str, str]:
    asm = getattr(compiled_kernel, "asm", None)
    if not isinstance(asm, dict):
        raise RuntimeError("Triton launch did not return a compiled kernel artifact")
    result = {name: value for name, value in asm.items() if isinstance(value, str)}
    if not result:
        raise RuntimeError("Triton compiled artifact exposes no textual backend IR")
    return result


def _native_resources(kernel):
    metadata = kernel.metadata
    fields = (
        ("registers_per_thread", getattr(kernel, "n_regs", None), "registers", "CompiledKernel.n_regs", "native_loading"),
        # NVIDIA load_binary reports CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES / 4.
        ("local_memory_words_per_thread", getattr(kernel, "n_spills", None), "32-bit words",
         "CompiledKernel.n_spills (CUDA LOCAL_SIZE_BYTES / 4)", "native_loading"),
        ("shared_memory_bytes", getattr(metadata, "shared", None), "bytes", "CompiledKernel.metadata.shared", "native_compilation"),
        ("max_threads_per_block", getattr(kernel, "n_max_threads", None), "threads", "CompiledKernel.n_max_threads", "native_loading"),
    )
    return tuple(resource(name, value, unit, "triton." + source, stage,
                          unavailable="The selected Triton compiled kernel does not expose this field")
                 for name, value, unit, source, stage in fields)


def materialize_triton_artifact(
    source: str,
    module_text: str,
    entry_name: str,
    device: int,
    contract,
    *,
    source_path: Path | None = None,
) -> CompiledArtifact:
    return materialize_gpu_program(
        provider_name="triton", provider_type=TritonProgram, contract=contract,
        source=source,
        module_text=module_text,
        entry_name=entry_name,
        device=device,
        backend_ir_collector=_collect_triton_ir,
        source_path=source_path,
    )
