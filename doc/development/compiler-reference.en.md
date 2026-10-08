# Compiler development entry points

This page locates the public implementation along one compilation call. Read the [compiler specification](../compiler/README.md) for language and IR contracts, and [extending passes](passes.md) for new transformations. Source links point to current `main`; installed tools expose the actual APIs, available providers and options.

## From definition to execution

```text
@intent.kernel / @intent.fn
  → typed Python frontend
  → canonical KIR
  → GPU / CPU / DSA physical IR
  → family passes
  → provider legalization
  → provider source + interface metadata
  → native materialization
  → explicit runtime call
```

`intent.generate` returns a portable `GeneratedProgram` without launching a kernel; `intent.compile` combines generation and runtime binding. External JIT or tuning may remain deferred until the first actual call. Authors call multiple kernels explicitly from host code; this compilation path preserves each kernel's algorithm and interface.

| Stage | Current implementation | Scope |
|---|---|---|
| Kernel and helper declarations | [python/intent/api/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/api/) | Decorators, definitions and author interfaces |
| Typed frontend | [python/intent/frontend/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/frontend/) | Typed capture, scopes, control and intrinsic construction |
| Public compiler API | [compiler/pipeline.py](https://github.com/Tianchen-ai/Intent/blob/main/python/intent/compiler/pipeline.py) | `compile`, `generate`, `compile_ir`, `generate_from_ir` and stage diagnostics |
| Native driver | [lib/Compiler/](https://github.com/Tianchen-ai/Intent/tree/main/lib/Compiler/) | Requests, target selection and family/provider pipelines; CLI parsing and file I/O live in [tools/intent-compile/](https://github.com/Tianchen-ai/Intent/tree/main/tools/intent-compile/) |
| KIR normalization | [lib/Transforms/](https://github.com/Tianchen-ai/Intent/tree/main/lib/Transforms/) | Cross-target normalization of canonical KIR |
| Execution-model construction | [lib/Conversion/](https://github.com/Tianchen-ai/Intent/tree/main/lib/Conversion/) | KIR to GPU, CPU or DSA physical programs |
| Family analyses and transformations | [lib/Dialect/](https://github.com/Tianchen-ai/Intent/tree/main/lib/Dialect/) | Each execution model's typed IR, analyses, transformations and verifiers |
| Provider implementation | [lib/Target/](https://github.com/Tianchen-ai/Intent/tree/main/lib/Target/) | Target legalization, native forms and serialization |
| Artifact persistence | [compiler/artifact.py](https://github.com/Tianchen-ai/Intent/blob/main/python/intent/compiler/artifact.py) | Source, IR, metadata and `save/load/materialize` |
| Targets and actual calls | [python/intent/targets/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/targets/), [python/intent/runtime/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/runtime/) | Capabilities, environment binding, ABI, JIT, tuning and launch |
| CLI and MCP | [python/intent/tools/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/tools/), [python/intent/mcp/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/mcp/) | Command-line and MCP access to the same public compiler API |

## Locate reusable components

The frontend builds typed values, operations, blocks and regions in `frontend/mlir/`, then serializes once. Helpers, control merges and structured intrinsics reuse scope, product, shape and region construction in `frontend/lowering/`. Intrinsic families live in `frontend/lowering/intrinsics/`. New rules consume the existing context rather than keeping a second type or SSA-visibility system.

Semantic analyses live in `include/Intent/Analysis/` and `lib/Analysis/`; physical relations live in the corresponding dialect's `Analysis/`. GPU, CPU and DSA have distinct execution models. Reuse applicable semantic proofs while keeping physical transformations in each family's `Transforms/`. Responsibility directories such as `Reduction/`, `Storage/` and `Task/` group closely related implementations at the same level.

A provider's `Transforms/` selects and legalizes native forms; `Serialization/` spells the already formed program. Provider-native reductions, scans, layouts and instruction lowering stay with the provider when their contracts match. Public headers live under matching `include/Intent/` paths; local implementation helpers may remain adjacent to their source.

## Discover interfaces and diagnose failures

Query current declarations and dependencies in the installed environment:

```bash
intent describe --target triton --json
intent doctor --target triton --json
```

`describe` discovers interfaces; `doctor` checks the actual toolchain and target environment. Neither command compiles or executes a kernel.

These commands use an existing complete algorithm and stop at different stages:

```bash
python -m examples.use softmax --target triton --stage source
python -m examples.use softmax --target triton --stage native
python -m examples.use softmax --target triton
```

The complete-program entry prepares its existing inputs and runs its existing host composition. KIR needs no target or device; shared IR needs an explicit compilation target; native compilation and execution need the corresponding toolchain and device. See [compile and run](../getting-started/usage.md) for commands selecting one definition and inspecting KIR.

On failure, use the returned `stage`, diagnostic and `cache_directory` to find actual input, IR, source and logs. Locate the failure in the frontend, physical construction, family pass, provider legalization, external compiler or runtime. Fix that boundary rather than guessing missing semantics in a later layer.

## Configuration, numerics and performance

Empirical configuration tables live in each provider's `Transforms/` or `Transforms/Configuration/`. They declare complete correlated rows; runtime tuning evaluates those rows. [Configuration and tuning](../getting-started/configuration.md) explains the format, overrides and first-call behavior.

`CompileOptions` and target parameters are distinct interfaces. The former chooses defined numerical permissions and optional optimizations; the latter provides capabilities and compilation facts. Enabling an optimization does not grant extra numerical changes. Read [types, numerics and effects](../dsl/types-numerics-and-effects.md) for the contract.

A performance contribution explains the consumed facts, legality, actual IR rewrite and expected structural benefit. Verify affected complete programs and record source generation, native compilation, execution, numerical comparisons and warm-call time separately. Do not mix JIT, tuning or input preparation into a kernel warm-call claim. See the [contribution guide](https://github.com/Tianchen-ai/Intent/blob/main/CONTRIBUTING.md).
