# Compiler development entry points

This page locates the public implementation along one compilation call. Read the [compiler specification](../compiler/README.md) for language and IR contracts, and [extending passes](passes.md) for new transformations. Source links point to current `main`; installed tools expose the actual APIs, available providers and options.

## From definition to execution

```text
@intent.kernel / @intent.fn
  → typed Python frontend
  → canonical KIR
  → Block / CPU task program
  → model passes / target storage realization
  → provider legalization
  → provider source + interface metadata
  → native materialization
  → explicit runtime call
```

Ordinary calls use `intent.jit(definition, backend=...)`. Actual inputs select the device and the provider discovers local facts. `prepare` binds arguments and generates necessary source; `compile` compiles native candidates; `launch` executes, including candidate selection when needed. `intent.generate` and `intent.compile` remain explicit-target artifact APIs. Authors compose kernels in host code; compilation preserves each kernel's algorithm and interface.

| Stage | Current implementation | Scope |
|---|---|---|
| Kernel and helper declarations | [python/intent/api/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/api/) | Decorators, definitions and author interfaces |
| Typed frontend | [python/intent/frontend/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/frontend/) | Typed capture, scopes, control and intrinsic construction |
| Public compiler API | [compiler/pipeline.py](https://github.com/Tianchen-ai/Intent/blob/main/python/intent/compiler/pipeline.py) | `compile`, `generate`, `compile_ir`, `generate_from_ir` and stage diagnostics |
| Native driver | [lib/Compiler/](https://github.com/Tianchen-ai/Intent/tree/main/lib/Compiler/) | Requests, target selection and family/provider pipelines; CLI parsing and file I/O live in [tools/intent-compile/](https://github.com/Tianchen-ai/Intent/tree/main/tools/intent-compile/) |
| KIR normalization | [lib/Transforms/](https://github.com/Tianchen-ai/Intent/tree/main/lib/Transforms/) | Cross-target normalization of canonical KIR |
| Execution-model construction | [lib/Conversion/](https://github.com/Tianchen-ai/Intent/tree/main/lib/Conversion/) | KIR to Block or CPU task; Block-to-BANG-C storage realization belongs to the target component |
| Family analyses and transformations | [lib/Dialect/](https://github.com/Tianchen-ai/Intent/tree/main/lib/Dialect/) | Each execution model's typed IR, analyses, transformations and verifiers |
| Provider implementation | [lib/Target/](https://github.com/Tianchen-ai/Intent/tree/main/lib/Target/) | Target legalization, native forms and serialization |
| Artifact persistence | [compiler/artifact.py](https://github.com/Tianchen-ai/Intent/blob/main/python/intent/compiler/artifact.py) | Source, IR, metadata and `save/load/materialize` |
| Targets and actual calls | [python/intent/targets/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/targets/), [python/intent/runtime/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/runtime/) | Capabilities, environment binding, ABI, JIT, tuning and launch |
| CLI and MCP | [python/intent/tools/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/tools/), [python/intent/mcp/](https://github.com/Tianchen-ai/Intent/tree/main/python/intent/mcp/) | Command-line and MCP access to the same public compiler API |

## Locate reusable components

The frontend builds typed values, operations, blocks and regions in `frontend/mlir/`, then serializes once. Helpers, control merges and structured intrinsics reuse scope, product, shape and region construction in `frontend/lowering/`. Intrinsic families live in `frontend/lowering/intrinsics/`. New rules consume the existing context rather than keeping a second type or SSA-visibility system.

Semantic analyses live in `include/Intent/Analysis/` and `lib/Analysis/`; physical relations live in the corresponding dialect's `Analysis/`. GPU and Triton–Ascend share Block computation. BANG C realizes explicit storage and engine cooperation from Block. CPU task retains its task, join and supply responsibilities. Directories such as `Reduction/`, `Storage/` and `Task/` group closely related implementations at the same level.

A provider's `Transforms/` selects and legalizes native forms; `Serialization/` spells the already formed program. Provider-native reductions, scans, layouts and instruction lowering stay with the provider when their contracts match. Public headers live under matching `include/Intent/` paths; local implementation helpers may remain adjacent to their source.

## Extension points for developers and Agents

| Capability | Component entry point | Boundary |
|---|---|---|
| Devices, toolchains and necessary specialization | `python/intent/targets/components.py` and component `Provider/LocalExecution` | Declaration queries do not import SDKs; ordinary JIT does not branch on provider names |
| Buffers, native compilation and launch | Component `python/intent/runtime/`; framework storage adapters in `integrations/` | Artifacts and buffer bindings remain separate; prepared calls retain actual resources |
| Models, pipelines and configuration resources | Component CMake registration and `Compilation` implementation | The common driver invokes stages rather than maintaining provider enums or filename tables |
| NVIDIA capabilities and GPU policy | `Target/NVIDIA` and Block GPU mapping | Common Block verification does not require CUDA resource fields |
| BANG C storage and engines | `Target/DSA/BangC/BlockLowering`, `Physical`, `Transforms` | Consume current Block; keep NRAM, Union and memory cores out of common DSA contracts |
| Ascend implementation | `Target/DSA/Ascend` and Python `targets/ascend` | Lower Block through official Triton–Ascend without BANG C storage realization |
| CPU implementations | `Target/Mojo` or `Target/Weft` | Components choose materialization stages; Triton–Weft uses formal `.weftir` JIT and its native launcher |

Locate the current IR stage before extending it. Reuse types, effects, coordinates and lifetime analyses. A pass defines input invariants, its transformation and postconditions; required lowering is separate from optional optimization. Target API spelling alone does not justify a common IR extension. Add a model fact only when execution needs it, updating verification, analyses and consumers together.

`LocalExecution.host_storage` registers explicit allocation, upload, view, snapshots and release for examples or applications choosing host staging. Ordinary JIT continues to bind caller-provided device buffers without automatic transfers. Invocation-private workspace is declared as real typed native ABI parameters and retained by the prepared call. It adds neither author parameters nor an optimization plan for runtime reinterpretation.

## Discover interfaces and diagnose failures

Query current declarations and dependencies in the installed environment:

```bash
intent describe --target triton --json
intent doctor --target triton --json
```

`describe` discovers interfaces; `doctor` checks the actual toolchain and target environment. Neither command compiles or executes a kernel.

These commands use an existing complete algorithm and stop at different stages:

```bash
python -m intent.examples softmax --backend triton --stage source
python -m intent.examples softmax --backend triton --stage native
python -m intent.examples softmax --backend triton
```

The complete-program entry prepares its existing inputs and runs its existing host composition. KIR needs no target or device; shared IR needs an explicit compilation target; native compilation and execution need the corresponding toolchain and device. See [compile and run](../getting-started/usage.md) for commands selecting one definition and inspecting KIR.

On failure, use the returned `stage`, diagnostic and `cache_directory` to find actual input, IR, source and logs. Locate the failure in the frontend, physical construction, family pass, provider legalization, external compiler or runtime. Fix that boundary rather than guessing missing semantics in a later layer.

## Configuration, numerics and performance

Empirical configuration tables live in each provider's `Transforms/` or `Transforms/Configuration/`. They declare complete correlated rows; runtime tuning evaluates those rows. [Configuration and tuning](../getting-started/configuration.md) explains the format, overrides and first-call behavior.

`CompileOptions` and target parameters are distinct interfaces. The former chooses defined numerical permissions and optional optimizations; the latter provides capabilities and compilation facts. Enabling an optimization does not grant extra numerical changes. Read [types, numerics and effects](../dsl/types-numerics-and-effects.md) for the contract.

A performance contribution explains the consumed facts, legality, actual IR rewrite and expected structural benefit. Verify affected complete programs and record source generation, native compilation, execution, numerical comparisons and warm-call time separately. Do not mix JIT, tuning or input preparation into a kernel warm-call claim. See the [contribution guide](https://github.com/Tianchen-ai/Intent/blob/main/CONTRIBUTING.md).
