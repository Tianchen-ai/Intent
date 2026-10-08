# Extending passes

Passes are the entry point for optimization knowledge. A pass reads current typed semantics and execution relationships, rewrites real IR and hands a complete program to later consumers. Before adding an optimization, read [passes and analyses](../compiler/passes-and-analyses.md), the relevant [GPU](../compiler/gpu-program-ir.md)/[CPU](../compiler/cpu-program-ir.md) contract and [target lowering](../compiler/target-lowering.md).

## Locate the responsible layer

| Change | Responsible location |
|---|---|
| Domain, dtype, effects and algorithm semantics | DSL / canonical KIR; discuss design changes separately |
| Def-use, coordinates, aliasing, dependence and working-set queries | The dialect's `Analysis/` |
| Ownership, traversal, blocking, reuse and materialization | Responsibility groups in GPU/CPU/DSA `Transforms/` |
| Capabilities, implementations, local forms and legality | `lib/Target/<Provider>/` |
| Source API spelling | Target serialization |
| ABI binding, toolchain invocation and tuning calls | `python/intent/runtime/<provider>/` |

GPU and CPU may reuse analysis ideas without sharing a physical topology. A pass need not run on every backend. Shared facts, independent decisions and appropriate target consumers are useful reuse.

## A complete optimization component

1. Define legality from current typed operands, regions, def-use, access relations, effects, lifetimes and capabilities, never kernel names, operation counts or string labels.
2. Let analyses provide recomputable facts. Unknown is not permission to replay, alias or assume initialization.
3. Rewrite operations, types, regions, coordinates, carries or resource lifetimes. Do not introduce a schema/plan that jointly interprets execution alongside IR.
4. Preserve logical members, dtypes, numerical permissions, ordered control, effects, ABI and the author's kernel count.
5. Declare invalidated analyses and check postconditions after a complete transformation group. Verifiers do not repair programs; serializers do not add loops, workspace or parameters.

Use existing MLIR pass declaration, registration and pipeline mechanisms. Inspect `include/Intent/Dialect/<Family>/Transforms/Passes.td`, `lib/Dialect/<Family>/Transforms/` and its `Passes.cpp`; target-local transformations live in `lib/Target/<Provider>/Transforms/`. `intent-opt --help` lists actual callable passes and options.

For example, GPU reduction optimizations live in `lib/Dialect/GPU/Transforms/Reduction/`, declarations in GPU `Passes.td`, and complete transformation entry points and ordering in GPU `Transforms/Passes.cpp`. CPU task partitioning and storage reuse live in CPU `Transforms/Task/` and `Transforms/Storage/`; CPU `Transforms/Passes.cpp` assembles that family's pipeline. Do not move a GPU ownership transformation there to force reuse. Provider empirical configuration tables live in the corresponding `Transforms/` or `Transforms/Configuration/`.

Add new implementations to the adjacent CMake list. Reuse an existing transformation wrapper's registration, dependent dialects, error propagation and completed-boundary verification. Insert the pass where its input conditions hold rather than adding a bypass for one program.

Changing complete empirical configurations does not require a new pass. When a provider owns collective trees, thread communication, distributed layouts or machine pipelines, supply legal input instead of rebuilding that mechanism in Intent.

## Observe existing programs

Choose affected complete programs, inspect before/after IR and generated source, verify actual calls and observe warm execution for the same algorithm and input. Do not copy algorithms or expand a test matrix. Benefits on another related program show the reuse scope. Report generation, native compilation, execution, numerical checking and performance improvement separately.

Use a standard MLIR pipeline on already generated IR that satisfies the pass's preconditions:

```bash
intent optimize /path/to/current.mlir --pipeline 'builtin.module(canonicalize,cse)' --json
```

Replace the pipeline with transformations listed by current `intent-opt --help` and applicable to that stage. Shared GPU passes cannot consume KIR or modules with provider-native forms already added. Native compiler MLIR IR-printing options locate a pass's input in the complete compilation. Re-run the corresponding complete program to observe the optimization; isolated IR optimization does not establish runtime results.

The repository [contribution guide](https://github.com/Tianchen-ai/Intent/blob/main/CONTRIBUTING.md) covers development setup; [compiler development entry points](compiler-reference.md) locate public APIs, implementation layers and stage diagnostics.
