# Kernel and host boundaries

## 1. Kernel definition

An `@intent.kernel` defines one independent logical kernel computation. A compilation call selects the target outside source. For a given specialization, interface and target, one compilation produces one target kernel artifact. Host/runtime determines when and how to launch that artifact.

A kernel:

- Receives parameters through `In`, `Out`, `InOut` views and runtime scalars.
- Does not return a device tensor to the host through Python return.
- May contain hardware-independent control, state, structured operations and effects.
- Does not query targets, device resources, provider capabilities or launch configurations.
- Does not allocate or launch a hidden second kernel.

The compiler must not split one kernel into multiple host-visible invocations or create cross-kernel workspace and synchronization invisible to the host.

## 2. Multi-kernel algorithms

An author may implement the same external callable contract with one or multiple kernels, preserving the task's required numerics, effects and interface semantics. Authors may design internal kernel interfaces, logical groupings and intermediate tensor shapes; these need not be specified by the external caller. Observable interfaces also include views and scalars actually passed to each kernel inside the wrapper.

With multiple kernels, the author explicitly defines multiple `@intent.kernel` functions and uses ordinary Python host code to:

1. Select algorithm specialization and compilation target.
2. Allocate outputs and intermediate tensors.
3. Establish invocation order or concurrency relationships.
4. Manage cross-kernel visible state and lifetime.

Intent compiles kernels individually; the Python wrapper is the authority for multi-kernel orchestration. Compilation, artifact creation and launch are separate stages, not a single Kernel IR operation.

## 3. Helper functions

`@intent.fn` is a hardware-independent helper inside a kernel:

- Parameters and results are typed DSL values.
- It may return scalars, tensors, tuples or records.
- It may have read/write effects with the same semantics permitted at the call site; effects come from the function body.
- Runtime values must be explicit parameters; only immutable `Constexpr` values may be captured.
- Recursion, host dispatch, kernel launch, target queries and provider selection are forbidden.

Inlining a helper or making it a target-local device function is an implementation choice that does not change call semantics.

Helper calls also preserve typed index relations. Coordinate values from `I.indices`, subregions or indexed relations retain the same source identity, axis mapping and SSA provenance through helper parameters/results. Inlining must not turn them into integer tensors without provenance.

## 4. Runtime and `Constexpr`

Runtime parameters take values on each invocation and may participate in computation and runtime control flow.

`Constexpr[T]` is bound at specialization and controls only hardware-independent algorithm variants, for example:

- Causal versus non-causal behavior.
- Whether an activation is applied.
- Algorithm-defined group counts, sparse formats or numerical modes.

`Constexpr` must not query or choose:

- Target/provider, GPU model, ISA or capabilities.
- TMA/MMA/layout.
- Warps, stages, tiles or memory spaces.
- Autotune configurations.

The selected hardware-independent branch enters canonical Kernel IR; target-specific selection occurs afterward.

## 5. Control within a kernel

A kernel permits:

- Runtime `if` with a scalar runtime condition.
- Algorithmic constexpr `if`.
- Ordered `for/while`, loop-carried values, `break` and `continue`.
- Unordered `parallel` iteration.

Tensor predicates use value-level `select`, not structured `if`. There is no independent `stop` operation; loop conditions, `break` and domain/subregion endpoints express termination.

All these controls belong to the same kernel and must not be interpreted as multi-kernel dispatch.

## 6. Interfaces and kernel-local resources

An external view exposes element dtype, logical shape, runtime element strides, access direction, bounds and necessary alias relations. Authors do not calculate target pointer arithmetic or make alignment, contiguity, TMA eligibility or vector width algorithm parameters.

A kernel-local logical buffer is algorithm state. The author defines shape, dtype, initialization and read/write relations without selecting its eventual SSA, register, stack, shared/local-memory or other target storage. An intermediate tensor that must survive across kernels is allocated explicitly by the host and is not a kernel-local buffer.

Concurrent access to an external allocation across kernels or host/device is defined by Python wrapper/runtime invocation and synchronization semantics. A kernel body does not use physical atomic scope to expand its participants implicitly.
