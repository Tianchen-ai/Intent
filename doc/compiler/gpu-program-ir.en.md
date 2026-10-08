# Shared Executable GPU Program IR

## 1. Role

Shared GPU IR is a provider-neutral, launchable block program that can be verified independently. It sits after canonical KIR and before Triton/cuTile source legalization.

It expresses the program-level facts shared by GPU source providers, without expressing lower-level distributed layouts or machine instructions. A provider serializer must not defer missing execution structure to the string-emission stage.

## 2. Top-level structure

A physical module contains:

- An external target-capability object.
- One or more physical kernels produced from different KIR specializations.
- Each kernel's public ABI and compiler-private resource parameters.
- Compile-time physical parameter declarations.
- The kernel's program space, body, and origin map.

A physical kernel corresponds to one target artifact and one launch. Compiler-private workspace may become an additional kernel argument allocated by the runtime, but it must not introduce a hidden second launch or cross-kernel synchronization.

## 3. Program space

Program space defines a finite set of physical program instances. The IR explicitly represents:

- Program-space extents.
- Current program coordinates.
- Execution-group segment lengths and prefix offsets.
- A bijection or guarded mapping from coordinates to an execution group and logical workset coordinates.
- Tail/empty program validity.
- Optional grouping, swizzling, or grid-stride traversal.

Program space may have any logical rank; provider legalization flattens it or maps it to the grid rank supported by the target surface. Mapping must be a first-class IR relation, not reconstructed by a serializer from axis roles or result shapes. Runtime extents may depend only on launch-visible ABI scalars and shape relations. A ragged/member domain whose length requires reading device data must remain inside the program body unless KIR itself already provides a mapping directly usable for launch.

Each workset's actual execution body is held by an `execution_group` region. Its operands hold the current linear program coordinate and runtime extents; typed attributes hold the corresponding launch extents, coordinate roles, and segment identities/bounds. Region arguments are index coordinates decoded in row-major order using these extents. The outer dispatch guard remains explicit structured control. The group itself adds neither a launch nor a barrier and does not change effects in its body. A mapping change must rebind both the region arguments and their execution body.

After forming the native grid, the provider entry uniformly expands execution groups: region arguments are replaced with the results of pure `delinearize`, and the body is expanded in place. `delinearize` expresses only the mathematical decoding of `linear` using `extents`; it holds no launch, segment, or ownership facts. Ordinary CSE/DCE may merge or remove it without a rule to retain unused operations.

Program coordinates exist only in physical IR and do not flow back into DSL/KIR values.

### 3.1 LaunchExpr

Program-space extents, execution-group segment offsets, and provider grids use a restricted typed `LaunchExpr`. Its leaves may only be:

- Integer/bool constants and bound physical parameters.
- Scalar parameters passed by value in the public ABI.
- Host-visible shape/stride metadata from the external-view ABI.

Permitted operations are effect-free integer/bool arithmetic, comparisons, `select`, `min/max`, floor/ceil division, and tuple indexing. Canonical equality/bounds facts are used only to prove an expression legal; they do not replace runtime values. A `LaunchExpr` cannot load device memory, call a kernel helper, or depend on kernel-body SSA.

The provider wrapper evaluates `LaunchExpr` before launch to form the grid; the kernel body reconstructs dispatch/validity from the same typed expression. If a provider's launch ABI cannot directly share a metadata value on both sides, lowering declares the wrapper-computed result as a compiler-private scalar argument. This argument and its producer must explicitly exist in the physical module; the serializer must not add them on demand.

## 4. Values and types

The shared GPU IR distinguishes at least the following:

### 4.1 Scalar

Canonical scalars, logical indices, and predicates retain their precise definitions. A scalar is not disguised as a one-element fragment to unify code paths.

### 4.2 Fragment

A fragment is a statically shaped or compile-time-parameterized shaped SSA value owned by a program instance:

```text
fragment<element_type, [physical_extent_exprs]>
```

A fragment holds:

- Element dtype.
- Physical shape expressions.
- Compositional mappings from fragment axes to logical coordinates and their source-axis provenance.
- Shape relations for the active member/index set and validity.
- The owning program instance/workset.

A fragment does not hold register/lane/warp/CTA distribution, shared/TMEM layout, MMA encoding, or provider memory space. These are determined by the external provider compiler or local extensions required by actual differences.

### 4.3 View

An external view preserves allocation identity, element offset, logical shape, element strides, access mode, bounds, and alias semantics from the public ABI. Physical accesses consume the view using explicit coordinates rather than deferring pointer arithmetic to the emitter.

### 4.4 Buffer

A physical buffer is a mutable resource inside the kernel, explicitly holding:

- Element type and shape.
- Allocation scope: program-private, iteration-private, or invocation workspace.
- Allocation-instance identity and ownership domain.
- Initialization mode and an optional full initial value, or an uninitialized-first-write obligation.
- Lifetime and uses.
- Sharing/visibility obligations.
- Whether a compiler-private workspace ABI is required.

A buffer with a full initial value is initialized exactly once per logical allocation instance. An uninitialized buffer generates no default fill and preserves the author's write-before-read obligation. The verifier does not reject a program because static initialization proof is insufficient. An optimization that relies on initialization facts must still obtain the required proof; Unknown must not be treated as initialized. The runtime allocates invocation workspace before the same launch. Each slice must have a unique owner or explicit atomic/scatter-reduction semantics; it must not depend on a hidden initialization kernel or kernel-global barrier.

The shared IR does not specify registers, local memory, shared memory, or TMEM. It first decides between an SSA value, a program-private resource, and invocation workspace; provider/storage passes then select the target memory form from lifetime, sharing, resource limits, and capabilities.

## 5. Control

The shared IR reuses structured SSA control:

- Scalar-condition `if`.
- Ordered `for/while` and loop-carried values.
- Execution-group dispatch.
- Compiler-generated physical traversal/blocking loops.

KIR's unordered `parallel` is realized during construction through program mapping or an explicit independent workset inside a program. It does not remain a semantic marker waiting for a provider to interpret it.

Physical loops may use runtime bounds or compile-time physical parameters. Blocking loops unobservable to the author exist only in physical IR; author-defined ordered control is preserved through origins and semantic-preservation rules.

## 6. Coordinate and access relations

Every physical access explicitly includes:

- The resource/view/buffer.
- Result axes.
- A typed coordinate expression for each source axis.
- SSA dependencies of coordinates on program IDs, loops, fragments, runtime indices, and source subregions.
- The active member/index set and validity composed from logical predicates and source bounds.
- Load fill or write collision/effect semantics.

Shared access operations are:

- Load.
- Store/unique scatter.
- Gather.
- Scatter-reduce.
- Atomic load/store/RMW/CAS.
- Buffer load/store.

An invalid load performs no memory access and returns its explicit fill; an invalid write produces no effect. Pointer tensors, cuTile tile indices, and descriptors are provider representations rather than shared access identities.

An atomic operation also explicitly holds memory order, logical sharing domain, RMW kind, and the old-value/CAS result schema. Provider scope is derived from that sharing domain and the current program mapping rather than selected as a serializer default.

## 7. Value operations

The shared IR includes operations with explicit definitions for both scalars and fragments:

- Arithmetic, comparisons, select, cast, and bitcast.
- Broadcast, reshape, transpose, join, tuple, and record operations.
- Explicit scalar↔fragment broadcast/extract.
- Immutable value reuse and rematerialization.
- Typed pure helper/combiner regions.

Unary/binary operations preserve KIR's per-operation `approximate` and `flush_to_zero` numerical attributes, both false by default. Legal operator/dtype combinations and accuracy ranges are defined by the DSL numerical specification. These attributes do not change fragment shape, ownership, or ABI and are not provider parameters. Operations with identical operands/operator kinds but different numerical attributes are not the same pure value.

Broadcast, reshape, transpose, slice, tuple/record extraction, and pure helper calls compose both value def-use and coordinate provenance. If a physical pass chooses to recompute a pure producer, it must actually change def-use while preserving an equivalent coordinate map. If it chooses materialization, it must create a buffer/value and corresponding uses. Writing only `replay=true`, a provenance ID, or a list of value IDs is insufficient.

## 8. Structured compute

Physical structured operations consume current scalar/fragment SSA:

- Reduce: physical axes, component-wise identity, a typed pure combine region declared associative and commutative, dynamic extent, result relation, and accumulator flow.
- Scan: physical axis, component-wise identity, an associative typed pure combine region, direction, inclusive/exclusive mode, prefix result relation, and carry. It preserves prefix member order and does not inherit reduce's reordering permission.
- Region fold: physical segment loop, source-slice operands, typed summarizer region, summary identity/combine, captures, and result flow.
- Region scan: physical segment loop, summarizer and transition combine, incoming-state application, slice emitter, source-aligned output assembly, and final-state flow.
- Contract: physical lhs/rhs/accumulator fragments, paired reduction and batch axis maps, free/result-axis order, the zero-reduction result rule, result relation, and accumulator dtype.
- Scaled/sparse contract: KIR's logical format/schema together with physical operand/access mapping.
- Histogram: physical input fragment, validity, and count result.
- Other local structured operations formally defined by canonical KIR.

Composite results such as arg-reduce also hold tie, NaN, and index semantics. Ordinary ordered loops explicitly hold runtime conditions/bounds, loop-carried values, effects, and terminators. Contracts/reduces already explicit inside a region summarizer remain separate physical structured operations; the compiler need not infer them from summary combine. These operations carry no provider primitive names, MMA versions, input-precision hints, K-pack, warp policy, or pipeline stages. A provider may directly map an operation to a native primitive, legally expand it, or explicitly reject it, but must not change its KIR semantic schema.

Ordinary reduce's associative/commutative contract comes from operation semantics and requires no additional algebraic proof of combine or reordering plan. Shared passes form blocking, accesses, and accumulator flow. When a provider has an equivalent collective, its internal reduction tree, lane/warp communication, and synchronization are handled by the provider compiler. Scans and ordered region operations continue to lower under their respective ordering contracts; ordinary reduce is not converted into scan followed by taking the final element merely to unify implementation.

A structured operation may additionally hold a physical effective source range obtained from typed predicate analysis. This range may narrow only physical traversal; it does not modify KIR's logical source. Excluded members must have been proven to produce identity/no effect for every free lane. Predicates may be removed in all-true intervals; mixed intervals retain original validity; all-false intervals may be removed from the physical loop/access graph. The effective range and rewritten loop/access SSA belong to the current program. Its proof is an analysis result recomputed from current coordinate/predicate relations; it may be cached but does not participate in execution interpretation.

## 9. Dependencies, storage, and synchronization

The shared IR uses SSA, memory effects, resource identities, and explicit dependence edges to express producer-consumer, visibility, and lifetime obligations. It does not preselect TMA/cp.async, mbarrier, named barriers, or a software pipeline.

The shared IR defines no barrier across physical program instances. Conflicts across instances can be resolved only through atomic/scatter-reduction semantics already in KIR; otherwise the mapping is illegal. A dependency token/operation with execution meaning across providers may be introduced only when a shared physical transformation actually creates asynchronous producer-consumer behavior inside one program instance. The shared IR must not be polluted merely to spell a provider's copy API or NVIDIA mbarrier.

A provider-local pass may expand shared dependencies/lifetimes into explicit allocations, copies, waits, and barriers or delegate them to the lower compiler.

## 10. Completeness invariants

Before and after every shared GPU pass, the program satisfies:

1. Exactly one executable physical kernel body per KIR kernel specialization.
2. Complete program space and logical effect/result coverage.
3. A legal scalar/fragment/resource type for every SSA value.
4. Fragment extents that are constants or declared physical parameter expressions.
5. A resource, compositional coordinates/provenance, active member set, validity, and fill/effect for every access.
6. Complete arguments, yields, and dominance for every control region.
7. Initialization or a first-write obligation, lifetime, and ownership for every buffer.
8. Complete physical operands/results and a semantic schema for every structured operation.
9. No non-atomic conflicting effects.
10. Runtime grid extents do not depend on device data readable only after launch.
11. Every physical range narrowing has a subset/identity proof against the original logical relation and has actually rewritten loops, accesses, and validity.
12. Execution does not depend on a KIR clone, axis/role strings, or side decision records.

## 11. Explicit exclusions

The shared GPU IR does not contain:

- Provider API spelling.
- Kernel-name or whole-operation template identity.
- Warp/lane/register distribution layouts.
- TTGIR-style blocked/MMA/shared encodings and `convert_layout`.
- CUDA shared/TMEM address spaces.
- Target copy instructions, TMA, WGMMA, TCGEN05, or MFMA.
- Software-pipeline stages, warp specialization, or barrier protocols.
- An autotune winner.

These belong to provider-local extensions or the external provider compiler.

## 12. GEMM example

After physicalization of a full-domain contract, shared GPU IR should have the following structure rather than a contract record plus KIR shapes:

```text
kernel @gemm<BM, BN, BK>(A, B, C, M, N, K) {
  grid = (ceil_div(M, BM), ceil_div(N, BN))
  (pm, pn) = program_coord(grid)
  m = pm * BM + range(0, BM)
  n = pn * BN + range(0, BN)
  acc = full<fragment<f32, [BM, BN]>>(0)

  for k0 = 0 to K step BK iter_args(acc):
    k = k0 + range(0, BK)
    lhs = load A[m, k] valid (m < M && k < K) fill 0
    rhs = load B[k, n] valid (k < K && n < N) fill 0
    acc = contract lhs, rhs, acc

  store C[m, n] = acc valid (m < M && n < N)
}
```

BM/BN/BK, the grid, loops, fragments, coordinates, validity, loads, accumulator, and store all belong to the current program. The Triton serializer mechanically spells them as `tl.program_id`, `tl.arange`, `tl.load`, `tl.dot`, and `tl.store`; cuTile maps them to `ct.bid`, tile indices, and `ct.load/mma/store`.
