# KIR to GPU Program

## 1. Immutable semantic input

KIR is frozen on entering physical construction. It is the sole authority for:

- Kernel ABI, logical shapes, views, aliasing, and effects.
- Domains, source-derived subregions, indexed relations, logical identities, coordinate expressions, and predicate/member sets.
- Ordered control, unordered parallelism, loop carries, and stop conditions.
- The reduce/scan/region-fold/region-scan/contract families, histogram, atomics, RNG, and their numerical semantics.
- How many kernels an algorithm uses and each kernel's boundary.

GPU conversion may analyze these facts and produce an equivalent physical program, but must not change logical members, operation semantics, effect order, or the number of kernels.

## 2. Logical workset

A GPU program instance does not originate from a fixed syntactic node. The compiler first constructs logical worksets.

A logical workset consists of a finite instance domain and a set of connected computations/effects, satisfying:

1. Every instance executes the group's program slice once.
2. There is no ordered data dependence between instances.
3. Overlapping effects use only explicit scatter-reduction or atomic semantics.
4. The group's values/effects have unique ownership.
5. Multiple outputs and shared producers may belong to the same workset.

Workset facts may come from:

- Explicit unordered `parallel` iterations.
- Unique pointwise/tensor writes.
- Batch and free axes of reduce/scan/contract.
- The iteration domain of an indexed write, atomic, or scatter operation.
- Other independent dimensions proven by def-use, alias, effect, and dependence analyses.

Ordered loop axes, reduction axes, scan prefix axes, strict recurrences, and mutable state across iterations remain inside a program instance; they do not become independent program instances.

## 3. Execution group

Each output element must not mechanically become a program: multiple outputs may share a producer, a whole-tensor structured operation may have no explicit point loop, and scan or recurrence cannot be split along a dependent axis.

Construction forms execution groups from connected worksets that must share ownership or have dependencies:

- Outputs sharing a producer or buffer state remain in the same group.
- Independent groups may have different instance domains.
- A program slice whose independence cannot be proven becomes a single-instance group, preserving ordered loops internally.
- Logical-buffer dependence across instances merges the relevant instances unless KIR already defines concurrent updates using atomic/reduction semantics.

An execution group is only an internal dispatch region inside the same physical kernel body, not a kernel, artifact, or launch. One `@intent.kernel` specialization still produces one launch. Multiple execution groups execute through the same physical program space without forming hidden kernels. Authors define multiple kernels and orchestrate them through a host wrapper for a multi-kernel algorithm. This is an Intent semantic boundary, not a description of Triton/cuTile's general capabilities.

## 4. Initial physical program

The first GPU program uses a conservative but complete uniform mapping:

1. For each execution group, only the finite instance domain available before launch is bijectively flattened into a one-dimensional segment in logical row-major order.
2. Each segment's runtime length `L_g` is a typed launch expression formed from ABI scalar parameters, constexprs, and verified shape relations. Its prefix offset is `O_g = sum_{h<g} L_h`, and the kernel's program-space length is `sum_g L_g`. The provider wrapper and kernel body bind the grid and dispatch from the same expression.
3. `L_g == 0` produces an empty segment and launches no program belonging to that group. If the provider grid requires rounding up, explicit program-validity guards prevent programs beyond the total length from executing effects.
4. A physical program ID selects a segment using `O_g <= id < O_g + L_g`, then decodes `id - O_g` into that group's logical coordinates.
5. Data-dependent/ragged domains whose member counts require reading a device tensor must not determine launch cardinality. Initial conversion maps only their launch-visible outer domains into program instances; data-derived subregions/members remain inside an instance and are traversed in order or by an existing structured operation. If even the outer extent is unavailable before launch, the group initially uses a length-one segment and traverses the entire domain inside its sole instance. Without an additional author-written kernel, the compiler must not first generate a prefix/count launch.
6. All dependent axes, runtime subregions, and strict control in a group body remain ordered structured loops.
7. Scalar computations remain scalar; tensor computations initially use the smallest legal fragment or scalar loop.
8. Every load/store/gather/scatter/atomic immediately produces an explicit access relation, typed coordinate SSA, source-provenance dependencies, an active member set, and validity.
9. Logical buffers immediately receive allocation scope, initialization mode or first-write obligation, reads/writes, ownership, and lifetime. These facts are not deferred to the emitter.
10. Reduce/scan/region-fold/region-scan/contract immediately become executable physical structured operations. An unblocked version may be slow, but must not retain only a record awaiting interpretation by a materializer.

If a kernel contains only one indivisible ordered group, that group's segment length is one and the entire algorithm executes in order inside a single program instance. This follows from completeness; it is not the default GPU strategy for every kernel. Subsequent passes may enlarge a physical program instance's logical ownership extent, regroup/swizzle program space, or form persistent traversal, but every step must preserve workset coverage and effects. A pass may promote data-dependent member traversal into program space only when the new mapping can be calculated directly from launch-visible values or an existing index relation. It must not introduce a hidden launch to obtain the mapping.

## 5. Buffer and effect scope

Every logical buffer allocation in KIR has a lexical allocation instance. Construction must assign it to one of the following physical scopes:

- **Program-private:** one copy per owning physical program instance.
- **Iteration-private:** one copy for every dynamic execution of the ordered loop/region containing its declaration.
- **Invocation workspace:** one compiler-private resource accessed by multiple program instances in the same kernel invocation through explicit ownership slices. It is allocated by the runtime and passed as a hidden ABI argument, but must not introduce an additional launch.
- **External view:** passed through the public ABI; allocation, initial contents, and lifetime across kernels belong to the host program.

If KIR provides a full initial value, initialization executes exactly once per logical allocation instance. If KIR creates an uninitialized buffer, construction must not synthesize default initialization; it must preserve the author's write-before-read obligation and existing read/write order. Insufficient static initialization proof does not block construction or order-preserving storage lowering. A transformation depending on initialization facts must still prove its required conditions. If a mutable buffer creates non-atomic dependence between program instances, conservative construction must bring those computations back into the same program instance. It cannot depend on a global barrier absent inside the kernel. Invocation workspace can be shared in parallel only when every slice has a unique owner or KIR already has scatter-reduction/atomic semantics.

An atomic operation holds KIR memory order, logical allocation/address relations, its return-value schema, and logical sharing domain. A physical pass selects the provider-required scope according to program mapping. Scope must not be hard-coded in KIR, and old-value/CAS-success semantics must not be lost.

## 6. Multiple outputs and different worksets

One execution group may write multiple output relations from a single program instance. For example, two outputs sharing one normalization summary must share fragment SSA and program ownership.

Independent groups with different shapes use disjoint program-space segments. Each program executes only the matching group-dispatch branch; this dispatch is uniform control within a single program instance. Runtime-data-dependent `if`, validity, and predication inside a group body may still diverge and must preserve original effects. A subsequent profitability pass may merge compatible groups into one program mapping or retain the initial union.

Every output within a group retains its independent result relation and per-instance slice. Shared producers share only values/ownership and do not require outputs to have identical shapes. An empty output slice produces zero effects through its write relation; it does not cancel computation of other outputs by the same instance.

This representation preserves one launch per kernel without disguising every output as the same logical shape.

## 7. Initial mapping of structured operations

- Pointwise/unique write: result/free axes enter the workset; each initial instance handles the smallest legal value slice.
- Reduce: non-reduction axes enter the workset; reduction axes remain inside the instance.
- Scan: non-scan axes enter the workset; the scan axis is handled by a physical scan or ordered carry.
- Region fold: summary free axes enter the workset; the source axis remains inside the instance. A physical pass selects contiguous nonempty segments, executes the explicit summarizer for each, then merges using summary combine.
- Region scan: output/free axes enter the workset; the source axis remains inside the instance. The physical program explicitly holds the segment summarizer, transition combine, incoming-state application, slice emitter, and final-state flow.
- Contract: batch and free axes enter the workset; paired reduction axes remain inside the instance.
- Ordered recurrence: provably independent outer axes enter the workset; state-carry axes remain inside the instance.
- Dynamic subregion: begin/end/source provenance become runtime SSA and access validity without changing program-instance identity.
- Helper-produced coordinates, slices, broadcasts, reshapes, transposes, and predicates must compose into the same physical coordinate/relation graph. A pass recomputing them must actually rewrite SSA def-use rather than retain only a provenance ID or range record.
- Scatter/atomic: source iteration may become a workset; collision and ordering are preserved by operation semantics.

Ordinary `for/while` in physical IR holds runtime conditions/bounds, region arguments, loop-carried SSA, memory effects, `break/continue` edges, and yields. The compiler need not prove termination; it only preserves KIR control semantics. Reduce/scan hold physical axes, component-wise identity, a typed combine region, and an accumulator schema; scan additionally holds inclusive/exclusive mode, direction, and result relation. Region fold/scan also hold source-slicing relations, summarizer regions, and explicit captures. Region scan holds transition combine, apply, emit, output assembly, and final-state flow. Arg-reduce tie/NaN rules, dynamic extents, and all stop conditions are also explicit operands/attributes/regions rather than inferred by a provider from operation names.

## 8. Origins and semantic preservation

Construction establishes stable origin references for kernels, operations, values, regions, and block arguments. Origins serve only:

- Conversion legality and semantic-preservation verification.
- Diagnostics and generated-source attribution.
- Indexing to project immutable KIR facts into physical analysis.

A physical program must not rely on origins to fill in execution structure or coordinate provenance. Coordinate expressions, axis maps, active member sets, and validity become the current program's SSA/operation payload during conversion. Origins can only check whether these mappings preserve KIR semantics. Combine regions, ordered control, and other semantic payload needed for execution also lower into physical regions/operations during conversion. A provider need not read KIR to execute or serialize the program.

Origin mapping distinguishes at least KernelID, OpID, ValueID, RegionID, and region argument position. String operation names may be used for diagnostics only, not as identities.

## 9. Construction verifier

After initial conversion, verify that:

- Each KIR-observable effect is covered by exactly one physical execution slice.
- Ownership of every logical result/member is complete without overlap undefined by the semantics.
- Ordered dependencies are not broken across independent program instances.
- Unordered conflicts are resolved by unique, reduction, or atomic semantics.
- Coordinates, validity, and fill/effect are complete for every access.
- Every buffer preserves its initial value or the author's first-write obligation, with complete ordered read/write dependencies.
- Every structured operation has complete operands, results, regions, and accumulator.
- Every region fold/scan has complete segment source relations, summarizer, identity, combine, and scan apply/emit/result assembly, without segment identity/extent leaking into a KIR-observable value.
- All physical values have legal scalar/fragment types.
- Every runtime program-space extent depends only on launch-visible values; data-derived member domains still have complete traversal inside the program.
- Every buffer has complete allocation scope, instance identity, initialization mode/coverage, ownership, and visibility.
- Every dynamic/indexed access is proven to be within the target resource by an explicit guard, a verified relation, or canonical `assume_in_bounds`.
- Every physical coordinate expression has source identity/rank and complete SSA provenance; slice/broadcast/reshape/transpose/helper-call composition is equivalent to the KIR relation.
- Every active-member/range narrowing is an explicit subset of the original relation with a typed predicate/range proof. It must not be reconstructed from shape, operation name, or an origin side record.
- Every atomic operation preserves order, logical sharing domain, and result semantics, and a legal provider scope can be obtained from physical mapping.
- The physical program needs neither KIR nor side records to interpret execution.
