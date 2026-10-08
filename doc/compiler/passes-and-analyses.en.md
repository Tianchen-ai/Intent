# Analyses, passes and verification

## 1. Basic discipline

A pass is justified by changing the current program while preserving verifiable invariants, rather than by its filename, the number of passes or fields in a side table.

Each transformation declares:

- The KIR semantic facts and current physical facts it reads.
- Legality conditions and preconditions.
- The types, operations, regions or def-use relationships it rewrites.
- The logical values, control, effects and ABI it preserves.
- The analyses it invalidates or recomputes.
- Its post-pass verifier.

Kernel names, operation counts, whole-region templates and provider strings must not drive shared policy.

The module's `intent.compile_options` preserves effective compilation permissions and optional controls. `numerics` is `source` or `relaxed_normalization`; `onlineReduction` controls whether an online rewrite of an ordinary normalized summary is considered; `optimizationRemarks` controls adopted/rejected diagnostics. Defaults are `source/true/false`. Construction, branch cloning and provider containers carry this attribute. Continuing compilation from physical IR reads the existing attribute rather than silently replacing it or supplying different default permissions. Provider metadata's `compile_options` agrees with it, and compilation caches consume the actual options.

Online transformation checks current summary/axis relationships, numerical permissions, identity, dtype and replay/effect conditions before creating execution structure. Its switch does not bypass validation or other required lowering. Current additional-mode eligibility covers f32 score, maximum and mass/moment accumulation, with matching f16/bf16/f32 weight/value operand dtypes. Other cases retain ordinary blocking. This is optimization eligibility, not an additional restriction on legal DSL dtypes. [Numerical permissions §5.6](../dsl/types-numerics-and-effects.md#56-numerical-permissions-on-compilation-calls) defines finite-value preconditions and the specific allowed rounding changes.

Optional remarks report adoption, disabling, missing numerical permissions or unmet implementation conditions at the current operation. They are not execution plans. A family without the corresponding optimization may leave additional permission unused; that permission must not otherwise change the family's computation.

## 2. Analysis layers

### 2.1 Canonical analyses

These are computed only on immutable KIR:

- Shape, domain and subregion identity.
- Coordinate provenance, axis maps, active index sets, index relations and aliasing.
- Operation effects and collision semantics.
- Control dependence, loop carries and ordered state.
- Reduce, scan, region-fold, region-scan and contract schemas.
- Logical-buffer lifetimes.
- Def-use, producer/consumer relationships and logical reuse.

Stable KIR IDs key these analyses. A physical program does not rerun a `KernelModel` that pretends to be canonical.

### 2.2 Construction facts

KIR-to-GPU conversion uses canonical analyses to produce:

- Logical worksets and execution groups.
- Initial program-space mapping.
- The KIR origin map.
- The initial physical scalar/fragment/value/access/control graph.

After construction, these results are in GPU IR. Later passes do not treat construction facts as another program authority.

### 2.3 Physical analyses

These are computed only from current GPU IR:

- Program coverage and mapping.
- Physical dominance, def-use and loop dependence.
- Fragment axes, compositional coordinate maps, predicate ranges, access footprints and validity.
- Reuse, rematerialization, buffer lifetimes and sharing.
- Structured-operation operand/accumulator flow.
- Target-independent resource estimates.
- Provider-form eligibility.

Physical analyses may use the origin map to query immutable KIR semantics. They cannot use KIR adjacency or shapes to supply execution structure missing from current IR.

Relevant IR mutation invalidates analyses by default. Only analyses whose preservation is proven may remain cached.

A transformation group is a complete execution boundary. It owns the actual rewrite and maintenance of its dependent value/access/aggregate/accumulator relations. Intermediate repair helpers are not independently exposed passes. Verify the same current program after each group; failure diagnostics distinguish rewrite and postcondition failures and identify the group. Verifiers do not repair programs.

## 3. Canonical GPU pipeline

### 3.1 Construct an executable program

KIR, canonical analyses and selected GPU capabilities produce the complete initial program defined in [KIR to GPU](kir-to-gpu.md). Conversion closes types, regions, values, accesses and origin mapping without holes awaiting emitter interpretation.

Construction closes only the initial program and indexed-access composition. Structured-source normalization, ownership/blocking, online summaries, and region/reduction/contraction realization belong to subsequent explicit shared transformation groups. They are not hidden in a construction-completion function. Dependency ordering normalizes source relations before ownership changes, forms physical ranges before structured realization, and recomposes newly created accesses before provider legalization. Mapping refinement and candidate materialization consume the completed program.

### 3.2 Refine program mapping

Worksets, accesses, reuse, effects and device resources guide rewrites of:

- Program-space rank and linearization.
- Execution-group placement.
- Logical ownership extents of each physical program instance.
- Grouping and swizzling.
- Grid-stride or persistent-traversal eligibility and structure.

When a provider's lower compiler reliably forms a persistent form from ordinary program structure, the shared pass does not duplicate its loop. It retains the legal input that lower compiler needs. If Intent owns that structure, its pass must create an actual physical loop.

### 3.3 Form blocking and fragments

Rewrite scalar or smaller work ownership into parameterized fragments and nested physical loops:

- Enlarge the logical instances owned by one physical program instance.
- Separate free, ordered, reduction and scan axes.
- Create scalar-to-fragment and fragment-to-scalar conversions.
- Update all users, access coordinates and validity.
- Preserve source coordinates of dynamic subregions.

A blocking pass must not change KIR logical subregions or author-observable page/window/chunk boundaries.

### 3.4 Co-realize values, accesses and structured operations

Three uninformed side-table passes cannot make these decisions: fragment shapes change access footprints, access forms affect materialization, and contract/reduce skeletons determine accumulators and reuse.

Implementations may use multiple passes and iterate canonicalization, provided each step preserves a complete program:

- Replay/rematerialization changes physical def-use.
- Bufferization creates actual allocation, lifetime, reads and writes.
- Access realization creates coordinate, validity, fill and effect operands.
- Contract/reduce/scan/region-fold/region-scan realization creates actual fragments, loops, carries and accumulator graphs. Contracts in region summarizers remain explicit; attention-shaped algebraic recognition must not recover them later.
- Ordinary contraction and its sole same-dtype addition consumer may merge under DSL local-fusion semantics. Current def-use, zero initialization, result-coordinate correspondence and dominance prove legality. The other addition operand connects directly to the physical accumulator. Do not cross numerical casts or perform this fusion implicitly in the serializer.
- Sole-use floating multiplication and same-dtype addition reduction may form a contract during structured-source normalization under DSL local-fusion semantics. Current broadcast, reduction axes and result-coordinate relationships determine paired/free/batch axes. Reuse existing contraction blocking and lowering without crossing numerical casts or changing author-ordered control.
- Boundary neutralization removes or simplifies actual validity/fill, rather than merely deleting a padding record.

### 3.5 Predicate range narrowing and summary emptiness

Canonical coordinate-provenance analysis starts from `I.indices`, domains/subregions/index relations and composes helper calls, slices, broadcasts, reshapes/transposes, integer arithmetic and comparisons. After blocking forms the current physical fragment, predicate-range analysis projects a logical predicate onto that fragment's source ranges:

- Inputs must be typed coordinate expressions, current physical ownership/ranges and structured-operation identity/effect facts.
- Exact proofs of monotonicity, bounds and set inclusion are required to derive all-true, mixed and all-false contiguous intervals. Otherwise preserve the original traversal and predicate.
- An all-true interval may remove redundant validity. A mixed interval retains the original predicate. An all-false interval may be skipped only when the summarizer/reduction produces identity for every free lane and has no effects.
- The rewrite changes actual physical loop bounds, access coordinates, active sets and validity SSA without changing KIR logical source relations.

The all-false-to-identity proof uses local typed value propagation: substitute the predicate into `select/mask`, then use builtin reduction identities, zero contractions, pointwise constant folding and tuple/record component equality. It recognizes neither summarizer names nor attention shapes. If any component cannot be proven equal to identity, the whole interval remains.

For example, with current query fragment `q in [q_begin,q_end)` and predicate `q >= k`, `k in [0,q_begin)` is all-true, `[q_begin,q_end)` is mixed and `[q_end,K)` is all-false. This is a coordinate/index-set rule, not an attention or causal-kernel matcher.

After range narrowing, summary-emptiness analysis may prove that a physical traversal first produces a nonempty summary for every free lane, or that the selected concrete combine graph never combines two empty summaries. While preserving NaN, identity, source order and empty-input semantics, the pass may remove optional/valid components and their selects from physical carries. Canonical KIR's total identity remains unchanged. Keep validity when the proof fails or a path may combine two empty summaries.

### 3.6 Target-capability legalization

The shared GPU verifier first checks provider-neutral legality. Selected provider/hardware capabilities then drive local checks and extensions required by actual differences:

- Grid rank and static fragment requirements.
- Native structured-operation/input dtype support.
- Descriptor, tile-index, storage, copy and synchronization forms.
- Compile-time parameters and resource constraints.

Reject an inexpressible combination at the earliest layer with sufficient information. Do not emit pseudo-support that is an order of magnitude slower or wait for a JIT timeout.

### 3.7 Deterministic serialization

The serializer traverses the legalized current program and emits provider source. It must not invoke a canonical `KernelModel`, infer fragments from result shapes, parse role strings, create workspace/loops/masks/grid or append search parameters.

## 4. Semantics-preserving rewrites

Physical IR may differ from the KIR operation graph, provided changes are implementations permitted by KIR semantics and the compilation call's explicit numerical permissions. Examples include:

- Merging pure producers and consumers.
- Rematerializing pure values.
- Reassociating and permuting ordinary reduce under its associative/commutative contract; reassociating scan and ordered region operations while retaining member order under their own contracts; contract reassociation under its separate numerical contract.
- Merging zero-initialized contraction with its sole same-dtype addition consumer under the DSL's local contraction-add rule.
- Normalizing eligible floating product sums to contract under the DSL's local multiplication-reduction rule.
- Flattening or permuting paired contract axes.
- Creating blocking loops and fragment accumulators.
- Forming online segmented rescaling under explicit normalized-summary numerical permissions and applicability conditions.
- Proving all-true/all-false/mixed ranges from coordinate predicates and removing identity-only physical traversals or redundant summary validity.
- Mapping exact read-to-write relationships to bulk transfers.
- Grouping, swizzling or grid-stride traversal of program instances.

Illegal changes include:

- Converting an ordered recurrence into reduce.
- Changing KIR combine, dtype, NaN/tie or atomic order.
- Changing logical subregion members.
- Changing kernel counts or introducing hidden cross-kernel workspace.
- Replacing one algorithm with another whole-operator template.

## 5. Verifiers

### 5.1 KIR verifier

Check semantic legality defined by the programming model and DSL, not GPU profitability or provider capabilities.

### 5.2 Construction verifier

Check KIR coverage, origin mapping and the first complete physical program under [construction rules](kir-to-gpu.md).

### 5.3 Shared GPU verifier

At least the following are checked after every shared pass:

- Program space, execution groups and ownership coverage.
- Scalar/fragment shapes and parameter expressions.
- Structured control, dominance, loop carries and terminators.
- Accesses, validity, fill, effects and alias/conflict relationships.
- Coordinate provenance, axis-map composition, active-index-set subsets and predicate/range proofs.
- Buffers, initialization/first-write obligations, lifetime and visibility.
- Structured-operation operands, results and accumulators.
- Declared physical parameters with legal domains.
- Absence of KIR operations, unresolved decision records or unknown executable fields.

### 5.4 Provider verifier

Check only current-program legality for the selected surface and completeness of local extensions. Do not decide shared structure again.

### 5.5 Serializer verifier

Before serialization, prove that every operation has a unique provider spelling or is explicitly unsupported. Serializer fallback is forbidden.

## 6. Legal side information

The following may exist outside executable SSA:

- Immutable diagnostic origin/provenance.
- Analysis caches.
- Diagnostics.
- Target capabilities.
- Physical parameter declarations and candidate domains.
- Benchmark/tuning artifacts.

Provenance/range-analysis caches may be side information. Coordinates, active member sets, validity, narrowed loop ranges and identity elimination that affect execution must also be rewritten into current executable IR.

The following must not exist only in side records:

- Program ownership and grid mapping.
- Loop/range nesting.
- Physical value shapes and uses requiring residency.
- Pointers, indices and validity.
- Accumulators, replay and materialization.
- Buffer allocation and lifetime.
- Persistent/pipeline control structure.
- Provider-primitive operands and results.

## 7. Evidence of pass quality

A compiler capability must demonstrate:

1. Actual before/after IR differences.
2. Legality and semantic-preservation conditions.
3. Whether another kernel or shape produces a different decision.
4. Independence from a triggering kernel name.
5. Numerical and performance reproduction on affected programs.

Changing attribute strings or candidate names without changing the current program is not a new physical compilation capability.
