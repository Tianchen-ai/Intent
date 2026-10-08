# Logical program

## 1. Domains, subregions and identity

A domain is a finite, ordered logical coordinate set. A basic domain is a half-open integer sequence:

```text
[begin, end), step > 0
```

Empty domains are legal; begin/end/step may come from runtime scalars. Logical `index` has one numerical definition across targets, independent of provider integer differences.

Contiguous subregions come from a source domain:

```python
axis = I.domain(0, N)
prefix = axis[0:valid_length]
window = axis[begin:end]
```

A subregion denotes the half-open source interval `[begin,end)` and requires `source.begin <= begin <= end <= source.end`. Equal endpoints give an empty subregion; out-of-bounds intervals are not implicitly clamped. It retains source identity, bounds and provenance:

- Iteration yields source logical coordinates without renumbering them to local ordinals.
- `I.indices(subregion)` returns those source coordinates.
- Use `enumerate` or an explicit ordinal domain for local ordinals.
- Physical padding does not change logical members.

Noncontiguous, repeated or reordered members use indexed relations rather than pretending to be subregions. Only author boundaries, input relations or algorithm metadata create subregions in Kernel IR. Compiler blocking exists only in target physical programs.

### 1.1 Coordinate provenance

`I.indices(axis_or_subregion)` produces a logical `index` tensor. Each result element's canonical relation retains:

- Source domain identity, source axis/rank and logical coordinate dtype.
- A typed expression from result logical axes to the source coordinate.
- That expression's SSA dependencies on domains, subregions, indexed values and predicates.
- The active member set and known bounds.

Provenance is part of the canonical value/relation, not a string axis name or debug-only origin. `@intent.fn` calls pass it componentwise; inlined and non-inlined helper calls have the same result. Slices compose source bounds without renumbering coordinates. Tuple/record construction groups components without losing their individual provenance.

Broadcast preserves explicit axis maps; transpose/permute composes axis permutations; reshape composes old/new axis maps through logical row-major linear coordinates. Integer arithmetic, comparisons and selects on coordinate values retain representable typed expressions and dependencies. If an operation cannot be represented exactly by a canonical expression, its numerical result remains correct, but coordinate/range analysis returns unknown rather than guessing from shape, names or nearby structure.

## 2. Shapes and tensor values

Every dynamic extent of a tensor value is an identity-bearing runtime shape value. Two unknown extents are not equal merely because both are dynamic. Equality comes only from the same value or an operation's explicit shape relation.

The pointwise surface permits scalar and size-one broadcasting, which the frontend normalizes to explicit canonical broadcast relations:

- Tensor axes align from the trailing side.
- Aligned extents must be equal or one must be `1`.
- `0` and `1` broadcast to `0`; `0` and other positive extents are incompatible.
- Runtime equality/size-one conditions remain explicit and cannot be guessed from static shapes.

`reshape` preserves logical row-major element order and total element count. At most one extent may be inferred, and inference is legal only when its result is unique. Reshape changes neither dtype, bit representation nor physical storage. `transpose/permute` specifies a logical axis permutation. Their result relations compose coordinate maps rather than retaining only result shapes.

`join(lhs,rhs)` is the sole two-input value construction: inputs have identical dtypes and logical shapes, and the result adds a trailing logical axis:

```text
result[..., 0] = lhs[...]
result[..., 1] = rhs[...]
shape(result) = shape(lhs) + [2]
```

It is neither a record nor concatenate nor general interleave. Interleaving created by subsequent reshape follows this element mapping and logical reshape order.

## 3. Unordered parallel iteration

```python
for point in I.parallel(domain):
    ...
```

This defines an unordered forall: every logical point executes once, with no observable order between points.

- Loop-carried values across iterations are forbidden.
- Conflicting effects outside atomics/reductions make a program illegal.
- `break` is illegal; `continue` skips only the current logical iteration.
- No program count, thread, lane, vector width, tile or actual simultaneous execution is specified.

The compiler may serialize, flatten, thread or vectorize this semantics. Where pure whole-domain tensor expressions already have unique writes, it may infer equivalent data parallelism from dataflow without an extra author-written `parallel` wrapper.

## 4. Ordered control and carries

Ordinary `if`, `for` and `while` preserve program order:

- Scalar Boolean conditions control `if`; SSA merges combine branch values.
- Ordinary domain iteration follows logical order.
- Values updated in a loop are loop carries.
- `break`, `continue` and early return normalize under structured-control semantics.

Ordinary ordered programs need no `ordered` marker. The compiler changes execution organization only after dependence/effect analysis proves the same result.

## 5. Recurrences, homomorphic region operations and ordered control

Intent has no `state_stream` that permits arbitrary bodies to be resegmented by compiler-selected extents. Such a construct does not establish why changed segmentation is the same program: a body may observe invocation counts, tails, local shapes and effects, generally preventing unique semantics.

The language distinguishes five structures:

1. Each logical element is already a summary; combine is associative and commutative; only a final reassociable/permutable result is needed: `reduce`.
2. Each logical element is already a summary; combine is associative; every order-preserving logical prefix is needed: `scan`.
3. The author defines how any contiguous source slice produces a summary; only the final summary is needed: `region_fold`.
4. The author defines slice summary, summary composition, incoming-state application and slice output; a result is needed at each logical position: `region_scan`.
5. Updates require strict order, dynamic termination, nonassociative state or ordered effects: ordinary `for/while` with carries.

`region_fold` operates on one ordered source axis. For any order-preserving contiguous segmentation `R0 ... Rn` that completely covers that axis, the author supplies:

```text
summarize(Ri) -> Summary
combine(Summary, Summary) -> Summary
identity : Summary
```

The following equations must hold:

```text
summarize(A ++ B) == combine(summarize(A), summarize(B))
combine(identity, x) == combine(x, identity) == x
```

`A` and `B` are adjacent contiguous slices preserving source order. Combine allows any order-preserving parenthesization, not permutation. Choosing this operation declares the equations part of the algorithm definition. The compiler does not prove mathematical associativity; it verifies types, schemas, purity, effects and source-axis relationships.

`summarize` receives tensor components sliced along the same source axis and explicit captures. It may contain pure tensor operations, including reduce, scan and contract. It may not contain external/logical-buffer writes, scatter, atomics, RNG or other observable effects. It cannot read segment ordinal, segment count, chosen extent or chunk-relative coordinates. For coordinates, pass `I.indices(source_axis)` as a source component; slicing retains absolute source coordinates.

`region_scan` uses the same summary algebra and additionally supplies:

```text
apply(prefix_summary, initial_state) -> incoming_state
emit(source_slice, incoming_state, captures) -> output_slice
```

Transition identity and composition must form a valid action on state:

```text
apply(identity, state) == state
apply(combine(a, b), state) == apply(b, apply(a, state))
```

For adjacent slices `A` and `B`, also require:

```text
emit(A ++ B, state)
  == concat(emit(A, state),
            emit(B, apply(summarize(A), state)))
```

`summarize`, `combine`, `apply` and `emit` are typed pure helpers. `emit` produces output with the same logical member relation as its source slice. The operation reassembles slice outputs on the original source axis and returns `apply(summarize(full_source), initial_state)` as final state. Compiler-selected segment counts/boundaries and internal prefix states are unobservable and cannot become result shapes or ABI. If an algorithm outputs per-chunk states or exposes chunk count in its ABI, chunks are logical data: use explicit chunk domains, source subregions and ordinary scan instead of `region_scan`.

Reduce/scan process element summaries; region fold/scan apply only when the author actually writes a region-level summarizer or emitter. From these definitions the compiler forms legal segmented traversal, within-segment computation, summary merging and output structure, preserving each operation's source-order and effect contract. Inner operations retain their own lowering semantics.

If page, window, group or chunk boundaries affect read members, output shape or ABI, authors calculate them explicitly and form source-derived subregions. If boundaries serve only physical blocking, authors do not specify their extents. Only the homomorphism above makes compiler-selected segmentation uniquely defined.

## 6. Structured tensor operations

`reduce`, `scan`, `contract`, `scaled_contract`, `sparse_contract` and `histogram` are first-class logical tensor operations, not target-primitive requests:

- Reduce retains accumulator schema, axes, identity and the declared associative/commutative typed pure combine.
- Scan retains an associative typed pure combine and defines order-preserving direction and inclusive/exclusive prefixes.
- Contract retains paired batch/reduction axes, free axes and accumulator for binary multiply-add contraction.
- Scaled contract retains microscaling formats, scale relations and logical contraction through the closed `[M,G,C]/[M,G] × [G,C,N]/[N,G]` positional schema.
- Sparse contract retains compressed values, typed format schema, metadata interpretation and logical contraction.
- Histogram retains binning semantics from values to a count tensor.

These operations permit their explicitly defined reassociation without selecting reduction trees, MMA, layout, storage or pipelines. If a target lacks an equivalent native primitive, it may expand legally or reject explicitly, but must not change the operation's logical definition.

## 7. Logical parts do not use `partition`

When part count, identity, boundary formulas or intermediate tensor interfaces are observable, authors express these relationships through ordinary domains, source-derived subregions, index arithmetic and control. `parallel` declares unordered independent logical points; source slicing defines actual members. Empty parts, tails and intermediate shapes belong to the algorithm, not physical tiles, thread blocks or extra launches.

Intent has no `partition(auto/count/extent)` operation. Ordinary semantics fully express observable partitioning; unobservable blocking belongs to physical programs.

## 8. Ragged and indexed relations

Ragged data consists of ordinary relations:

- An outer domain.
- A member source domain.
- Offsets.
- Optional member index mapping.

For outer identity `g`, contiguous members are `member_source[offsets[g]:offsets[g+1]]`; noncontiguous mappings form indexed relations. Subregion/relation SSA identity and provenance distinguish simultaneously existing relations.

Canonical Kernel IR introduces no independent RaggedOp or MembersOp. `ragged(...)` and `members(...)` may be ordinary surface helpers, but must expand mechanically into this sole relation path.

## 9. Indexed memory and effects

All indexed accesses share typed index relations and active validity. A relation retains source identity/rank, result logical axes, a typed coordinate expression for every source axis and its SSA provenance from domains, subregions or data-derived indices. Composition combines the relations themselves rather than keeping only result shapes.

In that common representation:

- Immutable tensor selection is pure gather.
- External-view loads are external reads.
- Logical-buffer loads are mutable-resource reads.
- Invalid read lanes do not access source and return an explicit fill.
- Invalid write lanes produce no effects.

Ordinary assignment and unique scatter use arbitrary-index unique stores with a provably injective destination relation. Collision reduction uses separate `scatter_reduce` and typed combine. Atomic load/store/RMW/CAS retain indivisibility, modification order, old values and memory order.

Intent has no canonical copy operation. An immutable SSA read followed by indexed write fully defines snapshot, mapping, validity, cast, alias and effect order. Physical programs form bulk transfer, async copy, DMA/TMA and synchronization protocols.

## 10. Atomics and RNG

Atomic operations have no author-visible physical scope. They define modification order among executions in the current kernel invocation accessing the same atomic object through the same logical allocation/address relation. `relaxed/acquire/release/acq_rel` change happens-before and are operation semantics. Providers choose physical scope from logical sharing and target execution mapping.

Conflicting non-atomic accesses are illegal data races. The language guarantees neither simultaneous residency of parallel iterations nor spin-wait progress.

RNG is a pure stateless counter operation. Canonical random bits use fixed Philox4x32-10: seed, logical counter, uint32 wrap, round constants, output words and uniform conversion are identical across targets. It observes no program/thread/lane identity and depends on neither call order nor mutable provider RNG state.
