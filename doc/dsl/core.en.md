# Language constructs

## 1. Definitions

```python
@intent.kernel
def kernel(...):
    ...

@intent.fn
def helper(...):
    ...
```

`@intent.kernel` defines one logical kernel computation; an external compilation call selects its target. `@intent.fn` defines a typed helper inside a kernel, without creating hidden kernels or host dispatch.

A helper may return scalars, tensors, tuples or records and have effects permitted at its call site. Runtime captures must be explicit parameters; only immutable `Constexpr` values may be captured lexically. Recursion, target queries and kernel launches are illegal.

Helper arguments follow the public Python signature, including named and keyword-only parameters. Expressions evaluate in source order, not reordered parameter order; each lowers once before binding typed values. Named computations normalize identically in ordinary and structured pure helpers; pure sites check actual body effects.

Helpers do not turn logical coordinates into integer tensors without provenance. Components passed or returned through tensors, tuples and records retain source identity, axis mapping and typed coordinate expressions when they originate from `I.indices`, subregions or indexed relations. Inlining must not change these facts.

## 2. Parameters

```python
x: I.In[I.f16, ("M", "N")]
y: I.Out[I.f16, ("M", "N")]
state: I.InOut[I.f32, ("M",)]
scale: I.f32
CAUSAL: I.Constexpr[bool]
```

- `In` is read-only.
- `Out` is unreadable on kernel entry; the author must define its contents before reading.
- `InOut` retains input contents and permits writes.
- Scalars annotated with a dtype are runtime scalars.
- `Constexpr` selects only hardware-independent algorithm branches.

`I.Enum` is a user-defined finite, nonempty, closed named constexpr type. Member names and compile-time enumeration values are unique within the type. Values may appear only as `Constexpr[EnumType]` parameters, constexpr defaults, branch conditions or helper constexpr captures. An enum is not a runtime scalar or tensor dtype and does not enter external-view/logical-buffer ABI. Different enum types do not compare or convert implicitly, and they do not inherit Python `IntEnum` integer arithmetic semantics.

## 3. Domains and source-derived subregions

```python
rows = I.domain(0, M)
columns = I.domain(0, N)
value = x[rows, columns]
y[rows, columns] = value * scale
window = columns[begin:end]
```

`domain(begin,end,step=1)` is half-open with positive step. Reverse traversal uses forward ordinals and explicit reverse coordinates:

```python
for ordinal in I.domain(0, n):
    i = n - 1 - ordinal
    ...
```

`axis[begin:end]` forms a source-derived contiguous subregion, retaining source axis, bounds, empty/tail behavior and provenance without implicit clamping.

Iteration yields source logical coordinates, as does `I.indices(subregion)`. `I.indices` produces a logical `index` tensor whose values are source coordinates; its canonical relation retains source identity, source axis and member mapping. Helpers, slicing and shape transformations must not retain only plain `i32/i64` values. Noncontiguous, repeated or reordered members use indexed relations.

## 4. Tensor values and shape transforms

Scalar and size-one pointwise broadcast normalizes to explicit relations. Dynamic extents retain identity/equality conditions, not automatic compatibility.

Unary elementwise math includes `I.exp(value)`, `I.log(value)`, `I.log1p(value)`, `I.lgamma(value)`, `I.sin(value)`, `I.asin(value)`, `I.cos(value)`, `I.floor(value)`, `I.erf(value)`, `I.erfc(value)`, `I.i0(value)`, `I.rsqrt(value)`, `I.sqrt(value)`, `I.sigmoid(value)` and `I.abs(value)`. `I.erf` is the error function. Floating behavior and special values follow the numerical chapter. `I.exp2` and `I.tanh` with additional numerical options are described below.

`I.lgamma(value)` computes the natural logarithm `log|Gamma(value)|`. It accepts floating scalars/tensors, preserves dtype and shape, and is an ordinary pure unary operation.

`I.asin(value)` computes the principal arcsine in radians. It accepts floating scalars/tensors and preserves dtype and shape. It is a pure math-library entry, without requiring authors to expand an approximation.

`I.log1p(value)` computes `log(1 + value)`; `I.erfc(value)` computes the complementary error function; `I.i0(value)` computes the modified Bessel function of the first kind, order zero. They accept floating scalars/tensors, preserve dtype/shape and normalize to the same pure unary-operation family. They are independent math-library operations, not ordinary add/subtract expansions or author-written approximations. Special values and accuracy follow the [numerical contracts](types-numerics-and-effects.md).

Floating division has the named entry `I.fdiv(lhs, rhs, *, approximate=False, flush_to_zero=False)`, whose default matches ordinary `/`. `I.exp2(value, *, approximate=False, flush_to_zero=False)` and `I.tanh(value, *, approximate=False)` allow explicit per-operation approximate math. Options are hardware-independent constexpr bools. Non-default modes accept only `f32`, and FTZ requires `approximate=True`. They normalize to numerical attributes of the same canonical binary/unary operations, not another algorithm, a global fast-math environment or target queries. Accuracy and special values follow the numerical chapter.

`reshape` preserves logical row-major element order and total element count. `transpose/permute` requires a permutation equal in length to input rank, containing every axis in `0..rank-1` exactly once. Omitted `transpose` permutation reverses all axes.

Ranked tensors and external views expose `.shape`, a logical-extent tuple retaining dynamic-extent identity. Scalars, tuples, records and domains/subregions have no common `.shape`. Shapes may be used in shape arithmetic, domains, shape transforms and tensor construction; they are not physical fragment or block shapes.

```python
mask = I.full(scores.shape, fill=True, dtype=I.bool)
zeros = I.full((M, N), fill=0.0, dtype=I.f32)
```

`I.full(shape, fill, dtype=...)` produces an effect-free, alias-free ranked tensor SSA value. Each shape dimension is a nonnegative logical extent, either static or an identity-preserving runtime shape value. Scalar fill is instantiated to explicit dtype and broadcast to all logical elements. Empty shape `()` creates a rank-zero tensor, useful for explicitly aligning scalar and rank-zero tensor branch/carry schemas. It neither allocates logical buffers nor initializes external views and carries no storage/layout/padding. `I.zeros(shape,dtype)` is shorthand for `I.full(shape, fill=0, dtype=dtype)`.

`join` stacks two values with the same dtype and shape along a new trailing logical axis:

```python
pair = I.join(lhs, rhs)

# pair[..., 0] == lhs[...]
# pair[..., 1] == rhs[...]
# pair.shape == lhs.shape + (2,)
```

It is neither a record nor concatenate nor general interleave.

Python tuple literals and tuple-valued helper results construct fixed-length, positional typed products. `I.record(field=value, ...)` constructs a nonempty typed named product with unique field names and fixed field order. Both are immutable SSA aggregates:

- Components may be scalars, ranked tensors, tuples or nested records with different dtypes, ranks and shapes.
- Tuples use static position selection/destructuring; records use static `.field` selection.
- Type identity contains tuple component order or record names, field order and per-field types.
- They may be helper values, loop carries and structured-operation accumulators/identities.
- They do not directly become public kernel runtime parameters, external-view elements or host-visible returns. Cross-kernel state uses explicit views/scalars.

Tuples/records have no common `.shape` or `.dtype`; select a tensor component before taking its shape. They differ from `join`, which creates a ranked tensor with one element dtype and a new logical axis.

## 5. Unordered parallel iteration

```python
for row in I.parallel(I.domain(0, M)):
    y[row] = f(x[row])
```

Parallel is unordered forall, exactly once per logical point, with no carries or conflicting non-atomic/non-reduction effects. It specifies no execution level. Whole-tensor assignment with unique writes needs no extra parallel wrapper.

## 6. Ordered control and carries

Ordinary Python if/for/while lower to structured control. Domain iteration is ordered; SSA updates are carries. Initialization, region arguments and updates preserve dtype/rank/provably equal logical shapes componentwise. Body broadcasting cannot enlarge the initial schema.

```python
state = initial
for index in I.domain(0, N):
    state = transition(state, x[index])
```

There is no public ordered marker. Break/continue have ordinary structured meaning; tensor predicates use select. No arbitrary resegmentable state_stream exists: element summaries use reduce/scan, region tensor summarization uses region fold/scan, strict sequence or dynamic stopping uses loops.

## 7. Generic reduce

```python
@intent.fn
def combine(lhs, rhs):
    return lhs + rhs

result = I.reduce(value, axis=axis,
                  identity=I.cast(0, I.f32), combine=combine)
```

Sources and accumulators may be scalars, tensors, tuples or typed records. A source component contains the reduced axes; removing them gives the component shape of accumulator, identity, combine parameters and result. Combine:

- Accepts two groups with the accumulator's schema.
- Returns that same schema.
- Is a typed pure helper.
- Has runtime captures as explicit operands.
- Contains no reads, writes, atomics, RNG or buffer mutation.

Choosing reduce declares associativity, commutativity and neutral identity. Elements may be reassociated/permuted, not a strict left fold or source-order guarantee. Floating-point tree differences follow the numerical chapter. The compiler checks schema/purity/effects rather than reproving arbitrary algebra. Empty reductions return componentwise identity.

Generic reduce has no hidden accumulator dtype independent of source components. Builtin acc_dtype/widening performs defined source conversions before constructing reduce; external input storage need not match accumulation.

```python
I.reduce.sum(value, *, axis, acc_dtype=None)
I.reduce.max(value, *, axis, acc_dtype=None)
I.reduce.any(value, *, axis)
I.reduce.all(value, *, axis)
```

Builtins accept ranked tensors and compile-time integer/nonempty-tuple axes, normalizing negatives. Sum identity is additive zero; max identity is result minimum or negative infinity; bool any/all use false/true. Builtins fix combine/identity, not caller-supplied generic parameters. Widening and NaN rules are in the numerical chapter. Empty reductions return identity.

Arg-reduce max normalizes to generic reduce with lowest-logical-index ties. It returns `(values,indices)` with reduced axes removed, scalars when none remain. Index dtype is index; positions start at0 in the input tensor's reduced axis, not absolute source/view coordinates.

## 8. Scan

```python
prefix = I.scan(value, axis=axis, identity=identity, combine=combine,
                inclusive=True, reverse=False)
```

Scan uses typed pure schemas and associativity without commutativity. It defines logical prefixes with order-preserving reassociation, not permutations. Inclusive/exclusive and direction are semantics. Strict recurrences use ordinary loops.

```python
I.cumsum(value, *, axis, inclusive=True, reverse=False, acc_dtype=None)
I.cummax(value, *, axis, inclusive=True, reverse=False, acc_dtype=None)
```

Inputs are ranked tensors with one explicit possibly negative axis. Results keep shape and values only. Cumsum uses sum's widening and zero; cummax uses propagating maximum/minimum identity and retains dtype by default. Explicit acc_dtype uses structured conversions. Both normalize to scan; empty input produces an empty tensor. Defaults are forward inclusive, not strict left fold.

## 9. Region fold and region scan

### 9.1 Region fold

```python
summary = I.region_fold(
    source=(keys, values, key_coordinates), axis=0,
    summarize=summarize_chunk, combine=merge_summaries,
    identity=empty_summary,
    operands=(queries, query_coordinates, scale))
```

Region fold defines a list homomorphism over one ordered source axis. A nonempty source may be split into any number of ordered contiguous nonempty slices covering it exactly. Extents are not DSL values/KIR parameters. Source components have one equal extent on axis and use identical boundaries. Explicit operands are unsliced captures; runtime captures are KIR operands, constexpr captures are specialization facts.

`summarize`:

- Receives each source component's current slice, followed by `operands`.
- Returns a fixed typed summary schema.
- May call pure pointwise operations, reduce, scan, contract, scaled/sparse contract and other pure helpers.
- May not write external views or logical buffers, scatter, perform atomics/RNG or depend on invocation count.
- May not read segment ordinal, segment count, chosen extent or chunk-relative coordinates.

For logical coordinates, pass `I.indices(source_axis)` as a source component. Its sliced components still retain absolute source coordinates.

Combine accepts two summaries, returns the same schema and shares that schema with identity. Authors declare:

```text
summarize(A ++ B) == combine(summarize(A), summarize(B))
combine(identity, x) == combine(x, identity) == x
```

Adjacent slices preserve order; parenthesization may vary, permutation may not. Empty source and physical padding safely use identity. NaN-producing sentinels are not identities; use explicit validity if needed. The operation lowers to segmentation, local computation and merging; nested structured operations retain their own lowering semantics.

### 9.2 Region scan

```python
outputs, final_state = I.region_scan(
    source=source, axis=0, summarize=summarize_transition,
    combine=compose_transitions, identity=identity_transition,
    initial_state=initial_state, apply=apply_transition,
    emit=emit_slice, operands=captures)
```

`region_scan` adds incoming-state application and slice output to the same homomorphism. `summarize`, `combine`, `apply` and `emit` are typed pure helpers:

- `summarize(slice, captures) -> Transition`.
- `combine(lhs, rhs) -> Transition`, composing in source order.
- `apply(prefix_transition, initial_state) -> incoming_state`.
- `emit(slice, incoming_state, captures) -> output_slice`.

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

Emit's result axis keeps the slice member relation. Outputs reassemble in source order; final state is apply(summarize(full_source),initial_state). Internal boundaries/prefixes/counts cannot be observed. Host-visible chunk states or fixed chunk ABI require explicit chunk domains/subregions/ordinary scan.

Ordinary scan is the element-summary/output case; degenerate region forms canonicalize to it. Region scan is not an effectful arbitrary state loop. [FlashAttention](https://github.com/Tianchen-ai/Intent/blob/main/doc/dsl/examples/flash_attention.py) explicitly writes QK/PV contractions rather than relying on pattern recognition. [Causal linear attention](https://github.com/Tianchen-ai/Intent/blob/main/doc/dsl/examples/causal_linear_attention.py) has unobservable legal segmentation; [Mamba](https://github.com/Tianchen-ai/Intent/blob/main/doc/dsl/examples/mamba_state_passing.py) exposes chunk ABI and therefore uses ordered carry.

## 10. Named multiplication and contract

### 10.1 Author entry points

```python
I.dot(lhs, rhs, *, acc_dtype)
I.matvec(matrix, vector, *, acc_dtype, transpose=False)
I.vecmat(vector, matrix, *, acc_dtype, transpose=False)
I.matmul(lhs, rhs, *, acc_dtype, transpose_lhs=False, transpose_rhs=False)
I.outer(lhs, rhs)
```

| Operation | Core input shapes | Result |
|---|---|---|
| dot | `[K]`, `[K]`, both rank1 | rank-zero tensor `[]` |
| matvec | `[...,M,K]`, `[...,K]` | `[...,M]` |
| vecmat | `[...,K]`, `[...,K,N]` | `[...,N]` |
| matmul | `[...,M,K]`, `[...,K,N]` | `[...,M,N]` |
| outer | `[M]`, `[N]`, both rank1 | `[M,N]` |

Matrix final two/vector final one axes are core; leading axes are batch. Matrix batch axes broadcast from the right; K must match without reduction-axis broadcast. Transpose swaps the matrix's last two axes only. Matmul inputs are at least rank2, without implicit vector promotion/result squeezing. Results list broadcast batch then M/N.

Dimensions describe post-transpose computation. Lhs `[64,32]`, rhs `[64,16]`, transpose_lhs=True gives `[32,16]`; transposed matvec on `[64,32]` needs vector64 and yields32. Matching remains required.

Multiplication operands share one numeric dtype; explicit casts handle mixed operands. Accumulator/result dtype is independent and explicit. No implicit conjugation, TF32, alpha/beta or mutable C initialization exists. Empty K yields accumulation zero. Frontend transpose/broadcast/paired axes normalize to contract retaining dynamic identity and runtime conditions, independent of provider matrix ranks.

Outer broadcasts ordinary multiplication, preserving dtype/pointwise numerics, without an empty-reduction contract or accumulator.

### 10.2 Generic contract

```python
acc = I.contract(a[m,k], b[k,n], reduce=((1,0),), batch=(), acc_dtype=I.f32)
```

Contract defines binary paired-axis multiply-add contraction:

- `reduce` is a nonempty list of unique, extent-compatible lhs/rhs axis pairs.
- `batch` has unique, extent-compatible pairs disjoint from reduction. Each pair appears once in the result through its lhs-axis representative.
- Lhs axes outside `reduce` become result axes in lhs order, including lhs representatives of batch pairs.
- Rhs axes outside `reduce` and not rhs members of batch pairs follow in rhs order.
- Multiply is fixed numerical multiplication and combine is fixed addition.
- Accumulator dtype is explicit.

The `reduce` list is nonempty, but a logical reduction extent may be zero; that output element is additive zero in accumulator dtype. Scaled and sparse contracts follow the same rule.

There are no pretend multiply/combine surface arguments. Other semirings use pointwise + generic reduce; outer products use broadcast multiply. Axis permutations, flattening into M/K/N, MMA and staging belong to compilation.

## 11. Scaled contract

```python
I.scaled_matmul(lhs, lhs_scale, rhs, rhs_scale, *,
                lhs_format, rhs_format, group_size, acc_dtype)
```

This shorthand uses the same closed positional schema, fixed reduce, empty batch and equal group sizes as:

```python
acc = I.scaled_contract(
    lhs_mgc, lhs_scale_mg, rhs_gcn, rhs_scale_ng,
    lhs_format=I.e2m1, rhs_format=I.e4m3,
    lhs_group_size=32, rhs_group_size=32,
    reduce=((1,0),(2,1)), batch=(), acc_dtype=I.f32)
```

Scaled contract is a first-class local tensor operation. Its closed positional schema states scale-axis relations explicitly rather than asking a shared pass to infer them from rank, shape or nearby indexing:

- Lhs carrier is `[M,G,C_lhs]`, lhs scale is `[M,G]`.
- Rhs carrier is `[G,C_rhs,N]`, rhs scale is `[N,G]`.
- `reduce` is fixed to `((1,0),(2,1))`, batch is empty, and result is `[M,N]`.
- Both group sizes agree; `G` is the same logical scale-group axis. Each format's logical elements per carrier and group size uniquely determine `C_lhs/C_rhs`.

G, formats, packing, scale relations, accumulation and rounding are algorithm meaning. Native scaled MMA/layout/storage/provider K flattening are lowering. E2M1/E4M3/E8M0 encodings/special values are in the numerical chapter. Other ranks/orders/batches explicitly normalize through ordinary transforms/control, not another schema. Packed INT4/INT2 uses explicit carrier decode/sign extension/zero point/scale and ordinary contract.

### 11.1 Closed quantized operations

`I.quantize(values, format=I.quant.q8_k)` and `I.quantized_dot(lhs, rhs, lhs_format=I.quant.q4_k, rhs_format=I.quant.q8_k, acc_dtype=I.f32)` are independent pure structured operations, not normalized to scaled contract. See [quantized operations](quantized-operations.md) for complete shapes, record mappings, quantization and dot-product numerical contracts. Formats do not select target implementations; ordinary author control may share quantized preparation intermediates.

## 12. Sparse contract

```python
I.sparse_matmul(compressed, metadata, rhs, *, format, acc_dtype)
I.sparse_contract(compressed, metadata, dense_rhs,
                  format=I.sparse.two_of_four(...),
                  reduce=((1,0),), batch=(), acc_dtype=I.f32)
```

Named sparse matmul takes rank2 data with lhs K compression and supplies fixed reduce/empty batch; format/metadata remain explicit. Formats are closed typed schemas, not provider strings/opaque descriptors. Each defines group/nonzero count/compressed ordering/logical position mapping/validity/dtype.

One_of_two and two_of_four logical metadata/order are defined in the numerical chapter. New formats require concrete semantics. Authors interpret external packed metadata; native metadata packing/storage/sparse MMA belongs to lowering. Format-specific shorthand shares the canonical path. Sparse contract retains ordinary contraction reduction/batch/free/result/accumulator rules, replacing one operand's value relation along its compression axis.

## 13. Histogram

```python
counts = I.histogram(values, bins=BINS, valid=valid, count_dtype=I.u32)
```

Histogram is a pure value-producing structured operation. Each active value must satisfy `0 <= value < bins` and contributes once to its bin. To ignore out-of-range values, incorporate their range condition into `valid`. Empty input returns zeros; count overflow follows count-dtype integer semantics.

`values` is an integer tensor and `bins` a positive logical index; result is a count tensor with shape `(bins,)`. `valid` is a Boolean scalar or Boolean tensor broadcastable to `values.shape`. A lane with `valid=False` does not test whether its value is in range and contributes no count.

This is not equivalent to author-written external-buffer initialization and atomic updates, which already select an effectful lowering.

## 14. Ordinary ragged composition

```python
member_source = I.domain(0, R)
members = member_source[offsets[group]:offsets[group+1]]
logical_members = mapping[members] if HAS_MAPPING else I.indices(members)
```

Offsets/subregions/optional mapping with SSA provenance fully express ragged semantics. Ragged/members helpers may expand mechanically but create no canonical dedicated operations.

## 15. Indexed access, buffers and copy

Ordinary indexing and explicit gather share typed index relations and active validity. Relations retain source identity/rank, result logical axes and each source axis's coordinate expression. Coordinates may come from constants, domain/subregion coordinates or data-derived integers. Relation composition retains SSA dependencies and provenance, not only result shape. Invalid reads avoid source and return an explicit same-dtype fill broadcastable to result shape; invalid writes have no effects.

Canonical operations distinguish:

- Pure tensor gather.
- External-view load/store.
- Logical-buffer load/store.
- Arbitrary-index unique store.
- `scatter_reduce`.

Unique stores require a provably injective destination relation. `scatter_reduce` retains typed combine and collision semantics.

Logical buffers are mutable kernel-local state with author shape/dtype/initialization/order and compiler placement. Read→immutable SSA→write fully specifies snapshots/effects without canonical copy. Providers may form bulk/vector/async/DMA transfers from these relations.

## 16. Atomics

```python
value = I.atomic.load(address, order="acquire")
I.atomic.store(address, value, order="release")
old = I.atomic.add(address, delta, order="relaxed")
result = I.atomic.compare_exchange(address, expected, desired, order="acq_rel")
# result.old_value, result.success
```

Canonical operations are:

- `atomic_load`.
- `atomic_store`.
- `atomic_rmw(kind=exchange/add/max/min/and/or/xor)`.
- `atomic_compare_exchange`, returning `old_value` and `success`.

Load permits only `relaxed/acquire`; store only `relaxed/release`; RMW/CAS permit `relaxed/acquire/release/acq_rel`. CAS returns typed record `{old_value, success}`. Failure order follows success order mechanically: `relaxed -> relaxed`, `acquire -> acquire`, `release -> relaxed`, `acq_rel -> acquire`.

No author scope exists. Allocation identity, alias/index relation and invocation determine logical participants; providers choose physical scope.

## 17. RNG

```python
bits = I.random.bits(seed, logical_counter)
uniform = I.random.uniform(seed, logical_counter, dtype=I.f32)
```

Canonical Philox4x32-10 maps scalar counters via block=counter//4, word=counter%4 to one output word. The numerical chapter fixes rounds/constants/conversion. It depends on no execution IDs/call ordering/mutable provider state. Composite distributions such as normal are author helpers.

## 18. Logical parts without partition

Observable part counts/identities/bounds/intermediate interfaces use domains/subregions/arithmetic/control. Empty/tail preserve member contracts. Logical parts specify no tiles/blocks/extra launches; unobservable intra-kernel blocking belongs to compilation. Partition(auto/count/extent) is neither public nor canonical.

## 19. Surface ownership

| Family | Canonical meaning | Shorthand | Excluded |
|---|---|---|---|
| Definitions/interface | kernel/helper/views/runtime/constexpr | decorators/types | target/hidden launch/provider dispatch |
| Domains/control | domains/subregions/control/parallel/carry | slices/indices/break/continue | auto/partition/state_stream/ordered/IDs |
| Tensor values | arithmetic/select/transforms/products/full/casts | zeros/activations/mask | physical tile/layout/padding |
| Structured compute | reduce/scan/region/contractions/quantize/quantized dot/histogram | named computations/prefixes/formats | whole-operator softmax/attention/MoE |
| Relations | subregions/index relations/sparse schema | ragged/member/index helpers | native metadata/MMA hints |
| Memory/effects | reads/writes/scatters/atomics/Philox | indexing/atomic conveniences | scope/storage/copy/barriers/pipelines |

Shorthand normalizes to one canonical path; physical facts never flow back into source through shorthand.
