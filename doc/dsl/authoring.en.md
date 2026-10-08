# Authoring quick reference

This is an entry point to [language constructs](core.md) and [numerical rules](types-numerics-and-effects.md), not a second semantics. MCP `search` finds concepts/APIs/diagnostics, `api` retrieves declarations and return rules, and `read` reads rules/interfaces. Declarations do not imply support on every target or measured efficiency.

Authors choose complete-callable organization subject to the task's numerics, effects and external interface. Internal kernel interfaces, logical groupings and intermediate shapes need not appear in task signatures. Actual inter-kernel tensor interfaces remain observable and preserved. The compiler constructs each declared kernel's physical program.

## From Triton algorithms to logical domains

Intent expresses Triton-style kernel algorithms in logical domains: keep independent work, algorithmic groups, local results and dependencies; delegate physical tiles, layouts and pipelines to compilation. It does not delegate entire algorithms to library selection.

- Establish inputs, outputs, members and dependencies before writing kernels. A Triton program ID may represent an observable local summary or merely assign physical tiles. Preserve the former's grouping/interface; use tensor free axes or `I.parallel` for the latter rather than reproducing every program.
- Tensor expressions retain semantics. After checking dtypes/members/numerics, `tl.sum`, `tl.dot` and `tl.associative_scan` correspond to structured reduce/contract/scan, without scalar expansion. An outer parallel does not parallelize ordinary inner loops. Structured results can feed independent points with conflict-free effects.
- Free tensor/structured-result axes already express independent coordinates. Do not create fixed logical subregions solely to reproduce BLOCK_M/BLOCK_N. Keep actual algorithm member sets, summaries and inter-kernel interfaces. A reduction axis never becomes free because tiles were omitted.
- A scalar output does not imply a whole-input single-group reduction. Parallel producers/consumers do not supply missing aggregation groups or cross-kernel merges.
- Preserve stages, intermediate tensors and host order in a multi-kernel algorithm. Expressions/helpers/local buffers are not substitutes. Authors choose grouping/intermediate interfaces; compilers do not invent missing stages.
- Consider complete-callable parallel work, reads/writes and launch costs rather than shortest source or fewest kernels. Preserve reduction/prefix/loop members, order, dtypes and numerics.

## Types, literals and shapes

Use `import intent` and `import intent.language as I`; `@intent.kernel` and `@intent.fn` are not in `I`. External annotations need dtype and shape, for example `I.In[I.f32,("M","N")]`; rank-zero views use `()`. `I.f32` is a runtime scalar dtype. Literals may instantiate contextually, but runtime operands of different dtypes require explicit `I.cast`.

Host `I.dtype("f32")` and `I.dtype("i64")` resolve canonical tokens, not Torch names such as float32/int64.

Optional third annotations use `I.constraints`, for example `I.Out[I.f32,("N",),I.constraints(noalias=True)]`. Views may alias by default. Noalias is a caller obligation that the allocation overlaps no other view, not inferred from Out. Equal `alias="group"` views share an allocation even with differing offsets/shapes or disjoint slices. Different groups do not prove disjointness. Alias and noalias cannot coexist and change no access direction/shape/dtype. Unsupported overlapping writable views are target restrictions, not implicit language noalias.

Read/write rank-zero views with `[()]`. A full slice contains one `:` per existing axis; `view[:, :]` on rank one is illegal and creates no new axis. Tensor slicing uses the same rank rule.

Without an expected dtype when first creating a runtime value, bool/int/float literals become bool/i64/f64. Later uses do not change dtype. Initialize f32 loop state explicitly, e.g. `I.cast(1.0,I.f32)`. Infinity is `I.inf` or `-I.inf`, optionally cast.

`I.exp2(...,approximate=True)`, `I.tanh(...,approximate=True)` and `I.fdiv(...,approximate=True)` opt into existing explicit approximate f32 contracts. Options are constexpr bool; exp2/fdiv allow `flush_to_zero=False`, enabled only with approximation. See [explicit approximate math](types-numerics-and-effects.md#51-explicit-approximate-math). These permissions do not change neighboring operations, accumulation or order.

Select's Boolean condition does not supply a numeric branch dtype. Both-literal branches do not derive dtype from the condition's tensor. For f32 signs use `I.cast(I.select(mask,-1.0,1.0),I.f32)`.

Tensors/views have `.shape`; scalars, tuples, records and domains do not share it. `I.full` makes immutable tensor SSA, not host workspace. Dot requires explicit acc_dtype and rank-one operands; it returns a rank-zero tensor, distinct from a scalar or `[1]`. `I.full((),value,dtype=...)` explicitly converts scalar state to a rank-zero tensor schema.

Zeros/full/read results are immutable. Assign or store only to writable views/buffers. Rebind tensor carries to new tensor values; use storage for address writes. For equal-shaped/dtyped values/delta and Boolean active, this keeps state schema unchanged:

```python
state = values
for step in I.domain(0, count):
    state = I.select(active, state + delta, state)
```

Buffer names denote mutable storage, not tensor values. Read before pointwise/reduce use, e.g. `value = state[:, :]`.

### Comparison operators

Python `== != < <= > >=` compare scalars/tensors and return bool with normal dtype/literal and trailing-broadcast rules. They are not `I.eq`-style intrinsic calls.

## Domains, indices and broadcasting

`I.domain(0,M)` is a coordinate domain, not an integer tensor or reduction-axis number. Use `I.indices` for coordinate values. Coordinates/loop variables have `I.index`, distinct from i64; cast runtime offsets/strides to index, and index to float before floating arithmetic. `x[rows,columns]` reads a logical region; axis=1 denotes the second tensor axis.

Broadcast aligns from the right. `[M]` does not automatically mean rows of `[M,N]`: reshape to `[M,1]`. `[N]` broadcasts directly. `[C]` for `[B,C,H,W]` needs `[1,C,1,1]`; variable names do not select axes. Unknown extents are not automatically compatible. Use annotation symbols or actual shapes, not literals from a coincidental current input.

Transpose permutation has exactly rank entries, each axis once; omission reverses axes. Reshape preserves row-major order, not arbitrary permutation. Domains/subregions and coordinate tensors are not interchangeable.

Domain indices form result axes in resource-index order; assignment aligns positional axes, not variable names. `output[rows,columns]=input[columns,rows]` is not a transpose for equal-length domains: explicitly transpose the tensor or construct matching coordinates. Independently dynamic subregions do not become equal because one input happens to match.

Indexing a tensor read from a subregion uses local positions starting at zero and that value's shape, not the original view's absolute coordinates.

Tensor indices broadcast jointly and pair coordinates pointwise. Two `[K]` indices of `x[r,c]` produce `[K]`, not a Cartesian product. For `[M,N]`, reshape r to `[M,1]` and c to `[1,N]`; `[M]` with `[1,N]` still compares M against N. Each domain index adds an independent axis; size-one tensor-index axes do not disappear. Assignment values must broadcast to the indexing result; destination views do not implicitly reduce rank.

Gather uses the same relation as indexing. Valid must be bool, fill has source dtype, both broadcast to but never enlarge the result shape. Invalid members yield fill without source access. Default fill is typed zero (False for bool).

## Reductions, tuples and helpers

Reduce permits associative/commutative parallel reassociation and permutations, not guaranteed source order or bits. Scan preserves each prefix's order with reassociation. Ordinary loops implement strict sequence accumulation.

Builtin reductions have no keepdim. Axis is a compile-time integer or nonempty tuple. Full reduction returns a scalar. Reshape tensor results to restore size-one axes; use full or assignment broadcast for scalars:

```python
reduced = I.reduce.sum(value, axis=1)
expanded = I.reshape(reduced, (value.shape[0], 1))
```

Cumsum of `[M,N]` on axis1 computes independent row prefixes, without flattening or cross-row state. Members come from its input, including only members of an already-read subregion. Scan/cummax use the same axis rule.

Arg-reduce max returns `(values,indices)`, not a lone index. Both use retained shape or scalars for full reduction. Index dtype is index and positions are relative to the reduced tensor axis, not source absolute coordinates. Cast explicitly before an i64 output even though both use 64 bits.

Generic reduce identity/combine arguments/result share the schema after deleting reduced axes. An `[M,N]` axis1 identity can be `I.full((M,),0.0,dtype=I.f32)`. Builtin accumulator conversions happen before combine, so external storage need not match accumulation.

Tuples/records permit components with different dtypes/shapes but each component matches its identity/combine/result. Tuples destructure statically, records use fields. They do not return host-visible kernel values.

Helpers are typed DSL calls, not arbitrary Python abs/math. Use declared `I.abs/sqrt/asin/lgamma/log1p/erfc/i0` and query unknown names rather than invent keepdim. Runtime captures are explicit; structured combines are pure. Helpers may return no value; natural end and bare return retain effects. Value returns must share schema across runtime branches.

## Control, effects and multiple kernels

Ordinary loops preserve ordered carries whose initialization/update dtype/rank/provable logical shape agree. Broadcast in a body does not enlarge carry schema. Initialize tensor state with full, not scalar cast. Parallel forbids carries. Tensor predicates select values, not statement branches. Out is undefined until written; InOut does not repair undefined reads.

Parallel nesting may capture outer immutable tensors. Per-outer-iteration values remain private logical instances, not globally shared mutable state.

A name used after if must be defined before it or in every continuing branch. Logical implications between separate conditions do not define a name; initialize common state first.

Kernel/helper range is positive-step domain shorthand. Loop variables remain index even for constant bounds. Reverse traversal uses forward ordinals and reverse-coordinate arithmetic.

Mask equals select and does not suppress writes. Masking destination coordinates redirects writes to fill addresses. Store/assignment have no mask/valid arguments. Use scalar structured conditions or actual participating domains/subregions. Parallel writes still need nonconflicting in-bounds addresses:

```python
for i in I.parallel(I.domain(0, values.shape[0])):
    if active[i]:
        out[destinations[i]] = values[i]
```

A kernel is not automatically split into launches. Hosts compile/call multiple kernels and own intermediates/lifetimes. Logical grouping counts/bounds/shapes are allowed and not hardware blocks. Local buffers never cross kernels; pass intermediate tensors as In/Out/InOut views.

### Host compilation and calls

`intent.compile(kernel,compiler=...,target=...,constexprs=...)` returns an artifact. `artifact(...)` receives every runtime parameter in signature order, including Out. `artifact.run(...)` omits only Out. For `A:In,B:Out,scale:f32`, call `(A,B,scale)` or `.run(A,scale)`. Constexprs are bound at compile time and omitted at runtime.

Run/prepared result return only declared Out: none→None, one→object, several→ordered tuple. InOut stays caller-owned and is not returned. Direct call/launch/prepared launch execute and return None. Prepare binds explicit outputs in Out order or allocates them, without execution. Result retrieves containers without synchronization; CPU joins/GPU stream/MLU queue retain their runtime completion rules.

Ordinary host loops/ifs may call compiled artifacts repeatedly with varying runtime scalars without recompilation; this differs from kernel control. Generate only produces source/IR. Target is selected on the host, e.g. TritonTarget, never queried inside kernels.

Scalar constexpr annotations use Python types (`I.Constexpr[int/float/bool]`), not runtime dtype tokens. Runtime values do not become constexpr because a call happens to be fixed. View static extents are nonnegative integers; strings are valid symbols, not stringified numbers. Annotation symbols do not bind Python names: read `M,K=input.shape`. Dynamic-shape-only kernels need no constexpr shape bindings. Equal symbols require equal call extents; distinct symbols do not declare equality.

Evaluation `build(context)` is a thin adapter: compile definitions individually and return a host callable allocating/calling them. It binds declared constexprs and does not constrain a task to one kernel.

## Diagnostics

For unknown intrinsics check declarations; integer-axis errors often mean a domain was passed; dtype errors need casts; shape errors need true trailing alignment/index relations. Do not repair tasks by changing output shape, dropping tuple components or loosening tolerances.

Legal expressions rejected by lowering are implementation gaps. Invalid compiler IR (binary schema or tuple type errors) requires preserving the complete source/diagnostic for compiler contributors, not changing language rules. MCP supplies rules/declarations without judging numerical correctness.
