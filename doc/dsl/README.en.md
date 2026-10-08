# Intent DSL

## 1. Language surface

Intent uses a restricted Python AST to express hardware-independent kernel algorithms. Python supplies syntax; canonical Kernel IR defines semantics, not arbitrary Python execution. Targets are selected by compilation calls, not DSL values, types, constexprs or control. This specification defines the author surface independently of implementation coverage and contains no progress or performance status.

## 2. Construct ownership

### Canonical algorithm semantics

Canonical KIR uniquely represents kernel interfaces/runtime/constexpr parameters/helpers; domains/source subregions/logical indices/compositional provenance/relations; structured control/parallel/carries; typed tensor operations including aggregates and shape transforms; reduce/scan/region operations/contractions/histograms; external/buffer access, scatter and atomics; and stateless counter RNG.

`I.Enum` is constexpr-only and normalizes to a closed canonical enum at specialization, never a runtime ABI value.

### Named computations and shorthand

Familiar named computations are author entry points, while structured constructs and `@intent.fn` support composition. Public names need not be canonical operations. These normalize completely and mechanically to the single canonical path:

- dot/matvec/vecmat/matmul/outer/scaled_matmul/sparse_matmul.
- reduce.sum/max/any/all, cumsum/cummax, arg_reduce.max.
- Python range/slicing/conditional expressions/break/continue.
- Zeros, pointwise helpers and value masks.
- Indices/endpoints/ragged/members relation helpers.
- Ordinary indexing/assignment and gather/scatter conveniences.
- Format-specific sparse spelling such as sparse_contract_2to4.

### Author libraries

Softmax, logsumexp, Welford, attention, normalization, MoE and quantized GEMM are author algorithms built from primitives, helpers and explicit multiple kernels. They are not language intrinsics or whole-operator templates selected by the compiler.

### Outside the DSL

There is no `auto("TILE")`, state_stream, partition, public ordered marker or dedicated RaggedOp/MembersOp. Program/grid/block/warp/thread/lane/hart identity, physical tiles/vector widths/storage/layout/padding/pointers/copy/pipelines/barriers/atomic scope, provider/device/ISA branches and autotune winners are physical concerns.

Their algorithmic capabilities remain expressible: reduce/scan/region operations/ordered loops cover aggregation and recurrence; ordinary parts/boundaries/subregions/intermediates cover observable partitioning; offsets/relations cover ragged data; indexed read→SSA→write covers logical copy. Physical programs construct blocking, storage, transfer, scope and execution mappings.

## 3. Criteria for canonical operations

Remove target spelling, physical participants, tiles, storage and instructions to state logical values/control/effects. Try existing constructs. If expansion is mechanical and preserves results, order, shape, effects, aliases, numerics and interfaces, no dedicated canonical operation is needed. If expansion chooses a tree, materialization, snapshot, collision behavior or format decoding, that structured meaning must remain first class.

GPU/CPU/RVV test semantic validity; target primitive names do not define public operation existence. “A scalar loop can simulate it” is not mechanical expansion if it loses associativity, prefix meaning, paired axes, metadata interpretation, collisions or atomicity and forces the compiler to rediscover them.

## 4. Pages

- [Language constructs](core.md).
- [Types, numerics and effects](types-numerics-and-effects.md).
- [Specification examples](examples/README.md).
