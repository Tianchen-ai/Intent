# Target extensions and lowering

## 1. Orthogonal choices

The compilation context selects these separately:

- A source provider: Triton, cuTile or a future direct backend.
- A hardware target: NVIDIA/AMD and a specific SM/gfx, or another GPU architecture.

Providers and hardware are different classifications. External Triton can compile Triton source to NVIDIA or AMD. Intent must not prebuild a vendor IR for every provider or create a dialect for each SM/gfx version.

## 2. Target capabilities

A physical module may read an external target-capability object. It expresses typed features:

- Grid/program limits.
- Warp/wave size and resource limits.
- Supported scalar/fragment dtypes.
- Structured-primitive and atomic capabilities.
- Provider source forms.
- Architecture-specific extension legality.

Passes query features rather than matching device names. Feature predicates, pipeline selection and lowering patterns on the same IR express differences between SM90, SM100, SM120 and gfx families.

## 3. Common IR and local extensions

Common GPU IR remains the sole complete executable authority. A target-local extension augments the current program without copying program mapping, value/access graphs or structured semantics.

A local extension must meet all these conditions:

1. Common IR cannot express it without loss, and making it common would be inappropriate.
2. It is more than an API name, argument order or string difference.
3. A later pass/consumer needs to read or rewrite it.
4. It needs independent legality checks and a verifier.
5. It becomes part of the current program before serialization.

Otherwise use thin legalization or direct serialization.

Compare source and provider operation contracts before designing a local form. When contracts match and a provider has a native primitive, prefer direct mapping. Intent's helper organization does not justify reclassifying or reimplementing reduction/scan trees, thread communication or layout strategies already owned by the provider. Stronger order or numerical requirements need explicit author-semantic justification; accidental restrictions in existing lowering or older documents cannot establish that necessity.

Explicit approximate math and FTZ are common numerical semantics of existing unary/binary operations, not new dialects named after provider APIs. Provider legality checks dtype and hardware capabilities; serialization mechanically emits primitives or wrappers satisfying those numerical attributes. Per-operation opt-in must not become a kernel-wide fast-math option. Cloning, bufferization and scalarization preserve both attributes. Existing lowering of ordinary operations does not change because a neighboring operation opts in.

## 4. Triton

The common GPU program maps mechanically:

- Program coordinates → `tl.program_id` and grid.
- Physical ranges/fragments → `tl.arange`, scalar/blocked tensor values.
- Explicit access/validity/fill → pointer expressions, mask/other and `tl.load/store`.
- Predicate-proven physical effective ranges → selected loop bounds, all-valid unmasked bodies and mixed-range guards.
- Gather/scatter/atomic → matching Triton operations.
- Physical reduce → `tl.reduce`, preserving selected component dtypes, identities and NaN/tie rules.
- Physical scan → `tl.associative_scan`, preserving prefix, direction and inclusive/exclusive semantics.
- Physical contract/scaled contract → `tl.dot/dot_scaled` or legal expansion satisfying the original contract.
- Physical region fold/scan → selected segment loops, summarizer bodies, summary combine, scan apply/emit and their explicit structured operations.
- Structured control → Python/Triton structured control.

Ordinary reduce already declares an associative/commutative contract. Providers need neither reprove it from the combine body nor generate an extra ordered tree or scan-plus-terminal path. Triton's default promotion, NaN rules or dot precision may differ from Intent's contract. Close those differences with selected conversions, callbacks and native precision options. Direct mapping does not mean blindly using defaults or copying internal instruction dtypes from the external compiler.

Ordinary floating multiply-add uses Triton's `enable_fp_fusion=True` under the [numerical contract](../dsl/types-numerics-and-effects.md#55-ordinary-multiply-add-fma-fusion), letting LLVM/PTX lowering form legal FMAs. Do not create another multiply-add matcher or tuning parameter. This switch is independent of arbitrary reassociation, dot input precision and FTZ; libdevice FTZ reflection stays disabled. Corresponding lowering must retain explicit numerical casts and operation-defined independent rounding boundaries; ordinary FMA enablement must not erase them.

Triton-local forms may include descriptors and descriptor accesses that affect the source program, certain provider compile-time branches, and `Config` binding. Pure pointer/descriptor spelling differences may serialize directly. If descriptor selection changes operands and static block constraints and multiple passes consume it, represent it with a local extension operation.

Tensor descriptors explicitly retain base, shape, strides, block shape, padding, alignment and accesses in the current provider program. They also declare runtime allocation size/alignment, allocator ABI and launch lifetime. Provider launchers bind the declared allocator requirement. If the selected Triton/runtime cannot satisfy it, provider legality rejects before serialization. Serializers must not invent undeclared descriptor workspace or allocator paths.

Intent does not copy external Triton's:

- TTGIR distributed encodings and `convert_layout`.
- Coalescing, thread locality and MMA/shared/TMEM representations.
- TMA lowering, software pipelines and warp specialization.
- Fences, barriers, registers and LLVM/PTX lowering.
- Autotune winner.

## 5. cuTile

cuTile and Triton share the block-program core: program/block identity, compile-time fragment extents, pointwise values, load/store/gather/scatter, reduction/prefix operations, MMA and control.

Thin mapping includes:

- Program coordinates → `ct.bid` and required grid values.
- Fragment shapes and coordinate relations → cuTile tile-space indices.
- Common accesses → `ct.load/store/gather/scatter`.
- Structured operations → cuTile-supported reduction/prefix forms such as `ct.sum`/`ct.max`/`ct.cumsum`, and `ct.mma/mma_scaled`; region fold/scan projects the segment loops, summaries and state/output flow already formed in the common program.

cuTile-local legality/forms include up to three-dimensional block identity, tile-space index multiplication, check-bounds/padding, advanced indexing, MMA-scaled layout and provider tuning constraints. They become local extensions only when they are more than mechanical index conversion.

If the selected cuTile surface lacks copy/barrier/synchronization forms needed by a common program's cooperative behavior, the provider must legalize to an existing equivalent capability or reject. It must not emit a slow serial path pretending to support the behavior.

## 6. Vendor and architecture extensions

Common IR with local extensions follows the same organizational principle as TritonGPU:

```text
common GPU program
    + optional provider/vendor extension ops
    → target-specific legalization
    → source or lower IR
```

An extension operation may be legal only with particular features; a pass may also choose different rewrite patterns from feature predicates. Architecture differences do not enter KIR or appear as kernel-name branches.

## 7. Provider verifier

After provider legalization, check:

- Grid rank, program coordinates and compile-time fragment requirements.
- A unique surface lowering for every common operation.
- Complete operands, results and regions of local extensions.
- Rejection of unsupported dtype/primitive/access/synchronization combinations before serialization.
- Complete binding of physical parameters to provider constexpr/config.
- No provider pass rereads KIR to infer shared ownership, axes, ranges, coordinate provenance, accesses or validity.

## 8. Terminal serialization

A serializer may only:

- Emit imports, signatures and decorators.
- Print values, control and operations in current-IR order.
- Mechanically convert types, attributes and provider API spelling.
- Emit declared `Config`/candidate sets and launch wrappers.

A serializer must not:

- Guess fragment width from logical result shape.
- Reconstruct pointers/masks from KIR relations.
- Choose ownership/primitives from roles or operation names.
- Create buffers, workspace, persistent loops or pipelines.
- Add undeclared physical parameters.
- Catch exceptions and switch to fallback.

## 9. Unsupported categories

Declare unsupported behavior at the earliest layer with sufficient information:

- Shared GPU legality: the physical program cannot satisfy common GPU execution constraints.
- Provider surface: the target language lacks an equivalent operation/form.
- Hardware capability: the selected architecture lacks a required dtype/resource/primitive.
- Lower compilation cost: source is legal, but the external compiler cannot complete within the given cost limit.

Do not collapse these four categories into one compilation failure or hide them by changing the algorithm, reducing scope or emitting an alternative path that is orders of magnitude slower.
