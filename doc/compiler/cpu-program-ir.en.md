# CPU executable programs

## 1. Programming model and compilation boundary

CPU is a non-SIMT family constructed directly from immutable canonical KIR. Invocations contain explicit tasks owning worksets, captures and outputs. Tasks contain ordered control/shaped computation, with dependencies and joins completed before return. Task coordinates are not fixed hart/thread IDs; blocks are computation ranges, not registers/matrix tiles.

```text
canonical KIR
    → shared CPU task/block program / analyses / passes
    → implementation selection, requirement coordination and microprogram expansion
        ├─ Mojo legalization / serialization → Mojo / LLVM
        └─ Weft legalization → Canonical Weft IR → Weft compiler
```

CPU/GPU share semantics and reusable analyses at comparable structured abstraction, not GPU program/warp/lane/fragment ownership. Providers and hardware are separate: Mojo/Weft vs x86/RVV/optional matrix capabilities, without changing algorithms/dtypes/control/effects.

## 2. Common CPU IR

The CPU dialect owns typed ABI/tasks/axis-access relations/resources/numerical obligations/verifiers. It may reuse func/arith/math/scf/memref/vector/linalg, not wrap every standard arithmetic or equate standard-dialect legality with CPU legality.

Current IR independently preserves value dtypes/axes/result-broadcast relations; views/coordinates/validity/fills/access/aliases; task captures/worksets/coverage/unique writes/completion; ordered/blocking loops/bounds/tails/carries; structured operands/axes/identities/combines/accumulators/numerics; and reuse/materialization/resource ownership/initialization/lifetime/actual bindings.

Structured operations need not become scalar loops. Shared IR retains meaning and surrounding connections without enumerating microprogram decoding/register layouts/trees or equating shaped extent with SIMD width. Formats/scales come from DSL/KIR, not provider shapes/names.

After construction, only current IR is transformed. Standard/local operations and expansion do not create a second planning IR; origins/caches/selection records never jointly define execution. Family IR/analysis/transforms, provider implementation and runtime follow their own responsibilities.

## 3. Compiler and implementation authors

| Responsibility | Boundary |
|---|---|
| Algorithms, interfaces, observable numerics/formats | Author and DSL/KIR |
| Tasks, outer blocks, cross-operation fusion/reuse, access/lifetime | Shared CPU passes |
| Decode, local subblocks/accumulation/conversion/stages | Expert target implementation within block semantics |
| Surrounding supply/shared representations/resources | Implementation declares needs; CPU/provider passes coordinate |
| Machine layouts/instructions/register allocation/scheduling | Provider-local lowering or external compiler |

Parameterized microkernels may contain loops/scratch/multiple operations, need not be one instruction or expressible linewise in Intent. Experts supply complex realizations; compilers determine applicability/bind/compose rather than invent every fast structure.

Scope determines ownership: shared panel existence/lifetime belongs to surrounding coordination, private packing/fragments to implementation/lower compiler. Shared passes must not fix widths/microblocks/packing before querying requirements. Reusable vector/reduction transformations are not a compulsory lowest-common representation.

## 4. Programmable target lowering

Each implementation declares operation/region semantics and operand/result/numerical relations; applicability over dtype/format/shape/tails/access/capability/resources; required outer blocks/input forms/supply/output/scratch; finite consumed parameters/domains; and actual IR expansion with current operands/regions/bindings.

A formal transformation queries finite implementations, coordinates requirements, fixes one candidate's implementation/bindings and expands it. Planning and expansion consume the same selection; local resources/effects connect to current IR. Implementation identities dispatch compilation but never replace input semantics or select whole kernels by name.

IRBuilder/structured macros/target helpers may implement expansions. Target source goes through its frontend with explicit captures/axes/symbols/domain/outputs/lifetime, not textual reconstruction. Direct add/load mapping is also legal; not every primitive needs a microprogram.

Expanded IR continues legalization/analysis/verification. An opaque op plus runtime/emitter recipe is insufficient. Programmable expansion is not a library bypass and does not become terminal emission merely because a class says Emitter.

## 5. Access, numerical and resource invariants

ABI retains element type/rank/static-dynamic identity/offset-strides/access-alias/input format. Native lowering expands pointer/descriptor/scalar parameters accordingly. Contiguity/alignment/disjointness needs actual view facts, not hidden copies/noalias assumptions or overwritten snapshots.

Tasks derive from worksets/dependence/unique writes. Resources retain size/alignment/owner/initialization/lifetime. Forwarding/rematerialization preserves coordinates/dominance/effect order/source stability. Microprograms neither repeat shared preparation nor hide cross-call repacking/workspace/caches.

Numerics follow [language rules](../dsl/types-numerics-and-effects.md). Quantized format/scale/group/zero point/conversion/accumulation are fixed before implementation selection, not interchangeable tuning formats. Undeclared formats cannot be supported with provider names, and Weft Encoding/Level is not a public author surface.

Experts preserve adopted computation/effects; partials/widening/narrowing/FMA/reassociation use permitted freedoms only. Authors undertake implementation preservation; compilers check applicability rather than reprove arbitrary quantization algebra per compilation. Ordinary arithmetic inherits no provider fast math/FTZ; floating environment preserves semantics and caller state.

Scalar/vector/matrix cooperation does not imply separate contexts. Synchronous instructions use SSA/effects/lifetimes; async engines/queues/cross-context synchronization require real capabilities and complete dependency semantics.

## 6. Capabilities, bindings and selection

Typed capabilities describe actual consumers' dtypes/resources/memory/vector/matrix needs, not device-name matching, one f32 width or AVX/RVV as the family definition. Do not prebuild unused hardware fields.

Shared parameters constrain task grain/outer blocks/cross-block organization; implementation parameters local microblocks/vector/replica/unroll; external parameters their compiler. Each cross-boundary binding has one owner with explicit constraints, not independently selected values. Adjacent finite configurations permit overrides, not arbitrary program-tree products.

Every CPU empirical row carries five outer shared bindings, local implementation parameters and an implementation set. The set denotes actual implementations of distinct typed kinds, not priorities/failure fallbacks. Each operation needs exactly one applicable member, then original type/layout/resource/numerical/supply legality. Zero or multiple matches reject that row. Bindings use existing ImplementationAttr. One row yields at most one candidate without registry-generated combinations.

Candidates fix complete bindings and implementation for one equivalent program. Strategies or measurements choose legal candidates, not replace missing lowering. No legal implementation diagnoses; serialization/execution failure does not switch algorithms. Compiled/winner caches are separate and account for specialization/views/provider/hardware/implementation definitions/bindings.

## 7. Providers and verification

Mojo implementations form needed loops/SIMD/math/resources/native ABI, then Mojo/LLVM perform machine lowering/allocation/scheduling. Shared CPU need not start with Mojo vectors.

Weft receives canonical Weft IR. Simple structured operations convert directly; experts may instantiate Weft author programs. Adapters retain host tasks/interfaces/coordinates/control/lifetime without implicit hart/grid. RVV/IME layouts/fragments/machine packing/resources/instructions remain Weft-owned, without forced Mojo SIMD expansion.

Providers may stop at different abstraction levels. Only structures needing independent legality/later consumers justify extensions, not symmetry.

Analyses recompute coverage/axes-access/dependence/reuse/lifetime/eligibility and invalidate on relevant changes. Complete transformation groups declare preconditions/rewrites/preservation/invalidation and verify postconditions. Verifiers do not repair; serializers do not choose implementations or create loops/scratch/packing/parameters.

## 8. Artifacts and calls

Runtime binds declared resources/executes tasks and joins before host return. Initialization/compilation/loading/calling remain separate. Python/Torch algorithm substitutes or hidden invocations are forbidden.

Source/IR generation is not native callable materialization. Native timing includes per-call packing/materialization/task dispatch/synchronization, excluding compilation/tuning/loading/external output allocation. Coverage, measurements and performance thresholds do not define these semantics.
