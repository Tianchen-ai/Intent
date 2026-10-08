# Intent compiler

## 1. Compilation input

The compiler receives Kernel IR after specialization, surface desugaring and canonical verification. KIR is the sole author-algorithm authority. The compiler does not select named templates or automatically split a kernel into launches.

It constructs executable physical programs for the selected execution family:

```text
canonical Intent KIR
    ├─ GPU executable program
    │    ├─ Triton legalization / serialization
    │    ├─ cuTile legalization / serialization
    │    └─ provider-local extensions justified by actual differences
    └─ CPU executable program
         ├─ Mojo legalization / serialization
         └─ Weft legalization / Canonical Weft IR
```

GPU/CPU reuse KIR semantics and shape/index/effect/dependence/reuse analysis methods, not a renamed GPU topology. Mojo and Weft consume the common CPU program; RVV is a CPU capability, not another Intent execution family.

## 2. GPU compilation boundary

The pipeline produces one complete shared executable program containing program-instance spaces/workset mappings, structured control/physical loops/carries, scalar/fragment SSA/accumulation/materialization/buffer lifetimes, coordinate provenance/predicates/index sets/accesses/validity/fills/collisions/atomics, physical structured compute, and compile-time parameters directly constraining types/loops/accesses/launch.

Execution decisions exist in current IR types/operations/regions/operands/results/attributes/def-use. Analysis caches, diagnostic origins and search declarations may be separate, but never jointly explain execution with the function. Derived coordinate provenance becomes current coordinate/access mappings, not only an origin side table.

## 3. Relation to Triton's IR layers

TTIR is already an author-written GPU block program; TTGIR adds distributed encodings, layout conversions, MMA/shared/TMEM, pipelines and vendor structures. Intent KIR is higher because authors do not prewrite program IDs, static block tensors or pointer/mask programs.

Intent therefore constructs a block program as complete as high-quality provider source without copying TTGIR/vendor lower compilers:

```text
Intent KIR → shared executable GPU IR → provider source
    → external provider frontend/lower compiler
```

Common GPU IR does not contain warp/lane distribution, register/shared/TMEM layouts, MMA instructions, TMA lowering, software-pipeline schedules or machine ISA. It retains program/fragment/access/structured facts needed by provider compilers.

## 4. Terminology

- Execution family: distinct execution models requiring different KIR-to-physical construction, such as GPU and CPU. CPU providers may implement scalar/vector/optional matrix computation in one family.
- Physical program structure: program mapping, loops/control/value/access/structured graphs and resource ownership.
- Execution group: internal dispatch region co-owning connected computation/effects in one kernel body, not another kernel/artifact/launch.
- Logical ownership extent: logical instances or a contiguous range owned by a physical instance.
- Provider form: alternative representation of one operation/access, e.g. Triton pointer vs descriptor.
- Target artifact: executable entry produced from a physical specialization, launched once by the host.
- Capability/resource limit: provider/hardware operation/dtype/grid/resource facts affecting legality, not author KIR.

## 5. Providers and hardware

Triton/cuTile are source providers; NVIDIA/AMD and SM/gfx versions are orthogonal hardware targets. CPU similarly distinguishes Mojo/Weft from x86/RISC-V. Shared CPU retains tasks/blocks/structured compute; lowering can choose/instantiate expert microprograms rather than only API mapping or whole-operator bypass. Vector widths and AMX/IME affect legality under [CPU rules](cpu-program-ir.md).

Pure API-spelling differences serialize deterministically from common IR. Add extensions only for genuine target-local structures that cannot be losslessly expressed and need independent legality/multiple consumers. Extensions augment the same program, not a second leaf program.

SM90/SM100/SM120/gfx versions do not get separate IRs. Typed capabilities and feature predicates drive passes/verifiers/patterns on one physical IR; KIR has no device/provider branches.

## 6. One executable authority

KIR remains immutable during physical construction. Once conversion creates an independent family program, passes transform only that current IR. Provider lowering/serialization do not revisit KIR, shape metadata or role names to rebuild ownership/ranges/accesses/validity/workspace/control.

KIR origin supports verification, diagnostics and tracing only. Physical programs independently verify/serialize without adjacent KIR clones or decision records.

## 7. Specification pages

- [KIR to GPU](kir-to-gpu.md): authority, worksets and first complete physical program.
- [GPU program IR](gpu-program-ir.md): types, operations and invariants.
- [Passes and analyses](passes-and-analyses.md): transformations and preservation.
- [Physical parameters](physical-parameters.md): legality and tuning.
- [Target lowering](target-lowering.md): local extensions and external compilers.
- [CPU program IR](cpu-program-ir.md): task/block model, implementations and Mojo/Weft boundaries.
