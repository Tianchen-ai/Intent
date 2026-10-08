# Intent programming model

## 1. Purpose

Intent is a structured kernel DSL. An author defines one hardware-independent logical kernel algorithm. Without changing that algorithm, the compiler constructs execution organization, value representations, accesses, storage and target-language programs.

Intent is neither a graph compiler nor a whole-operator library selector. Its compilation unit is a specialization of a kernel definition, not a model graph, Python call tree, provider template or operator name.

## 2. Targets are selected by compilation calls

Triton, cuTile, CPU and RISC-V/RVV targets are external compilation inputs. They are not DSL values, constexprs, types, function parameters or algorithm branches.

```text
Intent source + specialization + external target selection
    -> target artifact
```

Specifications may explain how semantics project to targets, but author source and canonical Kernel IR contain no provider, device model, ISA or capability identity. A canonical operation may map directly on one target, expand on another, or be rejected where it cannot be implemented. None changes its definition.

## 3. What authors express

Authors own every algorithm fact affecting observable results:

- Inputs, outputs, mutable parameters, shapes, bounds and alias semantics.
- Logical domains, source-derived subregions and indexed relations.
- Hardware-independent `if`, ordered `for/while`, unordered `parallel`, transitions and stopping conditions.
- Tensor indexing, broadcasting, shape transforms and numerical expressions.
- Complete reduce, scan, region fold/scan, contract, scaled/sparse contract and histogram semantics.
- Indexed reads/writes, collision reductions, atomics, logical buffers and deterministic RNG.
- Whether an algorithm uses one kernel or multiple kernels orchestrated by Python.

These facts make canonical Kernel IR the sole algorithm authority. The compiler may analyze and reference them, not silently substitute another algorithm.

## 4. What authors do not express

The following belong to the selected target and physical program:

- Program IDs, launch grids, blocks, warps, lanes, harts or vector lanes.
- Execution tiles, vector lengths, thread partitions and target-static block shapes.
- Registers, shared memory, TMEM, local memory and other storage levels.
- Pointer tensors, physical address widths, tail masks and neutral padding.
- Copy instructions, MMA variants, layouts, swizzles, pipelines, barriers and physical atomic scopes.
- Triton/cuTile API forms, autotune candidates and winners.
- Provider capabilities or device-model branches.

Authors can declare typed index bounds through `I.assume_in_bounds` and allocation relations through external-view `alias/noalias` annotations. There is no general Boolean precondition or optimization hint. Other relations must be expressed by types, ordinary operations or future closed typed constructs justified by real kernels.

## 5. Logical subregions are not physical tiles

A domain is an ordered logical coordinate set. A source-derived subregion is a contiguous interval obtained from source-domain bounds, input relations or algorithm metadata. It belongs to no program, block, warp, thread or register. Its runtime length need not be a power of two.

Subregions change members read by the body and therefore belong to the algorithm. Compiler-selected physical tiles do not change author tensor shapes or appear as logical subregions in KIR.

Coordinates produced by `I.indices`, subregions and indexed relations retain source identity and typed provenance through helper calls, slicing, broadcasting, reshaping and transposition. These facts let analyses prove physical all-valid/all-invalid ranges from author predicates; they do not authorize changes to logical members.

Authors may express a valid sequence prefix or `[begin_p,end_p)` for part `p`, not “128 tokens per CTA,” “16 elements per CPU step” or “current RVV VL.”

A region fold/scan source slice is not an ordinary author-created subregion and cannot escape into a value, shape or ABI. It is an internal parametric slice governed by homomorphism: every legal contiguous segmentation defines the same result. Only the physical program binds its extent. Authors consume sliced tensor components and absolute source coordinates without observing segment identity, count or extent. This grants no arbitrary resegmentation permission to ordinary loops.

## 6. Procedural and tensor semantics coexist

A kernel may combine whole-domain/subregion tensor operations, conditional SSA merges, ordered loops and carries, unordered parallel iteration, first-class collectives/contractions, indexed accesses, external views and mutable logical-buffer effects.

Tensor constructs do not erase program order; procedural control does not require machine execution units. The compiler preserves logical tensor flow together with control and effect semantics.

## 7. Observability and implementation freedom

Observable semantics include output values/shapes/dtypes; external effects, atomic ordering, collisions and aliases; ordered control, loop state and stopping; accumulation, rounding, approximation and structured reassociation; and host-visible kernel counts, call order and intermediate interfaces.

Within these contracts the compiler may introduce execution coordinates, blocking, vectorization, materialization, access forms, storage and target primitives separately for each target. GPU, CPU and RVV share algorithm facts and reusable analyses, not a renamed GPU topology.

## 8. Specification pages

- [Kernel and host](kernel-and-host.md): definitions, helpers, specialization and orchestration.
- [Logical program](logical-program.md): domains, control, state, structured operations, relations and effects.
- [Language reference](../dsl/README.md): Python DSL surface.
