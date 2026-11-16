# Execution models and source components

## Identity and lowering ownership

Device identity, execution model and compiler provider are independent. A device describes an actual CPU or accelerator; a model defines the execution meaning of current IR; a provider chooses its hand-off stage and owns native compilation and invocation. GPU, CPU and DSA directory placement does not force a particular lowering depth.

Canonical KIR preserves the author algorithm. Block represents program instances, ownership, shaped SSA, accesses, control, accumulation and dependencies. CPU represents tasks, captures, stage dependencies, joins and task computations. Explicit DSA execution represents participants, storage, transfers and completion. Lowering transforms the current program, without leaving multiple jointly interpreted execution plans.

Triton and cuTile consume GPU-mapped Block. Triton–Ascend takes Block into its NPU compiler without BANG C storage realization. BANG C consumes explicit DSA storage and engine realization. Mojo may consume lower-level CPU loops and vectors; Triton–Weft receives Canonical Weft `.weftir` through its official JIT. Providers need not consume the same lowering depth.

## CPU invariants

A task is not a fixed thread or hart, and block size is not SIMD width. Captures, output ownership, atomic effects, stage dependencies and joins are explicit. Dynamic shapes, strides, shared preparation and ordered control retain their semantics.

SSA values are distinguished from state requiring storage. Implementations declare representation, packing, local-block and scratch requirements; materialization passes produce storage and supply. Private/shared storage has ownership, initialization and lifetime. Fixed SIMD, RVV and matrix extensions participate through capability and implementation interfaces, rather than one width and global matrix booleans.

Hardware capabilities, affinity, thread budgets and workset budgets are separate. Each invocation has one scheduling owner, avoiding duplicate dispatch through Intent and provider pools. CPU calls complete before returning.

## DSA invariants

The computation layer retains Block values, control and accesses without forcing all values into on-chip buffers. The explicit execution layer represents participant groups, storage capacity/alignment/visibility/lifetime, transfers, compute, waits and dependencies actually decided by Intent.

Synchronization is defined by dependencies and scope, not vendor barrier spelling. Asynchronous transfers have explicit completion boundaries; consumers and storage reuse respect them. Current IR identifies sharing participants and visibility.

NRAM, WRAM, SRAM, memory cores, Union and fixed group sizes belong to BANG C. Cube, Vector and Ascend pipelines belong to Triton–Ascend or its target extensions. Physical capacity, SDK reservations and candidate budgets are separate; missing facts must not be fabricated. Storage or micro-program restrictions must not become source-language restrictions.

## Component contract

A source component registers device recognition, local discovery, specialization, compiler options, input model, pipeline, target dialects/passes, configuration resources, artifacts, binding and launch. Declaration queries neither load SDKs nor probe devices. Extension uses registration and rebuilding, without a stable binary plugin ABI.

The public driver invokes stage interfaces without provider-name-based resource or ABI reconstruction. Distribution conflicts are diagnosed through installation environments, never `sys.path` rewriting, global driver replacement or hidden subprocess routing.

Passes use MLIR pass, analysis, interface and pipeline mechanisms. Required lowering is separate from optional optimization; disabling optimization must leave a legal compilation route. Components define input stages, applicability, complete transformations and postconditions. Coordinate, effect, dependency and lifetime analyses can be shared while rewrites remain target-specific. Tuning configurations are finite, complete empirical candidates.

## Local JIT and extension

`intent.jit(definition, backend=...)` explicitly selects the provider; inputs identify the local device. Discovery supplies facts, compiler passes choose physical structure, and runtime binds resources and selects declared candidates. `prepare` neither executes nor tunes; `compile` compiles native code; `launch` executes; `result` returns output containers. Artifact caches, buffer bindings and tuning selection are separate. Hot calls do not repeat discovery or source generation.

Authors control algorithms, multi-kernel ordering and workspace. GPU/NPU calls retain stream semantics. Bench owns environments, inputs, correctness, comparisons and timing, without supplying missing product compilation or binding.

Agents normally add capabilities, lowering or optimization within the owning component, inspecting current IR and real execution. Core bugs still require fixes, but individual backends cannot silently change shared semantics. Product execution does not depend on online agent repair.
