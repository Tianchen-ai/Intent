include "mlir/Pass/PassBase.td"

class IntentGPUTransformPass<string passArgument, string summaryText,
                             string contract>
    : Pass<passArgument, "::mlir::ModuleOp"> {
  let summary = summaryText;
  let description = !strconcat([{
    Consumes a complete shared GPU module, before provider-native extensions,
    and verifies the shared GPU program on completion. Queries concern the
    current typed SSA, coordinate, control and effect relations; mutations do
    not preserve cached analyses. The group owns its required relation closure.

  }], contract);
  let dependentDialects = ["::intent::gpu::IntentGPUDialect",
      "::mlir::arith::ArithDialect", "::mlir::cf::ControlFlowDialect",
      "::mlir::func::FuncDialect", "::mlir::scf::SCFDialect"];
}

def HoistLoopInvariants : Option<"hoistLoopInvariants", "hoist-loop-invariants",
    "bool", "true", "Run bounded invariant value and stable-read placement at this group's cleanup points">;

def ReductionResidencyPreference : Option<"preferredResidentPrograms",
    "preferred-resident-programs", "unsigned", "1",
    "Preferred native program residency for the nominal full-tile adoption budget">;

def NormalizeStructuredSourcesPass : IntentGPUTransformPass<
    "intent-gpu-normalize-structured-sources",
    "Normalize structured GPU sources", [{
    Runs before ownership and blocking. Folds current scalar/shape values, then
    normalizes contraction sources from actual operand axes, uses and numerical
    permissions. Closes value relations for subsequent structured realization.
    It neither chooses a provider primitive nor changes the public interface.

    hoist-loop-invariants controls only optional invariant placement between
    canonicalization and CSE. Disabling it retains all value cleanup and source
    normalization; the independent hoist pass can be scheduled separately.
  }]> {
  let options = [HoistLoopInvariants];
}

def PredicateScalarControlPass : IntentGPUTransformPass<
    "intent-gpu-predicate-scalar-control",
    "Predicate scalar control in the GPU program", [{
    Consumes scalar structured control before pointwise ownership. Converts
    eligible branch bodies to explicit predicates/selects using their actual
    effects, captures and workset dependence. Launch-uniform choices remain
    lazy; predicated accesses retain validity and fill semantics. This prepares
    eligible control for fragment lifting, not arbitrary speculative execution.
  }]>;

def PointwiseOwnershipPass : IntentGPUTransformPass<
    "intent-gpu-form-pointwise-ownership",
    "Form pointwise GPU ownership", [{
    Consumes normalized sources and current execution groups. Chooses logical
    ownership from access geometry, effects and selected GPU capabilities,
    updating workset mappings and their dependent values together. Blocking
    consumes this ownership; this stage must not be replaced by serializer
    inference or skipped while required ownership remains unformed.
  }]>;

def ReductionConsumersPass : IntentGPUTransformPass<
    "intent-gpu-realize-reduction-consumers",
    "Realize shared GPU summary and output traversals", [{
    Consumes formed ownership before independent pointwise blocking. Normalizes
    ordinary reduction chains and native tuples, then queries closed input,
    compact summary and pointwise output groups. Uses current nominal liveness
    to retain fitting complete tiles or form bounded input/output traversals.
    A native adapter supplies its residency preference as a pass option. Only
    groups whose minimum whole working set exceeds that nominal budget adopt
    streaming; fitting native primitives and their original configuration
    family remain intact. This is profitability policy, not native allocation.
    Preserves source read epochs and captures. Additional output reads require
    real control with disjoint, nonoverlapping views; the other arm retains the
    complete original program. Collectives retain their typed callbacks and
    identities for provider-native lowering. Owns value-relation closure.
  }]> {
  let options = [ReductionResidencyPreference];
}

def PointwiseBlockingPass : IntentGPUTransformPass<
    "intent-gpu-form-pointwise-blocking",
    "Form pointwise GPU blocking and its dependent value graph", [{
    Consumes formed pointwise ownership. Creates bounded fragment traversals,
    rebuilds their value/access schemas, realizes dependent vector contractions
    and guards inactive predicated loops. Closes value relations within the
    group. Physical extents come from declared parameters and capabilities;
    source logical domains and ordered control remain unchanged.
  }]>;

def ScanConsumersPass : IntentGPUTransformPass<
    "intent-gpu-realize-scan-consumers",
    "Realize GPU scan consumer traversals", [{
    Consumes blocked pointwise values with typed scan/helper and access
    relations. Realizes eligible bounded scan consumers and terminal reads,
    retaining member order, actual validity, empty behavior and output effects.
    Owns value-relation closure; it does not replace scan with ordinary reduce
    or implement the provider's internal scan communication.
  }]>;

def RetainedValuesPass : IntentGPUTransformPass<
    "intent-gpu-materialize-retained-values",
    "Materialize retained GPU values", [{
    Consumes current producer/consumer SSA, storage versions and ownership.
    Selects retained values or legal replay using effects, dominance, access
    geometry and bounded working-set facts, emitting actual buffers, reads and
    writes where needed. Closes value relations. This optimization cannot
    stand in for mandatory workspace lowering or provider storage legality.
  }]>;

def OnlineReductionsPass : IntentGPUTransformPass<
    "intent-gpu-co-realize-online-reductions",
    "Co-realize GPU online reductions", [{
    Consumes structured reductions and contractions before region realization.
    This optional optimization checks intent.compile_options, current typed
    summary/identity/axis relations and stable replay before forming online
    traversal and carry. A disabled or inapplicable optimization leaves the
    ordinary structured program for its existing realization stages. Numerical
    permission is never inferred from provider choice or kernel identity.
  }]>;

def RegionFoldsPass : IntentGPUTransformPass<
    "intent-gpu-realize-region-folds",
    "Realize GPU region folds", [{
    Consumes typed region-fold sources, physical ownership and summarizer/
    combine schemas. Forms segment loops, captures, identities and state flow,
    preserving ordered membership and empty behavior, and closes value
    relations. This is required realization while region-fold operations
    remain, not an optional substitute for the author's algorithm.
    The optional simplify-first-summary rewrite directly supplies the complete
    state from an existing first-summary traversal using the fold's neutral
    identity contract. It preserves the nonempty guard, empty identity, source
    slices and subsequent combine. It does not infer neutrality from an
    arbitrary lowered loop or force additional peeling. The option defaults
    to false, retaining the initial identity combine; mandatory realization
    and relation closure run with either setting.
  }]> {
  let options = [Option<"simplifyFirstSummary", "simplify-first-summary",
      "bool", "false", "Forward an existing first summary using the fold's neutral identity contract">];
}

def RegionScansPass : IntentGPUTransformPass<
    "intent-gpu-realize-region-scans",
    "Realize GPU region scans", [{
    Consumes typed region-scan sources, physical ownership and summary,
    transition and emission schemas. Forms ordered segment traversal, state
    application and source-aligned outputs, then closes value relations.
    This is required realization while region-scan operations remain; native
    collectives and contractions remain visible for their respective stages.
  }]>;

def ReductionsPass : IntentGPUTransformPass<
    "intent-gpu-realize-reductions",
    "Realize GPU reduction programs", [{
    Consumes structured ordinary reductions after ownership and enclosing
    region traversal have been formed. Fuses eligible independent reductions
    using native tuple capability, decomposes axes, forms bounded traversals
    and neutralizes inactive members with their actual identities. Completes
    accumulator/value relations. Direct sole-use ordinary reduction chains
    with the same complete tuple combine, identities and scalar captures can
    first combine their axes; masks, numerical casts and different contracts
    retain the original structure. This uses the same callback-contract query
    as completed reduction normalization.

    A complete static outer axis with independent output lanes may use bounded
    outer and inner member traversals when current replay and producer-removal
    facts permit it. Deferred full-coverage axes retain their representation
    unless current fragment resource pressure selects bounded supply. Uniform
    tuple fields use their actual scalar values and the same identity predicates,
    without fictitious load roots. Physical parameters continue to come from
    the existing correlated profile and actual capacity.
    Required reduction realization remains even
    when a fusion or blocking opportunity is inapplicable; provider-native
    reduction trees and communication are not reconstructed here.
  }]>;

def ContractionsPass : IntentGPUTransformPass<
    "intent-gpu-realize-contractions",
    "Realize GPU contraction programs", [{
    Consumes normalized contraction operands and formed ownership/ranges.
    Forms bounded source supply, contraction loops and typed accumulators from
    current axis, access and capability facts, preserving declared precision
    and zero-reduction behavior. Reduction granularity uses the same logical
    capacity bound as other traversals; operand shapes and loop steps consume
    one effective extent from the existing complete configuration row.
    Closes value relations. Provider primitive
    selection, lane layout and native register allocation remain downstream.
  }]>;

def AccessCompositionPass : IntentGPUTransformPass<
    "intent-gpu-compose-realized-accesses",
    "Compose realized GPU accesses and values", [{
    Consumes newly realized fragments, traversals and storage. Composes actual
    coordinates/validity, materializes retained values, recomposes resulting
    accesses and orients loop contractions as one closed group.
    Stable same-block reads reuse their current storage version only across
    proved non-clobbering effects, including writable external views. Exact
    coordinate and validity coverage preserves each consumer's original fill.
    Necessary access/schema composition remains part of this boundary even when no
    optional retention or orientation rewrite applies. No caller-supplied
    repair pass is required before shared verification.
  }]>;

def PrivateStoresPass : IntentGPUTransformPass<
    "intent-gpu-schedule-private-stores",
    "Schedule private GPU stores", [{
    Consumes explicit program-private storage and current access/effect
    relations. Optionally moves eligible stores to their legal consumption
    scope without changing values, initialization, lifetime or observable
    effect order. Inapplicable memory motion leaves the original accesses;
    storage lowering and provider synchronization remain separate obligations.
  }]>;

def BufferVectorizationPass : IntentGPUTransformPass<
    "intent-gpu-vectorize-buffer-loops",
    "Vectorize GPU buffer loops", [{
    Consumes explicit buffer traversals after access composition. Vectorizes
    eligible independent iterations using current coordinates, injectivity,
    alias/effect and bounds facts, rebuilding values and accesses together.
    Across multiple program instances, inferred read-free scalar external
    stores retain each non-singleton program coordinate on a distinct invariant
    resource axis. Unit iteration coordinates and non-overlapping view guards
    establish disjoint writes. This domain has no loop carries, nested regions,
    memory reads or other effects; mutable-buffer inference remains restricted
    to a single instance. Existing physical parameters bound the chunks, and
    inactive lanes and zero-trip loops retain their original effects.
    Owns relation closure. This optional optimization does not parallelize an
    ordinary loop whose iteration independence has not been proved.
  }]>;

def BufferPromotionPass : IntentGPUTransformPass<
    "intent-gpu-promote-buffer-values",
    "Promote GPU buffer values", [{
    Consumes explicit initialized buffers and complete current use/effect
    relations. Optionally replaces eligible private storage with actual SSA
    values/carries, preserving guarded reads, first writes and lifetimes.
    Closes dependent value relations; nonpromoted storage remains explicit for
    downstream storage lowering rather than becoming an implicit descriptor.
  }]>;

def ProgramMappingPass : IntentGPUTransformPass<
    "intent-gpu-refine-program-mapping",
    "Refine GPU program mapping", [{
    Consumes realized worksets and their current execution-group mapping.
    Refines eligible mapping geometry from coordinate roles, launch-visible
    bounds and resource capabilities, updating the executable mapping itself.
    This optional refinement preserves logical coverage and effects; an
    inapplicable mapping keeps the already executable original workset.
  }]>;

def RangePredicatesPass : IntentGPUTransformPass<
    "intent-gpu-simplify-range-predicates",
    "Remove predicates proven by complete physical tiles", [{
    Consumes formed physical ranges, access bounds and validity. Removes only
    predicates proved redundant for the current members; unknown/tail domains
    keep their original predicates and fill behavior. Aligned unit windows can
    prove their own lower predicate without asserting external resource bounds.
    This optional
    simplification changes actual SSA, never a side-table assertion of safety.
  }]>;

def CommonValuesPass : IntentGPUTransformPass<
    "intent-gpu-eliminate-common-values",
    "Eliminate common GPU values and simplify masked coordinates", [{
    Folds scalar/shape values and exact typed boolean identities, performs CSE
    and simplifies masked coordinates
    from the current typed program. This complete cleanup requires no separate
    invariant-motion or relation-repair pass. With hoist-loop-invariants=false
    it does not move loop-invariant reads; all value/access cleanup still runs.
    The default preserves canonicalization, optional hoisting, CSE, then
    masked-coordinate simplification in that order.
  }]> {
  let options = [HoistLoopInvariants];
}

def HoistLoopInvariantValuesPass : IntentGPUTransformPass<
    "intent-gpu-hoist-loop-invariant-values",
    "Hoist bounded invariant GPU values and stable reads", [{
    Optional scope placement on a verified shared GPU program. Performs value
    canonicalization, invariant movement and CSE as a complete group; it can
    run independently of normalization, traversal fusion or common-value
    cleanup. Those groups expose their own hoist-loop-invariants switch, so
    callers can keep their necessary work while scheduling placement here.

    Uses current dominance, effects, aliases, loop bounds and actual captures/
    carries. Stable reads preserve their original snapshot: guarded movement
    requires nonempty execution and proved or explicitly guarded disjointness;
    the aliasing path reads at the original point. Finite extent bounds and
    nominal working-set capacity are adoption conditions, not additional IR
    legality requirements or a guarantee of native register allocation.
    Unknown conditions keep the original program. No provider-native surface
    is accepted by this shared entry; provider finalization owns its verifier.
  }]>;

def RealizeSharedProgramPass : IntentGPUTransformPass<
    "intent-gpu-realize-shared-program",
    "Realize the complete shared GPU program including uniform branch dispatch", [{
    Consumes construction-complete shared GPU IR and resolved capabilities/
    tuning-profile context. Runs the registered realization groups in their
    dependency order, including uniform branch specialization where legal.
    The result is realized shared IR; finite configuration binding and final
    traversal fusion follow separately. For custom composition use the same
    registered groups with their stated preconditions and optional switches,
    rather than treating arbitrary pass permutations as supported.
  }]> {
  let options = [ReductionResidencyPreference];
}

def MaterializeConfigurationsPass : IntentGPUTransformPass<
    "intent-gpu-materialize-configurations",
    "Bind shared configurations from resolved current-module profiles", [{
    Consumes the realized shared program and its current-module profile
    context. Removes unused physical parameters, materializes complete finite
    configuration tuples and verifies them. This is required configuration
    closure before transformations that compare all selected tuples or before
    provider lowering; it neither benchmarks candidates nor selects a winner.
  }]>;

def FuseIndependentTraversalsPass : IntentGPUTransformPass<
    "intent-gpu-fuse-independent-traversals",
    "Fuse realized traversals and verify their closed configurations", [{
    Consumes realized traversals after finite configuration binding. Optionally
    fuses independent loops using current domains, captures, storage effects
    and all selected tuples, reuses stable loads and sinks their legal chains.
    Closes fused traversal relations and verifies the unchanged configuration
    contract. Completed reduction normalization is a separate optional group.

    hoist-loop-invariants controls only invariant placement in cleanup after
    fusion and, when changed, after stable-load reuse. False preserves fusion,
    reuse, required canonicalization/CSE and all postconditions. True retains
    the original canonicalization, hoisting, CSE order at both points. This
    controls Intent placement, not downstream provider compiler LICM.
  }]> {
  let options = [HoistLoopInvariants];
}

def NormalizeCompletedReductionsPass : IntentGPUTransformPass<
    "intent-gpu-normalize-completed-reductions",
    "Combine completed GPU reduction trees", [{
    Consumes verified, realized shared GPU IR after finite configuration
    binding. Combines eligible ordinary reduction chains using their current
    component schemas, combine bodies, identities, physical axes and complete
    member coverage. Keeps source computations and reads at their original
    points; only the collective trees and intermediate values change.

    Direct sole-use chains without intervening masks define their exact
    physical member union regardless of the source producer's representation.
    Removing an intervening identity mask additionally requires current
    logical coverage or completed-partial evidence for every tuple component.

    This optional group preserves the declared dtypes, numerical attributes,
    logical members, effects and configuration domain. Unknown coverage or
    incompatible combine contracts leave the existing legal program in place.
    It owns dead-value cleanup, dependent value-relation closure and shared
    configuration verification. Provider reductions and machine layouts remain
    owned by the selected native compiler. The default shared pipeline runs it
    after independent traversal fusion; explicit pipelines can omit or invoke
    this group without requiring a separate relation-repair pass.
  }]>;

def PeelTraversalTailsPass : IntentGPUTransformPass<
    "intent-gpu-peel-traversal-tails",
    "Separate complete GPU traversal blocks from their tail", [{
    Consumes verified, realized shared GPU IR after finite configuration
    binding. Splits eligible counted traversals at their actual aligned upper
    bound, preserving each iteration's body, captures, effects and complete
    loop-carried state. The complete prefix retains its loop; the zero-or-one
    tail becomes a conditional body with the original induction/state bindings.
    Both execute in their original order; empty intervals perform no access.

    Eligibility follows current unit-step fragment coordinates, the positive
    physical loop width, nonnegative bounds and representable aligned
    arithmetic. Unknown geometry/effects or an internal allocation retain the
    original legal traversal: cloning does not invent new resource identities
    or discard an allocation owner. The
    group proves and simplifies complete-block predicates, cleans pure values,
    and closes value relations and unchanged configuration tuples. It does
    not change a combine, numerical mode, member domain or provider layout.

    This is optional and independently runnable on closed shared IR. The
    default pipeline applies it after completed reduction normalization so
    interval splitting does not obscure the original completion proof.
  }]>;
