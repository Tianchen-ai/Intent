#ifndef INTENT_TARGET_BANGC_PASSES_TD
#define INTENT_TARGET_BANGC_PASSES_TD

include "mlir/Pass/PassBase.td"

class IntentBangCPass<string passArgument, string summaryText>
    : Pass<passArgument, "::mlir::ModuleOp"> {
  let summary = summaryText;
  let dependentDialects = [
    "::intent::IntentDialect", "::intent::dsa::IntentDSADialect", "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect", "::mlir::math::MathDialect",
    "::mlir::memref::MemRefDialect", "::mlir::scf::SCFDialect"
  ];
}

def BangCPrepareProgram : IntentBangCPass<"intent-bangc-prepare-program",
    "Verify the shared DSA program and bind the selected BANG C architecture"> {
  let options = [Option<"architecture", "architecture", "std::string", "\"\"",
      "Explicit BANG C architecture whose implementation profile will be used">];
  let description = [{
    Required target entry on a verified shared DSA program with its original
    public interface, compile options and DSA configuration. Binds the explicit
    supported architecture to the current module; unsupported profiles fail
    here. Does not select new author semantics or rewrite requested bindings.
    The result is the input to native computation and workspace realization,
    not a serialized or arena-bound program.
  }];
}
def BangCNativeComputations : IntentBangCPass<"intent-bangc-native-computations",
    "Realize BANG C numerical and matrix computations"> {
  let description = [{
    Required realization after target preparation and shared collective
    lowering. Consumes bounded local operations, actual matrix geometry and
    declared numeric modes. Produces native matrix preparation/computation,
    scalar or promoted arithmetic and explicit conversions while preserving
    active counts, accumulator precision, author control and effects.

    Unsupported numeric/type combinations fail rather than changing precision
    or delegating an entire algorithm. Verifies the resulting DSA program and
    invalidates affected analyses. Native workspace and implementation binding
    must still run before local optimization and target serialization.
  }];
}
def BangCNativeWorkspace : IntentBangCPass<"intent-bangc-native-workspace",
    "Realize workspace required by selected BANG C computations"> {
  let description = [{
    Required target closure after native computations. Materializes the actual
    scratch operands and initialized auxiliary values used by native arithmetic,
    comparisons, selections and collectives. Uses current dtype, shape, numeric
    attributes and resource bindings; bounded workspace is not permission to
    widen source precision or change strict/approximate behavior.

    Owns its dependent producer/storage cleanup and canonicalization, including
    cleanup needed before native composition. Outputs complete typed operations
    with explicit workspace, before implementation selection and arena binding.
    Rebuilds analyses on mutation and verifies the DSA program. Final storage
    lifetime and capacity remain the responsibility of storage binding.
  }];
}
def BangCNativeImplementations : IntentBangCPass<"intent-bangc-native-implementations",
    "Select BANG C native implementations and matrix layouts"> {
  let description = [{
    Required selection after native computations and workspace realization.
    Checks actual operation kind, dtype, numeric attributes and operand shape;
    attaches native spelling/implementation choices and the layout required by
    prepared WRAM operands. Unsupported surface combinations fail explicitly.

    Does not introduce a runtime candidate portfolio or modify the requested
    DSA configuration. Produces the selected native operations consumed by
    local optimization and later serialization. Verifies the DSA program;
    synchronization and physical arena offsets are not established here.
  }];
}
def BangCLocalComposition : IntentBangCPass<"intent-bangc-local-composition",
    "Compose local values and accesses in the BANG C program"> {
  let description = [{
    Optional local access/producer optimization after native computation,
    workspace and implementation selection, before synchronization and arena
    binding. Uses actual integer relations, local storage versions, complete
    views, alias/effect facts and the selected native capability.

    Composes index/mask programs, compact broadcast operands, transposed supply
    and bounded worksets. Owns canonicalization, copy/dead-write cleanup and
    any gather workspace/completion required by the access forms it creates.
    Nonmatching operations retain their existing legal representation. Rebuilds
    analyses after each mutation and verifies the DSA program. Stable supply
    placement and first-write elimination belong to storage-reuse; final stream
    scheduling and arena binding remain required even when this pass is omitted.
  }];
}
def BangCStorageReuse : IntentBangCPass<"intent-bangc-storage-reuse",
    "Reuse completed native storage versions and stable supply scopes"> {
  let description = [{
    Optional complete optimization group for selected native BANG C operations.
    Requires a verified native surface with explicit workspace, prepared matrix
    layouts, current views/effects and the original target bindings. Rejects
    existing arena offsets or capacity bindings; rewriting their lifetimes
    would leave stale physical storage. No hidden stage attribute is required.

    Owns entry canonicalization and copy/dead-write closure, invariant supply
    placement, reuse of completed prepared representations and bounded windows
    of existing immutable converted snapshots, positive-zero matrix first-write
    specialization, and the storage
    closure exposed by each rewrite. Finishes with supported in-place reuse and
    independent load coalescing. Every query consumes current SSA, storage
    versions and completion facts; analyses are rebuilt after mutation. Empty
    scopes, unresolved aliases, ordered state and unproved transfer completion
    retain the original behavior. Group participation is never inferred from a
    private-buffer optimization.

    The result is verified selected native IR with unbound storage. Omitting
    this group preserves the unoptimized operations, not a different algorithm.
    Continue through supply-synchronization and storage-binding before emitting
    source; this pass neither establishes final arena capacity nor turns a
    provider snapshot into shared IR for frontend resumption.
  }];
}
def BangCSupplySynchronization : IntentBangCPass<"intent-bangc-supply-synchronization",
    "Schedule BANG C supply, participants and synchronization"> {
  let description = [{
    Required closure after native computation/workspace/implementation and any
    optional local optimizations, before arena binding. Uses actual operand
    dependencies and existing work-unit/group completion contracts to insert
    waits, schedule supported transfers and form participant control. Existing
    explicit asynchronous operations retain their completion obligations.

    Forms executable synchronization and any supported double-buffer supply in
    current IR. Does not invent cross-queue visibility or reorder author state.
    Verifies the DSA program and invalidates storage/lifetime analyses. Final
    storage binding must consume this scheduled program so in-flight uses are
    included in allocation lifetimes and resource checks.
  }];
}
def BangCStorageBinding : IntentBangCPass<"intent-bangc-storage-binding",
    "Bind final BANG C physical storage and verify target legality"> {
  let description = [{
    Required terminal target closure on the selected, synchronized native
    program. Computes alias- and completion-aware allocation lifetimes, assigns
    NRAM/WRAM/SRAM intervals and checks actual arena capacity plus declared
    internal workspace against the selected resource limits.

    Writes physical offsets/capacities into the existing IR and verifies native
    operations, layouts and storage intervals before serialization. It does not
    repair numerical lowering, synthesize missing synchronization or introduce
    a runtime fallback for resource failure. Later storage/control rewrites
    require a new unbound program and the same scheduling/binding closure.
  }];
}

#endif
