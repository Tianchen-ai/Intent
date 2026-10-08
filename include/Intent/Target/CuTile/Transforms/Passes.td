include "mlir/Pass/PassBase.td"

class IntentCuTileTransformPass<string passArgument, string summaryText>
    : Pass<passArgument, "::mlir::ModuleOp"> {
  let summary = summaryText;
  let dependentDialects = ["::intent::gpu::IntentGPUDialect",
      "::intent::cutile::IntentCuTileDialect", "::mlir::arith::ArithDialect",
      "::mlir::cf::ControlFlowDialect", "::mlir::func::FuncDialect",
      "::mlir::scf::SCFDialect"];
}

def PrepareProgramPass : IntentCuTileTransformPass<
    "intent-cutile-prepare-program",
    "Prepare cuTile matrix shapes, buffers and workspace accesses"> {
  let description = [{
    Requires realized shared GPU IR, selected capabilities and resolved
    current-module tuning profiles. Expands execution groups, normalizes matrix
    operand shapes, prepares launch configurations and lowers workspace to
    explicit accesses; exact scalar division folding preserves its numerical
    contract. Public views and author control/effects are unchanged.

    This required preparation boundary produces another verified shared GPU
    program. Native access/collective/MMA selection follows in native-program;
    unsupported preparation or configuration diagnoses here rather than being
    delegated to the serializer. Native register/layout allocation is not part
    of this group, and storage facts must be recomputed after rewriting.
  }];
}

def NativeProgramPass : IntentCuTileTransformPass<
    "intent-cutile-native-program",
    "Form native cuTile accesses and computations from the current GPU program"> {
  let description = [{
    Requires the prepared shared GPU program with original capabilities and
    tuning profiles. Collects current accesses and structured computations,
    checks their actual axis/shape/dtype and hardware requirements, and commits
    native accesses and collective/MMA forms together. Consumes bounds
    assumptions only after their native replacements have been formed.

    This required transition preserves active coordinates, fill, atomics and
    declared numerical precision. Missing primitive or hardware support fails
    at the responsible operation without substituting another algorithm. Its
    output may contain cuTile extensions and is structurally verified;
    finalize-program still owns configuration, loop/view closure and complete
    provider-surface validation. Distributed layouts and communication remain
    the external cuTile compiler's responsibility.
  }];
}

def FinalizeProgramPass : IntentCuTileTransformPass<
    "intent-cutile-finalize-program",
    "Close cuTile configurations, loops and array views and verify the surface"> {
  let description = [{
    Consumes the current native cuTile program. Refines native MMA/loop forms,
    materializes closed configurations, preserves native indices and collapses
    array views. Completes value cleanup, array-bound facts and configuration
    requirements, then verifies the cuTile source surface. Native extensions
    are checked by the provider verifier, not the shared-GPU verifier.

    hoist-loop-invariants optionally runs the shared bounded invariant-value
    placement component between canonicalization and CSE at the existing
    finalization point. False retains all required native/value closure and
    verification. Adoption uses current effects, aliases, control and finite
    nominal resource bounds; unknown facts leave otherwise legal IR unchanged.
    This switch does not disable the external cuTile compiler's optimizations.
  }];
  let options = [Option<"hoistLoopInvariants", "hoist-loop-invariants", "bool",
      "true", "Run bounded invariant value and stable-read placement during finalization">];
}
