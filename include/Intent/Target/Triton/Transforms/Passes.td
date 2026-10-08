include "mlir/Pass/PassBase.td"

class IntentTritonTransformPass<string passArgument, string summaryText>
    : Pass<passArgument, "::mlir::ModuleOp"> {
  let summary = summaryText;
  let dependentDialects = ["::intent::gpu::IntentGPUDialect",
      "::intent::triton::IntentTritonDialect", "::mlir::arith::ArithDialect",
      "::mlir::func::FuncDialect",
      "::mlir::scf::SCFDialect"];
}

def ProgramGridPass : IntentTritonTransformPass<
    "intent-triton-program-grid",
    "Legalize the Triton program grid"> {
  let description = [{
    Consumes and preserves a verified shared GPU program before execution-group
    expansion. Uses current coordinate roles, launch extents and unambiguous
    write-coordinate order to expose an eligible workset as a native grid of
    at most three dimensions. Secondary dimensions require proved finite
    launch bounds; ineligible geometry retains the executable linear grid.

    This mapping optimization rebuilds actual program IDs and the group's
    linear coordinate while preserving its body, logical coverage and effects.
    It does not choose fragment layouts or eliminate the required downstream
    group/access preparation. Shared verification runs before and after it.
  }];
}

def PrepareMemoryPass : IntentTritonTransformPass<
    "intent-triton-prepare-memory",
    "Prepare Triton buffers, indexed accesses and matrix shapes"> {
  let description = [{
    Requires the realized, verified shared GPU program and its selected target
    capabilities. Expands execution groups, closes eligible scan tails and
    oversized/masked/expanding gather and scatter forms, and normalizes matrix
    operand shapes. Uses actual coordinates, validity, collision semantics and
    typed shapes, preserving public ABI and numerical/effect contracts.

    This is a preparation boundary required while such unlowered forms remain.
    It produces verified shared GPU IR, not native Triton extensions. Resource
    workspace, descriptors, native configuration legality and primitive choices
    are closed by native-forms; unsupported access forms diagnose here rather
    than leaving serializer-side reconstruction.
  }];
}

def NativeFormsPass : IntentTritonTransformPass<
    "intent-triton-native-forms",
    "Form native Triton accesses, supply and complete configurations"> {
  let description = [{
    Requires prepared shared GPU IR with current-module tuning profiles and
    capabilities. Consumes actual ordered dependencies, buffers and supply
    loops to declare provider options, lower invocation workspace, close
    ordered accesses and form eligible descriptors/pointer accesses. Selects
    legal finite configurations and native contraction forms from these facts.

    Ordered-view synchronization consumes complete current access coordinates.
    Disjoint represented windows or consecutive loop chunks can omit a barrier
    only with the existing layout and no-wrap proofs. Unknown overlaps and
    cross-view aliases retain their synchronization; a value dependency alone
    does not prove that all CTA reads have completed.
    Uniform branches propagate incoming accesses independently and conservatively
    join outstanding dependencies at their exits.

    Matrix multiplicands carried between iterations use the existing native
    single-stage loop hint to bound retained supply lifetimes. Independent
    matrix supply and ordinary accumulator carries keep the row's stage option;
    external Triton still constructs layouts and the actual pipeline schedule.

    This required transition produces explicit native Triton operations and
    configuration requirements; public source semantics remain unchanged and
    any compiler-private resources are represented in IR. Unsupported provider
    or capability combinations diagnose at their owning form. Once extensions
    are installed, structural verification replaces shared-surface validation;
    finalize-program must still close callbacks/resources and verify the full
    Triton surface. External layout, pipeline and ISA lowering remain external.
  }];
}

def FinalizeProgramPass : IntentTritonTransformPass<
    "intent-triton-finalize-program",
    "Close Triton callbacks and resource constraints and verify the surface"> {
  let description = [{
    Consumes the current native Triton program after native forms and finite
    configurations have been selected. Closes constant/shape values, collective
    callbacks and configuration resource requirements, then verifies the Triton
    source surface. No shared-GPU verifier is applied to native extensions.

    hoist-loop-invariants optionally runs the shared bounded invariant-value
    placement component between canonicalization and CSE at the existing
    finalization point. False retains all required cleanup, callback lowering
    and verification. Adoption uses current effects, aliases, control and
    finite nominal resource bounds; unknown facts do not reject otherwise
    legal IR. This switch does not disable the external Triton compiler's LICM.
  }];
  let options = [Option<"hoistLoopInvariants", "hoist-loop-invariants", "bool",
      "true", "Run bounded invariant value and stable-read placement during finalization">];
}
