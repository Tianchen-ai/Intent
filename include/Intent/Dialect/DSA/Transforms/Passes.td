#ifndef INTENT_DIALECT_DSA_TRANSFORMS_PASSES_TD
#define INTENT_DIALECT_DSA_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

class IntentDSAPass<string passArgument, string summaryText>
    : Pass<passArgument, "::mlir::ModuleOp"> {
  let summary = summaryText;
  let dependentDialects = [
    "::intent::IntentDialect", "::intent::dsa::IntentDSADialect", "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect", "::mlir::math::MathDialect",
    "::mlir::memref::MemRefDialect", "::mlir::scf::SCFDialect"
  ];
}

def DSARealizeCollectives : IntentDSAPass<"intent-dsa-realize-collectives",
    "Realize current DSA reduction and scan combine regions"> {
  let description = [{
    Requires a verified DSA program with explicit local sources, initial states,
    outputs, active counts and typed combine regions. Selects supported native
    reductions or materializes scalar/whole-slice combine traversal and private
    state snapshots. Leaves executable DSA, memref and structured control with
    no ordinary collective regions, before supply and resource binding.

    This is required realization, not a removable performance optimization.
    Keeps the declared combine, captures, identity, tuple publication, source
    order and empty-input behavior. Recomputes affected physical analyses and
    verifies the complete DSA program after rewriting.
  }];
}

def DSACollectiveGatherSupply : IntentDSAPass<
    "intent-dsa-collective-gather-supply",
    "Form collective row supply from current DSA address relations"> {
  let description = [{
    Optional supply optimization on a verified DSA program after collective
    realization, before target workspace, synchronization and arena binding.
    Requires actual external row accesses, work-owner coordinates, compatible
    destination geometry and complete alias/effect facts for a participating
    group. A nonmatching access remains an ordinary DSA gather.

    Forms explicit group participants, shared/local storage and collective row
    supply with its existing publication and completion contract. Preserves
    every active source/destination coordinate, fill and public interface; it
    does not authorize arbitrary cross-group synchronization. Rebuilds access,
    ownership and storage facts after mutation and verifies the resulting DSA
    program. Target realization and final resource binding are still required.
  }];
}
def DSAMatrixSupply : IntentDSAPass<"intent-dsa-matrix-supply",
    "Form resident or shared supply for a complete DSA matrix traversal"> {
  let description = [{
    Optional supply optimization on a verified, unbound DSA program. Consumes
    complete ordinary matrix worksets: current A/B/C views and access geometry,
    free/reduction coordinates, output coverage, read stability and explicit
    resource bindings. It does not recover these facts from kernel names or
    reinterpret the requested configuration as a hint.

    Forms resident packed RHS or shared panel supply, explicit group ownership,
    allocations, active counts and completion operations. Epilogue coordinates
    and output coverage follow the same workset. Unsupported or unprofitable
    supply geometry retains the existing legal program. Preserves source
    precision, accumulator type and author control boundaries. Discards
    analyses after rewriting and verifies the DSA program; native realization
    and the target's actual arena check remain later obligations.
  }];
}
def DSANormalizeIndices : IntentDSAPass<"intent-dsa-normalize-indices",
    "Normalize exact DSA index programs and canonicalize the result"> {
  let description = [{
    Optional normalization of scalar index/i64 expressions in a verified DSA
    program. Combines current typed add/subtract/multiply expressions using
    their modular integer semantics; loads and unsupported expressions remain
    opaque values. Does not infer new bounds or signed-no-wrap permissions.

    Rewrites actual SSA and runs canonicalization/CSE when changed, preserving
    memory coordinates, control, effects and public bindings. No side plan or
    target storage layout is produced. Affected analyses are invalidated and
    the complete DSA program is verified at the group boundary.
  }];
}

#endif
