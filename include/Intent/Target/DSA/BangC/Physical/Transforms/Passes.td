#ifndef INTENT_DIALECT_DSA_TRANSFORMS_PASSES_TD
#define INTENT_DIALECT_DSA_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

class IntentBangCPhysicalPass<string passArgument, string summaryText>
    : Pass<passArgument, "::mlir::ModuleOp"> {
  let summary = summaryText;
  let dependentDialects = [
    "::intent::IntentDialect", "::intent::bangc::physical::IntentBangCPhysicalDialect", "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect", "::mlir::math::MathDialect",
    "::mlir::memref::MemRefDialect", "::mlir::scf::SCFDialect"
  ];
}

def DSARealizeCollectives : IntentBangCPhysicalPass<"intent-dsa-realize-collectives",
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

def DSACollectiveGatherSupply : IntentBangCPhysicalPass<
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
def DSAMatrixSupply : IntentBangCPhysicalPass<"intent-dsa-matrix-supply",
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
def DSAComposePrivateVersions : IntentBangCPhysicalPass<"intent-bangc-compose-private-versions",
    "Reuse unobserved private versions across local updates and publication"> {
  let description = [{
    Coalesces a complete private copy, bounded local updates and copy-back when
    the original version is unobserved throughout and the temporary cannot
    escape. Preserves exact storage shape/alignment, effects and value order;
    asynchronous or unknown effects retain their original storage. The result
    is ordinary in-place physical computation, not an emitter recipe. Optional
    before shared supply; disabling it preserves the legal copy-based path.
  }];
}

def DSASimplifyWorksetBounds : IntentBangCPhysicalPass<"intent-bangc-simplify-workset-bounds",
    "Expose bounded worksets through exact scalar index and predicate proofs"> {
  let description = [{
    Uses current loop, configuration and physical participant ranges to fold
    proven constant integer expressions and redundant signed remainders.
    Canonicalization removes only proved control branches, exposing existing
    complete worksets to supply passes. No shape, task ownership, memory effect
    or numerical contract changes. Unknown bounds retain their original IR.
    Optional; the unchanged program remains a legal native lowering input.
  }];
}

def DSANormalizeIndices : IntentBangCPhysicalPass<"intent-dsa-normalize-indices",
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
