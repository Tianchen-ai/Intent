#ifndef INTENT_TARGET_MOJO_TRANSFORMS_PASSES_TD
#define INTENT_TARGET_MOJO_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

class MojoTransform<string argument, string summaryText> : Pass<argument, "mlir::ModuleOp"> {
  let summary = summaryText;
  let dependentDialects = [
    "intent::cpu::IntentCPUDialect", "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::math::MathDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::vector::VectorDialect"
  ];
}

def MojoMaterializeProgram : MojoTransform<"intent-mojo-materialize-program",
    "Expand selected CPU implementations and materialize native task dispatch"> {
  let description = [{
    Requires a CPU Buffers program with valid Mojo implementation and candidate
    bindings and supported vector capabilities. Expands the selected structured
    computations and their supplies, lowers ownership, and closes any new
    worksets through the existing task partition/isolation/dispatch transforms.
    The output remains a verified CPU buffer program with explicit native work;
    the public interface, numerical options and selected candidate domain stay
    unchanged. Storage, effects and ownership facts must be rebuilt after it.
  }];
}
def MojoFusePrivateComputations : MojoTransform<"intent-mojo-fuse-private-computations",
    "Replay private computations under coordinate and storage lifetime proofs"> {
  let description = [{
    Requires and preserves a CPU Buffers program with valid Mojo bindings.
    Runs the complete private intermediate-buffer transformation, including
    exact destination forwarding and copy/access composition, under current
    coordinate, snapshot and lifetime proofs. It does not expand implementations
    or vectorize loops. The group recomputes analysis after rewrites and verifies
    its resulting CPU program; provider materialization remains a separate step.
  }];
}
def MojoVectorizeProgram : MojoTransform<"intent-mojo-vectorize-program",
    "Materialize vector traversals using coordinated implementation bindings"> {
  let description = [{
    Requires the materialized CPU Buffers program and valid Mojo bindings.
    Always closes missing loop bindings from the already selected program
    implementation, then expands supported loops using their actual bindings.
    The optional shared-traversal group is the same complete transformation
    exposed by intent-cpu-fuse-shared-traversals; disabling it does not disable
    mandatory binding closure or vectorization legality checks.

    Preserves task ownership, effects, tails, numerical permissions and the
    candidate domain. It produces verified CPU buffer IR with native vector
    operations; storage, dominance and loop facts are invalidated. Final surface
    legalization and machine register allocation are separate responsibilities.
  }];
  let options = [Option<"fuseSharedTraversals", "fuse-shared-traversals", "bool", "true",
      "Run the shared traversal-reuse group before vector expansion">];
}
def MojoFinalizeProgram : MojoTransform<"intent-mojo-finalize-program",
    "Legalize the Mojo surface and remove equivalent finalized candidates"> {
  let description = [{
    Requires a materialized and vectorized CPU Buffers program with valid Mojo
    bindings. Closes native arithmetic and atomics, bounded private storage,
    required floating-point environments and final task constant captures.
    Removes equivalent finalized candidates and verifies the CPU Realized
    contract. No structured CPU computation may remain for the serializer to
    implement; this group preserves public ABI and numerical options and does
    not select a new implementation or performance winner. All earlier analysis
    snapshots are invalid after finalization.
  }];
}

#endif
