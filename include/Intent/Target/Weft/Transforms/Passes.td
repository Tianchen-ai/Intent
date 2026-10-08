include "mlir/Pass/PassBase.td"

def WeftPrepareTaskViews : Pass<"intent-weft-prepare-task-views", "mlir::ModuleOp"> {
  let summary = "Reify typed task views and close the captures for Weft conversion";
  let description = [{
    Requires a verified CPU Buffers program with explicit Tasks and valid Weft
    implementation bindings. Reifies each task's actual view metadata and
    captures, preserving descriptor coordinates, aliases, ownership, effects
    and the selected computation. Verifies the resulting CPU buffer program;
    native Weft layout and task conversion are not performed here. Previously
    computed capture, view and storage facts must be rebuilt after rewriting.
  }];
  let dependentDialects = ["intent::cpu::IntentCPUDialect",
      "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
      "mlir::memref::MemRefDialect"];
}

def WeftConvertProgram : Pass<"intent-weft-convert-program", "mlir::ModuleOp"> {
  let summary = "Convert CPU tasks into an explicit host and Canonical Weft program";
  let description = [{
    Requires a CPU Buffers program with prepared task-view captures and valid
    Weft implementation bindings. Closes ownership and converts the selected
    task computations into Canonical Weft kernels plus an explicit host module,
    task bindings, accesses and native argument requirements. Host residual
    computations are materialized through the existing scalar CPU path.

    Public ABI, numerical options, logical coordinates, storage interpretations
    and task completion are preserved. The complete host/device program is
    verified before return. This is a representation-changing boundary: CPU
    analysis snapshots do not survive it. External Weft compilation owns native
    layouts, instruction selection and machine resources; serialization does
    not choose a new computation or repair missing task facts.
  }];
  let dependentDialects = ["intent::cpu::IntentCPUDialect",
      "intent::weft_provider::IntentWeftDialect", "weft::kernel::WEFTKernelDialect",
      "mlir::arith::ArithDialect", "mlir::func::FuncDialect", "mlir::linalg::LinalgDialect",
      "mlir::math::MathDialect", "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect"];
}

def WeftVerifyProgram : Pass<"intent-weft-verify-program", "mlir::ModuleOp"> {
  let summary = "Verify the complete Weft host, device and task interface";
  let description = [{
    Requires the converted Weft host/device container and checks its complete
    task bindings, native interfaces, encodings, accesses and selected
    implementation closure. Performs no rewrite, inference repair or native
    compilation; successful verification preserves the program and its analyses.
  }];
  let dependentDialects = ["intent::cpu::IntentCPUDialect",
      "intent::weft_provider::IntentWeftDialect", "weft::kernel::WEFTKernelDialect"];
}
