#ifndef INTENT_COMPILER_PASSES_TD
#define INTENT_COMPILER_PASSES_TD
include "mlir/Pass/PassBase.td"

def ConstructGPU : Pass<"intent-construct-gpu", "mlir::ModuleOp"> {
  let summary = "Construct the executable GPU program from canonical KIR";
  let description = [{
    Required KIR-to-GPU construction, not an optional performance transform.
    Input is canonical typed KIR; target capability options describe the selected
    device and provider. The conversion realizes logical integer storage and
    forms GPU values, access relations and execution structure. Unsupported
    source constructs fail conversion instead of being forwarded to a serializer.
    The result still needs shared GPU realization and provider legalization.
    This pass does not import tuning profiles or select a runtime winner.
  }];
  let options = [
    Option<"computeUnits", "compute-units", "int64_t", "0", "Compute units">,
    Option<"sharedMemoryPerUnit", "shared-memory-per-unit", "int64_t", "0", "Shared bytes per unit">,
    Option<"maxDynamicSharedMemoryPerBlock", "max-dynamic-shared-memory-per-block", "int64_t", "0", "Dynamic shared limit">,
    Option<"registersPerUnit", "registers-per-unit", "int64_t", "0", "Registers per unit">,
    Option<"maxThreadsPerBlock", "max-threads-per-block", "int64_t", "0", "Threads per block">,
    Option<"computeCapabilityMajor", "compute-capability-major", "int64_t", "0", "Architecture major">,
    Option<"computeCapabilityMinor", "compute-capability-minor", "int64_t", "-1", "Architecture minor">,
    Option<"singleToDoublePrecisionPerfRatio", "single-to-double-precision-perf-ratio", "int64_t", "0", "FP32/FP64 throughput ratio">,
    Option<"matrixUnits", "matrix-units", "bool", "false", "Matrix capability">,
    Option<"dynamicVectorWidth", "dynamic-vector-width", "bool", "false", "Dynamic vector width">,
    Option<"nativeTupleReductions", "native-tuple-reductions", "bool", "false", "Provider tuple reductions">,
    Option<"nativeTupleReductionRequiresConstantIdentity", "native-tuple-reduction-requires-constant-identity", "bool", "false", "Provider tuple reduction requires compile-time identities">,
    Option<"nativeFragmentGather", "native-fragment-gather", "bool", "false", "Provider fragment gather">
  ];
  let dependentDialects = ["intent::gpu::IntentGPUDialect", "mlir::arith::ArithDialect", "mlir::func::FuncDialect", "mlir::scf::SCFDialect"];
}

def ConstructCPU : Pass<"intent-construct-cpu", "mlir::ModuleOp"> {
  let summary = "Construct the executable CPU program from canonical KIR";
  let description = [{
    Required KIR-to-CPU construction. Consumes canonical typed KIR and creates
    the CPU program on which structured computation passes operate. The
    strided-inputs option changes external input layout representation for the
    selected provider; it is an ABI decision, not a vectorization switch.
    Logical integer storage and source conversion are performed here. CPU
    bufferization, implementation selection and provider legalization remain
    separate stages. Unsupported source constructs fail conversion.
  }];
  let options = [Option<"stridedInputs", "strided-inputs", "bool", "false", "Preserve external input strides">];
  let dependentDialects = ["intent::cpu::IntentCPUDialect", "mlir::arith::ArithDialect", "mlir::func::FuncDialect", "mlir::linalg::LinalgDialect", "mlir::math::MathDialect", "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect", "mlir::vector::VectorDialect"];
}

def ConstructDSA : Pass<"intent-construct-dsa", "mlir::ModuleOp"> {
  let summary = "Construct complete local-memory DSA candidates";
  let description = [{
    Reads complete provider configurations and constructs each supplied row
    from the same canonical typed KIR in a separate standard module. Tile,
    task and local-storage bindings form the checked DSA configuration;
    shapes and strides bind parameter facts for this compilation. No parameter
    combinations or prewritten operator implementations are invented.
    Invalid bindings or unsupported conversion fail. Each candidate requires
    the shared DSA pipeline and target native legalization before serialization;
    it is not yet a BANG C program with synchronized, arena-bound storage.
  }];
  let options = [
    Option<"directory", "directory", "std::string", [{""}], "Directory of complete backend configurations">,
    Option<"overrides", "overrides", "std::string", [{""}], "Complete configuration overrides">,
    Option<"shapes", "shapes", "std::string", [{"{}"}], "JSON parameter shapes">,
    Option<"strides", "strides", "std::string", [{"{}"}], "JSON parameter strides">
  ];
  let dependentDialects = ["intent::dsa::IntentDSADialect", "mlir::arith::ArithDialect", "mlir::func::FuncDialect", "mlir::math::MathDialect", "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect"];
}

def ResolveGPUProfiles : Pass<"intent-resolve-gpu-profiles", "mlir::ModuleOp"> {
  let summary = "Resolve finite profile inputs into the current GPU module";
  let description = [{
    Reads the selected profile resource directory and explicit overrides,
    validates the finite tables and attaches typed profiles to the current
    GPU module. A missing directory or invalid profile is an error. This is
    profile input resolution, not autotuning or configuration selection.
    Later passes consume current IR attributes without rereading these files.
    Shared-IR resume does not rerun this pass or accept replacement overrides.
  }];
  let options = [
    Option<"provider", "provider", "std::string", [{""}], "Provider owning the complete configuration table">,
    Option<"directory", "directory", "std::string", [{""}], "Directory of compiler profile resources">,
    Option<"overrides", "overrides", "std::string", [{""}], "Explicit profile overrides">
  ];
  let dependentDialects = ["intent::gpu::IntentGPUDialect"];
}
#endif
