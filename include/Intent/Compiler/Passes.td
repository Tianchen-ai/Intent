#ifndef INTENT_COMPILER_PASSES_TD
#define INTENT_COMPILER_PASSES_TD
include "mlir/Pass/PassBase.td"

def ConstructBlock : Pass<"intent-construct-block", "mlir::ModuleOp"> {
  let summary = "Construct the executable Block program from canonical KIR";
  let description = [{
    Required KIR-to-Block construction, not an optional performance transform.
    Input is canonical typed KIR; target capability options describe the selected
    device and provider. The conversion realizes logical integer storage and
    forms Block values, access relations and execution structure. Unsupported
    source constructs fail conversion instead of being forwarded to a serializer.
    The result still needs shared Block realization and provider legalization.
    This pass does not import tuning profiles or select a runtime winner.
  }];
  let options = [
    Option<"target", "target", "std::string", [{""}], "Typed target attribute implementing the block resource interface">,
    Option<"matrixUnits", "matrix-units", "bool", "false", "Matrix capability">,
    Option<"dynamicVectorWidth", "dynamic-vector-width", "bool", "false", "Dynamic vector width">,
    Option<"nativeTupleReductions", "native-tuple-reductions", "bool", "false", "Provider tuple reductions">,
    Option<"nativeTupleReductionRequiresConstantIdentity", "native-tuple-reduction-requires-constant-identity", "bool", "false", "Provider tuple reduction requires compile-time identities">,
    Option<"nativeFragmentGather", "native-fragment-gather", "bool", "false", "Provider fragment gather">
  ];
  let dependentDialects = ["intent::block::IntentBlockDialect", "mlir::arith::ArithDialect", "mlir::func::FuncDialect", "mlir::scf::SCFDialect"];
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
    Option<"shapes", "shapes", "std::string", [{"{}"}], "Typed parameter shape bindings">,
    Option<"strides", "strides", "std::string", [{"{}"}], "Typed parameter stride bindings">,
    Option<"capabilities", "capabilities", "std::string", [{""}], "Typed BANG C hardware capabilities">
  ];
  let dependentDialects = ["intent::dsa::IntentDSADialect", "mlir::arith::ArithDialect", "mlir::func::FuncDialect", "mlir::math::MathDialect", "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect"];
}

def ResolveBlockProfiles : Pass<"intent-resolve-block-profiles", "mlir::ModuleOp"> {
  let summary = "Resolve finite profile inputs into the current Block module";
  let description = [{
    Reads the selected profile resource directory and explicit overrides,
    validates the finite tables and attaches typed profiles to the current
    Block module. A missing directory or invalid profile is an error. This is
    profile input resolution, not autotuning or configuration selection.
    Later passes consume current IR attributes without rereading these files.
    Shared-IR resume does not rerun this pass or accept replacement overrides.
  }];
  let options = [
    Option<"provider", "provider", "std::string", [{""}], "Provider owning the complete configuration table">,
    Option<"directory", "directory", "std::string", [{""}], "Directory of compiler profile resources">,
    Option<"overrides", "overrides", "std::string", [{""}], "Explicit profile overrides">
  ];
  let dependentDialects = ["intent::block::IntentBlockDialect"];
}
#endif
