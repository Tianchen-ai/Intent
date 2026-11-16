#ifndef INTENT_TARGET_BANGC_CONSTRUCTION_TD
#define INTENT_TARGET_BANGC_CONSTRUCTION_TD
include "mlir/Pass/PassBase.td"
def ConstructBangC : Pass<"intent-construct-bangc", "mlir::ModuleOp"> {
  let summary = "Construct BANG C physical candidates from complete configurations";
  let description = [{
    Bind complete BANG C configuration rows and explicit shape/stride facts,
    construct local storage and preserve author control and numerical semantics.
    Native storage arenas and transfer completion are realized by later passes.
  }];
  let options = [
    Option<"directory", "directory", "std::string", [{""}], "Configuration directory">,
    Option<"overrides", "overrides", "std::string", [{""}], "Complete configuration overrides">,
    Option<"shapes", "shapes", "std::string", [{"{}"}], "Typed parameter shape bindings">,
    Option<"strides", "strides", "std::string", [{"{}"}], "Typed parameter stride bindings">,
    Option<"capabilities", "capabilities", "std::string", [{""}], "Typed BANG C hardware capabilities">
  ];
  let dependentDialects = ["intent::block::IntentBlockDialect", "intent::bangc::physical::IntentBangCPhysicalDialect", "mlir::arith::ArithDialect", "mlir::func::FuncDialect", "mlir::math::MathDialect", "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect"];
}
#endif
