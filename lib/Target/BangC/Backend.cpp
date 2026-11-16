#include "Intent/Compiler/Backend.h"
#include "Intent/Compiler/TargetOptions.h"
#include "Intent/Target/BangC/Passes.h"

namespace intent::compiler {
mlir::LogicalResult configureBangCTarget(mlir::ModuleOp module, Request &request) {
  auto options = readTargetOptions(module, request, {"architecture", "capabilities", "shapes", "strides"});
  if (mlir::failed(options)) return mlir::failure();
  auto architecture = options->getString("architecture");
  if (!architecture) return module.emitError("BANG C architecture must be a string");
  request.dsa.architecture = architecture->str();
  request.dsa.shapes = llvm::formatv("{0}", *options->get("shapes")).str();
  request.dsa.strides = llvm::formatv("{0}", *options->get("strides")).str();
  request.dsa.capabilities = llvm::formatv("{0}", *options->get("capabilities")).str();
  return mlir::success();
}

Backend createBangCBackend() {
  return {"bangc", "bangc", true, DSABackend{"bangc.json"}, nullptr,
      [] { bangc::registerBangCPasses(); bangc::registerBangCPipelines(); },
      [](mlir::OpPassManager &manager, const Request &request) {
        bangc::buildBangCPipeline(manager.nest<mlir::ModuleOp>(), request.dsa.architecture);
      }, bangc::serializeProgram, configureBangCTarget};
}
}
