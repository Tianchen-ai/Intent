#include "Intent/Compiler/Backend.h"
#include "Intent/Target/DSA/BangC/Compilation.h"
#include "Intent/Compiler/TargetOptions.h"
#include "Intent/Target/DSA/BangC/Passes.h"
#include "Intent/Target/DSA/BangC/Physical/Transforms/Passes.h"
#include "Intent/Target/DSA/BangC/ConstructionPasses.h"

namespace intent::compiler {
mlir::LogicalResult configureBangCTarget(mlir::ModuleOp module, Request &request) {
  auto options = readTargetOptions(module, request, {"architecture", "capabilities", "shapes", "strides"});
  if (mlir::failed(options)) return mlir::failure();
  auto architecture = options->getString("architecture");
  if (!architecture) return module.emitError("BANG C architecture must be a string");
  if (*architecture != "mtp_372")
    return module.emitError("BANG C supports the mtp_372 implementation only");
  auto hardware = readDSACapabilities(module, llvm::formatv("{0}", *options->get("capabilities")).str());
  auto shapes = parseDSAParameterBindings(module, llvm::formatv("{0}", *options->get("shapes")).str(), true);
  auto strides = parseDSAParameterBindings(module, llvm::formatv("{0}", *options->get("strides")).str(), false);
  if (mlir::failed(hardware) || mlir::failed(shapes) || mlir::failed(strides)) return mlir::failure();
  request.target = *hardware;
  mlir::Builder builder(module.getContext());
  request.bindings = builder.getDictionaryAttr({
      builder.getNamedAttr("architecture", builder.getStringAttr(*architecture)),
      builder.getNamedAttr("shapes", *shapes), builder.getNamedAttr("strides", *strides)});
  return mlir::success();
}

Backend createBangCBackend() {
  return {"bangc", "bangc", true, std::make_shared<BangCBackend>(),
      [](mlir::DialectRegistry &registry) { registry.insert<bangc::physical::IntentBangCPhysicalDialect>(); },
      [] {
        bangc::registerConstructionPasses();
        bangc::physical::registerDSAPasses();
        bangc::physical::registerDSAPipelines();
        bangc::registerBangCPasses(); bangc::registerBangCPipelines();
      },
      [](mlir::OpPassManager &manager, const Request &request) {
        bangc::buildBangCPipeline(manager.nest<mlir::ModuleOp>(),
            request.bindings.getAs<mlir::StringAttr>("architecture").getValue());
      }, bangc::serializeProgram, configureBangCTarget};
}
}
