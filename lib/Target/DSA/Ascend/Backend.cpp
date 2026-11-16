#include "Intent/Compiler/Backend.h"
#include "Intent/Compiler/Models/Block.h"
#include "Intent/Compiler/TargetOptions.h"
#include "Intent/Dialect/Block/IR/BlockDialect.h"
#include "Intent/Target/DSA/Ascend/IR/Target.h"
#include "Intent/Target/Triton/IR/TritonDialect.h"
#include "Intent/Target/Triton/Serialization/Serializer.h"
#include "Intent/Target/Triton/Transforms/Configuration/TuningProfiles.h"
#include "Intent/Target/Triton/Transforms/Passes.h"

using namespace mlir;
namespace intent::compiler {
Backend createTritonAscendBackend() {
  auto profiles = triton::tuningProfileSchema();
  profiles.space = "triton_ascend";
  return {"triton_ascend", "triton_ascend", true,
      std::make_shared<BlockBackend>(1, profiles, "triton_ascend.json"),
      [](DialectRegistry &registry) { registry.insert<triton::IntentTritonDialect, ascend::IntentAscendDialect>(); },
      [] {},
      [](OpPassManager &manager, const Request &) { triton::buildTritonPipeline(manager); },
      triton::serializeProgram, [](ModuleOp module, Request &request) -> LogicalResult {
        auto options = readTargetOptions(module, request, {"architecture", "cube_cores", "vector_cores"});
        if (failed(options)) return failure();
        auto architecture = options->getString("architecture");
        if (!architecture || architecture->empty())
          return module.emitError("Ascend target requires its actual architecture");
        int64_t cube, vector;
        if (failed(readIntegerOption(module, *options, "cube_cores", cube)) ||
            failed(readIntegerOption(module, *options, "vector_cores", vector))) return failure();
        module.getContext()->getOrLoadDialect<ascend::IntentAscendDialect>();
        module.getContext()->getOrLoadDialect<block::IntentBlockDialect>();
        auto device = ascend::DeviceAttr::getChecked([&] { return module.emitError(); },
            module.getContext(), StringAttr::get(module.getContext(), *architecture), cube, vector);
        if (!device) return failure();
        request.target = block::CapabilitiesAttr::getChecked([&] { return module.emitError(); },
            module.getContext(), Attribute(device), cube > 0, false, true, false, true);
        return success(bool(request.target));
      }};
}
}
