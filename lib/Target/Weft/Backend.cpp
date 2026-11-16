#include "Intent/Compiler/Backend.h"
#ifdef INTENT_HAS_WEFT_CANONICAL
#include "Intent/Target/Weft/IR/WeftDialect.h"
#include "Intent/Target/Weft/Serialization/Serializer.h"
#include "Intent/Target/Weft/Transforms/Passes.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#endif

namespace intent::compiler {
Backend createWeftBackend() {
#ifdef INTENT_HAS_WEFT_CANONICAL
  return {"weft", "weft", true,
      CPUBackend{false, "weft.json", weft_provider::implementations},
      [](mlir::DialectRegistry &registry) {
        registry.insert<weft_provider::IntentWeftDialect, ::weft::kernel::WEFTKernelDialect>();
      }, weft_provider::registerWeftPasses,
      [](mlir::OpPassManager &manager, const Request &) { weft_provider::buildWeftPipeline(manager); },
      weft_provider::serializeProgram, configureCPUTarget};
#else
  return {"weft", "weft", false, CPUBackend{false, "weft.json", nullptr},
      nullptr, nullptr, nullptr, nullptr, configureCPUTarget};
#endif
}
}
