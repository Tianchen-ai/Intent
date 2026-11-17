#include "Intent/Compiler/Backend.h"
#include "Intent/Compiler/Models/CPU.h"
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
      std::make_shared<CPUBackend>(false, "weft.json", weft_provider::implementations(),
          cpu::CPUProgramStage::Values, nullptr),
      [](mlir::DialectRegistry &registry) {
        registry.insert<weft_provider::IntentWeftDialect, ::weft::kernel::WEFTKernelDialect>();
      }, weft_provider::registerWeftPasses,
      [](mlir::OpPassManager &manager, const Request &) {
        cpu::CPUCompilationOptions options;
        options.provider = "weft";
        manager.addPass(cpu::createCPUBufferizeValues());
        cpu::buildCPUBufferPipeline(manager, options);
        weft_provider::buildWeftPipeline(manager);
      },
      weft_provider::serializeProgram, configureCPUTarget};
#else
  return {"weft", "weft", false,
      std::make_shared<CPUBackend>(false, "weft.json", cpu::ImplementationRegistry{},
                                  cpu::CPUProgramStage::Values, nullptr),
      nullptr, nullptr, nullptr, nullptr, configureCPUTarget};
#endif
}
}
