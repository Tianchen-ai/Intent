#include "Intent/Compiler/Backend.h"
#include "Intent/Compiler/Models/CPU.h"
#include "Intent/Target/Mojo/Serialization/Serializer.h"
#include "Intent/Target/Mojo/Transforms/Passes.h"

namespace intent::compiler {
Backend createMojoBackend() {
  return {"mojo", "mojo", true, std::make_shared<CPUBackend>(true, "mojo.json", mojo::implementations(),
      cpu::CPUProgramStage::Buffers,
      [](mlir::OpPassManager &manager, const cpu::CPUCompilationOptions &options) {
        manager.addPass(cpu::createCPUBufferizeValues());
        cpu::buildCPUBufferPipeline(manager, options);
      }),
      nullptr, mojo::registerMojoPasses,
      [](mlir::OpPassManager &manager, const Request &) { mojo::buildMojoPipeline(manager); },
      mojo::serializeProgram, configureCPUTarget};
}
}
