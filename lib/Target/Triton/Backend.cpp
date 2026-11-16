#include "Intent/Compiler/Backend.h"
#include "Intent/Target/NVIDIA/IR/Target.h"
#include "Intent/Target/Triton/IR/TritonDialect.h"
#include "Intent/Target/Triton/Serialization/Serializer.h"
#include "Intent/Target/Triton/Transforms/Configuration/TuningProfiles.h"
#include "Intent/Target/Triton/Transforms/Passes.h"

namespace intent::compiler {
Backend createTritonBackend() {
  return {"triton", "triton", true,
      BlockBackend{true, false, true, 1, triton::tuningProfileSchema(), "triton.json"},
      [](mlir::DialectRegistry &registry) { registry.insert<triton::IntentTritonDialect, nvidia::IntentNVIDIADialect>(); },
      triton::registerTritonPasses,
      [](mlir::OpPassManager &manager, const Request &) { triton::buildTritonPipeline(manager); },
      triton::serializeProgram, configureNVIDIATarget};
}
}
