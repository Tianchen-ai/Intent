#include "Intent/Compiler/Backend.h"
#include "Intent/Target/CuTile/IR/CuTileDialect.h"
#include "Intent/Target/CuTile/Serialization/Serializer.h"
#include "Intent/Target/CuTile/Transforms/Configuration/TuningProfiles.h"
#include "Intent/Target/CuTile/Transforms/Passes.h"

namespace intent::compiler {
Backend createCuTileBackend() {
  return {"cutile", "cutile", true,
      GPUBackend{true, true, false, 2, cutile::tuningProfileSchema(), "cutile.json"},
      [](mlir::DialectRegistry &registry) { registry.insert<cutile::IntentCuTileDialect>(); },
      cutile::registerCuTilePasses,
      [](mlir::OpPassManager &manager, const Request &) { cutile::buildCuTilePipeline(manager); },
      cutile::serializeProgram, configureNVIDIATarget};
}
}
