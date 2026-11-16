#include "Intent/Compiler/Backend.h"
#include "Intent/Target/NVIDIA/IR/Target.h"
#include "Intent/Target/CuTile/IR/CuTileDialect.h"
#include "Intent/Target/CuTile/Serialization/Serializer.h"
#include "Intent/Target/CuTile/Transforms/Configuration/TuningProfiles.h"
#include "Intent/Target/CuTile/Transforms/Passes.h"

namespace intent::compiler {
Backend createCuTileBackend() {
  return {"cutile", "cutile", true,
      BlockBackend{true, true, false, 2, cutile::tuningProfileSchema(), "cutile.json"},
      [](mlir::DialectRegistry &registry) { registry.insert<cutile::IntentCuTileDialect, nvidia::IntentNVIDIADialect>(); },
      cutile::registerCuTilePasses,
      [](mlir::OpPassManager &manager, const Request &) { cutile::buildCuTilePipeline(manager); },
      cutile::serializeProgram, configureNVIDIATarget};
}
}
