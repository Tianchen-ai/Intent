#include "../PassSupport.h"
#include "Intent/Target/DSA/BangC/Physical/Transforms/Collective/Collectives.h"

using namespace mlir;

namespace intent::bangc::physical {

#define GEN_PASS_DEF_DSAREALIZECOLLECTIVES
#include "Intent/Target/DSA/BangC/Physical/Transforms/Passes.h.inc"

namespace {

struct RealizeCollectivesPass
    : impl::DSARealizeCollectivesBase<RealizeCollectivesPass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyProgram(module)))
      return signalPassFailure();
    auto function = *module.getOps<func::FuncOp>().begin();
    if (failed(realizeCollectives(function))) {
      module.emitError() << "DSA transformation failed: " << getArgument();
      return signalPassFailure();
    }
    if (failed(verifyRealizedProgram(module)))
      signalPassFailure();
  }
};

} // namespace
} // namespace intent::bangc::physical
