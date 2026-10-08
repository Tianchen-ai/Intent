#include "../PassSupport.h"
#include "Intent/Dialect/DSA/Transforms/Collective/Collectives.h"

using namespace mlir;

namespace intent::dsa {

#define GEN_PASS_DEF_DSAREALIZECOLLECTIVES
#include "Intent/Dialect/DSA/Transforms/Passes.h.inc"

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
} // namespace intent::dsa
