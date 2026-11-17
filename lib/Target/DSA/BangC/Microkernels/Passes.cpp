#include "Intent/Target/DSA/BangC/Microkernels.h"
#include "../Transforms/PassDetail.h"

using namespace mlir;
namespace intent::bangc {
#define GEN_PASS_DEF_BANGCMATRIXMICROKERNELS
#define GEN_PASS_DEF_BANGCCOLLECTIVEMICROKERNELS
#include "Intent/Target/DSA/BangC/Passes.h.inc"
namespace {
LogicalResult verifyInput(ModuleOp module) {
  if (failed(physical::verifyRealizedProgram(module))) return failure();
  auto architecture = module->getAttrOfType<StringAttr>("bangc.architecture");
  if (!architecture || architecture.getValue() != "mtp_372")
    return module.emitError("BANG C microkernels require the selected MLU370 implementation");
  return success();
}

class MatrixMicrokernelsPass : public impl::BangCMatrixMicrokernelsBase<MatrixMicrokernelsPass> {
public:
  using Base::Base;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyInput(module))) return signalPassFailure();
    for (auto function : module.getOps<func::FuncOp>()) {
      auto config = function->getAttrOfType<physical::ConfigurationAttr>("intent_bangc.configuration");
      SmallVector<physical::MatMulOp> operations;
      function.walk([&](physical::MatMulOp op) { operations.push_back(op); });
      for (auto operation : operations)
        if (failed(realizeMatMul(operation, config))) return signalPassFailure();
    }
    if (failed(physical::verifyRealizedProgram(module))) signalPassFailure();
  }
};

class CollectiveMicrokernelsPass : public impl::BangCCollectiveMicrokernelsBase<CollectiveMicrokernelsPass> {
public:
  using Base::Base;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyInput(module))) return signalPassFailure();
    for (auto function : module.getOps<func::FuncOp>())
      realizeRowSumChannels(function,
          function->getAttrOfType<physical::ConfigurationAttr>("intent_bangc.configuration"));
    if (failed(physical::verifyRealizedProgram(module))) signalPassFailure();
  }
};
}
}
