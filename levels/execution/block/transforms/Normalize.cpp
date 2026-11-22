#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Pass/Pass.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
struct Normalize : PassWrapper<Normalize, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Normalize)
  void runOnOperation() final {
    for (auto fn : getOperation().getOps<func::FuncOp>())
      if (intentdsl::kir::orderedOwners(fn) &&
          failed(intentdsl::block::forwardOwners(fn))) {
        signalPassFailure();
        return;
      }
    getOperation().walk([](Operation *op) {
      if (op->getName().getDialectNamespace() != "block" ||
          op->getNumResults() != 1)
        return;
      auto result = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!result || result.getRank() != 2)
        return;
      auto name = op->getName().stripDialect();
      if (name != "add" && name != "sub" && name != "mul" && name != "div")
        return;
      OpBuilder b(op);
      for (auto &operand : op->getOpOperands()) {
        auto t = dyn_cast<RankedTensorType>(operand.get().getType());
        if (!t || t.getRank() != 1)
          continue;
        OperationState expand(op->getLoc(), "block.broadcast_rows");
        expand.addOperands(operand.get());
        expand.addTypes(RankedTensorType::get({ShapedType::kDynamic, 1},
                                              t.getElementType()));
        operand.set(b.create(expand)->getResult(0));
      }
    });
  }
};
void normalize(OpPassManager &pm) { pm.addPass(std::make_unique<Normalize>()); }
} // namespace CI_UNIT_NAMESPACE
