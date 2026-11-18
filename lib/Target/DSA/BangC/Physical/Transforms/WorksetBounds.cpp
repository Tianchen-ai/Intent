#include "PassSupport.h"
#include "Intent/Target/DSA/BangC/Physical/Analysis/PhysicalProgram.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
namespace intent::bangc::physical {
#define GEN_PASS_DEF_DSASIMPLIFYWORKSETBOUNDS
#include "Intent/Target/DSA/BangC/Physical/Transforms/Passes.h.inc"

namespace {
class SimplifyWorksetBoundsPass final
    : public impl::DSASimplifyWorksetBoundsBase<SimplifyWorksetBoundsPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyRealizedProgram(module))) return signalPassFailure();
    bool changed = false;
    for (auto function : module.getOps<func::FuncOp>()) {
      SmallVector<Operation *> arithmetic;
      function.walk([&](Operation *operation) {
        if (operation->getDialect() == getContext().getLoadedDialect<arith::ArithDialect>() &&
            operation->getNumResults() == 1 &&
            operation->getResult(0).getType().isIntOrIndex() &&
            !isa<arith::ConstantOp>(operation))
          arithmetic.push_back(operation);
      });
      for (Operation *operation : arithmetic) {
        Value result = operation->getResult(0);
        if (auto remainder = dyn_cast<arith::RemSIOp>(operation)) {
          auto lhs = integerInterval(remainder.getLhs(), function);
          auto rhs = integerInterval(remainder.getRhs(), function);
          if (lhs && rhs && lhs->first >= 0 && rhs->first > lhs->second) {
            result.replaceAllUsesWith(remainder.getLhs());
            operation->erase();
            changed = true;
            continue;
          }
        }
        auto range = integerInterval(result, function);
        if (!range || range->first != range->second) continue;
        OpBuilder builder(operation);
        Value constant = builder.create<arith::ConstantOp>(operation->getLoc(),
            builder.getIntegerAttr(result.getType(), range->first));
        result.replaceAllUsesWith(constant);
        operation->erase();
        changed = true;
      }
    }
    LogicalResult result = success();
    if (changed) {
      OpPassManager cleanup(ModuleOp::getOperationName());
      cleanup.addPass(createCanonicalizerPass());
      cleanup.addPass(createCSEPass());
      result = runPipeline(cleanup, module);
    }
    if (failed(detail::finishTransform(module, getArgument(), result)))
      signalPassFailure();
  }
};
} // namespace
} // namespace intent::bangc::physical
