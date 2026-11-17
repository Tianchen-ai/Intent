#include "../PassSupport.h"
#include "ProducerReuse.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
namespace intent::cpu {
#define GEN_PASS_DEF_CPUFUSETENSORCOMPUTATIONS
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

namespace {
bool fusibleProducer(OpOperand *operand) {
  auto producer = operand->get().getDefiningOp<linalg::GenericOp>();
  auto consumer = dyn_cast<linalg::GenericOp>(operand->getOwner());
  if (!producer || !consumer || consumer.getOutputs().size() != 1 ||
      consumer.getNumReductionLoops()) return false;
  // Standard tensor fusion does not protect mutable captures in the payload.
  if (producer.getRegion().walk([](Operation *nested) {
    return isMemoryEffectFree(nested) ? WalkResult::advance() : WalkResult::interrupt();
  }).wasInterrupted()) return false;
  if (!linalg::getPreservedProducerResults(producer, consumer, operand).empty()) return false;
  unsigned resultNumber = cast<OpResult>(operand->get()).getResultNumber();
  AffineMap outputMap = producer.getMatchingIndexingMap(producer.getDpsInitOperand(resultNumber));
  if (!outputMap.isPermutation()) return false;
  AffineMap coordinates = inversePermutation(outputMap).compose(consumer.getMatchingIndexingMap(operand));
  StorageAnalysis storage(producer->getParentOfType<func::FuncOp>());
  auto payload = analyzeProducerResult(producer, storage, resultNumber);
  if (failed(payload)) return false;
  bool removesProducer = llvm::all_of(producer->getResults(), [&](Value value) {
    return llvm::all_of(value.getUses(), [&](OpOperand &use) { return &use == operand; });
  });
  Value yielded = producer.getRegion().front().getTerminator()->getOperand(resultNumber);
  return shouldFuseProducerResult(*payload, yielded, coordinates,
                                 consumer.getStaticLoopRanges(), removesProducer);
}

class FuseTensorComputationsPass
    : public impl::CPUFuseTensorComputationsBase<FuseTensorComputationsPass> {
public:
  using CPUFuseTensorComputationsBase::CPUFuseTensorComputationsBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(detail::verifyPassInput(module, getArgument(), CPUProgramStage::Values)))
      return signalPassFailure();
    RewritePatternSet patterns(&getContext());
    linalg::populateElementwiseOpsFusionPatterns(patterns, fusibleProducer);
    if (failed(detail::finishTransform(module, getArgument(),
            applyPatternsGreedily(module, std::move(patterns)), CPUProgramStage::Values)))
      signalPassFailure();
  }
};
} // namespace
} // namespace intent::cpu
