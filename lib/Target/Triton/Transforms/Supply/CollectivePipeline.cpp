#include "Intent/Target/Triton/Transforms/Passes.h"
#include "Intent/Target/Triton/IR/TritonDialect.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "Intent/Target/NVIDIA/IR/Target.h"
#include "Intent/Dialect/Block/IR/BlockDialect.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "Intent/Analysis/ControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
namespace intent::triton {
#define GEN_PASS_DEF_MIXEDCOLLECTIVEPIPELINEPASS
#include "Intent/Target/Triton/Transforms/Passes.h.inc"

namespace {
llvm::DenseSet<Value> inputLoads(ValueRange roots, scf::ForOp loop) {
  llvm::DenseSet<Value> visited, loads;
  SmallVector<Value> pending(roots);
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second) continue;
    Operation *producer = value.getDefiningOp();
    if (!producer || !loop->isProperAncestor(producer)) continue;
    if (isa<block::LoadOp>(producer)) {
      loads.insert(value);
      continue;
    }
    auto incoming = queryControlFlowIncoming(value);
    if (incoming.complete && !incoming.edges.empty()) {
      for (const auto &edge : incoming.edges)
        if (edge.operand) pending.push_back(edge.operand->get());
      continue;
    }
    if (producer->getNumRegions() || !isMemoryEffectFree(producer)) continue;
    llvm::append_range(pending, producer->getOperands());
  }
  return loads;
}

class MixedCollectivePipelinePass final
    : public impl::MixedCollectivePipelinePassBase<MixedCollectivePipelinePass> {
public:
  void runOnOperation() final {
    auto kernel = block::getPhysicalKernel(getOperation());
    if (failed(kernel)) return signalPassFailure();
    auto device = nvidia::deviceOf(*kernel);
    if (!device || device.getComputeCapabilityMajor() != 9) return;
    auto options = readCompileOptions(getOperation());
    if (failed(options)) return signalPassFailure();
    kernel->walk([&](scf::ForOp loop) {
      llvm::DenseSet<Value> matrixLoads, collectiveLoads;
      loop.walk([&](block::ContractOp contract) {
        auto element = contract.getLhs().getType().getElementType();
        if (!element.isF16() && !element.isBF16()) return;
        auto loads = inputLoads({contract.getLhs(), contract.getRhs()}, loop);
        matrixLoads.insert(loads.begin(), loads.end());
      });
      if (matrixLoads.empty()) return;
      loop.walk([&](block::ReduceOp reduce) {
        auto loads = inputLoads(reduce.getSources(), loop);
        collectiveLoads.insert(loads.begin(), loads.end());
      });
      if (!llvm::any_of(matrixLoads, [&](Value value) { return collectiveLoads.contains(value); })) return;
      // The supported Hopper native pipeline can reuse shared operand storage
      // before a mixed reduction/WGMMA consumer completes. Keep this loop in
      // its native single-stage form; other matrix loops retain their tuning.
      // This is provider legalization, not a restriction of Block semantics or
      // a new layout/synchronization implementation in Intent.
      loop->setAttr(loopStagesAttr, IntegerAttr::get(IntegerType::get(&getContext(), 64), 1));
      if (options->getOptimizationRemarks())
        loop.emitRemark("Hopper mixed collective/matrix supply uses one native stage");
    });
  }
};
} // namespace
} // namespace intent::triton
