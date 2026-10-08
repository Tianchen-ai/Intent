#include "ContractionDetail.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <functional>
#include <limits>
#include <tuple>


using namespace mlir;

namespace intent::gpu::contraction {



class ContractionWorklist {
public:
  struct Item {
    ContractOp operation;
    bool retiledEpilogue;
  };

  explicit ContractionWorklist(func::FuncOp kernel) {
    kernel.walk([&](ContractOp contract) {
      append(ArrayRef<ContractOp>(contract));
    });
  }

  void append(ArrayRef<ContractOp> products, bool retiledEpilogue = false) {
    for (ContractOp product : products) {
      if (scheduled.insert(product.getOperation()).second)
        entries.push_back(product);
      if (retiledEpilogue)
        epilogues.insert(product.getOperation());
    }
  }

  std::optional<Item> take() {
    if (cursor == entries.size())
      return std::nullopt;
    ContractOp operation = entries[cursor++];
    scheduled.erase(operation.getOperation());
    bool epilogue = epilogues.erase(operation.getOperation());
    return Item{operation, epilogue};
  }

private:
  SmallVector<ContractOp> entries;
  llvm::SmallPtrSet<Operation *, 16> scheduled;
  llvm::SmallPtrSet<Operation *, 16> epilogues;
  size_t cursor = 0;
};



} // namespace intent::gpu::contraction

namespace intent::gpu {
using namespace contraction;

static LogicalResult realizeContractionBlockingImpl(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  fuseContractionAdds(kernel);
  if (failed(realizeAccessComposition(module)))
    return failure();

  // Source normalization and physical preparation see different axis
  // realizations. Keep both opportunities, but share the snapshot rewrite.
  auto collapsed = rewriteContractionSnapshot(kernel, collapseMultiReductionContract);
  if (failed(collapsed) || (*collapsed && failed(realizeAccessComposition(module))))
    return failure();
  auto decomposed = rewriteContractionSnapshot(kernel, [](ContractOp contract)
      -> FailureOr<bool> {
    if (contract.getLhsReductionAxes().size() <= 1)
      return false;
    if (failed(decomposeMultiReductionContract(contract)))
      return failure();
    return true;
  });
  if (failed(decomposed) || failed(normalizeMatrixContractForms(kernel)) ||
      failed(prepareRetainedContractionReads(kernel)))
    return failure();

  ContractionWorklist worklist(kernel);
  auto realizeSelectedContract = [&](ContractOp contract) -> LogicalResult {
    SmallVector<ContractOp> created;
    if (failed(realizeContract(contract, kernel, created)))
      return failure();
    worklist.append(created, /*retiledEpilogue=*/true);
    return success();
  };
  while (auto item = worklist.take()) {
    ContractOp contract = item->operation;
    if (contract.getResult().use_empty()) {
      contract.erase();
      continue;
    }
    if (!hasFragmentSchema(contract))
      return contract.emitOpError(
          "shared contraction blocking requires fragment operands and result");

    const bool needsRealization = requiresPhysicalRealization(contract);
    const bool rangeSingleReduction = hasRangeContractForm(contract);
    // A complete result/store path owns its epilogue before any retained
    // full-result representation could hide that ownership.
    if (!needsRealization || !rangeSingleReduction) {
      SmallVector<ContractOp> created;
      auto fullResult = realizeFullResultTraversal(contract, kernel, created);
      if (failed(fullResult))
        return failure();
      worklist.append(created);
      if (*fullResult)
        continue;
    }
    if (outputCoordinatesNeedRealization(contract)) {
      if (failed(realizeSelectedContract(contract)))
        return failure();
      continue;
    }
    if (!needsRealization) {
      if (failed(markNativeCoverage(kernel, contract)))
        return failure();
      continue;
    }
    auto nativeSegment = realizeSegmentNativeReduction(contract, kernel);
    if (failed(nativeSegment))
      return failure();
    if (*nativeSegment)
      continue;
    SmallVector<ContractOp> created;
    auto nativeStructured =
        realizeStructuredNativeReduction(contract, kernel, created);
    if (failed(nativeStructured))
      return failure();
    worklist.append(created);
    if (*nativeStructured)
      continue;

    const bool singleReduction =
        contract.getLhsReductionAxes().size() == 1 &&
        contract.getRhsReductionAxes().size() == 1;
    if ((!rangeSingleReduction || item->retiledEpilogue) &&
        singleReduction &&
        freeAxesReadyForReductionTraversal(contract, kernel) &&
        reductionAxesNeedTraversal(contract, kernel) &&
        hasExplicitPairedReductionRanges(contract)) {
      created.clear();
      if (failed(realizeReductionTraversal(contract, kernel, created)))
        return failure();
      worklist.append(created);
    } else if (!rangeSingleReduction && singleReduction &&
               contract.getLhsBatchAxes().empty() &&
               contract.getRhsBatchAxes().empty() &&
               hasCompleteEpilogue(contract) &&
               freeAxesNeedRealization(contract, kernel)) {
      if (failed(realizeSelectedContract(contract)))
        return failure();
    } else if (!rangeSingleReduction && singleReduction) {
      if (failed(markNativeCoverage(kernel, contract)))
        return failure();
    } else if (failed(realizeSelectedContract(contract))) {
      return failure();
    }
  }

  SmallVector<SparseContractOp> sparseContracts;
  SmallVector<ScaledContractOp> scaledContracts;
  kernel.walk(
      [&](SparseContractOp contract) { sparseContracts.push_back(contract); });
  kernel.walk(
      [&](ScaledContractOp contract) { scaledContracts.push_back(contract); });
  for (ScaledContractOp contract : scaledContracts)
    if (!hasFragmentSchema(contract))
      return contract.emitOpError(
          "shared scaled-contraction blocking requires fragment operands, accumulator, and result");
  for (SparseContractOp contract : sparseContracts)
    if (!hasFragmentSchema(contract))
      return contract.emitOpError(
          "shared sparse-contraction blocking requires fragment operands, metadata, accumulator, and result");
  for (SparseContractOp contract : sparseContracts)
    if (failed(realizeSparseReductionTraversal(contract, kernel)))
      return failure();
  for (ScaledContractOp contract : scaledContracts)
    if (requiresPhysicalRealization(contract)) {
      if (failed(realizeScaledContract(contract, kernel)))
        return failure();
    } else if (failed(markNativeCoverage(kernel, contract))) {
      return failure();
    }

  eraseDeadPhysicalValues(kernel);
  WalkResult tails = kernel.walk([&](ContractOp contract) {
    if (failed(neutralizeFullCoverageOperand(
            contract, contract.getLhsMutable(), contract.getLhsReductionAxes(),
            kernel)) ||
        failed(neutralizeFullCoverageOperand(
            contract, contract.getRhsMutable(), contract.getRhsReductionAxes(),
            kernel)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (tails.wasInterrupted())
    return failure();
  eraseDeadPhysicalValues(kernel);
  pruneContractionProgramCoordinates(kernel);
  eraseDeadPhysicalValues(kernel);
  return success();
}

LogicalResult realizeContractionBlocking(ModuleOp module) {
  if (failed(realizeContractionBlockingImpl(module))) return failure();
  auto kernel = getPhysicalKernel(module);
  return failed(kernel) ? failure() : closeValueRelations(*kernel);
}

} // namespace intent::gpu
