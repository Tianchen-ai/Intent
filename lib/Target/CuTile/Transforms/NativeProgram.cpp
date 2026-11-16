#include "NativeRewrite.h"
#include "Access/Accesses.h"
#include "Compute/ComputeForms.h"
#include "Intent/Target/CuTile/Analysis/Program.h"
#include "Program.h"
#include "mlir/IR/Verifier.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <cassert>

using namespace mlir;
namespace intent::cutile {
LogicalResult formNativeProgram(ModuleOp module) {
  auto physicalKernel = block::getPhysicalKernel(module);
  if (failed(physicalKernel)) return failure();
  func::FuncOp kernel = *physicalKernel;
  NativeAccessInputs accesses;
  NativeComputeInputs computations;
  SmallVector<block::AssumeInBoundsOp> assumptions;
  NativeProgramFeatures features;
  kernel.walk([&](Operation *operation) {
    features.observe(operation);
    if (auto op = dyn_cast<block::LoadOp>(operation)) accesses.loads.push_back(op);
    else if (auto op = dyn_cast<block::GatherOp>(operation)) accesses.gathers.push_back(op);
    else if (auto op = dyn_cast<block::StoreOp>(operation)) accesses.stores.push_back(op);
    else if (auto op = dyn_cast<block::AtomicRMWOp>(operation)) accesses.atomics.push_back(op);
    else if (auto op = dyn_cast<block::ContractOp>(operation)) computations.contracts.push_back(op);
    else if (auto op = dyn_cast<block::ScaledContractOp>(operation)) computations.scaledContracts.push_back(op);
    else if (auto op = dyn_cast<block::ReduceOp>(operation)) computations.reductions.push_back(op);
    else if (auto op = dyn_cast<block::ScanOp>(operation)) computations.scans.push_back(op);
    else if (auto op = dyn_cast<block::AssumeInBoundsOp>(operation)) assumptions.push_back(op);
  });
  NativeFormRewriter rewriter;
  if (failed(formNativeAccesses(kernel, accesses, features.matrixCompute,
                                rewriter)) ||
      failed(formComputePrimitives(kernel, computations, rewriter)))
    return failure();
  rewriter.commit();
  for (block::AssumeInBoundsOp assumption : assumptions)
    assumption.erase();
  block::eraseDeadPhysicalValues(kernel);
  return mlir::verify(module);
}
} // namespace intent::cutile
