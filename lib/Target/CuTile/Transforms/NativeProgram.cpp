#include "NativeRewrite.h"
#include "Access/Accesses.h"
#include "Compute/ComputeForms.h"
#include "Intent/Target/CuTile/Analysis/Program.h"
#include "Program.h"
#include "mlir/IR/Verifier.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <cassert>

using namespace mlir;
namespace intent::cutile {
LogicalResult formNativeProgram(ModuleOp module) {
  auto physicalKernel = gpu::getPhysicalKernel(module);
  if (failed(physicalKernel)) return failure();
  func::FuncOp kernel = *physicalKernel;
  NativeAccessInputs accesses;
  NativeComputeInputs computations;
  SmallVector<gpu::AssumeInBoundsOp> assumptions;
  NativeProgramFeatures features;
  kernel.walk([&](Operation *operation) {
    features.observe(operation);
    if (auto op = dyn_cast<gpu::LoadOp>(operation)) accesses.loads.push_back(op);
    else if (auto op = dyn_cast<gpu::GatherOp>(operation)) accesses.gathers.push_back(op);
    else if (auto op = dyn_cast<gpu::StoreOp>(operation)) accesses.stores.push_back(op);
    else if (auto op = dyn_cast<gpu::AtomicRMWOp>(operation)) accesses.atomics.push_back(op);
    else if (auto op = dyn_cast<gpu::ContractOp>(operation)) computations.contracts.push_back(op);
    else if (auto op = dyn_cast<gpu::ScaledContractOp>(operation)) computations.scaledContracts.push_back(op);
    else if (auto op = dyn_cast<gpu::ReduceOp>(operation)) computations.reductions.push_back(op);
    else if (auto op = dyn_cast<gpu::HistogramOp>(operation)) computations.histograms.push_back(op);
    else if (auto op = dyn_cast<gpu::ScanOp>(operation)) computations.scans.push_back(op);
    else if (auto op = dyn_cast<gpu::AssumeInBoundsOp>(operation)) assumptions.push_back(op);
  });
  NativeFormRewriter rewriter;
  if (failed(formNativeAccesses(kernel, accesses, features.matrixCompute,
                                rewriter)) ||
      failed(formComputePrimitives(kernel, computations, rewriter)))
    return failure();
  rewriter.commit();
  for (gpu::AssumeInBoundsOp assumption : assumptions)
    assumption.erase();
  gpu::eraseDeadPhysicalValues(kernel);
  return mlir::verify(module);
}
} // namespace intent::cutile
