#include "ConfigurationFacts.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::triton {
ProgramConfigurationFacts::ProgramConfigurationFacts(func::FuncOp kernel) {
  bool mayFormDot = false, hasReductionOrScan = false;
  bool hasScaledOrSparse = false, loopMemory = false, straightLine = true;
  for (Attribute attribute : gpu::getParameterDeclarations(kernel)) {
      auto schema = cast<gpu::ParameterAttr>(attribute);
      auto category = schema.getCategory();
      if (category != gpu::ParameterCategory::Coverage &&
          category != gpu::ParameterCategory::Provider &&
          !llvm::is_contained(categories, category))
        categories.push_back(category);
  }
  kernel.walk([&](Operation *operation) {
    mayFormDot |= isa<gpu::ReduceOp, ReduceOp>(operation);
    hasReductionOrScan |= isa<gpu::ReduceOp, gpu::ScanOp>(operation);
    if (auto contract = dyn_cast<gpu::ContractOp>(operation)) {
      hasContraction = true;
      allContractionsF32 &= contract.getLhs().getType().getElementType().isF32() &&
          contract.getRhs().getType().getElementType().isF32() &&
          contract.getResult().getType().getElementType().isF32();
    }
    hasScaledOrSparse |= isa<gpu::ScaledContractOp, gpu::SparseContractOp>(operation);
    loopMemory |= operation->getParentOfType<scf::ForOp>() && !isMemoryEffectFree(operation);
    if (operation->getBlock() == &kernel.front())
      straightLine &= operation->getNumRegions() == 0;
  });
  if (hasReductionOrScan && !llvm::is_contained(categories, gpu::ParameterCategory::Reduction))
    categories.push_back(gpu::ParameterCategory::Reduction);
  if (hasContraction && !llvm::is_contained(categories, gpu::ParameterCategory::Contraction))
    categories.push_back(gpu::ParameterCategory::Contraction);
  straightLinePointwise = straightLine &&
      llvm::is_contained(categories, gpu::ParameterCategory::Pointwise) &&
      llvm::all_of(categories, [](gpu::ParameterCategory category) {
        return category == gpu::ParameterCategory::Pointwise;
      });
  pipelineStagesAffectProgram = (hasContraction && !allContractionsF32) ||
      hasScaledOrSparse || ((hasContraction || mayFormDot) && loopMemory);
}

} // namespace intent::triton
