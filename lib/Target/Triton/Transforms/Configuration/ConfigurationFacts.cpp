#include "ConfigurationFacts.h"
#include "Intent/Dialect/Block/Analysis/PhysicalParameters.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::triton {
ProgramConfigurationFacts::ProgramConfigurationFacts(func::FuncOp kernel) {
  bool mayFormDot = false, hasReductionOrScan = false;
  bool hasScaledOrSparse = false, loopMemory = false, straightLine = true;
  for (Attribute attribute : block::getParameterDeclarations(kernel)) {
      auto schema = cast<block::ParameterAttr>(attribute);
      auto category = schema.getCategory();
      if (category != block::ParameterCategory::Coverage &&
          category != block::ParameterCategory::Provider &&
          !llvm::is_contained(categories, category))
        categories.push_back(category);
  }
  kernel.walk([&](Operation *operation) {
    mayFormDot |= isa<block::ReduceOp, ReduceOp>(operation);
    hasReductionOrScan |= isa<block::ReduceOp, block::ScanOp>(operation);
    if (auto contract = dyn_cast<block::ContractOp>(operation)) {
      hasContraction = true;
      allContractionsF32 &= contract.getLhs().getType().getElementType().isF32() &&
          contract.getRhs().getType().getElementType().isF32() &&
          contract.getResult().getType().getElementType().isF32();
    }
    hasScaledOrSparse |= isa<block::ScaledContractOp, block::SparseContractOp>(operation);
    loopMemory |= operation->getParentOfType<scf::ForOp>() && !isMemoryEffectFree(operation);
    if (operation->getBlock() == &kernel.front())
      straightLine &= operation->getNumRegions() == 0;
  });
  if (hasReductionOrScan && !llvm::is_contained(categories, block::ParameterCategory::Reduction))
    categories.push_back(block::ParameterCategory::Reduction);
  if (hasContraction && !llvm::is_contained(categories, block::ParameterCategory::Contraction))
    categories.push_back(block::ParameterCategory::Contraction);
  straightLinePointwise = straightLine &&
      llvm::is_contained(categories, block::ParameterCategory::Pointwise) &&
      llvm::all_of(categories, [](block::ParameterCategory category) {
        return category == block::ParameterCategory::Pointwise;
      });
  pipelineStagesAffectProgram = (hasContraction && !allContractionsF32) ||
      hasScaledOrSparse || ((hasContraction || mayFormDot) && loopMemory);
}

} // namespace intent::triton
