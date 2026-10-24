#include "Legalize.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace intent::mojo {

void isolateLargeComputations(ModuleOp module) {
  // Bound the structured body optimized as one Mojo function. Keep small
  // register loops together, and never pass hardware tile values across a
  // function boundary. This changes no task, storage or arithmetic ordering.
  constexpr unsigned minimum = 256;
  llvm::DenseMap<Operation *, unsigned> sizes;
  SmallVector<Operation *> selected;
  module.walk<WalkOrder::PostOrder>([&](Operation *operation) {
    unsigned size = 1;
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (Operation &child : block) size += sizes.lookup(&child);
    sizes[operation] = size;
    if (size < minimum || operation->getNumResults() ||
        !isa<scf::IfOp, scf::ForOp>(operation)) return WalkResult::advance();
    llvm::SetVector<Value> captures;
    for (Region &region : operation->getRegions())
      getUsedValuesDefinedAbove(region, region, captures);
    captures.insert(operation->operand_begin(), operation->operand_end());
    if (!llvm::all_of(captures, [](Value value) {
          Type type = value.getType();
          return type.isIntOrIndexOrFloat() || isa<MemRefType, VectorType>(type);
        })) return WalkResult::advance();
    if (operation->walk([](cpu::TaskDispatchOp) {
          return WalkResult::interrupt();
        }).wasInterrupted()) return WalkResult::advance();
    selected.push_back(operation);
    // An enclosing scope sees a call after its children have been isolated.
    // This also bounds a large outer CFG instead of only cutting its leaves.
    sizes[operation] = 1;
    return WalkResult::advance();
  });
  for (Operation *operation : selected) {
    OpBuilder builder(operation);
    auto scope = builder.create<scf::ExecuteRegionOp>(operation->getLoc(), TypeRange{});
    Block *body = builder.createBlock(&scope.getRegion());
    operation->moveBefore(body, body->end());
    builder.setInsertionPointToEnd(body);
    builder.create<scf::YieldOp>(scope.getLoc());
    llvm::SetVector<Value> captures;
    getUsedValuesDefinedAbove(scope.getRegion(), scope.getRegion(), captures);
    builder.setInsertionPointToStart(body);
    for (Value value : captures) {
      Operation *definition = value.getDefiningOp();
      if (!definition || !definition->hasTrait<OpTrait::ConstantLike>()) continue;
      Operation *copy = builder.clone(*definition);
      for (auto [before, after] : llvm::zip(definition->getResults(), copy->getResults()))
        replaceAllUsesInRegionWith(before, after, scope.getRegion());
    }
  }
}

} // namespace intent::mojo
