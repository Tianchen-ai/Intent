#include "Legalize.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

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
  llvm::SmallPtrSet<Operation *, 32> boundaries(selected.begin(), selected.end());
  llvm::SmallPtrSet<Operation *, 32> matrixOwners, matrixScopes;
  module.walk([&](Operation *operation) {
    auto tile = [](Type type) { return isa<amx::TileType>(type); };
    if (!llvm::any_of(operation->getOperandTypes(), tile) &&
        !llvm::any_of(operation->getResultTypes(), tile)) return;
    bool owned = false;
    for (Operation *parent = operation->getParentOp(); parent; parent = parent->getParentOp()) {
      bool boundary = boundaries.contains(parent);
      bool entry = isa<cpu::TaskDispatchOp, func::FuncOp>(parent);
      if (boundary) matrixScopes.insert(parent);
      if (!owned && (boundary || entry)) {
        matrixOwners.insert(parent);
        owned = true;
      }
      if (entry) break;
    }
  });
  // Tile configuration is caller-saved even when no tile SSA crosses a call.
  // A function with direct AMX operations must not call an outlined AMX child:
  // that child's release can invalidate the caller's configured tile state.
  // Keep such nested microprograms together; independent AMX siblings and
  // non-matrix computation scopes can still be isolated.
  llvm::erase_if(selected, [&](Operation *operation) {
    if (!matrixScopes.contains(operation)) return false;
    for (Operation *parent = operation->getParentOp(); parent; parent = parent->getParentOp()) {
      if (matrixOwners.contains(parent)) return true;
      if (isa<cpu::TaskDispatchOp, func::FuncOp>(parent)) break;
    }
    return false;
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
