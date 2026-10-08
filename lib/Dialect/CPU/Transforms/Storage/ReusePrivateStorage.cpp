#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <memory>

using namespace mlir;

namespace intent::cpu {

void eraseDeadPrivateBuffers(func::FuncOp function) {
  SmallVector<memref::AllocOp> allocations;
  function.walk([&](memref::AllocOp allocation) { allocations.push_back(allocation); });
  std::unique_ptr<StorageAnalysis> storage;
  for (auto allocation : llvm::reverse(allocations)) {
    SmallVector<Operation *> users;
    llvm::SmallPtrSet<Operation *, 8> seen;
    for (Operation *user : allocation.getResult().getUsers())
      if (seen.insert(user).second) users.push_back(user);
    bool unused = llvm::all_of(users, [&](Operation *user) {
      if (auto generic = dyn_cast<linalg::LinalgOp>(user)) {
        if (generic.getNumDpsInits() != 1 || generic->getNumResults() ||
            generic.getDpsInits()[0] != allocation.getResult()) return false;
        return llvm::all_of(generic->getRegion(0).front().without_terminator(), [](Operation &nested) {
          return !nested.getNumRegions() && isMemoryEffectFree(&nested);
        });
      }
      if (auto copy = dyn_cast<memref::CopyOp>(user)) return copy.getTarget() == allocation.getResult();
      if (auto store = dyn_cast<memref::StoreOp>(user)) return store.getMemref() == allocation.getResult();
      if (auto store = dyn_cast<vector::StoreOp>(user)) return store.getBase() == allocation.getResult();
      if (auto dimension = dyn_cast<memref::DimOp>(user)) return dimension.getConstantIndex().has_value();
      return isa<memref::DeallocOp>(user);
    });
    if (!unused) continue;
    if (!storage) storage = std::make_unique<StorageAnalysis>(function);
    {
      auto aliases = storage->aliases(allocation);
      if (!aliases.complete || aliases.values.size() != 1) continue;
    }
    // Failed candidates share this unchanged snapshot. Drop it before any
    // dimension replacement or erasure changes the current function.
    storage.reset();
    for (Operation *user : users) {
      if (auto dimension = dyn_cast<memref::DimOp>(user)) {
        OpBuilder b(dimension);
        int64_t axis = *dimension.getConstantIndex();
        Value extent = allocation.getType().isDynamicDim(axis)
            ? allocation.getDynamicSizes()[allocation.getType().getDynamicDimIndex(axis)]
            : Value(b.create<arith::ConstantIndexOp>(dimension.getLoc(), allocation.getType().getDimSize(axis)));
        dimension.getResult().replaceAllUsesWith(extent);
      }
      user->erase();
    }
    allocation.erase();
  }
}

namespace {

bool reusePrivateInput(linalg::GenericOp consumer, Value buffer,
                       StorageAnalysis &storage) {
  auto source = buffer.getDefiningOp<memref::AllocOp>();
  if (!source || consumer.getNumResults() || consumer.getOutputs().size() != 1 ||
      consumer.getNumReductionLoops() || source->getBlock() != consumer->getBlock()) return false;
  auto output = consumer.getOutputs()[0].getDefiningOp<memref::AllocOp>();
  if (!output || output == source || output->getBlock() != source->getBlock() ||
      !source->isBeforeInBlock(output) || !output->isBeforeInBlock(consumer) ||
      source.getType() != output.getType() || !source.getType().getLayout().isIdentity() ||
      !llvm::equal(source.getDynamicSizes(), output.getDynamicSizes()) ||
      source.getAlignment().value_or(0) < output.getAlignment().value_or(0)) return false;
  Block &body = consumer.getRegion().front();
  auto pure = [](Block &block) {
    return llvm::all_of(block.without_terminator(), [](Operation &operation) {
      return !operation.getNumRegions() && isMemoryEffectFree(&operation);
    });
  };
  if (!body.getArguments().back().use_empty() || !pure(body)) return false;
  auto maps = consumer.getIndexingMapsArray();
  if (!maps.back().isIdentity()) return false;
  bool readsSource = false;
  for (auto [number, input] : llvm::enumerate(consumer.getInputs())) {
    if (!isa<MemRefType>(input.getType())) continue;
    Value root = storage.uniqueOrigin(input);
    if (root == output.getResult()) return false;
    if (root == buffer) {
      if (input != buffer || maps[number] != maps.back()) return false;
      readsSource |= !body.getArgument(number).use_empty();
    } else if (!storage.disjoint(input, buffer)) return false;
  }
  if (!readsSource) return false;

  auto lifetime = [&](memref::AllocOp allocation, bool oldValues) -> memref::DeallocOp {
    auto lifetime = storage.lifetime(allocation);
    if (!lifetime || !lifetime->aliases.complete) return {};
    for (Operation *user : lifetime->aliases.users) {
      if (user == lifetime->end || isStorageAliasOperation(user) || isa<memref::DimOp>(user)) continue;
      if (auto computation = dyn_cast<linalg::LinalgOp>(user)) {
        if (computation->getNumResults() || computation->getNumRegions() != 1 ||
            !pure(computation->getRegion(0).front())) return {};
      } else if (auto reduction = dyn_cast<ReduceOp>(user)) {
        if (!pure(reduction.getCombine().front())) return {};
      } else if (!isa<memref::LoadOp, memref::StoreOp, memref::CopyOp>(user)) return {};
      if (user == consumer) continue;
      Operation *stage = allocation->getBlock()->findAncestorOpInBlock(*user);
      if (oldValues ? !stage->isBeforeInBlock(consumer) : !consumer->isBeforeInBlock(stage)) return {};
    }
    return lifetime->end;
  };
  auto sourceEnd = lifetime(source, true), outputEnd = lifetime(output, false);
  if (!sourceEnd || !outputEnd) return false;
  auto yielded = dyn_cast<BlockArgument>(body.getTerminator()->getOperand(0));
  bool identity = body.without_terminator().empty() && yielded && yielded.getOwner() == &body &&
      yielded.getArgNumber() < consumer.getInputs().size() &&
      consumer.getInputs()[yielded.getArgNumber()] == buffer;
  // The pointwise phase reads each old element before overwriting that element.
  // All other old-value observations have finished; later phases use the result.
  if (sourceEnd->isBeforeInBlock(outputEnd)) sourceEnd->moveAfter(outputEnd);
  outputEnd.erase();
  output.getResult().replaceAllUsesWith(buffer);
  output.erase();
  if (identity) consumer.erase();
  return true;
}

} // namespace

LogicalResult reusePrivateStorage(func::FuncOp function) {
  // Preserve separate values until region predicates and pointwise bodies have
  // folded, so a constant replacement can discard its old computation first.
  eraseDeadPrivateBuffers(function);
  SmallVector<linalg::GenericOp> consumers;
  function.walk([&](linalg::GenericOp operation) { consumers.push_back(operation); });
  for (auto consumer : llvm::reverse(consumers)) {
    SmallVector<Value> inputs(consumer.getInputs());
    for (Value input : inputs) {
      StorageAnalysis analysis(function);
      if (reusePrivateInput(consumer, input, analysis)) break;
    }
  }
  return success();
}

} // namespace intent::cpu
