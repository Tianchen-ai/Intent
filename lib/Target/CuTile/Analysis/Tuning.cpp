#include "Intent/Target/CuTile/Analysis/Tuning.h"

#include "Intent/Dialect/Block/Analysis/ValueSchema.h"
#include "Intent/Dialect/Block/Analysis/ProgramInterface.h"
#include "Intent/Dialect/Block/IR/BlockOps.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::cutile {
namespace {

bool isRuntimeScalar(BlockArgument argument) {
  auto kernel = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!kernel)
    return false;
  auto binding = block::getArgumentBinding(argument);
  return binding && binding.getKind() == block::ArgumentKind::Public &&
         !isa<block::ViewType>(argument.getType());
}

// Payload stores may feed later loads, including through another external
// view's alias. Do not stop the proof at memory: conservatively include every
// read result in the kernel, regardless of address or relative ordering.
struct MemoryReads {
  SmallVector<Value> values;
  bool unknown = false;

  explicit MemoryReads(func::FuncOp kernel) {
    kernel.walk([&](Operation *operation) {
      if (isa<TileLoadOp, ScalarLoadOp, GatherLoadOp, AtomicRMWOp,
              block::LoadOp, block::GatherOp, block::AtomicLoadOp,
              block::AtomicRMWOp, block::AtomicCompareExchangeOp>(operation)) {
        llvm::append_range(values, operation->getResults());
        return;
      }
      // Structured owners are covered by the recursive walk. Unknown
      // leaf readers, calls and effects must not turn into an alias proof.
      if (isa<func::FuncOp, scf::IfOp, scf::ForOp, scf::WhileOp>(operation))
        return;
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(operation)) {
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        unknown |= llvm::any_of(instances, [](const auto &effect) {
          return isa<MemoryEffects::Read>(effect.getEffect());
        });
      } else if (!isMemoryEffectFree(operation)) {
        unknown = true;
      }
    });
  }
};

bool isStoredPayload(OpOperand &use) {
  Operation *operation = use.getOwner();
  if (auto store = dyn_cast<TileStoreOp>(operation))
    return &use == &store.getValueMutable();
  if (auto store = dyn_cast<ScalarStoreOp>(operation))
    return &use == &store.getValueMutable();
  if (auto store = dyn_cast<ScatterStoreOp>(operation))
    return &use == &store.getValueMutable();
  if (auto store = dyn_cast<TileAtomicAddOp>(operation))
    return &use == &store.getValueMutable();
  if (auto store = dyn_cast<AtomicRMWOp>(operation))
    return &use == &store.getValueMutable();
  if (auto store = dyn_cast<block::StoreOp>(operation))
    return &use == &store.getValueMutable();
  if (auto store = dyn_cast<block::ScatterReduceOp>(operation))
    return &use == &store.getValueMutable();
  return false;
}

bool isDataOperation(Operation *operation) {
  // A whitelist is intentional: Pure also includes range/index construction,
  // so it is not sufficient to establish a data-only dependency.
  return isa<block::SplatOp, block::BroadcastOp, block::UnaryOp, block::BinaryOp,
             block::CompareOp, block::SelectOp, block::CastOp, block::BitcastOp,
             block::ReshapeOp, block::TransposeOp, MMAOp, ScaledMMAOp>(operation);
}

bool hasOnlyDataUses(Value argument, const MemoryReads &reads) {
  SmallVector<Value> pending{argument};
  llvm::DenseSet<Value> visited;
  bool followedMemory = false;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *operation = use.getOwner();
      if (isStoredPayload(use)) {
        if (reads.unknown)
          return false;
        if (!followedMemory) {
          llvm::append_range(pending, reads.values);
          followedMemory = true;
        }
        // An atomic payload also flows to the returned old value through a
        // possible earlier aliasing write, covered by the read closure above.
        continue;
      }
      if (isDataOperation(operation)) {
        llvm::append_range(pending, operation->getResults());
        continue;
      }
      if (auto extract = dyn_cast<ExtractOp>(operation)) {
        if (&use != &extract.getSourceMutable())
          return false;
        pending.push_back(extract.getResult());
        continue;
      }
      if (auto yield = dyn_cast<scf::YieldOp>(operation)) {
        // A branch result preserves data dependence; the if condition is a
        // separate use and never accepted here. Loop-carried values remain
        // unknown, including any effects reached through later iterations.
        if (auto branch = dyn_cast<scf::IfOp>(yield->getParentOp())) {
          pending.push_back(branch.getResult(use.getOperandNumber()));
          continue;
        }
      }
      return false;
    }
  }
  return true;
}

} // namespace

llvm::SmallBitVector getTuningKeyScalarArguments(func::FuncOp kernel) {
  llvm::SmallBitVector retained(kernel.getNumArguments());
  // Scalar ABI dependencies can survive only in parameter, type or launch
  // attributes. Walking SSA alone would miss them after canonicalization.
  AttrTypeWalker expressions;
  expressions.addWalk([&](block::PhysicalExprAttr expression) {
    if (block::isShapeBound(expression) ||
        expression.getKind() !=
            block::PhysicalExprKind::ScalarABI)
      return;
    auto argument = block::resolveArgument(kernel, expression.getArgumentReference());
    if (argument && isRuntimeScalar(argument))
      retained.set(argument.getArgNumber());
  });
  kernel.walk([&](Operation *operation) {
    expressions.walk(operation->getAttrDictionary());
    for (Type type : operation->getResultTypes())
      expressions.walk(type);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          expressions.walk(argument.getType());
  });

  MemoryReads reads(kernel);
  for (BlockArgument argument : kernel.getArguments())
    if (isRuntimeScalar(argument) && !retained.test(argument.getArgNumber()) &&
        !hasOnlyDataUses(argument, reads))
      retained.set(argument.getArgNumber());
  return retained;
}

} // namespace intent::cutile
