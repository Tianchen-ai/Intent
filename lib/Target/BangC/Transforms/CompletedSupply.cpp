#include "PassDetail.h"
#include "Intent/Dialect/DSA/IR/MemoryEffects.h"
#include "Intent/Dialect/DSA/Transforms/LocalSupplyRelations.h"
#include "Intent/Dialect/DSA/Transforms/StoragePatterns.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace intent::bangc {
namespace {

bool sameAttributes(Operation *first, Operation *second) {
  NamedAttrList lhs(first->getAttrs()), rhs(second->getAttrs());
  lhs.erase("intent.origin");
  rhs.erase("intent.origin");
  return lhs.getDictionary(first->getContext()) == rhs.getDictionary(second->getContext());
}

bool privateStorage(Value memory, int64_t space) {
  auto allocation = memory.getDefiningOp<memref::AllocaOp>();
  auto type = allocation ? allocation.getType() : MemRefType{};
  return type && type.getRank() == 2 && type.hasStaticShape() &&
         type.getNumElements() > 0 && type.getLayout().isIdentity() &&
         type.getMemorySpaceAsInt() == space;
}

bool readOnlyResult(Value memory, Operation *writer,
                    dsa::StorageAnalysis &storage, DominanceInfo &dominance) {
  auto allocation = memory.getDefiningOp<memref::AllocaOp>();
  if (!allocation || allocation->getBlock() != writer->getBlock() ||
      !allocation->isBeforeInBlock(writer) ||
      storage.uniqueWriter(memory) != writer) return false;
  auto aliases = storage.aliases(memory);
  if (!aliases.complete) return false;
  for (Operation *user : aliases.users) {
    if (user == writer) continue;
    auto effects = storage.effects(user);
    if (!effects.complete || effects.ordered) return false;
    for (const auto &entry : effects.entries) {
      if (storage.disjoint(memory, entry.effect.getValue())) continue;
      if (!isa<MemoryEffects::Read>(entry.effect.getEffect()) ||
          !dominance.properlyDominates(writer, entry.operation)) return false;
    }
  }
  return true;
}

bool exclusiveStorage(Value memory, ArrayRef<Operation *> users) {
  return memory.getDefiningOp<memref::AllocaOp>() &&
      llvm::all_of(memory.getUsers(), [&](Operation *user) {
        return llvm::is_contained(users, user);
      });
}

// Detaching a writer must also remove its operands from current storage use
// lists. Budget queries then see precisely the prospective executable program.
struct RemovedOperation {
  Operation *operation, *next;
  Block *block;
  SmallVector<Value> operands;
};

SmallVector<RemovedOperation, 0> detach(func::FuncOp function,
                                      const llvm::SetVector<Operation *> &removed) {
  SmallVector<RemovedOperation, 0> saved;
  function.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    if (removed.contains(operation))
      saved.push_back({operation, operation->getNextNode(), operation->getBlock(),
                       llvm::to_vector(operation->getOperands())});
  });
  for (auto &entry : saved) {
    entry.operation->setOperands(ValueRange{});
    entry.operation->remove();
  }
  return saved;
}

void restore(ArrayRef<RemovedOperation> saved) {
  for (const auto &entry : saved) entry.operation->setOperands(entry.operands);
  for (const auto &entry : llvm::reverse(saved)) {
    OpBuilder builder(entry.operation->getContext());
    if (entry.next) builder.setInsertionPoint(entry.next);
    else builder.setInsertionPointToEnd(entry.block);
    builder.insert(entry.operation);
  }
}

void eraseDetached(ArrayRef<RemovedOperation> saved) {
  for (const auto &entry : llvm::reverse(saved)) entry.operation->destroy();
}

bool reusePreparedMatrix(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::PrepareMatrixOp> preparations;
  function.walk([&](dsa::PrepareMatrixOp operation) { preparations.push_back(operation); });
  dsa::StorageAnalysis storage(function);
  DominanceInfo dominance(function);
  for (auto [number, first] : llvm::enumerate(preparations)) {
    if (!privateStorage(first.getInput(), dsa::nramSpace) ||
        !privateStorage(first.getOutput(), dsa::matrixSpace) ||
        !readOnlyResult(first.getOutput(), first, storage, dominance)) continue;
    Operation *writer = storage.lastWriterBefore(first.getInput(), first);
    auto completed = writer ? dsa::detail::completeLocalOutput(writer) : nullptr;
    if (!completed || !dsa::detail::sameCompleteView(storage, completed->get(), first.getInput()))
      continue;
    for (dsa::PrepareMatrixOp second : ArrayRef(preparations).drop_front(number + 1)) {
      if (first->getBlock() != second->getBlock() || !first->isBeforeInBlock(second) ||
          !sameAttributes(first, second) ||
          first.getInput().getType() != second.getInput().getType() ||
          first.getOutput().getType() != second.getOutput().getType() ||
          !dsa::detail::sameCompleteView(storage, first.getInput(), second.getInput()) ||
          !privateStorage(second.getOutput(), dsa::matrixSpace) ||
          !sameAttributes(first.getOutput().getDefiningOp(), second.getOutput().getDefiningOp()) ||
          !readOnlyResult(second.getOutput(), second, storage, dominance) ||
          !storage.disjoint(first.getInput(), first.getOutput()) ||
          !storage.disjoint(first.getInput(), second.getOutput()) ||
          !storage.disjoint(first.getOutput(), second.getOutput()) ||
          storage.lastWriterBefore(second.getInput(), second) != writer ||
          !storage.contentsUnchangedBetween(first.getInput(), first, second)) continue;
      llvm::SetVector<Operation *> removed;
      removed.insert(second);
      removed.insert(second.getOutput().getDefiningOp());
      bool closed = true;
      for (Value scratch : {Value(second.getScratch()), Value(second.getReshaped())}) {
        if (!scratch) continue;
        if (!privateStorage(scratch, dsa::nramSpace) ||
            !exclusiveStorage(scratch, {second.getOperation()}) ||
            !storage.disjoint(scratch, first.getInput()) ||
            !storage.disjoint(scratch, first.getOutput())) { closed = false; break; }
        removed.insert(scratch.getDefiningOp());
      }
      if (!closed) continue;
      SmallVector<OpOperand *> uses;
      for (OpOperand &use : second.getOutput().getUses())
        if (use.getOwner() != second) uses.push_back(&use);
      for (OpOperand *use : uses) use->set(first.getOutput());
      auto saved = detach(function, removed);
      if (!storageFitsBudget(function, config, measureStorage(function))) {
        restore(saved);
        for (OpOperand *use : uses) use->set(second.getOutput());
        continue;
      }
      eraseDetached(saved);
      return true;
    }
  }
  return false;
}

struct ConvertedWindow {
  dsa::LoadTileOp load;
  dsa::CastOp conversion;
};

// An existing full conversion can supply a narrower read without moving any
// operation that mutates the requested window. Its consumers still own that
// independent destination, including subsequent in-place numerical updates.
bool supplyConvertedWindows(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::CastOp> conversions;
  function.walk([&](dsa::CastOp operation) { conversions.push_back(operation); });
  for (dsa::CastOp full : conversions) {
    dsa::StorageAnalysis storage(function);
    dsa::LocalSupplyRelations relations(function);
    DominanceInfo dominance(function);
    if (!privateStorage(full.getInput(), dsa::nramSpace) ||
        !privateStorage(full.getOutput(), dsa::nramSpace) ||
        !readOnlyResult(full.getOutput(), full, storage, dominance)) continue;
    auto load = dyn_cast_or_null<dsa::LoadTileOp>(
        storage.lastWriterBefore(full.getInput(), full));
    if (!load || load.getAsynchronous() || load.getOutput() != full.getInput() ||
        load->getBlock() != full->getBlock() || !load->isBeforeInBlock(full) ||
        !exclusiveStorage(full.getInput(), {load.getOperation(), full.getOperation()})) continue;
    auto rawType = cast<MemRefType>(full.getInput().getType());
    auto fullType = cast<MemRefType>(full.getOutput().getType());
    if (rawType.getShape() != fullType.getShape() ||
        !relations.equal(load.getRows(), fullType.getDimSize(0)) ||
        !relations.equal(load.getColumns(), fullType.getDimSize(1)) ||
        !relations.equal(load.getColumnStride(), 1)) continue;
    Value source = load.getSource();
    auto argument = dyn_cast<BlockArgument>(source);
    // This motion reads an immutable invocation input. A mutable snapshot is
    // not entitled to cross waits merely because two allocation names match.
    if (!argument || argument.getOwner() != &function.front() ||
        !storage.preservesContents(function, source)) continue;

    for (Operation &operation : *full->getBlock()) {
      if (&operation == load.getOperation()) break;
      auto loop = dyn_cast<scf::ForOp>(operation);
      if (!loop || !relations.equal(loop.getLowerBound(), 0)) continue;
      auto members = relations.interval(loop.getInductionVar());
      if (!members || members->first < 0 ||
          members->second >= fullType.getDimSize(1)) continue;
      bool ordered = false;
      for (Operation *current = loop; current != load; current = current->getNextNode()) {
        if (!storage.effects(current).complete) { ordered = true; break; }
        current->walk([&](Operation *nested) {
          if (!nested->getNumRegions() && storage.effects(nested).ordered &&
              !isa<dsa::SynchronizeOp>(nested))
            ordered = true;
        });
      }
      if (ordered) continue;
      SmallVector<ConvertedWindow> windows;
      for (dsa::CastOp conversion : loop.getBody()->getOps<dsa::CastOp>()) {
        if (!sameAttributes(conversion, full) ||
            !privateStorage(conversion.getInput(), dsa::nramSpace) ||
            !privateStorage(conversion.getOutput(), dsa::nramSpace)) continue;
        auto read = dyn_cast_or_null<dsa::LoadTileOp>(
            storage.lastWriterBefore(conversion.getInput(), conversion));
        if (!read || read.getAsynchronous() || read->getBlock() != loop.getBody() ||
            read.getOutput() != conversion.getInput() || read.getSource() != source ||
            !exclusiveStorage(conversion.getInput(), {read.getOperation(), conversion.getOperation()}))
          continue;
        auto input = cast<MemRefType>(conversion.getInput().getType());
        auto output = cast<MemRefType>(conversion.getOutput().getType());
        auto columns = relations.interval(read.getColumns());
        if (input.getShape() != output.getShape() || input.getElementType() != rawType.getElementType() ||
            output.getElementType() != fullType.getElementType() ||
            output.getDimSize(0) != fullType.getDimSize(0) ||
            output.getDimSize(1) >= fullType.getDimSize(1) ||
            !relations.equal(read.getRows(), fullType.getDimSize(0)) ||
            !relations.equal(read.getRowStride(), load.getRowStride()) ||
            !relations.equal(read.getColumnStride(), 1) ||
            !columns || columns->first <= 0 || columns->second > output.getDimSize(1) ||
            members->second > fullType.getDimSize(1) - columns->second ||
            relations.difference(read.getOffset(), load.getOffset()) !=
                relations.difference(loop.getInductionVar(), loop.getLowerBound())) continue;
        windows.push_back({read, conversion});
      }
      if (windows.empty()) continue;

      llvm::SetVector<Operation *> moving;
      std::function<bool(Value)> available = [&](Value value) {
        if (dominance.properlyDominates(value, loop)) return true;
        Operation *definition = value.getDefiningOp();
        if (!definition || definition->getBlock() != loop->getBlock() ||
            !loop->isBeforeInBlock(definition) || !definition->isBeforeInBlock(full)) return false;
        if (moving.contains(definition)) return true;
        bool allocation = definition == full.getInput().getDefiningOp() ||
                          definition == full.getOutput().getDefiningOp();
        if ((!allocation && (definition->getNumRegions() || !isMemoryEffectFree(definition) ||
                             !isSpeculatable(definition))) ||
            !llvm::all_of(definition->getOperands(), available)) return false;
        moving.insert(definition);
        return true;
      };
      if (!llvm::all_of(load->getOperands(), available) || !available(full.getOutput())) continue;
      moving.insert(load);
      moving.insert(full);
      SmallVector<std::pair<Operation *, Operation *>> positions;
      for (Operation &current : *loop->getBlock())
        if (moving.contains(&current)) positions.emplace_back(&current, current.getNextNode());
      for (auto [current, next] : positions) current->moveBefore(loop);

      llvm::SetVector<Operation *> removed;
      SmallVector<Operation *> inserted;
      for (auto window : windows) {
        OpBuilder builder(window.conversion);
        Location location = window.conversion.getLoc();
        auto stride = builder.create<arith::ConstantIndexOp>(location, fullType.getDimSize(1));
        auto one = builder.create<arith::ConstantIndexOp>(location, 1);
        auto supplied = builder.create<dsa::LoadTileOp>(location, full.getOutput(),
            window.conversion.getOutput(), loop.getInductionVar(), stride, one,
            window.load.getRows(), window.load.getColumns(), builder.getBoolAttr(false));
        if (auto origin = window.conversion->getAttr("intent.origin"))
          supplied->setAttr("intent.origin", origin);
        llvm::append_range(inserted, ArrayRef<Operation *>{stride, one, supplied});
        removed.insert(window.load);
        removed.insert(window.conversion);
        removed.insert(window.load.getOutput().getDefiningOp());
      }
      auto saved = detach(function, removed);
      if (!storageFitsBudget(function, config, measureStorage(function))) {
        restore(saved);
        for (Operation *current : llvm::reverse(inserted)) current->erase();
        for (auto [current, next] : llvm::reverse(positions)) current->moveBefore(next);
        continue;
      }
      eraseDetached(saved);
      return true;
    }
  }
  return false;
}

} // namespace

bool reuseCompletedSupply(func::FuncOp function, dsa::ConfigurationAttr config) {
  return reusePreparedMatrix(function, config) || supplyConvertedWindows(function, config);
}

} // namespace intent::bangc
