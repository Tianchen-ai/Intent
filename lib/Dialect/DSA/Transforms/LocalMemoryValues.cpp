#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "Intent/Dialect/DSA/Analysis/Storage.h"
#include "Intent/Dialect/DSA/IR/MemoryEffects.h"
#include "Intent/Dialect/DSA/IR/Views.h"
#include "Intent/Dialect/DSA/Transforms/StoragePatterns.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::dsa {
namespace {

struct ScalarAccess {
  Value memory, value;
  SmallVector<Value> indices;
  bool write, linear;
};

std::optional<ScalarAccess> scalarAccess(Operation *operation) {
  if (auto load = dyn_cast<memref::LoadOp>(operation))
    return ScalarAccess{load.getMemref(), load.getResult(),
                        llvm::to_vector(load.getIndices()), false, false};
  if (auto store = dyn_cast<memref::StoreOp>(operation))
    return ScalarAccess{store.getMemref(), store.getValue(),
                        llvm::to_vector(store.getIndices()), true, false};
  if (auto load = dyn_cast<LoadScalarOp>(operation))
    return ScalarAccess{load.getSource(), load.getResult(), {load.getOffset()}, false, true};
  if (auto store = dyn_cast<StoreScalarOp>(operation))
    return ScalarAccess{store.getDestination(), store.getValue(), {store.getOffset()}, true, true};
  return std::nullopt;
}

bool sameIndex(Value lhs, Value rhs) {
  if (lhs == rhs) return true;
  auto a = getConstantIntValue(lhs), b = getConstantIntValue(rhs);
  return a && b && *a == *b;
}

// This describes existing dense descriptor coordinates, not a new index IR.
// An unknown/dynamic layout stays at exact descriptor-and-index matching.
struct ScalarAddress {
  Value origin;
  int64_t constant = 0;
  SmallVector<std::pair<Value, int64_t>> terms;
};

std::optional<ScalarAddress> address(const ScalarAccess &access,
                                     StorageAnalysis &storage) {
  auto type = cast<MemRefType>(access.memory.getType());
  Value origin = storage.uniqueOrigin(access.memory);
  if (!origin || !type.hasStaticShape() || !type.getLayout().isIdentity() ||
      !isCompleteStorageViewOf(access.memory, origin)) return std::nullopt;
  ScalarAddress result{origin, 0, {}};
  auto append = [&](Value coordinate, int64_t stride) {
    if (auto constant = getConstantIntValue(coordinate)) {
      int64_t offset;
      return !llvm::MulOverflow(*constant, stride, offset) &&
             !llvm::AddOverflow(result.constant, offset, result.constant);
    }
    for (auto &term : result.terms)
      if (term.first == coordinate)
        return !llvm::AddOverflow(term.second, stride, term.second);
    result.terms.emplace_back(coordinate, stride);
    return true;
  };
  if (access.linear) {
    if (!append(access.indices.front(), 1)) return std::nullopt;
  } else {
    int64_t stride = 1;
    for (int64_t axis = type.getRank(); axis-- > 0;) {
      if (!append(access.indices[axis], stride)) return std::nullopt;
      if (axis && llvm::MulOverflow(stride, type.getDimSize(axis), stride))
        return std::nullopt;
    }
  }
  return result;
}

bool sameAddress(const ScalarAccess &lhs, const ScalarAccess &rhs,
                 StorageAnalysis &storage) {
  if (lhs.value.getType() != rhs.value.getType()) return false;
  if (lhs.memory == rhs.memory && lhs.linear == rhs.linear &&
      lhs.indices.size() == rhs.indices.size() &&
      llvm::all_of(llvm::zip(lhs.indices, rhs.indices), [](auto values) {
        return sameIndex(std::get<0>(values), std::get<1>(values));
      })) return true;
  auto a = address(lhs, storage), b = address(rhs, storage);
  return a && b && a->origin == b->origin && a->constant == b->constant &&
      a->terms.size() == b->terms.size() &&
      llvm::all_of(a->terms, [&](auto term) { return llvm::is_contained(b->terms, term); });
}

bool completeView(Value memory, Value origin) {
  return memory == origin || isCompleteStorageViewOf(memory, origin);
}

bool sameCompleteView(Value lhs, Value rhs, StorageAnalysis &storage) {
  Value origin = storage.uniqueOrigin(lhs);
  return origin && origin == storage.uniqueOrigin(rhs) &&
      completeView(lhs, origin) && completeView(rhs, origin);
}

bool coversMemory(Value complete, Value memory, StorageAnalysis &storage) {
  Value origin = storage.uniqueOrigin(complete);
  return origin && origin == storage.uniqueOrigin(memory) &&
      completeView(complete, origin);
}

bool sameTileRead(LoadTileOp lhs, LoadTileOp rhs) {
  if (lhs.getSource() != rhs.getSource() ||
      lhs.getOutput().getType() != rhs.getOutput().getType() ||
      lhs.getAsynchronous() || rhs.getAsynchronous() ||
      lhs->getAttrDictionary() != rhs->getAttrDictionary()) return false;
  return sameIndex(lhs.getOffset(), rhs.getOffset()) &&
      sameIndex(lhs.getRowStride(), rhs.getRowStride()) &&
      sameIndex(lhs.getColumnStride(), rhs.getColumnStride()) &&
      sameIndex(lhs.getRows(), rhs.getRows()) && sameIndex(lhs.getColumns(), rhs.getColumns());
}

struct AvailableValue {
  Value memory, value;
  std::optional<ScalarAccess> scalar;
};
struct PendingWrite {
  Operation *operation;
  Value memory;
  std::optional<ScalarAccess> scalar;
};

class LocalMemoryValues {
public:
  explicit LocalMemoryValues(func::FuncOp function)
      : storage(function), dominance(function) {}

  void analyze(Block &block, SmallVector<AvailableValue> available = {}) {
    SmallVector<PendingWrite> pending;
    SmallVector<LoadTileOp> tiles;
    auto read = [&](Value memory) {
      llvm::erase_if(pending, [&](const PendingWrite &write) {
        return !memory || !storage.disjoint(memory, write.memory);
      });
    };
    for (Operation &operation : block) {
      if (operation.hasTrait<OpTrait::IsTerminator>()) break;
      auto effects = storage.effects(&operation);
      if (!effects.complete || effects.ordered) {
        available.clear(); tiles.clear();
        llvm::erase_if(pending, [&](const PendingWrite &write) {
          return !preservesPendingWrite(write, &operation, effects);
        });
        for (Region &region : operation.getRegions())
          for (Block &nested : region) analyze(nested);
        continue;
      }
      auto scalar = scalarAccess(&operation);
      if (scalar && !scalar->write) {
        Value value;
        for (const AvailableValue &known : llvm::reverse(available)) {
          bool matches = known.scalar ? sameAddress(*known.scalar, *scalar, storage)
              : coversMemory(known.memory, scalar->memory, storage);
          if (matches && known.value.getType() == scalar->value.getType() &&
              dominance.properlyDominates(known.value, &operation)) {
            value = resolve(known.value);
            break;
          }
        }
        if (value) {
          replacements.map(scalar->value, value);
          erased.insert(&operation);
          continue;
        }
      }
      Value retained;
      auto tile = dyn_cast<LoadTileOp>(&operation);
      if (tile && !tile.getAsynchronous()) {
        for (LoadTileOp previous : llvm::reverse(tiles)) {
          if (!sameTileRead(previous, tile)) continue;
          if (previous.getOutput() != tile.getOutput() &&
              !storage.disjoint(previous.getOutput(), tile.getOutput())) continue;
          retained = previous.getOutput();
          break;
        }
        if (retained) {
          // The replacement copy reads this already completed local snapshot.
          read(retained);
          tileCopies.emplace_back(tile, retained);
          if (retained == tile.getOutput()) continue;
        }
      }
      for (const auto &entry : effects.entries)
        if (isa<MemoryEffects::Read>(entry.effect.getEffect()) && !retained)
          read(entry.effect.getValue());

      // Child facts cannot escape their branch. A parent snapshot enters a
      // repeated/conditional region only if the whole operation preserves it.
      for (Region &region : operation.getRegions()) {
        SmallVector<AvailableValue> inherited;
        if (!operation.hasTrait<OpTrait::IsIsolatedFromAbove>())
          for (const AvailableValue &known : available)
            if (storage.preserves(&operation, known.memory)) inherited.push_back(known);
        for (Block &nested : region) analyze(nested, inherited);
      }

      auto output = detail::completeLocalOutput(&operation);
      Value complete = output ? output->get() : Value{};
      if (scalar && scalar->write) {
        auto type = cast<MemRefType>(scalar->memory.getType());
        if (type.hasStaticShape() && type.getNumElements() == 1)
          complete = scalar->memory;
      }
      for (const auto &entry : effects.entries) {
        const auto &effect = entry.effect;
        if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect())) continue;
        Value memory = effect.getValue();
        for (const PendingWrite &write : pending) {
          bool covered = complete && sameCompleteView(complete, write.memory, storage);
          covered |= scalar && scalar->write && write.scalar &&
              sameAddress(*scalar, *write.scalar, storage);
          if (isa<MemoryEffects::Write>(effect.getEffect()) && memory &&
              !storage.disjoint(memory, write.memory) && covered)
            erased.insert(write.operation);
        }
        read(memory);
        llvm::erase_if(available, [&](const AvailableValue &known) {
          return !memory || !storage.disjoint(memory, known.memory);
        });
        llvm::erase_if(tiles, [&](LoadTileOp known) {
          return !memory || !storage.disjoint(memory, known.getSource()) ||
                 !storage.disjoint(memory, known.getOutput());
        });
      }
      if (scalar)
        available.push_back({scalar->memory, resolve(scalar->value), *scalar});
      else if (auto fill = dyn_cast<FillOp>(&operation))
        available.push_back({fill.getOutput(), resolve(fill.getValue()), std::nullopt});
      if (tile && !retained && !tile.getAsynchronous() &&
          storage.disjoint(tile.getSource(), tile.getOutput())) tiles.push_back(tile);

      Value written = scalar && scalar->write ? scalar->memory : complete;
      if (!written || operation.getNumRegions() || operation.getNumResults()) continue;
      auto type = cast<MemRefType>(written.getType());
      if (type.getMemorySpaceAsInt() != nramSpace) continue;
      bool onlyThisWrite = llvm::all_of(effects.entries, [&](const auto &entry) {
        return isa<MemoryEffects::Read>(entry.effect.getEffect()) ||
            (isa<MemoryEffects::Write>(entry.effect.getEffect()) && entry.effect.getValue() == written);
      });
      if (onlyThisWrite)
        pending.push_back({&operation, written, scalar && scalar->write ? scalar : std::nullopt});
    }
  }

  bool apply() {
    // All queries precede mutation. Re-run after changing uses or storage.
    for (auto [before, after] : replacements.getValueMap())
      before.replaceAllUsesWith(resolve(after));
    for (auto [tile, retained] : tileCopies) {
      if (erased.contains(tile)) continue;
      if (retained != tile.getOutput()) {
        OpBuilder builder(tile);
        builder.create<memref::CopyOp>(tile.getLoc(), retained, tile.getOutput());
      }
      erased.insert(tile);
    }
    for (Operation *operation : erased) operation->erase();
    return !erased.empty();
  }

private:
  bool preservesPendingWrite(const PendingWrite &write, Operation *scope,
                             const BufferStorageEffects &effects) {
    Value origin = storage.uniqueOrigin(write.memory);
    auto allocation = origin ? origin.getDefiningOp<memref::AllocaOp>()
                             : memref::AllocaOp{};
    auto output = detail::completeLocalOutput(write.operation);
    if (!effects.complete || !allocation ||
        allocation.getType().getMemorySpaceAsInt() != nramSpace || !output ||
        !isCompleteStorageViewOf(output->get(), origin) ||
        !isCompleteStorageViewOf(write.memory, origin)) return false;
    if (llvm::any_of(effects.entries, [&](const auto &entry) {
          return !storage.disjoint(origin, entry.effect.getValue());
        })) return false;
    // Group participation is not a private-work-unit ordering boundary.
    bool group = false;
    scope->walk([&](Operation *operation) {
      group |= requiredCompletion(operation) == CompletionScope::ExecutionGroup ||
               completesTransfers(operation, CompletionScope::ExecutionGroup);
    });
    if (group) return false;
    auto accesses = storage.accesses(origin);
    if (!accesses.complete) return false;
    // A disjoint current prefetch says nothing about older outstanding uses.
    // All asynchronous uses of this private storage must have completed before
    // the pending write; an ambiguous use/backedge keeps the original write.
    for (const auto &entry : accesses.entries) {
      if (requiredCompletion(entry.operation) == CompletionScope::Immediate)
        continue;
      auto completion = storage.completionOfUse(entry.operation);
      if (failed(completion) ||
          !dominance.properlyDominates(*completion, write.operation)) return false;
    }
    return true;
  }

  Value resolve(Value value) const {
    while (Value next = replacements.lookupOrNull(value)) value = next;
    return value;
  }
  StorageAnalysis storage;
  DominanceInfo dominance;
  IRMapping replacements;
  llvm::SetVector<Operation *> erased;
  SmallVector<std::pair<LoadTileOp, Value>> tileCopies;
};

} // namespace

bool reuseLocalMemoryValues(func::FuncOp function) {
  LocalMemoryValues values(function);
  for (Block &block : function.getBody()) values.analyze(block);
  return values.apply();
}

} // namespace intent::dsa
