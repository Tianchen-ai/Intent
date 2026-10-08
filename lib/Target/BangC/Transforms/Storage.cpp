#include "PassDetail.h"

using namespace mlir;
namespace intent::bangc {
namespace {
StorageUsage visitStorage(func::FuncOp function,
    llvm::function_ref<void(memref::AllocaOp, int64_t, int64_t)> binding) {
  struct Live { uint64_t finish; int64_t offset, bytes, space; };
  SmallVector<Live> live;
  StorageUsage usage;
  for (const auto &range : dsa::analyzeStorageLifetimes(function)) {
    auto allocation = range.allocation;
    auto type = allocation.getType();
    int64_t space = type.getMemorySpaceAsInt();
    bool matrix = space == dsa::matrixSpace;
    int64_t payload = type.getNumElements() * llvm::divideCeil(type.getElementTypeBitWidth(), 8u);
    // MTP372 WRAM addresses are per-bank offsets across sixteen LT banks.
    int64_t bytes = matrix ? llvm::alignTo(llvm::divideCeil(payload, int64_t(16)), int64_t(64))
                           : llvm::alignTo(payload, int64_t(128));
    llvm::erase_if(live, [&](const Live &value) { return value.finish < range.start; });
    llvm::sort(live, [](const Live &a, const Live &b) { return a.offset < b.offset; });
    int64_t offset = 0;
    for (const Live &value : live) {
      if (value.space != space) continue;
      if (offset + bytes <= value.offset) break;
      offset = std::max(offset, value.offset + value.bytes);
    }
    live.push_back({range.finish, offset, bytes, space});
    int64_t &peak = matrix ? usage.wram : space == dsa::sharedSpace ? usage.sram : usage.nram;
    peak = std::max(peak, (offset + bytes) * (matrix ? 16 : 1));
    binding(allocation, offset, bytes * (matrix ? 16 : 1));
  }
  return usage;
}
} // namespace

StorageUsage measureStorage(func::FuncOp function) {
  return visitStorage(function, [](memref::AllocaOp, int64_t, int64_t) {});
}

void measureStorage(func::FuncOp function, int64_t &nram, int64_t &wram) {
  auto usage = measureStorage(function);
  nram = usage.nram; wram = usage.wram;
}

StorageUsage bindStorage(func::FuncOp function) {
  Builder builder(function.getContext());
  auto usage = visitStorage(function, [&](memref::AllocaOp allocation, int64_t offset, int64_t bytes) {
    allocation->setAttr("bangc.offset", builder.getI64IntegerAttr(offset));
    allocation->setAttr("bangc.allocation_bytes", builder.getI64IntegerAttr(bytes));
  });
  function->setAttr("bangc.sram_bytes", builder.getI64IntegerAttr(usage.sram));
  return usage;
}

Value allocate(OpBuilder &b, Location loc, Type element, ArrayRef<int64_t> shape, int64_t space) {
  auto type = MemRefType::get(shape, element, MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(space));
  auto result = b.create<memref::AllocaOp>(loc, type);
  result.setAlignment(128);
  return result;
}

namespace {
struct InPlaceCandidate {
  Operation *operation;
  unsigned input, output;
};

bool reuseConsumedStorage(func::FuncOp function, dsa::ConfigurationAttr config,
                          ArrayRef<InPlaceCandidate> candidates) {
  int64_t currentNram = 0, currentWram = 0;
  measureStorage(function, currentNram, currentWram);
  bool changed = false;
  for (auto candidate : candidates) {
    Operation *operation = candidate.operation;
    Value input = operation->getOperand(candidate.input);
    Value output = operation->getOperand(candidate.output);
    dsa::StorageAnalysis storage(function);
    Value inputOrigin = storage.uniqueOrigin(input), outputOrigin = storage.uniqueOrigin(output);
    auto source = inputOrigin ? inputOrigin.getDefiningOp<memref::AllocaOp>() : memref::AllocaOp{};
    auto destination = outputOrigin ? outputOrigin.getDefiningOp<memref::AllocaOp>() : memref::AllocaOp{};
    if (!source || !destination || source == destination ||
        source->getBlock() != operation->getBlock() || destination->getBlock() != operation->getBlock() ||
        source.getType() != destination.getType() || !source.getType().getLayout().isIdentity() ||
        source.getType().getMemorySpaceAsInt() != dsa::nramSpace ||
        source.getAlignment().value_or(0) < destination.getAlignment().value_or(0) ||
        input.getType() != output.getType() ||
        !dsa::isCompleteStorageViewOf(input, source) ||
        !dsa::isCompleteStorageViewOf(output, destination) ||
        !storage.allUsesCompleteBefore(source, operation, operation)) continue;
    if (llvm::any_of(llvm::enumerate(operation->getOperands()), [&](auto entry) {
          Value operand = entry.value();
          return entry.index() != candidate.output && operand != input &&
              isa<MemRefType>(operand.getType()) &&
              (!storage.disjoint(operand, input) || !storage.disjoint(operand, output));
        })) continue;
    // Descriptor forwarding is harmless; every actual use of the old output
    // must follow this complete overwrite, including uses through other views.
    DominanceInfo dominance(function);
    auto outputUses = storage.accesses(destination);
    if (!outputUses.complete || !dominance.dominates(source.getResult(), destination) ||
        llvm::any_of(outputUses.entries, [&](const auto &entry) {
          return entry.operation != operation &&
              !isa<MemoryEffects::Allocate>(entry.effect.getEffect()) &&
              !dominance.properlyDominates(operation, entry.operation);
        })) continue;
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : destination.getResult().getUses()) uses.push_back(&use);
    for (OpOperand *use : uses) use->set(source);
    Operation *next = destination->getNextNode();
    destination->remove();
    int64_t nram = 0, wram = 0;
    measureStorage(function, nram, wram);
    if (nram > currentNram || wram > currentWram || nram > config.getLocalBytes() ||
        nram > 768 * 1024 || wram > 1024 * 1024) {
      OpBuilder restore(next);
      restore.insert(destination.getOperation());
      for (OpOperand *use : uses) use->set(destination);
    } else {
      destination->destroy();
      currentNram = nram;
      currentWram = wram;
      changed = true;
    }
  }
  return changed;
}
} // namespace

bool reuseConsumedBinaryInputs(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<InPlaceCandidate> candidates;
  function.walk([&](dsa::BinaryOp operation) {
    // Exact native input/output aliasing is a provider capability. Storage
    // liveness and view completeness are proved by the common implementation.
    if (operation.getApproximate() || operation.getFlushToZero() || operation.getScratch()) return;
    if (!isa<MemRefType>(operation.getRhs().getType()) && !supportedScalarBinary(operation.getKind())) return;
    switch (operation.getKind()) {
    case BinaryOperator::Add: case BinaryOperator::Subtract: case BinaryOperator::Multiply:
    case BinaryOperator::Maximum: case BinaryOperator::Minimum:
    case BinaryOperator::MaximumNum: case BinaryOperator::MinimumNum: break;
    default: return;
    }
    Type element = cast<MemRefType>(operation.getOutput().getType()).getElementType();
    if (!element.isF16() && !element.isF32()) return;
    if (auto loop = dyn_cast<scf::ForOp>(operation->getParentOp())) {
      dsa::StorageAnalysis storage(function);
      dsa::UniformMemoryAnalysis uniforms(function, storage);
      if (Value scalar = uniforms.read(operation.getLhs(), operation);
          scalar && loop.isDefinedOutsideOfLoop(scalar)) return;
    }
    candidates.push_back({operation, 0, 2});
  });
  return reuseConsumedStorage(function, config, candidates);
}

bool reuseConsumedExp2Inputs(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<InPlaceCandidate> candidates;
  function.walk([&](dsa::UnaryOp op) {
    if (op.getKind() != UnaryOperator::Exp2 || !op.getApproximate() || !op.getFlushToZero() ||
        !cast<MemRefType>(op.getOutput().getType()).getElementType().isF32() ||
        (op.getScratch() && !cast<MemRefType>(op.getScratch().getType()).getElementType().isInteger(32))) return;
    candidates.push_back({op, 0, 1});
  });
  return reuseConsumedStorage(function, config, candidates);
}

bool storageFitsBudget(func::FuncOp function, dsa::ConfigurationAttr config,
                       StorageUsage usage) {
  int64_t internalNram = 0;
  function.walk([&](Operation *op) {
    if (auto bytes = op->getAttrOfType<IntegerAttr>("bangc.internal_nram_bytes"))
      internalNram = std::max(internalNram, bytes.getInt());
  });
  return usage.nram <= std::min<int64_t>(config.getLocalBytes(), 768 * 1024) - internalNram &&
         usage.wram <= 1024 * 1024 && usage.sram <= 3968 * 1024;
}

LogicalResult bindProgramStorage(ModuleOp module) {
  auto function = *module.getOps<func::FuncOp>().begin();
  auto config = function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration");
  StorageUsage usage = bindStorage(function);
  int64_t nram = usage.nram, wram = usage.wram, shared = usage.sram;
  if (shared > 3968 * 1024) return function.emitError("selected shared supply exceeds the MLU370 SRAM budget");
  if (!storageFitsBudget(function, config, usage))
    return function.emitError("selected local implementation exceeds the MLU370 per-task storage budget")
        << "; NRAM=" << nram << ", WRAM=" << wram;
  Builder builder(module.getContext());
  function->setAttr("bangc.nram_bytes", builder.getI64IntegerAttr(nram));
  function->setAttr("bangc.wram_bytes", builder.getI64IntegerAttr(wram));
  function->setAttr("bangc.wram_align", builder.getI64IntegerAttr(16));
  return success();
}
} // namespace intent::bangc
