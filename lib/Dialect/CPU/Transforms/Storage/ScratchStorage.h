#ifndef INTENT_CPU_TRANSFORMS_STORAGE_SCRATCHSTORAGE_H
#define INTENT_CPU_TRANSFORMS_STORAGE_SCRATCHSTORAGE_H

#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"

namespace intent::cpu::detail {

// Ephemeral facts about one current allocation and its closed local uses.
struct ScratchAllocation {
  mlir::Operation *operation;
  mlir::Value memory;
  mlir::MemRefType type;
  mlir::Operation *lastUse;
  mlir::memref::DeallocOp release;
  StorageAliasFacts aliases;
  bool stack;
};

struct ScratchCandidate {
  mlir::Operation *operation;
  std::optional<ScratchAllocation> allocation;
  bool queried = false;
};

// One read-only search across slot reuse and placement. Candidates retain their
// original order, and the driver discards all facts after the first rewrite.
class ScratchSnapshot {
public:
  explicit ScratchSnapshot(mlir::func::FuncOp function);
  llvm::ArrayRef<ScratchCandidate> candidates() const { return allocations; }
  ScratchAllocation *get(unsigned number);
  mlir::DominanceInfo &getDominance();
  std::optional<bool> hasLoopOnlyDataUses(const ScratchAllocation &scratch,
                                        mlir::scf::ForOp loop);

private:
  mlir::func::FuncOp function;
  llvm::SmallVector<ScratchCandidate> allocations;
  std::optional<StorageAnalysis> storage;
  std::optional<mlir::DominanceInfo> dominance;
};

std::optional<ScratchAllocation> scratchAllocation(
    mlir::Operation *operation, StorageAnalysis &storage);
std::optional<int64_t> scratchCapacity(const ScratchAllocation &scratch,
                                      int64_t byteLimit);
std::optional<int64_t> scratchBytes(const ScratchAllocation &scratch,
                                  int64_t capacity);
int64_t scratchAlignment(const ScratchAllocation &scratch);
mlir::Value createScratchBacking(mlir::OpBuilder &builder,
    const ScratchAllocation &scratch, int64_t capacity, int64_t alignment);
// Keep the original logical shape and dense strides even when backing capacity
// is larger or has a different rank. No initialization or read is moved.
mlir::Value scratchDescriptor(mlir::OpBuilder &builder,
    const ScratchAllocation &scratch, mlir::Value backing);

bool reuseScratchSlots(ScratchSnapshot &snapshot, int64_t byteLimit,
                       ScratchRepresentation representation);
bool placeScratchAllocations(ScratchSnapshot &snapshot, int64_t byteLimit,
                             ScratchRepresentation representation);

} // namespace intent::cpu::detail
#endif
