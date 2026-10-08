#include "PassDetail.h"
#include "Intent/Dialect/DSA/Transforms/LocalSupplyRelations.h"

using namespace mlir;

namespace intent::bangc {
namespace {

enum class BroadcastAxis { Rows, Columns };

struct BroadcastOperand {
  Value source;
  BroadcastAxis axis;
};

std::optional<BroadcastOperand> queryBroadcastOperand(dsa::BinaryOp binary,
    dsa::StorageAnalysis &storage, dsa::LocalSupplyRelations &extents) {
  auto type = cast<MemRefType>(binary.getOutput().getType());
  if (!type.getLayout().isIdentity() || type.getDimSize(0) <= 1 ||
      binary.getRhs().getType() != type || binary.getScratch() ||
      binary.getApproximate() || binary.getFlushToZero() ||
      !supportedScalarBinary(binary.getKind()) ||
      (!type.getElementType().isF32() && !type.getElementType().isF16()))
    return std::nullopt;

  Value previous = binary.getRhs();
  Value origin = storage.uniqueOrigin(previous);
  if (!origin || !origin.getDefiningOp<memref::AllocaOp>() ||
      !storage.aliases(origin).complete ||
      !dsa::isCompleteStorageViewOf(previous, origin))
    return std::nullopt;
  auto broadcast = dyn_cast_or_null<dsa::LoadTileOp>(
      storage.lastWriterBefore(previous, binary));
  if (!broadcast || broadcast.getAsynchronous() ||
      broadcast->getBlock() != binary->getBlock() ||
      !broadcast->isBeforeInBlock(binary) ||
      broadcast.getOutput().getType() != type ||
      !dsa::isCompleteStorageViewOf(broadcast.getOutput(), origin) ||
      !extents.equal(broadcast.getRows(), type.getDimSize(0)) ||
      !extents.equal(broadcast.getColumns(), type.getDimSize(1)) ||
      !extents.equal(broadcast.getOffset(), 0))
    return std::nullopt;

  Value source = broadcast.getSource();
  auto sourceType = cast<MemRefType>(source.getType());
  Value sourceOrigin = storage.uniqueOrigin(source);
  if (!sourceOrigin || !sourceOrigin.getDefiningOp<memref::AllocaOp>() ||
      !storage.aliases(sourceOrigin).complete ||
      !dsa::isCompleteStorageViewOf(source, sourceOrigin) ||
      sourceType.getRank() != 2 || !sourceType.getLayout().isIdentity() ||
      sourceType.getElementType() != type.getElementType() ||
      !storage.disjoint(binary.getOutput(), source) ||
      !storage.readStable(source, broadcast, binary))
    return std::nullopt;

  if (sourceType.getShape() == ArrayRef<int64_t>({1, type.getDimSize(1)}) &&
      extents.equal(broadcast.getRowStride(), 0) &&
      extents.equal(broadcast.getColumnStride(), 1))
    return BroadcastOperand{source, BroadcastAxis::Columns};

  // Per-row scalar instructions have a launch per row. Keep their existing
  // profitability domain separate from the general typed broadcast contract.
  int64_t rows = type.getDimSize(0), columns = type.getDimSize(1);
  if (rows < 16 && columns >= 1024 &&
      sourceType.getShape() == ArrayRef<int64_t>({1, rows}) &&
      extents.equal(broadcast.getRowStride(), 1) &&
      extents.equal(broadcast.getColumnStride(), 0))
    return BroadcastOperand{source, BroadcastAxis::Rows};
  return std::nullopt;
}

} // namespace

bool bindBroadcastOperands(func::FuncOp function) {
  SmallVector<dsa::BinaryOp> binaries;
  function.walk([&](dsa::BinaryOp binary) { binaries.push_back(binary); });
  bool changed = false;
  for (auto binary : binaries) {
    dsa::StorageAnalysis storage(function);
    dsa::LocalSupplyRelations extents(function);
    auto broadcast = queryBroadcastOperand(binary, storage, extents);
    if (!broadcast) continue;
    Value source = broadcast->source;
    if (broadcast->axis == BroadcastAxis::Rows) {
      auto type = cast<MemRefType>(source.getType());
      int64_t rows = type.getDimSize(1);
      auto column = MemRefType::get({rows, 1}, type.getElementType(),
          MemRefLayoutAttrInterface{}, type.getMemorySpace());
      OpBuilder builder(binary);
      source = builder.create<memref::ReinterpretCastOp>(binary.getLoc(), column,
          source, int64_t(0), ArrayRef<int64_t>{rows, 1}, ArrayRef<int64_t>{1, 1});
    }
    binary.getRhsMutable().assign(source);
    binary->setAttr("bangc.implementation", StringAttr::get(function.getContext(),
        broadcast->axis == BroadcastAxis::Rows ? "row_scalar" : "cycle"));
    // Other consumers may still observe this broadcast version. The common
    // local-memory cleanup removes it only after its last actual read is gone.
    changed = true;
  }
  return changed;
}

} // namespace intent::bangc
