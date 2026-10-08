#include "ContiguousMemory.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace intent::cpu {

ContiguousMemoryGuard materializeContiguousMemoryGuard(
    OpBuilder &builder, Location location, ValueRange memories) {
  ContiguousMemoryGuard guard;
  llvm::SmallDenseSet<Value> seen;
  for (Value memory : memories) {
    if (!seen.insert(memory).second) continue;
    auto type = cast<MemRefType>(memory.getType());
    auto [strides, offset] = type.getStridesAndOffset();
    if (!ShapedType::isDynamic(strides.back())) continue;
    auto metadata = builder.create<memref::ExtractStridedMetadataOp>(location, memory);
    guard.descriptors.push_back(metadata);
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    Value unit = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq,
                                              metadata.getStrides().back(), one);
    guard.condition = guard.condition
        ? Value(builder.create<arith::AndIOp>(location, guard.condition, unit)) : unit;
  }
  return guard;
}

void ContiguousMemoryGuard::bind(OpBuilder &builder, Location location,
                                 IRMapping &mapping) const {
  for (auto metadata : descriptors) {
    auto sourceType = cast<MemRefType>(metadata.getSource().getType());
    auto [strides, offset] = sourceType.getStridesAndOffset();
    strides.back() = 1;
    SmallVector<OpFoldResult> sizes, steps;
    for (int64_t axis = 0; axis < sourceType.getRank(); ++axis) {
      sizes.push_back(sourceType.isDynamicDim(axis)
          ? OpFoldResult(metadata.getSizes()[axis])
          : OpFoldResult(builder.getIndexAttr(sourceType.getDimSize(axis))));
      steps.push_back(ShapedType::isDynamic(strides[axis])
          ? OpFoldResult(metadata.getStrides()[axis])
          : OpFoldResult(builder.getIndexAttr(strides[axis])));
    }
    auto viewType = MemRefType::get(sourceType.getShape(), sourceType.getElementType(),
        StridedLayoutAttr::get(builder.getContext(), offset, strides), sourceType.getMemorySpace());
    OpFoldResult begin = ShapedType::isDynamic(offset)
        ? OpFoldResult(metadata.getOffset()) : OpFoldResult(builder.getIndexAttr(offset));
    mapping.map(metadata.getSource(), builder.create<memref::ReinterpretCastOp>(
        location, viewType, metadata.getBaseBuffer(), begin, sizes, steps));
  }
}

} // namespace intent::cpu
