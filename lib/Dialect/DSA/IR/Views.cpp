#include "Intent/Dialect/DSA/IR/Views.h"
#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <cstdint>
#include <optional>

using namespace mlir;

namespace intent::dsa {
bool isStorageElementType(Type type) {
  if (type.isF16() || type.isBF16() || type.isF32() || type.isF64() ||
      isa<Float8E4M3FNType, Float8E5M2Type>(type))
    return true;
  auto integer = dyn_cast<IntegerType>(type);
  return integer && llvm::is_contained(ArrayRef<unsigned>{1, 8, 16, 32, 64},
                                      integer.getWidth());
}
namespace {

bool localCapacity(MemRefType type) {
  if (!type || !type.hasStaticShape() || type.getNumElements() <= 0 ||
      !type.getLayout().isIdentity() ||
      type.getMemorySpaceAsInt() != nramSpace)
    return false;
  Type element = type.getElementType();
  return isStorageElementType(element);
}

SmallVector<int64_t> contiguousStrides(MemRefType type) {
  SmallVector<int64_t> strides(type.getRank());
  int64_t stride = 1;
  for (int64_t axis = type.getRank(); axis > 0; --axis) {
    strides[axis - 1] = stride;
    stride *= type.getDimSize(axis - 1);
  }
  return strides;
}

bool completeContiguousView(memref::ReinterpretCastOp view) {
  auto input = cast<MemRefType>(view.getSource().getType());
  auto output = view.getType();
  return input.hasStaticShape() && output.hasStaticShape() &&
         input.getLayout().isIdentity() && output.getLayout().isIdentity() &&
         input.getMemorySpace() == output.getMemorySpace() &&
         input.getElementType() == output.getElementType() &&
         input.getNumElements() == output.getNumElements() &&
         view.getOffsets().empty() && view.getSizes().empty() &&
         view.getStrides().empty() &&
         view.getStaticOffsets() == ArrayRef<int64_t>({0}) &&
         view.getStaticSizes() == output.getShape() &&
         view.getStaticStrides() == ArrayRef<int64_t>(contiguousStrides(output));
}

} // namespace

bool isCompleteLocalStorageView(Value value) {
  if (!localCapacity(dyn_cast<MemRefType>(value.getType()))) return false;
  while (true) {
    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
    } else if (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) {
      if (!completeContiguousView(view))
        return false;
      value = view.getSource();
    } else {
      break;
    }
    if (!localCapacity(dyn_cast<MemRefType>(value.getType())))
      return false;
  }
  if (value.getDefiningOp<memref::AllocaOp>()) return true;
  auto formal = dyn_cast<BlockArgument>(value);
  return formal && isCollectiveBorrowedArgument(formal);
}

bool isBoundedContiguousLocalView(Value value) {
  auto view = value.getDefiningOp<memref::ReinterpretCastOp>();
  if (!view || !isCompleteLocalStorageView(view.getSource()) ||
      !view.getOffsets().empty() || !view.getSizes().empty() ||
      !view.getStrides().empty()) return false;
  auto source = cast<MemRefType>(view.getSource().getType());
  auto result = view.getType();
  if (result.getRank() != 2 || !result.hasStaticShape() ||
      result.getElementType() != source.getElementType() ||
      result.getMemorySpace() != source.getMemorySpace() ||
      view.getStaticOffsets().size() != 1 ||
      view.getStaticSizes() != result.getShape()) return false;
  auto elements = [](MemRefType type) -> std::optional<int64_t> {
    int64_t count = 1;
    for (int64_t extent : type.getShape()) {
      if (extent <= 0 || extent > INT64_MAX / count) return std::nullopt;
      count *= extent;
    }
    return count;
  };
  auto capacity = elements(source), count = elements(result);
  int64_t offset = view.getStaticOffsets().front();
  if (!capacity || !count || offset < 0 || offset > *capacity ||
      *count > *capacity - offset) return false;
  auto dense = contiguousStrides(result);
  SmallVector<int64_t> strides;
  int64_t typeOffset;
  if (failed(result.getStridesAndOffset(strides, typeOffset)) ||
      strides != dense || view.getStaticStrides() != ArrayRef<int64_t>(dense) ||
      (typeOffset != ShapedType::kDynamic && typeOffset != offset)) return false;
  return true;
}

bool isCompleteStorageViewOf(Value value, Value origin) {
  if (!value || !origin || !isa<BaseMemRefType>(value.getType()) ||
      !isa<BaseMemRefType>(origin.getType()))
    return false;
  while (value != origin) {
    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
    } else if (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) {
      if (!completeContiguousView(view))
        return false;
      value = view.getSource();
    } else {
      return false;
    }
  }
  return true;
}

FailureOr<Value> materializeCollectiveView(OpBuilder &builder, Location location,
                                          Value source, MemRefType requested) {
  auto original = dyn_cast<MemRefType>(source.getType());
  if (!isCompleteLocalStorageView(source) || !localCapacity(requested) ||
      requested.getElementType() != original.getElementType() ||
      requested.getNumElements() != original.getNumElements())
    return emitError(location,
                     "collective view must preserve complete contiguous local storage"),
           failure();
  if (source.getType() == requested) return source;
  Value backing = source;
  while (true) {
    if (auto view = backing.getDefiningOp<memref::ReinterpretCastOp>())
      backing = view.getSource();
    else if (auto cast = backing.getDefiningOp<memref::CastOp>())
      backing = cast.getSource();
    else
      break;
    if (backing.getType() == requested) return backing;
  }
  return builder.create<memref::ReinterpretCastOp>(
      location, requested, backing, int64_t(0), requested.getShape(),
      contiguousStrides(requested)).getResult();
}

} // namespace intent::dsa
