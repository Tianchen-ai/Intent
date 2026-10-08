#include "Intent/Target/BangC/NativeWorkspace.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir;
namespace intent::bangc {
Type arithmeticStorageType(Type element) {
  return element.isBF16() ? Float32Type::get(element.getContext()) : element;
}
SmallVector<int64_t, 2> exponentialScratchShape(Type element, int64_t elements) {
  if (!element.isF32() || elements < 1024) return {};
  return {4, std::min<int64_t>(8192, elements)};
}
SmallVector<int64_t, 2> extremaScratchShape(Type element, int64_t elements) {
  if ((!element.isF16() && !element.isF32()) || elements <= 0) return {};
  return {1, std::min<int64_t>(8192, elements)};
}
SmallVector<int64_t, 2> selectionScratchShape(Type element, int64_t elements,
                                            bool tensorFalseValue) {
  if (!element.isF32() || elements < 64) return {};
  return {tensorFalseValue ? 2 : 1,
          int64_t(1) << llvm::Log2_64(std::min<int64_t>(8192, elements))};
}
SmallVector<int64_t, 2> rowOffsetsShape(int64_t rows) { return {1, rows}; }
std::optional<GatherWorkspace> gatherWorkspace(int64_t rows) {
  if (rows < 64 || rows % 64 || rows > 65536) return std::nullopt;
  return GatherWorkspace{{3, rows}, {1, rows}, 512};
}

std::optional<NativeWorkspace> queryNativeWorkspace(Operation *operation,
                                                    ArrayRef<int64_t> shape) {
  Builder builder(operation->getContext());
  NativeWorkspace result;
  int64_t elements = 1;
  for (int64_t extent : shape) {
    if (extent <= 0 || elements > std::numeric_limits<int64_t>::max() / extent)
      return std::nullopt;
    elements *= extent;
  }
  auto add = [&](Type element, ArrayRef<int64_t> sizes) {
    if (!sizes.empty()) result.buffers.push_back(MemRefType::get(sizes, element,
        MemRefLayoutAttrInterface{}, builder.getI64IntegerAttr(dsa::nramSpace)));
  };
  auto arithmetic = [&](Type element, unsigned inputs) {
    Type storage = arithmeticStorageType(element);
    if (storage != element)
      for (unsigned slot = 0; slot <= inputs; ++slot) add(storage, shape);
    return storage;
  };
  auto unary = [&](UnaryOperator kind, Type element, bool approximate, bool flush,
                   bool hasScratch) {
    if (kind != UnaryOperator::Exp || approximate || flush ||
        (!element.isF16() && !element.isBF16() && !element.isF32())) return false;
    Type storage = arithmetic(element, 1);
    if (!hasScratch) add(storage, exponentialScratchShape(storage, elements));
    return true;
  };
  auto binary = [&](BinaryOperator kind, Type element, bool approximate,
                    bool flush, unsigned inputs, bool hasScratch) {
    if (approximate || flush ||
        (kind != BinaryOperator::Add && kind != BinaryOperator::Subtract &&
         kind != BinaryOperator::Multiply && kind != BinaryOperator::Maximum &&
         kind != BinaryOperator::Minimum && kind != BinaryOperator::MaximumNum &&
         kind != BinaryOperator::MinimumNum)) return false;
    Type storage = arithmetic(element, inputs);
    if (!hasScratch && (kind == BinaryOperator::Maximum || kind == BinaryOperator::Minimum ||
                        kind == BinaryOperator::MaximumNum || kind == BinaryOperator::MinimumNum))
      add(storage, extremaScratchShape(storage, elements));
    return true;
  };
  auto gather = [&](int64_t rows, bool offsets) {
    if (offsets) add(builder.getI64Type(), rowOffsetsShape(rows));
    if (auto workspace = gatherWorkspace(rows)) {
      add(builder.getI64Type(), workspace->plan);
      add(builder.getF32Type(), workspace->indices);
      result.internalBytes = workspace->internalBytes;
    }
  };
  if (auto op = dyn_cast<intent::UnaryOp>(operation)) {
    if (!unary(op.getOperatorKind(), getElementTypeOrSelf(op.getResult().getType()),
               op.getApproximate(), op.getFlushToZero(), false)) return std::nullopt;
  } else if (auto op = dyn_cast<dsa::UnaryOp>(operation)) {
    if (!unary(op.getKind(), cast<MemRefType>(op.getOutput().getType()).getElementType(),
               op.getApproximate(), op.getFlushToZero(), bool(op.getScratch()))) return std::nullopt;
  } else if (auto op = dyn_cast<intent::BinaryOp>(operation)) {
    unsigned inputs = llvm::count_if(op->getOperandTypes(), [](Type type) { return isa<ShapedType>(type); });
    if (!binary(op.getOperatorKind(), getElementTypeOrSelf(op.getResult().getType()),
                op.getApproximate(), op.getFlushToZero(), inputs, false)) return std::nullopt;
  } else if (auto op = dyn_cast<dsa::BinaryOp>(operation)) {
    if (!binary(op.getKind(), cast<MemRefType>(op.getOutput().getType()).getElementType(),
                op.getApproximate(), op.getFlushToZero(),
                isa<MemRefType>(op.getRhs().getType()) ? 2 : 1, bool(op.getScratch()))) return std::nullopt;
  } else if (isa<intent::ViewLoadOp>(operation)) {
    gather(shape.empty() ? 1 : shape.front(), true);
  } else if (auto op = dyn_cast<dsa::GatherRowsOp>(operation)) {
    if (!op.getPlan()) gather(shape.front(), false);
  } else if (isa<intent::SelectOp, intent::MaskOp>(operation)) {
    add(builder.getI32Type(), selectionScratchShape(
        getElementTypeOrSelf(operation->getResult(0).getType()), elements, true));
  } else if (auto op = dyn_cast<dsa::SelectOp>(operation)) {
    if (!op.getScratch() && isa<MemRefType>(op.getCondition().getType()))
      add(builder.getI32Type(), selectionScratchShape(
          cast<MemRefType>(op.getOutput().getType()).getElementType(), elements,
          isa<MemRefType>(op.getFalseValue().getType())));
  } else if (!isa<intent::ContractOp, intent::CastOp, intent::TransposeOp,
                  intent::BroadcastOp, intent::FullOp, intent::IndicesOp,
                  intent::GatherOp, dsa::FillOp, dsa::LoadTileOp,
                  dsa::StoreTileOp, dsa::LoadScalarOp, dsa::CastOp,
                  dsa::TransposeOp>(operation)) return std::nullopt;
  return result;
}
} // namespace intent::bangc
