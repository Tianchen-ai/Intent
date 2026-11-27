#pragma once
#include "compilerinfra/Contract.h"
#include "mlir/IR/BuiltinTypes.h"

namespace CI_UNIT_NAMESPACE {
inline std::optional<int64_t> approximateScratch(mlir::Operation *op) {
  using namespace mlir;
  StringRef name = op->getName().getStringRef();
  if (name != "block.sqrt" && name != "block.div")
    return {};
  auto out = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto flags = name == "block.sqrt" ? arith::FastMathFlags::afn
                                    : arith::FastMathFlags::arcp;
  if (!out || out.getRank() != 2 || !out.getElementType().isF32() ||
      !arith::bitEnumContainsAll(compilerinfra::effectiveFastMath(op), flags) ||
      (name == "block.sqrt" && (!compilerinfra::allowsInputSubnormalFlush(op) ||
                                !compilerinfra::allowsOutputSubnormalFlush(op))))
    return {};
  unsigned tiles = 0;
  for (auto v : op->getOperands()) {
    if (auto t = dyn_cast<RankedTensorType>(v.getType())) {
      if (!t.getElementType().isF32() ||
          (t.getRank() != 2 && !(name == "block.div" && t.getRank() == 1)))
        return {};
      tiles += t.getRank() == 2;
    } else if (!v.getType().isF32()) {
      return {};
    }
  }
  if (!tiles)
    return {};
  if (name == "block.div" && (!compilerinfra::allowsInputSubnormalFlush(op) ||
                              !compilerinfra::allowsOutputSubnormalFlush(op)))
    return 5 * sizeof(float);
  return tiles == 2 ? sizeof(float) : 0;
}
}
