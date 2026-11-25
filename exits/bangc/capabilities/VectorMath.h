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
      !compilerinfra::allowsInputSubnormalFlush(op) ||
      !compilerinfra::allowsOutputSubnormalFlush(op))
    return {};
  for (auto v : op->getOperands()) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (t ? t.getRank() != 2 || !t.getElementType().isF32() : !v.getType().isF32())
      return {};
  }
  bool vectorPair = name == "block.div" &&
                    isa<RankedTensorType>(op->getOperand(0).getType()) &&
                    isa<RankedTensorType>(op->getOperand(1).getType());
  return vectorPair ? sizeof(float) : 0;
}
}
