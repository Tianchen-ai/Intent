#include "StrictArithmetic.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"

using namespace mlir;

namespace intent::cutile {

void legalizeStrictArithmetic(func::FuncOp kernel) {
  SmallVector<gpu::BinaryOp> operations;
  kernel.walk([&](gpu::BinaryOp operation) {
    if (operation.getStrictRounding()) operations.push_back(operation);
  });
  for (gpu::BinaryOp operation : operations) {
    OpBuilder builder(operation);
    Type original = operation.getResult().getType();
    Type wide = builder.getF64Type();
    if (auto fragment = dyn_cast<gpu::FragmentType>(original))
      wide = gpu::FragmentType::get(kernel.getContext(), wide,
          fragment.getShape(), fragment.getAxisMaps(), fragment.getValidity(),
          fragment.getOwner());
    Location location = operation.getLoc();
    Value lhs = builder.create<gpu::CastOp>(location, wide, operation.getLhs());
    Value rhs = builder.create<gpu::CastOp>(location, wide, operation.getRhs());
    // cuTile's frontend may contract even explicitly RN f32 add/mul. Products
    // of two f32 values are exact in f64. For add/sub, either the exact sum
    // fits f64 or the smaller operand is too small to affect f32 rounding.
    // Thus this explicit narrowing is the original R32 boundary, including
    // gradual underflow and signed zero; it must not become a fused f32 op.
    Value computed = builder.create<gpu::BinaryOp>(
        location, wide, lhs, rhs, operation.getOperatorKind());
    Value rounded = builder.create<gpu::CastOp>(location, original, computed);
    operation.getResult().replaceAllUsesWith(rounded);
    operation.erase();
  }
}

} // namespace intent::cutile
