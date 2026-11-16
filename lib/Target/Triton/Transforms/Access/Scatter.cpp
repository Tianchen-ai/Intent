#include "Scatter.h"
#include "Intent/Dialect/Block/Analysis/Helpers.h"
#include "llvm/ADT/DenseSet.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Block/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/Block/Analysis/UniformValues.h"
#include "Intent/Dialect/Block/Analysis/ValueSchema.h"
#include "Intent/Dialect/Block/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/Block/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/Block/Transforms/Storage/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include <algorithm>
#include <limits>
#include <optional>


using namespace mlir;

namespace intent::triton::detail {

namespace {

bool isTritonAtomicAddType(Type type) {
  type = block::uniformElementType(type);
  if (isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(type))
    return true;
  auto integer = dyn_cast<IntegerType>(type);
  return integer && (integer.getWidth() == 32 || integer.getWidth() == 64);
}

bool isAddCombine(block::ScatterReduceOp scatter) {
  auto combine = block::queryBinaryCombine(scatter.getCombine());
  return combine && combine->kind() == BinaryOperator::Add;
}

} // namespace

LogicalResult legalizeScatterAdd(func::FuncOp kernel) {
  SmallVector<block::ScatterReduceOp> scatters;
  kernel.walk([&](block::ScatterReduceOp scatter) { scatters.push_back(scatter); });
  for (block::ScatterReduceOp scatter : scatters) {
    if (!isAddCombine(scatter) || !isTritonAtomicAddType(scatter.getValue().getType()))
      continue;
    OpBuilder builder(scatter);
    auto atomic = builder.create<block::AtomicRMWOp>(
        scatter.getLoc(), scatter.getValue().getType(), scatter.getResource(),
        scatter.getCoordinates(), scatter.getValue(), scatter.getValid(),
        AtomicRMWKind::Add, AtomicOrdering::Relaxed, scatter.getSharing(),
        scatter.getSourceAxes());
    if (Attribute origin = scatter->getAttr(block::originAttr))
      atomic->setAttr(block::originAttr, origin);
    scatter.erase();
  }
  return success();
}

} // namespace intent::triton::detail
