#include "Collectives.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "llvm/ADT/DenseSet.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Storage.h"
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

void foldIntegerScanTails(func::FuncOp kernel) {
  SmallVector<gpu::GatherOp> gathers;
  kernel.walk([&](gpu::GatherOp gather) { gathers.push_back(gather); });
  bool changed = false;
  for (gpu::GatherOp gather : gathers) {
    auto type = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    if (!type || type.getShape().size() != 1 ||
        (!type.getElementType().isInteger(32) &&
         !type.getElementType().isInteger(64)) ||
        gather.getType() != type.getElementType() ||
        gather.getSourceAxes() != ArrayRef<int64_t>{0} ||
        gather.getCoordinates().size() != 1 ||
        (gather.getValid() &&
         (!gather.getValid().getType().isInteger(1) || !gather.getFill())))
      continue;
    auto extent = cast<gpu::PhysicalExprAttr>(type.getShape()[0]);
    bool positive = extent.getKind() ==
                        gpu::PhysicalExprKind::Constant &&
                    extent.getValue() > 0;
    if (extent.getKind() ==
        gpu::PhysicalExprKind::Parameter) {
      auto parameter = gpu::queryParameterBySymbol(kernel, extent.getParameterReference().getName());
      positive = succeeded(parameter) && llvm::all_of(
          parameter->getCandidates().asArrayRef(),
          [](int64_t candidate) { return candidate > 0; });
    }
    if (!positive)
      continue;
    Value index = gather.getCoordinates().front();
    auto coordinate = gpu::queryLaunchExpression(index);
    bool last = coordinate &&
                coordinate.getKind() ==
                    gpu::PhysicalExprKind::Constant &&
                extent.getKind() == coordinate.getKind() &&
                coordinate.getValue() == extent.getValue() - 1;
    if (auto subtract = index.getDefiningOp<gpu::BinaryOp>();
        subtract && subtract.getOperatorKind() == BinaryOperator::Subtract) {
      auto one = gpu::queryLaunchExpression(subtract.getRhs());
      last |= gpu::queryLaunchExpression(subtract.getLhs()) == extent && one &&
              one.getKind() ==
                  gpu::PhysicalExprKind::Constant &&
              one.getValue() == 1;
    }
    if (!last)
      continue;

    Value source = gather.getSource();
    Value base;
    auto addition = source.getDefiningOp<gpu::BinaryOp>();
    if (addition && addition.getOperatorKind() == BinaryOperator::Add) {
      auto scalar = [&](Value value) -> Value {
        if (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>())
          value = broadcast.getValue();
        else if (auto splat = value.getDefiningOp<gpu::SplatOp>())
          value = splat.getValue();
        return value.getType() == type.getElementType() ? value : Value();
      };
      if ((base = scalar(addition.getLhs())))
        source = addition.getRhs();
      else if ((base = scalar(addition.getRhs())))
        source = addition.getLhs();
    }
    auto scan = source.getDefiningOp<gpu::ScanOp>();
    if (!scan || scan.getSources().size() != 1 || scan.getIdentities().size() != 1 ||
        scan.getCaptures().size() || scan.getAxis() != 0 || !scan.getInclusive() ||
        scan.getReverse() || source.getType() != type ||
        !gpu::isLiteralZeroProjection(scan.getIdentities().front()))
      continue;
    Block &body = scan.getCombine().front();
    auto combine = dyn_cast<gpu::BinaryOp>(body.front());
    auto structured = cast<StructuredOpInterface>(scan.getOperation());
    auto binaryCombine = gpu::queryBinaryCombine(scan.getCombine());
    if (!llvm::hasSingleElement(body.without_terminator()) || !combine ||
        combine.getLhs() != structured.getCombineLhs().front() ||
        !binaryCombine || binaryCombine->kind() != BinaryOperator::Add)
      continue;

    // A scalar cross-warp gather materializes the whole prefix in shared memory.
    // Integer addition gives the same terminal value through a scalar reduction.
    OpBuilder builder(gather);
    Value zero = builder.create<arith::ConstantOp>(
        gather.getLoc(), builder.getZeroAttr(type.getElementType()));
    auto reduced = builder.create<gpu::ReduceOp>(gather.getLoc(),
        TypeRange{type.getElementType()}, scan.getSources(), ValueRange{zero},
        ValueRange{}, ArrayRef<int64_t>{0});
    if (Attribute origin = gather->getAttr(gpu::originAttr))
      reduced->setAttr(gpu::originAttr, origin);
    {
      OpBuilder::InsertionGuard guard(builder);
      Block *scalarBody = new Block();
      reduced.getCombine().push_back(scalarBody);
      Value lhs = scalarBody->addArgument(type.getElementType(), gather.getLoc());
      Value rhs = scalarBody->addArgument(type.getElementType(), gather.getLoc());
      builder.setInsertionPointToEnd(scalarBody);
      auto sum = builder.create<gpu::BinaryOp>(
          gather.getLoc(), type.getElementType(), lhs, rhs, BinaryOperator::Add);
      sum->setAttrs(combine->getAttrs());
      builder.create<gpu::YieldOp>(gather.getLoc(), sum.getResult());
    }
    Value result = reduced.getResult(0);
    if (base) {
      auto sum = builder.create<gpu::BinaryOp>(
          gather.getLoc(), type.getElementType(), base, result,
          BinaryOperator::Add);
      sum->setAttrs(addition->getAttrs());
      result = sum;
    }
    if (gather.getValid())
      result = builder.create<gpu::SelectOp>(
          gather.getLoc(), type.getElementType(), gather.getValid(), result,
          gather.getFill());
    gather.getResult().replaceAllUsesWith(result);
    gather.erase();
    changed = true;
  }
  if (changed)
    gpu::eraseDeadPhysicalValues(kernel);
}

} // namespace intent::triton::detail
