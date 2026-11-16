#include "Collectives.h"
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

void foldIntegerScanTails(func::FuncOp kernel) {
  SmallVector<block::GatherOp> gathers;
  kernel.walk([&](block::GatherOp gather) { gathers.push_back(gather); });
  bool changed = false;
  for (block::GatherOp gather : gathers) {
    auto type = dyn_cast<block::FragmentType>(gather.getSource().getType());
    if (!type || type.getShape().size() != 1 ||
        (!type.getElementType().isInteger(32) &&
         !type.getElementType().isInteger(64)) ||
        gather.getType() != type.getElementType() ||
        gather.getSourceAxes() != ArrayRef<int64_t>{0} ||
        gather.getCoordinates().size() != 1 ||
        (gather.getValid() &&
         (!gather.getValid().getType().isInteger(1) || !gather.getFill())))
      continue;
    auto extent = cast<block::PhysicalExprAttr>(type.getShape()[0]);
    bool positive = extent.getKind() ==
                        block::PhysicalExprKind::Constant &&
                    extent.getValue() > 0;
    if (extent.getKind() ==
        block::PhysicalExprKind::Parameter) {
      auto parameter = block::queryParameterBySymbol(kernel, extent.getParameterReference().getName());
      positive = succeeded(parameter) && llvm::all_of(
          parameter->getCandidates().asArrayRef(),
          [](int64_t candidate) { return candidate > 0; });
    }
    if (!positive)
      continue;
    Value index = gather.getCoordinates().front();
    auto coordinate = block::queryLaunchExpression(index);
    bool last = coordinate &&
                coordinate.getKind() ==
                    block::PhysicalExprKind::Constant &&
                extent.getKind() == coordinate.getKind() &&
                coordinate.getValue() == extent.getValue() - 1;
    if (auto subtract = index.getDefiningOp<block::BinaryOp>();
        subtract && subtract.getOperatorKind() == BinaryOperator::Subtract) {
      auto one = block::queryLaunchExpression(subtract.getRhs());
      last |= block::queryLaunchExpression(subtract.getLhs()) == extent && one &&
              one.getKind() ==
                  block::PhysicalExprKind::Constant &&
              one.getValue() == 1;
    }
    if (!last)
      continue;

    Value source = gather.getSource();
    Value base;
    auto addition = source.getDefiningOp<block::BinaryOp>();
    if (addition && addition.getOperatorKind() == BinaryOperator::Add) {
      auto scalar = [&](Value value) -> Value {
        if (auto broadcast = value.getDefiningOp<block::BroadcastOp>())
          value = broadcast.getValue();
        else if (auto splat = value.getDefiningOp<block::SplatOp>())
          value = splat.getValue();
        return value.getType() == type.getElementType() ? value : Value();
      };
      if ((base = scalar(addition.getLhs())))
        source = addition.getRhs();
      else if ((base = scalar(addition.getRhs())))
        source = addition.getLhs();
    }
    auto scan = source.getDefiningOp<block::ScanOp>();
    if (!scan || scan.getSources().size() != 1 || scan.getIdentities().size() != 1 ||
        scan.getCaptures().size() || scan.getAxis() != 0 || !scan.getInclusive() ||
        scan.getReverse() || source.getType() != type ||
        !block::isLiteralZeroProjection(scan.getIdentities().front()))
      continue;
    Block &body = scan.getCombine().front();
    auto combine = dyn_cast<block::BinaryOp>(body.front());
    auto structured = cast<StructuredOpInterface>(scan.getOperation());
    auto binaryCombine = block::queryBinaryCombine(scan.getCombine());
    if (!llvm::hasSingleElement(body.without_terminator()) || !combine ||
        combine.getLhs() != structured.getCombineLhs().front() ||
        !binaryCombine || binaryCombine->kind() != BinaryOperator::Add)
      continue;

    // A scalar cross-warp gather materializes the whole prefix in shared memory.
    // Integer addition gives the same terminal value through a scalar reduction.
    OpBuilder builder(gather);
    Value zero = builder.create<arith::ConstantOp>(
        gather.getLoc(), builder.getZeroAttr(type.getElementType()));
    auto reduced = builder.create<block::ReduceOp>(gather.getLoc(),
        TypeRange{type.getElementType()}, scan.getSources(), ValueRange{zero},
        ValueRange{}, ArrayRef<int64_t>{0});
    if (Attribute origin = gather->getAttr(block::originAttr))
      reduced->setAttr(block::originAttr, origin);
    {
      OpBuilder::InsertionGuard guard(builder);
      Block *scalarBody = new Block();
      reduced.getCombine().push_back(scalarBody);
      Value lhs = scalarBody->addArgument(type.getElementType(), gather.getLoc());
      Value rhs = scalarBody->addArgument(type.getElementType(), gather.getLoc());
      builder.setInsertionPointToEnd(scalarBody);
      auto sum = builder.create<block::BinaryOp>(
          gather.getLoc(), type.getElementType(), lhs, rhs, BinaryOperator::Add);
      sum->setAttrs(combine->getAttrs());
      builder.create<block::YieldOp>(gather.getLoc(), sum.getResult());
    }
    Value result = reduced.getResult(0);
    if (base) {
      auto sum = builder.create<block::BinaryOp>(
          gather.getLoc(), type.getElementType(), base, result,
          BinaryOperator::Add);
      sum->setAttrs(addition->getAttrs());
      result = sum;
    }
    if (gather.getValid())
      result = builder.create<block::SelectOp>(
          gather.getLoc(), type.getElementType(), gather.getValid(), result,
          gather.getFill());
    gather.getResult().replaceAllUsesWith(result);
    gather.erase();
    changed = true;
  }
  if (changed)
    block::eraseDeadPhysicalValues(kernel);
}

} // namespace intent::triton::detail
