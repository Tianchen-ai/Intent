#include "Collectives.h"
#include "Intent/Dialect/Block/Transforms/Value/Helpers.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
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

LogicalResult legalizeCollectiveCallbacks(func::FuncOp kernel) {
  SmallVector<Operation *> collectives;
  kernel.walk([&](Operation *operation) {
    if (isa<block::ReduceOp, block::ScanOp>(operation))
      collectives.push_back(operation);
  });
  for (Operation *operation : collectives) {
    auto reduce = dyn_cast<block::ReduceOp>(operation);
    auto scan = dyn_cast<block::ScanOp>(operation);
    ValueRange sources = reduce ? reduce.getSources() : scan.getSources();
    ValueRange identities = reduce ? reduce.getIdentities() : scan.getIdentities();
    auto sourceType = sources.empty() ? block::FragmentType()
        : dyn_cast<block::FragmentType>(sources.front().getType());
    bool allAxes = reduce && sourceType && sourceType.getShape().size() > 1 &&
        reduce.getAxes().size() == sourceType.getShape().size();
    if ((reduce && ((!allAxes && reduce.getAxes().size() != 1) || reduce.getCaptures().size())) ||
        (scan && scan.getCaptures().size()))
      return operation->emitOpError("native collective requires one axis or a full reduction and a capture-free callback");
    // ODS inferred builders treat failure as a construction error. Diagnose
    // unsupported native schemas before invoking that builder, and retain the
    // result contract of the shared operation being replaced.
    SmallVector<Type> resultTypes;
    int64_t axis = reduce ? (allAxes ? -1 : reduce.getAxes().front()) : scan.getAxis();
    if (failed(block::inferScalarCollectiveResultTypes(
            operation->getLoc(), sources, axis, bool(scan), resultTypes)))
      return failure();
    if (!llvm::equal(resultTypes, operation->getResultTypes()))
      return operation->emitOpError(
          "native collective cannot preserve the shared result schema");
    Region &region = reduce ? reduce.getCombine() : scan.getCombine();
    OpBuilder builder(operation);
    Operation *native;
    Region *combine;
    if (reduce) {
      auto replacement = builder.create<ReduceOp>(operation->getLoc(),
          sources, identities, axis, false);
      native = replacement;
      combine = &replacement.getCombine();
    } else {
      auto replacement = builder.create<ScanOp>(operation->getLoc(),
          sources, identities, axis, scan.getReverse());
      native = replacement;
      combine = &replacement.getCombine();
    }
    if (Attribute origin = operation->getAttr(block::originAttr)) native->setAttr(block::originAttr, origin);
    if (failed(block::scalarizeElementwiseCallback(region, *combine)))
      return failure();
    SmallVector<Value> results(native->getResults());
    if (scan && !scan.getInclusive()) {
      builder.setInsertionPointAfter(native);
      Location location = scan.getLoc();
      for (unsigned component = 0; component < sources.size(); ++component) {
        auto type = dyn_cast<block::FragmentType>(results[component].getType());
        if (!type || scan.getAxis() >= type.getShape().size())
          return scan.emitOpError("exclusive scan requires a ranked physical result");
        auto axis = cast<block::AxisMapAttr>(type.getAxisMaps()[scan.getAxis()]);
        auto extent = cast<block::PhysicalExprAttr>(type.getShape()[scan.getAxis()]);
        auto ordinalAxis = block::AxisMapAttr::get(
            kernel.getContext(), axis.getSourceId(), axis.getSourceAxis(),
            axis.getDimensionId(), 0, axis.getDerived());
        auto rangeType = block::FragmentType::get(
            kernel.getContext(), builder.getIndexType(),
            builder.getArrayAttr({extent}), builder.getArrayAttr({ordinalAxis}),
            type.getValidity(), type.getOwner());
        auto indexType = block::FragmentType::get(
            kernel.getContext(), builder.getIndexType(), type.getShape(),
            type.getAxisMaps(), type.getValidity(), type.getOwner());
        auto predicateType = block::FragmentType::get(
            kernel.getContext(), builder.getI1Type(), type.getShape(),
            type.getAxisMaps(), type.getValidity(), type.getOwner());
        Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
        Value one = builder.create<arith::ConstantIndexOp>(location, 1);
        Value size = builder.create<block::PhysicalExprOp>(
            location, builder.getIndexType(), extent);
        Value ordinal = builder.create<block::MakeRangeOp>(
            location, rangeType, zero, size, one, zero, size,
            axis.getSourceId(), axis.getSourceAxis(), axis.getDerived());
        ordinal = builder.create<block::BroadcastOp>(location, indexType, ordinal);
        Value step = builder.create<block::BroadcastOp>(location, indexType, one);
        Value first = builder.create<block::BroadcastOp>(location, indexType, zero);
        Value shifted = builder.create<block::BinaryOp>(
            location, indexType, ordinal, step,
            scan.getReverse() ? BinaryOperator::Add : BinaryOperator::Subtract);
        Value bound = scan.getReverse()
                          ? Value(builder.create<block::BroadcastOp>(location, indexType, size))
                          : first;
        Value valid = builder.create<block::CompareOp>(
            location, predicateType, shifted, bound,
            scan.getReverse() ? ComparePredicate::Lt : ComparePredicate::Ge);
        Value safeIndex = builder.create<block::SelectOp>(
            location, indexType, valid, shifted, first);
        Value prefix = builder.create<block::GatherOp>(
            location, type, results[component], ValueRange{safeIndex},
            Value(), Value(), ArrayRef<int64_t>{static_cast<int64_t>(scan.getAxis())});
        Value identity = builder.create<block::BroadcastOp>(
            location, type, scan.getIdentities()[component]);
        results[component] = builder.create<block::SelectOp>(
            location, type, valid, prefix, identity);
      }
    }
    operation->replaceAllUsesWith(results);
    operation->erase();
  }
  return success();
}

} // namespace intent::triton::detail
