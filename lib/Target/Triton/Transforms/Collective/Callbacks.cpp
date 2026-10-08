#include "Collectives.h"
#include "Intent/Dialect/GPU/Transforms/Value/Helpers.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
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

LogicalResult legalizeCollectiveCallbacks(func::FuncOp kernel) {
  SmallVector<Operation *> collectives;
  kernel.walk([&](Operation *operation) {
    if (isa<gpu::ReduceOp, gpu::ScanOp>(operation))
      collectives.push_back(operation);
  });
  for (Operation *operation : collectives) {
    auto reduce = dyn_cast<gpu::ReduceOp>(operation);
    auto scan = dyn_cast<gpu::ScanOp>(operation);
    ValueRange sources = reduce ? reduce.getSources() : scan.getSources();
    ValueRange identities = reduce ? reduce.getIdentities() : scan.getIdentities();
    auto sourceType = sources.empty() ? gpu::FragmentType()
        : dyn_cast<gpu::FragmentType>(sources.front().getType());
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
    if (failed(gpu::inferScalarCollectiveResultTypes(
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
    if (Attribute origin = operation->getAttr(gpu::originAttr)) native->setAttr(gpu::originAttr, origin);
    if (failed(gpu::scalarizeElementwiseCallback(region, *combine)))
      return failure();
    SmallVector<Value> results(native->getResults());
    if (scan && !scan.getInclusive()) {
      builder.setInsertionPointAfter(native);
      Location location = scan.getLoc();
      for (unsigned component = 0; component < sources.size(); ++component) {
        auto type = dyn_cast<gpu::FragmentType>(results[component].getType());
        if (!type || scan.getAxis() >= type.getShape().size())
          return scan.emitOpError("exclusive scan requires a ranked physical result");
        auto axis = cast<gpu::AxisMapAttr>(type.getAxisMaps()[scan.getAxis()]);
        auto extent = cast<gpu::PhysicalExprAttr>(type.getShape()[scan.getAxis()]);
        auto ordinalAxis = gpu::AxisMapAttr::get(
            kernel.getContext(), axis.getSourceId(), axis.getSourceAxis(),
            axis.getDimensionId(), 0, axis.getDerived());
        auto rangeType = gpu::FragmentType::get(
            kernel.getContext(), builder.getIndexType(),
            builder.getArrayAttr({extent}), builder.getArrayAttr({ordinalAxis}),
            type.getValidity(), type.getOwner());
        auto indexType = gpu::FragmentType::get(
            kernel.getContext(), builder.getIndexType(), type.getShape(),
            type.getAxisMaps(), type.getValidity(), type.getOwner());
        auto predicateType = gpu::FragmentType::get(
            kernel.getContext(), builder.getI1Type(), type.getShape(),
            type.getAxisMaps(), type.getValidity(), type.getOwner());
        Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
        Value one = builder.create<arith::ConstantIndexOp>(location, 1);
        Value size = builder.create<gpu::PhysicalExprOp>(
            location, builder.getIndexType(), extent);
        Value ordinal = builder.create<gpu::MakeRangeOp>(
            location, rangeType, zero, size, one, zero, size,
            axis.getSourceId(), axis.getSourceAxis(), axis.getDerived());
        ordinal = builder.create<gpu::BroadcastOp>(location, indexType, ordinal);
        Value step = builder.create<gpu::BroadcastOp>(location, indexType, one);
        Value first = builder.create<gpu::BroadcastOp>(location, indexType, zero);
        Value shifted = builder.create<gpu::BinaryOp>(
            location, indexType, ordinal, step,
            scan.getReverse() ? BinaryOperator::Add : BinaryOperator::Subtract);
        Value bound = scan.getReverse()
                          ? Value(builder.create<gpu::BroadcastOp>(location, indexType, size))
                          : first;
        Value valid = builder.create<gpu::CompareOp>(
            location, predicateType, shifted, bound,
            scan.getReverse() ? ComparePredicate::Lt : ComparePredicate::Ge);
        Value safeIndex = builder.create<gpu::SelectOp>(
            location, indexType, valid, shifted, first);
        Value prefix = builder.create<gpu::GatherOp>(
            location, type, results[component], ValueRange{safeIndex},
            Value(), Value(), ArrayRef<int64_t>{static_cast<int64_t>(scan.getAxis())});
        Value identity = builder.create<gpu::BroadcastOp>(
            location, type, scan.getIdentities()[component]);
        results[component] = builder.create<gpu::SelectOp>(
            location, type, valid, prefix, identity);
      }
    }
    operation->replaceAllUsesWith(results);
    operation->erase();
  }
  return success();
}

} // namespace intent::triton::detail
