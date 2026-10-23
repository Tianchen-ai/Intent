#include "ComputeForms.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Workspace.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace intent::cutile {

LogicalResult materializeHistograms(func::FuncOp kernel) {
  SmallVector<gpu::HistogramOp> histograms;
  kernel.walk([&](gpu::HistogramOp histogram) { histograms.push_back(histogram); });
  for (gpu::HistogramOp histogram : histograms) {
    OpBuilder builder(histogram);
    Location location = histogram.getLoc();
    auto source = cast<gpu::FragmentType>(histogram.getValues().getType());
    auto result = cast<gpu::FragmentType>(histogram.getResult().getType());
    auto count = cast<IntegerType>(result.getElementType());
    // Native atomics use word-sized storage; the final integer conversion
    // preserves the source count dtype's modular arithmetic.
    Type storage = count.getWidth() <= 32 ? Type(builder.getI32Type())
                                         : Type(builder.getI64Type());
    auto fragment = [&](gpu::FragmentType type, Type element) {
      return gpu::FragmentType::get(kernel.getContext(), element, type.getShape(),
          type.getAxisMaps(), type.getValidity(), type.getOwner());
    };
    auto counts = fragment(result, storage);
    auto buffer = gpu::createProgramBuffer(builder, location, storage,
                                           result.getShape(), result.getOwner());
    auto map = cast<gpu::AxisMapAttr>(result.getAxisMaps()[0]);
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    Value bins = builder.create<gpu::MakeRangeOp>(location,
        fragment(result, builder.getIndexType()), zero, histogram.getBins(), one,
        zero, histogram.getBins(), map.getSourceId(), map.getSourceAxis(),
        map.getDerived());
    Value zeroCount = *gpu::materializeScalarConstant(builder, location,
        builder.getIntegerAttr(storage, 0), storage);
    Value initial = builder.create<gpu::SplatOp>(location, counts, zeroCount);
    // Every physical program owns its own workspace slice. Initialization,
    // atomic updates and readback are ordered by cuTile's alias token chain;
    // no initialization launch or cross-program synchronization is introduced.
    builder.create<gpu::StoreOp>(location, buffer, ValueRange{bins}, initial,
                                Value{}, ArrayRef<int64_t>{0});
    Value indices = builder.create<gpu::CastOp>(location,
        fragment(source, builder.getIndexType()), histogram.getValues());
    auto predicate = fragment(source, builder.getI1Type());
    Value lower = builder.create<gpu::SplatOp>(location, indices.getType(), zero);
    Value upper = builder.create<gpu::SplatOp>(location, indices.getType(),
                                             histogram.getBins());
    Value nonnegative = builder.create<gpu::CompareOp>(location, predicate,
        indices, lower, ComparePredicate::Ge);
    Value below = builder.create<gpu::CompareOp>(location, predicate,
        indices, upper, ComparePredicate::Lt);
    // Histogram's active-member contract supplies these bounds. Express them
    // at the memory access so workspace projection need not reconstruct the
    // source's subtracted bin origin through narrower integer arithmetic.
    Value valid = builder.create<gpu::BinaryOp>(location, predicate,
        nonnegative, below, BinaryOperator::LogicalAnd);
    if (histogram.getValid()) {
      Value active = builder.create<gpu::BroadcastOp>(location, predicate,
                                                      histogram.getValid());
      valid = builder.create<gpu::BinaryOp>(location, predicate,
          valid, active, BinaryOperator::LogicalAnd);
    }
    Value scalar = *gpu::materializeScalarConstant(builder, location,
        builder.getIntegerAttr(storage, 1), storage);
    auto updates = fragment(source, storage);
    Value ones = builder.create<gpu::SplatOp>(location, updates, scalar);
    builder.create<gpu::AtomicRMWOp>(location, updates, buffer,
        ValueRange{indices}, ones, valid, AtomicRMWKind::Add,
        AtomicOrdering::Relaxed, gpu::AtomicSharingDomain::ProgramInstance,
        ArrayRef<int64_t>{0});
    Value output = builder.create<gpu::LoadOp>(location, counts, buffer,
        ValueRange{bins}, Value{}, Value{}, ArrayRef<int64_t>{0});
    if (counts != result)
      output = builder.create<gpu::CastOp>(location, result, output);
    histogram.replaceAllUsesWith(output);
    histogram.erase();
  }
  return success();
}

} // namespace intent::cutile
