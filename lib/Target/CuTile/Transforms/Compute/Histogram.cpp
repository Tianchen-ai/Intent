#include "ComputeForms.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Workspace.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <optional>

using namespace mlir;

namespace intent::cutile {
namespace {

struct CountAccumulation {
  scf::ForOp loop;
  gpu::BinaryOp add;
};

std::optional<CountAccumulation> countAccumulation(gpu::HistogramOp histogram) {
  auto loop = dyn_cast<scf::ForOp>(histogram->getParentOp());
  if (!loop || loop.getNumResults() != 1 || !histogram->hasOneUse() ||
      !loop.isDefinedOutsideOfLoop(histogram.getBins())) return std::nullopt;
  auto add = dyn_cast<gpu::BinaryOp>(*histogram->getUsers().begin());
  Value carry = loop.getRegionIterArgs()[0];
  if (!add || add.getOperatorKind() != BinaryOperator::Add || !carry.hasOneUse() ||
      !add->hasOneUse() || add->getBlock() != loop.getBody() ||
      !((add.getLhs() == carry && add.getRhs() == histogram.getResult()) ||
        (add.getRhs() == carry && add.getLhs() == histogram.getResult())) ||
      loop.getBody()->getTerminator()->getOperand(0) != add.getResult())
    return std::nullopt;
  return CountAccumulation{loop, add};
}

void replaceAccumulation(CountAccumulation accumulation,
                         gpu::HistogramOp histogram, Value output) {
  auto loop = accumulation.loop;
  loop.getResult(0).replaceAllUsesWith(output);
  OpBuilder builder(loop);
  auto replacement = builder.create<scf::ForOp>(loop.getLoc(), loop.getLowerBound(),
      loop.getUpperBound(), loop.getStep());
  replacement->setAttrs(loop->getAttrs());
  loop.getInductionVar().replaceAllUsesWith(replacement.getInductionVar());
  loop.getBody()->getTerminator()->setOperands({});
  accumulation.add.erase();
  histogram.erase();
  for (Operation &operation : llvm::make_early_inc_range(loop.getBody()->without_terminator()))
    operation.moveBefore(replacement.getBody()->getTerminator());
  loop.erase();
}

} // namespace

LogicalResult materializeHistograms(func::FuncOp kernel) {
  SmallVector<gpu::HistogramOp> histograms;
  kernel.walk([&](gpu::HistogramOp histogram) { histograms.push_back(histogram); });
  for (gpu::HistogramOp histogram : histograms) {
    OpBuilder builder(histogram);
    Location location = histogram.getLoc();
    auto source = cast<gpu::FragmentType>(histogram.getValues().getType());
    auto result = cast<gpu::FragmentType>(histogram.getResult().getType());
    auto count = cast<IntegerType>(result.getElementType());
    auto accumulation = countAccumulation(histogram);
    if (accumulation) builder.setInsertionPoint(accumulation->loop);
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
    if (accumulation) {
      initial = accumulation->loop.getInitArgs()[0];
      if (initial.getType() != counts)
        initial = builder.create<gpu::CastOp>(location, counts, initial);
    }
    // Every physical program owns its own workspace slice. Initialization,
    // atomic updates and readback are ordered by cuTile's alias token chain;
    // no initialization launch or cross-program synchronization is introduced.
    builder.create<gpu::StoreOp>(location, buffer, ValueRange{bins}, initial,
                                Value{}, ArrayRef<int64_t>{0});
    builder.setInsertionPoint(histogram);
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
    // Integer sum of local histograms is the same modular count update. Keep
    // one private accumulator across the proved loop instead of clearing and
    // reading the same bins after every input chunk.
    if (accumulation) builder.setInsertionPointAfter(accumulation->loop);
    Value output = builder.create<gpu::LoadOp>(location, counts, buffer,
        ValueRange{bins}, Value{}, Value{}, ArrayRef<int64_t>{0});
    if (counts != result)
      output = builder.create<gpu::CastOp>(location, result, output);
    if (accumulation) replaceAccumulation(*accumulation, histogram, output);
    else {
      histogram.replaceAllUsesWith(output);
      histogram.erase();
    }
  }
  return success();
}

} // namespace intent::cutile
