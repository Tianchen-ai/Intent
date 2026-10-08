#include "Gathers.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Storage/FragmentSnapshot.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Workspace.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>
#include <optional>


using namespace mlir;

namespace intent::triton::detail {

namespace {

struct GatherSourceCapacity {
  int64_t minimumElements = 1;
  int64_t maximumElements = 1;
  bool constant = true;
};

std::optional<GatherSourceCapacity> sourceCapacity(gpu::FragmentType source,
                                                  func::FuncOp kernel) {
  GatherSourceCapacity capacity;
  int64_t bytes = (source.getElementType().getIntOrFloatBitWidth() + 7) / 8;
  for (Attribute attribute : source.getShape()) {
    auto extent = cast<gpu::PhysicalExprAttr>(attribute);
    auto bounds = gpu::queryPositiveExtentBounds(extent, kernel);
    if (!bounds || capacity.maximumElements >
                       std::numeric_limits<int64_t>::max() / bounds->second)
      return std::nullopt;
    capacity.minimumElements *= bounds->first;
    capacity.maximumElements *= bounds->second;
    capacity.constant &= extent.getKind() == gpu::PhysicalExprKind::Constant;
  }
  if (capacity.maximumElements > std::numeric_limits<int64_t>::max() / bytes)
    return std::nullopt;
  return capacity;
}

scf::ForOp repeatedReadLoop(gpu::GatherOp gather, func::FuncOp kernel) {
  gpu::IndexRelations relations;
  for (Operation *parent = gather->getParentOp(); parent != kernel;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop || !loop.isDefinedOutsideOfLoop(gather.getSource())) continue;
    auto lower = relations.constant(loop.getLowerBound());
    auto upper = relations.constant(loop.getUpperBound());
    auto step = relations.constant(loop.getStep());
    if (lower && upper && step && *step > 0 &&
        static_cast<__int128>(*upper) - *lower <= *step)
      continue;
    llvm::DenseMap<Value, bool> varying;
    if (llvm::any_of(gather.getCoordinates(), [&](Value coordinate) {
          return gpu::variesWithIteration(coordinate, loop, varying);
        }))
      return loop;
  }
  return {};
}

bool isRepeatedScalarRead(gpu::GatherOp gather, func::FuncOp kernel,
                          const GatherSourceCapacity &capacity,
                          gpu::PhysicalProgramAnalysis &analysis) {
  auto source = cast<gpu::FragmentType>(gather.getSource().getType());
  auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (isa<gpu::FragmentType>(gather.getResult().getType()) ||
      gather.getSourceAxes().size() != source.getShape().size() ||
      !llvm::all_of(gather.getCoordinates(), [](Value value) {
        return value.getType().isIndex();
      }) || capacity.maximumElements <= capabilities.getMaxThreadsPerBlock())
    return false;
  for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
    if (analysis.axisRealization(gather.getSource(), axis).constructionScalarSeed)
      return false;
  return bool(repeatedReadLoop(gather, kernel));
}

bool endsSourceLifetimeBeforeReads(Value source,
                                  ArrayRef<gpu::GatherOp> readers,
                                  func::FuncOp kernel) {
  SmallVector<Operation *> loops;
  llvm::SmallPtrSet<Operation *, 8> selected;
  for (gpu::GatherOp reader : readers) {
    selected.insert(reader);
    auto loop = repeatedReadLoop(reader, kernel);
    if (!loop) return false;
    if (!llvm::is_contained(loops, loop.getOperation())) loops.push_back(loop);
  }
  DominanceInfo dominance(kernel);
  SmallVector<Value> values{source};
  llvm::DenseSet<Value> visited;
  for (unsigned position = 0; position < values.size(); ++position) {
    if (!visited.insert(values[position]).second) continue;
    for (Operation *user : values[position].getUsers()) {
      if (selected.contains(user)) continue;
      // Region init/carry and record packing can forward an entire payload.
      // Without their component/control flow proof, preceding the loop alone
      // does not establish that the native representation has become dead.
      if (user->getNumRegions() || llvm::any_of(user->getResultTypes(), [](Type type) {
            return isa<gpu::RecordType>(type);
          }))
        return false;
      // A full-value consumer in, beside or after the loop retains this SSA
      // representation. Adding a snapshot then creates two live versions.
      if (!llvm::all_of(loops, [&](Operation *loop) {
            return dominance.properlyDominates(user, loop, /*enclosingOpOk=*/false);
          }))
        return false;
      // These values may share the source's native representation. A wrapper
      // before the loop does not end its lifetime if its users are still inside.
      if (isa<gpu::CastOp, gpu::BitcastOp, gpu::BroadcastOp,
              gpu::ReshapeOp, gpu::TransposeOp, gpu::SelectOp>(user))
        llvm::append_range(values, user->getResults());
    }
  }
  return true;
}

Value largeSourceGuard(Value source, int64_t threshold) {
  auto type = cast<gpu::FragmentType>(source.getType());
  OpBuilder builder(source.getContext());
  if (Operation *definition = source.getDefiningOp())
    builder.setInsertionPointAfter(definition);
  else
    builder.setInsertionPointToStart(cast<BlockArgument>(source).getOwner());
  Location location = source.getLoc();
  Value elements = builder.create<arith::ConstantIndexOp>(location, 1);
  for (Attribute attribute : type.getShape()) {
    Value extent = builder.create<gpu::PhysicalExprOp>(
        location, builder.getIndexType(), cast<gpu::PhysicalExprAttr>(attribute));
    elements = builder.create<gpu::BinaryOp>(location, builder.getIndexType(),
                                            elements, extent, BinaryOperator::Multiply);
  }
  Value limit = builder.create<arith::ConstantIndexOp>(location, threshold);
  return builder.create<gpu::CompareOp>(location, builder.getI1Type(), elements,
                                        limit, ComparePredicate::Gt);
}

} // namespace

LogicalResult materializeGatherSources(func::FuncOp kernel) {
  auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  gpu::PhysicalProgramAnalysis analysis(kernel);
  llvm::MapVector<Value, SmallVector<gpu::GatherOp>> readers;
  kernel.walk([&](gpu::GatherOp gather) {
    if (!gpu::isProgramAllocationContext(gather, kernel))
      return;
    auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
    if (!source || gather.getCoordinates().empty() ||
        belongsToSplitGatherPair(gather) ||
        !isa<FloatType, IntegerType>(source.getElementType()))
      return;
    auto capacity = sourceCapacity(source, kernel);
    if (!capacity) return;
    int64_t bytes = (source.getElementType().getIntOrFloatBitWidth() + 7) / 8;
    bool oversized = capacity->constant && capacity->maximumElements * bytes >
                                               capabilities.getMaxDynamicSharedMemoryPerBlock();
    if (!oversized && !isRepeatedScalarRead(gather, kernel, *capacity, analysis)) return;
    SmallVector<Value> selected(source.getShape().size());
    for (auto [coordinate, axis] :
         llvm::zip(gather.getCoordinates(), gather.getSourceAxes()))
      selected[axis] = coordinate;
    for (unsigned axis = 0; axis < source.getShape().size(); ++axis) {
      Type coordinateType;
      if (selected[axis]) {
        coordinateType = selected[axis].getType();
      } else {
        if (!result)
          return;
        auto mapping = cast<gpu::AxisMapAttr>(source.getAxisMaps()[axis]);
        auto projection = gpu::queryFragmentAxis(result, gpu::sourceAxisIdentity(mapping));
        if (!projection.isExact() ||
            projection.dimensionId != mapping.getDimensionId() ||
            result.getShape()[projection.fragmentAxis] != source.getShape()[axis])
          return;
        auto ordinal = gpu::AxisMapAttr::get(kernel.getContext(),
            mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDimensionId(),
            0, mapping.getDerived());
        coordinateType = gpu::FragmentType::get(kernel.getContext(),
            IndexType::get(kernel.getContext()), ArrayAttr::get(kernel.getContext(),
                {source.getShape()[axis]}), ArrayAttr::get(kernel.getContext(), {ordinal}),
            source.getValidity(), source.getOwner());
      }
      if (auto coordinate = dyn_cast<gpu::FragmentType>(coordinateType)) {
        if (!result)
          return;
        auto projected = gpu::FragmentType::get(kernel.getContext(),
            coordinate.getElementType(), result.getShape(), result.getAxisMaps(),
            result.getValidity(), result.getOwner());
        if (!gpu::queryBroadcastProjection(coordinate, projected).isExact())
          return;
      }
    }
    readers[gather.getSource()].push_back(gather);
  });
  if (readers.empty())
    return success();
  for (auto &readerGroup : readers) {
    auto &gathers = readerGroup.second;
    Value source = gathers.front().getSource();
    auto payload = cast<gpu::FragmentType>(source.getType());
    auto capacity = sourceCapacity(payload, kernel);
    if (!capacity) return gathers.front().emitOpError("gather snapshot lost its finite source capacity");
    int64_t bytes = (payload.getElementType().getIntOrFloatBitWidth() + 7) / 8;
    bool oversized = capacity->constant && capacity->maximumElements * bytes >
                                               capabilities.getMaxDynamicSharedMemoryPerBlock();
    if (!oversized && !endsSourceLifetimeBeforeReads(source, gathers, kernel)) continue;
    // Large-source selection is a profitability choice, not a proof about the
    // native layout. Bind unknown coverage sizes with the actual constexpr
    // extent so inexpensive small-domain gathers retain their native path.
    Value sizeEnabled;
    if (!oversized && capacity->minimumElements <= capabilities.getMaxThreadsPerBlock())
      sizeEnabled = largeSourceGuard(source, capabilities.getMaxThreadsPerBlock());
    Value domainEnabled = gpu::materializeFragmentSnapshotReadDomain(source, gathers);
    Value enabled = sizeEnabled ? sizeEnabled : domainEnabled;
    if (sizeEnabled && domainEnabled) {
      Operation *after = sizeEnabled.getDefiningOp();
      if (Operation *domain = domainEnabled.getDefiningOp();
          domain && domain->getBlock() == after->getBlock() &&
          after->isBeforeInBlock(domain))
        after = domain;
      OpBuilder builder(after);
      builder.setInsertionPointAfter(after);
      enabled = builder.create<gpu::BinaryOp>(source.getLoc(), builder.getI1Type(),
          sizeEnabled, domainEnabled, BinaryOperator::LogicalAnd);
    }
    auto storage = gpu::materializeFragmentSnapshot(source, enabled);
    if (failed(storage)) return failure();
    for (gpu::GatherOp gather : gathers) {
      Value replacement;
      if (!sizeEnabled) {
        auto value = gpu::loadFragmentSnapshot(gather, *storage);
        if (failed(value)) return failure();
        replacement = *value;
      } else {
        OpBuilder builder(gather);
        auto branch = builder.create<scf::IfOp>(gather.getLoc(),
            TypeRange{gather.getResult().getType()}, sizeEnabled, true);
        builder.setInsertionPointToStart(branch.elseBlock());
        auto native = cast<gpu::GatherOp>(builder.clone(*gather));
        builder.create<scf::YieldOp>(gather.getLoc(), native.getResult());
        gather->moveBefore(branch.thenBlock(), branch.thenBlock()->end());
        auto value = gpu::loadFragmentSnapshot(gather, *storage);
        if (failed(value)) return failure();
        builder.setInsertionPointToEnd(branch.thenBlock());
        builder.create<scf::YieldOp>(gather.getLoc(), *value);
        replacement = branch.getResult(0);
      }
      gather.getResult().replaceAllUsesWith(replacement);
      gather.erase();
    }
  }
  gpu::eraseDeadPhysicalValues(kernel);
  return success();
}

} // namespace intent::triton::detail
