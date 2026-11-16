#include "Gathers.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/Analysis/IndexRelations.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Block/Analysis/ValueSchema.h"
#include "Intent/Dialect/Block/Transforms/Storage/FragmentSnapshot.h"
#include "Intent/Dialect/Block/Transforms/Storage/Workspace.h"
#include "Intent/Dialect/Block/Transforms/Value/ValueMaterialization.h"
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

std::optional<GatherSourceCapacity> sourceCapacity(block::FragmentType source,
                                                  func::FuncOp kernel) {
  GatherSourceCapacity capacity;
  int64_t bytes = (source.getElementType().getIntOrFloatBitWidth() + 7) / 8;
  for (Attribute attribute : source.getShape()) {
    auto extent = cast<block::PhysicalExprAttr>(attribute);
    auto bounds = block::queryPositiveExtentBounds(extent, kernel);
    if (!bounds || capacity.maximumElements >
                       std::numeric_limits<int64_t>::max() / bounds->second)
      return std::nullopt;
    capacity.minimumElements *= bounds->first;
    capacity.maximumElements *= bounds->second;
    capacity.constant &= extent.getKind() == block::PhysicalExprKind::Constant;
  }
  if (capacity.maximumElements > std::numeric_limits<int64_t>::max() / bytes)
    return std::nullopt;
  return capacity;
}

scf::ForOp repeatedReadLoop(block::GatherOp gather, func::FuncOp kernel) {
  block::IndexRelations relations;
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
          return block::variesWithIteration(coordinate, loop, varying);
        }))
      return loop;
  }
  return {};
}

bool isRepeatedScalarRead(block::GatherOp gather, func::FuncOp kernel,
                          const GatherSourceCapacity &capacity,
                          block::PhysicalProgramAnalysis &analysis) {
  auto source = cast<block::FragmentType>(gather.getSource().getType());
  auto capabilities = kernel->getAttrOfType<block::CapabilitiesAttr>(block::capabilitiesAttr);
  if (isa<block::FragmentType>(gather.getResult().getType()) ||
      gather.getSourceAxes().size() != source.getShape().size() ||
      !llvm::all_of(gather.getCoordinates(), [](Value value) {
        return value.getType().isIndex();
      }) || capacity.maximumElements <= capabilities.getPreferredVectorElements())
    return false;
  for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
    if (analysis.axisRealization(gather.getSource(), axis).constructionScalarSeed)
      return false;
  return bool(repeatedReadLoop(gather, kernel));
}

bool endsSourceLifetimeBeforeReads(Value source,
                                  ArrayRef<block::GatherOp> readers,
                                  func::FuncOp kernel) {
  SmallVector<Operation *> loops;
  llvm::SmallPtrSet<Operation *, 8> selected;
  for (block::GatherOp reader : readers) {
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
            return isa<block::RecordType>(type);
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
      if (isa<block::CastOp, block::BitcastOp, block::BroadcastOp,
              block::ReshapeOp, block::TransposeOp, block::SelectOp>(user))
        llvm::append_range(values, user->getResults());
    }
  }
  return true;
}

Value largeSourceGuard(Value source, int64_t threshold) {
  auto type = cast<block::FragmentType>(source.getType());
  OpBuilder builder(source.getContext());
  if (Operation *definition = source.getDefiningOp())
    builder.setInsertionPointAfter(definition);
  else
    builder.setInsertionPointToStart(cast<BlockArgument>(source).getOwner());
  Location location = source.getLoc();
  Value elements = builder.create<arith::ConstantIndexOp>(location, 1);
  for (Attribute attribute : type.getShape()) {
    Value extent = builder.create<block::PhysicalExprOp>(
        location, builder.getIndexType(), cast<block::PhysicalExprAttr>(attribute));
    elements = builder.create<block::BinaryOp>(location, builder.getIndexType(),
                                            elements, extent, BinaryOperator::Multiply);
  }
  Value limit = builder.create<arith::ConstantIndexOp>(location, threshold);
  return builder.create<block::CompareOp>(location, builder.getI1Type(), elements,
                                        limit, ComparePredicate::Gt);
}

} // namespace

LogicalResult materializeGatherSources(func::FuncOp kernel) {
  auto capabilities = kernel->getAttrOfType<block::CapabilitiesAttr>(block::capabilitiesAttr);
  block::PhysicalProgramAnalysis analysis(kernel);
  llvm::MapVector<Value, SmallVector<block::GatherOp>> readers;
  kernel.walk([&](block::GatherOp gather) {
    if (!block::isProgramAllocationContext(gather, kernel))
      return;
    auto source = dyn_cast<block::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<block::FragmentType>(gather.getResult().getType());
    if (!source || gather.getCoordinates().empty() ||
        belongsToSplitGatherPair(gather) ||
        !isa<FloatType, IntegerType>(source.getElementType()))
      return;
    auto capacity = sourceCapacity(source, kernel);
    if (!capacity) return;
    int64_t bytes = (source.getElementType().getIntOrFloatBitWidth() + 7) / 8;
    bool oversized = capacity->constant && capacity->maximumElements * bytes >
                                               capabilities.getLocalStorageBudgetBytes();
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
        auto mapping = cast<block::AxisMapAttr>(source.getAxisMaps()[axis]);
        auto projection = block::queryFragmentAxis(result, block::sourceAxisIdentity(mapping));
        if (!projection.isExact() ||
            projection.dimensionId != mapping.getDimensionId() ||
            result.getShape()[projection.fragmentAxis] != source.getShape()[axis])
          return;
        auto ordinal = block::AxisMapAttr::get(kernel.getContext(),
            mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDimensionId(),
            0, mapping.getDerived());
        coordinateType = block::FragmentType::get(kernel.getContext(),
            IndexType::get(kernel.getContext()), ArrayAttr::get(kernel.getContext(),
                {source.getShape()[axis]}), ArrayAttr::get(kernel.getContext(), {ordinal}),
            source.getValidity(), source.getOwner());
      }
      if (auto coordinate = dyn_cast<block::FragmentType>(coordinateType)) {
        if (!result)
          return;
        auto projected = block::FragmentType::get(kernel.getContext(),
            coordinate.getElementType(), result.getShape(), result.getAxisMaps(),
            result.getValidity(), result.getOwner());
        if (!block::queryBroadcastProjection(coordinate, projected).isExact())
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
    auto payload = cast<block::FragmentType>(source.getType());
    auto capacity = sourceCapacity(payload, kernel);
    if (!capacity) return gathers.front().emitOpError("gather snapshot lost its finite source capacity");
    int64_t bytes = (payload.getElementType().getIntOrFloatBitWidth() + 7) / 8;
    bool oversized = capacity->constant && capacity->maximumElements * bytes >
                                               capabilities.getLocalStorageBudgetBytes();
    if (!oversized && !endsSourceLifetimeBeforeReads(source, gathers, kernel)) continue;
    // Large-source selection is a profitability choice, not a proof about the
    // native layout. Bind unknown coverage sizes with the actual constexpr
    // extent so inexpensive small-domain gathers retain their native path.
    Value sizeEnabled;
    if (!oversized && capacity->minimumElements <= capabilities.getPreferredVectorElements())
      sizeEnabled = largeSourceGuard(source, capabilities.getPreferredVectorElements());
    Value domainEnabled = block::materializeFragmentSnapshotReadDomain(source, gathers);
    Value enabled = sizeEnabled ? sizeEnabled : domainEnabled;
    if (sizeEnabled && domainEnabled) {
      Operation *after = sizeEnabled.getDefiningOp();
      if (Operation *domain = domainEnabled.getDefiningOp();
          domain && domain->getBlock() == after->getBlock() &&
          after->isBeforeInBlock(domain))
        after = domain;
      OpBuilder builder(after);
      builder.setInsertionPointAfter(after);
      enabled = builder.create<block::BinaryOp>(source.getLoc(), builder.getI1Type(),
          sizeEnabled, domainEnabled, BinaryOperator::LogicalAnd);
    }
    auto storage = block::materializeFragmentSnapshot(source, enabled);
    if (failed(storage)) return failure();
    for (block::GatherOp gather : gathers) {
      Value replacement;
      if (!sizeEnabled) {
        auto value = block::loadFragmentSnapshot(gather, *storage);
        if (failed(value)) return failure();
        replacement = *value;
      } else {
        OpBuilder builder(gather);
        auto branch = builder.create<scf::IfOp>(gather.getLoc(),
            TypeRange{gather.getResult().getType()}, sizeEnabled, true);
        builder.setInsertionPointToStart(branch.elseBlock());
        auto native = cast<block::GatherOp>(builder.clone(*gather));
        builder.create<scf::YieldOp>(gather.getLoc(), native.getResult());
        gather->moveBefore(branch.thenBlock(), branch.thenBlock()->end());
        auto value = block::loadFragmentSnapshot(gather, *storage);
        if (failed(value)) return failure();
        builder.setInsertionPointToEnd(branch.thenBlock());
        builder.create<scf::YieldOp>(gather.getLoc(), *value);
        replacement = branch.getResult(0);
      }
      gather.getResult().replaceAllUsesWith(replacement);
      gather.erase();
    }
  }
  block::eraseDeadPhysicalValues(kernel);
  return success();
}

} // namespace intent::triton::detail
