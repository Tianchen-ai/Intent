#include "Intent/Dialect/GPU/Transforms/Storage/FragmentSnapshot.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Workspace.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include <limits>
#include <numeric>

using namespace mlir;

namespace intent::gpu {
namespace {

Type storageElementType(Type type) {
  return type.isIndex() ? Type(IntegerType::get(type.getContext(), 64)) : type;
}

FragmentType ordinalType(PhysicalExprAttr extent, AxisMapAttr mapping,
                         uint64_t owner) {
  auto context = extent.getContext();
  return FragmentType::get(context, IndexType::get(context),
      ArrayAttr::get(context, {extent}), ArrayAttr::get(context, {AxisMapAttr::get(
          context, mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), 0, mapping.getDerived())}), 1, owner);
}

Value ordinal(OpBuilder &builder, Location location, PhysicalExprAttr extent,
              AxisMapAttr mapping, uint64_t owner) {
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value width = builder.create<PhysicalExprOp>(location, builder.getIndexType(), extent);
  return builder.create<MakeRangeOp>(location, ordinalType(extent, mapping, owner),
      zero, width, one, zero, width, mapping.getSourceId(), mapping.getSourceAxis(),
      mapping.getDerived());
}

Type withStorageElement(Type type, Type storage) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return FragmentType::get(type.getContext(), storage, fragment.getShape(),
        fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
  return storage;
}

} // namespace

Value materializeFragmentSnapshotReadDomain(Value source,
                                            ArrayRef<GatherOp> readers) {
  if (readers.empty() || !isa<FragmentType>(source.getType()) ||
      !llvm::all_of(readers, [&](GatherOp reader) {
        return reader.getSource() == source;
      }))
    return {};
  auto kernel = source.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel) return {};
  Operation *definition = source.getDefiningOp();
  Block *block = definition ? definition->getBlock()
                            : cast<BlockArgument>(source).getOwner();
  DominanceInfo dominance(kernel);
  for (Operation *parent = readers.front()->getParentOp();
       parent && parent != kernel; parent = parent->getParentOp()) {
    auto branch = dyn_cast<scf::IfOp>(parent);
    if (!branch || branch->getBlock() != block ||
        !dominance.dominates(source, parent))
      continue;
    auto readArm = [&](GatherOp reader) -> Region * {
      Operation *nested = reader;
      while (nested && nested->getParentOp() != parent)
        nested = nested->getParentOp();
      return nested ? nested->getParentRegion() : nullptr;
    };
    Region *arm = readArm(readers.front());
    if (!llvm::all_of(readers, [&](GatherOp reader) {
          return readArm(reader) == arm;
        }))
      continue;
    if (arm == &branch.getThenRegion()) return branch.getCondition();
    if (arm != &branch.getElseRegion()) continue;
    OpBuilder builder(branch);
    Value disabled = builder.create<arith::ConstantIntOp>(branch.getLoc(), 0, 1);
    return builder.create<CompareOp>(branch.getLoc(), builder.getI1Type(),
        branch.getCondition(), disabled, ComparePredicate::Eq);
  }
  return {};
}

FailureOr<Value> materializeFragmentSnapshot(Value source, Value enabled) {
  auto type = dyn_cast<FragmentType>(source.getType());
  auto kernel = source.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!type || !kernel) return failure();
  Type storageElement = storageElementType(type.getElementType());
  if (!isa<IntegerType, FloatType>(storageElement)) return failure();
  uint64_t elements = 1;
  for (Attribute attribute : type.getShape()) {
    auto bounds = queryPositiveExtentBounds(cast<PhysicalExprAttr>(attribute), kernel);
    if (!bounds || bounds->second <= 0 ||
        elements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                       static_cast<uint64_t>(bounds->second))
      return kernel.emitError("fragment snapshot requires finite representable physical extents");
    elements *= bounds->second;
  }
  unsigned bytes = (storageElement.getIntOrFloatBitWidth() + 7) / 8;
  if (elements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / bytes)
    return kernel.emitError("fragment snapshot byte capacity is not representable");
  Operation *definition = source.getDefiningOp();
  Block *block = definition ? definition->getBlock()
                            : cast<BlockArgument>(source).getOwner();
  if (!block || block->empty()) return failure();
  Operation *anchor = definition ? definition : &block->front();
  if (!isProgramAllocationContext(anchor, kernel))
    return kernel.emitError("fragment snapshot cannot escape an isolated helper");
  Operation *insertionAfter = definition;
  if (enabled) {
    if (!enabled.getType().isInteger(1) ||
        enabled.getParentRegion()->getParentOfType<func::FuncOp>() != kernel)
      return kernel.emitError("fragment snapshot enable must be a scalar i1 in the same program");
    if (Operation *predicate = enabled.getDefiningOp();
        predicate && predicate->getBlock() == block &&
        (!insertionAfter || insertionAfter->isBeforeInBlock(predicate)))
      insertionAfter = predicate;
    Operation *point = insertionAfter ? insertionAfter : &block->front();
    DominanceInfo dominance(kernel);
    // Results of point become available immediately after point. All other
    // values must already dominate it; do not move into another control epoch.
    auto available = [&](Value value) {
      return (insertionAfter && value.getDefiningOp() == insertionAfter) ||
             dominance.dominates(value, point);
    };
    if (!isProgramAllocationContext(point, kernel) ||
        !available(source) || !available(enabled))
      return kernel.emitError("fragment snapshot enable does not dominate its initialization");
  }

  OpBuilder builder(kernel.getContext());
  if (insertionAfter) builder.setInsertionPointAfter(insertionAfter);
  else builder.setInsertionPointToStart(block);
  Location location = source.getLoc();
  auto buffer = createProgramBuffer(builder, location, storageElement,
                                    type.getShape(), type.getOwner());
  auto [sourceId, dimension] = nextPhysicalAxisIdentities(kernel);
  SmallVector<Attribute> mappings, groups;
  SmallVector<Value> coordinates;
  SmallVector<int64_t> axes(type.getShape().size());
  std::iota(axes.begin(), axes.end(), 0);
  for (int64_t axis : axes) {
    auto mapping = AxisMapAttr::get(kernel.getContext(), sourceId, axis,
                                    dimension++, axis, true);
    mappings.push_back(mapping);
    groups.push_back(ReshapeGroupAttr::get(kernel.getContext(),
        builder.getDenseI64ArrayAttr({axis}), builder.getDenseI64ArrayAttr({axis})));
    coordinates.push_back(ordinal(builder, location,
        cast<PhysicalExprAttr>(type.getShape()[axis]), mapping, type.getOwner()));
  }
  // Position axes preserve every completed lane, including existing padding.
  // They grant no permission to load the producer's external source again.
  auto storedType = FragmentType::get(kernel.getContext(), type.getElementType(),
      type.getShape(), builder.getArrayAttr(mappings), type.getValidity(), type.getOwner());
  Value stored = builder.create<ReshapeOp>(location, storedType, source,
                                          builder.getArrayAttr(groups));
  if (storageElement != type.getElementType())
    stored = builder.create<CastOp>(location,
        withStorageElement(storedType, storageElement), stored);
  Value valid;
  if (enabled) {
    auto predicateType = FragmentType::get(kernel.getContext(), builder.getI1Type(),
        storedType.getShape(), storedType.getAxisMaps(), storedType.getValidity(),
        storedType.getOwner());
    valid = builder.create<SplatOp>(location, predicateType, enabled);
  }
  builder.create<StoreOp>(location, buffer, coordinates, stored, valid, axes);
  return buffer.getResult();
}

FailureOr<Value> loadFragmentSnapshot(GatherOp gather, Value snapshot) {
  auto kernel = gather->getParentOfType<func::FuncOp>();
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  auto storage = dyn_cast<BufferType>(snapshot.getType());
  if (!kernel || !source || !storage ||
      storage.getScope().getValue() != BufferScope::ProgramPrivate ||
      storage.getShape() != source.getShape() ||
      storage.getOwner() != source.getOwner() ||
      storage.getElementType() != storageElementType(source.getElementType()) ||
      !DominanceInfo(kernel).dominates(snapshot, gather.getOperation()))
    return gather.emitOpError("fragment snapshot does not dominate a matching physical gather");
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  SmallVector<Value> indices(source.getShape().size());
  for (auto [coordinate, axis] : llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (axis < 0 || static_cast<size_t>(axis) >= indices.size() || indices[axis])
      return failure();
    indices[axis] = coordinate;
  }
  // Check all retained axes against the actual gather result before creating
  // their ordinals. Scalar coordinates stay scalar; fragment coordinates may
  // broadcast only along the result's current axis/extent relation.
  for (unsigned axis = 0; axis < indices.size(); ++axis) {
    Type coordinateType = indices[axis] ? indices[axis].getType() : Type(ordinalType(
        cast<PhysicalExprAttr>(source.getShape()[axis]),
        cast<AxisMapAttr>(source.getAxisMaps()[axis]), source.getOwner()));
    if (auto fragment = dyn_cast<FragmentType>(coordinateType)) {
      if (!result) return failure();
      auto projected = FragmentType::get(kernel.getContext(), fragment.getElementType(),
          result.getShape(), result.getAxisMaps(), result.getValidity(), result.getOwner());
      if (!queryBroadcastProjection(fragment, projected).isExact())
        return gather.emitOpError("fragment snapshot index has no result-axis projection");
    }
  }
  OpBuilder builder(gather);
  SmallVector<int64_t> axes(indices.size());
  std::iota(axes.begin(), axes.end(), 0);
  for (int64_t axis : axes)
    if (!indices[axis])
      indices[axis] = ordinal(builder, gather.getLoc(),
          cast<PhysicalExprAttr>(source.getShape()[axis]),
          cast<AxisMapAttr>(source.getAxisMaps()[axis]), source.getOwner());
  Type loadedType = withStorageElement(gather.getResult().getType(), storage.getElementType());
  Value fill = gather.getFill();
  if (fill && storage.getElementType() != source.getElementType())
    fill = builder.create<CastOp>(gather.getLoc(),
        withStorageElement(fill.getType(), storage.getElementType()), fill);
  auto loaded = builder.create<LoadOp>(gather.getLoc(), loadedType,
      snapshot, indices, gather.getValid(), fill, axes);
  if (Attribute origin = gather->getAttr(originAttr)) loaded->setAttr(originAttr, origin);
  Value value = loaded;
  if (loadedType != gather.getResult().getType())
    value = builder.create<CastOp>(gather.getLoc(), gather.getResult().getType(), value);
  return value;
}

} // namespace intent::gpu
