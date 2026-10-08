#include "Intent/Dialect/GPU/IR/AccessOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

#include "Intent/Dialect/GPU/IR/AccessOpInterface.cpp.inc"

namespace intent::gpu {

namespace {
bool sameSource(AxisMapAttr lhs, AxisMapAttr rhs) {
  return lhs.getSourceId() == rhs.getSourceId() &&
         lhs.getSourceAxis() == rhs.getSourceAxis() &&
         lhs.getDerived() == rhs.getDerived();
}

bool sameOccurrence(AxisMapAttr lhs, AxisMapAttr rhs) {
  return sameSource(lhs, rhs) && lhs.getDimensionId() == rhs.getDimensionId();
}

bool unitExtent(Attribute attribute) {
  auto extent = cast<PhysicalExprAttr>(attribute);
  return extent.getKind() == PhysicalExprKind::Constant && extent.getValue() == 1;
}
} // namespace

BroadcastProjection queryAccessCoordinateAxes(AccessOpInterface access,
                                              unsigned coordinateIndex) {
  BroadcastProjection result;
  auto coordinates = access.getAccessCoordinates();
  auto resourceAxes = access.getAccessSourceAxes();
  if (coordinateIndex >= coordinates.size() || resourceAxes.size() != coordinates.size())
    return result;
  llvm::SmallDenseSet<int64_t> seenResourceAxes;
  for (int64_t axis : resourceAxes)
    if (axis < 0 || !seenResourceAxes.insert(axis).second)
      return result;
  auto target = dyn_cast<FragmentType>(access.getAccessValueType());
  auto source = dyn_cast<FragmentType>(coordinates[coordinateIndex].getType());
  Type element = source ? source.getElementType() : coordinates[coordinateIndex].getType();
  if (!isa<IntegerType, IndexType>(element))
    return result;
  if (target)
    result.targetToSource.resize(target.getShape().size());
  if (!source) {
    result.state = BroadcastProjectionState::Exact;
    return result;
  }
  if (!target || source.getOwner() != target.getOwner())
    return result;

  if (source.getAxisMaps() == target.getAxisMaps())
    return queryAxisProjection(source, target);

  unsigned fragmentCount = 0, coordinateSlot = 0;
  bool cartesian = true;
  for (auto [index, coordinate] : llvm::enumerate(coordinates)) {
    auto type = dyn_cast<FragmentType>(coordinate.getType());
    if (!type)
      continue;
    if (index == coordinateIndex)
      coordinateSlot = fragmentCount;
    ++fragmentCount;
    cartesian &= type.getShape().size() == 1;
  }
  cartesian &= fragmentCount == target.getShape().size();
  if (cartesian && sameOccurrence(cast<AxisMapAttr>(source.getAxisMaps()[0]),
                                  cast<AxisMapAttr>(target.getAxisMaps()[coordinateSlot]))) {
    result.targetToSource[coordinateSlot] = 0;
    result.state = BroadcastProjectionState::Exact;
    return result;
  }

  // Reuse the standard physical broadcast relation when its matched axes are
  // supported by this access's identities. Its generic trailing-axis fallback
  // alone is not evidence that a coordinate varies over that payload axis.
  auto broadcast = queryAxisProjection(source, target);
  if (broadcast.isExact()) {
    bool supported = true;
    for (auto [targetAxis, sourceAxis] : llvm::enumerate(broadcast.targetToSource)) {
      if (!sourceAxis || unitExtent(source.getShape()[*sourceAxis]))
        continue;
      auto from = cast<AxisMapAttr>(source.getAxisMaps()[*sourceAxis]);
      auto to = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
      bool positional = source.getShape().size() == target.getShape().size() &&
                        targetAxis == *sourceAxis;
      bool related = sameSource(from, to) ||
                     (from.getDimensionId() > 0 && from.getDimensionId() == to.getDimensionId());
      auto selects = [&](AxisMapAttr candidate) {
        if (sameOccurrence(from, to))
          return sameOccurrence(from, candidate);
        if (sameSource(from, to))
          return sameSource(from, candidate);
        return from.getDimensionId() == candidate.getDimensionId();
      };
      unsigned targetOccurrences = llvm::count_if(target.getAxisMaps(), [&](Attribute axis) {
        return selects(cast<AxisMapAttr>(axis));
      });
      unsigned sourceOccurrences = llvm::count_if(source.getAxisMaps(), [&](Attribute axis) {
        return selects(cast<AxisMapAttr>(axis));
      });
      supported &= related && (positional || (targetOccurrences == 1 && sourceOccurrences == 1));
    }
    if (supported)
      return broadcast;
  }

  SmallVector<bool> used(source.getShape().size(), false);
  auto bindPhase = [&](auto eligible, auto matches) {
    for (auto [sourceAxis, attribute] : llvm::enumerate(source.getAxisMaps())) {
      auto mapping = cast<AxisMapAttr>(attribute);
      if (used[sourceAxis] || unitExtent(source.getShape()[sourceAxis]) ||
          !eligible(mapping))
        continue;
      std::optional<unsigned> selected;
      for (auto [targetAxis, attribute] : llvm::enumerate(target.getAxisMaps())) {
        if (result.targetToSource[targetAxis] ||
            !matches(mapping, cast<AxisMapAttr>(attribute)))
          continue;
        if (selected) {
          result.state = BroadcastProjectionState::Ambiguous;
          return false;
        }
        selected = targetAxis;
      }
      if (!selected)
        continue;
      auto targetMapping = cast<AxisMapAttr>(target.getAxisMaps()[*selected]);
      unsigned sources = llvm::count_if(llvm::enumerate(source.getAxisMaps()), [&](auto item) {
        auto candidate = cast<AxisMapAttr>(item.value());
        return !used[item.index()] && !unitExtent(source.getShape()[item.index()]) &&
               eligible(candidate) && matches(candidate, targetMapping);
      });
      if (sources != 1) {
        result.state = BroadcastProjectionState::Ambiguous;
        return false;
      }
      result.targetToSource[*selected] = sourceAxis;
      used[sourceAxis] = true;
    }
    return true;
  };
  // Resolve every exact occurrence before considering weaker source/dimension
  // evidence. A weaker match must not consume another source axis's exact slot.
  // Each phase also requires uniqueness in both directions, not first-use wins.
  auto allAxes = [](AxisMapAttr) { return true; };
  if (!bindPhase(allAxes, sameOccurrence) || !bindPhase(allAxes, sameSource))
    return result;
  auto uniqueDimension = [&](AxisMapAttr mapping) {
    return mapping.getDimensionId() > 0 &&
        llvm::count_if(llvm::enumerate(source.getAxisMaps()), [&](auto item) {
          return !unitExtent(source.getShape()[item.index()]) &&
                 cast<AxisMapAttr>(item.value()).getDimensionId() == mapping.getDimensionId();
        }) == 1;
  };
  if (!bindPhase(uniqueDimension, [](AxisMapAttr lhs, AxisMapAttr rhs) {
        return lhs.getDimensionId() == rhs.getDimensionId();
      }))
    return result;
  // An inserted singleton contributes no coordinate variation. Assign it only
  // after varying axes, without taking a target axis away from a real relation.
  for (unsigned sourceAxis = 0; sourceAxis < used.size(); ++sourceAxis) {
    if (!unitExtent(source.getShape()[sourceAxis]))
      continue;
    for (auto [targetAxis, bound] : llvm::enumerate(result.targetToSource)) {
      if (bound)
        continue;
      result.targetToSource[targetAxis] = sourceAxis;
      used[sourceAxis] = true;
      break;
    }
  }
  if (llvm::all_of(used, [](bool bound) { return bound; }))
    result.state = BroadcastProjectionState::Exact;
  return result;
}

BroadcastProjection queryAccessCoordinateProjection(AccessOpInterface access,
                                                    unsigned coordinateIndex) {
  auto result = queryAccessCoordinateAxes(access, coordinateIndex);
  if (!result.isExact())
    return result;
  auto source = dyn_cast<FragmentType>(access.getAccessCoordinates()[coordinateIndex].getType());
  if (!source)
    return result;
  auto target = cast<FragmentType>(access.getAccessValueType());
  for (auto [targetAxis, sourceAxis] : llvm::enumerate(result.targetToSource)) {
    if (!sourceAxis || unitExtent(source.getShape()[*sourceAxis]))
      continue;
    auto from = cast<AxisMapAttr>(source.getAxisMaps()[*sourceAxis]);
    auto to = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
    if (source.getShape()[*sourceAxis] != target.getShape()[targetAxis] ||
        (sameSource(from, to) && from.getDimensionId() != to.getDimensionId())) {
      result.state = BroadcastProjectionState::Unknown;
      return result;
    }
  }
  return result;
}

bool AccessOpInterface::writesMemory() {
  switch (getAccessKind()) {
  case AccessKind::Store:
  case AccessKind::ScatterReduce:
  case AccessKind::AtomicStore:
  case AccessKind::AtomicRMW:
  case AccessKind::AtomicCompareExchange: return true;
  case AccessKind::Load:
  case AccessKind::Gather:
  case AccessKind::AtomicLoad: return false;
  }
  llvm_unreachable("unknown physical access kind");
}

LogicalResult AccessOpInterface::updateAccessOperands(
    ValueRange coordinates, ValueRange payloads, Value valid, Value fill) {
  if (coordinates.size() != getAccessSourceAxes().size())
    return emitOpError("access update must preserve its explicit coordinate-axis mapping");
  if (payloads.size() != getAccessPayloads().size())
    return emitOpError("access update must preserve its payload arity");
  bool hasFillGroup = getAccessFillMutable().has_value();
  if ((hasFillGroup && bool(valid) != bool(fill)) || (!hasFillGroup && fill))
    return emitOpError("access update violates the operation's validity/fill contract");
  SmallVector<Value> copiedCoordinates(coordinates), copiedPayloads(payloads);
  getAccessCoordinatesMutable().assign(copiedCoordinates);
  getAccessPayloadsMutable().assign(copiedPayloads);
  getAccessValidityMutable().assign(valid ? ValueRange{valid} : ValueRange{});
  if (hasFillGroup)
    getAccessFillMutable()->assign(fill ? ValueRange{fill} : ValueRange{});
  return success();
}

} // namespace intent::gpu
