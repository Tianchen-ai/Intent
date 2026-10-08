#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/MemoryEffects.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"
#include "Verification.h"

using namespace mlir;
namespace intent::gpu {
using namespace operation_detail;

namespace {

unsigned rankOf(Type type) {
  if (auto view = dyn_cast<ViewType>(type))
    return view.getRank();
  if (auto fragment = dyn_cast<FragmentType>(type))
    return fragment.getShape().size();
  if (auto buffer = dyn_cast<BufferType>(type))
    return buffer.getShape().size();
  return 0;
}

Type resourceElementType(Type type) {
  if (auto view = dyn_cast<ViewType>(type))
    return view.getElementType();
  if (auto buffer = dyn_cast<BufferType>(type))
    return buffer.getElementType();
  return {};
}

LogicalResult verifyWritableResource(Operation *owner, Type resource) {
  auto view = dyn_cast<ViewType>(resource);
  return !view || view.getAccess() != 0
             ? success()
             : owner->emitOpError("In-only external view cannot be written");
}

LogicalResult verifyResourceSharing(Operation *owner, Type resource,
                                    AtomicSharingDomain sharing) {
  AtomicSharingDomain expected;
  if (auto buffer = dyn_cast<BufferType>(resource))
    expected = buffer.getScope().getValue() == BufferScope::InvocationWorkspace
                   ? AtomicSharingDomain::KernelInvocation
                   : AtomicSharingDomain::ProgramInstance;
  else if (isa<ViewType>(resource))
    expected = AtomicSharingDomain::KernelInvocation;
  else
    return owner->emitOpError("sharing requires a physical resource");
  return sharing == expected
             ? success()
             : owner->emitOpError(
                   "sharing domain disagrees with the physical resource scope");
}

LogicalResult verifyAtomicSemantics(AccessOpInterface access,
                                    AtomicOrdering ordering,
                                    AtomicSharingDomain sharing) {
  Operation *owner = access;
  Type resource = access.getAccessResource().getType();
  bool orderingLegal =
      (access.getAccessKind() == AccessKind::AtomicLoad &&
       (ordering == AtomicOrdering::Relaxed ||
        ordering == AtomicOrdering::Acquire)) ||
      (access.getAccessKind() == AccessKind::AtomicStore &&
       (ordering == AtomicOrdering::Relaxed ||
        ordering == AtomicOrdering::Release)) ||
      access.getAccessKind() == AccessKind::AtomicRMW ||
      access.getAccessKind() == AccessKind::AtomicCompareExchange;
  if (!orderingLegal)
    return owner->emitOpError(
        "atomic ordering is illegal for this physical operation");
  if (failed(verifyResourceSharing(owner, resource, sharing)))
    return failure();
  if (auto view = dyn_cast<ViewType>(resource); view && view.getAccess() != 2)
    return owner->emitOpError(
        "external atomic target must use InOut access semantics");
  return success();
}

} // namespace

LogicalResult verifyAccessSchema(AccessOpInterface access) {
  Type resource = access.getAccessResource().getType();
  Type payload = access.getAccessValueType();
  Value valid = access.getAccessValidity(), fill = access.getAccessFill();
  unsigned coordinateCount = access.getAccessCoordinates().size();
  bool gather = access.getAccessKind() == AccessKind::Gather;
  if ((gather && (!isa<FragmentType>(resource) || !coordinateCount)) ||
      (!gather && coordinateCount != rankOf(resource)) ||
      access.getAccessSourceAxes().size() != coordinateCount)
    return access.emitOpError("access coordinate partition/rank is inconsistent");
  if (access.getAccessFillMutable() && bool(valid) != bool(fill))
    return access.emitOpError("read requires validity and fill together");
  if (valid && (!elementType(valid.getType()).isInteger(1) ||
                !sameShape(valid.getType(), payload)))
    return access.emitOpError("access validity must match its value schema")
           << "; value=" << payload << "; valid=" << valid.getType();
  if (fill && !sameShape(fill.getType(), payload))
    return access.emitOpError("read fill must match its value schema")
           << "; value=" << payload << "; fill=" << fill.getType();
  llvm::DenseSet<int64_t> axes;
  for (int64_t axis : access.getAccessSourceAxes())
    if (axis < 0 || axis >= static_cast<int64_t>(rankOf(resource)) ||
        !axes.insert(axis).second)
      return access.emitOpError(gather ? "gather source axes must be a unique subset"
                                       : "memory source axes must be a bijection");
  Type resourceElement = gather ? cast<FragmentType>(resource).getElementType()
                                : resourceElementType(resource);
  if (resourceElement != elementType(payload))
    return access.emitOpError("access resource/value element types disagree");
  for (auto [index, coordinate] : llvm::enumerate(access.getAccessCoordinates())) {
    auto projection = queryAccessCoordinateProjection(access, index);
    if (!projection.isExact())
      return access.emitOpError(
          "access coordinate has no exact projection into its payload lanes")
             << "; coordinate_slot=" << index
             << "; resource_axis=" << access.getAccessSourceAxes()[index]
             << "; coordinate=" << coordinate.getType()
             << "; payload=" << payload;
  }
  return success();
}

LogicalResult AssumeInBoundsOp::verify() {
  if (!isa<IntegerType, IndexType>(elementType(getIndex().getType())))
    return emitOpError("in-bounds assumption requires an integer/index value");
  return getAxis() < rankOf(getResource().getType())
             ? success()
             : emitOpError("assumed source axis is outside the resource rank");
}

void LoadOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getResourceMutable());
}

void AssumeInBoundsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), AssumptionResource::get());
}

LogicalResult StoreOp::verify() {
  return verifyWritableResource(getOperation(), getResource().getType());
}

void StoreOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getResourceMutable());
}

LogicalResult ScatterReduceOp::verify() {
  if (failed(verifyWritableResource(getOperation(), getResource().getType())) ||
      failed(verifyResourceSharing(getOperation(), getResource().getType(),
                                   getSharing())))
    return failure();
  SmallVector<Type> arguments{getValue().getType(), getValue().getType()};
  SmallVector<Type> results{getValue().getType()};
  return verifyHelperRegion(getOperation(), getCombine(), arguments, results);
}

void ScatterReduceOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getResourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getResourceMutable());
}

LogicalResult AtomicLoadOp::verify() {
  return verifyAtomicSemantics(cast<AccessOpInterface>(getOperation()),
                               getOrdering(), getSharing());
}

LogicalResult AtomicStoreOp::verify() {
  return verifyAtomicSemantics(cast<AccessOpInterface>(getOperation()),
                               getOrdering(), getSharing());
}

LogicalResult AtomicRMWOp::verify() {
  if (getResult().getType() != getValue().getType())
    return emitOpError("atomic RMW physical value/kind schema is invalid")
           << "; resource element="
           << resourceElementType(getResource().getType())
           << ", value=" << getValue().getType()
           << ", result=" << getResult().getType()
           << ", kind=" << stringifyAtomicRMWKind(getKind());
  return verifyAtomicSemantics(cast<AccessOpInterface>(getOperation()),
                               getOrdering(), getSharing());
}

LogicalResult AtomicCompareExchangeOp::verify() {
  auto result = getResult().getType();
  if (getExpected().getType() != getDesired().getType() ||
      result.getFieldTypes().size() != 2 ||
      cast<TypeAttr>(result.getFieldTypes()[0]).getValue() !=
          getExpected().getType() ||
      !elementType(cast<TypeAttr>(result.getFieldTypes()[1]).getValue())
           .isInteger(1))
    return emitOpError("compare-exchange physical result schema is invalid");
  return verifyAtomicSemantics(cast<AccessOpInterface>(getOperation()),
                               getOrdering(), getSharing());
}

LogicalResult BufferOp::verify() {
  auto type = getResult().getType();
  if (type.isInvocationWorkspace()) {
    auto kernel = getOperation()->getParentOfType<func::FuncOp>();
    if (!kernel || getOperation()->getBlock() != &kernel.front())
      return emitOpError("invocation allocation must belong to the kernel entry block");
    if (type.getInitialization().getValue() != BufferInitialization::FirstWrite)
      return emitOpError("invocation allocation requires explicit first writes");
  }
  if ((type.getInitialization().getValue() ==
       BufferInitialization::FullValue) !=
      static_cast<bool>(getInitialValue()))
    return emitOpError(
        "buffer initializer disagrees with its initialization obligation");
  if (getInitialValue() &&
      elementType(getInitialValue().getType()) != getResult().getType().getElementType())
    return emitOpError("buffer initializer element type disagrees");
  if (Value initialValue = getInitialValue())
    if (auto initial = dyn_cast<FragmentType>(initialValue.getType());
        initial && (initial.getShape() != type.getShape() ||
                    initial.getOwner() != type.getOwner()))
      return emitOpError(
          "buffer full-value initializer must cover its physical shape and owner")
          << "; initializer=" << initial << "; buffer=" << type;
  return success();
}

void BufferOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       cast<OpResult>(getResult()));
  if (getInitialValue())
    effects.emplace_back(MemoryEffects::Write::get(), cast<OpResult>(getResult()));
}

} // namespace intent::gpu
