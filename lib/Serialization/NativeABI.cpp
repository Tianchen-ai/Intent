#include "Intent/Serialization/NativeABI.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace intent {

std::string NativeSlot::name() const {
  std::string result = "a" + std::to_string(parameter);
  if (axis) result += (role == NativeSlotRole::Extent ? "_d" : "_s") + std::to_string(*axis);
  return result;
}

llvm::json::Array NativeABI::serialize() const {
  llvm::json::Array result;
  for (const NativeSlot &slot : slots) {
    StringRef role;
    switch (slot.role) {
    case NativeSlotRole::Pointer: role = "pointer"; break;
    case NativeSlotRole::Extent: role = "extent"; break;
    case NativeSlotRole::Stride: role = "stride"; break;
    case NativeSlotRole::Scalar: role = "scalar"; break;
    }
    llvm::json::Object entry{{"role", role}, {"parameter", slot.parameter},
        {"carrier", slot.role == NativeSlotRole::Pointer ? "ptr" : scalarABIName(slot.carrier)}};
    if (slot.axis) entry["axis"] = *slot.axis;
    if (slot.element) entry["element"] = scalarABIName(slot.element);
    result.push_back(std::move(entry));
  }
  return result;
}

llvm::json::Object NativeEntryRequirements::serialize() const {
  llvm::json::Array viewEntries, pairs;
  for (const auto &[parameter, requirements] : views)
    viewEntries.push_back(llvm::json::Object{
        {"parameter", parameter},
        {"layout", requirements.layout == NativeViewLayout::Contiguous ? "contiguous" : "strided"},
        {"alignment", requirements.alignment}});
  for (auto [left, right] : disjoint)
    pairs.push_back(llvm::json::Object{{"left", left}, {"right", right}});
  return llvm::json::Object{{"views", std::move(viewEntries)}, {"disjoint", std::move(pairs)}};
}

FailureOr<NativeEntryRequirements> queryNativeEntryRequirements(
    func::FuncOp function, InterfaceAttr interface, bool disjointWritableViews,
    function_ref<FailureOr<NativeViewRequirements>(unsigned, ViewType)> viewRequirements) {
  if (failed(verifyPublicInterface(function, interface))) return failure();
  if (function.getNumArguments() != interface.getArguments().size())
    return function.emitError("native entry requirements disagree with the public parameters"), failure();
  NativeEntryRequirements result;
  for (unsigned ordinal = 0; ordinal < function.getNumArguments(); ++ordinal) {
    auto view = getPublicView(interface, ordinal);
    if (!view) continue;
    auto requirements = viewRequirements(ordinal, view);
    if (failed(requirements)) return failure();
    if (requirements->alignment <= 0)
      return function.emitError("native view alignment must be positive"), failure();
    if (disjointWritableViews)
      for (const auto &[previous, unused] : result.views) {
        if (view.getAccess() != 0 || getPublicView(interface, previous).getAccess() != 0)
          result.disjoint.emplace_back(previous, ordinal);
      }
    result.views.emplace_back(ordinal, *requirements);
  }
  return result;
}

FailureOr<NativeABI> queryNativeABI(
    func::FuncOp function, InterfaceAttr interface,
    function_ref<FailureOr<Type>(Type)> scalarCarrier) {
  if (failed(verifyPublicInterface(function, interface))) return failure();
  if (function.getNumArguments() != interface.getArguments().size())
    return function.emitError("native entry arguments disagree with the public interface"), failure();
  NativeABI result;
  Type indexCarrier = IntegerType::get(function.getContext(), 64);
  for (auto [ordinal, argument] : llvm::enumerate(function.getArguments())) {
    auto view = getPublicView(interface, ordinal);
    auto memory = dyn_cast<MemRefType>(argument.getType());
    if (static_cast<bool>(view) != static_cast<bool>(memory))
      return function.emitError("native entry storage kind disagrees with its public parameter"), failure();
    if (memory) {
      result.slots.push_back({NativeSlotRole::Pointer, static_cast<unsigned>(ordinal),
          std::nullopt, {}, memory.getElementType()});
      for (auto role : {NativeSlotRole::Extent, NativeSlotRole::Stride})
        for (unsigned axis = 0; axis < memory.getRank(); ++axis)
          result.slots.push_back({role, static_cast<unsigned>(ordinal), axis, indexCarrier, {}});
      continue;
    }
    auto carrier = scalarCarrier(argument.getType());
    if (failed(carrier))
      return function.emitError("native entry has no scalar carrier for ") << argument.getType(), failure();
    auto integer = dyn_cast<IntegerType>(*carrier);
    bool supportedInteger = integer && integer.isSignless() &&
        llvm::is_contained(ArrayRef<unsigned>{1, 8, 16, 32, 64}, integer.getWidth());
    if (!supportedInteger && !carrier->isF32() && !carrier->isF64())
      return function.emitError("unsupported native scalar carrier ") << *carrier, failure();
    result.slots.push_back({NativeSlotRole::Scalar, static_cast<unsigned>(ordinal), std::nullopt, *carrier, {}});
  }
  return result;
}

} // namespace intent
