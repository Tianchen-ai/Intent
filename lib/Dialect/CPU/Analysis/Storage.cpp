#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/IR/CollectiveHelpers.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
namespace intent::cpu {
namespace {

BufferStoragePolicy storagePolicy(func::FuncOp function) {
  BufferStoragePolicy policy;
  policy.isBorrowedArgument = isCollectiveArgument;
  policy.isOrderingBarrier = [](Operation *operation) {
    return isa<AtomicLoadOp, AtomicStoreOp, AtomicRMWOp,
               AtomicCompareExchangeOp>(operation);
  };
  policy.provenDisjointOrigins = [function](Value lhs, Value rhs) {
    auto first = dyn_cast<BlockArgument>(lhs);
    auto second = dyn_cast<BlockArgument>(rhs);
    auto entry = function;
    if (!first || !second || first.getOwner() != &entry.front() ||
        second.getOwner() != &entry.front()) return false;
    auto requirements = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
    auto interface = getPublicInterface(entry);
    if (!requirements || !requirements.getDisjointOutputs() || !interface) return false;
    auto source = getPublicView(interface, first.getArgNumber());
    auto target = getPublicView(interface, second.getArgNumber());
    return source && target && (source.getAccess() != 0 || target.getAccess() != 0);
  };
  return policy;
}

} // namespace

StorageAnalysis::StorageAnalysis(func::FuncOp function)
    : BufferStorageAnalysis(function, storagePolicy(function)), function(function) {}

std::optional<int64_t> constantDimensionUpperBound(Value memory, unsigned axis) {
  auto type = dyn_cast<MemRefType>(memory.getType());
  if (!type || axis >= static_cast<unsigned>(type.getRank())) return std::nullopt;
  return constantExtentUpperBound(ValueBoundsConstraintSet::Variable(memory, axis));
}

intent::ViewType StorageAnalysis::externalView(Value memory) const {
  auto argument = dyn_cast_or_null<BlockArgument>(uniqueOrigin(memory));
  auto currentFunction = function;
  if (!argument || argument.getOwner() != &currentFunction.front()) return {};
  auto interface = getPublicInterface(currentFunction);
  return interface ? getPublicView(interface, argument.getArgNumber())
                   : intent::ViewType{};
}

bool StorageAnalysis::isReadOnly(Value memory) const {
  if (auto formal = dyn_cast_or_null<BlockArgument>(memory);
      formal && isReadOnlyCollectiveArgument(formal)) return true;
  auto source = origins(memory);
  if (source.values.empty()) return false;
  return llvm::all_of(source.values, [&](Value value) {
    if (auto view = externalView(value)) return view.getAccess() == 0;
    auto argument = dyn_cast<BlockArgument>(value);
    return argument && isReadOnlyCollectiveArgument(argument);
  });
}

} // namespace intent::cpu
