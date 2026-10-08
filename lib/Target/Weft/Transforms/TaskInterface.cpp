#include "TaskInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
namespace wk = ::weft::kernel;

namespace intent::weft_provider {
namespace {

void removeDeadValues(wk::KernelOp kernel) {
  SmallVector<Operation *> operations;
  kernel.walk([&](Operation *operation) {
    // Symbols and domains also bind type-level identities; ordinary SSA use
    // counts do not describe their liveness. Keep structured control intact.
    if (operation->getNumRegions() || isa<wk::SymbolOp>(operation) ||
        llvm::any_of(operation->getResultTypes(), [](Type type) { return isa<wk::DomainType>(type); })) return;
    operations.push_back(operation);
  });
  // Users follow their dominating leaf definitions, including values captured
  // by nested regions. Removing unused reads/casts exposes their slices here.
  for (Operation *operation : llvm::reverse(operations))
    if (isOpTriviallyDead(operation)) operation->erase();
}

} // namespace

LogicalResult finalizeTaskInterface(
    wk::KernelOp kernel, SmallVectorImpl<unsigned> &argumentPositions) {
  removeDeadValues(kernel);
  Block &body = kernel.getBody().front();
  SmallVector<Attribute> names, accesses;
  SmallVector<int64_t> aliases;
  SmallVector<unsigned> positions;
  for (auto [index, argument] : llvm::enumerate(body.getArguments())) {
    if (argument.use_empty()) continue;
    names.push_back(kernel.getArgNames()[index]);
    accesses.push_back(kernel.getArgAccess()[index]);
    aliases.push_back(kernel.getArgAliasSets()[index]);
    positions.push_back(argumentPositions[index]);
  }
  body.eraseArguments([](BlockArgument argument) { return argument.use_empty(); });
  Builder builder(kernel.getContext());
  kernel.setArgNamesAttr(builder.getArrayAttr(names));
  kernel.setArgAccessAttr(builder.getArrayAttr(accesses));
  kernel.setArgAliasSetsAttr(builder.getDenseI64ArrayAttr(aliases));
  argumentPositions.assign(positions.begin(), positions.end());
  return verify(kernel);
}

} // namespace intent::weft_provider
