#include "Intent/Transforms/Passes.h"

#include "Intent/Analysis/CanonicalKernel.h"
#include "Intent/Analysis/ProductSchema.h"
#include "Intent/Dialect/Intent/IR/IntentAttrs.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent {
namespace {

void collectCoordinateSources(Type type,
                              llvm::DenseMap<uint64_t, unsigned> &ranks) {
  if (auto domain = dyn_cast<DomainType>(type))
    ranks.try_emplace(domain.getOriginId(), domain.getRank());
  if (auto components = getProductComponents(type))
    for (Attribute attribute : components)
      collectCoordinateSources(cast<TypeAttr>(attribute).getValue(), ranks);
}

LogicalResult verifyCoordinateSources(
    Operation *owner, Type type,
    const llvm::DenseMap<uint64_t, unsigned> &ranks) {
  if (auto index = dyn_cast<LogicalIndexType>(type)) {
    auto source = ranks.find(index.getSourceId());
    if (source == ranks.end() || index.getAxis() >= source->second)
      return owner->emitOpError(
          "logical index references an unknown source identity or axis");
    return success();
  }
  if (auto region = dyn_cast<RegionType>(type)) {
    auto source = ranks.find(region.getSourceId());
    if (source == ranks.end() || region.getRank() > source->second)
      return owner->emitOpError(
          "subregion references an unknown source identity/rank");
    return success();
  }
  if (auto components = getProductComponents(type))
    for (Attribute attribute : components)
      if (failed(verifyCoordinateSources(
              owner, cast<TypeAttr>(attribute).getValue(), ranks)))
        return failure();
  return success();
}

LogicalResult verifyResultProvenance(Operation *operation,
                                     DenseSet<int64_t> &valueIDs) {
  auto nodes = operation->getAttrOfType<ArrayAttr>("intent.result_nodes");
  auto names = operation->getAttrOfType<ArrayAttr>("intent.result_names");
  if (!nodes || !names || nodes.size() != operation->getNumResults() ||
      names.size() != operation->getNumResults())
    return operation->emitOpError(
        "result provenance must align with every SSA result");
  for (auto [index, nodeAttribute] : llvm::enumerate(nodes)) {
    auto node = dyn_cast<IntegerAttr>(nodeAttribute);
    auto name = dyn_cast<StringAttr>(names[index]);
    if (!node || node.getInt() < 0 || !valueIDs.insert(node.getInt()).second ||
        !name || name.getValue().empty())
      return operation->emitOpError(
          "SSA result provenance IDs must be unique and names non-empty");
    Type type = operation->getResult(index).getType();
    if (auto domain = dyn_cast<DomainType>(type);
        domain && domain.getOriginId() != static_cast<uint64_t>(node.getInt()))
      return operation->emitOpError(
          "domain origin must equal its canonical SSA provenance ID");
    if (auto region = dyn_cast<RegionType>(type);
        region && region.getOriginId() != static_cast<uint64_t>(node.getInt()))
      return operation->emitOpError(
          "subregion origin must equal its canonical SSA provenance ID");
    if (auto buffer = dyn_cast<BufferType>(type);
        buffer && buffer.getOriginId() != static_cast<uint64_t>(node.getInt()))
      return operation->emitOpError(
          "buffer origin must equal its canonical SSA provenance ID");
    if (failed(verifyCanonicalType(operation, type)))
      return failure();
  }
  return success();
}

LogicalResult verifyRegionProvenance(Operation *operation,
                                     DenseSet<int64_t> &valueIDs) {
  auto nodes =
      operation->getAttrOfType<ArrayAttr>("intent.region_argument_nodes");
  auto names =
      operation->getAttrOfType<ArrayAttr>("intent.region_argument_names");
  if (operation->getNumRegions() == 0) {
    if (nodes || names)
      return operation->emitOpError(
          "region provenance is illegal on an operation without regions");
    return success();
  }
  if (!nodes || !names || nodes.size() != operation->getNumRegions() ||
      names.size() != operation->getNumRegions())
    return operation->emitOpError(
        "region provenance must align with every structured region");
  for (auto [regionIndex, region] : llvm::enumerate(operation->getRegions())) {
    auto regionNodes = dyn_cast<ArrayAttr>(nodes[regionIndex]);
    auto regionNames = dyn_cast<ArrayAttr>(names[regionIndex]);
    if (!regionNodes || !regionNames ||
        regionNodes.size() != region.getBlocks().size() ||
        regionNames.size() != region.getBlocks().size())
      return operation->emitOpError(
          "region provenance must align with every block");
    unsigned blockIndex = 0;
    for (Block &block : region) {
      auto blockNodes = dyn_cast<ArrayAttr>(regionNodes[blockIndex]);
      auto blockNames = dyn_cast<ArrayAttr>(regionNames[blockIndex]);
      if (!blockNodes || !blockNames ||
          blockNodes.size() != block.getNumArguments() ||
          blockNames.size() != block.getNumArguments())
        return operation->emitOpError(
            "region provenance must align with every block argument");
      for (auto [argumentIndex, nodeAttribute] : llvm::enumerate(blockNodes)) {
        auto node = dyn_cast<IntegerAttr>(nodeAttribute);
        auto name = dyn_cast<StringAttr>(blockNames[argumentIndex]);
        if (!node || node.getInt() < 0 ||
            !valueIDs.insert(node.getInt()).second || !name ||
            name.getValue().empty())
          return operation->emitOpError(
              "block argument provenance IDs must be unique and names non-empty");
        if (failed(verifyCanonicalType(operation,
                                       block.getArgument(argumentIndex).getType())))
          return failure();
      }
      ++blockIndex;
    }
  }
  return success();
}

LogicalResult verifyNoLegacyFacts(Operation *operation) {
  static constexpr StringLiteral forbidden[] = {
      "intent.result_types", "intent.result_shapes", "intent.effects",
      "intent.spec",         "intent.mode",          "intent.scope"};
  for (StringRef name : forbidden)
    if (operation->hasAttr(name))
      return operation->emitOpError()
             << "contains legacy duplicate semantic fact '" << name << "'";
  StringRef name = operation->getName().getStringRef();
  if (llvm::is_contained(
          {StringRef("intent.partition"), StringRef("intent.state_stream"),
           StringRef("intent.ragged"), StringRef("intent.members"),
           StringRef("intent.ragged_member"), StringRef("intent.ragged_outer"),
           StringRef("intent.random"), StringRef("intent.atomic_add"),
           StringRef("intent.atomic_cas")},
          name))
    return operation->emitOpError("is a removed legacy canonical KIR node");
  return success();
}

LogicalResult verifyFunction(func::FuncOp function, bool &sawKernel,
                             DenseSet<int64_t> &valueIDs,
                             DenseSet<int64_t> &operationIDs) {
  auto kind = function->getAttrOfType<FunctionKindAttr>("intent.kind");
  if (failed(verifySourceInterface(function))) return failure();
  if (!llvm::hasSingleElement(function.getBody()) ||
      function.getBody().front().empty() ||
      function.getBody().front().back().getName().getStringRef() !=
          "intent.return")
    return function.emitOpError(
        "requires one entry block ending in intent.return");
  if (kind.getKind() == 0) {
    if (sawKernel)
      return function.emitOpError("module contains more than one kernel entry");
    sawKernel = true;
  }
  for (BlockArgument argument : function.getArguments()) {
    if (!valueIDs.insert(getSourceParameter(argument).getOriginId()).second)
      return function.emitOpError(
          "parameter provenance IDs must be unique non-negative integers");
  }
  Operation &returnOp = function.getBody().front().back();
  if (!llvm::equal(returnOp.getOperandTypes(), function.getResultTypes()))
    return returnOp.emitOpError("return operands do not match function results");

  WalkResult walk = function.walk([&](Operation *operation) -> WalkResult {
    if (operation == function.getOperation())
      return WalkResult::advance();
    if (operation->getName().getDialectNamespace() != "intent" ||
        !operation->getRegisteredInfo()) {
      operation->emitOpError("is not a registered canonical Intent KIR operation");
      return WalkResult::interrupt();
    }
    auto node = operation->getAttrOfType<IntegerAttr>("intent.node");
    if (!node || node.getInt() < 0 ||
        !operationIDs.insert(node.getInt()).second) {
      operation->emitOpError(
          "requires a unique non-negative intent.node provenance ID");
      return WalkResult::interrupt();
    }
    if (failed(verifyNoLegacyFacts(operation)) ||
        failed(verifyResultProvenance(operation, valueIDs)) ||
        failed(verifyRegionProvenance(operation, valueIDs)))
      return WalkResult::interrupt();
    for (Value operand : operation->getOperands())
      if (failed(verifyCanonicalType(operation, operand.getType())))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return failure();

  llvm::DenseMap<uint64_t, unsigned> coordinateRanks;
  for (Type type : function.getArgumentTypes())
    collectCoordinateSources(type, coordinateRanks);
  function.walk([&](Operation *operation) {
    for (Type type : operation->getResultTypes())
      collectCoordinateSources(type, coordinateRanks);
  });
  LogicalResult coordinateResult = success();
  function.walk([&](Operation *operation) {
    if (failed(coordinateResult))
      return WalkResult::interrupt();
    for (Type type : operation->getOperandTypes())
      if (failed(verifyCoordinateSources(operation, type, coordinateRanks))) {
        coordinateResult = failure();
        return WalkResult::interrupt();
      }
    for (Type type : operation->getResultTypes())
      if (failed(verifyCoordinateSources(operation, type, coordinateRanks))) {
        coordinateResult = failure();
        return WalkResult::interrupt();
      }
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          if (failed(verifyCoordinateSources(operation, argument.getType(),
                                             coordinateRanks))) {
            coordinateResult = failure();
            return WalkResult::interrupt();
          }
    return WalkResult::advance();
  });
  return coordinateResult;
}

class VerifyKernelIRPass
    : public PassWrapper<VerifyKernelIRPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyKernelIRPass)

  StringRef getArgument() const final { return "verify-intent-kernel"; }
  StringRef getDescription() const final {
    return "Verify typed canonical Intent Kernel IR";
  }
  void runOnOperation() final {
    if (failed(verifyKernelModule(getOperation())))
      signalPassFailure();
  }
};

} // namespace

LogicalResult verifyKernelStructure(ModuleOp module) {
  if (failed(mlir::verify(module)))
    return failure();
  DenseSet<int64_t> operationIDs;
  DenseSet<int64_t> valueIDs;
  bool sawKernel = false;
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (failed(verifyFunction(function, sawKernel, valueIDs, operationIDs)))
      return failure();
  if (!sawKernel)
    return module.emitError("Intent module requires exactly one kernel entry");
  return success();
}

LogicalResult verifyKernelModule(ModuleOp module) {
  if (failed(verifyKernelStructure(module)))
    return failure();
  CanonicalKernelAnalysis analysis(module);
  return analysis.verify();
}

std::unique_ptr<Pass> createVerifyKernelIRPass() {
  return std::make_unique<VerifyKernelIRPass>();
}

void registerIntentPasses() {
  PassRegistration<VerifyKernelIRPass>();
  registerPass([] { return createNormalizeKernelIRPass(); });
}

} // namespace intent
