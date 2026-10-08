#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Analysis/Contractions.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>

using namespace mlir;

namespace intent::cpu {

std::optional<SmallVector<std::pair<unsigned, unsigned>>> unitReshapeAxes(Operation *operation) {
  if (!operation) return std::nullopt;
  bool expanding = isa<memref::ExpandShapeOp>(operation);
  if (!expanding && !isa<memref::CollapseShapeOp>(operation)) return std::nullopt;
  auto source = cast<MemRefType>(operation->getOperand(0).getType());
  auto result = cast<MemRefType>(operation->getResult(0).getType());
  auto groups = expanding ? cast<memref::ExpandShapeOp>(operation).getReassociationIndices()
                          : cast<memref::CollapseShapeOp>(operation).getReassociationIndices();
  auto wider = expanding ? result : source;
  auto narrower = expanding ? source : result;
  SmallVector<std::pair<unsigned, unsigned>> axes;
  for (auto [axis, group] : llvm::enumerate(groups)) {
    std::optional<unsigned> nonunit;
    for (int64_t member : group)
      if (wider.getDimSize(member) != 1) {
        if (nonunit) return std::nullopt;
        nonunit = member;
      }
    if (!nonunit) {
      if (narrower.getDimSize(axis) != 1) return std::nullopt;
      continue;
    }
    if (wider.getDimSize(*nonunit) != narrower.getDimSize(axis)) return std::nullopt;
    axes.emplace_back(expanding ? axis : *nonunit, expanding ? *nonunit : axis);
  }
  return axes;
}

bool isMatrixContraction(linalg::GenericOp operation) {
  if (operation.getInputs().size() != 2 || operation.getOutputs().size() != 1 ||
      operation.getNumResults()) return false;
  AffineExpr m, n, k;
  bindDims(operation.getContext(), m, n, k);
  SmallVector<AffineMap> maps = {
      AffineMap::get(3, 0, {m, k}, operation.getContext()),
      AffineMap::get(3, 0, {k, n}, operation.getContext()),
      AffineMap::get(3, 0, {m, n}, operation.getContext())};
  if (operation.getIndexingMapsArray() != maps ||
      operation.getIteratorTypesArray() != SmallVector<utils::IteratorType>{
          utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::reduction}) return false;
  return queryContractionAxes(operation).has_value();
}

SmallVector<AllocationFacts> PhysicalProgramAnalysis::allocations() {
  SmallVector<AllocationFacts> result;
  StorageAnalysis storage(function);
  function.walk([&](Operation *operation) {
    if (!isa<memref::AllocOp, memref::AllocaOp>(operation)) return;
    Value value = operation->getResult(0);
    auto type = cast<MemRefType>(value.getType());
    std::optional<int64_t> bytes;
    int64_t elementBytes = type.getElementType().isIndex() ? 8 : (type.getElementTypeBitWidth() + 7) / 8;
    if (type.hasStaticShape() && type.getNumElements() <= std::numeric_limits<int64_t>::max() / elementBytes)
      bytes = type.getNumElements() * elementBytes;
    Operation *writer = nullptr;
    auto aliases = storage.aliases(value);
    bool multiple = !aliases.complete;
    for (Operation *user : aliases.users) {
      auto effects = storage.effects(user);
      multiple |= !effects.complete || effects.ordered;
      for (const StorageEffect &entry : effects.entries) {
        if (!isa<MemoryEffects::Write>(entry.effect.getEffect())) continue;
        Value affected = entry.effect.getValue();
        if (affected && storage.disjoint(value, affected)) continue;
        if (!affected || storage.uniqueOrigin(affected) != value) {
          multiple = true;
          continue;
        }
        if (writer && writer != entry.operation) multiple = true;
        writer = entry.operation;
      }
    }
    result.push_back({value, operation->getParentOp(), multiple ? nullptr : writer,
                      bytes, isa<memref::AllocaOp>(operation)});
  });
  return result;
}

LogicalResult PhysicalProgramAnalysis::verify(CPUProgramStage stage) {
  StorageAnalysis storage(function);
  auto interface = getPublicInterface(function);
  auto requirements = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
  if (failed(verifyPublicInterface(function, interface))) return failure();
  if (interface.getArguments().empty() || interface.getArguments().size() != function.getNumArguments() ||
      !requirements || !requirements.getDisjointOutputs())
    return function.emitError("CPU function requires its complete typed native interface");
  for (auto [argument, field] : llvm::zip(function.getArguments(), interface.getArguments())) {
    auto storageType = [&](Type logical) -> Type {
      if (auto integer = dyn_cast<IntegerType>(logical)) return IntegerType::get(function.getContext(), integer.getWidth());
      return logical;
    };
    Type logical = cast<PublicParameterAttr>(field).getType();
    if (auto view = dyn_cast<intent::ViewType>(logical)) {
      auto tensor = publicViewTensor(view);
      Type element = tensor.getElementType();
      if (!element.isF16() && !element.isBF16() && !element.isF32() && !element.isF64() &&
          !isa<Float8E4M3FNType, Float8E5M2Type>(element) && !isa<IntegerType>(element))
        return function.emitError("CPU view has unsupported storage element type");
      auto type = dyn_cast<MemRefType>(argument.getType());
      if (!type || type.getElementType() != storageType(element) || type.getShape() != tensor.getShape())
        return function.emitError("CPU view type disagrees with its physical ABI");
      SmallVector<int64_t> strides;
      int64_t offset;
      if (failed(type.getStridesAndOffset(strides, offset)) || offset != 0 ||
          ((requirements.getContiguousViews() || view.getAccess() != 0) && !type.getLayout().isIdentity()))
        return function.emitError("CPU view layout disagrees with its physical ABI");
      if (view.getConstraints().getHasStrides())
        for (auto [constraint, stride] : llvm::zip(view.getConstraints().getStrides(), strides)) {
          if (!isa<UnitAttr, IntegerAttr>(constraint))
            return function.emitError("CPU stride constraint must be an integer or unconstrained");
          if (auto fixed = dyn_cast<IntegerAttr>(constraint);
              fixed && !ShapedType::isDynamic(stride) && fixed.getInt() != stride)
            return function.emitError("CPU view layout contradicts its declared stride constraint");
        }
    } else {
      if (!logical.isF32() && !logical.isF64() && !logical.isIndex() && !isa<IntegerType>(logical))
        return function.emitError("CPU scalar requires a supported C ABI numeric type");
      if (storageType(logical) != argument.getType())
        return function.emitError("CPU scalar type disagrees with its physical ABI");
    }
  }
  auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>(
      "intent_cpu.capabilities");
  for (auto facts : allocations()) {
    if (facts.stack && capabilities && (!facts.bytes || *facts.bytes > capabilities.getPrivateBytes()))
      return facts.value.getDefiningOp()->emitError("CPU stack allocation exceeds its declared budget");
  }
  if (stage != CPUProgramStage::Values && failed(verifyStorageOwnership(function)))
    return failure();
  bool invalid = false;
  for (const StorageEffect &entry : storage.effects(function).entries) {
    if (!isa<MemoryEffects::Write>(entry.effect.getEffect())) continue;
    for (Value origin : storage.origins(entry.effect.getValue()).values) {
      auto view = storage.externalView(origin);
      if (view && view.getAccess() == 0) {
        entry.operation->emitError("CPU write contradicts its input-only ABI");
        invalid = true;
        break;
      }
    }
  }
  function.walk([&](Operation *operation) {
    if (stage != CPUProgramStage::Values) {
      auto tensor = [](Type type) { return isa<TensorType>(type); };
      bool hasTensor = llvm::any_of(operation->getOperandTypes(), tensor) ||
                       llvm::any_of(operation->getResultTypes(), tensor);
      for (Region &region : operation->getRegions())
        for (Block &block : region)
          hasTensor |= llvm::any_of(block.getArgumentTypes(), tensor);
      if (hasTensor) {
        operation->emitError("CPU buffer program still contains an unmaterialized tensor value");
        invalid = true;
      }
    }
    if (stage == CPUProgramStage::Realized && (isa<RegionFoldOp, RegionScanOp, ReduceOp, SliceReduceOp, ScanOp, HistogramOp, QuantizeOp, QuantizedDotOp>(operation) || operation->getName().getDialectNamespace() == "linalg")) {
      operation->emitError("CPU structured operation has not been materialized for the provider");
      invalid = true;
    }
  });
  return failure(invalid);
}

bool supportsVectorScan(ScanOp operation) {
  if (operation.isDestinationPassing()) return false;
  auto type = cast<MemRefType>(operation.getSources()[0].getType());
  if (operation.getAxis() + 1 != static_cast<uint64_t>(type.getRank())) return false;
  auto contiguous = [](Value memory) {
    auto type = cast<MemRefType>(memory.getType());
    SmallVector<int64_t> strides;
    int64_t offset;
    return !type.getElementType().isInteger(1) && succeeded(type.getStridesAndOffset(strides, offset)) &&
        (strides.back() == 1 || ShapedType::isDynamic(strides.back()));
  };
  return llvm::all_of(operation.getSources(), contiguous) && llvm::all_of(operation.getOutputs(), contiguous) &&
      llvm::all_of(operation.getCombine().front().without_terminator(), [](Operation &instruction) {
        return isa<arith::ConstantOp>(instruction) ||
            (instruction.hasTrait<OpTrait::Elementwise>() && instruction.getNumResults() == 1);
      });
}

LogicalResult verifyCPUProgram(ModuleOp module, CPUProgramStage stage) {
  if (failed(mlir::verify(module))) return failure();
  bool invalid = false;
  module.walk([&](Operation *operation) {
    llvm::StringRef dialect = operation->getName().getDialectNamespace();
    if (dialect != "builtin" && dialect != "func" && dialect != "arith" &&
        dialect != "math" && dialect != "memref" && dialect != "scf" &&
        dialect != "vector" && dialect != "linalg" && dialect != "intent_cpu" &&
        !(stage != CPUProgramStage::Realized && dialect == "bufferization") &&
        !(stage == CPUProgramStage::Values && dialect == "tensor")) {
      operation->emitError("operation is outside the current CPU execution family");
      invalid = true;
    }
  });
  func::FuncOp first;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.isExternal() && function->hasAttr("cpu.external_runtime")) continue;
    if (failed(PhysicalProgramAnalysis(function).verify(stage))) invalid = true;
    if (!first) first = function;
    else if (getPublicInterface(function) != getPublicInterface(first) ||
             function.getFunctionType() != first.getFunctionType() ||
             function->getAttr(entryRequirementsAttr) != first->getAttr(entryRequirementsAttr)) {
      function.emitError("CPU candidates must share one public interface and physical entry ABI");
      invalid = true;
    }
  }
  return failure(invalid);
}

}
