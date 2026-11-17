#include "Intent/Target/Weft/Serialization/HostSource.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Target/Weft/Serialization/HostScalar.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace intent::weft_provider {
namespace {

LogicalResult verifyHostType(Operation *owner, Type type) {
  if (auto memory = dyn_cast<MemRefType>(type)) {
    if (failed(verifyNativeMemoryType(owner, memory))) return failure();
    if (Attribute space = memory.getMemorySpace()) {
      auto integer = dyn_cast<IntegerAttr>(space);
      if (!integer || integer.getInt() != 0)
        return owner->emitOpError("requires native host memory space");
    }
    if (!hostStorageType(memory.getElementType()))
      return owner->emitOpError("element has no native host C storage representation: ")
             << memory.getElementType();
    return success();
  }
  if (!hostScalarType(type))
    return owner->emitOpError("type has no native host C representation: ")
           << type;
  return success();
}

LogicalResult verifyHostTypes(Operation *operation) {
  for (Type type : llvm::concat<Type>(operation->getOperandTypes(),
                                      operation->getResultTypes()))
    if (failed(verifyHostType(operation, type))) return failure();
  for (Region &region : operation->getRegions())
    for (Block &block : region)
      for (Type type : block.getArgumentTypes())
        if (failed(verifyHostType(operation, type))) return failure();
  return success();
}

LogicalResult verifyInvocation(cpu::InvokeOp invocation) {
  auto declaration = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      invocation, invocation.getCalleeAttr());
  if (!declaration || !declaration.isExternal() ||
      declaration.getNumResults() != 0)
    return invocation.emitOpError(
        "native host invocation requires a void external task declaration");
  return success();
}

} // namespace

HostSourceEmitter::HostSourceEmitter(ModuleOp module, llvm::raw_ostream &output)
    : NativeSourceEmitter(output, NativeSourceSyntax::C), module(module) {}

std::string HostSourceEmitter::nativeType(Type type) {
  if (auto memory = dyn_cast<MemRefType>(type))
    return hostStorageType(memory.getElementType())->name + " *";
  return hostScalarType(type)->name;
}

std::string HostSourceEmitter::offsetPointer(StringRef base, StringRef offset) {
  if (offset == "0") return base.str();
  return "(" + base.str() + " + (" + offset.str() + "))";
}

std::string HostSourceEmitter::pointerAsIndex(StringRef base) {
  return "(int64_t)(uintptr_t)(" + base.str() + ")";
}

const OperationEmitters<HostSourceEmitter> &HostSourceEmitter::emitters() {
  static const auto table = [] {
    OperationEmitters<HostSourceEmitter> result;
    auto allocate = [](auto operation, HostSourceEmitter &emitter) {
      return emitter.emitAllocation(operation);
    };
    auto checkAllocation = [](auto operation) {
      return verifyNativeAllocation(operation, /*requireStaticShape=*/false);
    };
    result.add<memref::AllocOp>(checkAllocation, allocate);
    result.add<memref::AllocaOp>(checkAllocation, allocate);
    result.add<memref::DeallocOp>(
        [](memref::DeallocOp) { return success(); },
        [](memref::DeallocOp operation, HostSourceEmitter &emitter) {
          emitter.line("free(" + emitter.memories.at(operation.getMemref()).base +
                       ");");
          return success();
        });
    result.add<memref::LoadOp>(
        [](memref::LoadOp) { return success(); },
        [](memref::LoadOp operation, HostSourceEmitter &emitter) {
          emitter.bindExpression(
              operation.getResult(),
              "*(" + emitter.memoryPointer(operation.getMemref(),
                                           operation.getIndices()) + ")");
          return success();
        });
    result.add<memref::StoreOp>(
        [](memref::StoreOp) { return success(); },
        [](memref::StoreOp operation, HostSourceEmitter &emitter) {
          emitter.line("*(" + emitter.memoryPointer(operation.getMemref(),
                                                    operation.getIndices()) +
                       ") = " + emitter.valueString(operation.getValue()) + ";");
          return success();
        });
    result.add<cpu::InvokeOp>(
        verifyInvocation,
        [](cpu::InvokeOp operation, HostSourceEmitter &emitter) {
          std::string text = operation.getCallee().str() + "(intent_context";
          for (auto [index, argument] : llvm::enumerate(operation.getOperands())) {
            text += ", ";
            text += isa<MemRefType>(argument.getType())
                        ? emitter.memoryPointer(argument)
                        : emitter.valueString(argument);
          }
          emitter.line(text + ");");
          return success();
        });
    result.add<cpu::TaskDispatchOp>(
        [](cpu::TaskDispatchOp operation) { return operation.verify(); },
        [](cpu::TaskDispatchOp operation, HostSourceEmitter &emitter) {
          return emitter.emitTaskDispatch(operation);
        });
    result.add<cpu::TaskYieldOp>(
        [](cpu::TaskYieldOp operation) { return operation.verify(); },
        [](cpu::TaskYieldOp, HostSourceEmitter &) { return success(); });
    result.add<scf::YieldOp>(
        [](scf::YieldOp) { return success(); },
        [](scf::YieldOp, HostSourceEmitter &) { return success(); });
    result.add<scf::ConditionOp>(
        [](scf::ConditionOp) { return success(); },
        [](scf::ConditionOp, HostSourceEmitter &) { return success(); });
    result.add<func::ReturnOp>(
        [](func::ReturnOp operation) -> LogicalResult {
          if (operation.getNumOperands())
            return operation.emitOpError("native host entry must return void");
          return success();
        },
        [](func::ReturnOp, HostSourceEmitter &emitter) {
          emitter.line("return;");
          return success();
        });
    return result;
  }();
  return table;
}

LogicalResult HostSourceEmitter::verifyOperation(Operation *operation) {
  if (mlir::failed(verifyHostTypes(operation))) return failure();
  if (isStandardScalarOperation(operation)) return verifyHostScalar(operation);
  if (metadataEmitters().contains(operation))
    return metadataEmitters().verify(operation);
  if (controlEmitters().contains(operation))
    return controlEmitters().verify(operation);
  return emitters().verify(operation);
}

LogicalResult HostSourceEmitter::emitNativeOperation(Operation *operation) {
  if (mlir::failed(verifyOperation(operation))) return failure();
  if (isStandardScalarOperation(operation)) {
    SmallVector<std::string> operands;
    for (Value operand : operation->getOperands())
      operands.push_back(valueString(operand));
    auto expression = emitHostScalar(operation, operands);
    if (mlir::failed(expression)) return failure();
    if (isa<arith::ConstantOp>(operation))
      bind(operation->getResult(0), *expression);
    else
      bindExpression(operation->getResult(0), *expression);
    return success();
  }
  if (metadataEmitters().contains(operation))
    return metadataEmitters().emit(operation, *this);
  if (controlEmitters().contains(operation))
    return controlEmitters().emit(operation, *this);
  return emitters().emit(operation, *this);
}

LogicalResult HostSourceEmitter::emitAllocation(Operation *operation) {
  auto type = cast<MemRefType>(operation->getResult(0).getType());
  ValueRange dynamicSizes = isa<memref::AllocOp>(operation)
                                ? cast<memref::AllocOp>(operation).getDynamicSizes()
                                : cast<memref::AllocaOp>(operation).getDynamicSizes();
  std::string name = newName(), elements = "1";
  auto memory = allocationDescriptor(type, name, dynamicSizes);
  for (const std::string &size : memory.sizes)
    elements = "(" + elements + ") * (" + size + ")";
  auto element = hostStorageType(type.getElementType())->name;
  int64_t alignment = 16;
  if (auto attribute = operation->getAttrOfType<IntegerAttr>("alignment"))
    alignment = std::max(alignment, attribute.getInt());
  if (isa<memref::AllocaOp>(operation)) {
    line("_Alignas(" + std::to_string(alignment) + ") " + element + " " + name +
         "[" + elements + "];");
  } else {
    std::string bytes = "sizeof(" + element + ") * (" + elements + ")";
    line(element + " *" + name + " = aligned_alloc(" +
         std::to_string(alignment) + ", ((" + bytes + " + " +
         std::to_string(alignment - 1) + ") / " + std::to_string(alignment) +
         ") * " + std::to_string(alignment) + ");");
    line("if (!" + name + " && (" + elements + ")) abort();");
  }
  memories[operation->getResult(0)] = std::move(memory);
  return success();
}

LogicalResult HostSourceEmitter::emitTaskDispatch(Operation *operation) {
  auto dispatch = cast<cpu::TaskDispatchOp>(operation);
  line("#pragma omp parallel for num_threads(" +
       valueString(dispatch.getWorkerCount()) + ")");
  std::string iv = newName();
  bind(dispatch.getBody().front().getArgument(0), iv);
  line("for (int64_t " + iv + " = 0; " + iv + " < " +
       valueString(dispatch.getCount()) + "; ++" + iv + ") {");
  ++indent;
  if (mlir::failed(emitNativeBlock(dispatch.getBody().front()))) return failure();
  --indent;
  line("}");
  return success();
}

LogicalResult verifyHostSourceProgram(ModuleOp module) {
  for (Operation &operation : module.getBody()->getOperations()) {
    auto function = dyn_cast<func::FuncOp>(operation);
    if (!function)
      return operation.emitOpError("native host module requires functions");
    if (function.getNumResults())
      return function.emitOpError("native host functions must return void");
    for (Type type : function.getArgumentTypes())
      if (failed(verifyHostType(function, type))) return failure();
    if (function.isExternal()) continue;
    if (!llvm::hasSingleElement(function.getBody()))
      return function.emitOpError("native host function requires a single block");
    auto result = function.walk([&](Operation *nested) {
      if (nested == function.getOperation()) return WalkResult::advance();
      return failed(HostSourceEmitter::verifyOperation(nested))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    if (result.wasInterrupted()) return failure();
  }
  return success();
}

} // namespace intent::weft_provider
