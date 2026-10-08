#include "Intent/Target/Weft/Serialization/Serializer.h"
#include "Intent/Target/Weft/Serialization/HostSource.h"
#include "Intent/Target/Weft/Serialization/HostScalar.h"
#include "TaskABI.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;

namespace intent::weft_provider {

FailureOr<NativeABI> queryHostABI(func::FuncOp function) {
  return queryNativeABI(
      function, getPublicInterface(function), [](Type type) -> FailureOr<Type> {
        if (type.isIndex()) return IntegerType::get(type.getContext(), 64);
        if (type.isInteger(1)) return IntegerType::get(type.getContext(), 32);
        if (type.isSignlessInteger() || type.isF32() || type.isF64()) return type;
        return failure();
      });
}

namespace {

class HostSerializer : public HostSourceEmitter {
public:
  HostSerializer(ModuleOp module, llvm::raw_ostream &output)
      : HostSourceEmitter(module, output), module(module) {}

  LogicalResult run() {
    output << "#include <stdint.h>\n#include <stddef.h>\n"
              "#include <stdlib.h>\n#include <math.h>\n";
    for (auto function : module.getOps<func::FuncOp>()) {
      if (!function.isExternal()) continue;
      output << "extern void " << function.getName() << "(";
      for (auto [index, type] : llvm::enumerate(function.getArgumentTypes())) {
        if (index) output << ", ";
        output << (isa<MemRefType>(type) ? "void *" : nativeType(type));
      }
      output << ");\n";
    }
    for (auto function : module.getOps<func::FuncOp>()) {
      if (function.isExternal()) continue;
      auto abi = queryHostABI(function);
      if (mlir::failed(abi)) return failure();
      ScopedValues local(*this);
      memories.clear();
      output << "void " << function.getName() << "(";
      bool first = true;
      for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
        std::string name = "a" + std::to_string(index);
        if (isa<MemRefType>(argument.getType())) {
          if (mlir::failed(bindEntryMemory(argument, name))) return failure();
        } else {
          bind(argument, name);
        }
      }
      for (const NativeSlot &slot : abi->slots) {
        std::string name = slot.name();
        if (!first) output << ", ";
        output << (slot.role == NativeSlotRole::Pointer
                       ? hostStorageType(slot.element)->name + " *"
                       : nativeType(slot.carrier))
               << " " << name;
        first = false;
        reserveName(name);
        if (slot.axis) {
          auto &memory =
              memories.find(function.getArgument(slot.parameter))->second;
          auto &entries = slot.role == NativeSlotRole::Extent ? memory.sizes
                                                            : memory.strides;
          entries[*slot.axis] = name;
        }
      }
      output << ") {\n";
      indent = 1;
      if (mlir::failed(emitNativeBlock(function.front())) ||
          mlir::failed(emitNativeOperation(function.front().getTerminator())))
        return failure();
      output << "}\n";
    }
    return failure(hasFailed());
  }

private:
  ModuleOp module;
};

} // namespace

LogicalResult serializeHostProgram(ModuleOp program, std::string &source) {
  if (mlir::failed(verify(program)) ||
      mlir::failed(verifyHostSourceProgram(program)))
    return failure();
  llvm::raw_string_ostream stream(source);
  return HostSerializer(program, stream).run();
}

} // namespace intent::weft_provider
