#include "Intent/Target/Mojo/Serialization/Matrix.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;

namespace intent::mojo {
namespace {

std::string constant(int64_t value) {
  return "__mlir_op.`llvm.mlir.constant`[_type=__mlir_type.i16, value=__mlir_attr.`" +
      std::to_string(value) + " : i16`]()";
}

std::string intrinsic(StringRef name, ArrayRef<std::string> arguments, bool result = true) {
  return "__mlir_op.`llvm.call_intrinsic`[_type=" +
      std::string(result ? "__mlir_type.`!llvm.x86_amx`" : "None") +
      ", _properties=__mlir_attr.`{intrin = \"llvm.x86." + name.str() +
      ".internal\", operandSegmentSizes = array<i32: " + std::to_string(arguments.size()) +
      ", 0>, op_bundle_sizes = array<i32>}`](" + llvm::join(arguments, ", ") + ")";
}

LogicalResult capability(Operation *operation) {
  auto facts = operation->getParentOfType<ModuleOp>()->getAttrOfType<cpu::CapabilitiesAttr>(
      "intent_cpu.capabilities");
  if (!facts || !facts.getMatrixI8I32())
    return operation->emitOpError("AMX integer source requires the native matrix_i8_i32 capability");
  return success();
}

template <typename Op> LogicalResult memory(Op operation) {
  if (failed(capability(operation))) return failure();
  auto type = operation.getMemRefType();
  SmallVector<int64_t> strides;
  int64_t offset;
  if (type.getRank() != 2 || failed(type.getStridesAndOffset(strides, offset)) || strides[1] != 1)
    return operation.emitOpError("AMX native memory requires a two-dimensional row-contiguous view");
  if (!type.getElementType().isInteger(8) && !type.getElementType().isInteger(32))
    return operation.emitOpError("AMX native memory currently requires integer storage");
  return success();
}

SmallVector<std::string> access(NativeSourceEmitter &out, Value memory, ValueRange indices,
                                amx::TileType type) {
  auto memoryType = cast<MemRefType>(memory.getType());
  auto bytes = memoryType.getElementTypeBitWidth() / 8;
  auto pointer = "__mlir_op.`builtin.unrealized_conversion_cast`[_type=__mlir_type.`!llvm.ptr`](" +
      out.memoryPointer(memory, indices) + ")";
  auto stride = "__mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.i64](Int64((" +
      out.memories.at(memory).strides[0] + ") * " + std::to_string(bytes) + ")._mlir_value)";
  return {constant(type.getShape()[0]), constant(type.getShape()[1] * bytes), pointer, stride};
}

void assign(NativeSourceEmitter &out, Value value, StringRef expression) {
  auto name = out.newName();
  out.bind(value, name);
  out.line("var " + name + " = " + expression);
}

} // namespace

const OperationEmitters<NativeSourceEmitter> &matrixEmitters() {
  static const auto table = [] {
    OperationEmitters<NativeSourceEmitter> result;
    result.add<amx::TileZeroOp>(capability, [](amx::TileZeroOp op, NativeSourceEmitter &out) {
      auto type = op.getTileType();
      assign(out, op.getRes(), intrinsic("tilezero", {constant(type.getShape()[0]),
          constant(type.getShape()[1] * type.getElementTypeBitWidth() / 8)}));
      return success();
    });
    result.add<amx::TileLoadOp>(memory<amx::TileLoadOp>, [](amx::TileLoadOp op, NativeSourceEmitter &out) {
      assign(out, op.getRes(), intrinsic("tileloadd64", access(out, op.getBase(), op.getIndices(), op.getTileType())));
      return success();
    });
    result.add<amx::TileStoreOp>(memory<amx::TileStoreOp>, [](amx::TileStoreOp op, NativeSourceEmitter &out) {
      auto arguments = access(out, op.getBase(), op.getIndices(), op.getTileType());
      arguments.push_back(out.valueString(op.getVal()));
      out.line(intrinsic("tilestored64", arguments, false));
      return success();
    });
    result.add<amx::TileMulIOp>(capability, [](amx::TileMulIOp op, NativeSourceEmitter &out) {
      auto output = op.getTileType();
      std::string name = std::string("tdpb") + (op.getIsZextLhs() ? "u" : "s") +
          (op.getIsZextRhs() ? "u" : "s") + "d";
      assign(out, op.getRes(), intrinsic(name, {constant(output.getShape()[0]),
          constant(output.getShape()[1] * 4), constant(op.getLhsTileType().getShape()[1]),
          out.valueString(op.getAcc()), out.valueString(op.getLhs()), out.valueString(op.getRhs())}));
      return success();
    });
    return result;
  }();
  return table;
}

} // namespace intent::mojo
