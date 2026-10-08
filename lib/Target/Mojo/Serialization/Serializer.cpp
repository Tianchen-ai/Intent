#include "Intent/Target/Mojo/Serialization/Serializer.h"
#include "Intent/Target/Mojo/Serialization/Scalar.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Serialization/NativeABI.h"
#include "Intent/Serialization/NativeSource.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace intent::mojo {
namespace {

using Memory = MemoryDescriptor;

std::string join(ArrayRef<std::string> values, llvm::StringRef separator = ", ") {
  return llvm::join(values, separator);
}

std::string dtype(Type type) { return scalarDType(type); }

std::string memoryElement(Type type) {
  return "SIMD[DType." + dtype(type) + ", 1]";
}

std::string valueType(Type type) { return scalarValueType(type); }

std::string ordering(AtomicOrdering order) {
  switch (order) {
  case AtomicOrdering::Relaxed: return "Ordering.RELAXED";
  case AtomicOrdering::Acquire: return "Ordering.ACQUIRE";
  case AtomicOrdering::Release: return "Ordering.RELEASE";
  case AtomicOrdering::AcquireRelease: return "Ordering.ACQUIRE_RELEASE";
  }
  llvm_unreachable("unknown atomic ordering");
}

class Serializer : public NativeSourceEmitter {
public:
  explicit Serializer(llvm::raw_ostream &out)
      : NativeSourceEmitter(out, NativeSourceSyntax::Mojo) {}

  std::string nativeType(Type type) override {
    if (auto memory = dyn_cast<MemRefType>(type))
      return "Pointer[" + memoryElement(memory.getElementType()) + ", MutUntrackedOrigin]";
    return valueType(type);
  }
  std::string offsetPointer(StringRef base, StringRef offset) override {
    return base.str() + ".unsafe_offset(" + offset.str() + ")";
  }
  std::string pointerAsIndex(StringRef base) override {
    return "Int(" + base.str() + ")";
  }

  LogicalResult function(func::FuncOp function, const NativeABI &abi) {
    ScopedValues local(*this);
    memories.clear();
    SmallVector<std::string> signature;
    for (auto [number, argument] : llvm::enumerate(function.getArguments())) {
      std::string name = "a" + std::to_string(number);
      values[argument] = name;
      if (isa<MemRefType>(argument.getType())) {
        if (mlir::failed(bindEntryMemory(argument, name))) return failure();
      } else {
        if (argument.getType().isIndex()) values[argument] = "Int(" + name + ")";
      }
    }
    for (const NativeSlot &slot : abi.slots) {
      Value argument = function.getArgument(slot.parameter);
      std::string name = slot.name();
      if (slot.role == NativeSlotRole::Pointer) {
        signature.push_back(name + ": Pointer[" + memoryElement(slot.element) + ", MutUntrackedOrigin]");
      } else if (slot.role == NativeSlotRole::Scalar) {
        signature.push_back(name + ": " + valueType(slot.carrier));
      } else {
        signature.push_back(name + ": Int64");
        auto &memory = memories.find(argument)->second;
        auto &entry = slot.role == NativeSlotRole::Extent ? memory.sizes[*slot.axis] : memory.strides[*slot.axis];
        if (entry.empty()) entry = "Int(" + name + ")";
      }
    }
    line("@export(\"" + function.getName().str() + "\")");
    line("@no_inline");
    line("def " + function.getName().str() + "(" + join(signature) + ") abi(\"C\"):");
    ++indent;
    line("initialize_runtime()");
    if (mlir::failed(emitNativeBlock(function.front()))) return failure();
    --indent;
    line("");
    return failure(hasFailed());
  }

private:
  std::string name(Value value) { return valueString(value); }

  std::string fresh(Value value) {
    std::string result = newName();
    bind(value, result);
    return result;
  }

  void assign(Value value, const std::string &expression, bool constant = false) {
    std::string identifier = fresh(value);
    line(std::string(constant ? "comptime " : "var ") + identifier + " = " + expression);
  }

  LogicalResult dispatch(cpu::TaskDispatchOp dispatch) {
    Block &body = dispatch.getBody().front();
    llvm::SetVector<Value> used;
    getUsedValuesDefinedAbove(dispatch.getBody(), dispatch.getBody(), used);
    SmallVector<std::pair<Value, SmallVector<std::string>>> bindings;
    SmallVector<std::string> captures;
    for (Value value : used) {
      auto expressions = components(value);
      SmallVector<std::string> fields;
      for (auto [type, expression] :
           llvm::zip_equal(componentTypes(value.getType()), expressions)) {
        fields.push_back(newName("capture_"));
        line("var " + fields.back() + ": " + nativeType(type) + " = " + expression);
        captures.push_back("imm " + fields.back());
      }
      bindings.emplace_back(value, std::move(fields));
    }
    std::string task = newName("task_");
    LogicalResult emitted = success();
    {
      ScopedValues local(*this);
      auto enclosingMemories = std::move(memories);
      memories.clear();
      auto restoreMemories = llvm::make_scope_exit([&] {
        memories = std::move(enclosingMemories);
      });
      for (auto &[value, fields] : bindings) {
        for (const std::string &field : fields) reserveName(field);
        if (auto memory = dyn_cast<MemRefType>(value.getType()))
          memories[value] = Memory::fromComponents(memory, fields);
        else bind(value, fields.front());
      }
      std::string iv = fresh(body.getArgument(0));
      line("def " + task + "(" + iv + ": Int) {" + join(captures) + "}:");
      ++indent;
      emitted = emitNativeBlock(body);
      --indent;
    }
    if (mlir::failed(emitted)) return failure();
    line("parallelize(" + task + ", " + name(dispatch.getCount()) + ", " + name(dispatch.getWorkerCount()) + ")");
    return success();
  }

  LogicalResult allocation(Operation *operation, Value memory, ValueRange dynamicSizes, bool stack) {
    auto type = cast<MemRefType>(memory.getType());
    std::string element = memoryElement(type.getElementType());
    std::string value = fresh(memory);
    Memory descriptor = allocationDescriptor(type, value, dynamicSizes);
    if (stack) {
      int64_t elementBytes = type.getElementType().isIndex() ? 8 : (type.getElementTypeBitWidth() + 7) / 8;
      int64_t alignment = cast<memref::AllocaOp>(operation).getAlignment().value_or(elementBytes);
      line("var " + value + " = unsafe_stack_allocation[" +
          std::to_string(type.getNumElements()) + ", " + element + ", alignment=" +
          std::to_string(alignment) + "]()");
    } else {
      int64_t elementBytes = type.getElementType().isIndex() ? 8 : (type.getElementTypeBitWidth() + 7) / 8;
      int64_t alignment = std::max<int64_t>(16, cast<memref::AllocOp>(operation).getAlignment().value_or(elementBytes));
      std::string elements = descriptor.sizes.empty() ? "1"
          : "(" + descriptor.strides.front() + ") * (" + descriptor.sizes.front() + ")";
      std::string bytes = "max(Int(1), (" + elements + ") * " + std::to_string(elementBytes) + ")";
      std::string size = "((" + bytes + " + " + std::to_string(alignment - 1) + ") // " +
          std::to_string(alignment) + ") * " + std::to_string(alignment);
      line("var " + value + " = external_call[\"aligned_alloc\", Pointer[" + element +
           ", MutUntrackedOrigin]](UInt(" + std::to_string(alignment) + "), UInt(" + size + "))");
    }
    memories[memory] = std::move(descriptor);
    return success();
  }

public:
  static const OperationEmitters<Serializer> &targetEmitters() {
    static const auto table = [] {
      OperationEmitters<Serializer> result;
      addNative<cpu::InvokeOp>(result);
      addNative<func::CallOp>(result, [](func::CallOp op) {
        auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
        bool entry = op.getCallee() == "intent_cpu_enter_ieee" &&
            op.getNumOperands() == 0 && op.getNumResults() == 1 &&
            op.getResult(0).getType().isInteger(32);
        bool exit = op.getCallee() == "intent_cpu_leave_ieee" &&
            op.getNumOperands() == 1 && op.getNumResults() == 0 &&
            op.getOperand(0).getType().isInteger(32);
        return callee && callee.isExternal() && callee->hasAttr("cpu.external_runtime") && (entry || exit)
            ? success() : op.emitOpError("has no declared Mojo runtime C ABI spelling");
      });
      addNative<memref::AllocOp>(result, [](memref::AllocOp op) {
        return verifyNativeAllocation(op, false);
      });
      addNative<memref::AllocaOp>(result, [](memref::AllocaOp op) {
        return verifyNativeAllocation(op, true);
      });
      addNative<memref::DeallocOp>(result);
      addNative<memref::LoadOp>(result);
      addNative<memref::StoreOp>(result);
      addNative<memref::PrefetchOp>(result, [](memref::PrefetchOp op) {
        return !op.getIsWrite() && op.getLocalityHint() == 3 && op.getIsDataCache()
            ? success() : op.emitOpError("requires read/data-cache/high-locality Mojo prefetch");
      });
      auto atomic = [](auto op) -> LogicalResult {
        Type element = cast<MemRefType>(op.getTarget().getType()).getElementType();
        return element.isF16() || element.isBF16() || directAtomicAdd(element)
            ? success() : op.emitOpError("has no supported Mojo atomic element representation");
      };
      addNative<cpu::AtomicLoadOp>(result, atomic);
      addNative<cpu::AtomicStoreOp>(result, atomic);
      addNative<cpu::AtomicCompareExchangeOp>(result, atomic);
      addNative<cpu::AtomicRMWOp>(result, [](cpu::AtomicRMWOp op) {
        return op.getKind() == AtomicRMWKind::Add && directAtomicAdd(op.getValue().getType())
            ? success() : op.emitOpError("requires native add or prior Mojo atomic expansion");
      });
      addNative<vector::LoadOp>(result);
      addNative<vector::StoreOp>(result);
      addNative<vector::BroadcastOp>(result);
      addNative<vector::FromElementsOp>(result);
      addNative<vector::StepOp>(result);
      addNative<vector::ShuffleOp>(result);
      addNative<vector::ExtractElementOp>(result);
      addNative<math::RoundEvenOp>(result);
      addNative<vector::ReductionOp>(result, [](vector::ReductionOp op) -> LogicalResult {
        auto vectorType = cast<VectorType>(op.getVector().getType());
        Type element = vectorType.getElementType();
        bool floating = element.isF32() || element.isF64();
        bool integer = element.isSignlessInteger(8) || element.isSignlessInteger(16) ||
                       element.isSignlessInteger(32) || element.isSignlessInteger(64);
        bool arithmetic = op.getKind() == vector::CombiningKind::ADD ||
                          op.getKind() == vector::CombiningKind::MUL;
        bool extrema = op.getKind() == vector::CombiningKind::MAXIMUMF ||
                       op.getKind() == vector::CombiningKind::MINIMUMF;
        if (vectorType.getRank() != 1 || vectorType.isScalable() || (!floating && !integer) ||
            (!arithmetic && !(floating && extrema)))
          return op.emitOpError("requires a fixed one-dimensional same-dtype add/mul or floating maximum/minimum reduction for Mojo");
        if (floating && arithmetic &&
            (op.getFastmath() & arith::FastMathFlags::reassoc) == arith::FastMathFlags::none)
          return op.emitOpError("requires floating reassociation permission for Mojo SIMD reduction");
        return success();
      });
      addNative<cpu::TaskDispatchOp>(result);
      return result;
    }();
    return table;
  }

  LogicalResult emitNativeOperation(Operation *operation) override {
    if (mlir::failed(verifySourceOperation(operation))) return failure();
    if (isa<math::RoundEvenOp>(operation))
      return targetEmitters().emit(operation, *this);
    if (isStandardScalarOperation(operation)) {
      SmallVector<std::string> operands;
      for (Value value : operation->getOperands()) operands.push_back(name(value));
      auto expression = emitScalar(operation, operands);
      if (mlir::failed(expression)) return failure();
      assign(operation->getResult(0), *expression, isa<arith::ConstantOp>(operation));
      return success();
    }
    if (metadataEmitters().contains(operation))
      return metadataEmitters().emit(operation, *this);
    if (controlEmitters().contains(operation))
      return controlEmitters().emit(operation, *this);
    return targetEmitters().emit(operation, *this);
  }

private:
  static bool directAtomicAdd(Type element) {
    return element.isF32() || element.isF64() ||
           element.isSignlessInteger(32) || element.isSignlessInteger(64);
  }
  template <typename Op, typename Check>
  static void addNative(OperationEmitters<Serializer> &table, Check check) {
    table.add<Op>(check, [](Op operation, Serializer &serializer) {
      return serializer.emitTyped(operation);
    });
  }
  template <typename Op> static void addNative(OperationEmitters<Serializer> &table) {
    addNative<Op>(table, [](Op) { return success(); });
  }

  LogicalResult emitTyped(cpu::InvokeOp op) {
    SmallVector<std::string> arguments;
    for (Value argument : op.getArguments())
      arguments.push_back(isa<MemRefType>(argument.getType())
          ? memoryPointer(argument) : name(argument));
    line("external_call[\"" + op.getCallee().str() + "\", NoneType](" +
         join(arguments, ", ") + ")");
    return success();
  }
  LogicalResult emitTyped(func::CallOp op) {
    if (op.getCallee() == "intent_cpu_enter_ieee")
      assign(op.getResult(0), "external_call[\"intent_cpu_enter_ieee\", UInt32]()");
    else
      line("external_call[\"intent_cpu_leave_ieee\", NoneType](" + name(op.getOperand(0)) + ")");
    return success();
  }
  LogicalResult emitTyped(memref::LoadOp op) {
    std::string expression = memoryPointer(op.getMemref(), op.getIndices()) + ".unsafe_load()";
    if (op.getType().isIndex()) expression = "Int(" + expression + ")";
    if (op.getType().isInteger(1)) expression = "Bool(" + expression + ")";
    assign(op.getResult(), expression);
    return success();
  }
  LogicalResult emitTyped(memref::StoreOp op) {
    std::string value = name(op.getValue());
    if (op.getValue().getType().isIndex()) value = "Int64(" + value + ")";
    if (op.getValue().getType().isInteger(1)) value = "SIMD[DType.bool, 1](" + value + ")";
    line(memoryPointer(op.getMemref(), op.getIndices()) + ".unsafe_store(" + value + ")");
    return success();
  }
  LogicalResult emitTyped(cpu::AtomicLoadOp op) {
    assign(op.getValue(), "Atomic[DType." + dtype(op.getValue().getType()) + "].load[ordering=" +
        ordering(op.getOrdering()) + "](" + memoryPointer(op.getTarget(), op.getIndices()) + ")");
    return success();
  }
  LogicalResult emitTyped(cpu::AtomicStoreOp op) {
    line("Atomic[DType." + dtype(op.getValue().getType()) + "].store[ordering=" + ordering(op.getOrdering()) +
        "](" + memoryPointer(op.getTarget(), op.getIndices()) + ", " + name(op.getValue()) + ")");
    return success();
  }
  LogicalResult emitTyped(cpu::AtomicRMWOp op) {
    assign(op.getOldValue(), "Atomic[DType." + dtype(op.getValue().getType()) + "].fetch_add[ordering=" +
        ordering(op.getOrdering()) + "](" + memoryPointer(op.getTarget(), op.getIndices()) + ", " + name(op.getValue()) + ")");
    return success();
  }
  LogicalResult emitTyped(cpu::AtomicCompareExchangeOp op) {
    assign(op.getOldValue(), name(op.getExpected()));
    auto failureOrder = op.getOrdering() == AtomicOrdering::Acquire || op.getOrdering() == AtomicOrdering::AcquireRelease
        ? AtomicOrdering::Acquire : AtomicOrdering::Relaxed;
    assign(op.getSuccess(), "Atomic[DType." + dtype(op.getExpected().getType()) + "].compare_exchange[success_ordering=" +
        ordering(op.getOrdering()) + ", failure_ordering=" + ordering(failureOrder) + ", weak=False](" +
        memoryPointer(op.getTarget(), op.getIndices()) + ", " + name(op.getOldValue()) + ", " + name(op.getDesired()) + ")");
    return success();
  }
  LogicalResult emitTyped(vector::LoadOp op) {
    assign(op.getResult(), memoryPointer(op.getBase(), op.getIndices()) +
        ".unsafe_load[width=" + std::to_string(op.getVectorType().getNumElements()) + "]()");
    return success();
  }
  LogicalResult emitTyped(vector::StoreOp op) {
    line(memoryPointer(op.getBase(), op.getIndices()) + ".unsafe_store(" + name(op.getValueToStore()) + ")");
    return success();
  }
  LogicalResult emitTyped(memref::PrefetchOp op) {
    line("prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](" +
        memoryPointer(op.getMemref(), op.getIndices()) + ")");
    return success();
  }
  LogicalResult emitTyped(vector::BroadcastOp op) {
    assign(op.getResult(), valueType(op.getType()) + "(" +
        (op.getType().getElementType().isInteger(1) ? "fill=" : "") + name(op.getSource()) + ")");
    return success();
  }
  LogicalResult emitTyped(vector::FromElementsOp op) {
    SmallVector<std::string> lanes;
    Type element = op.getType().getElementType();
    for (Value value : op.getElements())
      lanes.push_back(element.isInteger(1) || element.isIndex()
          ? memoryElement(element) + "(" + name(value) + ")" : name(value));
    assign(op.getResult(), valueType(op.getType()) + "(" + join(lanes) + ")");
    return success();
  }
  LogicalResult emitTyped(vector::StepOp op) {
    SmallVector<std::string> lanes;
    for (int64_t lane = 0; lane < op.getType().getNumElements(); ++lane)
      lanes.push_back(std::to_string(lane));
    assign(op.getResult(), valueType(op.getType()) + "(" + join(lanes) + ")", true);
    return success();
  }
  LogicalResult emitTyped(vector::ShuffleOp op) {
    SmallVector<std::string> lanes;
    int64_t lhsSize = cast<VectorType>(op.getV1().getType()).getNumElements();
    for (int64_t lane : op.getMask())
      lanes.push_back(name(lane < lhsSize ? op.getV1() : op.getV2()) + "[" +
          std::to_string(lane < lhsSize ? lane : lane - lhsSize) + "]");
    assign(op.getResult(), valueType(op.getType()) + "(" + join(lanes) + ")");
    return success();
  }
  LogicalResult emitTyped(vector::ExtractElementOp op) {
    std::string expression = name(op.getVector()) + "[" + name(op.getPosition()) + "]";
    if (op.getResult().getType().isIndex()) expression = "Int(" + expression + ")";
    assign(op.getResult(), expression);
    return success();
  }
  LogicalResult emitTyped(math::RoundEvenOp op) {
    assign(op.getResult(), "llvm_intrinsic[\"llvm.roundeven\", " + valueType(op.getType()) +
        ", has_side_effect=False](" + name(op.getOperand()) + ")");
    return success();
  }
  LogicalResult emitTyped(vector::ReductionOp op) {
    if (op.getKind() == vector::CombiningKind::MAXIMUMF ||
        op.getKind() == vector::CombiningKind::MINIMUMF) {
      bool maximum = op.getKind() == vector::CombiningKind::MAXIMUMF;
      std::string intrinsic = maximum ? "llvm.vector.reduce.fmaximum" : "llvm.vector.reduce.fminimum";
      std::string type = valueType(op.getDest().getType());
      std::string expression = "llvm_intrinsic[\"" + intrinsic + "\", " + type + "](" +
                               name(op.getVector()) + ")";
      if (op.getAcc())
        expression = "llvm_intrinsic[\"" + std::string(maximum ? "llvm.maximum" : "llvm.minimum") +
                     "\", " + type + "](" + name(op.getAcc()) + ", " + expression + ")";
      assign(op.getDest(), expression);
      return success();
    }
    bool add = op.getKind() == vector::CombiningKind::ADD;
    std::string expression = "(" + name(op.getVector()) + ")" +
                            (add ? ".reduce_add()" : ".reduce_mul()");
    std::string accumulator;
    if (op.getAcc()) accumulator = name(op.getAcc());
    else if (add && isa<FloatType>(op.getDest().getType()))
      accumulator = valueType(op.getDest().getType()) + "(0.0)";
    if (!accumulator.empty())
      expression = "(" + accumulator + (add ? " + " : " * ") + expression + ")";
    assign(op.getDest(), expression);
    return success();
  }
  LogicalResult emitTyped(memref::AllocaOp op) {
    return allocation(op, op.getResult(), op.getDynamicSizes(), true);
  }
  LogicalResult emitTyped(memref::AllocOp op) {
    return allocation(op, op.getResult(), op.getDynamicSizes(), false);
  }
  LogicalResult emitTyped(memref::DeallocOp op) {
    line("external_call[\"free\", NoneType](" + memories.at(op.getMemref()).base + ")");
    return success();
  }
  LogicalResult emitTyped(cpu::TaskDispatchOp op) { return dispatch(op); }

};

}

LogicalResult verifySourceOperation(Operation *operation) {
  auto verifyType = [&](Type type) -> LogicalResult {
    if (auto memory = dyn_cast<MemRefType>(type)) {
      if (failed(verifyNativeMemoryType(operation, memory))) return failure();
      auto space = dyn_cast_or_null<IntegerAttr>(memory.getMemorySpace());
      if (memory.getMemorySpace() && (!space || space.getInt() != 0))
        return operation->emitOpError("Mojo source requires host memory space");
      type = memory.getElementType();
      if (!type.isIntOrIndexOrFloat())
        return operation->emitOpError("Mojo memory requires scalar numeric elements");
    }
    return supportsScalarType(type) ? success()
        : operation->emitOpError("type has no Mojo source representation: ") << type;
  };
  for (Type type : llvm::concat<Type>(operation->getOperandTypes(), operation->getResultTypes()))
    if (failed(verifyType(type))) return failure();
  for (Region &region : operation->getRegions())
    for (Block &block : region)
      for (Type type : block.getArgumentTypes())
        if (failed(verifyType(type))) return failure();
  if (isa<ModuleOp, func::FuncOp, func::ReturnOp, scf::YieldOp,
          scf::ConditionOp, cpu::TaskYieldOp>(operation))
    return success();
  if (isa<math::RoundEvenOp>(operation))
    return Serializer::targetEmitters().verify(operation);
  if (isStandardScalarOperation(operation)) return verifyScalar(operation);
  const auto &metadata = NativeSourceEmitter::metadataEmitters();
  if (metadata.contains(operation)) return metadata.verify(operation);
  const auto &control = NativeSourceEmitter::controlEmitters();
  if (control.contains(operation)) return control.verify(operation);
  return Serializer::targetEmitters().verify(operation);
}

LogicalResult verifySourceProgram(ModuleOp module) {
  auto result = module.walk([&](Operation *operation) {
    return succeeded(verifySourceOperation(operation))
        ? WalkResult::advance() : WalkResult::interrupt();
  });
  return failure(result.wasInterrupted());
}

LogicalResult serializeProgram(ModuleOp module, std::string &source, std::string &metadata) {
  auto options = readCompileOptions(module);
  if (mlir::failed(options)) return failure();
  if (mlir::failed(cpu::verifyCPUProgram(module, cpu::CPUProgramStage::Realized)) ||
      mlir::failed(verifySourceProgram(module))) return failure();
  llvm::raw_string_ostream output(source);
  output << "from std.ffi import external_call\n"
            "from std.atomic import Atomic, Ordering\n"
            "from std.memory import unsafe_stack_allocation, bitcast\n"
            "from std.sys import prefetch, llvm_intrinsic\n"
            "from std.sys.intrinsics import PrefetchOptions\n"
            "from std.math import fma, sqrt, exp, exp2, log, tanh, sin, cos, floor, erf, abs, min, max\n"
            "from std.runtime import initialize_runtime\n"
            "from max.algorithm import parallelize\n\n";
  Serializer serializer(output);
  llvm::json::Object interface;
  auto capabilities = module->getAttrOfType<cpu::CapabilitiesAttr>("intent_cpu.capabilities");
  interface["provider"] = "mojo";
  interface["compile_options"] = serializeCompileOptions(*options);
  interface["target"] = llvm::json::Object{
      {"family", "cpu"}, {"vector_bits", capabilities.getVectorBits()},
      {"workers", capabilities.getWorkers()},
      {"private_bytes", capabilities.getPrivateBytes()},
      {"matrix_i8_i32", capabilities.getMatrixI8I32()}};
  interface["native_dependencies"] = llvm::json::Array{"std", "max"};
  interface["source_prelude_end"] = static_cast<int64_t>(output.tell());
  llvm::json::Array candidates;
  bool first = true;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.isExternal()) continue;
    auto nativeABI = queryNativeABI(function, getPublicInterface(function), [](Type type) -> FailureOr<Type> {
      if (type.isIndex()) return IntegerType::get(type.getContext(), 64);
      if (auto integer = dyn_cast<IntegerType>(type))
        return IntegerType::get(type.getContext(), integer.getWidth());
      if (type.isF32() || type.isF64()) return type;
      return failure();
    });
    if (mlir::failed(nativeABI)) return failure();
    int64_t sourceBegin = output.tell();
    if (mlir::failed(serializer.function(function, *nativeABI))) return failure();
    int64_t sourceEnd = output.tell();
    if (first) {
      auto abi = serializePublicInterface(function, getPublicInterface(function));
      if (mlir::failed(abi)) return failure();
      auto entry = function->getAttrOfType<cpu::EntryRequirementsAttr>(cpu::entryRequirementsAttr);
      auto requirements = queryNativeEntryRequirements(function, getPublicInterface(function),
          entry.getDisjointOutputs(), [&](unsigned, intent::ViewType view) -> FailureOr<NativeViewRequirements> {
            auto layout = entry.getContiguousViews() || view.getAccess() != 0
                ? NativeViewLayout::Contiguous : NativeViewLayout::Strided;
            return NativeViewRequirements{layout, 1};
          });
      if (mlir::failed(requirements)) return failure();
      interface["entry_name"] = function.getName();
      interface["interface"] = std::move(*abi);
      interface["native"] = llvm::json::Object{{"requirements", requirements->serialize()},
          {"slots", nativeABI->serialize()}};
      first = false;
    }
    auto configuration = function->getAttrOfType<cpu::ConfigurationAttr>("intent_cpu.configuration");
    llvm::json::Array implementations;
    for (Attribute entry : function->getAttrOfType<ArrayAttr>("intent_cpu.implementations")) {
      auto binding = cast<cpu::ImplementationAttr>(entry);
      llvm::json::Object values;
      for (NamedAttribute parameter : binding.getParameters())
        values[parameter.getName().getValue()] = cast<IntegerAttr>(parameter.getValue()).getInt();
      implementations.push_back(llvm::json::Object{{"name", binding.getName().getValue().str()}, {"parameters", std::move(values)}});
    }
    candidates.push_back(llvm::json::Object{
        {"entry", function.getName().str()},
        {"source_range", llvm::json::Array{sourceBegin, sourceEnd}},
        {"values", llvm::json::Array{configuration.getTaskGrain(),
            configuration.getTileM(), configuration.getTileN(), configuration.getTileK(), configuration.getRegionSize()}},
        {"implementations", std::move(implementations)}});
  }
  interface["candidates"] = std::move(candidates);
  llvm::raw_string_ostream metadataOutput(metadata);
  metadataOutput << llvm::json::Value(std::move(interface));
  return success();
}

}
