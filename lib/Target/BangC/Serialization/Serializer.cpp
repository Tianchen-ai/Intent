#include "Intent/Target/BangC/Passes.h"
#include "Scalar.h"
#include "Surface.h"
#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "Intent/Serialization/NativeABI.h"
#include "Intent/Serialization/NativeSource.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
using namespace mlir;
namespace intent::bangc {
namespace {
#include "../Runtime/TileImplementations.inc"
std::string ctype(Type type) { return scalarType(type)->name; }
LogicalResult verifyAllocationGeometry(memref::AllocaOp allocation) {
  if (mlir::failed(verifyNativeAllocation(allocation, true))) return failure();
  auto space = allocation.getType().getMemorySpaceAsInt();
  if (space != dsa::nramSpace && space != dsa::matrixSpace && space != dsa::sharedSpace)
    return allocation.emitOpError("BANG C allocation requires NRAM, WRAM, or SRAM storage");
  return success();
}
class Serializer : public NativeSourceEmitter {
public:
  Serializer(func::FuncOp function, llvm::raw_ostream &output, unsigned candidate)
      : NativeSourceEmitter(output, NativeSourceSyntax::C), function(function),
        deviceEntry("intent_device_" + std::to_string(candidate)),
        launchEntry("intent_launch_" + std::to_string(candidate)) {}
  static const OperationEmitters<Serializer> &nativeEmitters();
  std::string nativeType(Type type) override {
    if (auto memory = dyn_cast<MemRefType>(type))
      return ctype(memory.getElementType()) + " *";
    return ctype(type);
  }
  std::string offsetPointer(StringRef base, StringRef offset) override {
    return "(" + base.str() + " + (" + offset.str() + "))";
  }
  std::string pointerAsIndex(StringRef base) override {
    return "reinterpret_cast<int64_t>(" + base.str() + ")";
  }
  LogicalResult emit(llvm::json::Object &metadata) {
    auto options = readCompileOptions(function);
    if (mlir::failed(options)) return failure();
    auto interface = getPublicInterface(function);
    auto publicMetadata = serializePublicInterface(function, interface);
    if (mlir::failed(publicMetadata)) return failure();
    auto nativeABI = queryNativeABI(function, interface, [](Type type) -> FailureOr<Type> {
      if (type.isIndex()) return IntegerType::get(type.getContext(), 64);
      if (type.isInteger(1) || type.isInteger(32) || type.isInteger(64))
        return IntegerType::get(type.getContext(), cast<IntegerType>(type).getWidth());
      if (type.isF32()) return type;
      return failure();
    });
    if (mlir::failed(nativeABI)) return failure();
    auto entry = function->getAttrOfType<dsa::EntryRequirementsAttr>(dsa::entryRequirementsAttr);
    auto requirements = queryNativeEntryRequirements(function, interface,
        entry.getDisjointOutputs(), [](unsigned, intent::ViewType) -> FailureOr<NativeViewRequirements> {
          return NativeViewRequirements{NativeViewLayout::Strided, 1};
        });
    if (mlir::failed(requirements)) return failure();
    auto config = function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration");
    auto architecture = function->getParentOfType<ModuleOp>()->getAttrOfType<StringAttr>("bangc.architecture");
    if (!architecture) return function.emitError("BANG C serialization requires a bound architecture");
    for (const NativeSlot &slot : nativeABI->slots) {
      std::string name = slot.name();
      Value argument = function.getArgument(slot.parameter);
      reserveName(name);
      call.push_back(name);
      if (slot.role == NativeSlotRole::Pointer) {
        auto view = getPublicView(interface, slot.parameter);
        signature.push_back((view.getAccess() == 0 ? "const " : "") + ctype(slot.element) + " *" + name);
        if (mlir::failed(bindEntryMemory(argument, name))) return failure();
      } else signature.push_back(ctype(slot.carrier) + " " + name);
      if (slot.role == NativeSlotRole::Extent) {
        auto &size = memories.find(argument)->second.sizes[*slot.axis];
        if (size.empty()) size = name;
      } else if (slot.role == NativeSlotRole::Stride) {
        auto &stride = memories.find(argument)->second.strides[*slot.axis];
        if (stride.empty()) stride = name;
      } else if (slot.role == NativeSlotRole::Scalar)
        SourceEmitter::bind(argument, name);
    }
    auto fullExtentRequirements = dsa::queryFullExtentRequirements(function);
    if (mlir::failed(fullExtentRequirements)) return failure();
    llvm::json::Array fullExtents;
    for (int64_t value : fullExtentRequirements->asArrayRef())
      fullExtents.push_back(value);
    metadata = llvm::json::Object{{"provider", "bangc"}, {"entry_name", function.getName()},
        {"compile_options", serializeCompileOptions(*options)},
        {"interface", std::move(*publicMetadata)}, {"entry", launchEntry},
        {"target", llvm::json::Object{{"family", "dsa"}, {"architecture", architecture.getValue()}}},
        {"configuration", llvm::json::Object{
            {"tile", config.getTile()}, {"tasks", config.getTasks()},
            {"tile_m", config.getTileM()}, {"tile_n", config.getTileN()}, {"tile_k", config.getTileK()},
            {"region_tile", config.getRegionTile()}, {"local_bytes", config.getLocalBytes()}}},
        {"full_extent_dimensions", std::move(fullExtents)},
        {"native", llvm::json::Object{{"requirements", requirements->serialize()},
            {"slots", nativeABI->serialize()}}}};
    output << "__mlu_global__ void " << deviceEntry << "(" << llvm::join(signature, ", ") << ") {\n";
    auto bytes = [&](StringRef name) { return function->getAttrOfType<IntegerAttr>(name).getInt(); };
    if (bytes("bangc.nram_bytes")) line("__nram__ __attribute__((aligned(128))) unsigned char local_nram[" + std::to_string(bytes("bangc.nram_bytes")) + "];", 1);
    if (bytes("bangc.wram_bytes")) line("__wram__ __attribute__((aligned(128))) unsigned char local_wram[" + std::to_string(bytes("bangc.wram_bytes")) + "];", 1);
    if (bytes("bangc.sram_bytes")) line("__mlu_shared__ __attribute__((aligned(128))) unsigned char group_sram[" + std::to_string(bytes("bangc.sram_bytes")) + "];", 1);
    indent = 1;
    if (mlir::failed(emitNativeBlock(function.front()))) return failure();
    indent = 0;
    output << "}\n\nextern \"C\" int " << launchEntry << "(void *stream";
    if (!signature.empty()) output << ", " << llvm::join(signature, ", ");
    auto group = function->getAttrOfType<IntegerAttr>("intent_dsa.group_width");
    output << ") {\n  cnrtDim3_t dim = {" << (group ? group.getInt() : config.getTasks()) << ", "
        << (group ? config.getTasks() / group.getInt() : 1) << ", 1};\n"
        << "  " << deviceEntry << "<<<dim, " << (group ? "cnrtFuncTypeUnion1" : "cnrtFuncTypeBlock") << ", static_cast<cnrtQueue_t>(stream)>>>("
        << llvm::join(call, ", ") << ");\n  return static_cast<int>(cnrtGetLastError());\n}\n";
    return failure(hasFailed());
  }
private:
  std::string name(Value value) {
    return isa<MemRefType>(value.getType()) ? memoryPointer(value) : valueString(value);
  }
  std::string bind(Value value) {
    std::string result = newName();
    SourceEmitter::bind(value, result);
    return result;
  }
  std::string count(Value value) { return std::to_string(cast<MemRefType>(value.getType()).getNumElements()); }
  std::string shape(Value value) {
    auto type = cast<MemRefType>(value.getType());
    return ctype(type.getElementType()) + ", " + std::to_string(type.getDimSize(0)) + ", " + std::to_string(type.getDimSize(1));
  }
  LogicalResult emitNativeOperation(Operation *op) override {
    if (isa<func::ReturnOp, scf::YieldOp>(op)) return success();
    if (nativeEmitters().contains(op)) return nativeEmitters().emit(op, *this);
    if (auto sync = dyn_cast<dsa::SynchronizeOp>(op)) {
      line(sync.getLocalOnly() ? "intent_sync_local();" : "__sync();"); return success();
    }
    if (isa<dsa::GroupSynchronizeOp>(op)) { line("__sync(); __sync_cluster();"); return success(); }
    if (auto load = dyn_cast<dsa::LoadTileOp>(op)) {
      bool local = cast<MemRefType>(load.getSource().getType()).getMemorySpaceAsInt() == dsa::nramSpace;
      bool shared = cast<MemRefType>(load.getSource().getType()).getMemorySpaceAsInt() == dsa::sharedSpace;
      line(std::string(local ? "intent_load_local_tile<" : "intent_load_tile<") + shape(load.getOutput()) +
          (shared ? (load.getAsynchronous() ? ", true, true" : ", false, true") : (load.getAsynchronous() ? ", true" : "")) + ">(" + name(load.getOutput()) + ", " + name(load.getSource()) +
          ", " + name(load.getOffset()) + ", " + name(load.getRowStride()) + ", " + name(load.getColumnStride()) +
          ", " + name(load.getRows()) + ", " + name(load.getColumns()) + ");"); return success();
    }
    if (auto stage = dyn_cast<dsa::StageTileOp>(op)) {
      line("intent_stage_tile<" + shape(stage.getOutput()) + ">(" + name(stage.getOutput()) + ", " +
          name(stage.getSource()) + ", " + name(stage.getOffset()) + ", " + name(stage.getRowStride()) + ", " +
          name(stage.getColumnStride()) + ", " + name(stage.getRows()) + ", " + name(stage.getColumns()) + ");");
      return success();
    }
    if (auto plan = dyn_cast<dsa::GatherPlanOp>(op)) {
      line("intent_prepare_gather_runs<" + count(plan.getRowOffsets()) + ">(" + name(plan.getRowOffsets()) + ", " +
          name(plan.getRows()) + ", " + name(plan.getOutput()) + ", " + name(plan.getLaneIndices()) + ");");
      return success();
    }
    if (auto gather = dyn_cast<dsa::GatherRowsOp>(op)) {
      line(std::string(gather.getPlan() ? "intent_gather_runs<" : "intent_gather_rows<") + shape(gather.getOutput()) + ">(" + name(gather.getOutput()) + ", " +
          name(gather.getSource()) + ", " + name(gather.getRowOffsets()) + ", " + name(gather.getColumnStride()) +
          ", " + name(gather.getRows()) + ", " + name(gather.getColumns()) +
          (gather.getPlan() ? ", " + name(gather.getPlan()) : "") + ");"); return success();
    }
    if (auto gather = dyn_cast<dsa::GroupGatherRowsOp>(op)) {
      line("intent_group_gather_rows<" + shape(gather.getOutput()) + ">(" + name(gather.getOutput()) + ", " +
          name(gather.getSource()) + ", " + name(gather.getRowOffsets()) + ", " + name(gather.getPlan()) + ", " +
          name(gather.getSharedData()) + ", " + name(gather.getSharedMetadata()) + ", " +
          name(gather.getRows()) + ", " + name(gather.getLane()) + ");");
      return success();
    }
    if (auto store = dyn_cast<dsa::StoreTileOp>(op)) {
      bool local = cast<MemRefType>(store.getDestination().getType()).getMemorySpaceAsInt() == dsa::nramSpace;
      line(std::string(local ? "intent_store_local_tile<" : "intent_store_tile<") + shape(store.getInput()) + ">(" + name(store.getDestination()) + ", " + name(store.getInput()) +
          ", " + name(store.getOffset()) + ", " + name(store.getRowStride()) + ", " + name(store.getColumnStride()) +
          ", " + name(store.getRows()) + ", " + name(store.getColumns()) + ");"); return success();
    }
    if (auto iota = dyn_cast<dsa::IotaOp>(op)) {
      line("intent_iota_local<" + count(iota.getOutput()) + ">(" + name(iota.getOutput()) + ");");
      return success();
    }
    if (auto broadcast = dyn_cast<dsa::BroadcastRowsOp>(op)) {
      line("intent_broadcast_rows<" + shape(broadcast.getOutput()) + ">(" + name(broadcast.getOutput()) + ", " +
          name(broadcast.getInput()) + ", " + name(broadcast.getScratch()) + ", " + name(broadcast.getOffset()) + ", " +
          name(broadcast.getRows()) + ", " + name(broadcast.getColumns()) + ");");
      return success();
    }
    if (auto layout = dyn_cast<dsa::IndexLayoutOp>(op)) {
      auto output = cast<MemRefType>(layout.getOutput().getType());
      std::string width = std::to_string(output.getDimSize(1));
      std::string dst = "reinterpret_cast<uint32_t *>(" + name(layout.getOutput()) + ")";
      if (layout.getInput().getType().isInteger(64)) {
        line("__bang_write_value(" + dst + ", " + width + ", uint32_t(" + name(layout.getInput()) + "));");
        line("__bang_write_value(" + dst + " + " + width + ", " + width + ", uint32_t(uint64_t(" + name(layout.getInput()) + ") >> 32));");
      } else {
        bool split = cast<MemRefType>(layout.getInput().getType()).getElementType().isInteger(64);
        line("__bang_transpose(" + dst + ", reinterpret_cast<const uint32_t *>(" + name(layout.getInput()) + "), " +
             (split ? width + ", 2" : "2, " + width) + ");");
      }
      return success();
    }
    if (auto binary = dyn_cast<dsa::IndexBinaryOp>(op)) {
      std::string width = std::to_string(cast<MemRefType>(binary.getOutput().getType()).getDimSize(1));
      std::string dst = "reinterpret_cast<uint32_t *>(" + name(binary.getOutput()) + ")";
      std::string lhs = "reinterpret_cast<const uint32_t *>(" + name(binary.getLhs()) + ")";
      bool shift = binary.getKind() == BinaryOperator::LeftShift || binary.getKind() == BinaryOperator::RightShift;
      if (shift) {
        std::string amount = std::to_string(binary.getRhs().getDefiningOp<arith::ConstantIntOp>().value());
        line(std::string(binary.getKind() == BinaryOperator::LeftShift ? "intent_index_shl<" : "intent_index_sar<") +
             width + ", " + amount + ">(" + dst + ", " + lhs + ");");
      } else {
        StringRef kind = binary.getKind() == BinaryOperator::Add ? "add" : binary.getKind() == BinaryOperator::Subtract ? "sub" : "mul";
        std::string rhs = isa<MemRefType>(binary.getRhs().getType()) ?
            "reinterpret_cast<const uint32_t *>(" + name(binary.getRhs()) + ")" : name(binary.getRhs());
        line("intent_index_" + kind.str() + "<" + width + ">(" + dst + ", " + lhs + ", " + rhs + ");");
      }
      return success();
    }
    if (auto fill = dyn_cast<dsa::FillOp>(op)) {
      if (fill.getValue().getType().isF16() || fill.getValue().getType().isF32())
        line("__bang_write_value(" + name(fill.getOutput()) + ", " + count(fill.getOutput()) + ", " + name(fill.getValue()) + ");");
      else line("intent_fill_local<" + ctype(fill.getValue().getType()) + ", " + count(fill.getOutput()) + ">(" + name(fill.getOutput()) + ", " + name(fill.getValue()) + ");");
      return success();
    }
    if (auto select = dyn_cast<dsa::SelectOp>(op)) {
      if (Value scratch = select.getScratch()) {
        line("intent_select_bits<" + count(select.getOutput()) + ", " + std::to_string(cast<MemRefType>(scratch.getType()).getDimSize(1)) + ">(" +
            name(select.getOutput()) + ", " + name(select.getCondition()) + ", " + name(select.getTrueValue()) + ", " +
            name(select.getFalseValue()) + ", " + name(scratch) + ");");
        return success();
      }
      line("intent_select_local<" + ctype(cast<MemRefType>(select.getOutput().getType()).getElementType()) + ", " +
          count(select.getOutput()) + ">(" + name(select.getOutput()) + ", " + name(select.getCondition()) + ", " +
          name(select.getTrueValue()) + ", " + name(select.getFalseValue()) + ");"); return success();
    }
    if (auto prepare = dyn_cast<dsa::PrepareMatrixViewOp>(op)) {
      auto output = cast<MemRefType>(prepare.getOutput().getType());
      auto input = cast<MemRefType>(prepare.getInputSlice().getType());
      line("intent_prepare_matrix_view<" + ctype(output.getElementType()) + ", " +
          std::to_string(output.getDimSize(0)) + ", " + std::to_string(output.getDimSize(1)) + ", " +
          std::to_string(input.getDimSize(0)) + ", " + std::to_string(input.getDimSize(1)) + ">(" + name(prepare.getOutput()) + ", " + name(prepare.getSource()) +
          ", " + name(prepare.getInputSlice()) + ", " + name(prepare.getTransposedSlice()) + ", " +
          name(prepare.getOffset()) + ", " + name(prepare.getRowStride()) + ", " + name(prepare.getColumnStride()) +
          ", " + name(prepare.getColumns()) + ");");
      return success();
    }
    if (auto transpose = dyn_cast<dsa::TransposeOp>(op)) {
      line("intent_transpose_tile<" + shape(transpose.getInput()) + ">(" + name(transpose.getOutput()) + ", " +
          name(transpose.getInput()) + ", " + name(transpose.getRows()) + ", " + name(transpose.getColumns()) + ");");
      return success();
    }
    if (auto compare = dyn_cast<dsa::CompareRangeOp>(op)) {
      auto type = cast<MemRefType>(compare.getOutput().getType());
      line("intent_compare_range<" + std::to_string(type.getDimSize(0)) + ", " +
          std::to_string(type.getDimSize(1)) + ">(" + name(compare.getOutput()) + ", " +
          name(compare.getRowCoordinates()) + ", " + name(compare.getRows()) + ", " +
          std::to_string(compare.getBaseAttr().getInt()) + "LL);");
      return success();
    }
    if (auto compare = dyn_cast<dsa::CompareRampOp>(op)) {
      auto type = cast<MemRefType>(compare.getOutput().getType());
      line("intent_compare_ramp<" + std::to_string(type.getDimSize(0)) + ", " +
          std::to_string(type.getDimSize(1)) + ">(" + name(compare.getOutput()) + ", " +
          (compare.getScratch() ? name(compare.getScratch()) : "nullptr") + ", " + name(compare.getRowBegin()) + ", " +
          std::to_string(compare.getBaseAttr().getInt()) + "LL);");
      return success();
    }
    if (auto masked = dyn_cast<dsa::MaskedFillOp>(op)) {
      line("intent_masked_fill<" + count(masked.getOutput()) + ">(" + name(masked.getOutput()) + ", " +
          name(masked.getInput()) + ", " + name(masked.getMask()) + ", " + name(masked.getValue()) + ");");
      return success();
    }
    if (auto compare = dyn_cast<dsa::CompareOp>(op)) {
      if (compare.getScratch()) {
        auto workspace = cast<MemRefType>(compare.getScratch().getType());
        line("intent_compare_i64<" + count(compare.getOutput()) + ", " + std::to_string(workspace.getDimSize(1)) +
            ", " + std::to_string(static_cast<unsigned>(compare.getPredicate())) + ">(" + name(compare.getOutput()) +
            ", " + name(compare.getLhs()) + ", " + name(compare.getRhs()) + ", " + name(compare.getScratch()) + ");");
        return success();
      }
      auto callee = op->getAttrOfType<StringAttr>("bangc.callee");
      if (!callee) return op->emitError("comparison requires a selected BANG C primitive");
      line(callee.getValue().str() + "(" + count(compare.getOutput()) + ", " + name(compare.getOutput()) + ", " +
          name(compare.getLhs()) + ", " + name(compare.getRhs()) + ");");
      return success();
    }
    if (auto store = dyn_cast<dsa::StoreScalarOp>(op)) {
      line(name(store.getDestination()) + "[" + name(store.getOffset()) + "] = " + name(store.getValue()) + ";"); return success();
    }
    if (auto atomic = dyn_cast<dsa::AtomicAddOp>(op)) {
      line("__bang_atomic_add(" + name(atomic.getOutput()) + ", " +
           name(atomic.getSource()) + " + " + name(atomic.getOffset()) + ", " +
           name(atomic.getValue()) + ", 1);");
      line("__sync();");
      return success();
    }
    if (auto binary = dyn_cast<dsa::BinaryOp>(op)) {
      if (auto implementation = op->getAttrOfType<StringAttr>("bangc.implementation");
          implementation && implementation.getValue() == "cycle") {
        StringRef callee = binary.getKind() == BinaryOperator::Add ? "__bang_cycle_add" :
            binary.getKind() == BinaryOperator::Subtract ? "__bang_cycle_sub" : "__bang_cycle_mul";
        line(callee.str() + "(" + name(binary.getOutput()) + ", " + name(binary.getLhs()) + ", " +
            name(binary.getRhs()) + ", " + count(binary.getOutput()) + ", " + count(binary.getRhs()) + ");");
        return success();
      }
      if (auto implementation = op->getAttrOfType<StringAttr>("bangc.implementation");
          implementation && implementation.getValue() == "row_scalar") {
        auto type = cast<MemRefType>(binary.getOutput().getType());
        line("intent_binary_rows<" + ctype(type.getElementType()) + ", " +
            std::to_string(type.getDimSize(0)) + ", " + std::to_string(type.getDimSize(1)) + ", " +
            std::to_string(static_cast<int>(binary.getKind())) + ">(" + name(binary.getOutput()) + ", " +
            name(binary.getLhs()) + ", " + name(binary.getRhs()) + ");");
        return success();
      }
      if (auto implementation = op->getAttrOfType<StringAttr>("bangc.implementation");
          implementation && implementation.getValue() == "reciprocal_f32_ftz") {
        line("intent_reciprocal_ftz<" + count(binary.getOutput()) + ", " +
            std::to_string(cast<MemRefType>(binary.getScratch().getType()).getDimSize(1)) + ">(" +
            name(binary.getOutput()) + ", " + name(binary.getRhs()) + ", " + name(binary.getScratch()) + ");");
        return success();
      }
      if (binary.getScratch()) {
        auto workspace = cast<MemRefType>(binary.getScratch().getType());
        auto implementation = op->getAttrOfType<StringAttr>("bangc.implementation");
        bool propagating = implementation && implementation.getValue() == "propagating_extrema";
        line(std::string(propagating ? "intent_propagating_extrema<" : "intent_numeric_extrema<") +
            ctype(workspace.getElementType()) + ", " + count(binary.getOutput()) +
            ", " + std::to_string(workspace.getNumElements()) + ", " +
            (binary.getKind() == BinaryOperator::MaximumNum || binary.getKind() == BinaryOperator::Maximum ? "true" : "false") + ">(" + name(binary.getOutput()) +
            ", " + name(binary.getLhs()) + ", " + name(binary.getRhs()) + ", " + name(binary.getScratch()) + ");");
        return success();
      }
      if (cast<MemRefType>(binary.getOutput().getType()).getElementType().isInteger(64)) {
        auto callee = op->getAttrOfType<StringAttr>("bangc.callee");
        if (!callee) return op->emitError("unbound BANG integer tile operation");
        line(callee.getValue().str() + "(" + count(binary.getOutput()) + ", " + name(binary.getOutput()) + ", " +
            name(binary.getLhs()) + ", " + name(binary.getRhs()) + ");");
        return success();
      }
      if (binary.getKind() == BinaryOperator::TrueDivide) {
        if (op->getAttrOfType<StringAttr>("bangc.implementation").getValue() == "divide_f32") {
          line("intent_divide_f32<" + count(binary.getOutput()) + ", " + (binary.getApproximate() ? "true" : "false") + ", " +
              (binary.getFlushToZero() ? "true" : "false") + ">(" + name(binary.getOutput()) + ", " + name(binary.getLhs()) + ", " + name(binary.getRhs()) + ");");
          return success();
        }
        line("intent_divide_local<" + ctype(cast<MemRefType>(binary.getOutput().getType()).getElementType()) + ", " +
            count(binary.getOutput()) + ">(" + name(binary.getOutput()) + ", " + name(binary.getLhs()) + ", " + name(binary.getRhs()) + ");");
        return success();
      }
      std::string intrinsic;
      switch (binary.getKind()) {
      case BinaryOperator::Add: intrinsic = "__bang_add"; break;
      case BinaryOperator::Subtract: intrinsic = "__bang_sub"; break;
      case BinaryOperator::Multiply: intrinsic = "__bang_mul"; break;
      case BinaryOperator::Maximum:
      case BinaryOperator::Minimum: break; // The selected, verified callee supplies the native spelling.
      case BinaryOperator::MaximumNum: intrinsic = "__bang_maximum"; break;
      case BinaryOperator::MinimumNum: intrinsic = "__bang_minimum"; break;
      default: return op->emitError("unbound BANG binary operation");
      }
      if (!isa<MemRefType>(binary.getRhs().getType())) intrinsic += "_scalar";
      if (auto callee = op->getAttrOfType<StringAttr>("bangc.callee")) intrinsic = callee.getValue().str();
      line(intrinsic + "(" + name(binary.getOutput()) + ", " + name(binary.getLhs()) + ", " +
          name(binary.getRhs()) + ", " + count(binary.getOutput()) + ");"); return success();
    }
    if (auto unary = dyn_cast<dsa::UnaryOp>(op)) {
      if (Value scratch = unary.getScratch()) {
        if (cast<MemRefType>(scratch.getType()).getElementType().isF32()) {
          line("intent_exp_f32_tile<" + count(unary.getOutput()) + ", " +
              std::to_string(cast<MemRefType>(scratch.getType()).getDimSize(1)) +
              (unary.getKind() == UnaryOperator::Exp2 ? ", true>(" : ">(") +
              name(unary.getOutput()) + ", " + name(unary.getInput()) + ", " + name(scratch) + ");");
          return success();
        }
        line("intent_exp2_ftz_tile<" + count(unary.getOutput()) + ", " + count(scratch) +
            (unary->hasAttr("bangc.input_non_subnormal") ? ", true>(" : ">(") +
            name(unary.getOutput()) + ", " + name(unary.getInput()) + ", " + name(scratch) + ");");
        return success();
      }
      if (auto callee = op->getAttrOfType<StringAttr>("bangc.callee")) {
        line(callee.getValue().str() + "(" + count(unary.getOutput()) + ", " + name(unary.getOutput()) + ", " + name(unary.getInput()) + ");");
        return success();
      }
      if (auto implementation = op->getAttrOfType<StringAttr>("bangc.implementation");
          implementation && implementation.getValue() == "exp2_f32") {
        line("intent_exp2_tile<" + count(unary.getOutput()) + ", " + (unary.getFlushToZero() ? "true" : "false") + ">(" +
            name(unary.getOutput()) + ", " + name(unary.getInput()) + ");");
        return success();
      }
      std::string intrinsic;
      switch (unary.getKind()) {
      case UnaryOperator::Abs: intrinsic = "__bang_abs"; break;
      case UnaryOperator::Negate:
        line("__bang_mul_scalar(" + name(unary.getOutput()) + ", " + name(unary.getInput()) + ", " +
            ctype(cast<MemRefType>(unary.getInput().getType()).getElementType()) + "(-1), " + count(unary.getOutput()) + ");"); return success();
      default: return op->emitError("unbound BANG unary operation");
      }
      line(intrinsic + "(" + name(unary.getOutput()) + ", " + name(unary.getInput()) + ", " + count(unary.getOutput()) + ");"); return success();
    }
    if (auto castOp = dyn_cast<dsa::CastOp>(op)) {
      if (auto callee = op->getAttrOfType<StringAttr>("bangc.callee")) {
        line(callee.getValue().str() + "(" + count(castOp.getOutput()) + ", " + name(castOp.getOutput()) + ", " + name(castOp.getInput()) + ");");
        return success();
      }
      auto from = cast<MemRefType>(castOp.getInput().getType()).getElementType();
      auto to = cast<MemRefType>(castOp.getOutput().getType()).getElementType();
      if (from == to) line("__memcpy(" + name(castOp.getOutput()) + ", " + name(castOp.getInput()) + ", " +
          count(castOp.getOutput()) + " * sizeof(" + ctype(to) + "), NRAM2NRAM);");
      else if (from.isInteger(1) && to.isF32())
        line("intent_bool_to_f32_tile<" + count(castOp.getOutput()) + ">(" + name(castOp.getOutput()) + ", " +
            name(castOp.getInput()) + ");");
      else if ((from.isF16() && to.isF32()) || (from.isF32() && to.isF16()))
        line(std::string(to.isF32() ? "__bang_half2float(" : "__bang_float2half_rn(") + name(castOp.getOutput()) + ", " +
            name(castOp.getInput()) + ", " + count(castOp.getOutput()) + ");");
      else if ((from.isBF16() && to.isF32()) || (from.isF32() && to.isBF16()))
        line(std::string(to.isF32() ? "intent_bf16_to_f32_tile<" : "intent_f32_to_bf16_tile<") + count(castOp.getOutput()) + ">(" +
            name(castOp.getOutput()) + ", " + name(castOp.getInput()) + ");");
      else line("intent_cast_local<" + ctype(from) + ", " + ctype(to) + ", " + count(castOp.getOutput()) + ">(" +
          name(castOp.getOutput()) + ", " + name(castOp.getInput()) + ");");
      return success();
    }
    if (auto reduce = dyn_cast<dsa::ReduceOp>(op)) {
      auto input = cast<MemRefType>(reduce.getInput().getType());
      if (input.getElementType().isF16()) {
        line("intent_reduce_half_extrema<" + std::to_string(input.getDimSize(0)) + ", " +
            std::to_string(input.getDimSize(1)) + ", " + std::to_string(static_cast<int>(reduce.getKind())) + ">(" +
            name(reduce.getOutput()) + ", " + name(reduce.getInput()) + ", " + name(reduce.getScratch()) + ", " +
            name(reduce.getCount()) + ", " + name(reduce.getIdentity()) + ");");
        return success();
      }
      if (reduce.getAxis() == 0) {
        line("intent_reduce_rows<" + std::to_string(cast<MemRefType>(reduce.getInput().getType()).getDimSize(1)) + ">(" +
            name(reduce.getOutput()) + ", " + name(reduce.getInput()) + ", " + name(reduce.getScratch()) + ", " +
            name(reduce.getCount()) + ", " + name(reduce.getIdentity()) + ");");
        return success();
      }
      if (input.getDimSize(0) > 1) {
        auto scratch = cast<MemRefType>(reduce.getScratch().getType());
        if (input.getDimSize(0) < 32 && input.getDimSize(1) >= 1024 && scratch.getDimSize(1) == input.getDimSize(1)) {
          line("intent_reduce_row_tiles<" + std::to_string(input.getDimSize(0)) + ", " +
              std::to_string(input.getDimSize(1)) + ", " + std::to_string(static_cast<int>(reduce.getKind())) + ">(" +
              name(reduce.getOutput()) + ", " + name(reduce.getInput()) + ", " + name(reduce.getScratch()) + ", " +
              name(reduce.getCount()) + ", " + name(reduce.getIdentity()) + ");");
          return success();
        }
        std::string helper = reduce.getKind() == BinaryOperator::Add ? "intent_reduce_row_sums<" : "intent_reduce_row_extrema<";
        std::string kind = reduce.getKind() == BinaryOperator::Add ? "" : ", " + std::to_string(static_cast<int>(reduce.getKind()));
        line(helper + std::to_string(input.getDimSize(0)) + ", " +
            std::to_string(input.getDimSize(1)) + kind + ">(" + name(reduce.getOutput()) + ", " +
            name(reduce.getInput()) + ", " + name(reduce.getScratch()) + ", " +
            name(reduce.getCount()) + ", " + name(reduce.getIdentity()) + ");");
        return success();
      }
      line("intent_reduce<" + count(reduce.getInput()) + ", " + std::to_string(static_cast<int>(reduce.getKind())) + ">(" +
          name(reduce.getOutput()) + ", " + name(reduce.getInput()) + ", " + name(reduce.getScratch()) + ", " +
          name(reduce.getCount()) + ", " + name(reduce.getIdentity()) + ");"); return success();
    }
    if (auto divide = dyn_cast<dsa::DivideRNOp>(op)) {
      line("intent_divide_rn<" + count(divide.getOutput()) + ", " +
          std::to_string(cast<MemRefType>(divide.getScratch().getType()).getDimSize(1)) + ">(" +
          name(divide.getOutput()) + ", " + name(divide.getLhs()) + ", " + name(divide.getRhs()) + ", " +
          name(divide.getScratch()) + ", " + name(divide.getLaneIndices()) + ");");
      return success();
    }
    if (auto divide = dyn_cast<dsa::DivideCastOp>(op)) {
      line("intent_divide_cast_f16<" + count(divide.getOutput()) + ">(" +
          name(divide.getOutput()) + ", " + name(divide.getLhs()) + ", " + name(divide.getRhs()) + ", " +
          name(divide.getQuotient()) + ", " + name(divide.getBounds()) + ", " + name(divide.getAccepted()) + ", " +
          name(divide.getNarrowBounds()) + ", " + name(divide.getLaneIndices()) + ");"); return success();
    }
    if (auto prepare = dyn_cast<dsa::PrepareMatrixOp>(op)) {
      line("intent_prepare_matrix<" + shape(prepare.getOutput()) + ", " + (prepare.getInputTransposed() ? "true" : "false") + ">(" +
          name(prepare.getOutput()) + ", " + name(prepare.getInput()) + ", " +
          (prepare.getScratch() ? name(prepare.getScratch()) : "nullptr") + ", " +
          (prepare.getReshaped() ? name(prepare.getReshaped()) : "nullptr") + ");"); return success();
    }
    if (auto matrix = dyn_cast<dsa::MatrixTileOp>(op)) {
      auto a = cast<MemRefType>(matrix.getLhs().getType()), c = cast<MemRefType>(matrix.getAccumulator().getType());
      line("intent_matmul<" + ctype(a.getElementType()) + ", " + std::to_string(a.getDimSize(0)) + ", " +
          std::to_string(a.getDimSize(1)) + ", " + std::to_string(c.getDimSize(1)) +
          (matrix.getAccumulate() ? "" : ", false") + ">(" + name(matrix.getAccumulator()) + ", " +
          name(matrix.getLhs()) + ", " + name(matrix.getRhs()) + ");"); return success();
    }
    std::string expression;
    if (isa<dsa::TaskIdOp>(op)) expression = "taskId";
    else if (isa<dsa::TaskCountOp>(op)) expression = "taskDim";
    else if (isa<dsa::GroupIdOp>(op)) expression = "taskIdY";
    else if (isa<dsa::GroupCountOp>(op)) expression = "taskDimY";
    else if (isa<dsa::LocalIdOp>(op)) expression = "taskIdX";
    else if (isa<dsa::IsMemoryCoreOp>(op)) expression = "__is_mpu()";
    else if (auto scalar = dyn_cast<dsa::LoadScalarOp>(op)) expression = name(scalar.getSource()) + "[" + name(scalar.getOffset()) + "]";
    else if (isStandardScalarOperation(op)) {
      SmallVector<std::string> operands;
      for (Value value : op->getOperands()) operands.push_back(name(value));
      auto scalar = emitScalar(op, operands);
      if (mlir::failed(scalar)) return failure();
      expression = std::move(*scalar);
    }
    if (expression.empty() || op->getNumResults() != 1) return op->emitError("operation has no BANG C spelling");
    line(ctype(op->getResult(0).getType()) + " " + bind(op->getResult(0)) + " = " + expression + ";");
    return success();
  }
  func::FuncOp function;
  std::string deviceEntry, launchEntry;
  SmallVector<std::string> signature, call;
};

const OperationEmitters<Serializer> &Serializer::nativeEmitters() {
  static const auto handlers = [] {
    OperationEmitters<Serializer> table;
    auto metadataCheck = [](Operation *operation) {
      return NativeSourceEmitter::metadataEmitters().verify(operation);
    };
    auto metadataEmit = [](Operation *operation, Serializer &out) {
      return NativeSourceEmitter::metadataEmitters().emit(operation, out);
    };
    table.add<memref::DimOp>(metadataCheck, metadataEmit);
    table.add<memref::ReinterpretCastOp>(metadataCheck, metadataEmit);
    auto controlCheck = [](Operation *operation) -> LogicalResult {
      if (operation->getNumResults())
        return operation->emitOpError("BANG C control results require explicit state storage");
      if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
        auto condition = cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
        if (loop.getNumOperands() || !condition.getArgs().empty())
          return loop.emitOpError("BANG C while state requires explicit state storage");
      }
      return NativeSourceEmitter::controlEmitters().verify(operation);
    };
    auto controlEmit = [](Operation *operation, Serializer &out) {
      return NativeSourceEmitter::controlEmitters().emit(operation, out);
    };
    table.add<scf::ForOp>(controlCheck, controlEmit);
    table.add<scf::IfOp>(controlCheck, controlEmit);
    table.add<scf::WhileOp>(controlCheck, controlEmit);
    table.add<memref::AllocaOp>([](memref::AllocaOp allocation) -> LogicalResult {
      if (mlir::failed(verifyAllocationGeometry(allocation))) return failure();
      if (!allocation->getAttrOfType<IntegerAttr>("bangc.offset") ||
          !allocation->getAttrOfType<IntegerAttr>("bangc.allocation_bytes"))
        return allocation.emitOpError("BANG C allocation requires completed storage binding");
      return success();
    }, [](memref::AllocaOp allocation, Serializer &out) {
      auto type = allocation.getType();
      std::string buffer = type.getMemorySpaceAsInt() == dsa::matrixSpace ? "local_wram" :
          type.getMemorySpaceAsInt() == dsa::sharedSpace ? "group_sram" : "local_nram";
      std::string pointer = out.newName();
      out.line(out.nativeType(type) + pointer + " = reinterpret_cast<" + out.nativeType(type) +
          ">(" + buffer + " + " +
          std::to_string(allocation->getAttrOfType<IntegerAttr>("bangc.offset").getInt()) + ");");
      out.memories[allocation.getResult()] = out.allocationDescriptor(type, pointer, {});
      return success();
    });
    auto localAccess = [](Operation *operation, Value memory) -> LogicalResult {
      auto type = cast<MemRefType>(memory.getType());
      if (mlir::failed(verifyNativeMemoryType(operation, type))) return failure();
      if (type.getMemorySpaceAsInt() != dsa::nramSpace)
        return operation->emitOpError("BANG C scalar memref access requires NRAM storage; other spaces require their explicit DSA access operation");
      return success();
    };
    table.add<memref::LoadOp>([=](memref::LoadOp load) {
      return localAccess(load, load.getMemref());
    }, [](memref::LoadOp load, Serializer &out) {
      out.bindExpression(load.getResult(), "intent_read_local(" +
          out.memoryPointer(load.getMemref(), load.getIndices()) + ")");
      return success();
    });
    table.add<memref::StoreOp>([=](memref::StoreOp store) {
      return localAccess(store, store.getMemref());
    }, [](memref::StoreOp store, Serializer &out) {
      out.line("intent_write_local(" + out.memoryPointer(store.getMemref(), store.getIndices()) +
          ", " + out.valueString(store.getValue()) + ");");
      return success();
    });
    table.add<memref::CopyOp>([=](memref::CopyOp copy) -> LogicalResult {
      for (Value memory : {copy.getSource(), copy.getTarget()}) {
        if (mlir::failed(localAccess(copy, memory))) return failure();
        auto type = cast<MemRefType>(memory.getType());
        if (!type.hasStaticShape() || !type.getLayout().isIdentity())
          return copy.emitOpError("BANG C local copy requires static contiguous source and destination views");
      }
      return success();
    }, [](memref::CopyOp copy, Serializer &out) {
      out.line("intent_copy_local<" + ctype(cast<MemRefType>(copy.getSource().getType()).getElementType()) +
          ", " + out.count(copy.getSource()) + ">(" + out.memoryPointer(copy.getTarget()) +
          ", " + out.memoryPointer(copy.getSource()) + ");");
      return success();
    });
    table.add<dsa::StrideOp>([](dsa::StrideOp stride) {
      return verifyNativeMemoryType(stride, cast<MemRefType>(stride.getSource().getType()));
    }, [](dsa::StrideOp stride, Serializer &out) {
      out.SourceEmitter::bind(stride.getResult(), out.memories.at(stride.getSource()).strides[stride.getAxis()]);
      return success();
    });
    return table;
  }();
  return handlers;
}
}

std::optional<LogicalResult> verifyNativeSourceOperation(Operation *operation) {
  const auto &table = Serializer::nativeEmitters();
  if (!table.contains(operation)) return std::nullopt;
  return table.verify(operation);
}

std::optional<LogicalResult> verifyUnboundNativeSourceOperation(Operation *operation) {
  if (auto allocation = dyn_cast<memref::AllocaOp>(operation))
    return verifyAllocationGeometry(allocation);
  return verifyNativeSourceOperation(operation);
}

LogicalResult serializeProgram(ModuleOp module, std::string &source, std::string &metadata) {
  auto entry = module->getAttrOfType<StringAttr>("intent.entry_name");
  if (!entry)
    return module.emitError("BANG C portfolio requires a public entry and bound candidate modules");
  for (Operation &operation : *module.getBody())
    if (!isa<ModuleOp>(operation))
      return operation.emitError("BANG C portfolio contains an operation outside its candidate modules");
  auto options = module->getAttrOfType<CompileOptionsAttr>(compileOptionsAttr);
  if (!options)
    return module.emitError("BANG C portfolio requires its numerical compile options");
  if (module.getOps<ModuleOp>().empty())
    return module.emitError("BANG C portfolio has no bound candidates");
  IntegerAttr wramAlignment;
  for (ModuleOp candidate : module.getOps<ModuleOp>()) {
    if (candidate->getAttrOfType<CompileOptionsAttr>(compileOptionsAttr) != options)
      return candidate.emitError("BANG C candidate compile options disagree with the portfolio");
    if (mlir::failed(verifyProgram(candidate))) return failure();
    auto function = *candidate.getOps<func::FuncOp>().begin();
    auto alignment = function->getAttrOfType<IntegerAttr>("bangc.wram_align");
    if (!alignment)
      return function.emitError("BANG C candidate requires its bound WRAM alignment");
    if (wramAlignment && alignment.getInt() != wramAlignment.getInt())
      return function.emitError("BANG C candidates disagree on their bound WRAM alignment");
    wramAlignment = alignment;
  }
  llvm::raw_string_ostream output(source);
  output << "#pragma bang walign(" << wramAlignment.getInt() << ")\n"
         << tileImplementations << "\n";
  int64_t preludeEnd = output.tell();
  llvm::json::Object interface;
  llvm::json::Array candidates;
  unsigned ordinal = 0;
  for (ModuleOp candidate : module.getOps<ModuleOp>()) {
    int64_t begin = output.tell();
    llvm::json::Object description;
    Serializer serializer(*candidate.getOps<func::FuncOp>().begin(), output, ordinal++);
    if (mlir::failed(serializer.emit(description))) return failure();
    for (StringRef name : {"provider", "compile_options", "target", "interface", "native"}) {
      if (ordinal == 1) interface[name] = std::move(*description.get(name));
      else if (*interface.get(name) != *description.get(name))
        return candidate.emitError("BANG C candidates disagree on their public compilation contract: ") << name;
    }
    candidates.push_back(llvm::json::Object{
        {"entry", std::move(*description.get("entry"))},
        {"configuration", std::move(*description.get("configuration"))},
        {"full_extent_dimensions", std::move(*description.get("full_extent_dimensions"))},
        {"source_range", llvm::json::Array{begin, static_cast<int64_t>(output.tell())}}});
  }
  interface["entry_name"] = entry.getValue();
  interface["source_prelude_end"] = preludeEnd;
  interface["candidates"] = std::move(candidates);
  llvm::raw_string_ostream metadataOutput(metadata);
  metadataOutput << llvm::json::Value(std::move(interface));
  return success();
}
}
