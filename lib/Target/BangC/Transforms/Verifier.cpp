#include "PassDetail.h"
#include "../Serialization/Scalar.h"
#include "../Serialization/Surface.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
namespace intent::bangc {

LogicalResult verifyNativeProgram(ModuleOp module) {
  if (failed(dsa::verifyRealizedProgram(module))) return failure();
  auto architecture = module->getAttrOfType<StringAttr>("bangc.architecture");
  if (!architecture || architecture.getValue() != "mtp_372")
    return module.emitError("BANG C program requires the selected mtp_372 implementation profile");
  auto function = *module.getOps<func::FuncOp>().begin();
  if (failed(verifySurfaceOperations(function))) return failure();
  auto walk = function.walk([&](Operation *operation) {
    if (auto allocation = dyn_cast<memref::AllocaOp>(operation)) {
      auto space = allocation.getType().getMemorySpaceAsInt();
      if (space == dsa::matrixSpace) {
        auto layout = operation->getAttrOfType<StringAttr>("bangc.layout");
        if (!layout || layout.getValue() != "matrix_filter_interleaved64") {
          operation->emitError("BANG C matrix storage requires its selected filter layout");
          return WalkResult::interrupt();
        }
      }
    }
    if (isa<dsa::MatrixTileOp>(operation)) {
      auto implementation = operation->getAttrOfType<StringAttr>("bangc.implementation");
      if (!implementation || implementation.getValue() != "matmul_local_f32_accumulator") {
        operation->emitError("BANG C matrix tile has no selected accumulator implementation");
        return WalkResult::interrupt();
      }
    }
    if (auto binary = dyn_cast<dsa::BinaryOp>(operation)) {
      auto rhs = dyn_cast<MemRefType>(binary.getRhs().getType());
      auto implementation = operation->getAttrOfType<StringAttr>("bangc.implementation");
      if (binary.getKind() == BinaryOperator::Maximum || binary.getKind() == BinaryOperator::Minimum) {
        auto callee = operation->getAttrOfType<StringAttr>("bangc.callee");
        StringRef expected = binary.getKind() == BinaryOperator::Maximum ? "__bang_maxequal" : "__bang_minequal";
        if (binary.getScratch() ? (!implementation || implementation.getValue() != "propagating_extrema")
                                : (!callee || callee.getValue() != expected)) {
          binary.emitError("propagating extrema requires a selected MTP372 implementation");
          return WalkResult::interrupt();
        }
        if (binary.getScratch() && cast<MemRefType>(binary.getScratch().getType()).getNumElements() <
                                       cast<MemRefType>(binary.getOutput().getType()).getNumElements()) {
          dsa::StorageAnalysis storage(function);
          Value output = binary.getOutput(), origin = storage.uniqueOrigin(output);
          for (Value input : ValueRange{binary.getLhs(), binary.getRhs()})
            if (!storage.disjoint(input, output) && input != output &&
                !(origin && dsa::isCompleteStorageViewOf(input, origin) &&
                  dsa::isCompleteStorageViewOf(output, origin))) {
              binary.emitError("windowed propagating extrema requires disjoint or identical output storage");
              return WalkResult::interrupt();
            }
        }
      }
      bool compact = rhs && rhs != binary.getLhs().getType();
      bool selected = implementation && (implementation.getValue() == "cycle" ||
                                         implementation.getValue() == "row_scalar");
      if (compact || selected) {
        StringRef expected = compact && rhs.getDimSize(0) == 1 ? "cycle" : "row_scalar";
        dsa::StorageAnalysis storage(function);
        if (!compact || !selected || implementation.getValue() != expected ||
            !storage.disjoint(binary.getOutput(), binary.getRhs())) {
          binary.emitError("broadcast primitive requires the matching unit-axis operand and independent short input");
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}

LogicalResult verifyUnboundNativeProgram(ModuleOp module) {
  if (failed(verifyNativeProgram(module))) return failure();
  auto walk = module.walk([&](Operation *operation) {
    if (operation->hasAttr("bangc.offset") || operation->hasAttr("bangc.allocation_bytes") ||
        operation->hasAttr("bangc.nram_bytes") || operation->hasAttr("bangc.wram_bytes") ||
        operation->hasAttr("bangc.sram_bytes") || operation->hasAttr("bangc.wram_align")) {
      operation->emitError("BANG C storage reuse requires unbound storage; run it before storage binding");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}

LogicalResult verifyProgram(ModuleOp module) {
  if (failed(verifyNativeProgram(module))) return failure();
  auto function = *module.getOps<func::FuncOp>().begin();
  auto nram = function->getAttrOfType<IntegerAttr>("bangc.nram_bytes");
  auto wram = function->getAttrOfType<IntegerAttr>("bangc.wram_bytes");
  auto sram = function->getAttrOfType<IntegerAttr>("bangc.sram_bytes");
  if (!nram || !wram || !sram || nram.getInt() < 0 || wram.getInt() < 0 || sram.getInt() < 0)
    return function.emitError("BANG C program requires completed storage binding");
  auto walk = function.walk([&](memref::AllocaOp allocation) {
    auto offset = allocation->getAttrOfType<IntegerAttr>("bangc.offset");
    auto bytes = allocation->getAttrOfType<IntegerAttr>("bangc.allocation_bytes");
    auto space = allocation.getType().getMemorySpaceAsInt();
    int64_t banks = space == dsa::matrixSpace ? 16 : 1;
    int64_t capacity = space == dsa::matrixSpace ? wram.getInt()
        : space == dsa::sharedSpace ? sram.getInt() : nram.getInt();
    if (!offset || !bytes || offset.getInt() < 0 || bytes.getInt() < 0 ||
        bytes.getInt() > capacity || offset.getInt() > (capacity - bytes.getInt()) / banks) {
      allocation.emitError("BANG C allocation is missing or exceeds its bound storage interval");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}


LogicalResult verifySurfaceOperations(func::FuncOp function) {
  auto walk = function.walk([&](Operation *op) {
    auto supportedType = [](Type type) {
      if (auto memory = dyn_cast<MemRefType>(type))
        type = memory.getElementType();
      return scalarType(type).has_value();
    };
    for (Type type : llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes()))
      if (!supportedType(type)) {
        op->emitError("type has no BANG C representation: ") << type;
        return WalkResult::interrupt();
      }
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          if (!supportedType(argument.getType())) {
            op->emitError("block argument has no BANG C representation: ")
                << argument.getType();
            return WalkResult::interrupt();
          }
    if (isStandardScalarOperation(op))
      return failed(verifyScalar(op)) ? WalkResult::interrupt() : WalkResult::advance();
    if (auto native = verifyUnboundNativeSourceOperation(op))
      return failed(*native) ? WalkResult::interrupt() : WalkResult::advance();
    if (!isa<dsa::SynchronizeOp, dsa::GroupSynchronizeOp, dsa::GroupIdOp, dsa::GroupCountOp, dsa::LocalIdOp,
             dsa::IsMemoryCoreOp, dsa::StageTileOp, dsa::TaskIdOp, dsa::TaskCountOp, dsa::LoadScalarOp, dsa::StoreScalarOp, dsa::AtomicAddOp,
             dsa::LoadTileOp, dsa::GatherPlanOp, dsa::GatherRowsOp, dsa::GroupGatherRowsOp, dsa::StoreTileOp, dsa::FillOp, dsa::IotaOp,
             dsa::IndexLayoutOp, dsa::IndexBinaryOp, dsa::BroadcastRowsOp, dsa::TransposeOp, dsa::SelectOp, dsa::MaskedFillOp, dsa::UnaryOp, dsa::BinaryOp,
             dsa::CastOp, dsa::CompareOp, dsa::CompareRangeOp, dsa::CompareRampOp, dsa::DivideCastOp, dsa::DivideRNOp, dsa::ReduceOp,
             dsa::PrepareMatrixOp, dsa::PrepareMatrixViewOp, dsa::MatrixTileOp,
             scf::ConditionOp, scf::YieldOp, func::FuncOp, func::ReturnOp>(op)) {
      op->emitError("operation is outside the bound BANG C surface"); return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}
} // namespace intent::bangc
