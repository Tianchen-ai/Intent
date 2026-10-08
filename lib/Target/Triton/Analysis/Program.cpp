#include "Intent/Target/Triton/Analysis/Program.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Target/Triton/Analysis/Configuration.h"
#include "Intent/Target/Triton/IR/Configuration.h"
#include "Intent/Target/Triton/IR/Program.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::triton {
namespace {
bool isTritonScalarType(Type type) {
  if (type.isIndex() || isa<Float16Type, BFloat16Type, Float32Type,
                            Float64Type, Float8E4M3FNType,
                            Float8E5M2Type>(type))
    return true;
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth() == 1 || integer.getWidth() == 8 ||
           integer.getWidth() == 16 || integer.getWidth() == 32 ||
           integer.getWidth() == 64;
  return false;
}

bool isTritonDataType(Type type) {
  if (isTritonScalarType(type))
    return true;
  if (auto fragment = dyn_cast<gpu::FragmentType>(type))
    return isTritonScalarType(fragment.getElementType());
  if (auto record = dyn_cast<gpu::RecordType>(type)) {
    for (Attribute field : record.getFieldTypes())
      if (!isTritonDataType(cast<TypeAttr>(field).getValue()))
        return false;
    return true;
  }
  return false;
}

LogicalResult verifyAccess(gpu::AccessOpInterface access) {
  Operation *operation = access.getOperation();
  auto view = dyn_cast<gpu::ViewType>(access.getAccessResource().getType());
  auto argument = dyn_cast<BlockArgument>(access.getAccessResource());
  auto kernel = operation->getParentOfType<func::FuncOp>();
  if (!view || !argument || !kernel ||
      argument.getOwner() != &kernel.getBody().front())
    return operation->emitOpError(
        "Triton pointer access requires a kernel-entry view or workspace argument");
  for (unsigned index = 0; index < access.getAccessCoordinates().size(); ++index)
    if (!gpu::queryAccessCoordinateProjection(access, index).isExact())
      return operation->emitOpError(
          "Triton pointer access requires a complete coordinate projection");
  return success();
}

LogicalResult verifyKernel(
    func::FuncOp kernel,
    llvm::function_ref<LogicalResult(Operation *)> verifySource) {
  auto space = kernel->getAttrOfType<ArrayAttr>(gpu::programSpaceAttr);
  if (!space || space.empty() || space.size() > 3)
    return kernel.emitError(
        "Triton source surface requires a one-to-three dimensional grid");
  for (Attribute extent : space)
    if (!isTritonExpression(cast<gpu::PhysicalExprAttr>(extent)))
      return kernel.emitError(
          "Triton launch expression has no deterministic terminal spelling");

  for (Type type : kernel.getArgumentTypes()) {
    if (auto view = dyn_cast<gpu::ViewType>(type)) {
      if (!isTritonScalarType(view.getElementType()))
        return kernel.emitError("Triton view element type is unsupported");
      continue;
    }
    if (!isTritonScalarType(type) ||
        isa<Float8E4M3FNType, Float8E5M2Type>(type))
      return kernel.emitError("Triton physical ABI type is unsupported");
  }

  auto schema = ConfigurationSchema::read(kernel);
  if (failed(schema) || failed(verifyConfigurationRequirements(kernel)))
    return failure();

  WalkResult result = kernel.walk([&](Operation *operation) {
    if (isa<gpu::AtomicLoadOp>(operation)) {
      operation->emitOpError(
          "has no Triton atomic-load primitive with preserved memory-order semantics");
      return WalkResult::interrupt();
    }
    if (auto scatter = dyn_cast<gpu::ScatterReduceOp>(operation)) {
      scatter.emitOpError(
          "requires an add combine and Triton-native atomic-add dtype legalization");
      return WalkResult::interrupt();
    }
    if (isa<gpu::BufferOp>(operation)) {
      operation->emitOpError(
          "requires provider-local mutable-buffer realization before Triton serialization");
      return WalkResult::interrupt();
    }
    if (isa<gpu::SparseContractOp>(operation)) {
      operation->emitOpError(
          "has no native Triton structured-sparse contraction surface");
      return WalkResult::interrupt();
    }
    for (Type type : operation->getResultTypes()) {
      if (auto fragment = dyn_cast<gpu::FragmentType>(type)) {
        if (!isTritonScalarType(fragment.getElementType()) ||
            !llvm::all_of(fragment.getShape(), isTritonFragmentExtent)) {
          operation->emitOpError(
              "has no statically blocked Triton fragment representation");
          return WalkResult::interrupt();
        }
      } else if (isa<gpu::RecordType>(type)) {
        if (!isTritonDataType(type)) {
          operation->emitOpError("contains a record field outside Triton data types");
          return WalkResult::interrupt();
        }
      } else if (!isa<gpu::ViewType, gpu::RangeType>(type) &&
                 !isTritonScalarType(type)) {
        operation->emitOpError("has a result type outside the Triton surface");
        return WalkResult::interrupt();
      }
    }
    if (auto access = dyn_cast<gpu::AccessOpInterface>(operation)) {
      switch (access.getAccessKind()) {
      case gpu::AccessKind::AtomicCompareExchange:
        if (access.getAccessValidity()) {
          operation->emitOpError(
              "Triton tl.atomic_cas has no mask and cannot preserve physical validity");
          return WalkResult::interrupt();
        }
        [[fallthrough]];
      case gpu::AccessKind::Load:
      case gpu::AccessKind::Store:
      case gpu::AccessKind::AtomicStore:
      case gpu::AccessKind::AtomicRMW:
        if (failed(verifyAccess(access))) return WalkResult::interrupt();
        break;
      default:
        // Pure fragment gathers and unsupported atomic/scatter kinds retain
        // their provider-specific legality checks.
        break;
      }
    }
    if (auto contract = dyn_cast<gpu::ContractOp>(operation)) {
      std::string reason;
      auto axes = gpu::queryContractionAxes(contract, &reason);
      if (!axes) {
        contract.emitOpError("invalid Triton contraction axis schema: ") << reason;
        return WalkResult::interrupt();
      }
      if (!axes->hasCanonicalMatrixAxes()) {
        contract.emitOpError(
            "requires provider legalization to [...,M,K] x [...,K,N] tl.dot form");
        return WalkResult::interrupt();
      }
      auto form = contract->getAttrOfType<StringAttr>(contractFormAttr);
      if (form && form.getValue() == "fma" &&
          !contract.getAccumulator().getType().getElementType().isF32()) {
        contract.emitOpError(
            "Triton explicit FMA contraction requires f32 accumulation");
        return WalkResult::interrupt();
      }
      if ((!form || form.getValue() != "multiply_sum") &&
          isa<BFloat16Type>(contract.getResult().getType().getElementType())) {
        contract.emitOpError(
            "Triton tl.dot does not support a bfloat16 accumulator; requires precision-preserving provider legalization");
        return WalkResult::interrupt();
      }
    } else if (auto range = dyn_cast<gpu::MakeRangeOp>(operation)) {
      auto fragment = dyn_cast<gpu::FragmentType>(range.getResult().getType());
      auto physicalExtent =
          fragment && fragment.getShape().size() == 1
              ? dyn_cast<gpu::PhysicalExprAttr>(fragment.getShape()[0])
              : gpu::PhysicalExprAttr();
      if (!physicalExtent || !isTritonFragmentExtent(physicalExtent)) {
        InFlightDiagnostic diagnostic = range.emitOpError(
            "Triton tl.arange physical extent must be a compile-time physical expression");
        diagnostic << "; source_id=" << range.getSourceId();
        if (fragment && fragment.getAxisMaps().size() == 1)
          diagnostic << ", source_dimension="
                     << cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[0])
                            .getDimensionId();
        unsigned loopDepth = 0;
        for (Operation *parent = range->getParentOp(); parent;
             parent = parent->getParentOp())
          loopDepth += isa<scf::ForOp>(parent);
        diagnostic << ", loop_depth=" << loopDepth
                   << ", physical_extent=" << physicalExtent;
        diagnostic << (range.getResult().use_empty() ? " and is dead"
                                                      : " and still has live uses");
        for (Operation *user : range.getResult().getUsers()) {
          diagnostic << " " << user->getName();
          if (auto load = dyn_cast<gpu::LoadOp>(user)) {
            if (auto view = dyn_cast<gpu::ViewType>(load.getResource().getType()))
              diagnostic << "(source=" << view.getSourceId() << ",source_axes=["
                         << load.getSourceAxes() << "])";
            for (Operation *loadUser : load.getResult().getUsers()) {
              diagnostic << "->" << loadUser->getName();
              for (Value result : loadUser->getResults())
                for (Operation *resultUser : result.getUsers()) {
                  diagnostic << "->" << resultUser->getName();
                  for (Value nextResult : resultUser->getResults())
                    for (Operation *nextUser : nextResult.getUsers()) {
                      diagnostic << "->" << nextUser->getName();
                      if (auto reduction = dyn_cast<gpu::ReduceOp>(nextUser))
                        diagnostic << "(axes=[" << reduction.getAxes()
                                   << "],sources="
                                   << reduction.getSources().size() << ")";
                    }
                }
            }
          }
        }
        return WalkResult::interrupt();
      }
    } else if (auto gather = dyn_cast<gpu::GatherOp>(operation)) {
      if (gather.getValid()) {
        gather.emitOpError(
            "requires safe-index legalization before Triton tl.gather: source=")
            << gather.getSource().getType()
            << ", coordinate=" << gather.getCoordinates().front().getType()
            << ", valid=" << gather.getValid().getType()
            << ", fill=" << gather.getFill().getType()
            << ", result=" << gather.getResult().getType();
        return WalkResult::interrupt();
      }
      if (gather.getCoordinates().size() != 1 ||
          gather.getSourceAxes().size() != 1) {
        gather.emitOpError(
            "requires provider legalization to one-axis tl.gather");
        return WalkResult::interrupt();
      }
      if (!isa<gpu::FragmentType>(gather.getResult().getType()) &&
          (cast<gpu::FragmentType>(gather.getSource().getType()).getShape().size() != 1 ||
           gather.getSourceAxes().front() != 0 ||
           !isa<IndexType, IntegerType>(gather.getCoordinates().front().getType()))) {
        gather.emitOpError("Triton scalar gather requires one scalar index into a vector");
        return WalkResult::interrupt();
      }
    } else if (isa<gpu::ReduceOp, gpu::ScanOp>(operation)) {
      operation->emitOpError(
          "Triton collective requires an explicitly scalarized callback");
      return WalkResult::interrupt();
    } else if (auto contract = dyn_cast<gpu::ScaledContractOp>(operation)) {
      unsigned lhsRank = contract.getLhs().getType().getShape().size();
      unsigned rhsRank = contract.getRhs().getType().getShape().size();
      Type lhsScaleElement =
          contract.getLhsScale().getType().getElementType();
      Type rhsScaleElement =
          contract.getRhsScale().getType().getElementType();
      if (lhsRank != 3 || rhsRank != 3 ||
          contract.getLhsFormat() == ScaledFormat::E8M0 ||
          contract.getRhsFormat() == ScaledFormat::E8M0 ||
          contract.getLhsGroupSize() != 32 ||
          contract.getRhsGroupSize() != 32 ||
          !lhsScaleElement.isUnsignedInteger(8) ||
          !rhsScaleElement.isUnsignedInteger(8) ||
          !contract.getResult().getType().getElementType().isF32() ||
          contract.getLhsReductionAxes() !=
              ArrayRef<int64_t>{1, 2} ||
          contract.getRhsReductionAxes() !=
              ArrayRef<int64_t>{0, 1}) {
        contract.emitOpError(
            "is outside Triton tl.dot_scaled rank/format/e8m0-scale/group/axis legality");
        return WalkResult::interrupt();
      }
    } else if (auto histogram = dyn_cast<gpu::HistogramOp>(operation)) {
      if (!histogram.getBins().getDefiningOp<arith::ConstantOp>() &&
          !histogram.getBins().getDefiningOp<gpu::ParameterOp>()) {
        histogram.emitOpError(
            "Triton histogram bin count must be compile-time bound");
        return WalkResult::interrupt();
      }
    }

    return succeeded(verifySource(operation))
        ? WalkResult::advance() : WalkResult::interrupt();
  });
  return result.wasInterrupted() ? failure() : success();
}

} // namespace

LogicalResult verifyTritonProgram(
    ModuleOp module,
    llvm::function_ref<LogicalResult(Operation *)> verifySource) {
  if (failed(mlir::verify(module.getOperation())))
    return failure();
  SmallVector<func::FuncOp> kernels;
  WalkResult attributes = module.walk([&](Operation *operation) {
    if (failed(verifyProgramAttributes(operation)))
      return WalkResult::interrupt();
    if (auto function = dyn_cast<func::FuncOp>(operation);
        function && function->hasAttr(gpu::kernelAttr))
      kernels.push_back(function);
    return WalkResult::advance();
  });
  if (attributes.wasInterrupted())
    return failure();
  if (kernels.size() != 1)
    return module.emitError("Triton provider program requires one physical kernel");
  return verifyKernel(kernels.front(), verifySource);
}

} // namespace intent::triton
