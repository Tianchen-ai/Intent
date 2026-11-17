#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/ShapeRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace intent::cpu;

#include "Intent/Dialect/CPU/IR/CPUDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "Intent/Dialect/CPU/IR/CPUAttrs.cpp.inc"

void IntentCPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Intent/Dialect/CPU/IR/CPUAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Intent/Dialect/CPU/IR/CPUOps.cpp.inc"
      >();
}

namespace {

struct CanonicalViewDimension : OpRewritePattern<memref::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DimOp operation, PatternRewriter &rewriter) const override {
    auto argument = dyn_cast<BlockArgument>(operation.getSource());
    auto axis = operation.getConstantIndex();
    if (!argument || !axis) return failure();
    auto dimension = queryPublicDimension(argument, *axis);
    if (!dimension) return failure();
    if (dimension->constant) {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(operation, *dimension->constant);
      return success();
    }
    if (dimension->argument == argument && dimension->axis == *axis) return failure();
    rewriter.replaceOpWithNewOp<memref::DimOp>(operation, dimension->argument, dimension->axis);
    return success();
  }
};

}

void IntentCPUDialect::getCanonicalizationPatterns(RewritePatternSet &patterns) const {
  patterns.add<CanonicalViewDimension>(getContext());
}

LogicalResult EntryRequirementsAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, bool, bool disjointOutputs) {
  return disjointOutputs ? success()
      : error() << "CPU entry requires disjoint writable views";
}

LogicalResult CapabilitiesAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, VectorCapabilitiesAttr vector,
    ArrayAttr contractions, ExecutionConfigAttr execution) {
  if (!vector || !execution || !contractions)
    return error() << "CPU capabilities require vector facts, a contraction catalog and execution configuration";
  for (Attribute capability : contractions)
    if (!mlir::isa<ContractionCapabilityAttr>(capability))
      return error() << "CPU contraction capability must preserve operand and accumulator types";
  return success();
}
LogicalResult VectorCapabilitiesAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, int64_t minimumBits, bool) {
  if (minimumBits < 32 || minimumBits % 32)
    return error() << "CPU vector capability requires a byte-addressable minimum width";
  return success();
}
LogicalResult ExecutionConfigAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, int64_t workers, int64_t privateBytes) {
  if (workers <= 0 || privateBytes <= 0)
    return error() << "CPU execution requires positive worker and private-storage budgets";
  return success();
}
LogicalResult ContractionCapabilityAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, Type lhs, Type rhs, Type accumulator, bool) {
  if (!mlir::isa<IntegerType, FloatType>(lhs) || !mlir::isa<IntegerType, FloatType>(rhs) ||
      !mlir::isa<IntegerType, FloatType>(accumulator))
    return error() << "CPU contraction capability requires numeric element types";
  return success();
}
bool CapabilitiesAttr::supportsContraction(Type lhs, Type rhs, Type accumulator) const {
  return llvm::any_of(getContractions(), [&](Attribute attr) {
    auto capability = mlir::cast<ContractionCapabilityAttr>(attr);
    return capability.getLhs() == lhs && capability.getRhs() == rhs && capability.getAccumulator() == accumulator;
  });
}
int64_t CapabilitiesAttr::getVectorBits() const { return getVector().getMinimumBits(); }
int64_t CapabilitiesAttr::getWorkers() const { return getExecution().getWorkers(); }
int64_t CapabilitiesAttr::getPrivateBytes() const { return getExecution().getPrivateBytes(); }
bool CapabilitiesAttr::getMatrixI8I32() const {
  auto input = IntegerType::get(getContext(), 8);
  return supportsContraction(input, input, IntegerType::get(getContext(), 32));
}
bool CapabilitiesAttr::getMatrixBF16F32() const {
  auto input = BFloat16Type::get(getContext());
  return supportsContraction(input, input, Float32Type::get(getContext()));
}

LogicalResult ConfigurationAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error,
    int64_t grain, int64_t m, int64_t n, int64_t k, int64_t regionSize) {
  if (grain <= 0 || m <= 0 || n <= 0 || k <= 0 || regionSize <= 0)
    return error() << "CPU task/cache blocking requires positive extents";
  return success();
}

LogicalResult ImplementationAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, StringAttr name,
    DictionaryAttr parameters) {
  if (!name || name.empty() || !parameters)
    return error() << "CPU implementation requires a registered identity and explicit bindings";
  for (NamedAttribute parameter : parameters) {
    auto integer = mlir::dyn_cast<IntegerAttr>(parameter.getValue());
    if (!integer || !integer.getValue().isSignedIntN(64) || integer.getInt() <= 0)
      return error() << "CPU implementation parameter must be a positive integer";
  }
  return success();
}

LogicalResult ProfileRowAttr::verify(llvm::function_ref<InFlightDiagnostic()> error,
    ConfigurationAttr shared, DictionaryAttr local, ArrayAttr implementations) {
  if (!shared || !local || !implementations || implementations.empty())
    return error() << "CPU profile rows require complete shared/local parameters and implementations";
  for (NamedAttribute parameter : local) {
    auto integer = mlir::dyn_cast<IntegerAttr>(parameter.getValue());
    if (!integer || !integer.getValue().isSignedIntN(64) || integer.getInt() <= 0)
      return error() << "CPU profile parameters must be positive i64 values";
  }
  llvm::DenseSet<Attribute> names;
  for (Attribute item : implementations) {
    auto name = mlir::dyn_cast<StringAttr>(item);
    if (!name || name.empty() || !names.insert(name).second)
      return error() << "CPU profile implementations require distinct nonempty names";
  }
  return success();
}

LogicalResult TuningProfilesAttr::verify(llvm::function_ref<InFlightDiagnostic()> error,
    StringAttr provider, DictionaryAttr families) {
  if (!provider || provider.empty() || !families || families.empty())
    return error() << "CPU profiles require a provider and finite families";
  for (NamedAttribute family : families) {
    auto rows = mlir::dyn_cast<ArrayAttr>(family.getValue());
    if (family.getName().empty() || !rows || rows.empty() ||
        !llvm::all_of(rows, [](Attribute row) { return mlir::isa<ProfileRowAttr>(row); }))
      return error() << "CPU profile families require nonempty typed row arrays";
  }
  return success();
}

LogicalResult ReductionOrderAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, bool adjacentReassociation, bool elementPermutation) {
  if (elementPermutation && !adjacentReassociation)
    return error() << "CPU reduction element permutation requires reassociation permission";
  return success();
}

LogicalResult MicrotileAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, int64_t rows,
    int64_t columns, int64_t width) {
  if (rows <= 0 || columns <= 0 || width <= 0 || (width & (width - 1)) ||
      columns % width)
    return error() << "CPU register tile must contain complete positive issue groups";
  return success();
}
