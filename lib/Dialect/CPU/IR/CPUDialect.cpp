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
    llvm::function_ref<InFlightDiagnostic()> error, int64_t vectorBits,
    int64_t workers, int64_t privateBytes, bool) {
  if (vectorBits < 32 || vectorBits % 32 || workers <= 0 || privateBytes <= 0)
    return error() << "CPU capabilities require positive byte-addressable vector, worker and private-storage budgets";
  return success();
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
