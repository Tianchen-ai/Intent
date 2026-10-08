#include "ComputeForms.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Transforms/Value/Helpers.h"
#include "Intent/Target/CuTile/Analysis/Program.h"
#include "Intent/Target/CuTile/Analysis/IndexBounds.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/APFloat.h"
#include <numeric>

using namespace mlir;
namespace intent::cutile {
std::optional<BinaryOperator> nativeCombineKind(Region &region) {
  auto combine = gpu::queryBinaryCombine(region);
  if (!combine)
    return std::nullopt;
  if (combine->operation.getStrictRounding())
    return std::nullopt;
  // Every selected native kind below is commutative and has no non-default
  // math mode. Other combines retain the actual ordered callback operations.
  std::optional<BinaryOperator> kind = combine->kind();
  if (*kind == BinaryOperator::Add || *kind == BinaryOperator::MaximumNum ||
      *kind == BinaryOperator::MinimumNum)
    return kind;
  Type element = region.front().getArgument(0).getType();
  if (auto fragment = dyn_cast<gpu::FragmentType>(element))
    element = fragment.getElementType();
  if (element.isIntOrIndex()) {
    if (*kind == BinaryOperator::Maximum)
      return BinaryOperator::MaximumNum;
    if (*kind == BinaryOperator::Minimum)
      return BinaryOperator::MinimumNum;
  }
  if (element.isInteger(1)) {
    if (*kind == BinaryOperator::LogicalOr)
      return BinaryOperator::LogicalOr;
    if (*kind == BinaryOperator::LogicalAnd)
      return BinaryOperator::LogicalAnd;
  }
  return std::nullopt;
}

bool isNativeReductionIdentity(BinaryOperator kind, Value identity) {
  Attribute constant = getCompileTimeScalar(identity);
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant)) {
    switch (kind) {
    case BinaryOperator::Add:
    case BinaryOperator::LogicalOr:
      return integer.getValue().isZero();
    case BinaryOperator::LogicalAnd:
      return integer.getValue().isAllOnes();
    default:
      return false;
    }
  }
  auto floating = dyn_cast<FloatAttr>(constant);
  if (!floating)
    return false;
  const llvm::APFloat &value = floating.getValue();
  if (kind == BinaryOperator::Add)
    return value.isZero();
  if (!value.isInfinity())
    return false;
  if (kind == BinaryOperator::MaximumNum || kind == BinaryOperator::Maximum)
    return value.isNegative();
  if (kind == BinaryOperator::MinimumNum || kind == BinaryOperator::Minimum)
    return !value.isNegative();
  return false;
}

LogicalResult formComputePrimitives(func::FuncOp kernel,
    const NativeComputeInputs &inputs, NativeFormRewriter &rewriter) {
  auto checkCollectiveResults = [](Operation *operation, ValueRange sources,
                                   int64_t axis, bool scan) -> LogicalResult {
    // Preserve a diagnostic for unsupported native schemas: the ODS inferred
    // builder assumes inference succeeds once the provider selects the form.
    SmallVector<Type> resultTypes;
    if (failed(gpu::inferScalarCollectiveResultTypes(
            operation->getLoc(), sources, axis, scan, resultTypes)))
      return failure();
    if (!llvm::equal(resultTypes, operation->getResultTypes()))
      return operation->emitOpError(
          "native collective cannot preserve the shared result schema");
    return success();
  };
  for (gpu::HistogramOp histogram : inputs.histograms) {
    auto source = cast<gpu::FragmentType>(histogram.getValues().getType());
    auto result = cast<gpu::FragmentType>(histogram.getResult().getType());
    if (source.getShape().size() != 1)
      return histogram.emitOpError(
          "cuTile histogram requires a realized one-axis input fragment");
    OpBuilder builder(histogram);
    Location location = histogram.getLoc();
    auto sourceInteger = dyn_cast<IntegerType>(source.getElementType());
    Type comparisonElement = sourceInteger && sourceInteger.isUnsigned()
                                 ? IntegerType::get(kernel.getContext(), 64,
                                                    IntegerType::Unsigned)
                                 : builder.getI64Type();
    auto binMap = cast<gpu::AxisMapAttr>(result.getAxisMaps()[0]);
    auto inputMap = cast<gpu::AxisMapAttr>(source.getAxisMaps()[0]);
    auto inputAxis = gpu::AxisMapAttr::get(
        kernel.getContext(), inputMap.getSourceId(), inputMap.getSourceAxis(),
        inputMap.getDimensionId(), 1, inputMap.getDerived());
    auto shape = builder.getArrayAttr(
        {result.getShape()[0], source.getShape()[0]});
    auto maps = builder.getArrayAttr({binMap, inputAxis});
    auto relation = [&](Type element) {
      return gpu::FragmentType::get(kernel.getContext(), element, shape, maps,
                                    result.getValidity(), result.getOwner());
    };
    auto binType = gpu::FragmentType::get(
        kernel.getContext(), builder.getIndexType(), result.getShape(),
        result.getAxisMaps(), result.getValidity(), result.getOwner());
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    Value bins = builder.create<gpu::MakeRangeOp>(
        location, binType, zero, histogram.getBins(), one, zero,
        histogram.getBins(), binMap.getSourceId(), binMap.getSourceAxis(),
        binMap.getDerived());
    bins = builder.create<gpu::CastOp>(
        location, withElementType(binType, comparisonElement), bins);
    Value input = builder.create<gpu::CastOp>(
        location, withElementType(source, comparisonElement),
        histogram.getValues());
    Value values = builder.create<gpu::BroadcastOp>(
        location, relation(comparisonElement), input);
    Value selectedBins = builder.create<gpu::BroadcastOp>(
        location, relation(comparisonElement), bins);
    Value members = builder.create<gpu::CompareOp>(
        location, relation(builder.getI1Type()), values, selectedBins,
        ComparePredicate::Eq);
    if (histogram.getValid()) {
      Value valid = builder.create<gpu::BroadcastOp>(
          location, relation(builder.getI1Type()), histogram.getValid());
      members = builder.create<gpu::BinaryOp>(
          location, relation(builder.getI1Type()), members, valid,
          BinaryOperator::LogicalAnd);
    }
    Value counts = builder.create<gpu::CastOp>(
        location, relation(result.getElementType()), members);
    auto replacement = builder.create<ReduceOp>(
        location, ValueRange{counts}, ValueRange{}, 1, false,
        BinaryOperatorAttr::get(kernel.getContext(), BinaryOperator::Add));
    if (Attribute origin = histogram->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    rewriter.replace(histogram, ValueRange{replacement.getResult(0)});
  }

  for (gpu::ReduceOp reduce : inputs.reductions) {
    auto sourceType = cast<gpu::FragmentType>(reduce.getSources().front().getType());
    bool allAxes = sourceType.getShape().size() > 1 &&
        reduce.getAxes().size() == sourceType.getShape().size();
    if (!reduce.getCaptures().empty() || (!allAxes && reduce.getAxes().size() != 1))
      return reduce.emitOpError(
          "cuTile reduce requires one axis or a full reduction and no captures");
    SmallVector<Value> sources(reduce.getSources());
    int64_t axis = reduce.getAxes().front();
    if (allAxes) {
      if (failed(checkCollectiveResults(reduce, sources, -1, false)))
        return failure();
      // A full reduction has no retained coordinate to preserve. Flatten the
      // complete tuple in one common order, then use the existing scalar tree.
      // This is target IR construction, not a serializer layout decision.
      auto capacity = gpu::fragmentElementCount(sourceType);
      if (!gpu::queryPositiveExtentBounds(capacity, kernel))
        return reduce.emitOpError("whole reduction extent has no finite representable bound");
      if (auto count = gpu::constantPhysicalExpression(capacity))
        capacity = gpu::PhysicalExprAttr::get(kernel.getContext(),
            gpu::PhysicalExprKind::Constant, *count,
            StringAttr::get(kernel.getContext(), ""),
            ArrayAttr::get(kernel.getContext(), {}));
      auto [sourceId, dimensionId] = gpu::nextPhysicalAxisIdentities(kernel);
      SmallVector<int64_t> axes(sourceType.getShape().size());
      std::iota(axes.begin(), axes.end(), 0);
      OpBuilder builder(reduce);
      auto groups = builder.getArrayAttr({gpu::ReshapeGroupAttr::get(kernel.getContext(),
          builder.getDenseI64ArrayAttr(axes), builder.getDenseI64ArrayAttr({0}))});
      auto shape = builder.getArrayAttr({capacity});
      auto mappings = builder.getArrayAttr({gpu::AxisMapAttr::get(
          kernel.getContext(), sourceId, 0, dimensionId, 0, true)});
      for (Value &source : sources) {
        auto type = cast<gpu::FragmentType>(source.getType());
        auto flat = gpu::FragmentType::get(kernel.getContext(), type.getElementType(),
            shape, mappings, type.getValidity(), type.getOwner());
        source = builder.create<gpu::ReshapeOp>(reduce.getLoc(), flat, source, groups);
      }
      axis = 0;
    }
    if (failed(checkCollectiveResults(reduce, sources, axis, false)))
      return failure();
    std::optional<BinaryOperator> kind = nativeCombineKind(reduce.getCombine());
    bool native = reduce.getSources().size() == 1 && kind.has_value();
    if (native) {
      // Shared realization has already filled inactive lanes with the declared
      // identity. As with tl.reduce, the native tree consumes those values;
      // it does not require that identity to be a compile-time literal.
      OpBuilder builder(reduce);
      auto replacement = builder.create<ReduceOp>(
          reduce.getLoc(), sources, ValueRange{}, axis,
          false, BinaryOperatorAttr::get(kernel.getContext(), *kind));
      if (Attribute origin = reduce->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      rewriter.replace(reduce, replacement.getResults());
      continue;
    }

    auto canonicalCombine = gpu::queryBinaryCombine(reduce.getCombine());
    bool propagatingMaximum =
        reduce.getSources().size() == 1 && canonicalCombine &&
        canonicalCombine->kind() == BinaryOperator::Maximum &&
        isNativeReductionIdentity(
            BinaryOperator::Maximum,
            reduce.getIdentities().front());
    if (propagatingMaximum) {
      auto sourceType =
          dyn_cast<gpu::FragmentType>(sources.front().getType());
      if (!sourceType || !isa<FloatType>(sourceType.getElementType()))
        return reduce.emitOpError(
            "cuTile propagating maximum reduction requires a floating tile");
      OpBuilder builder(reduce);
      Location location = reduce.getLoc();
      Type sourcePredicateType =
          withElementType(sourceType, builder.getI1Type());
      auto isNan = builder.create<gpu::CompareOp>(
          location, sourcePredicateType, sources.front(),
          sources.front(), ComparePredicate::Ne);
      auto anyNan = builder.create<ReduceOp>(
          location, ValueRange{isNan.getResult()},
          ValueRange{}, axis, false,
          BinaryOperatorAttr::get(kernel.getContext(), BinaryOperator::LogicalOr));
      auto numericMaximum = builder.create<ReduceOp>(
          location, sources,
          ValueRange{}, axis, false,
          BinaryOperatorAttr::get(kernel.getContext(), BinaryOperator::MaximumNum));
      auto floatType = cast<FloatType>(sourceType.getElementType());
      auto nan = builder.create<arith::ConstantOp>(
          location, floatType,
          builder.getFloatAttr(
              floatType,
              llvm::APFloat::getNaN(floatType.getFloatSemantics())));
      Value nanValue = nan.getResult();
      Type resultType = numericMaximum.getResult(0).getType();
      if (isa<gpu::FragmentType>(resultType))
        nanValue = builder.create<gpu::SplatOp>(location, resultType, nanValue);
      auto replacement = builder.create<gpu::SelectOp>(
          location, resultType, anyNan.getResult(0), nanValue,
          numericMaximum.getResult(0));
      if (Attribute origin = reduce->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      rewriter.replace(reduce, ValueRange{replacement.getResult()});
      continue;
    }

    OpBuilder builder(reduce);
    auto replacement = builder.create<ReduceOp>(
        reduce.getLoc(), sources,
        reduce.getIdentities(), axis, false,
        BinaryOperatorAttr());
    if (failed(gpu::scalarizeElementwiseCallback(reduce.getCombine(),
                                                 replacement.getCombine())))
      return failure();
    if (Attribute origin = reduce->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    rewriter.replace(reduce, replacement.getResults());
  }

  for (gpu::ScanOp scan : inputs.scans) {
    if (!scan.getCaptures().empty())
      return scan.emitOpError("cuTile scan lowering requires no captures");
    std::optional<BinaryOperator> kind = nativeCombineKind(scan.getCombine());
    bool native = scan.getSources().size() == 1 && kind &&
                  *kind == BinaryOperator::Add &&
                  isNativeReductionIdentity(*kind, scan.getIdentities().front());
    Type element = gpu::uniformElementType(scan.getSources().front().getType());
    bool exclusiveSum = !scan.getInclusive() && native &&
                        isa<IntegerType, IndexType>(element) &&
                        !element.isInteger(1);
    if (!scan.getInclusive() && !exclusiveSum)
      return scan.emitOpError(
          "cuTile exclusive scan requires an additive integer prefix");
    if (failed(checkCollectiveResults(scan, scan.getSources(), scan.getAxis(), true)))
      return failure();
    OpBuilder builder(scan);
    auto replacement = builder.create<ScanOp>(
        scan.getLoc(), scan.getSources(),
        native ? ValueRange{} : ValueRange(scan.getIdentities()),
        scan.getAxis(), scan.getReverse(),
        native ? BinaryOperatorAttr::get(kernel.getContext(), *kind)
               : BinaryOperatorAttr());
    if (!native &&
        failed(gpu::scalarizeElementwiseCallback(scan.getCombine(),
                                                replacement.getCombine())))
      return failure();
    if (Attribute origin = scan->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    SmallVector<Value> results(replacement.getResults());
    if (exclusiveSum)
      // Modular integer addition has an exact inverse. This also preserves
      // reverse prefixes; floating subtraction is not an equivalent rewrite.
      results[0] = builder.create<gpu::BinaryOp>(
          scan.getLoc(), results[0].getType(), results[0], scan.getSources().front(),
          BinaryOperator::Subtract);
    rewriter.replace(scan, results);
  }

  for (gpu::ContractOp contract : inputs.contracts) {
    std::string reason;
    auto axes = gpu::queryContractionAxes(contract, &reason);
    if (!axes)
      return contract.emitOpError("invalid cuTile contraction axis schema: ") << reason;
    if (!axes->hasCanonicalMatrixAxes() || axes->lhsResultAxes.size() > 3)
      return contract.emitOpError(
          "cuTile native MMA requires [M,K] x [K,N] or [B,M,K] x [B,K,N] "
          "physical axes");
    OpBuilder builder(contract);
    Attribute origin = contract->getAttr(gpu::originAttr);
    auto inheritOrigin = [&](Operation *operation) {
      if (origin)
        operation->setAttr(gpu::originAttr, origin);
    };
    auto replacement = builder.create<MMAOp>(
        contract.getLoc(), contract.getResult().getType(), contract.getLhs(),
        contract.getRhs(), contract.getAccumulator(), IntegerAttr());
    inheritOrigin(replacement);
    rewriter.replace(contract, ValueRange{replacement.getResult()});
  }

  for (gpu::ScaledContractOp contract : inputs.scaledContracts) {
    if (failed(formScaledMMA(kernel, contract, rewriter)))
      return failure();
  }

  return success();
}
} // namespace intent::cutile
