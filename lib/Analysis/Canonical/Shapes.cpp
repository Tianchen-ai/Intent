#include "Intent/Analysis/CanonicalKernel.h"
#include "Intent/Analysis/ProductSchema.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "Intent/Interfaces/StructuredOpInterface.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include <functional>

using namespace mlir;

namespace intent {
namespace {

RankedTensorType tensorType(Type type, ArrayRef<unsigned> path = {}) {
  type = getProductComponentType(type, path);
  if (!type) return {};
  if (auto view = dyn_cast<ViewType>(type)) type = view.getTensor();
  if (auto buffer = dyn_cast<BufferType>(type)) type = buffer.getTensor();
  return dyn_cast<RankedTensorType>(type);
}

TensorExtentFact shapedLeaf(Value value, unsigned axis,
                           ArrayRef<unsigned> path = {}) {
  TensorExtentFact fact;
  fact.source = value;
  fact.axis = axis;
  fact.fieldPath.assign(path.begin(), path.end());
  return fact;
}

std::optional<int64_t> identity(Value value, unsigned axis,
                              ArrayRef<unsigned> path = {}) {
  auto type = tensorType(value.getType(), path);
  auto shape = type ? dyn_cast_or_null<TensorShapeAttr>(type.getEncoding())
                    : TensorShapeAttr();
  if (!shape || axis >= shape.getDimensions().size()) return std::nullopt;
  return shape.getDimensions()[axis];
}

bool sameFact(const TensorExtentFact &lhs, const TensorExtentFact &rhs) {
  if (!lhs.isKnown() || !rhs.isKnown()) return false;
  return lhs.constant == rhs.constant && lhs.value == rhs.value &&
         lhs.domain == rhs.domain && lhs.source == rhs.source &&
         lhs.axis == rhs.axis && lhs.fieldPath == rhs.fieldPath &&
         lhs.inferred == rhs.inferred;
}

class ExtentQuery {
public:
  explicit ExtentQuery(CanonicalKernelAnalysis &analysis,
      function_ref<bool(Value, ArrayRef<unsigned>, unsigned)> stop = nullptr)
      : analysis(analysis), stop(stop) {}

  TensorExtentFact scalar(Value value) {
    if (auto dim = value.getDefiningOp<DimOp>())
      return query(dim.getSource(), dim.getAxis());
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant)) && constant.isSignedIntN(64))
      return {constant.getSExtValue(), {}};
    if (auto constant = value.getDefiningOp<ConstantOp>())
      if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
        return {integer.getInt(), {}};
    return {std::nullopt, value};
  }

  TensorExtentFact query(Value value, unsigned axis,
                         ArrayRef<unsigned> path = {}) {
    if (!value) return {};
    auto leaf = shapedLeaf(value, axis, path);
    if (stop && stop(value, path, axis)) return leaf;
    for (const auto &entry : active)
      if (sameFact(entry, leaf)) return leaf;
    active.push_back(leaf);
    auto finish = llvm::make_scope_exit([&] { active.pop_back(); });

    // Product extraction preserves the selected component, not merely its type.
    if (auto extract = value.getDefiningOp<ExtractOp>()) {
      SmallVector<unsigned> nested{static_cast<unsigned>(
          extract->getAttrOfType<IntegerAttr>("field").getInt())};
      llvm::append_range(nested, path);
      return query(extract->getOperand(0), axis, nested);
    }
    if (!path.empty())
      if (auto *operation = value.getDefiningOp();
          isa_and_nonnull<MakeTupleOp, MakeRecordOp>(operation))
        return query(operation->getOperand(path.front()), axis, path.drop_front());

    Type type = getProductComponentType(value.getType(), path);
    if (!type) return {};
    if (auto product = value.getDefiningOp<DomainProductOp>()) {
      for (Value component : product.getDomains()) {
        unsigned rank = cast<DomainType>(component.getType()).getRank();
        if (axis < rank) return query(component, axis);
        axis -= rank;
      }
      return {};
    }
    if (isa<DomainType, RegionType>(type)) {
      unsigned rank = isa<DomainType>(type) ? cast<DomainType>(type).getRank()
                                          : cast<RegionType>(type).getRank();
      if (axis >= rank || !path.empty()) return {};
      TensorExtentFact result;
      result.domain = value;
      result.axis = axis;
      return result;
    }
    auto tensor = tensorType(type);
    if (!tensor || axis >= tensor.getRank()) return {};
    if (!tensor.isDynamicDim(axis)) return {tensor.getDimSize(axis), {}};

    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Operation *owner = argument.getOwner()->getParentOp();
      if (auto loop = dyn_cast<ForOp>(owner)) {
        auto carries = loop.getRegionIterArgs();
        auto found = llvm::find(carries, argument);
        if (found != carries.end())
          return query(loop.getInitArgs()[found - carries.begin()], axis, path);
      }
      if (auto loop = dyn_cast<WhileOp>(owner))
        return query(loop.getInitArgs()[argument.getArgNumber()], axis, path);
      if (auto structured = dyn_cast<StructuredOpInterface>(owner))
        for (const auto &relation : structured.getRegionArgumentRelations(
                 *argument.getOwner()->getParent())) {
          if (relation.to != argument) continue;
          if (relation.kind == StructuredRelationKind::SourceSlice &&
              llvm::is_contained(relation.axes, axis))
            return leaf;
          if (relation.kind == StructuredRelationKind::Capture ||
              relation.kind == StructuredRelationKind::SameSchema ||
              relation.kind == StructuredRelationKind::SourceSlice)
            return query(relation.from, axis, path);
        }
      return leaf;
    }

    auto result = cast<OpResult>(value);
    Operation *operation = result.getOwner();
    if (auto loop = dyn_cast<ForOp>(operation))
      return query(loop.getInitArgs()[result.getResultNumber()], axis, path);
    if (auto loop = dyn_cast<WhileOp>(operation))
      return query(loop.getInitArgs()[result.getResultNumber()], axis, path);
    if (auto branch = dyn_cast<IfOp>(operation)) {
      std::optional<TensorExtentFact> common;
      Value first;
      DominanceInfo dominance(operation->getParentOfType<func::FuncOp>());
      for (Region &region : branch->getRegions()) {
        if (region.empty() || region.front().empty()) return leaf;
        Value yielded = region.front().getTerminator()->getOperand(
            result.getResultNumber());
        if (first && !analysis.equalTensorExtents(first, axis, yielded, axis,
                                                  path, path))
          return leaf;
        if (!first) first = yielded;
        TensorExtentFact candidate = query(yielded, axis, path);
        Value origin = candidate.value ? candidate.value
                       : candidate.domain ? candidate.domain : candidate.source;
        if (!candidate.isKnown() ||
            (origin && !dominance.properlyDominates(origin, operation)))
          continue;
        // Shape equality is enough; branch tensor contents remain distinct.
        // Use only a representative available before the conditional.
        if (!common) common = candidate;
      }
      return common ? *common : leaf;
    }
    if (auto structured = dyn_cast<StructuredOpInterface>(operation)) {
      auto kind = structured.getStructuredKind();
      unsigned number = result.getResultNumber();
      if (kind == StructuredOpKind::RegionFold)
        return query(structured.getIdentities()[number], axis, path);
      if (kind == StructuredOpKind::RegionScan) {
        unsigned emitted = structured.getEmittedResults().size();
        if (number >= emitted)
          return query(structured.getInitialStates()[number - emitted], axis, path);
        auto member = analysis.emissionAxis(result, path);
        if (failed(member)) return {};
        if (axis == *member)
          return query(structured.getSources().front(),
                       structured.getIterationAxes().front());
        auto extent = query(structured.getEmitYields()[number], axis, path);
        // A helper-local value cannot be exported as an outer shape operand.
        Value source = extent.value ? extent.value
                       : extent.domain ? extent.domain : extent.source;
        if (source && !DominanceInfo(operation->getParentOfType<func::FuncOp>())
                           .properlyDominates(source, operation))
          return leaf;
        return extent;
      }
      Value source = structured.getSources()[number];
      unsigned sourceAxis = axis;
      if (kind == StructuredOpKind::Reduce) {
        auto sourceType = tensorType(source.getType(), path);
        if (!sourceType) return {};
        unsigned remaining = 0;
        auto reduced = structured.getIterationAxes();
        for (sourceAxis = 0; sourceAxis < sourceType.getRank(); ++sourceAxis)
          if (!llvm::is_contained(reduced, sourceAxis) && remaining++ == axis)
            break;
      }
      return query(source, sourceAxis, path);
    }
    if (!path.empty() && !isa<IndexedAccessOpInterface>(operation)) return leaf;
    if (auto indices = dyn_cast<IndicesOp>(operation))
      return query(indices.getSource(), axis);
    if (auto histogram = dyn_cast<HistogramOp>(operation))
      return scalar(histogram.getBins());
    if (auto quantize = dyn_cast<QuantizeOp>(operation))
      return query(quantize.getInput(), axis);
    if (auto random = dyn_cast<RandomBitsOp>(operation))
      return query(random.getLogicalCounter(), axis);

    ShapeRelationAttr shape;
    ValueRange operands = operation->getOperands();
    if (auto reshape = dyn_cast<ReshapeOp>(operation)) shape = reshape.getShape();
    else if (auto broadcast = dyn_cast<BroadcastOp>(operation)) shape = broadcast.getShape();
    else if (auto full = dyn_cast<FullOp>(operation)) shape = full.getShape();
    else if (auto buffer = dyn_cast<BufferOp>(operation)) {
      shape = buffer.getShape();
      operands = buffer.getExtents();
    }
    if (shape) {
      auto extent = cast<ShapeExprAttr>(shape.getAxes()[axis]);
      if (extent.getKind() == 0) return {extent.getPayload(), {}};
      if (extent.getKind() == 1) return scalar(operands[extent.getPayload()]);
      if (extent.getKind() == 2 && isa<ReshapeOp>(operation)) {
        leaf.inferred = true;
        return leaf;
      }
      return {};
    }
    if (isa<IndexedAccessOpInterface>(operation)) {
      auto access = analysis.indexRelation(operation);
      if (failed(access)) return {};
      for (const IndexTermFact &term : access->terms) {
        if (!llvm::is_contained(term.resultAxes, axis)) continue;
        if (term.kind == 4) return query(term.operands.front(), 0);
        if (term.kind == 0 && term.sourceAxis)
          return query(access->source, *term.sourceAxis);
        if (term.kind == 1) return {1, {}};
        if (term.kind == 3)
          for (auto [indexAxis, mapped] : llvm::enumerate(term.indexAxes))
            if (mapped == axis &&
                cast<RankedTensorType>(term.operands.front().getType())
                        .getDimSize(indexAxis) != 1)
              return query(term.operands.front(), indexAxis);
      }
      return leaf;
    }
    auto projections = analysis.operandProjections(result);
    if (succeeded(projections))
      for (const auto &projection : *projections)
        for (auto [sourceAxis, resultAxis] : llvm::enumerate(projection.resultAxes))
          if (resultAxis == axis) {
            Value source = operation->getOperand(projection.operandNumber);
            if (cast<RankedTensorType>(source.getType()).getDimSize(sourceAxis) != 1)
              return query(source, sourceAxis);
          }
    return leaf;
  }

private:
  CanonicalKernelAnalysis &analysis;
  function_ref<bool(Value, ArrayRef<unsigned>, unsigned)> stop;
  SmallVector<TensorExtentFact> active;
};

} // namespace

TensorExtentFact CanonicalKernelAnalysis::tensorExtent(
    Value value, unsigned axis, ArrayRef<unsigned> path,
    function_ref<bool(Value, ArrayRef<unsigned>, unsigned)> stop) {
  return ExtentQuery(*this, stop).query(value, axis, path);
}

bool CanonicalKernelAnalysis::equalTensorExtents(
    Value lhs, unsigned lhsAxis, Value rhs, unsigned rhsAxis,
    ArrayRef<unsigned> lhsPath, ArrayRef<unsigned> rhsPath) {
  auto left = tensorExtent(lhs, lhsAxis, lhsPath);
  auto right = tensorExtent(rhs, rhsAxis, rhsPath);
  if (left.constant || right.constant)
    return left.constant && right.constant && left.constant == right.constant;
  if (sameFact(left, right)) return true;
  // Identity is an equality fact, never a license to substitute another SSA
  // producer when materializing the dimension.
  auto leftID = identity(lhs, lhsAxis, lhsPath);
  auto rightID = identity(rhs, rhsAxis, rhsPath);
  return leftID && rightID && *leftID > 0 && leftID == rightID;
}

FailureOr<unsigned> CanonicalKernelAnalysis::emissionAxis(
    OpResult result, ArrayRef<unsigned> path) const {
  auto scan = dyn_cast<RegionScanOp>(result.getOwner());
  if (!scan || result.getResultNumber() >= scan.getEmittedResults().size())
    return failure();
  auto structured = cast<StructuredOpInterface>(scan.getOperation());
  Value slice = structured.getSummarizeSources().front();
  auto member = identity(slice, scan.getAxis());
  Value emitted = structured.getEmitYields()[result.getResultNumber()];
  auto type = tensorType(emitted.getType(), path);
  if (!member || !type) return failure();
  std::optional<unsigned> axis;
  for (unsigned current = 0; current < type.getRank(); ++current)
    if (identity(emitted, current, path) == member) {
      if (axis) return failure();
      axis = current;
    }
  if (!axis) return failure();
  return *axis;
}

FailureOr<SmallVector<LogicalReshapeGroup, 4>>
CanonicalKernelAnalysis::reshapeGroups(ReshapeOp reshape) {
  Value source = reshape.getInputs().front(), result = reshape.getResult();
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto resultType = cast<RankedTensorType>(result.getType());
  unsigned sourceRank = sourceType.getRank(), resultRank = resultType.getRank();
  struct Product {
    int64_t constant = 1;
    SmallVector<unsigned> dynamic;
    bool valid = true;
  };
  auto product = [&](Value value, unsigned begin, unsigned end) {
    Product shape;
    for (unsigned axis = begin; axis < end; ++axis) {
      auto extent = tensorExtent(value, axis);
      if (extent.constant) {
        int64_t next;
        if (llvm::MulOverflow(shape.constant, *extent.constant, next)) {
          shape.valid = false;
          break;
        }
        shape.constant = next;
      } else {
        shape.dynamic.push_back(axis);
      }
    }
    return shape;
  };
  auto equalProducts = [&](unsigned sb, unsigned se, unsigned rb, unsigned re) {
    Product left = product(source, sb, se), right = product(result, rb, re);
    if (!left.valid || !right.valid || left.constant != right.constant)
      return false;
    // Zero proves only this whole product equal, never any constituent axis.
    if (left.constant == 0) return true;
    if (left.dynamic.size() != right.dynamic.size()) return false;
    SmallVector<bool> used(right.dynamic.size());
    for (unsigned from : left.dynamic) {
      bool matched = false;
      for (auto [index, to] : llvm::enumerate(right.dynamic))
        if (!used[index] && equalTensorExtents(source, from, result, to)) {
          used[index] = true;
          matched = true;
          break;
        }
      if (!matched) return false;
    }
    return true;
  };
  SmallVector<LogicalReshapeGroup, 4> groups;
  auto append = [&](unsigned sb, unsigned se, unsigned rb, unsigned re) {
    LogicalReshapeGroup group;
    for (unsigned axis = sb; axis < se; ++axis) group.sourceAxes.push_back(axis);
    for (unsigned axis = rb; axis < re; ++axis) group.resultAxes.push_back(axis);
    groups.push_back(std::move(group));
  };
  // Search contiguous groups in increasing size. Failed suffixes are memoized;
  // equal-sized axes retain their actual occurrence rather than being permuted.
  DenseSet<std::pair<unsigned, unsigned>> failedSuffixes;
  std::function<bool(unsigned, unsigned)> solve = [&](unsigned sb, unsigned rb) {
    if (sb == sourceRank && rb == resultRank) return true;
    if (failedSuffixes.contains({sb, rb})) return false;
    for (unsigned count = 1; count <= sourceRank + resultRank - sb - rb; ++count)
      for (unsigned sc = 0; sc <= count && sb + sc <= sourceRank; ++sc) {
        unsigned rc = count - sc;
        if (rb + rc > resultRank || (sc == 0 && rc == 0)) continue;
        if (!equalProducts(sb, sb + sc, rb, rb + rc)) continue;
        append(sb, sb + sc, rb, rb + rc);
        if (solve(sb + sc, rb + rc)) return true;
        groups.pop_back();
      }
    failedSuffixes.insert({sb, rb});
    return false;
  };
  if (solve(0, 0)) return groups;
  // The verified reshape supplies total element-count equality, including its
  // inferred axis. Preserve explicit boundary unit axes before using that
  // equality for the unresolved core. Only factors of one may be cancelled;
  // interior units must stay in the core so each group remains contiguous.
  groups.clear();
  auto isUnit = [&](Value value, unsigned axis) {
    return tensorExtent(value, axis).constant == 1;
  };
  unsigned sb = 0, rb = 0, se = sourceRank, re = resultRank;
  while (sb < se || rb < re) {
    bool sourceUnit = sb < se && isUnit(source, sb);
    bool resultUnit = rb < re && isUnit(result, rb);
    if (!sourceUnit && !resultUnit) break;
    append(sb, sb + sourceUnit, rb, rb + resultUnit);
    sb += sourceUnit;
    rb += resultUnit;
  }
  SmallVector<LogicalReshapeGroup, 4> suffix;
  while (sb < se || rb < re) {
    bool sourceUnit = sb < se && isUnit(source, se - 1);
    bool resultUnit = rb < re && isUnit(result, re - 1);
    if (!sourceUnit && !resultUnit) break;
    LogicalReshapeGroup group;
    if (sourceUnit) group.sourceAxes.push_back(--se);
    if (resultUnit) group.resultAxes.push_back(--re);
    suffix.push_back(std::move(group));
  }
  if (sb != se || rb != re) append(sb, se, rb, re);
  for (auto &group : llvm::reverse(suffix))
    groups.push_back(std::move(group));
  return groups;
}

} // namespace intent
