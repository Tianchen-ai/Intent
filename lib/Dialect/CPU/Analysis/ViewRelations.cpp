#include "Intent/Dialect/CPU/Analysis/ViewRelations.h"
#include "Intent/Analysis/ControlFlow.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::cpu {
namespace {

using Variable = ValueBoundsConstraintSet::Variable;

bool isContiguousDescriptorType(MemRefType type);

bool equalBound(OpFoldResult lhs, OpFoldResult rhs) {
  if (lhs == rhs) return true;
  auto left = getConstantIntValue(lhs), right = getConstantIntValue(rhs);
  if (left && right) return *left == *right;
  return haveEqualExtents(Variable(lhs), Variable(rhs));
}

bool unit(OpFoldResult value) { return getConstantIntValue(value) == 1; }

bool unitAxis(Value view, unsigned axis) {
  auto type = cast<MemRefType>(view.getType());
  if (!type.isDynamicDim(axis)) return type.getDimSize(axis) == 1;
  return haveEqualExtents(Variable(view, axis),
                         Variable(IntegerAttr::get(IndexType::get(view.getContext()), 1)));
}

Value stripCasts(Value value) {
  while (auto cast = value.getDefiningOp<memref::CastOp>())
    value = cast.getSource();
  return value;
}

// Compare a descriptor field with a currently available SSA/constant bound.
// Metadata extraction refers to the selected descriptor, including control
// results. Equal allocation origins do not establish field equality.
bool matchesMetadata(Value source, OpFoldResult bound,
                     std::optional<unsigned> strideAxis,
                     llvm::DenseSet<std::pair<Value, int64_t>> &active) {
  source = stripCasts(source);
  if (auto value = dyn_cast<Value>(bound)) {
    if (auto metadata = value.getDefiningOp<memref::ExtractStridedMetadataOp>()) {
      if (stripCasts(metadata.getSource()) == source &&
          (!strideAxis || *strideAxis < metadata.getStrides().size())) {
        Value field = strideAxis ? Value(metadata.getStrides()[*strideAxis])
                                 : Value(metadata.getOffset());
        if (value == field) return true;
      }
    }
  }
  auto type = dyn_cast<MemRefType>(source.getType());
  if (!type || (strideAxis && *strideAxis >= type.getRank())) return false;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset))) return false;
  int64_t fixed = strideAxis ? strides[*strideAxis] : offset;
  if (!ShapedType::isDynamic(fixed))
    return equalBound(bound, IntegerAttr::get(IndexType::get(source.getContext()), fixed));

  auto key = std::make_pair(source, strideAxis ? int64_t(*strideAxis) : int64_t(-1));
  if (!active.insert(key).second) return true;
  auto leave = llvm::make_scope_exit([&] { active.erase(key); });
  if (auto reinterpret = source.getDefiningOp<memref::ReinterpretCastOp>())
    return equalBound(bound, strideAxis ? reinterpret.getMixedStrides()[*strideAxis]
                                       : reinterpret.getMixedOffsets().front());
  if (auto subview = source.getDefiningOp<memref::SubViewOp>()) {
    if (!strideAxis) {
      if (!llvm::all_of(subview.getMixedOffsets(), [](OpFoldResult offset) {
            return getConstantIntValue(offset) == 0;
          })) return false;
      return matchesMetadata(subview.getSource(), bound, std::nullopt, active);
    }
    auto dropped = subview.getDroppedDims();
    unsigned resultAxis = 0;
    for (unsigned axis = 0; axis < dropped.size(); ++axis) {
      if (dropped.test(axis)) continue;
      if (resultAxis++ != *strideAxis) continue;
      return unit(subview.getMixedStrides()[axis]) &&
             matchesMetadata(subview.getSource(), bound, axis, active);
    }
    return false;
  }
  auto incoming = queryControlFlowIncoming(source);
  return incoming.complete && !incoming.edges.empty() &&
         llvm::all_of(incoming.edges, [&](const ControlFlowEdge &edge) {
           return edge.operand &&
                  matchesMetadata(edge.operand->get(), bound, strideAxis, active);
         });
}

bool matchesMetadata(Value source, OpFoldResult bound,
                     std::optional<unsigned> strideAxis) {
  llvm::DenseSet<std::pair<Value, int64_t>> active;
  return matchesMetadata(source, bound, strideAxis, active);
}

// Match the actual descriptor's stride against its suffix-size product. This
// only compares current integer expressions; it neither constructs arithmetic
// nor infers a coordinate map from coincidentally equal storage sizes.
bool matchesProduct(OpFoldResult value, ArrayRef<OpFoldResult> factors) {
  SmallVector<OpFoldResult> left, right;
  auto flatten = [&](auto &&self, OpFoldResult term,
                     SmallVectorImpl<OpFoldResult> &terms) -> void {
    if (unit(term)) return;
    if (auto operand = dyn_cast<Value>(term))
      if (auto multiply = operand.getDefiningOp<arith::MulIOp>()) {
        self(self, multiply.getLhs(), terms);
        self(self, multiply.getRhs(), terms);
        return;
      }
    terms.push_back(term);
  };
  flatten(flatten, value, left);
  for (OpFoldResult factor : factors) flatten(flatten, factor, right);
  auto constants = [](SmallVectorImpl<OpFoldResult> &terms) -> std::optional<int64_t> {
    int64_t product = 1;
    for (OpFoldResult term : terms)
      if (auto constant = getConstantIntValue(term))
        if (llvm::MulOverflow(product, *constant, product)) return std::nullopt;
    llvm::erase_if(terms, [](OpFoldResult term) { return getConstantIntValue(term).has_value(); });
    return product;
  };
  auto a = constants(left), b = constants(right);
  if (!a || !b || *a != *b || left.size() != right.size()) return false;
  for (OpFoldResult factor : left) {
    auto found = llvm::find_if(right, [&](OpFoldResult other) {
      return equalBound(factor, other);
    });
    if (found == right.end()) return false;
    right.erase(found);
  }
  return true;
}

bool contiguous(Value view, llvm::DenseSet<Value> &active) {
  auto type = dyn_cast<MemRefType>(view.getType());
  if (!type) return false;
  if (isContiguousDescriptorType(type)) return true;
  if (!active.insert(view).second) return true;
  auto leave = llvm::make_scope_exit([&] { active.erase(view); });
  if (auto cast = view.getDefiningOp<memref::CastOp>())
    return contiguous(cast.getSource(), active);
  if (auto reinterpret = view.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto sizes = reinterpret.getMixedSizes(), strides = reinterpret.getMixedStrides();
    for (unsigned axis = 0; axis < sizes.size(); ++axis)
      if (!unit(sizes[axis]) &&
          !matchesProduct(strides[axis], ArrayRef(sizes).drop_front(axis + 1)))
        return false;
    return true;
  }
  if (auto subview = view.getDefiningOp<memref::SubViewOp>()) {
    if (!contiguous(subview.getSource(), active)) return false;
    auto sizes = subview.getMixedSizes(), strides = subview.getMixedStrides();
    bool outerActive = false;
    for (unsigned axis = 0; axis < sizes.size(); ++axis) {
      if (!unit(sizes[axis]) && !unit(strides[axis])) return false;
      if (outerActive && !haveEqualExtents(
              Variable(sizes[axis]), Variable(subview.getSource(), axis)))
        return false;
      outerActive |= !unit(sizes[axis]);
    }
    return true;
  }
  if (auto expand = view.getDefiningOp<memref::ExpandShapeOp>())
    return contiguous(expand.getSrc(), active);
  if (auto collapse = view.getDefiningOp<memref::CollapseShapeOp>())
    return contiguous(collapse.getSrc(), active);
  if (auto transpose = view.getDefiningOp<memref::TransposeOp>()) {
    if (!contiguous(transpose.getIn(), active)) return false;
    SmallVector<unsigned> before, after;
    for (unsigned axis = 0; axis < type.getRank(); ++axis)
      if (!unitAxis(transpose.getIn(), axis)) before.push_back(axis);
    for (AffineExpr expr : transpose.getPermutation().getResults()) {
      auto axis = dyn_cast<AffineDimExpr>(expr);
      if (!axis) return false;
      if (!unitAxis(transpose.getIn(), axis.getPosition())) after.push_back(axis.getPosition());
    }
    return before == after;
  }
  // All exact incoming slots must preserve the property. A loop backedge is
  // considered under that invariant; its entry/bypass paths are checked too.
  // This proves geometry for the selected value, never equality of pointers.
  auto incoming = queryControlFlowIncoming(view);
  return incoming.complete && !incoming.edges.empty() &&
         llvm::all_of(incoming.edges, [&](const ControlFlowEdge &edge) {
           return edge.operand && contiguous(edge.operand->get(), active);
         });
}

bool isContiguousDescriptorType(MemRefType type) {
  if (!type) return false;
  if (type.getLayout().isIdentity()) return true;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset))) return false;
  std::optional<int64_t> expected = 1;
  for (int64_t axis = type.getRank(); axis-- > 0;) {
    int64_t size = type.getDimSize(axis);
    if (size != 1 && (!expected || ShapedType::isDynamic(strides[axis]) ||
                      strides[axis] != *expected)) return false;
    if (ShapedType::isDynamic(size)) expected.reset();
    else if (expected) {
      int64_t product;
      if (llvm::MulOverflow(*expected, size, product)) return false;
      expected = product;
    }
  }
  return true;
}

} // namespace

bool isContiguousDescriptor(Value view) {
  llvm::DenseSet<Value> active;
  return contiguous(view, active);
}

FailureOr<ViewAxisProjection> queryViewAxisProjection(Value value) {
  auto target = dyn_cast<MemRefType>(value.getType());
  if (!target) return failure();
  if (auto transpose = value.getDefiningOp<memref::TransposeOp>()) {
    ViewAxisProjection result{transpose.getIn(), {}};
    for (AffineExpr expr : transpose.getPermutation().getResults()) {
      auto axis = dyn_cast<AffineDimExpr>(expr);
      if (!axis) return failure();
      result.sourceAxes.push_back(axis.getPosition());
    }
    return result;
  }
  if (auto expand = value.getDefiningOp<memref::ExpandShapeOp>()) {
    ViewAxisProjection result{expand.getSrc(),
        SmallVector<std::optional<unsigned>>(target.getRank())};
    for (auto [source, group] : llvm::enumerate(expand.getReassociationIndices())) {
      std::optional<unsigned> active;
      for (int64_t axis : group) if (!unitAxis(value, axis)) {
        if (active) return failure();
        active = axis;
      }
      if (active) result.sourceAxes[*active] = source;
      else if (!unitAxis(expand.getSrc(), source)) return failure();
    }
    return result;
  }
  if (auto collapse = value.getDefiningOp<memref::CollapseShapeOp>()) {
    ViewAxisProjection result{collapse.getSrc(), {}};
    for (auto [axis, group] : llvm::enumerate(collapse.getReassociationIndices())) {
      std::optional<unsigned> active;
      for (int64_t source : group) if (!unitAxis(collapse.getSrc(), source)) {
        if (active) return failure();
        active = source;
      }
      if (!active && !unitAxis(value, axis)) return failure();
      result.sourceAxes.push_back(active);
    }
    return result;
  }
  auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>();
  if (!reinterpret) return failure();
  Value source = stripCasts(reinterpret.getSource());
  if (auto metadata = source.getDefiningOp<memref::ExtractStridedMetadataOp>();
      metadata && source == metadata.getBaseBuffer())
    source = metadata.getSource();
  auto type = dyn_cast<MemRefType>(source.getType());
  if (!type || type.getElementType() != target.getElementType() ||
      type.getMemorySpace() != target.getMemorySpace() ||
      !matchesMetadata(source, reinterpret.getMixedOffsets().front(), std::nullopt))
    return failure();
  ViewAxisProjection result{source, {}};
  SmallVector<bool> used(type.getRank(), false);
  auto sizes = reinterpret.getMixedSizes(), strides = reinterpret.getMixedStrides();
  for (unsigned axis = 0; axis < target.getRank(); ++axis) {
    if (unit(sizes[axis])) {
      result.sourceAxes.push_back(std::nullopt);
      continue;
    }
    std::optional<unsigned> match;
    for (unsigned input = 0; input < type.getRank(); ++input) {
      if (used[input] || unitAxis(source, input) ||
          !haveEqualExtents(Variable(sizes[axis]), Variable(source, input)) ||
          !matchesMetadata(source, strides[axis], input)) continue;
      if (match) return failure();
      match = input;
    }
    if (!match) return failure();
    used[*match] = true;
    result.sourceAxes.push_back(match);
  }
  for (unsigned axis = 0; axis < used.size(); ++axis)
    if (!used[axis] && !unitAxis(source, axis)) return failure();
  return result;
}

} // namespace intent::cpu
