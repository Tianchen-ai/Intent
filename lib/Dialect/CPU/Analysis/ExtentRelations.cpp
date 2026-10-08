#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/ShapeRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::cpu {
namespace {

using Variable = ValueBoundsConstraintSet::Variable;
using Dimension = std::pair<Value, int64_t>;

Operation *owner(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    return argument.getOwner()->getParentOp();
  return value.getDefiningOp();
}

Value taskCapture(Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument) return {};
  auto task = dyn_cast<TasksOp>(argument.getOwner()->getParentOp());
  if (!task || argument.getArgNumber() == 0) return {};
  return task.getCaptures()[argument.getArgNumber() - 1];
}

std::optional<unsigned> metadataSize(Value value) {
  auto metadata = value.getDefiningOp<memref::ExtractStridedMetadataOp>();
  if (!metadata) return std::nullopt;
  auto position = llvm::find(metadata.getSizes(), value);
  if (position == metadata.getSizes().end()) return std::nullopt;
  return position - metadata.getSizes().begin();
}

Value scalarCapture(Value value) {
  if (Value captured = taskCapture(value)) return captured;
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument) return {};
  ValueRange captures;
  unsigned count = 0;
  if (auto reduction = dyn_cast<SliceReduceOp>(owner(value))) {
    captures = reduction.getCaptures();
    count = reduction.getSources().size();
  } else if (auto scan = dyn_cast<ScanOp>(owner(value))) {
    captures = scan.getCaptures();
    count = scan.getSources().size();
  }
  if (argument.getArgNumber() >= 2 * count &&
      argument.getArgNumber() < 2 * count + captures.size())
    return captures[argument.getArgNumber() - 2 * count];
  auto region = dyn_cast<RegionOpInterface>(owner(value));
  if (!region) return {};
  auto schema = region.getRegionSchema(*argument.getOwner()->getParent());
  if (failed(schema)) return {};
  const auto &relation = (*schema)[argument.getArgNumber()];
  return relation.kind == RegionArgumentKind::Captures ? relation.prototype
                                                     : Value{};
}

// CPU logical index and i64 have the same width. Only unwrap casts that keep
// every bit, and only recognize max(x, 0) when x is already a descriptor size.
Value scalarSource(Value value) {
  llvm::SmallPtrSet<Operation *, 8> visited;
  while (value) {
    if (Value captured = scalarCapture(value)) {
      value = captured;
      continue;
    }
    Operation *operation = value.getDefiningOp();
    if (!operation || !visited.insert(operation).second) break;
    if (auto cast = dyn_cast<arith::IndexCastOp>(operation)) {
      if (cast.getIn().getType().isIndex() || cast.getIn().getType().isInteger(64)) {
        if (value.getType().isIndex() || value.getType().isInteger(64)) {
          value = cast.getIn();
          continue;
        }
      }
    }
    if (auto maximum = dyn_cast<arith::MaxSIOp>(operation)) {
      Value candidate;
      if (matchPattern(maximum.getLhs(), m_Zero())) candidate = maximum.getRhs();
      else if (matchPattern(maximum.getRhs(), m_Zero())) candidate = maximum.getLhs();
      if (candidate) {
        Value source = scalarSource(candidate);
        if (source.getDefiningOp<memref::DimOp>() || metadataSize(source)) {
          value = source;
          continue;
        }
      }
    }
    break;
  }
  return value;
}

std::optional<Dimension> helperDimension(BlockArgument argument, int64_t axis) {
  if (Value captured = taskCapture(argument)) return Dimension{captured, axis};
  Operation *operation = owner(argument);
  if (auto program = dyn_cast<RegionOpInterface>(operation)) {
    Region &region = *argument.getOwner()->getParent();
    auto schema = program.getRegionSchema(region);
    if (failed(schema)) return std::nullopt;
    const auto &relation = (*schema)[argument.getArgNumber()];
    if (!relation.sliceAxis || axis != *relation.sliceAxis)
      return Dimension{relation.prototype, axis};
    // Source slices in one helper invocation share their segment extent. It is
    // not the full source extent, nor a value shared by different helpers.
    for (const auto &entry : *schema)
      if (entry.kind == RegionArgumentKind::Sources && entry.sliceAxis)
        return Dimension{entry.argument, *entry.sliceAxis};
    return std::nullopt;
  }
  ValueRange sources, captures;
  SmallVector<int64_t> removed;
  bool destinationPassing;
  if (auto reduction = dyn_cast<SliceReduceOp>(operation)) {
    sources = reduction.getSources();
    captures = reduction.getCaptures();
    removed.assign(reduction.getAxes().begin(), reduction.getAxes().end());
    destinationPassing = reduction.isDestinationPassing();
  } else if (auto scan = dyn_cast<ScanOp>(operation)) {
    sources = scan.getSources();
    captures = scan.getCaptures();
    removed.push_back(scan.getAxis());
    destinationPassing = scan.isDestinationPassing();
  } else {
    return std::nullopt;
  }
  unsigned number = argument.getArgNumber(), count = sources.size();
  if (number >= 2 * count && number < 2 * count + captures.size())
    return Dimension{captures[number - 2 * count], axis};
  if (!destinationPassing) return std::nullopt;
  unsigned component = number < 2 * count ? number % count
                                        : number - 2 * count - captures.size();
  if (component >= count) return std::nullopt;
  auto source = cast<MemRefType>(sources[component].getType());
  int64_t remaining = 0;
  for (int64_t original = 0; original < source.getRank(); ++original)
    if (!llvm::is_contained(removed, original) && remaining++ == axis)
      return Dimension{sources[component], original};
  return std::nullopt;
}

std::optional<Dimension> collapsedDimension(memref::CollapseShapeOp operation,
                                           int64_t axis) {
  auto source = operation.getSrcType();
  std::optional<int64_t> nonunit;
  for (int64_t dimension : operation.getReassociationIndices()[axis]) {
    if (source.getDimSize(dimension) == 1) continue;
    if (nonunit) return std::nullopt;
    nonunit = dimension;
  }
  if (!nonunit) return std::nullopt; // Static unit result is handled by MLIR.
  return Dimension{operation.getSrc(), *nonunit};
}

std::optional<Dimension> expandedDimension(memref::ExpandShapeOp operation,
                                          int64_t axis) {
  auto result = operation.getResultType();
  for (auto [sourceAxis, group] :
       llvm::enumerate(operation.getReassociationIndices())) {
    if (!llvm::is_contained(group, axis)) continue;
    if (llvm::all_of(group, [&](int64_t other) {
          return other == axis || result.getDimSize(other) == 1;
        }))
      return Dimension{operation.getSrc(), sourceAxis};
    return std::nullopt;
  }
  return std::nullopt;
}

template <typename Op>
struct HelperBounds : ValueBoundsOpInterface::ExternalModel<HelperBounds<Op>, Op> {
  void populateBoundsForIndexValue(Operation *, Value value,
                                   ValueBoundsConstraintSet &constraints) const {
    if (Value source = scalarCapture(value); source && source.getType().isIndex())
      constraints.bound(value) == constraints.getExpr(source);
  }
  void populateBoundsForShapedValueDim(Operation *, Value value, int64_t axis,
                                       ValueBoundsConstraintSet &constraints) const {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument) return;
    if (auto source = helperDimension(argument, axis))
      if (source->first != value || source->second != axis)
        constraints.bound(value)[axis] ==
            constraints.getExpr(source->first, source->second);
  }
};

struct PublicBounds : ValueBoundsOpInterface::ExternalModel<PublicBounds, func::FuncOp> {
  void populateBoundsForIndexValue(Operation *, Value,
                                   ValueBoundsConstraintSet &) const {}
  void populateBoundsForShapedValueDim(Operation *, Value value, int64_t axis,
                                       ValueBoundsConstraintSet &constraints) const {
    auto dimension = queryPublicDimension(dyn_cast<BlockArgument>(value), axis);
    if (!dimension) return;
    if (dimension->constant)
      constraints.bound(value)[axis] == *dimension->constant;
    else if (dimension->argument != value || dimension->axis != axis)
      constraints.bound(value)[axis] ==
          constraints.getExpr(dimension->argument, dimension->axis);
  }
};

template <typename Op>
struct DescriptorBounds
    : ValueBoundsOpInterface::ExternalModel<DescriptorBounds<Op>, Op> {
  void populateBoundsForIndexValue(Operation *operation, Value value,
                                   ValueBoundsConstraintSet &constraints) const {
    if (auto metadata = dyn_cast<memref::ExtractStridedMetadataOp>(operation))
      if (auto axis = metadataSize(value))
        constraints.bound(value) == constraints.getExpr(metadata.getSource(), *axis);
  }
  void populateBoundsForShapedValueDim(Operation *operation, Value value,
                                       int64_t axis,
                                       ValueBoundsConstraintSet &constraints) const {
    if (auto view = dyn_cast<memref::ReinterpretCastOp>(operation))
      constraints.bound(value)[axis] == constraints.getExpr(view.getMixedSizes()[axis]);
    else if (auto expand = dyn_cast<memref::ExpandShapeOp>(operation)) {
      if (auto source = expandedDimension(expand, axis))
        constraints.bound(value)[axis] == constraints.getExpr(source->first, source->second);
      auto shape = getMixedValues(expand.getStaticOutputShape(),
                                  expand.getOutputShape(), expand.getContext());
      constraints.bound(value)[axis] == constraints.getExpr(shape[axis]);
    } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(operation)) {
      if (auto source = collapsedDimension(collapse, axis))
        constraints.bound(value)[axis] == constraints.getExpr(source->first, source->second);
    }
  }
};

bool stopAtExtentLeaf(Value value, std::optional<int64_t> axis,
                      ValueBoundsConstraintSet &constraints) {
  Operation *operation = owner(value);
  if (axis) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (isa<func::FuncOp>(operation)) {
        auto dimension = queryPublicDimension(argument, *axis);
        return !dimension || (!dimension->constant &&
            dimension->argument == value && dimension->axis == axis);
      }
      if (isa<TasksOp, SliceReduceOp, ScanOp, RegionOpInterface>(operation)) {
        auto source = helperDimension(argument, *axis);
        return !source || (source->first == value && source->second == axis);
      }
    }
    if (auto collapse = dyn_cast_or_null<memref::CollapseShapeOp>(operation))
      return !collapsedDimension(collapse, *axis);
    return !isa_and_nonnull<ValueBoundsOpInterface>(operation);
  }
  Value source = scalarSource(value);
  if (source != value && source.getType().isIndex()) {
    constraints.bound(value) == constraints.getExpr(source);
    return false;
  }
  if (isa_and_nonnull<arith::ConstantOp, memref::DimOp, memref::RankOp>(operation) ||
      metadataSize(value)) return false;
  // Signed selection bounds hold for the actual operands even when their
  // producers wrap. Keep the selected value as an exact SSA leaf for EQ
  // queries rather than replacing it with an incomplete affine expression.
  if (auto minimum = dyn_cast_or_null<arith::MinSIOp>(operation)) {
    constraints.bound(value) <= constraints.getExpr(minimum.getLhs());
    constraints.bound(value) <= constraints.getExpr(minimum.getRhs());
  } else if (auto maximum = dyn_cast_or_null<arith::MaxSIOp>(operation)) {
    constraints.bound(value) >= constraints.getExpr(maximum.getLhs());
    constraints.bound(value) >= constraints.getExpr(maximum.getRhs());
  }
  // Arbitrary index arithmetic wraps. Do not import its mathematical affine
  // model merely because the result is later used as a size.
  return true;
}

struct ExtentConstraints : ValueBoundsConstraintSet {
  explicit ExtentConstraints(MLIRContext *context)
      : ValueBoundsConstraintSet(context, stopAtExtentLeaf) {}
};

} // namespace

bool haveEqualExtents(const Variable &lhs, const Variable &rhs) {
  ExtentConstraints constraints(lhs.getContext());
  return constraints.populateAndCompare(lhs, ValueBoundsConstraintSet::EQ, rhs);
}

FailureOr<ExtentExpression> queryExtent(const Variable &extent) {
  ExtentExpression result;
  if (failed(ValueBoundsConstraintSet::computeBound(result.expression,
          result.operands, presburger::BoundType::EQ, extent, stopAtExtentLeaf)))
    return failure();
  return result;
}

FailureOr<ExtentExpression> queryExtent(OpFoldResult extent) {
  if (auto value = dyn_cast<Value>(extent)) {
    value = scalarSource(value);
    if (auto constant = getConstantIntValue(value))
      extent = IntegerAttr::get(IndexType::get(value.getContext()), *constant);
    else if (value.getType().isIndex()) extent = value;
    else return failure();
  }
  return queryExtent(Variable(extent));
}

std::optional<int64_t> constantExtentUpperBound(const Variable &extent) {
  auto bound = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::UB, extent, stopAtExtentLeaf, /*closedUB=*/true);
  if (failed(bound) || *bound < 0) return std::nullopt;
  return *bound;
}

std::optional<OpFoldResult> queryExtentValue(Value memory, int64_t axis) {
  AffineMap expression;
  ValueDimList operands;
  auto stop = [](Value value, std::optional<int64_t> dimension,
                 ValueBoundsConstraintSet &) {
    if (isa<BlockArgument>(value)) return true;
    if (!dimension)
      return !isa_and_nonnull<arith::ConstantOp>(value.getDefiningOp());
    return !isa_and_nonnull<ValueBoundsOpInterface>(value.getDefiningOp());
  };
  if (failed(ValueBoundsConstraintSet::computeBound(expression, operands,
          presburger::BoundType::EQ, Variable(memory, axis), stop)))
    return std::nullopt;
  AffineExpr result = expression.getResult(0);
  if (auto constant = dyn_cast<AffineConstantExpr>(result))
    return IntegerAttr::get(IndexType::get(memory.getContext()), constant.getValue());
  unsigned position;
  if (auto dimension = dyn_cast<AffineDimExpr>(result))
    position = dimension.getPosition();
  else if (auto symbol = dyn_cast<AffineSymbolExpr>(result))
    position = expression.getNumDims() + symbol.getPosition();
  else return std::nullopt;
  auto [value, dimension] = operands[position];
  if (dimension || !value.getType().isIndex()) return std::nullopt;
  return value;
}

void registerExtentRelations(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, func::FuncDialect *) {
    func::FuncOp::attachInterface<PublicBounds>(*context);
  });
  registry.addExtension(+[](MLIRContext *context, IntentCPUDialect *) {
    TasksOp::attachInterface<HelperBounds<TasksOp>>(*context);
    SliceReduceOp::attachInterface<HelperBounds<SliceReduceOp>>(*context);
    ScanOp::attachInterface<HelperBounds<ScanOp>>(*context);
    RegionFoldOp::attachInterface<HelperBounds<RegionFoldOp>>(*context);
    RegionScanOp::attachInterface<HelperBounds<RegionScanOp>>(*context);
  });
  registry.addExtension(+[](MLIRContext *context, memref::MemRefDialect *) {
    memref::ReinterpretCastOp::attachInterface<DescriptorBounds<memref::ReinterpretCastOp>>(*context);
    memref::ExtractStridedMetadataOp::attachInterface<DescriptorBounds<memref::ExtractStridedMetadataOp>>(*context);
    memref::ExpandShapeOp::attachInterface<DescriptorBounds<memref::ExpandShapeOp>>(*context);
    memref::CollapseShapeOp::attachInterface<DescriptorBounds<memref::CollapseShapeOp>>(*context);
  });
}

} // namespace intent::cpu
