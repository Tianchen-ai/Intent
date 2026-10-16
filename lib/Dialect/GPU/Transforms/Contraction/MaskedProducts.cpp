#include "ContractionDetail.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;

namespace intent::gpu::contraction {
namespace {

Value compact(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    Value input;
    if (auto broadcast = dyn_cast<BroadcastOp>(op))
      input = broadcast.getValue();
    else if (auto reshape = dyn_cast<ReshapeOp>(op))
      input = reshape.getValue();
    else
      break;
    auto source = dyn_cast<FragmentType>(input.getType());
    auto target = cast<FragmentType>(value.getType());
    if (!source || !queryBroadcastProjection(source, target).isExact())
      break;
    value = input;
  }
  return value;
}

FragmentType withElement(FragmentType type, Type element) {
  return FragmentType::get(type.getContext(), element, type.getShape(),
                           type.getAxisMaps(), type.getValidity(), type.getOwner());
}

bool projectsAs(Value value, FragmentType product,
                ArrayRef<std::optional<unsigned>> axes) {
  auto type = dyn_cast<FragmentType>(value.getType());
  if (!type) return false;
  auto projection = queryBroadcastProjection(type, product);
  return projection.isExact() && llvm::equal(projection.targetToSource, axes);
}

Value squeeze(OpBuilder &b, Location loc, Value value, unsigned axis) {
  auto source = cast<FragmentType>(value.getType());
  auto squeezed = eraseFragmentAxis(source, axis);
  SmallVector<Attribute> groups;
  unsigned output = 0;
  for (unsigned i = 0; i < source.getShape().size(); ++i) {
    SmallVector<int64_t> result;
    if (i != axis) result.push_back(output++);
    groups.push_back(ReshapeGroupAttr::get(b.getContext(),
        b.getDenseI64ArrayAttr({static_cast<int64_t>(i)}), b.getDenseI64ArrayAttr(result)));
  }
  return b.create<ReshapeOp>(loc, squeezed, value, b.getArrayAttr(groups));
}

// Take one ordinal from an already computed matrix. This is a pure SSA gather,
// not a replay of its loads or its contraction producer.
Value member(OpBuilder &b, Location loc, Value matrix, unsigned axis,
             Value ordinal) {
  auto source = cast<FragmentType>(matrix.getType());
  SmallVector<Attribute> shape(source.getShape().getValue());
  shape[axis] = expression(b.getContext(), PhysicalExprKind::Constant, 1);
  auto slice = FragmentType::get(b.getContext(), source.getElementType(),
      b.getArrayAttr(shape), source.getAxisMaps(), source.getValidity(), source.getOwner());
  Value indices = b.create<SplatOp>(loc, withElement(slice, b.getIndexType()), ordinal);
  Value capacity = b.create<PhysicalExprOp>(loc, b.getIndexType(),
      cast<PhysicalExprAttr>(source.getShape()[axis]));
  Value extent = b.create<SplatOp>(loc, withElement(slice, b.getIndexType()), capacity);
  Value valid = b.create<CompareOp>(loc, withElement(slice, b.getI1Type()),
                                   indices, extent, ComparePredicate::Lt);
  Value scalarZero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(source.getElementType()));
  Value fill = b.create<SplatOp>(loc, slice, scalarZero);
  Value gathered = b.create<GatherOp>(loc, slice, matrix, ValueRange{indices},
      valid, fill, ArrayRef<int64_t>{static_cast<int64_t>(axis)});
  return squeeze(b, loc, gathered, axis);
}

Value all(OpBuilder &b, Location loc, Value value, Value identity,
          ArrayRef<int64_t> axes) {
  auto reduce = b.create<ReduceOp>(loc, ValueRange{value}, ValueRange{identity},
                                  ValueRange{}, axes);
  Type type = reduce.getResult(0).getType();
  OpBuilder::InsertionGuard guard(b);
  Block *body = b.createBlock(&reduce.getCombine(), {}, {type, type}, {loc, loc});
  Value result = b.create<BinaryOp>(loc, type, body->getArgument(0),
      body->getArgument(1), BinaryOperator::LogicalAnd);
  b.create<YieldOp>(loc, result);
  return reduce.getResult(0);
}

Value finiteMembers(OpBuilder &b, Location loc, Value value) {
  auto type = cast<FragmentType>(value.getType());
  auto element = cast<FloatType>(type.getElementType());
  auto predicate = withElement(type, b.getI1Type());
  Value infinity = b.create<arith::ConstantOp>(loc, element,
      FloatAttr::get(element, APFloat::getInf(element.getFloatSemantics())));
  Value bound = b.create<SplatOp>(loc, type, infinity);
  Value magnitude = b.create<UnaryOp>(loc, type, value, UnaryOperator::Abs);
  return b.create<CompareOp>(loc, predicate, magnitude, bound, ComparePredicate::Lt);
}

struct GuardedProduct {
  scf::IfOp branch;
  Value checked;
};

void shareFiniteChecks(ArrayRef<GuardedProduct> products, func::FuncOp kernel) {
  DominanceInfo dominance(kernel);
  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<bool> joined(products.size(), false);
  auto aligned = [&](Value lhs, Value rhs) {
    auto left = cast<FragmentType>(lhs.getType());
    auto right = cast<FragmentType>(rhs.getType());
    if (left.getShape() != right.getShape() ||
        left.getElementType() != right.getElementType() ||
        left.getValidity() != right.getValidity()) return false;
    auto projection = queryBroadcastProjection(right, left);
    if (!projection.isExact() ||
        !llvm::equal(projection.targetToSource,
                     ArrayRef<std::optional<unsigned>>{0u, 1u})) return false;
    for (unsigned axis : {0u, 1u}) {
      if (!analysis.lockstepTraversal({lhs, rhs}, {axis, axis}).isExact())
        return false;
    }
    return true;
  };
  for (unsigned i = 0; i < products.size(); ++i) {
    if (joined[i]) continue;
    auto first = products[i];
    SmallVector<unsigned> group{i};
    for (unsigned j = i + 1; j < products.size(); ++j) {
      auto next = products[j];
      if (!joined[j] && first.branch->getBlock() == next.branch->getBlock() &&
          first.branch->isBeforeInBlock(next.branch) &&
          dominance.dominates(next.checked, first.branch) &&
          aligned(first.checked, next.checked))
        group.push_back(j);
    }
    if (group.size() == 1) continue;
    OpBuilder b(first.branch);
    Location loc = first.branch.getLoc();
    Value members = finiteMembers(b, loc, first.checked);
    for (unsigned j : llvm::drop_begin(group)) {
      Value next = finiteMembers(b, loc, products[j].checked);
      if (next.getType() != members.getType())
        next = b.create<BroadcastOp>(loc, members.getType(), next);
      members = b.create<BinaryOp>(loc, members.getType(), members, next,
                                   BinaryOperator::LogicalAnd);
    }
    Value truth = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));
    Value finite = all(b, loc, members, truth, {0, 1});
    // Each fallback computes the original masked product for every input.
    // A stronger shared guard is therefore safe, including mixed finite/NaN
    // operands. Keep contractions and branches in place; move no loads.
    for (unsigned j : group) {
      products[j].branch->setOperand(0, finite);
      joined[j] = true;
    }
  }
}

LogicalResult normalize(ReduceOp reduce, SmallVectorImpl<GuardedProduct> &products) {
  // Region predicate specialization must run first: all-false summaries can
  // disappear and all-true products use ordinary contraction normalization.
  if (reduce->getParentOfType<RegionFoldOp>() ||
      reduce->getParentOfType<RegionScanOp>() ||
      reduce.getSources().size() != 1 || reduce.getIdentities().size() != 1 ||
      !reduce.getCaptures().empty() || reduce.getNumResults() != 1 ||
      reduce.getAxes() != ArrayRef<int64_t>{1} ||
      !isLiteralZeroProjection(reduce.getIdentities().front())) return success();
  auto result = dyn_cast<FragmentType>(reduce.getResult(0).getType());
  auto combine = queryBinaryCombine(reduce.getCombine());
  auto select = reduce.getSources().front().getDefiningOp<SelectOp>();
  if (!result || result.getShape().size() != 2 || !result.getElementType().isF32() ||
      !combine || combine->kind() != BinaryOperator::Add ||
      combine->operation.getStrictRounding() ||
      combine->operation.getApproximate() || combine->operation.getFlushToZero() || !select ||
      !isLiteralZeroProjection(select.getFalseValue())) return success();
  auto multiply = select.getTrueValue().getDefiningOp<BinaryOp>();
  if (!multiply || !multiply.getResult().hasOneUse() ||
      multiply.getOperatorKind() != BinaryOperator::Multiply ||
      multiply.getStrictRounding() || multiply.getApproximate() ||
      multiply.getFlushToZero()) return success();
  auto product = dyn_cast<FragmentType>(multiply.getResult().getType());
  if (!product || product.getShape().size() != 3 || !product.getElementType().isF32())
    return success();
  Value lhs = compact(multiply.getLhs()), rhs = compact(multiply.getRhs());
  if (!projectsAs(lhs, product, {0u, 1u, std::nullopt}) ||
      !projectsAs(rhs, product, {std::nullopt, 0u, 1u})) return success();
  auto leftCast = lhs.getDefiningOp<CastOp>(), rightCast = rhs.getDefiningOp<CastOp>();
  if (!leftCast || !rightCast) return success();
  lhs = leftCast.getValue();
  rhs = rightCast.getValue();
  auto left = dyn_cast<FragmentType>(lhs.getType());
  auto right = dyn_cast<FragmentType>(rhs.getType());
  if (!left || !right || left.getShape().size() != 2 || right.getShape().size() != 2 ||
      left.getElementType() != right.getElementType() ||
      (!left.getElementType().isF16() && !left.getElementType().isBF16())) return success();
  for (FragmentType type : {left, right})
    if (!llvm::all_of(type.getShape(), [](Attribute extent) {
          return isCompileTimeExtent(cast<PhysicalExprAttr>(extent));
        })) return success();
  auto uniform = uniformFragmentAxes(select.getCondition());
  if (uniform.size() != 3 || !uniform[2]) return success();
  PhysicalProgramAnalysis analysis(reduce->getParentOfType<func::FuncOp>());
  auto columns = queryExactLogicalRange(analysis.axisRanges(rhs, 1));
  auto rows = queryExactLogicalRange(analysis.axisRanges(lhs, 0));
  if (failed(columns) || failed(rows) || !isUnitStepRange(*columns) ||
      !isUnitStepRange(*rows)) return success();

  OpBuilder b(reduce);
  Location loc = reduce.getLoc();
  auto predicate = projectPhysicalValueToSchema(b, loc, compact(select.getCondition()),
                                               withElement(left, b.getI1Type()));
  if (failed(predicate)) {
    // Project a composed tail predicate in its original rank before removing
    // the proved uniform column. This preserves every coordinate relation
    // through nested broadcasts/reshapes without expanding a boolean tile.
    if (!isKnownPositiveExtent(cast<PhysicalExprAttr>(product.getShape()[2]),
                               reduce->getParentOfType<func::FuncOp>())) return success();
    auto type = cast<FragmentType>(select.getCondition().getType());
    SmallVector<Attribute> shape(type.getShape().getValue());
    shape[2] = expression(b.getContext(), PhysicalExprKind::Constant, 1);
    auto compactType = FragmentType::get(b.getContext(), b.getI1Type(),
        b.getArrayAttr(shape), type.getAxisMaps(), type.getValidity(), type.getOwner());
    auto projected = projectPhysicalValueToSchema(b, loc, select.getCondition(), compactType);
    if (failed(projected)) return success();
    predicate = squeeze(b, loc, *projected, 2);
  }
  auto zero = materializeZeroFragment(b, loc, left);
  if (failed(zero)) return failure();
  Value weights = b.create<SelectOp>(loc, left, *predicate, lhs, *zero);
  // Masking an operand is equivalent to masking its product only when the
  // other operand is finite. Speculate the pure MMA without moving memory
  // effects; only the finite arm consumes its result.
  Value dot = b.create<ContractOp>(loc, result, weights, rhs, reduce.getIdentities().front(),
      ArrayRef<int64_t>{1}, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{}, ArrayRef<int64_t>{});
  Value truth = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));
  Value finite = all(b, loc, finiteMembers(b, loc, rhs), truth, {0, 1});
  auto branch = b.create<scf::IfOp>(loc, reduce.getResultTypes(), finite, true);
  for (Region &region : branch->getRegions())
    if (!region.front().empty()) region.front().back().erase();
  b.setInsertionPointToStart(branch.thenBlock());
  b.create<scf::YieldOp>(loc, dot);

  b.setInsertionPointToStart(branch.elseBlock());
  Value start = b.create<arith::ConstantIndexOp>(loc, 0);
  Value step = b.create<arith::ConstantIndexOp>(loc, 1);
  Value columnStop = b.create<PhysicalExprOp>(loc, b.getIndexType(),
      cast<PhysicalExprAttr>(right.getShape()[1]));
  Value rowStop = b.create<PhysicalExprOp>(loc, b.getIndexType(),
      cast<PhysicalExprAttr>(left.getShape()[0]));
  auto columnCoordinates = projectPhysicalValueToSchema(
      b, loc, (*columns).getResult(), withElement(eraseFragmentAxis(result, 0), b.getIndexType()));
  auto rowCoordinates = projectPhysicalValueToSchema(
      b, loc, (*rows).getResult(), withElement(eraseFragmentAxis(result, 1), b.getIndexType()));
  if (failed(columnCoordinates) || failed(rowCoordinates))
    return reduce.emitOpError("masked product has no physical output projection");
  // Every cell is written once; no speculative contraction result is needed.
  // The exceptional path keeps a reduction vector instead of a product tensor.
  auto empty = materializeZeroFragment(b, loc, result);
  if (failed(empty)) return failure();
  auto loop = b.create<scf::ForOp>(loc, start, rowStop, step, ValueRange{*empty},
                                  [](OpBuilder &, Location, Value, ValueRange) {});
  b.setInsertionPointToStart(loop.getBody());
  auto vectorType = eraseFragmentAxis(left, 0);
  auto project = [&](Value value, Type element) -> FailureOr<Value> {
    if (cast<FragmentType>(value.getType()).getElementType() != element) {
      auto converted = withElement(cast<FragmentType>(value.getType()), element);
      value = b.create<CastOp>(loc, converted, value);
    }
    return projectPhysicalValueToSchema(b, loc, value, withElement(vectorType, element));
  };
  auto a = project(member(b, loc, lhs, 0, loop.getInductionVar()), b.getF32Type());
  auto active = member(b, loc, *predicate, 0, loop.getInductionVar());
  auto columnsLoop = b.create<scf::ForOp>(loc, start, columnStop, step, loop.getRegionIterArgs(),
                                        [](OpBuilder &, Location, Value, ValueRange) {});
  b.setInsertionPointToStart(columnsLoop.getBody());
  auto c = project(member(b, loc, rhs, 1, columnsLoop.getInductionVar()), b.getF32Type());
  if (failed(a) || failed(c))
    return reduce.emitOpError("masked product cannot preserve its member projection");
  auto termType = withElement(vectorType, b.getF32Type());
  auto zeroTerm = materializeZeroFragment(b, loc, termType);
  Value identity = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0));
  if (failed(zeroTerm)) return failure();
  Value term = b.create<BinaryOp>(loc, termType, *a, *c, BinaryOperator::Multiply);
  term = b.create<SelectOp>(loc, termType, active, term, *zeroTerm);
  auto sum = b.create<ReduceOp>(loc, ValueRange{term}, ValueRange{identity},
                                ValueRange{}, ArrayRef<int64_t>{0});
  {
    OpBuilder::InsertionGuard guard(b);
    Type type = sum.getResult(0).getType();
    Block *body = b.createBlock(&sum.getCombine(), {}, {type, type}, {loc, loc});
    Value joined = b.create<BinaryOp>(loc, type, body->getArgument(0),
                                     body->getArgument(1), BinaryOperator::Add);
    b.create<YieldOp>(loc, joined);
  }
  Value value = b.create<SplatOp>(loc, result, sum.getResult(0));
  auto outputMask = [&](Value coordinates, Value origin, Value ordinal) -> FailureOr<Value> {
    Value coordinate = b.create<BinaryOp>(loc, b.getIndexType(), origin, ordinal, BinaryOperator::Add);
    auto coordinateType = cast<FragmentType>(coordinates.getType());
    Value selected = b.create<SplatOp>(loc, coordinateType, coordinate);
    Value writes = b.create<CompareOp>(loc, withElement(coordinateType, b.getI1Type()),
                                      coordinates, selected, ComparePredicate::Eq);
    return projectPhysicalValueToSchema(b, loc, writes, withElement(result, b.getI1Type()));
  };
  auto rowMask = outputMask(*rowCoordinates, (*rows).getStart(), loop.getInductionVar());
  auto columnMask = outputMask(*columnCoordinates, (*columns).getStart(), columnsLoop.getInductionVar());
  if (failed(rowMask) || failed(columnMask))
    return reduce.emitOpError("masked product has no output write projection");
  Value writeMask = b.create<BinaryOp>(loc, withElement(result, b.getI1Type()),
                                      *rowMask, *columnMask, BinaryOperator::LogicalAnd);
  Value total = b.create<SelectOp>(loc, result, writeMask, value,
                                  columnsLoop.getRegionIterArgs().front());
  b.create<scf::YieldOp>(loc, total);
  b.setInsertionPointToEnd(loop.getBody());
  b.create<scf::YieldOp>(loc, columnsLoop.getResults());
  b.setInsertionPointToEnd(branch.elseBlock());
  b.create<scf::YieldOp>(loc, loop.getResults());
  reduce.getResult(0).replaceAllUsesWith(branch.getResult(0));
  products.push_back({branch, rhs});
  reduce.erase();
  return success();
}

} // namespace

LogicalResult normalizeMaskedProducts(ModuleOp module) {
  auto kernel = getPhysicalKernel(module);
  if (failed(kernel)) return failure();
  SmallVector<ReduceOp> reductions;
  SmallVector<GuardedProduct> products;
  kernel->walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
  for (ReduceOp reduce : reductions)
    if (failed(normalize(reduce, products))) return failure();
  shareFiniteChecks(products, *kernel);
  eraseDeadPhysicalValues(*kernel);
  return success();
}

} // namespace intent::gpu::contraction
