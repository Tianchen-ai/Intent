#include "Lowering.h"
#include "Intent/Dialect/Block/Analysis/IndexRelations.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Block/Analysis/UniformValues.h"
#include "Intent/Dialect/Block/IR/AccessOpInterface.h"
#include "Intent/Dialect/Block/IR/FragmentOpInterface.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "Intent/Target/DSA/BangC/Physical/Analysis/PhysicalProgram.h"
#include "mlir/IR/Matchers.h"

namespace intent::bangc::block_lowering {
namespace {
bool uniform(const SmallVector<int64_t> &steps) {
  return llvm::all_of(steps, [](int64_t step) { return step == 0; });
}
}

// A current Block coordinate is linear in its fragment lanes. Only exact
// index arithmetic and shape-preserving projections participate; memory reads
// and narrowing casts are never rediscovered as address formulae.
std::optional<Lowering::AffineCoordinate> Lowering::affineCoordinate(Value value) {
  auto type = dyn_cast<block::FragmentType>(value.getType());
  if (!type) {
    Value scalar = get(value);
    if (!scalar || !scalar.getType().isIndex()) return std::nullopt;
    return AffineCoordinate{scalar, {}};
  }
  if (!type.getElementType().isIndex()) return std::nullopt;
  Operation *producer = value.getDefiningOp();
  if (!producer) return std::nullopt;
  if (auto range = dyn_cast<block::MakeRangeOp>(producer)) {
    auto step = block::IndexRelations().constant(range.getStep());
    if (!step) return std::nullopt;
    return AffineCoordinate{get(range.getStart()), {*step}};
  }
  auto project = [&](AffineCoordinate input, Type from) -> std::optional<AffineCoordinate> {
    SmallVector<int64_t> steps(type.getShape().size(), 0);
    if (auto fragment = dyn_cast<block::FragmentType>(from)) {
      auto dimensions = shape(fragment);
      auto relation = block::queryAxisProjection(fragment, type);
      if (failed(dimensions) || !relation.isExact()) return std::nullopt;
      for (auto [axis, sourceAxis] : llvm::enumerate(relation.targetToSource))
        if (sourceAxis && (*dimensions)[*sourceAxis] != 1) steps[axis] = input.steps[*sourceAxis];
    }
    return AffineCoordinate{input.base, steps};
  };
  if (isa<block::SplatOp, block::BroadcastOp, block::ReshapeOp>(producer)) {
    if (isa<block::ReshapeOp>(producer)) {
      auto relation = block::queryFragmentOperandRelations(producer);
      if (failed(relation) || !relation->front().preservesNonUnitAxes()) return std::nullopt;
    }
    Value input = producer->getOperand(0);
    auto affine = affineCoordinate(input);
    return affine ? project(*affine, input.getType()) : std::nullopt;
  }
  if (auto transpose = dyn_cast<block::TransposeOp>(producer)) {
    auto input = affineCoordinate(transpose.getValue());
    if (!input) return std::nullopt;
    SmallVector<int64_t> steps;
    for (int64_t axis : transpose.getPermutation()) steps.push_back(input->steps[axis]);
    return AffineCoordinate{input->base, steps};
  }
  auto binary = dyn_cast<block::BinaryOp>(producer);
  if (!binary) return std::nullopt;
  auto left = affineCoordinate(binary.getLhs()), right = affineCoordinate(binary.getRhs());
  if (!left || !right) return std::nullopt;
  left = project(*left, binary.getLhs().getType());
  right = project(*right, binary.getRhs().getType());
  if (!left || !right) return std::nullopt;
  SmallVector<int64_t> steps(left->steps.size());
  if (binary.getOperatorKind() == BinaryOperator::Add || binary.getOperatorKind() == BinaryOperator::Subtract) {
    bool add = binary.getOperatorKind() == BinaryOperator::Add;
    for (unsigned axis = 0; axis < steps.size(); ++axis) {
      bool overflow = add ? __builtin_add_overflow(left->steps[axis], right->steps[axis], &steps[axis])
                          : __builtin_sub_overflow(left->steps[axis], right->steps[axis], &steps[axis]);
      if (overflow) return std::nullopt;
    }
    Value base = add ? b.createOrFold<arith::AddIOp>(loc, left->base, right->base)
                     : b.createOrFold<arith::SubIOp>(loc, left->base, right->base);
    return AffineCoordinate{base, steps};
  }
  if (binary.getOperatorKind() == BinaryOperator::Multiply) {
    auto factor = uniform(left->steps) ? block::IndexRelations().constant(binary.getLhs()) : std::nullopt;
    if (!factor && uniform(right->steps)) {
      factor = block::IndexRelations().constant(binary.getRhs());
      std::swap(left, right);
    }
    if (!factor) return std::nullopt;
    for (unsigned axis = 0; axis < steps.size(); ++axis)
      if (__builtin_mul_overflow(*factor, right->steps[axis], &steps[axis])) return std::nullopt;
    return AffineCoordinate{b.createOrFold<arith::MulIOp>(loc, left->base, right->base), steps};
  }
  return std::nullopt;
}

FailureOr<bool> Lowering::rectangularAccess(Operation *operation, Value resource, Value result) {
  auto reject = [&](StringRef reason) -> FailureOr<bool> {
    if (auto options = readCompileOptions(source); succeeded(options) && options->getOptimizationRemarks())
      operation->emitRemark("rectangular-supply: ") << reason;
    return false;
  };
  auto nonnegative = [&](Value value) {
    auto range = physical::integerInterval(value, entry);
    return range && range->first >= 0;
  };
  auto access = cast<block::AccessOpInterface>(operation);
  bool store = access.getAccessKind() == block::AccessKind::Store;
  if (!store && access.getAccessKind() != block::AccessKind::Load) return false;
  auto fragment = dyn_cast<block::FragmentType>(access.getAccessValueType());
  if (!fragment || fragment.getShape().empty() ||
      cast<MemRefType>(resource.getType()).getMemorySpaceAsInt() != 0) return false;
  auto dimensions = shape(fragment);
  if (failed(dimensions)) return failure();
  // Local storage flattens all leading fragment dimensions into rows. Unit
  // batch/owner axes do not make that rectangle irregular; keep their bounds
  // below, but do not force otherwise two-dimensional copies into scalar lanes.
  std::optional<unsigned> rowAxis;
  for (unsigned axis = 0; axis + 1 < dimensions->size(); ++axis) {
    if ((*dimensions)[axis] == 1) continue;
    if (rowAxis) return reject("multiple nonunit row axes require a separate rectangular projection");
    rowAxis = axis;
  }
  auto boundary = block::PhysicalProgramAnalysis(source).boundaryValidity(operation, true);
  if (!boundary.isExact()) return reject("validity is not an exact rectangular boundary");
  SmallVector<AffineCoordinate> coordinates;
  for (auto [slot, coordinate] : llvm::enumerate(access.getAccessCoordinates())) {
    auto affine = affineCoordinate(coordinate);
    auto projection = block::queryAccessCoordinateAxes(access, slot);
    if (!affine || !projection.isExact()) return reject("coordinate has no exact affine lane projection");
    SmallVector<int64_t> steps(dimensions->size(), 0);
    auto from = shape(coordinate.getType());
    if (failed(from)) return failure();
    for (auto [axis, sourceAxis] : llvm::enumerate(projection.targetToSource))
      if (sourceAxis && (*from)[*sourceAxis] != 1) steps[axis] = affine->steps[*sourceAxis];
    coordinates.push_back({affine->base, steps});
  }
  // Boundary masks may cut only a prefix of each rectangular dimension. This
  // uses the exact common boundary proof, never the resource shape alone.
  SmallVector<Value> counts;
  for (int64_t size : *dimensions) counts.push_back(index(size));
  // A tiled loop's start is normally an induction variable, not a constant.
  // An exact range mask clips a rectangular prefix at the current iteration;
  // it need not describe a full compile-time tile. Preserve that clip in IR.
  for (auto [range, bound] : boundary.rangeBounds) {
    auto step = block::IndexRelations().constant(range.getStep());
    auto projection = block::queryAxisProjection(range.getResult().getType(), fragment);
    Value begin = get(range.getStart()), end = get(bound);
    if (!step || *step != 1 || !projection.isExact() || !begin || !end ||
        !nonnegative(begin) || !nonnegative(end))
      return reject("range mask is not a nonnegative unit-step prefix");
    std::optional<unsigned> axis;
    for (auto [lane, sourceAxis] : llvm::enumerate(projection.targetToSource)) {
      if (!sourceAxis) continue;
      if (axis) return reject("range mask maps to more than one tile axis");
      axis = lane;
    }
    Value remaining = b.createOrFold<arith::SubIOp>(loc, end, begin);
    remaining = b.createOrFold<arith::MaxSIOp>(loc, remaining, index(0));
    if (axis) counts[*axis] = b.createOrFold<arith::MinSIOp>(loc, counts[*axis], remaining);
    else {
      Value active = b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, begin, end);
      for (Value &count : counts) count = b.createOrFold<arith::SelectOp>(loc, active, count, index(0));
    }
  }
  for (int64_t axis : boundary.boundaryAxes) {
    auto found = llvm::find(access.getAccessSourceAxes(), axis);
    if (found == access.getAccessSourceAxes().end()) return false;
    auto &coordinate = coordinates[found - access.getAccessSourceAxes().begin()];
    Value original = access.getAccessCoordinates()[found - access.getAccessSourceAxes().begin()];
    std::optional<unsigned> varying;
    for (auto [lane, step] : llvm::enumerate(coordinate.steps)) {
      if (step == 0) continue;
      if (step != 1 || varying) return false;
      varying = lane;
    }
    Value extent = b.createOrFold<memref::DimOp>(loc, resource, axis);
    if (!varying) {
      Value valid = b.createOrFold<arith::AndIOp>(loc,
          b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, coordinate.base, index(0)),
          b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, coordinate.base, extent));
      for (Value &count : counts) count = b.createOrFold<arith::SelectOp>(loc, valid, count, index(0));
      continue;
    }
    if (!block::IndexRelations().nonnegative(original) &&
        !nonnegative(coordinate.base)) {
      return reject("resource boundary has no nonnegative tile origin proof");
    }
    // The separate lower-bound proof closes the native prefix form. Retain
    // the exact upper-bound mask as a clipped native transfer extent.
    Value available = b.createOrFold<arith::SubIOp>(loc, extent, coordinate.base);
    available = b.createOrFold<arith::MaxSIOp>(loc, available, index(0));
    counts[*varying] = b.createOrFold<arith::MinSIOp>(loc, counts[*varying], available);
  }
  Value offset = index(0);
  SmallVector<Value> steps(dimensions->size(), index(0));
  for (auto [axis, coordinate] : llvm::zip(access.getAccessSourceAxes(), coordinates)) {
    Value stride = b.create<physical::StrideOp>(loc, b.getIndexType(), resource, axis);
    offset = b.createOrFold<arith::AddIOp>(loc, offset, b.createOrFold<arith::MulIOp>(loc, coordinate.base, stride));
    for (unsigned lane = 0; lane < steps.size(); ++lane)
      steps[lane] = b.createOrFold<arith::AddIOp>(loc, steps[lane],
          b.createOrFold<arith::MulIOp>(loc, stride, index(coordinate.steps[lane])));
  }
  Value rows = rowAxis ? counts[*rowAxis] : index(1);
  for (unsigned axis = 0; axis + 1 < dimensions->size(); ++axis) {
    if (rowAxis && axis == *rowAxis) continue;
    Value active = b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                               counts[axis], index(0));
    rows = b.createOrFold<arith::SelectOp>(loc, active, rows, index(0));
  }
  Value columns = counts.back();
  Value rowStride = rowAxis ? steps[*rowAxis] : index(0);
  if (store) {
    Value payload = get(access.getAccessPayloads().front());
    b.create<physical::StoreTileOp>(loc, payload, resource, offset, rowStride, steps.back(), rows, columns);
  } else {
    b.create<physical::LoadTileOp>(loc, resource, result, offset, rowStride, steps.back(), rows, columns);
    Value fill = access.getAccessFill();
    Value uniformFill = fill ? block::uniformScalarSource(fill) : Value();
    bool zero = uniformFill && (matchPattern(uniformFill, m_Zero()) || matchPattern(uniformFill, m_AnyZeroFloat()));
    if (fill && !zero && (!boundary.boundaryAxes.empty() || !boundary.rangeBounds.empty())) {
      // Native transfers define zero padding. Restore the author's fill only
      // for lanes excluded by the exact boundary mask.
      if (failed(each(*dimensions, [&](ValueRange lane) -> LogicalResult {
        Value active = b.create<arith::ConstantIntOp>(loc, 1, 1);
        for (auto [index, coordinate] : llvm::enumerate(lane))
          active = b.createOrFold<arith::AndIOp>(loc, active,
              b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, coordinate, counts[index]));
        auto branch = b.create<scf::IfOp>(loc, active, true);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(branch.elseBlock());
        auto value = element(fill, fragment, lane);
        return failed(value) ? failure() : write(*value, result, lane);
      }))) return failure();
    }
    values.map(operation->getResult(0), result);
  }
  return true;
}
}
