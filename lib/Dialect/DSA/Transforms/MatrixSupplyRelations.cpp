#include "MatrixSupplyRelations.h"
#include "Intent/Dialect/DSA/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/DSA/IR/ExecutionRelations.h"
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include <limits>

using namespace mlir;

namespace intent::dsa::detail {
namespace {

// Standard affine expressions normalize the current scalar SSA. Unsupported
// operations stay distinct SSA atoms; they never acquire coordinate facts.
class Coordinates {
public:
  explicit Coordinates(func::FuncOp function)
      : function(function), context(function.getContext()),
        interface(intent::getPublicInterface(function)) {}

  AffineExpr constant(int64_t value) { return getAffineConstantExpr(value, context); }

  AffineExpr dimension(Value source, unsigned axis) {
    auto type = cast<MemRefType>(source.getType());
    if (!type.isDynamicDim(axis)) return constant(type.getDimSize(axis));
    if (auto argument = dyn_cast<BlockArgument>(source);
        argument && argument.getOwner() == &function.front()) {
      auto view = intent::getPublicView(interface, argument.getArgNumber());
      if (view) {
        int64_t id = intent::publicViewDimensions(view)[axis];
        if (id > 0) {
          auto [entry, inserted] = dimensions.try_emplace(id, AffineExpr{});
          if (inserted) entry->second = symbol();
          return entry->second;
        }
      }
    }
    auto [entry, inserted] = viewDimensions.try_emplace(std::make_pair(source, axis), AffineExpr{});
    if (inserted) entry->second = symbol();
    return entry->second;
  }

  AffineExpr stride(Value source, unsigned axis) {
    auto type = cast<MemRefType>(source.getType());
    SmallVector<int64_t> strides;
    int64_t offset;
    if (succeeded(type.getStridesAndOffset(strides, offset)) &&
        !ShapedType::isDynamic(strides[axis]))
      return constant(strides[axis]);
    if (auto argument = dyn_cast<BlockArgument>(source);
        argument && argument.getOwner() == &function.front()) {
      auto view = intent::getPublicView(interface, argument.getArgNumber());
      if (view && view.getConstraints().getHasStrides())
        if (auto fixed = dyn_cast<IntegerAttr>(view.getConstraints().getStrides()[axis]))
          return constant(fixed.getInt());
    }
    auto [entry, inserted] = viewStrides.try_emplace(std::make_pair(source, axis), AffineExpr{});
    if (inserted) entry->second = symbol();
    return entry->second;
  }

  AffineExpr expression(Value value) {
    if (auto found = expressions.find(value); found != expressions.end()) return found->second;
    auto infer = [&]() -> AffineExpr {
      APInt number;
      if (matchPattern(value, m_ConstantInt(&number)) && number.isSignedIntN(64))
        return constant(number.getSExtValue());
      if (!value.getType().isIndex() && !value.getType().isInteger(64)) return symbol();
      if (auto dim = value.getDefiningOp<memref::DimOp>(); dim && dim.getConstantIndex())
        return dimension(dim.getSource(), *dim.getConstantIndex());
      if (auto query = value.getDefiningOp<StrideOp>())
        return stride(query.getSource(), query.getAxis());
      if (auto cast = value.getDefiningOp<arith::IndexCastOp>(); cast &&
          (cast.getIn().getType().isIndex() || cast.getIn().getType().isInteger(64)))
        return expression(cast.getIn());
      if (auto add = value.getDefiningOp<arith::AddIOp>())
        return expression(add.getLhs()) + expression(add.getRhs());
      if (auto sub = value.getDefiningOp<arith::SubIOp>())
        return expression(sub.getLhs()) - expression(sub.getRhs());
      if (auto mul = value.getDefiningOp<arith::MulIOp>())
        return expression(mul.getLhs()) * expression(mul.getRhs());
      if (auto divide = value.getDefiningOp<arith::CeilDivSIOp>();
          divide && nonnegative(divide.getLhs()) && positive(divide.getRhs()))
        return expression(divide.getLhs()).ceilDiv(expression(divide.getRhs()));
      // On a nonnegative domain truncating division agrees with affine floor
      // division. A zero divisor is never executed by a legal original program.
      if (auto divide = value.getDefiningOp<arith::DivSIOp>();
          divide && nonnegative(divide.getLhs()) && nonnegative(divide.getRhs()))
        return expression(divide.getLhs()).floorDiv(expression(divide.getRhs()));
      if (auto remainder = value.getDefiningOp<arith::RemSIOp>();
          remainder && nonnegative(remainder.getLhs()) && nonnegative(remainder.getRhs()))
        return expression(remainder.getLhs()) % expression(remainder.getRhs());
      return symbol();
    };
    AffineExpr result = infer();
    expressions[value] = result;
    return result;
  }

  bool equal(AffineExpr lhs, AffineExpr rhs) {
    auto difference = dyn_cast<AffineConstantExpr>(
        simplifyAffineExpr(lhs - rhs, 0, symbols));
    return difference && difference.getValue() == 0;
  }
  bool equal(Value actual, AffineExpr expected) { return equal(expression(actual), expected); }

  bool clipped(Value actual, AffineExpr remaining, AffineExpr extent, int64_t width) {
    // Both folded complete tiles and dynamic tails use the same two bounds.
    if (equal(actual, remaining) && atMost(actual, width)) return true;
    if (equal(actual, constant(width)) &&
        (equal(extent % width, constant(0)) || atLeast(remaining, width))) return true;
    auto minimum = actual.getDefiningOp<arith::MinSIOp>();
    if (!minimum) return false;
    return (equal(minimum.getLhs(), remaining) && equal(minimum.getRhs(), constant(width))) ||
           (equal(minimum.getRhs(), remaining) && equal(minimum.getLhs(), constant(width)));
  }

  bool nonnegative(Value value) {
    auto bounds = integerInterval(value, function);
    return bounds && bounds->first >= 0;
  }

  int64_t dimensionUpperBound(Value source, unsigned axis) {
    auto type = cast<MemRefType>(source.getType());
    if (!type.isDynamicDim(axis)) return type.getDimSize(axis);
    int64_t upper = std::numeric_limits<int64_t>::max();
    function.walk([&](memref::DimOp dim) {
      if (dim.getSource() != source || dim.getConstantIndex() != axis) return;
      if (auto bounds = integerInterval(dim.getResult(), function))
        upper = std::min(upper, bounds->second);
    });
    return upper;
  }

private:
  bool positive(Value value) {
    auto bounds = integerInterval(value, function);
    return bounds && bounds->first > 0;
  }
  bool atMost(Value value, int64_t upper) {
    auto bounds = integerInterval(value, function);
    return bounds && bounds->second <= upper;
  }
  bool atLeast(AffineExpr expression, int64_t lower) {
    auto folded = dyn_cast<AffineConstantExpr>(simplifyAffineExpr(expression, 0, symbols));
    return folded && folded.getValue() >= lower;
  }
  AffineExpr symbol() { return getAffineSymbolExpr(symbols++, context); }

  func::FuncOp function;
  MLIRContext *context;
  intent::InterfaceAttr interface;
  unsigned symbols = 0;
  DenseMap<Value, AffineExpr> expressions;
  DenseMap<int64_t, AffineExpr> dimensions;
  DenseMap<std::pair<Value, unsigned>, AffineExpr> viewDimensions, viewStrides;
};

} // namespace

std::optional<MatrixWorksetGeometry> queryCompleteMatrixWorkset(scf::ForOp work, scf::ForOp reduction,
    MatMulOp matrix, LoadTileOp lhs, LoadTileOp rhs) {
  auto function = work->getParentOfType<func::FuncOp>();
  ExecutionRelations execution(function);
  if (!execution.readonlyView(lhs.getSource()) || !execution.readonlyView(rhs.getSource()))
    return std::nullopt;
  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(work.getRegion(), captures);
  if (!llvm::all_of(captures, [&](Value value) { return execution.isUniform(value); }))
    return std::nullopt;
  Coordinates coordinates(function);
  AffineExpr M = coordinates.dimension(lhs.getSource(), 0);
  AffineExpr N = coordinates.dimension(rhs.getSource(), 1);
  AffineExpr K = coordinates.dimension(lhs.getSource(), 1);
  if (!coordinates.equal(K, coordinates.dimension(rhs.getSource(), 0))) return std::nullopt;
  auto left = dyn_cast<MemRefType>(matrix.getLhs().getType());
  auto right = dyn_cast<MemRefType>(matrix.getRhs().getType());
  auto outputType = dyn_cast<MemRefType>(matrix.getAccumulator().getType());
  if (!left || !right || !outputType || left.getRank() != 2 || right.getRank() != 2 ||
      outputType.getRank() != 2 || !left.hasStaticShape() || !right.hasStaticShape() ||
      !outputType.hasStaticShape() || matrix.getRhsTransposed()) return std::nullopt;
  int64_t tm = left.getDimSize(0), tn = right.getDimSize(1), tk = left.getDimSize(1);
  if (tm <= 0 || tn <= 0 || tk <= 0 || right.getDimSize(0) != tk ||
      outputType.getShape() != ArrayRef<int64_t>({tm, tn})) return std::nullopt;
  auto upperTiles = [&](Value source, unsigned axis, int64_t width) {
    int64_t upper = coordinates.dimensionUpperBound(source, axis);
    return upper / width + (upper % width != 0);
  };
  int64_t upperM = upperTiles(lhs.getSource(), 0, tm);
  int64_t upperN = upperTiles(rhs.getSource(), 1, tn);
  // The rewritten traversal is a partition of a mathematical workset, not of
  // a wrapped product. Empty worksets need no supply transformation.
  if (upperM <= 0 || upperN <= 0 ||
      static_cast<__int128>(upperM) * upperN > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  AffineExpr gridM = M.ceilDiv(tm), gridN = N.ceilDiv(tn);
  if (!coordinates.equal(work.getUpperBound(), gridM * gridN) ||
      !coordinates.equal(reduction.getLowerBound(), coordinates.constant(0)) ||
      !coordinates.equal(reduction.getUpperBound(), K) ||
      !coordinates.equal(reduction.getStep(), coordinates.constant(tk)))
    return std::nullopt;
  AffineExpr ordinal = coordinates.expression(work.getInductionVar());
  AffineExpr row = ordinal.floorDiv(gridN) * tm;
  AffineExpr column = (ordinal % gridN) * tn;
  AffineExpr depth = coordinates.expression(reduction.getInductionVar());
  auto memory = [&](LoadTileOp load) {
    return coordinates.equal(load.getRowStride(), coordinates.stride(load.getSource(), 0)) &&
           coordinates.equal(load.getColumnStride(), coordinates.stride(load.getSource(), 1));
  };
  if (!memory(lhs) || !memory(rhs) ||
      !coordinates.equal(lhs.getOffset(), row * coordinates.expression(lhs.getRowStride()) +
                                             depth * coordinates.expression(lhs.getColumnStride())) ||
      !coordinates.equal(rhs.getOffset(), depth * coordinates.expression(rhs.getRowStride()) +
                                             column * coordinates.expression(rhs.getColumnStride())) ||
      !coordinates.clipped(matrix.getRows(), M - row, M, tm) ||
      !coordinates.clipped(matrix.getColumns(), N - column, N, tn) ||
      !coordinates.clipped(matrix.getDepth(), K - depth, K, tk) ||
      !coordinates.equal(lhs.getRows(), coordinates.expression(matrix.getRows())) ||
      !coordinates.equal(lhs.getColumns(), coordinates.expression(matrix.getDepth())) ||
      !coordinates.equal(rhs.getRows(), coordinates.expression(matrix.getDepth())) ||
      !coordinates.equal(rhs.getColumns(), coordinates.expression(matrix.getColumns())))
    return std::nullopt;
  // Each original output store must cover its corresponding tile. Reordering
  // the workset never changes an aliasing or partial-write program into a matrix
  // output merely because one read happened to have matrix-shaped storage.
  bool output = false;
  for (Operation *operation = reduction->getNextNode(); operation;
       operation = operation->getNextNode()) {
    auto store = dyn_cast<StoreTileOp>(operation);
    if (!store) continue;
    auto target = dyn_cast<MemRefType>(store.getDestination().getType());
    if (!target || target.getRank() != 2 || target.getMemorySpaceAsInt() != 0 ||
        !coordinates.equal(coordinates.dimension(store.getDestination(), 0), M) ||
        !coordinates.equal(coordinates.dimension(store.getDestination(), 1), N) ||
        !coordinates.equal(store.getRowStride(), coordinates.stride(store.getDestination(), 0)) ||
        !coordinates.equal(store.getColumnStride(), coordinates.stride(store.getDestination(), 1)) ||
        !coordinates.equal(store.getOffset(), row * coordinates.expression(store.getRowStride()) +
                                               column * coordinates.expression(store.getColumnStride())) ||
        !coordinates.equal(store.getRows(), coordinates.expression(matrix.getRows())) ||
        !coordinates.equal(store.getColumns(), coordinates.expression(matrix.getColumns())))
      return std::nullopt;
    output = true;
  }
  return output ? std::optional(MatrixWorksetGeometry{tm, tn, tk}) : std::nullopt;
}

} // namespace intent::dsa::detail
