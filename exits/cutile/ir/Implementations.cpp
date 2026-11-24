#include "../capabilities/Index.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/StorageLayout.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
namespace {
bool validMomentTuple(Type t, ArrayRef<int64_t> expectedRow = {}) {
  auto tuple = dyn_cast<TupleType>(t);
  if (!tuple || tuple.size() != 3)
    return false;
  auto c = dyn_cast<RankedTensorType>(tuple.getType(0));
  auto m1 = dyn_cast<RankedTensorType>(tuple.getType(1));
  auto m2 = dyn_cast<RankedTensorType>(tuple.getType(2));
  if (!c || !m1 || !m2 || c.getRank() != 1 || m1.getRank() != 1 ||
      m2.getRank() != 1 || !c.getElementType().isInteger(32) ||
      !m1.getElementType().isF32() || !m2.getElementType().isF32() ||
      c.getShape() != m1.getShape() || m1 != m2)
    return false;
  return expectedRow.empty() || c.getShape() == expectedRow;
}

bool validReducedShape(RankedTensorType input, int64_t axis, Type resultType) {
  if (getElementTypeOrSelf(resultType) != input.getElementType())
    return false;
  if (axis == -1)
    return !isa<ShapedType>(resultType);
  if (axis < 0 || axis >= input.getRank())
    return false;
  SmallVector<int64_t> dims(input.getShape());
  dims.erase(dims.begin() + axis);
  if (dims.empty())
    return !isa<ShapedType>(resultType);
  auto out = dyn_cast<RankedTensorType>(resultType);
  return out && out.getShape() == ArrayRef<int64_t>(dims);
}

bool validCollectiveElement(Type elem) {
  return elem.isInteger(1) || elem.isInteger(32) || elem.isInteger(64) ||
         elem.isF16() || elem.isBF16() || elem.isF32();
}

LogicalResult matrix(Operation *o) {
  if (!isMemoryEffectFree(o) || o->getNumResults() != 1)
    return failure();
  bool scaled = o->getName().stripDialect() == "scaled_mma";
  if (scaled ? (o->getNumOperands() != 5)
             : (o->getNumOperands() != 2 && o->getNumOperands() != 3))
    return failure();
  auto trAttr =
      scaled ? BoolAttr() : o->getAttrOfType<BoolAttr>("transpose_rhs");
  if (!scaled && !trAttr)
    return failure();
  auto aTy = dyn_cast<RankedTensorType>(o->getOperand(0).getType());
  auto bTy = dyn_cast<RankedTensorType>(o->getOperand(scaled ? 2 : 1).getType());
  auto resTy = dyn_cast<RankedTensorType>(o->getResult(0).getType());
  if (!aTy || !bTy || !resTy)
    return failure();
  auto x = aTy.getShape(), y = bTy.getShape(), z = resTy.getShape();
  bool transpose = !scaled && trAttr.getValue();
  if (x.size() != 2 || y.size() != 2 || z.size() != 2 ||
      ShapedType::isDynamic(x[0]) || ShapedType::isDynamic(x[1]) ||
      ShapedType::isDynamic(z[1]) || x[1] != y[transpose ? 1 : 0] ||
      x[0] != z[0] || y[transpose ? 0 : 1] != z[1])
    return failure();
  auto e = aTy.getElementType(), accum = resTy.getElementType();
  if (e != bTy.getElementType() ||
      !(e.isInteger(8) ? accum.isInteger(32) : accum.isF32()))
    return failure();
  if (!scaled)
    return success((e.isInteger(8) || e.isF16() || e.isBF16() || e.isF32() ||
                    isa<Float8E4M3FNType>(e)) &&
                   (o->getNumOperands() == 2 ||
                    o->getOperand(2).getType() == resTy));
  auto lsTy = dyn_cast<RankedTensorType>(o->getOperand(1).getType());
  auto rsTy = dyn_cast<RankedTensorType>(o->getOperand(3).getType());
  if (!lsTy || !rsTy)
    return failure();
  auto g = intentdsl::kir::MX8Block::group;
  return success(
      isa<Float8E4M3FNType>(e) && x[1] > 0 && x[1] % g == 0 &&
      o->getOperand(4).getType() == resTy &&
      lsTy.getElementType().isInteger(8) && rsTy.getElementType().isInteger(8) &&
      lsTy.getShape() == ArrayRef<int64_t>{x[0], x[1] / g} &&
      rsTy.getShape() == ArrayRef<int64_t>{x[1] / g, z[1]});
}
} // namespace
void registerContracts(Unit &unit) {
  for (auto [name, op, builtin] :
       {std::tuple{"native-dot", "tile_dot", "ct.mma"},
        {"scaled-mma", "scaled_mma", "ct.mma_scaled"},
        {"row-collective", "tile_reduce", "ct.sum/ct.max/ct.min"},
        {"dynamic-collective", "axis_reduce", "ct.sum/ct.max"},
        {"native-row-reduce", "reduce", "ct.sum/ct.max"},
        {"inclusive-scan", "scan_sum", "ct.cumsum"},
        {"moments-reduction", "summary_reduce", "merge_moments"},
        {"moments-merge", "summary_merge", "merge_moments"},
        {"random", "uniform", "uniform_at"},
        {"random-quad", "uniform", "uniform_quad"}}) {
    ImplementationContract c;
    c.name = name;
    c.operation = "ct_src." + std::string(op);
    c.symbol = builtin;
    auto n = StringRef(name);
    bool dot = n == "scaled-mma" || n == "native-dot";
    bool quad = n == "random-quad";
    if (n == "moments-reduction" || n == "moments-merge" || n == "random" ||
        quad) {
      c.source = ImplementationSource::Runtime;
      c.resource = (n == "random" || quad) ? "runtime/cutile/kernels/philox.py"
                                           : "runtime/cutile/kernels/moments.py";
    }
    c.operands = dot ? "M x K, K x N after projected transpose"
                     : "typed fragment with the operation's declared inputs";
    c.layout = dot ? "matrix tiles; MX scales grouped on the contracted axis"
                   : "logical member order";
    c.accumulator = dot ? "FP32 or INT32" : "declared element or Welford tuple";
    c.effects = "register values only";
    c.numerical =
        dot ? "signed INT8 or floating matrix accumulation; MX scales use E8M0"
            : "preserve the declared algebra, counter order and mask";
    c.estimator =
        dot ? "exits/cutile::matrix-issues" : "exits/cutile::collective-steps";
    c.costParameters = dot ? std::vector<std::string>{"M", "N", "K"}
                           : std::vector<std::string>{"rows", "width"};
    c.available = [scaled = n == "scaled-mma"](
                      const TargetProfile &profile) -> llvm::Error {
      auto cap = profile.services.get<intentdsl::cutile::ArrayIndices>();
      if (!cap || (scaled && !cap->scaledMma))
        return llvm::createStringError("missing cuTile target capability");
      return llvm::Error::success();
    };
    c.verify = [dot, quad](Operation *op) -> LogicalResult {
      if (dot)
        return matrix(op);
      if (!isMemoryEffectFree(op) || op->getNumResults() != 1)
        return failure();
      auto n = op->getName().stripDialect();
      if (n == "summary_merge") {
        auto alg = op->getAttrOfType<StringAttr>("algebra");
        if (op->getNumOperands() != 2 || !alg || alg.getValue() != "welford")
          return failure();
        Type t = op->getResult(0).getType();
        return success(validMomentTuple(t) && op->getOperand(0).getType() == t &&
                       op->getOperand(1).getType() == t);
      }
      if (n == "summary_reduce") {
        if (op->getNumOperands() != 2)
          return failure();
        auto samples = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
        auto mask = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
        if (!samples || !mask || samples.getRank() != 2 ||
            !samples.getElementType().isF32() ||
            !mask.getElementType().isInteger(1) ||
            samples.getShape() != mask.getShape())
          return failure();
        return success(validMomentTuple(op->getResult(0).getType(),
                                        samples.getShape().take_front(1)));
      }
      if (n == "uniform") {
        if (op->getNumOperands() != 2)
          return failure();
        auto pos = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
        auto out = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        return success(op->getOperand(0).getType().isUnsignedInteger(64) &&
                       pos && out && pos.getRank() >= 1 &&
                       pos.getElementType().isUnsignedInteger(64) &&
                       out.getElementType().isF32() &&
                       pos.getShape() == out.getShape() &&
                       op->hasAttr("quad_aligned") == quad);
      }
      if (n == "reduce") {
        if (op->getNumOperands() != 2)
          return failure();
        auto values = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
        auto mask = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
        auto out = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        auto alg = op->getAttrOfType<StringAttr>("algebra");
        return success(alg && (alg == "sum" || alg == "max_num") && values &&
                       mask && out && values.getRank() == 2 &&
                       mask.getElementType().isInteger(1) &&
                       values.getShape() == mask.getShape() &&
                       out.getRank() == 1 &&
                       out.getDimSize(0) == values.getDimSize(0) &&
                       out.getElementType() == values.getElementType() &&
                       values.getElementType().isF32());
      }
      if (op->getNumOperands() != 1)
        return failure();
      auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      if (!input || input.getRank() < 1)
        return failure();
      if (n == "scan_sum") {
        return success(input.getRank() == 2 &&
                       op->getResult(0).getType() == input &&
                       (input.getElementType().isInteger(32) ||
                        input.getElementType().isF32()));
      }
      if (n == "tile_reduce" || n == "axis_reduce") {
        auto axAttr = op->getAttrOfType<IntegerAttr>("axis");
        auto alg = op->getAttrOfType<StringAttr>("algebra");
        if (!axAttr || !alg || !validCollectiveElement(input.getElementType()) ||
            !(alg == "min" || alg == "max" || alg == "sum"))
          return failure();
        int64_t ax = axAttr.getInt();
        if (ax < (n == "axis_reduce" ? -1 : 0) || ax >= input.getRank())
          return failure();
        if (n == "tile_reduce" &&
            (input.getDimSize(ax) <= 0 ||
             !llvm::isPowerOf2_64(input.getDimSize(ax))))
          return failure();
        return success(
            validReducedShape(input, ax, op->getResult(0).getType()));
      }
      return failure();
    };
    unit.implementations.push_back({std::move(c), COMPILERINFRA_SOURCE});
  }
}
} // namespace CI_UNIT_NAMESPACE
