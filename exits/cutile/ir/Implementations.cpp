#include "../capabilities/Index.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/StorageLayout.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
namespace {
LogicalResult matrix(Operation *o) {
  bool scaled = o->getName().stripDialect() == "scaled_mma";
  auto ranks = [&](Value v) {
    return cast<RankedTensorType>(v.getType()).getShape();
  };
  Value a = o->getOperand(0), b = o->getOperand(scaled ? 2 : 1),
        result = o->getResult(0);
  auto x = ranks(a), y = ranks(b), z = ranks(result);
  bool transpose =
      !scaled && o->getAttrOfType<BoolAttr>("transpose_rhs").getValue();
  if (x.size() != 2 || y.size() != 2 || z.size() != 2 ||
      x[1] != y[transpose ? 1 : 0] || x[0] != z[0] ||
      y[transpose ? 0 : 1] != z[1])
    return failure();
  auto e = getElementTypeOrSelf(a.getType()),
       accum = getElementTypeOrSelf(result.getType());
  if (e != getElementTypeOrSelf(b.getType()) ||
      !(e.isInteger(8) ? accum.isInteger(32) : accum.isF32()))
    return failure();
  if (!scaled)
    return success((e.isInteger(8) || e.isF16() || e.isBF16() || e.isF32() ||
                    isa<Float8E4M3FNType>(e)) &&
                   (o->getNumOperands() == 2 ||
                    o->getOperand(2).getType() == result.getType()));
  auto g = intentdsl::kir::MX8Block::group;
  return success(
      isa<Float8E4M3FNType>(e) && x[1] % g == 0 &&
      o->getOperand(4).getType() == result.getType() &&
      getElementTypeOrSelf(o->getOperand(1).getType()).isInteger(8) &&
      getElementTypeOrSelf(o->getOperand(3).getType()).isInteger(8) &&
      ranks(o->getOperand(1)) == ArrayRef<int64_t>{x[0], x[1] / g} &&
      ranks(o->getOperand(3)) == ArrayRef<int64_t>{x[1] / g, z[1]});
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
    if (n == "moments-reduction" || n=="moments-merge" || n == "random" || quad) {
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
    c.verify = [dot, quad](Operation *op) {
      if (dot)
        return matrix(op);
      auto n = op->getName().stripDialect();
      if(n=="summary_merge"){
        auto tuple=dyn_cast<TupleType>(op->getOperand(0).getType());
        if(!tuple || tuple.size()!=3 || tuple!=op->getOperand(1).getType() || tuple!=op->getResult(0).getType())return failure();
        return success(getElementTypeOrSelf(tuple.getType(0)).isInteger(32) && getElementTypeOrSelf(tuple.getType(1)).isF32() && tuple.getType(1)==tuple.getType(2));
      }
      if (n == "uniform")
        return success(getElementTypeOrSelf(op->getOperand(1).getType())
                           .isUnsignedInteger(64) &&
                       op->hasAttr("quad_aligned") == quad);
      auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      if (!input)
        return failure();
      if (n == "tile_reduce" || n == "axis_reduce") {
        auto ax = op->getAttrOfType<IntegerAttr>("axis").getInt();
        auto alg = op->getAttrOfType<StringAttr>("algebra");
        if (ax < (n == "axis_reduce" ? -1 : 0) || ax >= input.getRank() ||
            !(alg == "min" || alg == "max" || alg == "sum"))
          return failure();
        if (n == "tile_reduce") {
          SmallVector<int64_t> shape(input.getShape());
          shape.erase(shape.begin() + ax);
          auto t = op->getResult(0).getType();
          auto ranked = dyn_cast<RankedTensorType>(t);
          return success((ranked ? ranked.getShape() : ArrayRef<int64_t>{}) ==
                             ArrayRef<int64_t>(shape) &&
                         getElementTypeOrSelf(t) == input.getElementType());
        }
      }
      return success();
    };
    unit.implementations.push_back({std::move(c), COMPILERINFRA_SOURCE});
  }
}
} // namespace CI_UNIT_NAMESPACE
