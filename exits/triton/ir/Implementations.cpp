#include "TileShape.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
namespace {
LogicalResult dotSignature(Operation *op) {
  auto l = shape(op->getOperand(0).getType()),
       r = shape(op->getOperand(1).getType()),
       m = shape(op->getResult(0).getType());
  bool tr = op->getAttrOfType<BoolAttr>("transpose_rhs").getValue();
  auto element = getElementTypeOrSelf(op->getOperand(0).getType());
  auto acc = getElementTypeOrSelf(op->getResult(0).getType());
  bool type = element.isInteger(8)
                  ? acc.isInteger(32)
                  : ((isa<Float8E4M3FNType>(element) || element.isBF16() ||
                      element.isF16() || element.isF32()) &&
                     acc.isF32());
  return success(
      type && element == getElementTypeOrSelf(op->getOperand(1).getType()) &&
      l.size() == 2 && r.size() == 2 && m.size() == 2 &&
      l[1] == r[tr ? 1 : 0] && m[0] == l[0] && m[1] == r[tr ? 0 : 1] &&
      l[0] >= 16 && l[1] >= 16 && m[1] >= 16 &&
      (op->getNumOperands() == 2 ||
       op->getOperand(2).getType() == op->getResult(0).getType()));
}
LogicalResult collectiveSignature(Operation *op) {
  auto input = op->getOperand(0).getType(), result = op->getResult(0).getType();
  SmallVector<int64_t> dims(shape(input));
  auto axis = op->getAttrOfType<IntegerAttr>("axis").getInt();
  auto alg = op->getAttrOfType<StringAttr>("algebra");
  if (axis < 0 || size_t(axis) >= dims.size() ||
      !llvm::isPowerOf2_64(dims[axis]))
    return failure();
  dims.erase(dims.begin() + axis);
  return success(ArrayRef<int64_t>(dims) == shape(result) &&
                 getElementTypeOrSelf(input) == getElementTypeOrSelf(result) &&
                 (alg == "sum" || alg == "min" || alg == "max"));
}
} // namespace
void tileContracts(Unit &unit) {
  for (auto [op, name, symbol] :
       {std::tuple{"axis_reduce", "dynamic-collective", "tl.sum/tl.max"},
        {"bin_count", "histogram", "tl.histogram"},
        {"reduce", "native-row-reduce", "tl.sum/tl.max"},
        {"scan_sum", "inclusive-scan", "tl.cumsum"},
        {"summary_reduce", "moments-reduction", "tl.reduce"}}) {
    ImplementationContract c;
    c.operation = "triton_src." + std::string(op);
    c.name = name;
    c.symbol = symbol;
    c.source = ImplementationSource::Builtin;
    c.minimum = Reuse::Native;
    if (c.name == "moments-reduction") {
      c.source = ImplementationSource::Runtime;
      c.symbol = "merge_moments";
      c.resource = "runtime/triton/kernels/moments.py";
    }
    c.operands =
        "typed register fragment in the selected ROWS and WIDTH parameter row";
    c.layout = "declared logical reduction axis";
    c.accumulator = "i32 counters or declared floating accumulator";
    c.effects = "pure within program instance";
    c.numerical = "declared reduction algebra and author identity";
    c.estimator = "exits/triton::collective-steps";
    c.costParameters = {"participants", "width"};
    c.available = [](const TargetProfile &p) -> llvm::Error {
      if (!p.services.get<intentdsl::block::OwnerMachine>())
        return llvm::createStringError("missing CUDA execution facts");
      return llvm::Error::success();
    };
    c.verify = [](Operation *o) {
      if (!isa<RankedTensorType>(o->getOperand(0).getType()))
        return failure();
      if (o->getName().stripDialect() == "axis_reduce") {
        auto a = o->getAttrOfType<IntegerAttr>("axis").getInt();
        auto kind = o->getAttrOfType<StringAttr>("algebra");
        return success(
            a >= -1 &&
            a < cast<RankedTensorType>(o->getOperand(0).getType()).getRank() &&
            (kind == "max" || kind == "sum"));
      }
      return success();
    };
    unit.implementations.push_back({std::move(c), COMPILERINFRA_SOURCE});
  }
  for (bool isDot : {true, false}) {
    ImplementationContract c;
    c.name = isDot ? "native-dot" : "row-collective";
    c.operation = isDot ? "triton_src.tile_dot" : "triton_src.tile_reduce";
    c.symbol = isDot ? "tl.dot" : "tl.sum/tl.min/tl.max";
    c.source = ImplementationSource::Builtin;
    c.minimum = Reuse::Native;
    c.operands = isDot ? "[M,K] x [K,N] or transposed RHS"
                       : "tile -> tile with declared reduction axis removed";
    c.layout = isDot ? "register tiles, contracted axis agrees"
                     : "row fragment, power-of-two width";
    c.accumulator = isDot ? "f32 or i32"
                          : "preserve f32, integer or predicate element type";
    c.effects = "pure register results";
    c.numerical = isDot ? "IEEE input precision, declared accumulator"
                        : "native sum, min or max along the explicit axis";
    c.estimator = isDot ? "exits/triton::matrix-issues"
                        : "exits/triton::collective-steps";
    c.costParameters = isDot ? std::vector<std::string>{"M", "N", "K"}
                             : std::vector<std::string>{"M", "K"};
    c.available = [](const TargetProfile &p) -> llvm::Error {
      if (!p.services.get<intentdsl::block::OwnerMachine>())
        return llvm::createStringError("missing CUDA execution capabilities");
      return llvm::Error::success();
    };
    c.verify = isDot ? dotSignature : collectiveSignature;
    unit.implementations.push_back({std::move(c), COMPILERINFRA_SOURCE});
  }
}
} // namespace CI_UNIT_NAMESPACE
