#include "TileShape.h"
#include "../capabilities/Target.h"
#include "intentdsl/levels/algorithm/kir/ir/StorageLayout.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

LogicalResult dotSignature(Operation *op) {
  if (!isMemoryEffectFree(op) || op->getNumResults() != 1 ||
      (op->getNumOperands() != 2 && op->getNumOperands() != 3))
    return failure();
  auto trAttr = op->getAttrOfType<BoolAttr>("transpose_rhs");
  if (!trAttr)
    return failure();
  auto l = shape(op->getOperand(0).getType()),
       r = shape(op->getOperand(1).getType()),
       m = shape(op->getResult(0).getType());
  bool tr = trAttr.getValue();
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
      !ShapedType::isDynamic(l[0]) && !ShapedType::isDynamic(l[1]) &&
      !ShapedType::isDynamic(m[1]) && l[1] == r[tr ? 1 : 0] && m[0] == l[0] &&
      m[1] == r[tr ? 0 : 1] && l[0] >= 16 && l[1] >= 16 && m[1] >= 16 &&
      (op->getNumOperands() == 2 ||
       op->getOperand(2).getType() == op->getResult(0).getType()));
}

LogicalResult collectiveSignature(Operation *op) {
  if (!isMemoryEffectFree(op) || op->getNumOperands() != 1 ||
      op->getNumResults() != 1)
    return failure();
  auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
  auto alg = op->getAttrOfType<StringAttr>("algebra");
  if (!input || !axisAttr || !alg ||
      !validCollectiveElement(input.getElementType()))
    return failure();
  int64_t axis = axisAttr.getInt();
  if (axis < 0 || axis >= input.getRank() || input.getDimSize(axis) <= 0 ||
      !llvm::isPowerOf2_64(input.getDimSize(axis)))
    return failure();
  return success((alg == "sum" || alg == "min" || alg == "max") &&
                 validReducedShape(input, axis, op->getResult(0).getType()));
}
} // namespace
void tileContracts(Unit &unit) {
  for (auto [op, name, symbol] :
       {std::tuple{"axis_reduce", "dynamic-collective", "tl.sum/tl.max"},
        {"bin_count", "histogram", "tl.histogram"},
        {"reduce", "native-row-reduce", "tl.sum/tl.max"},
        {"scan_sum", "inclusive-scan", "tl.cumsum"},
        {"summary_reduce", "moments-reduction", "tl.reduce"},
        {"summary_merge", "moments-merge", "merge_moments"}}) {
    ImplementationContract c;
    c.operation = "triton_src." + std::string(op);
    c.name = name;
    c.symbol = symbol;
    c.source = ImplementationSource::Builtin;
    c.minimum = Reuse::Native;
    if (c.name == "moments-reduction" || c.name == "moments-merge") {
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
    TypedContract tc;
    tc.effects = EffectMask::None;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    StringRef k(op);
    if (k == "summary_merge") {
      tc.shape.minOperands = 2;
      tc.shape.maxOperands = 2;
      tc.operandTypes = {ScalarTypeKind::Any};
      tc.resultTypes = {ScalarTypeKind::Any};
    } else if (k == "summary_reduce") {
      tc.shape.minOperands = 2;
      tc.shape.maxOperands = 2;
      tc.shape.operandRanks = {2, 2};
      tc.operandTypes = {ScalarTypeKind::F32, ScalarTypeKind::I1};
      tc.resultTypes = {ScalarTypeKind::Any};
    } else if (k == "bin_count") {
      tc.shape.minOperands = 1;
      tc.shape.maxOperands = 1;
      tc.shape.operandRanks = {1};
      tc.shape.resultRanks = {1};
      tc.operandTypes = {ScalarTypeKind::I32, ScalarTypeKind::I64};
      tc.resultTypes = {ScalarTypeKind::I32};
    } else if (k == "scan_sum") {
      tc.shape.minOperands = 1;
      tc.shape.maxOperands = 1;
      tc.shape.operandRanks = {2};
      tc.shape.resultRanks = {2};
      tc.shape.sameFirstOperandAndResultShape = true;
      tc.operandTypes = {ScalarTypeKind::I32, ScalarTypeKind::F32};
      tc.resultTypes = {ScalarTypeKind::I32, ScalarTypeKind::F32};
    } else if (k == "reduce") {
      tc.shape.minOperands = 1;
      tc.shape.maxOperands = 1;
      tc.shape.operandRanks = {2};
      tc.shape.resultRanks = {1};
      tc.operandTypes = {ScalarTypeKind::F32};
      tc.resultTypes = {ScalarTypeKind::F32};
    } else {
      tc.shape.minOperands = 1;
      tc.shape.maxOperands = 1;
      tc.operandTypes = {ScalarTypeKind::I1,  ScalarTypeKind::I32,
                         ScalarTypeKind::I64, ScalarTypeKind::F16,
                         ScalarTypeKind::BF16, ScalarTypeKind::F32};
      tc.resultTypes = tc.operandTypes;
    }
    c.typed = std::move(tc);
    c.available = [](const TargetProfile &p) -> llvm::Error {
      if (!p.services.get<intentdsl::block::OwnerMachine>())
        return llvm::createStringError("missing CUDA execution facts");
      return llvm::Error::success();
    };
    c.verify = [](Operation *o) -> LogicalResult {
      if (!isMemoryEffectFree(o) || o->getNumResults() != 1)
        return failure();
      auto kind = o->getName().stripDialect();
      if (kind == "summary_merge") {
        auto alg = o->getAttrOfType<StringAttr>("algebra");
        if (o->getNumOperands() != 2 || !alg || alg.getValue() != "welford")
          return failure();
        Type t = o->getResult(0).getType();
        return success(validMomentTuple(t) && o->getOperand(0).getType() == t &&
                       o->getOperand(1).getType() == t);
      }
      if (kind == "summary_reduce") {
        if (o->getNumOperands() != 2)
          return failure();
        auto samples = dyn_cast<RankedTensorType>(o->getOperand(0).getType());
        auto mask = dyn_cast<RankedTensorType>(o->getOperand(1).getType());
        if (!samples || !mask || samples.getRank() != 2 ||
            !samples.getElementType().isF32() || !mask.getElementType().isInteger(1) ||
            samples.getShape() != mask.getShape())
          return failure();
        return success(
            validMomentTuple(o->getResult(0).getType(), samples.getShape().take_front(1)));
      }
      if (o->getNumOperands() != 1)
        return failure();
      auto input = dyn_cast<RankedTensorType>(o->getOperand(0).getType());
      if (!input || input.getRank() < 1)
        return failure();
      if (kind == "bin_count") {
        auto out = dyn_cast<RankedTensorType>(o->getResult(0).getType());
        return success(input.getRank() == 1 &&
                       (input.getElementType().isInteger(32) ||
                        input.getElementType().isInteger(64)) &&
                       out && out.getRank() == 1 &&
                       out.getElementType().isInteger(32));
      }
      if (kind == "scan_sum") {
        return success(input.getRank() == 2 &&
                       o->getResult(0).getType() == input &&
                       (input.getElementType().isInteger(32) ||
                        input.getElementType().isF32()));
      }
      if (kind == "reduce") {
        auto alg = o->getAttrOfType<StringAttr>("algebra");
        auto out = dyn_cast<RankedTensorType>(o->getResult(0).getType());
        return success(alg && (alg == "sum" || alg == "max_num") &&
                       input.getRank() == 2 && out && out.getRank() == 1 &&
                       out.getDimSize(0) == input.getDimSize(0) &&
                       out.getElementType() == input.getElementType() &&
                       input.getElementType().isF32());
      }
      if (kind == "axis_reduce") {
        auto axisAttr = o->getAttrOfType<IntegerAttr>("axis");
        auto alg = o->getAttrOfType<StringAttr>("algebra");
        if (!axisAttr || !alg ||
            !validCollectiveElement(input.getElementType()) ||
            !(alg == "max" || alg == "sum" || alg == "min"))
          return failure();
        int64_t axis = axisAttr.getInt();
        return success(axis >= -1 && axis < input.getRank() &&
                       validReducedShape(input, axis, o->getResult(0).getType()));
      }
      return failure();
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
    TypedContract tc;
    tc.effects = EffectMask::None;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    if (isDot) {
      tc.shape.minOperands = 2;
      tc.shape.maxOperands = 3;
      tc.shape.operandRanks = {2, 2};
      tc.shape.resultRanks = {2};
      tc.operandTypes = {ScalarTypeKind::I8,  ScalarTypeKind::I32,
                         ScalarTypeKind::F8E4M3FN, ScalarTypeKind::F16,
                         ScalarTypeKind::BF16, ScalarTypeKind::F32};
      tc.resultTypes = {ScalarTypeKind::I32, ScalarTypeKind::F32};
      tc.accumulatorTypes = {ScalarTypeKind::I32, ScalarTypeKind::F32};
    } else {
      tc.shape.minOperands = 1;
      tc.shape.maxOperands = 1;
      tc.operandTypes = {ScalarTypeKind::I1,  ScalarTypeKind::I32,
                         ScalarTypeKind::I64, ScalarTypeKind::F16,
                         ScalarTypeKind::BF16, ScalarTypeKind::F32};
      tc.resultTypes = tc.operandTypes;
    }
    c.typed = std::move(tc);
    c.available = [](const TargetProfile &p) -> llvm::Error {
      if (!p.services.get<intentdsl::block::OwnerMachine>())
        return llvm::createStringError("missing CUDA execution capabilities");
      return llvm::Error::success();
    };
    c.verify = isDot ? dotSignature : collectiveSignature;
    unit.implementations.push_back({std::move(c), COMPILERINFRA_SOURCE});
  }
  {
    ImplementationContract sc;
    sc.name = "scaled-mma";
    sc.operation = "triton_src.scaled_mma";
    sc.symbol = "tl.dot_scaled";
    sc.source = ImplementationSource::Builtin;
    sc.minimum = Reuse::Native;
    sc.operands = "[M,K] x [K,N] FP8 E4M3FN with [M,K/32] and [N,K/32] E8M0 scales";
    sc.layout = "register or descriptor tiles; MX8 scales grouped by 32 along K";
    sc.accumulator = "f32";
    sc.effects = "pure register results";
    sc.numerical = "MX8 E8M0 block-scaled FP8 Tensor Core accumulation into FP32";
    sc.estimator = "exits/triton::matrix-issues";
    sc.costParameters = {"M", "N", "K"};
    TypedContract tc;
    tc.effects = EffectMask::None;
    tc.shape.minOperands = 5;
    tc.shape.maxOperands = 5;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    tc.shape.operandRanks = {2, 2, 2, 2, 2};
    tc.shape.resultRanks = {2};
    tc.operandTypes = {ScalarTypeKind::F8E4M3FN, ScalarTypeKind::I8,
                       ScalarTypeKind::F32};
    tc.resultTypes = {ScalarTypeKind::F32};
    tc.accumulatorTypes = {ScalarTypeKind::F32};
    sc.typed = std::move(tc);
    sc.available = [](const TargetProfile &p) -> llvm::Error {
      auto t = p.services.get<intentdsl::itsrc::TritonTarget>();
      if (!t || !t->scaledMma)
        return llvm::createStringError("missing Triton scaled MMA capability");
      return llvm::Error::success();
    };
    sc.verify = [](Operation *o) -> LogicalResult {
      if (!isMemoryEffectFree(o) || o->getNumOperands() != 5 ||
          o->getNumResults() != 1)
        return failure();
      auto aTy = dyn_cast<RankedTensorType>(o->getOperand(0).getType());
      auto lsTy = dyn_cast<RankedTensorType>(o->getOperand(1).getType());
      auto bTy = dyn_cast<RankedTensorType>(o->getOperand(2).getType());
      auto rsTy = dyn_cast<RankedTensorType>(o->getOperand(3).getType());
      auto resTy = dyn_cast<RankedTensorType>(o->getResult(0).getType());
      if (!aTy || !lsTy || !bTy || !rsTy || !resTy)
        return failure();
      auto g = intentdsl::kir::MX8Block::group;
      return success(
          aTy.getRank() == 2 && bTy.getRank() == 2 && resTy.getRank() == 2 &&
          isa<Float8E4M3FNType>(aTy.getElementType()) &&
          isa<Float8E4M3FNType>(bTy.getElementType()) &&
          resTy.getElementType().isF32() &&
          o->getOperand(4).getType() == resTy &&
          aTy.getDimSize(1) > 0 && aTy.getDimSize(1) % g == 0 &&
          aTy.getDimSize(1) == bTy.getDimSize(0) &&
          resTy.getDimSize(0) == aTy.getDimSize(0) &&
          resTy.getDimSize(1) == bTy.getDimSize(1) &&
          lsTy.getElementType().isInteger(8) &&
          rsTy.getElementType().isInteger(8) &&
          lsTy.getShape() ==
              ArrayRef<int64_t>{aTy.getDimSize(0), aTy.getDimSize(1) / g} &&
          rsTy.getShape() ==
              ArrayRef<int64_t>{bTy.getDimSize(1), aTy.getDimSize(1) / g});
    };
    unit.implementations.push_back({std::move(sc), COMPILERINFRA_SOURCE});
  }
}
} // namespace CI_UNIT_NAMESPACE
