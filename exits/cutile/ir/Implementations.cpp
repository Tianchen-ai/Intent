#include "../capabilities/Index.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/StorageLayout.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
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
      !(scaled ? accum.isF32()
               : (e.isInteger(8) ? accum.isInteger(32) : accum.isF32())))
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
  auto fmtAttr = o->getAttrOfType<StringAttr>("format");
  bool isFp4 = fmtAttr && fmtAttr.getValue() == "e2m1";
  auto g = intentdsl::kir::MX8Block::group;
  int64_t kElems = isFp4 ? x[1] * 2 : x[1];
  return success(
      (isFp4 ? e.isInteger(8) : isa<Float8E4M3FNType>(e)) &&
      (!fmtAttr || isFp4 || fmtAttr.getValue() == "e4m3") &&
      kElems > 0 && kElems % g == 0 &&
      o->getOperand(4).getType() == resTy &&
      lsTy.getElementType().isInteger(8) && rsTy.getElementType().isInteger(8) &&
      lsTy.getShape() == ArrayRef<int64_t>{x[0], kElems / g} &&
      rsTy.getShape() == ArrayRef<int64_t>{kElems / g, z[1]});
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
        {"sorted-run-scan", "run_length_scan", "ct.scan"},
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
    TypedContract tc;
    tc.effects = EffectMask::None;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    if (n == "native-dot") {
      tc.shape.minOperands = 2;
      tc.shape.maxOperands = 3;
      tc.shape.operandRanks = {2, 2};
      tc.shape.resultRanks = {2};
      tc.operandTypes = {ScalarTypeKind::I8,  ScalarTypeKind::I32,
                         ScalarTypeKind::F8E4M3FN, ScalarTypeKind::F16,
                         ScalarTypeKind::BF16, ScalarTypeKind::F32};
      tc.resultTypes = {ScalarTypeKind::I32, ScalarTypeKind::F32};
      tc.accumulatorTypes = {ScalarTypeKind::I32, ScalarTypeKind::F32};
    } else if (n == "scaled-mma") {
      tc.shape.minOperands = 5;
      tc.shape.maxOperands = 5;
      tc.shape.operandRanks = {2, 2, 2, 2, 2};
      tc.shape.resultRanks = {2};
      tc.operandTypes = {ScalarTypeKind::F8E4M3FN, ScalarTypeKind::I8,
                         ScalarTypeKind::F32};
      tc.resultTypes = {ScalarTypeKind::F32};
      tc.accumulatorTypes = {ScalarTypeKind::F32};
    } else if (n == "moments-merge") {
      tc.shape.minOperands = 2;
      tc.shape.maxOperands = 2;
      tc.operandTypes = {ScalarTypeKind::Any};
      tc.resultTypes = {ScalarTypeKind::Any};
    } else if (n == "moments-reduction") {
      tc.shape.minOperands = 2;
      tc.shape.maxOperands = 2;
      tc.shape.operandRanks = {2, 2};
      tc.operandTypes = {ScalarTypeKind::F32, ScalarTypeKind::I1};
      tc.resultTypes = {ScalarTypeKind::Any};
    } else if (n == "random" || quad) {
      tc.shape.minOperands = 2;
      tc.shape.maxOperands = 2;
      tc.operandTypes = {ScalarTypeKind::I64};
      tc.resultTypes = {ScalarTypeKind::F32};
    } else if (n == "native-row-reduce") {
      tc.shape.minOperands = 2;
      tc.shape.maxOperands = 2;
      tc.shape.operandRanks = {2, 2};
      tc.shape.resultRanks = {1};
      tc.operandTypes = {ScalarTypeKind::F32, ScalarTypeKind::I1};
      tc.resultTypes = {ScalarTypeKind::F32};
    } else if (n == "sorted-run-scan") {
      tc.shape.minOperands = tc.shape.maxOperands = 2;
      tc.shape.operandRanks = {1, 1};
      tc.shape.resultRanks = {1};
      tc.shape.sameFirstOperandAndResultShape = true;
      tc.operandTypes = tc.resultTypes = {ScalarTypeKind::I32};
      c.numerical = "inclusive integer sums over equal adjacent sorted keys";
    } else if (n == "inclusive-scan") {
      tc.shape.minOperands = 1;
      tc.shape.maxOperands = 1;
      tc.shape.operandRanks = {2};
      tc.shape.resultRanks = {2};
      tc.shape.sameFirstOperandAndResultShape = true;
      tc.operandTypes = {ScalarTypeKind::I32, ScalarTypeKind::F32};
      tc.resultTypes = {ScalarTypeKind::I32, ScalarTypeKind::F32};
    } else {
      tc.shape.minOperands = 1;
      tc.shape.maxOperands = 1;
      tc.operandTypes = {ScalarTypeKind::I1,  ScalarTypeKind::I32,
                         ScalarTypeKind::I64, ScalarTypeKind::F16,
                         ScalarTypeKind::BF16, ScalarTypeKind::F32};
      tc.resultTypes = tc.operandTypes;
    }
    c.typed = std::move(tc);
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
      if (n == "run_length_scan") {
        auto t = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        auto dir = op->getAttrOfType<IntegerAttr>("reverse");
        return success(t && t.getRank() == 1 && t.hasStaticShape() &&
                       llvm::isPowerOf2_64(t.getDimSize(0)) &&
                       t.getElementType().isInteger(32) && dir &&
                       (dir.getInt() == 0 || dir.getInt() == 1) &&
                       llvm::all_of(op->getOperandTypes(),
                                    [&](Type x) { return x == t; }));
      }
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
  {
    ImplementationContract ac;
    ac.name = "tile-atomic-add";
    ac.operation = "ct_src.tile_atomic_add";
    ac.symbol = "ct.atomic_add";
    ac.source = ImplementationSource::Builtin;
    ac.minimum = Reuse::Native;
    ac.operands = "target array, projected coordinate tuple, f32/i32 update, optional i1 mask";
    ac.layout = "global array coordinates with relaxed device-scope atomic RMW";
    ac.accumulator = "FP32 (with reassoc) or INT32";
    ac.effects = "global read-modify-write";
    ac.numerical = "associative/commutative relaxed atomic addition";
    ac.estimator = "exits/cutile::collective-steps";
    ac.costParameters = {"rows", "width"};
    TypedContract tc;
    tc.effects = EffectMask::ReadWrite;
    tc.shape.minOperands = 3;
    tc.shape.maxOperands = 4;
    tc.shape.minResults = 0;
    tc.shape.maxResults = 0;
    tc.operandTypes = {ScalarTypeKind::I1, ScalarTypeKind::I32,
                       ScalarTypeKind::I64, ScalarTypeKind::F32};
    tc.accumulatorTypes = {ScalarTypeKind::I32, ScalarTypeKind::F32};
    ac.typed = std::move(tc);
    ac.available = [](const TargetProfile &profile) -> llvm::Error {
      if (!profile.services.get<intentdsl::cutile::ArrayIndices>())
        return llvm::createStringError("missing cuTile target capability");
      return llvm::Error::success();
    };
    ac.verify = [](Operation *op) -> LogicalResult {
      if (op->getAttrOfType<StringAttr>("order") != "relaxed" ||
          op->getNumResults() != 0 ||
          (op->getNumOperands() != 3 && op->getNumOperands() != 4))
        return failure();
      auto mem = dyn_cast<MemRefType>(op->getOperand(0).getType());
      Type elem = getElementTypeOrSelf(op->getOperand(2).getType());
      auto shapeOf = [](Type t) -> ArrayRef<int64_t> {
        if (auto rt = dyn_cast<RankedTensorType>(t))
          return rt.getShape();
        return {};
      };
      if (!mem || mem.getElementType() != elem ||
          (!elem.isF32() && !elem.isInteger(32)) ||
          shapeOf(op->getOperand(1).getType()) !=
              shapeOf(op->getOperand(2).getType()))
        return failure();
      if (op->getNumOperands() == 4 &&
          (!getElementTypeOrSelf(op->getOperand(3).getType()).isInteger(1) ||
           shapeOf(op->getOperand(3).getType()) !=
               shapeOf(op->getOperand(1).getType())))
        return failure();
      if (elem.isF32()) {
        auto fm = op->getAttrOfType<arith::FastMathFlagsAttr>("fastmath");
        if (!fm ||
            !bitEnumContainsAll(fm.getValue(), arith::FastMathFlags::reassoc))
          return failure();
      }
      return success();
    };
    unit.implementations.push_back({std::move(ac), COMPILERINFRA_SOURCE});
  }
  for (bool isXchg : {false, true}) {
    ImplementationContract ac;
    ac.name = isXchg ? "tile-atomic-xchg" : "tile-atomic-fetch-add";
    ac.operation = isXchg ? "ct_src.tile_atomic_xchg"
                          : "ct_src.tile_atomic_fetch_add";
    ac.symbol = isXchg ? "ct.atomic_xchg" : "ct.atomic_add";
    ac.source = ImplementationSource::Builtin;
    ac.minimum = Reuse::Native;
    ac.operands = "target array, projected coordinate tuple, f32/i32 operand, optional i1 mask";
    ac.layout = "global array coordinates with declared atomic order and scope";
    ac.accumulator = "FP32 (with reassoc) or INT32";
    ac.effects = "global read-modify-write";
    ac.numerical = isXchg ? "atomic exchange" : "atomic fetch-add";
    ac.estimator = "exits/cutile::collective-steps";
    ac.costParameters = {"rows", "width"};
    TypedContract tc;
    tc.effects = EffectMask::ReadWrite;
    tc.shape.minOperands = 3;
    tc.shape.maxOperands = 4;
    tc.shape.minResults = isXchg ? 0 : 1;
    tc.shape.maxResults = isXchg ? 0 : 1;
    tc.operandTypes = {ScalarTypeKind::I1, ScalarTypeKind::I32,
                       ScalarTypeKind::I64, ScalarTypeKind::F32};
    if (!isXchg)
      tc.resultTypes = {ScalarTypeKind::I32, ScalarTypeKind::F32};
    tc.accumulatorTypes = {ScalarTypeKind::I32, ScalarTypeKind::F32};
    ac.typed = std::move(tc);
    ac.available = [](const TargetProfile &profile) -> llvm::Error {
      if (!profile.services.get<intentdsl::cutile::ArrayIndices>())
        return llvm::createStringError("missing cuTile target capability");
      return llvm::Error::success();
    };
    ac.verify = [isXchg](Operation *op) -> LogicalResult {
      if (op->getNumResults() != (isXchg ? 0u : 1u) ||
          (op->getNumOperands() != 3 && op->getNumOperands() != 4))
        return failure();
      auto ord = op->getAttrOfType<StringAttr>("order");
      auto sc = op->getAttrOfType<StringAttr>("scope");
      if (!ord || (ord != "relaxed" && ord != "acq_rel") || !sc ||
          (sc != "block" && sc != "device"))
        return failure();
      auto mem = dyn_cast<MemRefType>(op->getOperand(0).getType());
      Type elem = getElementTypeOrSelf(op->getOperand(2).getType());
      auto shapeOf = [](Type t) -> ArrayRef<int64_t> {
        if (auto rt = dyn_cast<RankedTensorType>(t))
          return rt.getShape();
        return {};
      };
      if (!mem || mem.getElementType() != elem ||
          (!elem.isF32() && !elem.isInteger(32)) ||
          shapeOf(op->getOperand(1).getType()) !=
              shapeOf(op->getOperand(2).getType()))
        return failure();
      if (!isXchg && op->getResult(0).getType() != op->getOperand(2).getType())
        return failure();
      if (op->getNumOperands() == 4 &&
          (!getElementTypeOrSelf(op->getOperand(3).getType()).isInteger(1) ||
           shapeOf(op->getOperand(3).getType()) !=
               shapeOf(op->getOperand(1).getType())))
        return failure();
      if (!isXchg && elem.isF32()) {
        auto fm = op->getAttrOfType<arith::FastMathFlagsAttr>("fastmath");
        if (!fm ||
            !bitEnumContainsAll(fm.getValue(), arith::FastMathFlags::reassoc))
          return failure();
      }
      return success();
    };
    unit.implementations.push_back({std::move(ac), COMPILERINFRA_SOURCE});
  }
  {
    ImplementationContract lc;
    lc.name = "cublas-gemm";
    lc.operation = "ct_src.library_gemm";
    lc.symbol = "torch.matmul";
    lc.source = ImplementationSource::Builtin;
    lc.minimum = Reuse::Library;
    lc.operands = "2D contiguous lhs [M,K], materialized rhs workspace [K,N], output [M,N]";
    lc.layout = "row-major contiguous 2D device arrays";
    lc.accumulator = "FP32";
    lc.effects = "reads lhs and workspace, writes output on active CUDA stream";
    lc.numerical = "cuBLAS GEMM with FP32 accumulation";
    lc.estimator = "exits/cutile::matrix-issues";
    lc.costParameters = {"M", "N", "K"};
    TypedContract tc;
    tc.effects = EffectMask::ReadWrite;
    tc.shape.minOperands = 3;
    tc.shape.maxOperands = 3;
    tc.shape.minResults = 0;
    tc.shape.maxResults = 0;
    tc.shape.operandRanks = {2, 2, 2};
    tc.operandTypes = {ScalarTypeKind::F16, ScalarTypeKind::BF16,
                       ScalarTypeKind::F32};
    tc.accumulatorTypes = {ScalarTypeKind::F32};
    lc.typed = std::move(tc);
    lc.available = [](const TargetProfile &profile) -> llvm::Error {
      if (!profile.services.get<intentdsl::block::MatrixTarget>())
        return llvm::createStringError("missing cuTile matrix capability");
      return llvm::Error::success();
    };
    lc.verify = [](Operation *op) -> LogicalResult {
      if (op->getNumOperands() != 3 || op->getNumResults() != 0)
        return failure();
      auto a = dyn_cast<MemRefType>(op->getOperand(0).getType());
      auto b = dyn_cast<MemRefType>(op->getOperand(1).getType());
      auto c = dyn_cast<MemRefType>(op->getOperand(2).getType());
      if (!a || !b || !c || a.getRank() != 2 || b.getRank() != 2 ||
          c.getRank() != 2)
        return failure();
      Type el = a.getElementType();
      return success((el.isF16() || el.isBF16() || el.isF32()) &&
                     b.getElementType() == el && c.getElementType() == el);
    };
    unit.implementations.push_back({std::move(lc), COMPILERINFRA_SOURCE});
  }
}
} // namespace CI_UNIT_NAMESPACE
