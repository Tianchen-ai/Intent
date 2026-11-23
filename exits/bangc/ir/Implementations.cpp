#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include "mlir/IR/BuiltinTypes.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;

namespace {
bool isSpace(Value v, int64_t expected) {
  auto m = dyn_cast<MemRefType>(v.getType());
  if (!m)
    return false;
  auto attr = dyn_cast_or_null<IntegerAttr>(m.getMemorySpace());
  int64_t space = attr ? attr.getInt() : 0;
  return space == expected;
}

bool isMomentTuple(Type t) {
  auto tup = dyn_cast<TupleType>(t);
  return tup && tup.size() == 3 && tup.getType(0).isInteger(32) &&
         tup.getType(1).isF32() && tup.getType(2).isF32();
}
} // namespace

void rowContracts(Unit &unit) {
  auto hasNram = [](const TargetProfile &p) -> llvm::Error {
    if (!p.services.get<intentdsl::dsa::LocalSupply>())
      return llvm::createStringError("missing MLU370 NRAM supply capabilities");
    return llvm::Error::success();
  };
  auto add = [&](StringRef name, StringRef op, StringRef symbol,
                 StringRef resource, StringRef operands, StringRef layout,
                 StringRef acc, StringRef effects, StringRef numerical,
                 std::function<LogicalResult(Operation *)> verify) {
    ImplementationContract c;
    c.name = name.str();
    c.operation = ("bangc_src." + op).str();
    c.symbol = symbol.str();
    c.resource = ("runtime/bangc/kernels/" + resource).str();
    c.source = ImplementationSource::Runtime;
    c.minimum = Reuse::Native;
    c.operands = operands.str();
    c.layout = layout.str();
    c.accumulator = acc.str();
    c.effects = effects.str();
    c.numerical = numerical.str();
    c.estimator = "exits/bangc::dma";
    c.costParameters = {"participants", "width", "batch"};
    c.available = hasNram;
    c.verify = std::move(verify);
    unit.implementations.push_back({std::move(c), COMPILERINFRA_SOURCE});
  };

  add("row-reduce", "row_reduce", "group_sum/group_max_num", "reduce.mlu",
      "NRAM f32 pitched input, NRAM f32 scratch, per-member f32 output",
      "contiguous pitched rows [members, pitch] in NRAM", "f32",
      "writes NRAM scratch and per-member f32 reduction results",
      "sumpool+reduce_sum or maxpool+argmax with IEEE NaN/subnormal fallback",
      [](Operation *op) {
        auto alg = op->getAttrOfType<StringAttr>("algebra");
        auto inTy = dyn_cast<MemRefType>(op->getOperand(2).getType());
        auto scTy = dyn_cast<MemRefType>(op->getOperand(1).getType());
        auto outTy = dyn_cast<MemRefType>(op->getOperand(0).getType());
        return success(alg && (alg == "sum" || alg == "max_num") &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) &&
                       (isSpace(op->getOperand(0), 1) ||
                        isSpace(op->getOperand(0), 2)) &&
                       inTy.getElementType().isF32() &&
                       scTy.getElementType().isF32() &&
                       outTy.getElementType().isF32());
      });

  add("validated-exp", "row_exp", "row_exp", "exp.mlu",
      "NRAM f32 input and NRAM f32 output over [members, pitch]",
      "128-byte aligned NRAM vector lanes", "f32",
      "writes NRAM f32 output buffer",
      "__bang_active_exphp in [-80, 80] with exact scalar expf repair outside",
      [](Operation *op) {
        auto outTy = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto inTy = dyn_cast<MemRefType>(op->getOperand(1).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       outTy.getElementType().isF32() &&
                       inTy.getElementType().isF32());
      });

  add("certified-half-quotient", "narrow_quotient", "group_narrow_quotient",
      "quotient.mlu",
      "NRAM f32 numerator, per-member f32 divisor, NRAM f32 quotient/scratch, "
      "NRAM f16 output",
      "pitched rows [members, pitch] in NRAM", "f32",
      "writes NRAM quotient/scratch and certified f16 output",
      "+-4 ULP f32 endpoint certification with exact RN f32->f16 fallback",
      [](Operation *op) {
        auto outTy = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto qTy = dyn_cast<MemRefType>(op->getOperand(1).getType());
        auto scTy = dyn_cast<MemRefType>(op->getOperand(2).getType());
        auto xTy = dyn_cast<MemRefType>(op->getOperand(3).getType());
        auto dTy = dyn_cast<MemRefType>(op->getOperand(4).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) &&
                       isSpace(op->getOperand(3), 1) &&
                       (isSpace(op->getOperand(4), 1) ||
                        isSpace(op->getOperand(4), 2)) &&
                       outTy.getElementType().isF16() &&
                       qTy.getElementType().isF32() &&
                       scTy.getElementType().isF32() &&
                       xTy.getElementType().isF32() &&
                       dTy.getElementType().isF32());
      });

  add("bf16-tile-convert", "bf16_tile", "widen_bf16_tile/narrow_bf16_tile",
      "bf16.mlu", "NRAM bf16 and f32 tile buffers",
      "64-lane aligned NRAM vector fragment", "f32",
      "writes destination NRAM tile",
      "NaN-preserving bf16<->f32 round-to-nearest-even",
      [](Operation *op) {
        auto mode = op->getAttrOfType<StringAttr>("mode");
        auto dst = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto src = dyn_cast<MemRefType>(op->getOperand(1).getType());
        if (!mode || !isSpace(op->getOperand(0), 1) ||
            !isSpace(op->getOperand(1), 1))
          return failure();
        if (mode == "widen")
          return success(dst.getElementType().isF32() &&
                         src.getElementType().isBF16());
        if (mode == "narrow")
          return success(dst.getElementType().isBF16() &&
                         src.getElementType().isF32());
        return failure();
      });

  add("bf16-scalar-convert", "bf16_scalar", "widen_bf16/round_bf16", "bf16.mlu",
      "scalar bf16 or f32 values", "scalar value", "f32", "pure scalar result",
      "IEEE round-to-nearest-even bf16 scalar conversion and product",
      [](Operation *op) {
        auto mode = op->getAttrOfType<StringAttr>("mode");
        if (!mode || op->getNumResults() != 1)
          return failure();
        auto res = op->getResult(0).getType();
        if (mode == "widen")
          return success(op->getNumOperands() == 1 &&
                         op->getOperand(0).getType().isBF16() && res.isF32());
        if (mode == "narrow")
          return success(op->getNumOperands() == 1 &&
                         op->getOperand(0).getType().isF32() && res.isBF16());
        if (mode == "product")
          return success(op->getNumOperands() == 2 &&
                         op->getOperand(0).getType().isBF16() &&
                         op->getOperand(1).getType().isBF16() && res.isBF16());
        return failure();
      });

  add("inclusive-scan", "scan_sum", "tile_prefix", "scan.mlu",
      "NRAM source, destination and scratch tiles of matching element type",
      "contiguous NRAM row fragment", "i32 or f32",
      "writes NRAM prefix destination and scratch",
      "Hillis-Steele power-of-two inclusive prefix sum in NRAM",
      [](Operation *op) {
        auto d = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto s = dyn_cast<MemRefType>(op->getOperand(1).getType());
        auto t = dyn_cast<MemRefType>(op->getOperand(2).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) &&
                       d.getElementType() == s.getElementType() &&
                       d.getElementType() == t.getElementType() &&
                       (d.getElementType().isF32() ||
                        d.getElementType().isInteger(32)));
      });

  add("moments-reduce", "summary_reduce", "welford_row", "moments.mlu",
      "NRAM f32 row fragment and active count", "contiguous NRAM row", "f32",
      "pure Moment tuple (n, mean, m2)",
      "sequential Welford moment accumulation with RN f32 division",
      [](Operation *op) {
        auto inTy = dyn_cast<MemRefType>(op->getOperand(0).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       inTy.getElementType().isF32() &&
                       isMomentTuple(op->getResult(0).getType()));
      });

  add("moments-merge", "summary_merge", "merge_moments", "moments.mlu",
      "two Moment tuples (n, mean, m2)", "scalar Moment struct", "f32",
      "pure Moment tuple result",
      "parallel Chan/Welford merge with RN f32 division",
      [](Operation *op) {
        return success(isMomentTuple(op->getOperand(0).getType()) &&
                       isMomentTuple(op->getOperand(1).getType()) &&
                       isMomentTuple(op->getResult(0).getType()));
      });

  add("philox4x32-linear", "philox_uniform", "philox4x32_linear", "philox.mlu",
      "NRAM f32 output, 8 KiB aligned NRAM u32 scratch, u64 seed and base",
      "contiguous logical counters [base .. base + live)", "u32/f32",
      "writes NRAM scratch and uniform f32 samples",
      "exact 10-round Philox-4x32 uniform float in [0, 1)",
      [](Operation *op) {
        auto dst = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto sc = dyn_cast<MemRefType>(op->getOperand(1).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       dst.getElementType().isF32() &&
                       sc.getElementType().isInteger(32) &&
                       op->getOperand(2).getType().getIntOrFloatBitWidth() ==
                           64);
      });

  add("philox4x32-element", "philox_at", "uniform_at", "philox.mlu",
      "u64 seed and u64 logical counter", "scalar value", "u32/f32",
      "pure scalar uniform f32 result",
      "exact 10-round Philox-4x32 uniform float in [0, 1)",
      [](Operation *op) {
        return success(
            op->getOperand(0).getType().getIntOrFloatBitWidth() == 64 &&
            op->getOperand(1).getType().getIntOrFloatBitWidth() == 64 &&
            op->getResult(0).getType().isF32());
      });
}
} // namespace CI_UNIT_NAMESPACE
