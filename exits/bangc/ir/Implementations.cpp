#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/storage/dsa/ir/Matrix.h"
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
  auto hasWram = [](const TargetProfile &p) -> llvm::Error {
    auto ms = p.services.get<intentdsl::dsa::MatrixStorage>();
    if (!p.services.get<intentdsl::dsa::LocalSupply>() || !ms ||
        ms->wramBytes <= 0)
      return llvm::createStringError("missing MLU370 NRAM/WRAM capabilities");
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
  auto addBuiltin = [&](StringRef name, StringRef op, StringRef symbol,
                        StringRef estimator, StringRef operands,
                        StringRef layout, StringRef acc, StringRef effects,
                        StringRef numerical,
                        std::function<llvm::Error(const TargetProfile &)> avail,
                        std::function<LogicalResult(Operation *)> verify) {
    ImplementationContract c;
    c.name = name.str();
    c.operation = ("bangc_src." + op).str();
    c.symbol = symbol.str();
    c.source = ImplementationSource::Builtin;
    c.minimum = Reuse::Native;
    c.operands = operands.str();
    c.layout = layout.str();
    c.accumulator = acc.str();
    c.effects = effects.str();
    c.numerical = numerical.str();
    c.estimator = estimator.str();
    c.costParameters = {"m", "n", "k"};
    c.available = std::move(avail);
    c.verify = std::move(verify);
    unit.implementations.push_back({std::move(c), COMPILERINFRA_SOURCE});
  };

  addBuiltin(
      "wram-conv-f16", "wram_conv", "__bang_conv/__bang_conv_partial",
      "exits/bangc::matrix-transfer",
      "NRAM f32[BM, BN] accumulator, NRAM f16[BM, BK] input, WRAM f16[BN, BK] "
      "filter and i1 accumulate flag",
      "NRAM row-major [BM, BK] x WRAM 64-channel filter [BN=64, BK]", "f32",
      "writes NRAM f32[BM, BN] accumulator",
      "MLU370 matrix-engine FP16xFP16 contraction accumulated in FP32", hasWram,
      [](Operation *op) {
        auto mode = op->getAttrOfType<StringAttr>("mode");
        auto bn = op->getAttrOfType<IntegerAttr>("bn");
        if (!mode || mode != "f16" || !bn || bn.getInt() != 64 ||
            op->getNumOperands() != 4)
          return failure();
        auto acc = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto lhs = dyn_cast<MemRefType>(op->getOperand(1).getType());
        auto flt = dyn_cast<MemRefType>(op->getOperand(2).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 3) &&
                       acc.getElementType().isF32() &&
                       lhs.getElementType().isF16() &&
                       flt.getElementType().isF16() &&
                       op->getOperand(3).getType().isInteger(1));
      });

  addBuiltin(
      "wram-conv-bf16", "wram_conv", "__bang_conv/__bang_conv_partial",
      "exits/bangc::matrix-transfer",
      "NRAM f32[BM, BN] accumulator, NRAM bf16[BM, BK] input, WRAM bf16[BN, "
      "BK] filter and i1 accumulate flag",
      "NRAM row-major [BM, BK] x WRAM 64-channel filter [BN=64, BK]", "f32",
      "writes NRAM f32[BM, BN] accumulator",
      "MLU370 matrix-engine BF16xBF16 contraction accumulated in FP32 over "
      "verified finite normal exponent window or kir.allow_subnormal_ftz",
      hasWram, [](Operation *op) {
        auto mode = op->getAttrOfType<StringAttr>("mode");
        auto bn = op->getAttrOfType<IntegerAttr>("bn");
        if (!mode || mode != "bf16" || !bn || bn.getInt() != 64 ||
            op->getNumOperands() != 4)
          return failure();
        auto acc = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto lhs = dyn_cast<MemRefType>(op->getOperand(1).getType());
        auto flt = dyn_cast<MemRefType>(op->getOperand(2).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 3) &&
                       acc.getElementType().isF32() &&
                       lhs.getElementType().isBF16() &&
                       flt.getElementType().isBF16() &&
                       op->getOperand(3).getType().isInteger(1));
      });

  addBuiltin(
      "wram-conv-i8", "wram_conv",
      "__bang_conv+__bang_float2int32_rn+__bang_add",
      "exits/bangc::matrix-transfer",
      "NRAM i32[BM, BN] accumulator, NRAM f32[BM, BN] partial scratch, NRAM "
      "i8[BM, BK] input and WRAM i8[BN, BK] filter",
      "NRAM row-major [BM, BK] x WRAM 64-channel filter [BN=64, BK<=1024]",
      "i32", "writes NRAM f32 partial scratch and NRAM i32[BM, BN] accumulator",
      "exact signed int8xint8 contraction (BK <= 1024 keeps |partial| <= 2^24 "
      "exact in binary32) accumulated in i32",
      hasWram, [](Operation *op) {
        auto mode = op->getAttrOfType<StringAttr>("mode");
        auto bn = op->getAttrOfType<IntegerAttr>("bn");
        auto bk = op->getAttrOfType<IntegerAttr>("bk");
        if (!mode || mode != "i8" || !bn || bn.getInt() != 64 || !bk ||
            bk.getInt() > 1024 || op->getNumOperands() != 4)
          return failure();
        auto acc = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto part = dyn_cast<MemRefType>(op->getOperand(1).getType());
        auto lhs = dyn_cast<MemRefType>(op->getOperand(2).getType());
        auto flt = dyn_cast<MemRefType>(op->getOperand(3).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) &&
                       isSpace(op->getOperand(3), 3) &&
                       acc.getElementType().isInteger(32) &&
                       part.getElementType().isF32() &&
                       lhs.getElementType().getIntOrFloatBitWidth() == 8 &&
                       flt.getElementType().getIntOrFloatBitWidth() == 8);
      });

  addBuiltin("cycle-mul", "cycle_mul", "__bang_cycle_mul",
             "exits/bangc::matrix-transfer",
             "NRAM f32 destination, NRAM f32 source and NRAM f32 cyclic vector",
             "contiguous NRAM f32 tile with periodic 1D multiplier", "f32",
             "writes NRAM f32 destination tile",
             "IEEE-754 binary32 elementwise cyclic product", hasNram,
             [](Operation *op) {
               auto dst = dyn_cast<MemRefType>(op->getOperand(0).getType());
               auto src = dyn_cast<MemRefType>(op->getOperand(1).getType());
               auto cyc = dyn_cast<MemRefType>(op->getOperand(2).getType());
               return success(isSpace(op->getOperand(0), 1) &&
                              isSpace(op->getOperand(1), 1) &&
                              isSpace(op->getOperand(2), 1) &&
                              dst.getElementType().isF32() &&
                              src.getElementType().isF32() &&
                              cyc.getElementType().isF32());
             });

  add("ieee-vec-min", "vec_min_scalar", "vec_min_scalar_ieee", "matrix.mlu",
      "NRAM f32 destination, NRAM f32 scratch, NRAM f32 input and scalar f32 "
      "bound",
      "64-lane aligned NRAM vector fragment", "f32",
      "writes NRAM f32 scratch and clamped NRAM f32 destination",
      "IEEE-754 minimumf(x, bound) preserving NaN propagation",
      [](Operation *op) {
        auto dst = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto sc = dyn_cast<MemRefType>(op->getOperand(1).getType());
        auto src = dyn_cast<MemRefType>(op->getOperand(2).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) &&
                       dst.getElementType().isF32() &&
                       sc.getElementType().isF32() &&
                       src.getElementType().isF32() &&
                       op->getOperand(3).getType().isF32());
      });

  add("bf16-panel-guard", "bf16_panel_ok", "ordinary_panel", "matrix.mlu",
      "NRAM bf16 panel and two NRAM u16 scratch buffers",
      "contiguous NRAM bf16 panel", "u16",
      "writes NRAM u16 scratch buffers and returns i1 finite-normal predicate",
      "certifies every BF16 element is zero or has normal exponent in "
      "[127-48, 127+46] so products avoid binary32 subnormals and overflow",
      [](Operation *op) {
        auto p = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto m = dyn_cast<MemRefType>(op->getOperand(1).getType());
        auto f = dyn_cast<MemRefType>(op->getOperand(2).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) &&
                       p.getElementType().getIntOrFloatBitWidth() == 16 &&
                       m.getElementType().getIntOrFloatBitWidth() == 16 &&
                       f.getElementType().getIntOrFloatBitWidth() == 16 &&
                       op->getResult(0).getType().isInteger(1));
      });

  add("bf16-ieee-dot", "bf16_ieee_dot", "bf16_ieee_dot_panel", "matrix.mlu",
      "NRAM f32[BM, BN] accumulator, NRAM bf16[BM, BK] lhs and NRAM bf16[BN, "
      "BK] transposed rhs",
      "pitched NRAM tiles [BM, BK] and [BN, BK]", "f32",
      "updates NRAM f32[BM, BN] accumulator in place",
      "exact IEEE-754 binary32 BF16 multiply-add preserving subnormals, signed "
      "zeros, Inf and NaN",
      [](Operation *op) {
        auto acc = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto lhs = dyn_cast<MemRefType>(op->getOperand(1).getType());
        auto rhs = dyn_cast<MemRefType>(op->getOperand(2).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) &&
                       acc.getElementType().isF32() &&
                       lhs.getElementType().isBF16() &&
                       rhs.getElementType().isBF16());
      });

  add("bf16-acc-narrow", "bf16_acc_narrow", "bf16_acc_narrow", "matrix.mlu",
      "NRAM bf16 destination, NRAM f32 accumulator and i1 strict flag",
      "contiguous NRAM tile", "f32", "writes NRAM bf16 destination",
      "round-to-nearest-even f32->bf16 preserving NaN payloads on strict path",
      [](Operation *op) {
        auto dst = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto acc = dyn_cast<MemRefType>(op->getOperand(1).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       dst.getElementType().isBF16() &&
                       acc.getElementType().isF32() &&
                       op->getOperand(3).getType().isInteger(1));
      });

  add("i32-to-bf16", "i32_to_bf16", "signed_to_bf16", "matrix.mlu",
      "scalar i32 value", "scalar value", "i32", "pure scalar bf16 result",
      "correctly rounded IEEE round-to-nearest-even signed i32 to bf16",
      [](Operation *op) {
        return success(op->getOperand(0).getType().isInteger(32) &&
                       op->getResult(0).getType().isBF16());
      });

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
      "NRAM f32 input, NRAM f32 maxpool scratch and NRAM f32 output over "
      "[members, pitch]",
      "128-byte aligned NRAM vector lanes", "f32",
      "writes NRAM f32 scratch and output buffer",
      "__bang_active_exphp in [-80, 80] with exact scalar expf repair outside",
      [](Operation *op) {
        auto outTy = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto scTy = dyn_cast<MemRefType>(op->getOperand(1).getType());
        auto inTy = dyn_cast<MemRefType>(op->getOperand(2).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) &&
                       outTy.getElementType().isF32() &&
                       scTy.getElementType().isF32() &&
                       inTy.getElementType().isF32());
      });

  add("validated-exp2", "row_exp2", "row_exp2", "exp.mlu",
      "NRAM f32 input, NRAM f32 maxpool scratch and NRAM f32 output over "
      "[members, pitch]",
      "128-byte aligned NRAM vector lanes", "f32",
      "writes NRAM f32 scratch and output buffer",
      "ln(2) scaling + __bang_active_exphp in [-115, 115] with exact scalar "
      "exp2f repair outside",
      [](Operation *op) {
        auto outTy = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto scTy = dyn_cast<MemRefType>(op->getOperand(1).getType());
        auto inTy = dyn_cast<MemRefType>(op->getOperand(2).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) &&
                       outTy.getElementType().isF32() &&
                       scTy.getElementType().isF32() &&
                       inTy.getElementType().isF32());
      });

  addBuiltin("reduce32-sum", "reduce32_sum", "__bang_reduce_sum",
             "exits/bangc::dma", "32-lane NRAM f32 buffer",
             "32-element aligned NRAM f32 vector", "f32",
             "reduces NRAM buffer in place and returns scalar f32 sum",
             "32-lane hardware tree sum in FP32", hasNram, [](Operation *op) {
               auto ty = dyn_cast<MemRefType>(op->getOperand(0).getType());
               return success(isSpace(op->getOperand(0), 1) &&
                              ty.getElementType().isF32() &&
                              op->getResult(0).getType().isF32());
             });

  addBuiltin("nram-sumpool", "sumpool", "__bang_sumpool", "exits/bangc::dma",
             "NRAM f32 destination [channel] and NRAM f32 source "
             "[height, channel]",
             "channel-last NRAM 2D f32 tile", "f32",
             "writes NRAM f32 channel sums",
             "batched NRAM column sumpool in FP32", hasNram, [](Operation *op) {
               auto dst = dyn_cast<MemRefType>(op->getOperand(0).getType());
               auto src = dyn_cast<MemRefType>(op->getOperand(1).getType());
               return success(isSpace(op->getOperand(0), 1) &&
                              isSpace(op->getOperand(1), 1) &&
                              dst.getElementType().isF32() &&
                              src.getElementType().isF32());
             });

  add("nram-transpose", "tile_transpose",
      "transpose_tile_b16/transpose_tile_b32", "transpose.mlu",
      "NRAM source tile [rows, cols] and NRAM destination tile [cols, rows]",
      "64-byte aligned 2D NRAM tile", "none",
      "writes transposed NRAM tile",
      "bit-exact 2D NRAM element permutation via __bang_transpose",
      [](Operation *op) {
        auto dstTy = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto srcTy = dyn_cast<MemRefType>(op->getOperand(1).getType());
        if (!isSpace(op->getOperand(0), 1) || !isSpace(op->getOperand(1), 1) ||
            dstTy.getElementType() != srcTy.getElementType())
          return failure();
        unsigned bits = dstTy.getElementTypeBitWidth();
        return success(bits == 8 || bits == 16 || bits == 32);
      });

  add("certified-half-quotient", "narrow_quotient", "group_narrow_quotient",
      "quotient.mlu",
      "NRAM f32 numerator, per-member f32 divisor, NRAM f32 quotient/scratch, "
      "NRAM f16 output",
      "pitched rows [members, pitch] in NRAM", "f32",
      "writes NRAM quotient/scratch and certified f16 output",
      "+-2 ULP f32 endpoint certification with exact RN f32->f16 fallback",
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

  add("bf16-tile-product", "bf16_product", "bf16_product_tile", "bf16.mlu",
      "NRAM bf16 destination, two NRAM bf16 inputs and NRAM f32 scratch",
      "64-lane aligned NRAM vector fragment", "f32",
      "writes NRAM f32 scratch and destination NRAM bf16 tile",
      "correctly rounded IEEE-754 binary32 product narrowed with NaN-preserving "
      "bf16 round-to-nearest-even",
      [](Operation *op) {
        auto dst = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto lhs = dyn_cast<MemRefType>(op->getOperand(1).getType());
        auto rhs = dyn_cast<MemRefType>(op->getOperand(2).getType());
        auto sc = dyn_cast<MemRefType>(op->getOperand(3).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) &&
                       isSpace(op->getOperand(3), 1) &&
                       dst.getElementType().isBF16() &&
                       lhs.getElementType().isBF16() &&
                       rhs.getElementType().isBF16() &&
                       sc.getElementType().isF32());
      });

  add("ieee-sqrt-rn", "vec_sqrt_rn", "vec_sqrt_rn", "ieee_math.mlu",
      "NRAM f32 destination and NRAM f32 input",
      "64-lane aligned NRAM vector fragment", "f32",
      "writes NRAM f32 destination tile",
      "correctly rounded IEEE-754 binary32 round-to-nearest-even (0 ULP)",
      [](Operation *op) {
        auto dst = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto src = dyn_cast<MemRefType>(op->getOperand(1).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       dst.getElementType().isF32() &&
                       src.getElementType().isF32());
      });

  add("ieee-div-rn", "vec_div_rn",
      "vec_div_rn/vec_div_scalar_rn/scalar_div_vec_rn", "ieee_math.mlu",
      "NRAM f32 destination, NRAM or scalar f32 dividend and divisor, and NRAM "
      "f32 scratch",
      "64-lane aligned NRAM vector fragment with 6x lanes f32 scratch", "f32",
      "writes NRAM f32 scratch and correctly rounded NRAM f32 quotient",
      "correctly rounded IEEE-754 binary32 round-to-nearest-even (0 ULP): "
      "exponent scaling + reciprocal Newton-Raphson + 48-bit "
      "Veltkamp-Dekker/TwoDiff residual + endpoint certification",
      [](Operation *op) {
        auto mode = op->getAttrOfType<StringAttr>("mode");
        auto dst = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto sc = dyn_cast<MemRefType>(op->getOperand(3).getType());
        if (!mode || !isSpace(op->getOperand(0), 1) ||
            !isSpace(op->getOperand(3), 1) || !dst.getElementType().isF32() ||
            !sc.getElementType().isF32())
          return failure();
        if (mode == "vv") {
          auto l = dyn_cast<MemRefType>(op->getOperand(1).getType());
          auto r = dyn_cast<MemRefType>(op->getOperand(2).getType());
          return success(isSpace(op->getOperand(1), 1) &&
                         isSpace(op->getOperand(2), 1) &&
                         l.getElementType().isF32() &&
                         r.getElementType().isF32());
        }
        if (mode == "vs") {
          auto l = dyn_cast<MemRefType>(op->getOperand(1).getType());
          return success(isSpace(op->getOperand(1), 1) &&
                         l.getElementType().isF32() &&
                         op->getOperand(2).getType().isF32());
        }
        if (mode == "sv") {
          auto r = dyn_cast<MemRefType>(op->getOperand(2).getType());
          return success(op->getOperand(1).getType().isF32() &&
                         isSpace(op->getOperand(2), 1) &&
                         r.getElementType().isF32());
        }
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
      "NRAM f32 row fragment, NRAM f32 sumpool scratch and active count",
      "contiguous NRAM row", "f32",
      "writes NRAM f32 scratch and returns Moment tuple (n, mean, m2)",
      "vectorized two-pass sumpool Welford moment accumulation with RN f32 "
      "division and sequential non-finite fallback",
      [](Operation *op) {
        auto inTy = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto scTy = dyn_cast<MemRefType>(op->getOperand(1).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       inTy.getElementType().isF32() &&
                       scTy.getElementType().isF32() &&
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
