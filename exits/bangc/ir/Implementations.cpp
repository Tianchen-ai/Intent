#include "Alignment.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"
#include "intentdsl/levels/algorithm/kir/ir/StorageLayout.h"
#include "intentdsl/levels/storage/dsa/ir/Matrix.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

TypedContract makeTyped(std::optional<unsigned> minIn,
                        std::optional<unsigned> maxIn, unsigned numOut,
                        std::vector<ScalarTypeKind> inTypes,
                        std::vector<ScalarTypeKind> outTypes,
                        std::vector<ScalarTypeKind> accTypes,
                        EffectMask effects, uint64_t minAlign = 64,
                        std::optional<uint64_t> space = 1) {
  TypedContract tc;
  tc.shape.minOperands = minIn;
  tc.shape.maxOperands = maxIn;
  tc.shape.minResults = numOut;
  tc.shape.maxResults = numOut;
  tc.operandTypes = std::move(inTypes);
  tc.resultTypes = std::move(outTypes);
  tc.accumulatorTypes = std::move(accTypes);
  tc.layout.minAlignmentBytes = minAlign;
  tc.layout.knownTargetAlignment = intentdsl::bangc::storageAlignment;
  tc.layout.requiredMemorySpace = space;
  tc.effects = effects;
  tc.verifyEffectsInterface = true;
  return tc;
}
} // namespace

void rowContracts(Unit &unit) {
  using STK = ScalarTypeKind;
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
                 TypedContract typed,
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
    c.typed = std::move(typed);
    unit.implementations.push_back({std::move(c), COMPILERINFRA_SOURCE});
  };
  auto addBuiltin = [&](StringRef name, StringRef op, StringRef symbol,
                        StringRef estimator, StringRef operands,
                        StringRef layout, StringRef acc, StringRef effects,
                        StringRef numerical, TypedContract typed,
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
    c.typed = std::move(typed);
    unit.implementations.push_back({std::move(c), COMPILERINFRA_SOURCE});
  };

  addBuiltin(
      "native-row-atomic-add", "atomic_add", "__bang_atomic_add",
      "exits/bangc::dma",
      "old NRAM row, global destination row, NRAM increments, length",
      "contiguous elements", "destination element type",
      "reads and updates GDRAM, writes old values to NRAM",
      "per-element atomic addition, BANG C Vector Operation Functions, mtp_372",
      makeTyped(4, 4, 0, {STK::F32, STK::I32, STK::I64}, {}, {},
                EffectMask::ReadWrite, 1, std::nullopt),
      hasNram, [](Operation *op) {
        auto type =
            cast<MemRefType>(op->getOperand(1).getType()).getElementType();
        return success(
            (type.isF32() || type.isInteger(32)) &&
            isSpace(op->getOperand(0), 1) && isSpace(op->getOperand(1), 0) &&
            isSpace(op->getOperand(2), 1) &&
            cast<MemRefType>(op->getOperand(0).getType()).getElementType() ==
                type &&
            cast<MemRefType>(op->getOperand(2).getType()).getElementType() ==
                type);
      });

  addBuiltin(
      "wram-conv-f16", "wram_conv", "__bang_conv/__bang_conv_partial",
      "exits/bangc::matrix-transfer",
      "NRAM f32[BM, BN] accumulator, NRAM f16[BM, BK] input, WRAM f16[BN, BK] "
      "filter and i1 accumulate flag",
      "NRAM row-major [BM, BK] x WRAM 64-channel filter [BN=64, BK]", "f32",
      "writes NRAM f32[BM, BN] accumulator",
      "MLU370 matrix-engine FP16xFP16 contraction accumulated in FP32",
      makeTyped(4, 4, 0, {STK::F32, STK::F16, STK::I1}, {}, {STK::F32},
                EffectMask::ReadWrite, 64, std::nullopt),
      hasWram, [](Operation *op) {
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

  {
    TypedContract bf16ConvTyped =
        makeTyped(4, 4, 0, {STK::F32, STK::BF16, STK::I1}, {}, {STK::F32},
                  EffectMask::ReadWrite, 64, std::nullopt);
    bf16ConvTyped.precondition =
        [](Operation *op, const TargetProfile &target) -> llvm::Error {
      if (allowsSubnormalFlush(op, target.numerical))
        return llvm::Error::success();
      if (op->getName().getStringRef() != "bangc_src.wram_conv")
        return llvm::Error::success();
      auto guarded = op->getAttrOfType<BoolAttr>("bangc.guarded_bf16_window");
      for (Operation *cur = op->getParentOp(); cur; cur = cur->getParentOp()) {
        if (auto parentIf = dyn_cast<scf::IfOp>(cur)) {
          bool hasPanelGuard = false;
          if (Block *enclosing = parentIf->getBlock()) {
            for (Operation &prev : *enclosing) {
              if (&prev == parentIf.getOperation())
                break;
              if (prev.getName().getStringRef() == "bangc_src.bf16_panel_ok")
                hasPanelGuard = true;
            }
          }
          if (guarded && guarded.getValue() && hasPanelGuard)
            return llvm::Error::success();
        }
      }
      return llvm::createStringError(
          "wram-conv-bf16 requires function denormal-fp-math FTZ permission or "
          "an explicit bf16_panel_ok normal-exponent guard");
    };
    addBuiltin(
        "wram-conv-bf16", "wram_conv", "__bang_conv/__bang_conv_partial",
        "exits/bangc::matrix-transfer",
        "NRAM f32[BM, BN] accumulator, NRAM bf16[BM, BK] input, WRAM bf16[BN, "
        "BK] filter and i1 accumulate flag",
        "NRAM row-major [BM, BK] x WRAM 64-channel filter [BN=64, BK]", "f32",
        "writes NRAM f32[BM, BN] accumulator",
        "MLU370 matrix-engine BF16xBF16 contraction accumulated in FP32 over "
        "verified finite normal exponent window or denormal-fp-math FTZ",
        std::move(bf16ConvTyped), hasWram, [](Operation *op) {
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
  }

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
      makeTyped(4, 4, 0, {STK::I32, STK::F32, STK::I8, STK::UI8}, {},
                {STK::I32}, EffectMask::ReadWrite, 64, std::nullopt),
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
             "IEEE-754 binary32 elementwise cyclic product",
             makeTyped(5, 5, 0, {STK::F32, STK::I64}, {}, {STK::F32},
                       EffectMask::ReadWrite, 64, 1),
             hasNram, [](Operation *op) {
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
      makeTyped(6, 6, 0, {STK::F32, STK::I64}, {}, {STK::F32},
                EffectMask::ReadWrite, 64, 1),
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
      makeTyped(4, 4, 1, {STK::BF16, STK::I16, STK::I64}, {STK::I1}, {STK::I16},
                EffectMask::ReadWrite, 64, 1),
      [](Operation *op) {
        auto p = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto m = dyn_cast<MemRefType>(op->getOperand(1).getType());
        auto f = dyn_cast<MemRefType>(op->getOperand(2).getType());
        auto capacity = op->getAttrOfType<IntegerAttr>("scratch_elements");
        auto fits = [&](Value v) {
          auto *alloc = v.getDefiningOp();
          int64_t byteOffset = 0;
          if (alloc && alloc->getName().stripDialect() == "nram_ptr") {
            auto c = alloc->getOperand(1).getDefiningOp<arith::ConstantIntOp>();
            if (!c || c.value() < 0)
              return false;
            byteOffset = c.value();
            alloc = alloc->getOperand(0).getDefiningOp();
          }
          if (!alloc || alloc->getName().stripDialect() != "nram_buffer")
            return false;
          auto size = alloc->getAttrOfType<IntegerAttr>("elements");
          auto width = cast<MemRefType>(alloc->getResult(0).getType())
                           .getElementTypeBitWidth() /
                       8;
          int64_t bytes;
          return size && capacity && capacity.getInt() >= 64 &&
                 capacity.getInt() % 64 == 0 &&
                 !__builtin_mul_overflow(size.getInt(), int64_t(width),
                                         &bytes) &&
                 byteOffset <= bytes &&
                 capacity.getInt() <= (bytes - byteOffset) / 2;
        };
        return success(fits(op->getOperand(1)) && fits(op->getOperand(2)) &&
                       isSpace(op->getOperand(0), 1) &&
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
      makeTyped(8, 8, 0, {STK::F32, STK::BF16, STK::I64}, {}, {STK::F32},
                EffectMask::ReadWrite, 64, 1),
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
      makeTyped(4, 4, 0, {STK::BF16, STK::F32, STK::I64, STK::I1}, {},
                {STK::F32}, EffectMask::ReadWrite, 64, 1),
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
      makeTyped(1, 1, 1, {STK::I32}, {STK::BF16}, {STK::I32}, EffectMask::None,
                0, std::nullopt),
      [](Operation *op) {
        return success(op->getOperand(0).getType().isInteger(32) &&
                       op->getResult(0).getType().isBF16());
      });

  add("row-reduce", "row_reduce", "group_sum/group_max_num", "reduce.mlu",
      "NRAM f32 pitched input, NRAM f32 scratch, per-member f32 output",
      "64-byte aligned NRAM input and scratch, naturally aligned per-member f32 output", "f32",
      "writes NRAM scratch and per-member f32 reduction results",
      "sumpool+reduce_sum or maxpool+argmax with IEEE NaN/subnormal fallback",
      makeTyped(6, 6, 0, {STK::F32, STK::I64}, {}, {STK::F32},
                EffectMask::ReadWrite, 4, std::nullopt),
      [](Operation *op) {
        auto alg = op->getAttrOfType<StringAttr>("algebra");
        auto inTy = dyn_cast<MemRefType>(op->getOperand(2).getType());
        auto scTy = dyn_cast<MemRefType>(op->getOperand(1).getType());
        auto outTy = dyn_cast<MemRefType>(op->getOperand(0).getType());
        return success(alg && (alg == "sum" || alg == "max_num") &&
                       intentdsl::bangc::storageAlignment(op->getOperand(1)) >= 64 &&
                       intentdsl::bangc::storageAlignment(op->getOperand(2)) >= 64 &&
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
      makeTyped(5, 5, 0, {STK::F32, STK::I64}, {}, {STK::F32},
                EffectMask::ReadWrite, 64, 1),
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
      makeTyped(5, 5, 0, {STK::F32, STK::I64}, {}, {STK::F32},
                EffectMask::ReadWrite, 64, 1),
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

  add("validated-log", "row_log", "row_log", "exp.mlu",
      "NRAM f32 input, NRAM f32 scratch and NRAM f32 output",
      "128-byte aligned NRAM vector lanes", "f32",
      "writes NRAM f32 scratch and output buffer",
      "__bang_active_loghp in [2^-120, 2^+120) with exact scalar logf repair "
      "outside",
      makeTyped(5, 5, 0, {STK::F32, STK::I64}, {}, {STK::F32},
                EffectMask::ReadWrite, 64, 1),
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

  add("ieee-vec-cmpf", "vec_cmpf_mask", "vec_cmpf_mask_vs/vec_cmpf_mask_vv",
      "exp.mlu",
      "NRAM i32 mask output, NRAM scratch, NRAM f32 lhs and f32/NRAM rhs",
      "64-lane aligned NRAM vector fragment", "i32",
      "writes NRAM scratch and NRAM i32 all-ones/zero lane mask",
      "IEEE-754 binary32 ordered/unordered predicate preserving subnormals, "
      "signed zero, Inf and NaN",
      makeTyped(5, 5, 0, {STK::I32, STK::F32, STK::I64}, {}, {STK::I32},
                EffectMask::ReadWrite, 64, 1),
      [](Operation *op) {
        auto mTy = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto lTy = dyn_cast<MemRefType>(op->getOperand(2).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) && mTy && lTy &&
                       mTy.getElementType().isInteger(32) &&
                       lTy.getElementType().isF32());
      });

  add("ieee-vec-bitselect", "vec_bitselect",
      "vec_bitselect_vv/vec_bitselect_vs/vec_bitselect_sv", "exp.mlu",
      "NRAM f32 destination, NRAM scratch, NRAM i32 mask and f32/NRAM arms",
      "64-lane aligned NRAM vector fragment", "i32",
      "writes NRAM scratch and bitwise-selected NRAM f32 destination",
      "exact 32-bit lane selection immune to NaN/Inf on unselected branch",
      makeTyped(6, 6, 0, {STK::F32, STK::I32, STK::I64}, {}, {STK::I32},
                EffectMask::ReadWrite, 64, 1),
      [](Operation *op) {
        auto dTy = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto mTy = dyn_cast<MemRefType>(op->getOperand(2).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) && dTy && mTy &&
                       dTy.getElementType().isF32() &&
                       mTy.getElementType().isInteger(32));
      });

  add("quant-lut-init", "quant_init_lut", "quant_init_byte_lut_f16",
      "quant.mlu", "256-entry NRAM i32 byte-pair LUT and format parameters",
      "aligned 256-entry NRAM LUT", "i32", "writes 256-entry NRAM LUT",
      "exact 256-byte to half2 codebook initialization from format descriptor",
      makeTyped(6, 6, 0, {STK::I32}, {}, {STK::I32}, EffectMask::Write, 64, 1),
      [](Operation *op) {
        auto lut = dyn_cast<MemRefType>(op->getOperand(0).getType());
        return success(isSpace(op->getOperand(0), 1) && lut &&
                       lut.getElementType().isInteger(32));
      });

  add("quant-record-lut-init", "quant_init_record_lut",
      "quant_init_record_2plane_lut", "quant.mlu",
      "1536-entry NRAM i16 bit-plane LUT and plane shift/width parameters",
      "aligned NRAM i16 LUT table", "i16", "writes NRAM i16 bit-plane LUT",
      "exact multi-plane integer unpack LUT initialization from format "
      "descriptor",
      makeTyped(7, 7, 0, {STK::I16, STK::I32}, {}, {STK::I16},
                EffectMask::Write, 64, 1),
      [](Operation *op) {
        auto lut = dyn_cast<MemRefType>(op->getOperand(0).getType());
        return success(isSpace(op->getOperand(0), 1) && lut &&
                       lut.getElementType().isInteger(16));
      });

  add("quant-e8m0-guard", "quant_e8m0_ok", "quant_e8m0_panel_ok", "quant.mlu",
      "two NRAM u8 E8M0 scale panels and NRAM scratch",
      "contiguous NRAM u8 scale panels", "i16",
      "writes NRAM scratch and returns i1 finite FP16-representable flag",
      "certifies E8M0 block scales lie in exact FP16 normal exponent window",
      makeTyped(7, 7, 1,
                {STK::F16, STK::I8, STK::UI8, STK::I16, STK::I32, STK::I64},
                {STK::I1}, {STK::I16}, EffectMask::ReadWrite, 64, 1),
      [](Operation *op) {
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(2), 1) &&
                       isSpace(op->getOperand(4), 1) &&
                       op->getResult(0).getType().isInteger(1));
      });

  add("quant-block-scaled", "quant_decode_block_scaled",
      "quant_decode_block_scaled_f16", "quant.mlu",
      "NRAM f16 panel output, NRAM packed bytes, NRAM E8M0 scales, LUT and "
      "NRAM scratch",
      "NRAM 2D tile [padded_rows, groups_per_panel * block_elems]", "f32",
      "writes NRAM scratch and decoded NRAM f16 operand panel",
      "exact __bang_lut nibble unpack + exact power-of-two FP16 block scaling",
      makeTyped(13, 13, 0,
                {STK::F16, STK::I8, STK::UI8, STK::I32, STK::I64}, {},
                {STK::F32}, EffectMask::ReadWrite, 64, 1),
      [](Operation *op) {
        auto dst = dyn_cast<MemRefType>(op->getOperand(0).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) &&
                       isSpace(op->getOperand(3), 1) && dst &&
                       dst.getElementType().isF16());
      });

  add("quant-grouped-rhs", "quant_decode_grouped_rhs",
      "quant_decode_grouped_rhs_f16", "quant.mlu",
      "NRAM f16 filter-major rhs output, NRAM packed weights, NRAM f16 scales, "
      "NRAM packed zeros, LUT and NRAM scratch",
      "NRAM filter-major [BN=64, BK] f16 panel", "f32",
      "writes NRAM scratch and decoded filter-major f16 rhs panel",
      "__bang_lut 4-bit unpack + lane permutation + grouped zero-point & scale",
      makeTyped(15, 15, 0,
                {STK::F16, STK::I8, STK::UI8, STK::I32, STK::I64,
                 STK::AnyInteger},
                {}, {STK::F32}, EffectMask::ReadWrite, 64, 1),
      [](Operation *op) {
        auto dst = dyn_cast<MemRefType>(op->getOperand(0).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) &&
                       isSpace(op->getOperand(3), 1) &&
                       isSpace(op->getOperand(4), 1) && dst &&
                       dst.getElementType().isF16());
      });

  add("quant-record-gemv", "quant_record_gemv", "quant_record_gemv_2plane_q8k",
      "quant.mlu",
      "NRAM f32[64] accumulator, NRAM 64-row record panel, NRAM quantized "
      "activation, bit-plane LUT and NRAM scratch",
      "64-row pitched 256B NRAM record panel", "f32",
      "updates NRAM f32[64] row accumulator in place",
      "exact 2-plane __bang_lut unpack + __bang_sumpool sub-block integer dot "
      "+ super-block FP32 accumulation",
      makeTyped(11, 11, 0,
                {STK::F32, STK::I8, STK::UI8, STK::I16, STK::I32, STK::I64}, {},
                {STK::F32}, EffectMask::ReadWrite, 64, 1),
      [](Operation *op) {
        auto acc = dyn_cast<MemRefType>(op->getOperand(0).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) &&
                       isSpace(op->getOperand(3), 1) &&
                       isSpace(op->getOperand(4), 1) && acc &&
                       acc.getElementType().isF32());
      });

  addBuiltin("reduce32-sum", "reduce32_sum", "__bang_reduce_sum",
             "exits/bangc::dma", "32-lane NRAM f32 buffer",
             "32-element aligned NRAM f32 vector", "f32",
             "reduces NRAM buffer in place and returns scalar f32 sum",
             "32-lane hardware tree sum in FP32",
             makeTyped(1, 1, 1, {STK::F32}, {STK::F32}, {STK::F32},
                       EffectMask::ReadWrite, 64, 1),
             hasNram, [](Operation *op) {
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
             "batched NRAM column sumpool in FP32",
             makeTyped(4, 4, 0, {STK::F32, STK::I64}, {}, {STK::F32},
                       EffectMask::ReadWrite, 64, 1),
             hasNram, [](Operation *op) {
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
      "64-byte aligned 2D NRAM tile", "none", "writes transposed NRAM tile",
      "bit-exact 2D NRAM element permutation via __bang_transpose",
      makeTyped(4, 4, 0,
                {STK::I8, STK::UI8, STK::I16, STK::F16, STK::BF16, STK::I32,
                 STK::F32, STK::I64},
                {}, {}, EffectMask::ReadWrite, 64, 1),
      [](Operation *op) {
        auto dstTy = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto srcTy = dyn_cast<MemRefType>(op->getOperand(1).getType());
        if (!isSpace(op->getOperand(0), 1) || !isSpace(op->getOperand(1), 1) ||
            dstTy.getElementType() != srcTy.getElementType())
          return failure();
        unsigned bits = dstTy.getElementTypeBitWidth();
        return success(bits == 8 || bits == 16 || bits == 32);
      });

  add("bf16-tile-convert", "bf16_tile",
      "widen_bf16_tile/narrow_bf16_tile/narrow_bf16_tile_fast", "bf16.mlu",
      "NRAM bf16 and f32 tile buffers (with optional u16 scratch)",
      "64-lane aligned NRAM vector fragment", "f32",
      "writes destination NRAM tile",
      "NaN-preserving bf16<->f32 round-to-nearest-even",
      makeTyped(4, 5, 0, {STK::BF16, STK::F32, STK::I16, STK::I64}, {},
                {STK::F32}, EffectMask::ReadWrite, 64, 1),
      [](Operation *op) {
        auto mode = op->getAttrOfType<StringAttr>("mode");
        if (!mode || (op->getNumOperands() != 4 && op->getNumOperands() != 5))
          return failure();
        auto dst = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto src = dyn_cast<MemRefType>(op->getOperand(1).getType());
        if (!isSpace(op->getOperand(0), 1) || !isSpace(op->getOperand(1), 1))
          return failure();
        if (mode == "widen")
          return success(op->getNumOperands() == 4 &&
                         dst.getElementType().isF32() &&
                         src.getElementType().isBF16());
        if (mode == "narrow") {
          if (op->getNumOperands() == 5) {
            auto sc = dyn_cast<MemRefType>(op->getOperand(2).getType());
            if (!isSpace(op->getOperand(2), 1) ||
                sc.getElementType().getIntOrFloatBitWidth() != 16)
              return failure();
          }
          return success(dst.getElementType().isBF16() &&
                         src.getElementType().isF32());
        }
        return failure();
      });

  add("bf16-tile-product", "bf16_product", "bf16_product_tile", "bf16.mlu",
      "NRAM bf16 destination, two NRAM bf16 inputs and NRAM f32 scratch",
      "64-lane aligned NRAM vector fragment", "f32",
      "writes NRAM f32 scratch and destination NRAM bf16 tile",
      "correctly rounded IEEE-754 binary32 product narrowed with NaN-preserving "
      "bf16 round-to-nearest-even",
      makeTyped(6, 6, 0, {STK::BF16, STK::F32, STK::I64}, {}, {STK::F32},
                EffectMask::ReadWrite, 64, 1),
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

  auto approximate = [&](unsigned operands, arith::FastMathFlags permission, bool ftz = true) {
    auto tc = makeTyped(operands, operands, 0, {STK::F32, STK::I64}, {}, {STK::F32},
                        EffectMask::ReadWrite, 64, 1);
    tc.numerical.requiredFastMath = permission;
    tc.numerical.requireInputSubnormalFlush = ftz;
    tc.numerical.requireOutputSubnormalFlush = ftz;
    return tc;
  };
  auto floatNram = [](Value v) {
    auto t = dyn_cast<MemRefType>(v.getType());
    return t && t.getElementType().isF32() && isSpace(v, 1);
  };
  addBuiltin("approximate-sqrt", "vec_sqrt", "__bang_sqrt", "exits/bangc::dma",
             "NRAM f32 destination and source vector, lane count",
             "contiguous NRAM vectors", "f32", "reads source and writes destination",
             "MLU vector square root under afn and input/output FTZ permission",
             approximate(3, arith::FastMathFlags::afn), hasNram, [=](Operation *op) {
               return success(floatNram(op->getOperand(0)) && floatNram(op->getOperand(1)));
             });
  add("reciprocal-product", "vec_div_approx", "reciprocal_product", "reciprocal.mlu",
      "NRAM f32 destination, vector or scalar f32 operands, reciprocal scratch, lane count",
      "contiguous NRAM vectors", "f32", "reads operands and writes destination and vector scratch",
      "multiply by reciprocal under arcp and input/output FTZ permission",
      approximate(5, arith::FastMathFlags::arcp), [=](Operation *op) {
        auto a = op->getOperand(1), b = op->getOperand(2);
        return success(floatNram(op->getOperand(0)) && floatNram(op->getOperand(3)) &&
                       (floatNram(a) || a.getType().isF32()) &&
                       (floatNram(b) || b.getType().isF32()) && (floatNram(a) || floatNram(b)));
      });

  add("guarded-reciprocal-product", "vec_guarded_div", "guarded_reciprocal_product", "guarded_reciprocal.mlu",
      "NRAM f32 quotient and operands, five scratch tiles, live and padded lanes",
      "contiguous NRAM", "f32", "read operands, write scratch and quotient",
      "LLVM arcp: reciprocal multiply for normal lanes; SDK RN for exceptional lanes",
      approximate(6, arith::FastMathFlags::arcp, false), [=](Operation *op) {
        auto a = op->getOperand(1), b = op->getOperand(2);
        return success(floatNram(op->getOperand(0)) && floatNram(op->getOperand(3)) &&
                       (floatNram(a) || a.getType().isF32()) && (floatNram(b) || b.getType().isF32()) &&
                       (floatNram(a) || floatNram(b)));
      });
  add("ieee-sqrt-rn", "vec_sqrt_rn", "vec_sqrt_rn", "ieee_math.mlu",
      "NRAM f32 destination, NRAM f32 input and NRAM f32 scratch",
      "64-lane aligned NRAM vector fragment with 6x lanes f32 scratch", "f32",
      "writes NRAM f32 scratch and correctly rounded NRAM f32 square root",
      "IEEE-754 squareRoot roundTiesToEven (0 ULP); cncc bang_math_decls.h vector sqrt is only a candidate: "
      "correctness guaranteed solely by unconditional symmetric exact runtime "
      "endpoint certification on exact Veltkamp-Dekker/TwoDiff residual "
      "x - r^2 with bit-exact integer RN-even fallback (a priori candidate "
      "accuracy only bounds fallback frequency)",
      makeTyped(5, 5, 0, {STK::F32, STK::I64}, {}, {STK::F32},
                EffectMask::ReadWrite, 64, 1),
      [](Operation *op) {
        if (op->getNumOperands() != 5)
          return failure();
        auto dst = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto src = dyn_cast<MemRefType>(op->getOperand(1).getType());
        auto sc = dyn_cast<MemRefType>(op->getOperand(2).getType());
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) &&
                       dst.getElementType().isF32() &&
                       src.getElementType().isF32() &&
                       sc.getElementType().isF32());
      });

  for (StringRef mode : {"vv", "vs", "sv"}) {
    add(("ieee-div-rn-" + mode).str(), "vec_div_rn", "certified_div_rn", "certified_div.mlu",
        "NRAM f32 quotient, vector or scalar operands, six scratch tiles, live and padded lanes",
        "aligned contiguous NRAM", "f32", "read operands and write scratch and quotient",
        "IEEE-754 binary32 roundTiesToEven (0 ULP); BANG C bang_math_decls.h "
        "__cn_scalar_div_f32_rn. Certify the reciprocal/Newton candidate against "
        "both rounding boundaries; repair uncertified lanes with SDK RN.",
        makeTyped(6, 6, 0, {STK::F32, STK::I64}, {}, {STK::F32}, EffectMask::ReadWrite, 64, 1),
        [mode, floatNram](Operation *op) {
                 if (op->getAttrOfType<StringAttr>("mode") != mode ||
                     !floatNram(op->getOperand(0)) || !floatNram(op->getOperand(3)))
                   return failure();
                 for (unsigned i : {0u, 1u}) {
                   Value v = op->getOperand(i + 1);
                   if (mode[i] == 's' ? !v.getType().isF32() : !floatNram(v))
                     return failure();
                 }
                 return success();
               });
  }
  addBuiltin("ieee-div-rn", "div_rn", "__cn_scalar_div_f32_rn",
             "exits/bangc::dma", "f32 dividend and divisor", "scalar", "f32",
             "pure quotient", "IEEE binary32 round to nearest, ties to even",
             makeTyped(2, 2, 1, {STK::F32}, {STK::F32}, {STK::F32},
                       EffectMask::None, 0, std::nullopt),
             hasNram, [](Operation *) { return success(); });

  add("bf16-scalar-convert", "bf16_scalar", "widen_bf16/round_bf16", "bf16.mlu",
      "scalar bf16 or f32 values", "scalar value", "f32", "pure scalar result",
      "IEEE round-to-nearest-even bf16 scalar conversion and product",
      makeTyped(1, 2, 1, {STK::BF16, STK::F32}, {STK::BF16, STK::F32},
                {STK::F32}, EffectMask::None, 0, std::nullopt),
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

  add("wram-key-count", "key_histogram", "tile_key_histogram", "histogram.mlu",
      "i32 bins, f32 or i64 samples, i16 keys and WRAM comparison rows",
      "64 bins per WRAM tile", "i32", "writes private counts and encoded keys",
      "integer equality after truncation toward zero; invalid keys skipped",
      makeTyped(7, 7, 0, {STK::I32, STK::F32, STK::I64, STK::I16}, {},
                {STK::I32}, EffectMask::ReadWrite, 64, std::nullopt),
      [](Operation *op) {
        auto elem = [&](int i) {
          return cast<MemRefType>(op->getOperand(i).getType()).getElementType();
        };
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       isSpace(op->getOperand(2), 1) &&
                       isSpace(op->getOperand(3), 3) && elem(0).isInteger(32) &&
                       (elem(1).isF32() || elem(1).isInteger(64)) &&
                       isSpace(op->getOperand(4), 1) && elem(4).isInteger(16) &&
                       elem(2).isInteger(16) && elem(3).isInteger(16));
      });
  add("finite-panel", "finite_panel", "finite_narrow_panel", "matrix.mlu",
      "narrow source, 4096-element u16 workspace, count, exponent mask",
      "contiguous NRAM", "u16", "writes temporary exponent mask",
      "checks the exponent bits for infinity and NaN",
      makeTyped(4, 4, 1, {STK::F16, STK::BF16, STK::I64, STK::I32}, {STK::I1},
                {STK::I16}, EffectMask::ReadWrite, 64, 1),
      [](Operation *op) {
        auto t = cast<MemRefType>(op->getOperand(0).getType()).getElementType();
        return success(isSpace(op->getOperand(0), 1) &&
                       isSpace(op->getOperand(1), 1) &&
                       (t.isF16() || t.isBF16()) &&
                       cast<MemRefType>(op->getOperand(1).getType())
                               .getElementType()
                               .getIntOrFloatBitWidth() == 16);
      });
  add("packed-nibble-dot", "record_dot", "packed_nibble_dot", "record.mlu",
      "accumulator, record panel, quantized activation, group sums, workspace, "
      "scale and rows",
      "64 records, groups of 32", "i32",
      "adds the scaled weighted integer product minus its minimum correction "
      "to each output",
      "exact integer subgroup products, followed by source binary32 scaling",
      makeTyped(7, 7, 0, {STK::I32, STK::I8, STK::F32, STK::I64}, {},
                {STK::I32}, EffectMask::ReadWrite, 1, std::nullopt),
      [](Operation *op) {
        auto layout = op->getAttrOfType<DenseI32ArrayAttr>("layout");
        using Q = intentdsl::kir::Q4KRecord;
        const int32_t fields[] = {Q::bytes,     Q::packed,    Q::factors,
                                  Q::d,         Q::dmin,      Q::halfGroups,
                                  Q::scaleBits, Q::nibbleBits};
        if (!layout || layout.asArrayRef() != ArrayRef<int32_t>(fields))
          return failure();
        for (unsigned i = 0; i < 5; ++i) {
          auto ty =
              cast<MemRefType>(op->getOperand(i).getType()).getElementType();
          if (!isSpace(op->getOperand(i), 1) ||
              !(i == 0 ? ty.isF32() : ty.isInteger(i == 1 ? 8 : 32)))
            return failure();
        }
        return success();
      });
  add("segmented-indexed-product", "segment_product", "segment_product",
      "segment.mlu",
      "segment starts, consecutive coefficients and i32 keys, resident f32 "
      "table and workspace",
      "unit stride", "f32",
      "reads three global streams and writes private results",
      "separate binary32 multiply and addition in source member order",
      makeTyped(10, 10, 0, {STK::I32, STK::F32, STK::I64}, {}, {STK::F32},
                EffectMask::ReadWrite, 1, std::nullopt),
      [](Operation *op) {
        for (unsigned i = 0; i < 7; ++i) {
          auto t =
              cast<MemRefType>(op->getOperand(i).getType()).getElementType();
          bool integer = i == 1 || i == 3 || i == 5 || i == 6;
          if (!isSpace(op->getOperand(i), i >= 1 && i <= 3 ? 0 : 1) ||
              !(integer ? t.isInteger(32) : t.isF32()))
            return failure();
        }
        return success();
      });
  add("key-reservation", "reserve_tickets", "reserve_tickets",
      "reservations.mlu",
      "i32 counters, pitched keys, counts, range bases, compact output tickets "
      "and shape",
      "GDRAM counters, NRAM tile", "i32",
      "atomic fetch-add on touched counters",
      "one ordered ticket per live key with relaxed atomic ordering",
      makeTyped(10, 10, 0, {STK::I32, STK::I64, STK::I1}, {}, {STK::I32},
                EffectMask::ReadWrite, 1, std::nullopt),
      [](Operation *op) {
        for (unsigned i = 0; i < 5; ++i)
          if (!isSpace(op->getOperand(i), i ? 1 : 0) ||
              !cast<MemRefType>(op->getOperand(i).getType())
                   .getElementType()
                   .isInteger(32))
            return failure();
        return success();
      });
  add("member-ramp", "member_ramp", "i32_ramp", "selection.mlu",
      "output, first, live", "contiguous NRAM", "i32",
      "writes one row of indices", "integer ordinal modulo 2^32",
      makeTyped(3, 3, 0, {STK::I32, STK::I64}, {}, {STK::I32},
                EffectMask::Write, 64, 1),
      [](Operation *op) {
        return success(isSpace(op->getOperand(0), 1) &&
                       cast<MemRefType>(op->getOperand(0).getType())
                           .getElementType()
                           .isInteger(32));
      });
  add("stable-filter", "stable_filter", "filter_i32", "selection.mlu",
      "destination, payload, predicate, scratch and length", "contiguous NRAM",
      "i32", "writes selected payloads in source order",
      "integer 0/1 mask or binary32 nonzero predicate including NaN and "
      "subnormals",
      makeTyped(5, 5, 1, {STK::F32, STK::I32, STK::I64}, {STK::I32}, {STK::I32},
                EffectMask::ReadWrite, 64, 1),
      [](Operation *op) {
        for (unsigned i = 0; i < 4; ++i) {
          auto t =
              cast<MemRefType>(op->getOperand(i).getType()).getElementType();
          if (!isSpace(op->getOperand(i), 1) ||
              !(t.isInteger(32) || (i == 2 && t.isF32())))
            return failure();
        }
        return success();
      });
  add("ordered-selection", "ordered_select", "select_ordered", "selection.mlu",
      "values, indices, contiguous f32 row, filter scratch, n, k, width",
      "private NRAM", "f32/i32", "writes carried values and indices",
      "ordered greater-than, NaNs skipped, equal values follow the original "
      "insertion chain",
      makeTyped(7, 7, 0, {STK::F32, STK::I32, STK::I64}, {}, {STK::F32},
                EffectMask::ReadWrite, 64, 1),
      [](Operation *op) {
        return success(cast<MemRefType>(op->getOperand(0).getType())
                           .getElementType()
                           .isF32() &&
                       cast<MemRefType>(op->getOperand(1).getType())
                           .getElementType()
                           .isInteger(32) &&
                       cast<MemRefType>(op->getOperand(2).getType())
                           .getElementType()
                           .isF32() &&
                       cast<MemRefType>(op->getOperand(3).getType())
                           .getElementType()
                           .isInteger(32));
      });
  add("inclusive-scan", "scan_sum", "tile_prefix", "scan.mlu",
      "NRAM source, destination and scratch tiles of matching element type",
      "contiguous NRAM row fragment", "i32 or f32",
      "writes NRAM prefix destination and scratch",
      "Hillis-Steele power-of-two inclusive prefix sum in NRAM",
      makeTyped(5, 5, 0, {STK::F32, STK::I32, STK::I64}, {},
                {STK::F32, STK::I32}, EffectMask::ReadWrite, 64, 1),
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
      "NRAM f32 row fragment, NRAM f32 transpose/tree scratch and active count",
      "contiguous NRAM row", "f32",
      "writes NRAM f32 scratch and returns Moment tuple (n, mean, m2)",
      "vectorized 5-level binary tree of KIR adjacent merge_moments summaries "
      "with sequential merge_moments block/tail/non-normal fallback",
      makeTyped(3, 3, 1, {STK::F32, STK::I64}, {}, {STK::F32},
                EffectMask::ReadWrite, 64, 1),
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
      makeTyped(2, 2, 1, {}, {}, {STK::F32}, EffectMask::None, 0, std::nullopt),
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
      makeTyped(5, 5, 0, {STK::F32, STK::I32, STK::I64}, {}, {STK::I32},
                EffectMask::ReadWrite, 64, 1),
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
      makeTyped(2, 2, 1, {STK::I64}, {STK::F32}, {STK::I32}, EffectMask::None,
                0, std::nullopt),
      [](Operation *op) {
        return success(
            op->getOperand(0).getType().getIntOrFloatBitWidth() == 64 &&
            op->getOperand(1).getType().getIntOrFloatBitWidth() == 64 &&
            op->getResult(0).getType().isF32());
      });

  auto hasCnnl = [hasNram](const TargetProfile &p) -> llvm::Error {
    if (auto err = hasNram(p))
      return err;
    if (!p.features.count("cnnl_runtime"))
      return llvm::createStringError(
          "BANG C host runtime disables external CNNL library dispatch");
    return llvm::Error::success();
  };
  auto checkRowFastMath = [](Operation *op, const TargetProfile &target,
                             arith::FastMathFlags req,
                             llvm::StringRef label) -> llvm::Error {
    auto effDefault = target.numerical.defaultFastMath;
    if (auto *fmp = target.services.get<intentdsl::kir::FastMathPolicy>())
      effDefault = effDefault | fmp->defaultFlags;
    if (auto fn = dyn_cast<func::FuncOp>(op)) {
      SmallVector<Operation *> mathOps;
      fn.walk([&](Operation *inner) {
        auto code = inner->getName().stripDialect();
        if (llvm::is_contained(
                ArrayRef<StringRef>{"add", "sub", "mul", "div", "exp", "rsqrt",
                                    "sqrt", "reduce", "combine",
                                    "summary_reduce", "summary_merge"},
                code))
          mathOps.push_back(inner);
      });
      auto actual = intersectFastMath(mathOps, effDefault);
      if (!arith::bitEnumContainsAll(actual, req)) {
        auto missing = req & ~actual;
        return llvm::createStringError(
            label.str() + ": missing required fastmath flags (" +
            arith::stringifyFastMathFlags(missing) + ")");
      }
      if (fn.getNumArguments() > 0) {
        if (auto mr = dyn_cast<MemRefType>(fn.getArgument(0).getType());
            mr && mr.getElementType().isBF16() &&
            !allowsSubnormalFlush(op, target.numerical))
          return llvm::createStringError(
              label.str() +
              ": BF16 CNNL operator requires denormal-fp-math FTZ permission");
      }
      return llvm::Error::success();
    }
    NumericalOptions opts = target.numerical;
    opts.defaultFastMath = effDefault;
    if (!hasFastMath(op, req, opts)) {
      auto actual = effectiveFastMath(op, effDefault);
      auto missing = req & ~actual;
      return llvm::createStringError(
          label.str() + ": missing required fastmath flags (" +
          arith::stringifyFastMathFlags(missing) + ")");
    }
    if (op->getNumOperands() > 0) {
      if (auto mr = dyn_cast<MemRefType>(op->getOperand(0).getType());
          mr && mr.getElementType().isBF16() &&
          !allowsSubnormalFlush(op, target.numerical))
        return llvm::createStringError(
            label.str() +
            ": BF16 CNNL operator requires denormal-fp-math FTZ permission");
    }
    return llvm::Error::success();
  };
  auto addCnnl = [&](StringRef name, StringRef op, StringRef symbol,
                     StringRef estimator, std::vector<std::string> costParams,
                     StringRef operands, StringRef layout, StringRef acc,
                     StringRef effects, StringRef numerical,
                     TypedContract typed,
                     std::function<LogicalResult(Operation *)> verify) {
    ImplementationContract c;
    c.name = name.str();
    c.operation = ("bangc_src." + op).str();
    c.symbol = symbol.str();
    c.source = ImplementationSource::External;
    c.minimum = Reuse::Library;
    c.dependencies = {"cnnl"};
    c.operands = operands.str();
    c.layout = layout.str();
    c.accumulator = acc.str();
    c.effects = effects.str();
    c.numerical = numerical.str();
    c.estimator = estimator.str();
    c.costParameters = std::move(costParams);
    c.available = hasCnnl;
    c.verify = std::move(verify);
    c.typed = std::move(typed);
    unit.implementations.push_back({std::move(c), COMPILERINFRA_SOURCE});
  };

  addCnnl(
      "cnnl-matmul-f16", "cnnl_matmul", "intent_cnnl_matmul",
      "exits/bangc::cnnl-matmul-cost", {"m", "n", "k", "batch"},
      "GDRAM f16[M,K]/[K,M] lhs, f16[K,N]/[N,K] rhs -> f16[M,N] output",
      "contiguous or transposed 2D/3D GDRAM tensors", "f32",
      "reads GDRAM lhs/rhs and writes GDRAM f16 output on CNRT queue",
      "Cambricon CNNL cnnlMatMul_v2 with CNNL_DTYPE_FLOAT compute/onchip "
      "(ALLOW_TF32=0), preserving IEEE FP16 subnormals",
      makeTyped(3, std::nullopt, 0,
                {STK::F16, STK::F32, STK::I32, STK::I64}, {}, {STK::F32},
                EffectMask::ReadWrite, 0, std::nullopt),
      [](Operation *op) {
        auto dt = op->getAttrOfType<IntegerAttr>("in_dtype");
        return success(dt && dt.getInt() == 1);
      });

  {
    TypedContract ftzMm = makeTyped(
        3, std::nullopt, 0, {STK::BF16, STK::F32, STK::I32, STK::I64}, {},
        {STK::F32}, EffectMask::ReadWrite, 0, std::nullopt);
    ftzMm.numerical.requireInputSubnormalFlush = true;
    ftzMm.numerical.requireOutputSubnormalFlush = true;
    addCnnl(
        "cnnl-matmul-ftz", "cnnl_matmul", "intent_cnnl_matmul",
        "exits/bangc::cnnl-matmul-cost", {"m", "n", "k", "batch"},
        "GDRAM bf16/f32[M,K]/[K,M] lhs, bf16/f32[K,N]/[N,K] rhs -> bf16/f32[M,N]",
        "contiguous or transposed 2D/3D GDRAM tensors", "f32",
        "reads GDRAM lhs/rhs and writes GDRAM bf16/f32 output on CNRT queue",
        "Cambricon CNNL cnnlMatMul_v2 or strided BF16 batch with f32 accumulation "
        "(ALLOW_TF32=0); flushes binary32 subnormals (requires denormal-fp-math FTZ)",
        std::move(ftzMm), [](Operation *op) {
          auto dt = op->getAttrOfType<IntegerAttr>("in_dtype");
          return success(dt && (dt.getInt() == 2 || dt.getInt() == 17));
        });
  }

  addCnnl(
      "cnnl-gather-rows", "cnnl_row_op", "intent_cnnl_gather_rows",
      "exits/bangc::cnnl-row-cost", {"m", "n"},
      "table[V,N], in-bounds signed indices[M] -> rows[M,N]",
      "contiguous table, index vector and output with disjoint writes", "none",
      "reads the index vector and addressed rows and writes the output",
      "bitwise row copy via CNNL embedding forward with padding and normalization disabled",
      makeTyped(3, 3, 0, {STK::AnyInteger, STK::AnyFloat}, {}, {},
                EffectMask::ReadWrite, 0, std::nullopt),
      [](Operation *op) {
        return success(op->getAttrOfType<StringAttr>("kind") == "gather_rows");
      });

  addCnnl(
      "cnnl-transpose", "cnnl_row_op", "intent_cnnl_transpose",
      "exits/bangc::cnnl-row-cost", {"m", "n"},
      "GDRAM [M, N] source and GDRAM [N, M] destination (8/16/32-bit elements)",
      "contiguous 2D GDRAM tensors", "none",
      "reads GDRAM [M, N] source and writes transposed GDRAM [N, M] output",
      "bitwise exact 2D transpose via Cambricon CNNL cnnlTranspose_v2",
      makeTyped(2, 2, 0,
                {STK::I8, STK::UI8, STK::I16, STK::F16, STK::BF16, STK::I32,
                 STK::F32},
                {}, {}, EffectMask::ReadWrite, 0, std::nullopt),
      [](Operation *op) {
        auto k = op->getAttrOfType<StringAttr>("kind");
        return success(k && k.getValue() == "transpose");
      });

  {
    TypedContract smTyped =
        makeTyped(2, 2, 0, {STK::F16, STK::BF16, STK::F32}, {}, {STK::F32},
                  EffectMask::ReadWrite, 0, std::nullopt);
    smTyped.precondition =
        [checkRowFastMath](Operation *op,
                           const TargetProfile &target) -> llvm::Error {
      return checkRowFastMath(
          op, target,
          arith::FastMathFlags::afn | arith::FastMathFlags::arcp |
              arith::FastMathFlags::reassoc,
          "exits/bangc::cnnl-softmax");
    };
    addCnnl(
        "cnnl-softmax", "cnnl_row_op", "intent_cnnl_softmax",
        "exits/bangc::cnnl-row-cost", {"m", "n"},
        "GDRAM [M, N] input and GDRAM [M, N] output (f16/bf16/f32)",
        "contiguous 2D GDRAM tensors reduced along minor axis", "f32",
        "reads GDRAM [M, N] input and writes normalized GDRAM [M, N] output",
        "Cambricon CNNL cnnlSoftmaxForward_v2 (ACCURATE, HIGH_PRECISION); "
        "requires fastmath afn,arcp,reassoc",
        std::move(smTyped), [](Operation *op) {
          auto k = op->getAttrOfType<StringAttr>("kind");
          return success(k && k.getValue() == "softmax");
        });
  }

  {
    TypedContract lnTyped =
        makeTyped(5, 5, 0, {STK::F16, STK::BF16, STK::F32}, {}, {STK::F32},
                  EffectMask::ReadWrite, 0, std::nullopt);
    lnTyped.precondition =
        [checkRowFastMath](Operation *op,
                           const TargetProfile &target) -> llvm::Error {
      return checkRowFastMath(
          op, target,
          arith::FastMathFlags::afn | arith::FastMathFlags::reassoc,
          "exits/bangc::cnnl-layer-norm");
    };
    addCnnl(
        "cnnl-layer-norm", "cnnl_row_op", "intent_cnnl_layer_norm",
        "exits/bangc::cnnl-row-cost", {"m", "n"},
        "GDRAM [M, N] input, [N] gain, [N] bias, [M, N] output and scalar f32 eps",
        "contiguous 2D/1D GDRAM tensors normalized along minor axis", "f32",
        "reads GDRAM input/gain/bias and writes normalized GDRAM [M, N] output",
        "Cambricon CNNL cnnlLayerNormForward with FP32 mean/rstd; "
        "requires fastmath afn,reassoc",
        std::move(lnTyped), [](Operation *op) {
          auto k = op->getAttrOfType<StringAttr>("kind");
          return success(k && k.getValue() == "layer_norm");
        });
  }
}
} // namespace CI_UNIT_NAMESPACE
