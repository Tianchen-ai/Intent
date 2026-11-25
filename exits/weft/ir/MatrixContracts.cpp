#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cstdlib>

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;

llvm::json::Object probeTritonWeftBrgemm() {
  llvm::json::Object out;
  out["source"] = "triton-weft-builtin";
  if (const char *env = std::getenv("ONEDNN_ROOT"); env && *env)
    out["onednn_root"] = env;
  else if (const char *dnnl = std::getenv("DNNL_ROOT"); dnnl && *dnnl)
    out["onednn_root"] = dnnl;
  out["note"] =
      "live capability and version discovery runs via triton.backends.weft.target.native_capabilities()";
  return out;
}

void registerWeftMatrixContracts(Unit &u) {
  u.dependencyProbes["triton-weft-brgemm"] = probeTritonWeftBrgemm;

  auto add = [&](ImplementationContract c) {
    u.implementations.push_back({std::move(c), COMPILERINFRA_SOURCE});
  };

  auto checkMatrixProgram = [](Operation *op, StringRef expectedId,
                               bool expectedBrgemm) -> bool {
    for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
      if (p->getName().getStringRef() == "weft_host.matrix_program") {
        auto impl = p->getAttrOfType<StringAttr>("implementation");
        auto brg = p->getAttrOfType<BoolAttr>("uses_brgemm");
        return impl && impl.getValue() == expectedId && brg &&
               brg.getValue() == expectedBrgemm;
      }
    }
    return false;
  };

  {
    ImplementationContract c;
    c.name = "weft-simd-fma";
    c.operation = "weft_host.task_contract";
    c.symbol = "weft_kernel.outer_contract";
    c.source = ImplementationSource::Builtin;
    c.minimum = Reuse::Native;
    c.operands = "f32|f16|bf16|i8 widened to f32|i32 Weft SIMD/RVV lanes";
    c.layout = "row-major or arbitrary strided LHS/RHS, optional [K, nr] panels";
    c.accumulator = "f32 or si32 Weft vector/tile registers";
    c.effects = "pure local value contraction inside task.weftir";
    c.numerical =
        "IEEE-754 f32 FMA preserving subnormals, Inf, and qNaN; exact si32 MAC";
    c.estimator = u.qualify("native-panels");
    c.costParameters = {"vector_bits", "vector_registers", "workers"};
    c.available = [](const TargetProfile &p) -> llvm::Error {
      auto *rm = p.services.get<intentdsl::cpu::RowMachine>();
      if (!rm || rm->vectorBits < 128)
        return llvm::createStringError("Weft SIMD contraction requires >=128-bit vector width");
      return llvm::Error::success();
    };
    c.verify = [checkMatrixProgram](Operation *op) -> LogicalResult {
      if ((op->getNumOperands() != 2 && op->getNumOperands() != 3) ||
          op->getNumResults() != 1)
        return failure();
      auto eng = op->getAttrOfType<StringAttr>("engine");
      auto ftz = op->getAttrOfType<BoolAttr>("allow_subnormal_ftz");
      auto kind = op->getAttrOfType<StringAttr>("kind");
      auto acc = op->getAttrOfType<StringAttr>("acc_type");
      if (!eng || (eng.getValue() != "auto" && eng.getValue() != "vector") ||
          !ftz || ftz.getValue() || op->hasAttr("kir.storage") || !kind || !acc)
        return failure();
      if ((op->getNumOperands() == 2 && kind.getValue() != "outer_contract") ||
          (op->getNumOperands() == 3 && kind.getValue() != "fma"))
        return failure();
      if (acc.getValue() != "f32" && acc.getValue() != "si32")
        return failure();
      if (checkMatrixProgram(op, "exits/weft::weft-simd-fma", false))
        return success();
      for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
        if (p->getName().getStringRef() == "weft_host.owner_program")
          return success(op->getNumOperands() == 2 &&
                         kind.getValue() == "outer_contract" &&
                         acc.getValue() == "f32");
      return failure();
    };
    TypedContract tc;
    tc.shape.minOperands = 2;
    tc.shape.maxOperands = 3;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    tc.operandTypes = {ScalarTypeKind::Index};
    tc.resultTypes = {ScalarTypeKind::Index};
    tc.accumulatorTypes = {ScalarTypeKind::F32, ScalarTypeKind::I32};
    tc.effects = EffectMask::None;
    c.typed = std::move(tc);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }

  {
    ImplementationContract c;
    c.name = "weft-f32-panel-fma";
    c.operation = "weft_host.task_contract";
    c.symbol = "weft_kernel.outer_contract";
    c.source = ImplementationSource::Builtin;
    c.minimum = Reuse::Native;
    c.operands = "f32 panels (including host-widened f16|bf16) -> f32 Weft SIMD/RVV lanes";
    c.layout = "row-major [M, K] x packed [K, nr] f32 panels";
    c.accumulator = "f32 Weft vector/tile registers";
    c.effects = "pure local value contraction inside task.weftir";
    c.numerical = "IEEE-754 f32 FMA preserving subnormals, Inf, and qNaN";
    c.estimator = u.qualify("native-panels");
    c.costParameters = {"vector_bits", "vector_registers", "workers"};
    c.available = [](const TargetProfile &p) -> llvm::Error {
      auto *rm = p.services.get<intentdsl::cpu::RowMachine>();
      if (!rm || rm->vectorBits < 128)
        return llvm::createStringError("Weft f32 panel FMA requires >=128-bit vector width");
      return llvm::Error::success();
    };
    c.verify = [checkMatrixProgram](Operation *op) -> LogicalResult {
      if ((op->getNumOperands() != 2 && op->getNumOperands() != 3) ||
          op->getNumResults() != 1)
        return failure();
      auto eng = op->getAttrOfType<StringAttr>("engine");
      auto ftz = op->getAttrOfType<BoolAttr>("allow_subnormal_ftz");
      auto kind = op->getAttrOfType<StringAttr>("kind");
      auto acc = op->getAttrOfType<StringAttr>("acc_type");
      auto inTys = op->getAttrOfType<ArrayAttr>("in_types");
      if (!eng || eng.getValue() != "auto" || !ftz || ftz.getValue() ||
          op->hasAttr("kir.storage") || !kind || !acc ||
          acc.getValue() != "f32" || !inTys ||
          inTys.size() != op->getNumOperands())
        return failure();
      if ((op->getNumOperands() == 2 && kind.getValue() != "outer_contract") ||
          (op->getNumOperands() == 3 && kind.getValue() != "fma"))
        return failure();
      for (Attribute a : inTys) {
        auto s = dyn_cast<StringAttr>(a);
        if (!s || (s.getValue() != "f32" &&
                   !s.getValue().starts_with("!weft_kernel.value<f32,[")))
          return failure();
      }
      return success(
          checkMatrixProgram(op, "exits/weft::weft-f32-panel-fma", false));
    };
    TypedContract tc;
    tc.shape.minOperands = 2;
    tc.shape.maxOperands = 3;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    tc.operandTypes = {ScalarTypeKind::Index};
    tc.resultTypes = {ScalarTypeKind::Index};
    tc.accumulatorTypes = {ScalarTypeKind::F32};
    tc.effects = EffectMask::None;
    c.typed = std::move(tc);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }

  {
    ImplementationContract c;
    c.name = "weft-q4k-dot";
    c.operation = "weft_host.task_contract";
    c.symbol = "weft_kernel.contract";
    c.source = ImplementationSource::Builtin;
    c.minimum = Reuse::Native;
    c.operands = "Q4_K[144B] x Q8_K[292B] -> f32";
    c.layout = "Q4_K weight super-blocks x shared Q8_K activation super-blocks";
    c.accumulator = "si32 8x32 subgroup dots scaled to f32 per 256-element super-block";
    c.effects = "pure local value contraction on admitted Q4_K/Q8_K subgroups";
    c.numerical =
        "exact si32 32-element subgroup dot products with IEEE f16->f32 super-block scales";
    c.estimator = u.qualify("q4k-panels");
    c.costParameters = {"vector_bits", "workers"};
    c.available = [](const TargetProfile &p) -> llvm::Error {
      auto *rm = p.services.get<intentdsl::cpu::RowMachine>();
      if (!rm || rm->vectorBits < 128)
        return llvm::createStringError("Weft Q4_K contraction requires >=128-bit vector width");
      return llvm::Error::success();
    };
    c.verify = [checkMatrixProgram](Operation *op) -> LogicalResult {
      auto eng = op->getAttrOfType<StringAttr>("engine");
      auto ftz = op->getAttrOfType<BoolAttr>("allow_subnormal_ftz");
      auto kind = op->getAttrOfType<StringAttr>("kind");
      auto acc = op->getAttrOfType<StringAttr>("acc_type");
      if (op->getNumOperands() != 2 || op->getNumResults() != 1 || !eng ||
          eng.getValue() != "auto" || !ftz || ftz.getValue() || !kind ||
          kind.getValue() != "contract" || !acc || acc.getValue() != "si32")
        return failure();
      if (op->getAttr("kir.storage") !=
          TypeAttr::get(intentdsl::kir::Q4KType::get(op->getContext()))) {
        auto fmt = op->getAttrOfType<StringAttr>("format");
        const auto *spec =
            fmt ? intentdsl::kir::lookupQuantFormat(fmt.getValue()) : nullptr;
        if (!spec ||
            spec->roleLayout != intentdsl::kir::QuantRoleLayout::RecordGemv)
          return failure();
      }
      return success(checkMatrixProgram(op, "exits/weft::weft-q4k-dot", false));
    };
    TypedContract tc;
    tc.shape.minOperands = 2;
    tc.shape.maxOperands = 2;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    tc.operandTypes = {ScalarTypeKind::Index};
    tc.resultTypes = {ScalarTypeKind::Index};
    tc.accumulatorTypes = {ScalarTypeKind::I32};
    tc.effects = EffectMask::None;
    c.typed = std::move(tc);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }

  // Downstream builtins provided by triton-weft >= 0.2.0 (tier: vendor)
  auto makeBrgemm = [&](StringRef name, StringRef featKey, StringRef lhsElem,
                        StringRef rhsElem, StringRef accElem, int64_t vnniPack,
                        StringRef operands, StringRef layout, StringRef acc,
                        StringRef numerical, bool requiresFtz = false,
                        bool isQ4k = false) {
    ImplementationContract c;
    c.name = name.str();
    c.operation = "weft_host.task_contract";
    c.symbol = isQ4k ? "weft_kernel.contract" : "weft_kernel.outer_contract";
    c.source = ImplementationSource::Builtin;
    c.minimum = Reuse::Vendor;
    c.operands = operands.str();
    c.layout = layout.str();
    c.accumulator = acc.str();
    c.effects =
        "invokes triton-weft x86 oneDNN SEQ BRGeMM ukernel inside compiled task.weftir";
    c.numerical = numerical.str();
    c.estimator = u.qualify("brgemm-panels");
    c.costParameters = {"vector_bits", "workers"};
    std::string reqFeat = featKey.str();
    std::string contractName = name.str();
    std::string expectedLhs = lhsElem.str();
    std::string expectedRhs = rhsElem.str();
    std::string expectedAcc = accElem.str();
    std::string qualifiedId = ("exits/weft::" + name).str();
    c.available = [reqFeat, contractName](const TargetProfile &p) -> llvm::Error {
      if (!p.features.count("triton_weft_v0_2"))
        return llvm::createStringError(
            contractName + " requires triton-weft weft_core >= 0.2.0");
      if (!p.features.count(reqFeat))
        return llvm::createStringError(
            contractName + " is unavailable on probed triton-weft oneDNN SEQ runtime");
      return llvm::Error::success();
    };
    c.verify = [checkMatrixProgram, contractName, qualifiedId, expectedLhs,
                expectedRhs, expectedAcc, vnniPack, requiresFtz,
                isQ4k](Operation *op) -> LogicalResult {
      if (op->getNumOperands() != 2 || op->getNumResults() != 1)
        return failure();
      auto eng = op->getAttrOfType<StringAttr>("engine");
      if (!eng || eng.getValue() != contractName)
        return failure();
      auto kind = op->getAttrOfType<StringAttr>("kind");
      if (!kind || kind.getValue() != (isQ4k ? "contract" : "outer_contract"))
        return failure();
      auto over = op->getAttrOfType<DenseI64ArrayAttr>("over");
      const int64_t singleAxis[] = {3};
      const int64_t packAxes[] = {3, 4};
      ArrayRef<int64_t> expectedOver =
          vnniPack > 1 ? ArrayRef<int64_t>(packAxes)
                       : ArrayRef<int64_t>(singleAxis);
      if (!over || over.asArrayRef() != expectedOver)
        return failure();
      auto accAttr = op->getAttrOfType<StringAttr>("acc_type");
      if (!accAttr || accAttr.getValue() != expectedAcc)
        return failure();
      auto inTys = op->getAttrOfType<ArrayAttr>("in_types");
      auto valTy = op->getAttrOfType<StringAttr>("val_type");
      if (!inTys || inTys.size() != 2 || !valTy)
        return failure();
      auto lhsTy = dyn_cast<StringAttr>(inTys[0]);
      auto rhsTy = dyn_cast<StringAttr>(inTys[1]);
      if (!lhsTy || !rhsTy ||
          !lhsTy.getValue().starts_with("!weft_kernel.value<" + expectedLhs +
                                        ",[") ||
          !rhsTy.getValue().starts_with("!weft_kernel.value<" + expectedRhs +
                                        ",[") ||
          !valTy.getValue().starts_with("!weft_kernel.value<" + expectedAcc +
                                        ",["))
        return failure();
      if (vnniPack > 1) {
        std::string packSuffix = "," + std::to_string(vnniPack) + "],[";
        if (!lhsTy.getValue().contains(packSuffix) ||
            !rhsTy.getValue().contains(packSuffix))
          return failure();
      }
      if (isQ4k &&
          (lhsTy.getValue() != "!weft_kernel.value<ui8,[-1,32],[1,3]>" ||
           rhsTy.getValue() != "!weft_kernel.value<si8,[32],[3]>"))
        return failure();
      auto ftz = op->getAttrOfType<BoolAttr>("allow_subnormal_ftz");
      if (!ftz || ftz.getValue() != requiresFtz)
        return failure();
      auto fn = op->getParentOfType<func::FuncOp>();
      if (!fn || (requiresFtz && !intentdsl::kir::allowsSubnormalFtz(fn)))
        return failure();
      auto q4kAttr =
          TypeAttr::get(intentdsl::kir::Q4KType::get(op->getContext()));
      if (isQ4k ? (op->getAttr("kir.storage") != q4kAttr)
                : op->hasAttr("kir.storage"))
        return failure();
      if (!checkMatrixProgram(op, qualifiedId, true))
        return failure();
      return success();
    };
    TypedContract tc;
    tc.shape.minOperands = 2;
    tc.shape.maxOperands = 2;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    tc.operandTypes = {ScalarTypeKind::Index};
    tc.resultTypes = {ScalarTypeKind::Index};
    tc.accumulatorTypes = {expectedAcc == "f32" ? ScalarTypeKind::F32
                                                : ScalarTypeKind::I32};
    tc.effects = EffectMask::None;
    tc.numerical.requireInputSubnormalFlush = requiresFtz;
    tc.numerical.requireOutputSubnormalFlush = requiresFtz;
    c.typed = std::move(tc);
    populateTypedContractSummaries(c);
    add(std::move(c));
  };

  makeBrgemm(
      "onednn-brgemm-f32", "weft_brgemm_f32", "f32", "f32", "f32", 1,
      "f32 x f32 -> f32",
      "LHS row-major (unit K stride), RHS row-major or [K, nr] panels",
      "f32 ZMM BRGeMM accumulator",
      "IEEE-754 f32 FMA preserving subnormals, signed zero, Inf, and qNaN");

  makeBrgemm(
      "onednn-brgemm-f16", "weft_brgemm_f16", "f16", "f16", "f32", 1,
      "f16 x f16 -> f32",
      "LHS row-major (unit K stride), RHS row-major or [K, nr] panels",
      "f32 AVX-512 FP16/FMA BRGeMM accumulator",
      "IEEE-754 f16 widened to f32 FMA preserving f16 subnormals, Inf, and qNaN");

  makeBrgemm(
      "onednn-brgemm-bf16", "weft_brgemm_bf16", "bf16", "bf16", "f32", 2,
      "bf16 x bf16 -> f32",
      "LHS 3D view [R, K/2, 2] (unit K stride, K%2==0), RHS zero-copy VNNI pack32 [K/2, nr, 2]",
      "f32 AMX-BF16 dot2 tile accumulator",
      "AMX-BF16 dot2 to f32; flushes BF16 subnormals to zero (requires allow_subnormal_ftz)",
      /*requiresFtz=*/true);

  makeBrgemm(
      "onednn-brgemm-s8", "weft_brgemm_s8", "si8", "si8", "si32", 4,
      "s8 x s8 -> s32",
      "LHS 3D view [R, K/4, 4] (unit K stride, K%4==0), RHS zero-copy VNNI pack32 [K/4, nr, 4]",
      "s32 AMX-INT8/VNNI tile accumulator",
      "exact two's-complement s8*s8->s32 accumulation");

  makeBrgemm(
      "onednn-brgemm-u8s8", "weft_brgemm_u8s8", "ui8", "si8", "si32", 1,
      "u8[R, 32] x s8[32] -> s32 (Q4_K subgroup)",
      "multi-row unpacked u8[R, 32] nibbles x Q8_K s8[32] subgroup",
      "s32 AMX-INT8/VNNI subgroup accumulator",
      "exact u8*s8->s32 32-element subgroup dot preserving Q4_K super-block R32 contract",
      /*requiresFtz=*/false, /*isQ4k=*/true);
}

void populateWeftMatrixContracts(intentdsl::cpu::MatrixMachine &m, int64_t v,
                                 bool isRvv) {
  auto add = [&](int64_t r, int64_t c, int64_t d, int64_t lanes,
                 llvm::StringRef id, llvm::StringRef est, llvm::StringRef dtype,
                 int64_t kAlign = 1, bool contigK = false, bool ftz = false) {
    intentdsl::cpu::MatrixImplementation impl;
    impl.rows = r;
    impl.cols = c;
    impl.depth = d;
    impl.lanes = lanes;
    impl.id = ("exits/weft::" + id).str();
    impl.estimator = ("exits/weft::" + est).str();
    impl.dtype = dtype.str();
    impl.kAlign = kAlign;
    impl.requiresContiguousLhsK = contigK;
    impl.flushesSubnormals = ftz;
    m.contracts.push_back(std::move(impl));
  };

  // Native Q4_K/Q8_K fused record contraction precedes per-subgroup u8s8 fallback
  if (isRvv) {
    for (int64_t r : {64, 256, 512, 16, 8})
      add(r, 1, 256, v, "weft-q4k-dot", "q4k-panels", "q4k");
  } else {
    for (int64_t r : {16, 8, 64})
      add(r, 1, 256, v, "weft-q4k-dot", "q4k-panels", "q4k");
  }

  if (!isRvv) {
    // Vendor triton-weft oneDNN BRGeMM builtin candidates on x86-64
    for (auto [r, c, d] :
         std::initializer_list<std::tuple<int64_t, int64_t, int64_t>>{
             {6, 4 * v, 64}, {12, 4 * v, 64}, {8, 3 * v, 64}, {12, 3 * v, 64},
             {12, 2 * v, 64}, {14, 2 * v, 64}, {8, 2 * v, 64}, {4, 2 * v, 64}}) {
      add(r, c, d, v, "onednn-brgemm-f32", "brgemm-panels", "f32", 1, true,
          false);
      add(r, c, d, v, "onednn-brgemm-f16", "brgemm-panels", "f16", 1, true,
          false);
    }
    for (auto [r, c, d] :
         std::initializer_list<std::tuple<int64_t, int64_t, int64_t>>{
             {16, 4 * v, 64},
             {32, 4 * v, 64},
             {16, 2 * v, 64},
             {32, 2 * v, 64}}) {
      add(r, c, d, v, "onednn-brgemm-bf16", "brgemm-panels", "bf16", 2, true,
          true);
      add(r, c, d, v, "onednn-brgemm-s8", "brgemm-panels", "s8", 4, true,
          false);
    }
    for (int64_t r : {8, 16, 32})
      add(r, 1, 256, v, "onednn-brgemm-u8s8", "brgemm-panels", "q4k", 32, false,
          false);
  }

  // Native SIMD / RVV candidates (power-of-two RVV vector widths fit 32 RVV regs)
  auto nativeShapes =
      isRvv ? std::vector<std::tuple<int64_t, int64_t, int64_t>>{
                  {4, 4 * v, 64}, {4, 2 * v, 64}, {8, 2 * v, 64}, {2, v, 32}}
            : std::vector<std::tuple<int64_t, int64_t, int64_t>>{
                  {4, 2 * v, 64}, {8, 4 * v, 128}, {8, 3 * v, 64}, {2, v, 32}};
  for (auto [r, c, d] : nativeShapes) {
    add(r, c, d, v, "weft-simd-fma", "native-panels", "any");
    add(r, c, d, v, "weft-f32-panel-fma", "native-panels", "f32");
  }
}
} // namespace CI_UNIT_NAMESPACE
