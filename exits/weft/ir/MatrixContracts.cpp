#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include <cstdlib>

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;

llvm::json::Object probeTritonWeftBrgemm() {
  llvm::json::Object out;
  out["source"] = "triton-weft-builtin";
  out["onednn_root"] = "/opt/rc-dependencies/onednn-3.13.4";
  out["note"] =
      "live capability and version discovery runs via triton.backends.weft.target.native_capabilities()";
  return out;
}

void registerWeftMatrixContracts(Unit &u) {
  u.dependencyProbes["triton-weft-brgemm"] = probeTritonWeftBrgemm;

  auto add = [&](ImplementationContract c) {
    u.implementations.push_back({std::move(c), COMPILERINFRA_SOURCE});
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
    c.costParameters = {"vector_bits", "workers"};
    c.available = [](const TargetProfile &p) -> llvm::Error {
      auto *rm = p.services.get<intentdsl::cpu::RowMachine>();
      if (!rm || rm->vectorBits < 128)
        return llvm::createStringError("Weft SIMD contraction requires >=128-bit vector width");
      return llvm::Error::success();
    };
    c.verify = [](Operation *op) -> LogicalResult {
      return success((op->getNumOperands() == 2 || op->getNumOperands() == 3) &&
                     op->getNumResults() == 1);
    };
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
    c.costParameters = {"vector_bits", "workers"};
    c.available = [](const TargetProfile &p) -> llvm::Error {
      auto *rm = p.services.get<intentdsl::cpu::RowMachine>();
      if (!rm || rm->vectorBits < 128)
        return llvm::createStringError("Weft f32 panel FMA requires >=128-bit vector width");
      return llvm::Error::success();
    };
    c.verify = [](Operation *op) -> LogicalResult {
      return success((op->getNumOperands() == 2 || op->getNumOperands() == 3) &&
                     op->getNumResults() == 1);
    };
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
    c.verify = [](Operation *op) -> LogicalResult {
      return success(
          op->getNumOperands() == 2 && op->getNumResults() == 1 &&
          op->getAttr("kir.storage") ==
              TypeAttr::get(intentdsl::kir::Q4KType::get(op->getContext())));
    };
    add(std::move(c));
  }

  // Downstream builtins provided by triton-weft >= 0.2.0 (tier: vendor)
  auto makeBrgemm = [&](StringRef name, StringRef featKey, StringRef operands,
                        StringRef layout, StringRef acc, StringRef numerical,
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
    c.available = [reqFeat, contractName](const TargetProfile &p) -> llvm::Error {
      if (!p.features.count("triton_weft_v0_2"))
        return llvm::createStringError(
            contractName + " requires triton-weft weft_core >= 0.2.0");
      if (!p.features.count(reqFeat))
        return llvm::createStringError(
            contractName + " is unavailable on probed triton-weft oneDNN SEQ runtime");
      return llvm::Error::success();
    };
    c.verify = [isQ4k](Operation *op) -> LogicalResult {
      if (op->getNumOperands() != 2 || op->getNumResults() != 1)
        return failure();
      auto eng = op->getAttrOfType<StringAttr>("engine");
      if (!eng || !eng.getValue().starts_with("onednn-brgemm"))
        return failure();
      if (isQ4k &&
          op->getAttr("kir.storage") !=
              TypeAttr::get(intentdsl::kir::Q4KType::get(op->getContext())))
        return failure();
      return success();
    };
    add(std::move(c));
  };

  makeBrgemm(
      "onednn-brgemm-f32", "weft_brgemm_f32", "f32 x f32 -> f32",
      "LHS row-major (unit K stride), RHS row-major or [K, nr] panels",
      "f32 ZMM BRGeMM accumulator",
      "IEEE-754 f32 FMA preserving subnormals, signed zero, Inf, and qNaN");

  makeBrgemm(
      "onednn-brgemm-f16", "weft_brgemm_f16", "f16 x f16 -> f32",
      "LHS row-major (unit K stride), RHS row-major or [K, nr] panels",
      "f32 AVX-512 FP16/FMA BRGeMM accumulator",
      "IEEE-754 f16 widened to f32 FMA preserving f16 subnormals, Inf, and qNaN");

  makeBrgemm(
      "onednn-brgemm-bf16", "weft_brgemm_bf16", "bf16 x bf16 -> f32",
      "LHS 3D view [R, K/2, 2] (unit K stride, K%2==0), RHS zero-copy VNNI pack32 [K/2, nr, 2]",
      "f32 AMX-BF16 dot2 tile accumulator",
      "AMX-BF16 dot2 to f32; flushes BF16 subnormals to zero (requires allow_subnormal_ftz)");

  makeBrgemm(
      "onednn-brgemm-s8", "weft_brgemm_s8", "s8 x s8 -> s32",
      "LHS 3D view [R, K/4, 4] (unit K stride, K%4==0), RHS zero-copy VNNI pack32 [K/4, nr, 4]",
      "s32 AMX-INT8/VNNI tile accumulator",
      "exact two's-complement s8*s8->s32 accumulation");

  makeBrgemm(
      "onednn-brgemm-u8s8", "weft_brgemm_u8s8",
      "u8[R, 32] x s8[32] -> s32 (Q4_K subgroup)",
      "multi-row unpacked u8[R, 32] nibbles x Q8_K s8[32] subgroup",
      "s32 AMX-INT8/VNNI subgroup accumulator",
      "exact u8*s8->s32 32-element subgroup dot preserving Q4_K super-block R32 contract",
      /*isQ4k=*/true);
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
  for (int64_t r : {16, 8, 64})
    add(r, 1, 256, v, "weft-q4k-dot", "q4k-panels", "q4k");

  if (!isRvv) {
    // Vendor triton-weft oneDNN BRGeMM builtin candidates on x86-64
    for (auto [r, c, d] :
         std::initializer_list<std::tuple<int64_t, int64_t, int64_t>>{
             {6, 4 * v, 64}, {8, 3 * v, 64}, {4, 2 * v, 64}, {16, 4 * v, 64}}) {
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
