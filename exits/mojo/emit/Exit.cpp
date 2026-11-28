#include "intentdsl/levels/algorithm/kir/ir/DotSemantics.h"
#include "intentdsl/levels/execution/cpu/ir/WeightExp.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include <algorithm>
using namespace compilerinfra;
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
void dialect(DialectRegistry &);
TerminalPrinter mojoMatrixPrinter();
void populateMojoMatrixContracts(intentdsl::cpu::MatrixMachine &, int64_t);
void publishABI(OpPassManager &);
LogicalResult verifyLoops(ModuleOp, const TargetProfile &, bool callable);
void realize(OpPassManager &, int64_t, const intentdsl::CallFacts &);
namespace {
std::string probeHostMojoCalibration() {
#if defined(__x86_64__) || defined(_M_X64)
  return "x86";
#else
  return "";
#endif
}

llvm::Expected<TargetProfile> capabilities(const llvm::json::Object &options) {
  TargetProfile p;
  if (options.empty()) {
    p.services.calibrationProfile = probeHostMojoCalibration();
    return p;
  }
  auto bits = options.getInteger("vector_bits"),
       workers = options.getInteger("workers");
  if (!bits || !workers || *workers <= 0 ||
      (*bits != 128 && *bits != 256 && *bits != 512 && *bits != 1024))
    return llvm::createStringError(
        "invalid execution group or vector capabilities");
  if (auto err = intentdsl::bindPerformance(
          p, options, "",
          options.getString("calibration").value_or("").str(), *workers))
    return std::move(err);
  if (*bits >= 256)
    p.features.insert("avx2");
  if (*bits >= 512)
    p.features.insert("avx512f");
  if (options.getBoolean("grouped_i8").value_or(false))
    p.features.insert("avx512_vnni");
  for (StringRef feat :
       {"avx512_bf16", "avx512_fp16", "amx_int8", "amx_bf16"}) {
    if (options.getBoolean(feat).value_or(false))
      p.features.insert(feat.str());
  }
  if (options.getBoolean("onednn_runtime").value_or(true))
    p.features.insert("onednn_runtime");
  auto caps = std::make_shared<intentdsl::cpu::RowMachine>();
  caps->maskedContiguousLoads = true;
  caps->vectorBits = *bits;
  // The x86-64 AVX-512 encoding addresses 32 vector registers, AVX2 16.
  caps->ownerRegisters = *bits == 512 ? 32 : *bits == 256 ? 16 : 0;
  caps->l1CacheBytes = options.getInteger("l1_cache_bytes").value_or(0);
  caps->l2CacheBytes = options.getInteger("l2_cache_bytes").value_or(0);
  caps->packedOwners = std::clamp<int64_t>(*bits / 32, 4, 16);
  // The wide replay loop groups 16-lane chunks in pairs and stages
  // values larger than a single 32-element register block.
  caps->orderedValueMinColumns = 64;
  caps->orderedValueColumnMultiple = 32;
  caps->workers = *workers;
  int64_t v = *bits / 32;
  auto matrix = std::make_shared<intentdsl::cpu::MatrixMachine>();
  if (*bits == 512 && options.getBoolean("grouped_i8").value_or(false))
    matrix->integerPacking = "signed-dot4-panels";
  matrix->choices = {{2, v, 32, v},
                     {4, 2 * v, 64, v},
                     {8, 2 * v, 64, v},
                     {8, 4 * v, 64, v},
                     {8, 3 * v, 128, v},
                     {12, 2 * v, 128, v}};
  matrix->estimate = "exits/mojo::register-block";
  matrix->records = {{8, 1, 256, v}, {16, 1, 256, v}, {64, 1, 256, v}};
  populateMojoMatrixContracts(*matrix, v);
  if (auto e = p.services.provide<intentdsl::cpu::MatrixMachine>(matrix))
    return e;
  caps->rows = {{1, 64},
                {2, *bits == 512 ? 128 : 64},
                {4, *bits == 512 ? 128 : 64},
                {8, 256}};
  caps->unrollFactors = {1, 4};
  caps->estimate = "exits/mojo::simd-row-primitive";
  if (auto e = p.services.provide<intentdsl::cpu::RowMachine>(caps))
    return e;
  return p;
}
} // namespace
void registerBlockCost(Unit &);
void registerOneDNN(Unit &);
void registerMatrixContracts(Unit &);
void registerMojoAddressOptimization(Unit &);
void bind(Unit &u) {
  intentdsl::cpu::registerWeightExp(u);
  intentdsl::kir::registerDotSemantics(
      u, "mojo", "bf16", "oneDNN BRGEMM; ATen CPU FlashAttentionKernel.cpp",
      "native",
      "oneDNN Data Types and Floating-point Math Mode; Intel SDM TDPBF16PS and "
      "VDPBF16PS: DAZ inputs, FTZ FP32 results independent of MXCSR. RVV BF16 "
      "extension vfwmaccbf16.vv: FP32 widening FMA, IEEE subnormals.",
      "amx_bf16");
  intentdsl::kir::registerDotSemantics(
      u, "mojo", "f16", "oneDNN data types; RISC-V V widening FMA", "source",
      "FP16 widened to FP32 for accumulation; RISC-V V §13 vector "
      "floating-point instructions and vfwmacc; Intel SDM VCVTPH2PS.");
  intentdsl::kir::registerDotSemantics(
      u, "mojo", "fp8", "target matrix primitive", "undocumented",
      "FP8 matrix unit on this CPU target: no documented implementation.");
  registerBlockCost(u);
  registerOneDNN(u);
  registerMatrixContracts(u);
  registerMojoAddressOptimization(u);
  auto addParity = [&](ParityDeclaration d) {
    u.parity.push_back({std::move(d), COMPILERINFRA_SOURCE});
  };
  addParity({"levels/execution/cpu::simd-matrix-contraction",
             "x86",
             ParityStatus::Implemented,
             {"exits/mojo::simd-fma", "levels/execution/cpu::contract-blocks",
              "levels/execution/cpu::matrix-partition"},
             "Register-blocked Mojo SIMD FMA and host-packed panel contraction"});
  addParity({"levels/execution/cpu::vendor-brgemm-contraction",
             "x86",
             ParityStatus::Implemented,
             {"exits/mojo::onednn-brgemm-f32", "exits/mojo::onednn-brgemm-f16",
              "exits/mojo::onednn-brgemm-bf16", "exits/mojo::onednn-brgemm-s8",
              "exits/mojo::onednn-brgemm-u8s8",
              "levels/execution/cpu::contract-blocks"},
             "oneDNN 3.13+ BRGeMM / AMX / VNNI ukernel contraction"});
  addParity({"levels/execution/cpu::vnni-int8-contraction",
             "x86",
             ParityStatus::Implemented,
             {"exits/mojo::vnni-dot4", "exits/mojo::onednn-brgemm-s8",
              "levels/execution/cpu::contract-blocks"},
             "AVX-512 VNNI vpdpbusd signed-dot4 and AMX-INT8 BRGeMM contraction"});
  addParity({"levels/execution/cpu::q4k-packed-contraction",
             "x86",
             ParityStatus::Implemented,
             {"exits/mojo::q4k-vnni", "exits/mojo::q4k-simd",
              "exits/mojo::onednn-brgemm-u8s8",
              "levels/execution/cpu::contract-blocks"},
             "GGUF Q4_K x Q8_K super-block VNNI, SIMD, and u8s8 BRGeMM contraction"});
  addParity({"levels/execution/cpu::ordered-owner-pairing",
             "x86",
             ParityStatus::Implemented,
             {"exits/mojo::simd-reduce-16x16",
              "levels/execution/cpu::shared-intervals",
              "levels/execution/cpu::ordered-owners"},
             "16-owner BRGeMM/SIMD QK/PV pairing, proven score replay, and SIMD inner-loop reduction"});
  addParity({"levels/execution/cpu::row-reduction-and-scan",
             "x86",
             ParityStatus::Implemented,
             {"exits/mojo::simd-scan-sum", "exits/mojo::welford-step",
              "exits/mojo::welford-reduce", "exits/mojo::welford-combine",
              "levels/execution/cpu::row-reduction",
              "levels/execution/cpu::row-scan"},
             "SIMD row reductions, Hillis-Steele scan, Welford moments, and L1d row caching"});
  addParity({"levels/execution/cpu::keyed-ticket-reservation",
             "x86",
             ParityStatus::Implemented,
             {"levels/execution/cpu::key-reduction",
              "levels/execution/cpu::keyed-atomics"},
             "Private stack bin tallying, sharded key owners, and single-owner atomic elision"});
  addParity({"levels/execution/cpu::tile-2d-transpose",
             "x86",
             ParityStatus::Implemented,
             {"exits/mojo::simd-transpose-8x8",
              "levels/execution/cpu::rectangular-copy",
              "levels/execution/cpu::permutation"},
             "In-register 8x8 SIMD butterfly transpose within 32x32 tiles and destination walks"});
  addParity({"levels/execution/cpu::counter-philox-rng",
             "x86",
             ParityStatus::Implemented,
             {"exits/mojo::philox-uniform", "levels/execution/cpu::pointwise",
              "levels/execution/cpu::allocate"},
             "10-round Philox-4x32 SIMD uniform microkernel with 4-lane contiguous reuse"});
  addParity({"levels/execution/cpu::narrow-i32-addressing",
             "x86",
             ParityStatus::Implemented,
             {"exits/mojo::narrow-proven-addresses"},
             "proveNarrowAddress-backed 32-bit signed coordinate and offset narrowing"});
  addParity({"levels/execution/cpu::riscv-vector-execution",
             "riscv64",
             ParityStatus::Unsupported,
             {},
             "Modular Mojo compiler toolchain only targets x86_64/aarch64; RISC-V RVV targets are served by exits/weft"});
  u.dialects = dialect;
  for (StringRef phase : {"legal", "realized", "callable"})
    u.boundaries.push_back(
        {{phase.str(),
          [phase](OpPassManager &pm, const TargetProfile &p) {
            if (phase == "callable")
              publishABI(pm);
            if (phase == "realized") {
              auto *caps = p.services.get<intentdsl::cpu::RowMachine>();
              if (caps)
                realize(pm, caps->vectorBits,
                        *p.services.get<intentdsl::CallFacts>());
            }
            return llvm::Error::success();
          },
          [phase](ModuleOp m, const TargetProfile &p) {
            if (phase == "legal")
              return intentdsl::cpu::verifyTask(m, p, "private-values");
            return verifyLoops(m, p, phase == "callable");
          }},
         COMPILERINFRA_SOURCE});
  ExitImplementation exit;
  exit.terminal = mojoMatrixPrinter();
  exit.capabilities = capabilities;
  u.exit = Located<ExitImplementation>{std::move(exit), COMPILERINFRA_SOURCE};
}
} // namespace CI_UNIT_NAMESPACE
