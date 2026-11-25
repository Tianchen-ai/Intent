#include "../legalize/Matrix.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include <algorithm>
#include <fstream>
#include <string>

using namespace compilerinfra;
using namespace mlir;

namespace CI_UNIT_NAMESPACE {
void dialect(DialectRegistry &);
TerminalPrinter weftTerminalPrinter();
void registerWeftMatrixContracts(Unit &);
void populateWeftMatrixContracts(intentdsl::cpu::MatrixMachine &, int64_t,
                                 bool);
void registerWeftAddressOptimization(Unit &);
void publishWeftABI(OpPassManager &);
void lower(OpPassManager &, StringRef, const intentdsl::CallFacts &, bool,
           int64_t, int64_t);
LogicalResult verifyHost(ModuleOp, const TargetProfile &, StringRef);

namespace {
std::string probeHostWeftCalibration() {
#if defined(__x86_64__) || defined(_M_X64)
  return "x86";
#elif defined(__riscv)
  std::ifstream in("/proc/cpuinfo");
  std::string line, model, mv, ma;
  while (std::getline(in, line)) {
    auto pos = line.find(':');
    if (pos == std::string::npos)
      continue;
    auto k = llvm::StringRef(line.data(), pos).trim();
    auto v = llvm::StringRef(line.data() + pos + 1).trim();
    if (k == "model name" && model.empty())
      model = v.str();
    else if (k == "mvendorid")
      mv = v.str();
    else if (k == "marchid") {
      ma = v.str();
      break;
    }
  }
  if (model == "Spacemit(R) X100")
    return "k3-x100";
  if (model == "Spacemit(R) A100")
    return "k3-a100";
  std::string id = mv + "/" + ma;
  if (id == "0x5b7/0x80000000090c0d00")
    return "sg2044";
  if (id == "0x710/0x8000000058000002")
    return "rvv-v100";
  return "";
#else
  return "";
#endif
}

bool isWeftAtLeastV02(llvm::StringRef ver) {
  if (ver.empty())
    return false;
  llvm::SmallVector<llvm::StringRef, 3> parts;
  ver.split(parts, '.');
  if (parts.size() < 2)
    return false;
  int maj = 0, min = 0;
  if (parts[0].getAsInteger(10, maj) || parts[1].getAsInteger(10, min))
    return false;
  return maj > 0 || (maj == 0 && min >= 2);
}

llvm::Expected<TargetProfile> capabilities(const llvm::json::Object &options) {
  TargetProfile p;
  if (options.empty()) {
    p.services.calibrationProfile = probeHostWeftCalibration();
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

  bool isRvv = options.getString("arch") == "riscv64";
  if (!isRvv) {
    if (*bits >= 256)
      p.features.insert("avx2");
    if (*bits >= 512)
      p.features.insert("avx512f");
    if (options.getBoolean("grouped_i8").value_or(false) ||
        options.getBoolean("avx512_vnni").value_or(false))
      p.features.insert("avx512_vnni");
    for (StringRef feat : {"avx512_fp16", "amx_bf16", "amx_int8"})
      if (options.getBoolean(feat).value_or(false))
        p.features.insert(feat.str());

    auto weftVer = options.getString("weft_version").value_or("");
    if (isWeftAtLeastV02(weftVer)) {
      p.features.insert("triton_weft_v0_2");
      if (options.getBoolean("weft_brgemm_available").value_or(false) &&
          options.getString("weft_brgemm_runtime").value_or("") == "SEQ") {
        for (StringRef dt : {"f32", "f16", "bf16", "s8", "u8s8"}) {
          std::string key = ("weft_brgemm_" + dt).str();
          if (options.getBoolean(key).value_or(false))
            p.features.insert(key);
        }
      }
    }
  }

  int64_t v = *bits / 32;
  int64_t vecRegs = options.getInteger("vector_registers")
                        .value_or((isRvv || *bits >= 512) ? 32 : 16);
  int64_t maxLmul8 =
      options.getInteger("max_lmul_eighths").value_or(isRvv ? 64 : 8);
  int64_t privStack =
      options.getInteger("private_stack_bytes").value_or(65536);
  bool hasZvfbfmin = options.getBoolean("zvfbfmin").value_or(false);
  if (vecRegs <= 0 || maxLmul8 <= 0 || privStack <= 0)
    return llvm::createStringError("invalid probed vector register budget");

  auto caps = std::make_shared<intentdsl::cpu::RowMachine>();
  caps->vectorBits = *bits;
  caps->structuredRows = false;
  caps->orderedDotTokens = intentdsl::cpu::RowMachine::streamedDotTokens;
  caps->workers = *workers;
  if (isRvv) {
    // Software f32->bf16 RNE requires (4 * groups + 1) <= vector_registers.
    int64_t maxCastGroups =
        hasZvfbfmin ? (vecRegs / 2) : std::max<int64_t>(1, (vecRegs - 1) / 4);
    int64_t maxLmulGroups = std::max<int64_t>(1, maxLmul8 / 8);
    int64_t limitGroups =
        std::min({int64_t(4), maxCastGroups, maxLmulGroups,
                  std::max<int64_t>(1, int64_t(64) / v)});
    int64_t p2Groups = 1;
    while (p2Groups * 2 <= limitGroups)
      p2Groups *= 2;
    caps->wideningPanel = p2Groups * v;
  } else {
    caps->wideningPanel = v;
  }
  caps->boundWidening = isRvv;
  if (!isRvv && *bits >= 256)
    caps->packedOwners = std::clamp<int64_t>(*bits / 32, 4, 16);

  auto matrix = std::make_shared<intentdsl::cpu::MatrixMachine>();
  matrix->choices = {
      {2, v, 32, v}, {4, 2 * v, 64, v}, {8, 4 * v, 128, v}, {8, 3 * v, 64, v}};
  matrix->estimate = "exits/weft::native-panels";
  matrix->records = {{8, 1, 256, v}, {16, 1, 256, v}, {64, 1, 256, v}};
  populateWeftMatrixContracts(*matrix, v, isRvv);
  if (auto e = p.services.provide<intentdsl::cpu::MatrixMachine>(matrix))
    return e;
  if (auto e = p.services.provide<RecordFields>(std::make_shared<RecordFields>(
          RecordFields{caps->boundWidening, vecRegs, maxLmul8, hasZvfbfmin,
                       privStack})))
    return e;

  // Locally reported mvendorid/marchid for the V100 core.
  bool v100 = options.getString("core") == "0x710/0x8000000058000002";
  int64_t columns =
      isRvv ? std::max<int64_t>(32, 4 * v) : (*bits == 128 ? 32 : 64);
  int64_t midGrain = (*bits == 1024 || (*bits == 256 && v100)) ? 2 : 4;
  caps->rows = {{1, columns}, {midGrain, columns}, {8, columns}, {32, columns}};
  caps->segmentWidths = (!isRvv && v >= 16) ? SmallVector<int64_t, 2>{8, 16}
                                            : SmallVector<int64_t, 2>{8};
  caps->estimate = "exits/weft::row-primitive";
  if (auto e = p.services.provide<intentdsl::cpu::RowMachine>(caps))
    return e;
  return p;
}
} // namespace

void nativePanelCost(Unit &);
void bind(Unit &u) {
  nativePanelCost(u);
  registerWeftMatrixContracts(u);
  registerWeftAddressOptimization(u);
  auto addParity = [&](ParityDeclaration d) {
    u.parity.push_back({std::move(d), COMPILERINFRA_SOURCE});
  };
  addParity({"levels/execution/cpu::simd-matrix-contraction",
             "*",
             ParityStatus::Implemented,
             {"exits/weft::weft-simd-fma", "exits/weft::weft-f32-panel-fma",
              "levels/execution/cpu::contract-blocks",
              "levels/execution/cpu::matrix-partition"},
             "Weft outer_contract / packed panel FMA on x86 SIMD and RISC-V RVV"});
  addParity({"levels/execution/cpu::vendor-brgemm-contraction",
             "x86",
             ParityStatus::Implemented,
             {"exits/weft::onednn-brgemm-f32", "exits/weft::onednn-brgemm-f16",
              "exits/weft::onednn-brgemm-bf16", "exits/weft::onednn-brgemm-s8",
              "exits/weft::onednn-brgemm-u8s8",
              "levels/execution/cpu::contract-blocks"},
             "triton-weft >= 0.2.0 oneDNN SEQ BRGeMM ukernels on x86_64"});
  addParity({"levels/execution/cpu::vendor-brgemm-contraction",
             "riscv64",
             ParityStatus::Unsupported,
             {},
             "oneDNN x86 BRGeMM microkernels target x86_64 AVX/AMX ISAs; RISC-V uses native RVV weft-simd-fma / weft-f32-panel-fma"});
  addParity({"levels/execution/cpu::vnni-int8-contraction",
             "x86",
             ParityStatus::Implemented,
             {"exits/weft::onednn-brgemm-s8", "exits/weft::weft-simd-fma",
              "levels/execution/cpu::contract-blocks"},
             "x86 VNNI/AMX-INT8 packed s8 BRGeMM with widened SIMD si32 fallback"});
  addParity({"levels/execution/cpu::vnni-int8-contraction",
             "riscv64",
             ParityStatus::Unsupported,
             {},
             "x86 AVX-512 VNNI dot4 instructions are x86_64-specific; RISC-V lowers INT8 contractions via widened RVV si32 MAC in weft-simd-fma"});
  addParity({"levels/execution/cpu::q4k-packed-contraction",
             "*",
             ParityStatus::Implemented,
             {"exits/weft::weft-q4k-dot",
              "levels/execution/cpu::contract-blocks"},
             "Declarative GGUF Q4_K x Q8_K record encoding_decl and subquant level contraction"});
  addParity({"levels/execution/cpu::ordered-owner-pairing",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/cpu::shared-intervals",
              "levels/execution/cpu::ordered-owners",
              "exits/weft::weft-simd-fma"},
             "Multi-owner 2D outer_contract QK/PV panels and proven score window replay"});
  addParity({"levels/execution/cpu::row-reduction-and-scan",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/cpu::row-reduction",
              "levels/execution/cpu::row-scan", "exits/weft::task-values"},
             "Vectorized Weft row reductions, inclusive scan, Welford moments, and L1d row caching"});
  addParity({"levels/execution/cpu::keyed-ticket-reservation",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/cpu::key-reduction",
              "levels/execution/cpu::keyed-atomics", "exits/weft::task-values"},
             "Private local_alloc range reservations, sharded key owners, and single-owner atomic RMW"});
  addParity({"levels/execution/cpu::tile-2d-transpose",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/cpu::rectangular-copy",
              "levels/execution/cpu::permutation", "exits/weft::task-values"},
             "2D tile and destination-walk rectangular copy tasks"});
  addParity({"levels/execution/cpu::counter-philox-rng",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/cpu::pointwise", "exits/weft::task-values"},
             "10-round Philox-4x32 counter RNG with 4-lane contiguous output reuse"});
  addParity({"levels/execution/cpu::narrow-i32-addressing",
             "*",
             ParityStatus::Implemented,
             {"exits/weft::narrow-proven-addresses"},
             "proveNarrowAddress-verified 32-bit signed host tile offset narrowing"});
  addParity({"levels/execution/cpu::riscv-vector-execution",
             "*",
             ParityStatus::Implemented,
             {"exits/weft::weft-simd-fma", "exits/weft::task-values"},
             "Native RVV 1.0 vector/panel compilation across SG2044, RVV V100, K3 X100, and K3 A100"});
  u.dialects = dialect;
  u.realizers.push_back(
      {{"task-values", "levels/execution/cpu::storage",
        [](Operation *op, Attribute requirement, const TargetProfile &p,
           RewriterBase &rw) -> llvm::Error {
          return intentdsl::cpu::realizeLocalValues(op, requirement, p, rw,
                                                    "weft-task-ssa");
        }},
       COMPILERINFRA_SOURCE});
  for (StringRef phase : {"legal", "realized", "callable"})
    u.boundaries.push_back(
        {{phase.str(),
          [phase](OpPassManager &pm, const TargetProfile &p) {
            int64_t tailVec = 1;
            if (auto *rm = p.services.get<intentdsl::cpu::RowMachine>())
              tailVec = std::max<int64_t>(1, rm->vectorBits / 32);
            const auto *rf = p.services.get<RecordFields>();
            lower(pm, phase, *p.services.get<intentdsl::CallFacts>(),
                  rf->grouped, tailVec, rf->privateStackBytes);
            if (phase == "callable")
              publishWeftABI(pm);
            return llvm::Error::success();
          },
          [phase](ModuleOp m, const TargetProfile &p) {
            return verifyHost(m, p, phase);
          }},
         COMPILERINFRA_SOURCE});
  ExitImplementation exit;
  exit.terminal = weftTerminalPrinter();
  exit.capabilities = capabilities;
  u.exit = Located<ExitImplementation>{std::move(exit), COMPILERINFRA_SOURCE};
}
} // namespace CI_UNIT_NAMESPACE
