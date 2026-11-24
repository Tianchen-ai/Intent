#include "../legalize/Matrix.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include <algorithm>

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
           int64_t);
LogicalResult verifyHost(ModuleOp, const TargetProfile &, StringRef);

namespace {
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
  if (options.empty())
    return p;
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
  bool hasZvfbfmin = options.getBoolean("zvfbfmin").value_or(false);
  if (vecRegs <= 0 || maxLmul8 <= 0)
    return llvm::createStringError("invalid probed vector register budget");

  auto caps = std::make_shared<intentdsl::cpu::RowMachine>();
  caps->vectorBits = *bits;
  caps->structuredRows = false;
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
          RecordFields{caps->boundWidening, vecRegs, maxLmul8, hasZvfbfmin})))
    return e;

  // Locally reported mvendorid/marchid for the V100 core.
  bool v100 = options.getString("core") == "0x710/0x8000000058000002";
  int64_t columns =
      isRvv ? std::max<int64_t>(32, 4 * v) : (*bits == 128 ? 32 : 64);
  int64_t midGrain = (*bits == 1024 || (*bits == 256 && v100)) ? 2 : 4;
  caps->rows = {{1, columns}, {midGrain, columns}, {32, columns}};
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
            lower(pm, phase, *p.services.get<intentdsl::CallFacts>(),
                  p.services.get<RecordFields>()->grouped, tailVec);
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
