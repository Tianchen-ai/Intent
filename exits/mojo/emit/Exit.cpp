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
  caps->vectorBits = *bits;
  caps->packedOwners = std::clamp<int64_t>(*bits / 32, 4, 16);
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
  registerBlockCost(u);
  registerOneDNN(u);
  registerMatrixContracts(u);
  registerMojoAddressOptimization(u);
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
