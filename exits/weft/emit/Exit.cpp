#include "../legalize/Matrix.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"

using namespace compilerinfra;
using namespace mlir;

namespace CI_UNIT_NAMESPACE {
llvm::Expected<std::string> taskSource(Operation *, const llvm::json::Array &);
std::string hostSource(Operation *, const llvm::json::Array &);
void dialect(DialectRegistry &);
bool hasWeftMatrixProgram(ModuleOp);
TerminalPrinter weftMatrixPrinter();
void registerWeftMatrixContracts(Unit &);
void populateWeftMatrixContracts(intentdsl::cpu::MatrixMachine &, int64_t, bool);
void registerWeftAddressOptimization(Unit &);
void publishWeftMatrixABI(OpPassManager &);
llvm::Error emitOrdered(func::FuncOp, llvm::raw_pwrite_stream &);
void lower(OpPassManager &, StringRef, const intentdsl::CallFacts &, bool);
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

  auto caps = std::make_shared<intentdsl::cpu::RowMachine>();
  caps->vectorBits = *bits;
  caps->structuredRows = true;
  caps->workers = *workers;
  caps->wideningPanel = *bits / 32;
  caps->boundWidening = isRvv;

  int64_t v = *bits / 32;
  auto matrix = std::make_shared<intentdsl::cpu::MatrixMachine>();
  matrix->choices = {
      {2, v, 32, v}, {4, 2 * v, 64, v}, {8, 4 * v, 128, v}, {8, 3 * v, 64, v}};
  matrix->estimate = "exits/weft::native-panels";
  matrix->records = {{8, 1, 256, v}, {16, 1, 256, v}, {64, 1, 256, v}};
  populateWeftMatrixContracts(*matrix, v, isRvv);
  if (auto e = p.services.provide<intentdsl::cpu::MatrixMachine>(matrix))
    return e;
  if (auto e = p.services.provide<RecordFields>(
          std::make_shared<RecordFields>(RecordFields{caps->boundWidening})))
    return e;

  // Locally reported mvendorid/marchid for the V100 core.
  bool v100 = options.getString("core") == "0x710/0x8000000058000002";
  int64_t columns = *bits == 128 ? 32 : *bits == 1024 ? 128 : 64;
  caps->rows = {
      {1, columns},
      {(*bits == 1024 || (*bits == 256 && v100)) ? 2 : 4, 2 * columns}};
  if (auto e = p.services.provide<intentdsl::cpu::RowMachine>(caps))
    return e;
  return p;
}

llvm::Error emit(ModuleOp m, llvm::raw_pwrite_stream &out) {
  if (hasWeftMatrixProgram(m))
    return weftMatrixPrinter().emit(m, out);
  auto fn = *m.getOps<func::FuncOp>().begin();
  if (fn.front().front().getName().getStringRef() == "cpu.owner_task")
    return emitOrdered(fn, out);
  auto abi = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!abi)
    return abi.takeError();
  auto &task = fn.front().front();
  llvm::json::Array rows;
  for (auto a : task.getAttrOfType<ArrayAttr>("rows")) {
    auto row = cast<DictionaryAttr>(a);
    rows.push_back(llvm::json::Object{
        {"grain", row.getAs<IntegerAttr>("grain").getInt()},
        {"columns", row.getAs<IntegerAttr>("columns").getInt()}});
  }
  auto program = taskSource(&task, *abi->getAsArray());
  if (!program)
    return program.takeError();
  llvm::json::Object fields{{"task", *program},
                            {"host", hostSource(&task, *abi->getAsArray())}};
  fields["shape_parameters"] =
      task.getAttrOfType<BoolAttr>("weft_host.absolute_origin").getValue()
          ? llvm::json::Array{"T", "N", "C", "O"}
          : llvm::json::Array{"T", "N", "C"};
  if (task.hasAttr("cpu.key_extent")) {
    auto *symbols = fields["shape_parameters"].getAsArray();
    symbols->push_back("KL");
    symbols->push_back("KH");
  }
  fields["version"] = 3;
  llvm::json::Array compact;
  for (unsigned i = 0; i < fn.getNumArguments(); ++i)
    if (fn.getArgAttr(i, "weft_host.contiguous"))
      compact.push_back(i);
  if (!compact.empty())
    fields["contiguous_slots"] = std::move(compact);
  if (fn->hasAttr("kir.owner_axis")) {
    auto ranks = task.getAttrOfType<DenseI64ArrayAttr>("weft_host.slot_ranks");
    auto rank = ranks[intentdsl::domainSlot(fn)];
    for (int i = 0; i < rank; ++i)
      fields["shape_parameters"].getAsArray()->push_back(
          "D" + std::to_string(i));
    fields["task_ranks"] = llvm::json::Array(ranks.asArrayRef());
  }
  fields["domain"] = intentdsl::publishDomain(fn);
  fields["abi"] = std::move(*abi);
  fields["rows"] = std::move(rows);
  out << llvm::json::Value(std::move(fields));
  return llvm::Error::success();
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
        [](Operation *op, Attribute requirement, const TargetProfile &,
           RewriterBase &rw) -> llvm::Error {
          if (requirement != rw.getStringAttr("private-values"))
            return llvm::createStringError(
                "Weft task requires private SSA storage");
          if (op->getName().getStringRef() == "cpu.matrix_task")
            return intentdsl::cpu::materializeMatrixBuffers(
                op->getParentOfType<func::FuncOp>(), rw, "weft-task-ssa");
          op->setAttr("cpu.realization", rw.getStringAttr("weft-task-ssa"));
          op->removeAttr("cpu.storage");
          return llvm::Error::success();
        }},
       COMPILERINFRA_SOURCE});
  for (StringRef phase : {"legal", "realized", "callable"})
    u.boundaries.push_back(
        {{phase.str(),
          [phase](OpPassManager &pm, const TargetProfile &p) {
            lower(pm, phase, *p.services.get<intentdsl::CallFacts>(),
                  p.services.get<RecordFields>()->grouped);
            if (phase == "callable")
              publishWeftMatrixABI(pm);
            return llvm::Error::success();
          },
          [phase](ModuleOp m, const TargetProfile &p) {
            return verifyHost(m, p, phase);
          }},
         COMPILERINFRA_SOURCE});
  ExitImplementation exit;
  exit.emit = emit;
  exit.capabilities = capabilities;
  u.exit = Located<ExitImplementation>{std::move(exit), COMPILERINFRA_SOURCE};
}
} // namespace CI_UNIT_NAMESPACE
