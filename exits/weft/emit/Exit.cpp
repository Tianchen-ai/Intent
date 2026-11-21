#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
using namespace compilerinfra;
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
llvm::Expected<std::string> taskSource(Operation *, const llvm::json::Array &);
std::string hostSource(Operation *, const llvm::json::Array &);
void dialect(DialectRegistry &);
void lower(OpPassManager &, bool);
LogicalResult verifyHost(ModuleOp, const TargetProfile &, bool);
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
  auto caps = std::make_shared<intentdsl::cpu::RowMachine>();
  caps->vectorBits = *bits;
  caps->structuredRows = true;
  caps->workers = *workers;
  caps->wideningPanel = *bits / 32;
  caps->boundWidening = options.getString("arch") == "riscv64";
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
  auto fn = *m.getOps<func::FuncOp>().begin();
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
  fields["version"] = 3;
  fields["domain"]=intentdsl::publishDomain(fn);
  fields["abi"] = std::move(*abi);
  fields["rows"] = std::move(rows);
  out << llvm::json::Value(std::move(fields));
  return llvm::Error::success();
}
} // namespace
void bind(Unit &u) {
  u.dialects = dialect;
  u.realizers.push_back(
      {{"task-values", "levels/execution/cpu::storage",
        [](Operation *op, Attribute requirement, const TargetProfile &,
           RewriterBase &rw) -> llvm::Error {
          if (requirement != rw.getStringAttr("private-values"))
            return llvm::createStringError(
                "Weft row task requires private SSA storage");
          op->setAttr("cpu.realization", rw.getStringAttr("weft-task-ssa"));
          op->removeAttr("cpu.storage");
          return llvm::Error::success();
        }},
       COMPILERINFRA_SOURCE});
  for (StringRef phase : {"legal", "callable"})
    u.boundaries.push_back({{phase.str(),
                             [phase](OpPassManager &pm, const TargetProfile &) {
                               lower(pm, phase == "callable");
                               return llvm::Error::success();
                             },
                             [phase](ModuleOp m, const TargetProfile &p) {
                               return verifyHost(m, p, phase == "callable");
                             }},
                            COMPILERINFRA_SOURCE});
  u.exit =
      Located<ExitImplementation>{{emit, capabilities}, COMPILERINFRA_SOURCE};
}
} // namespace CI_UNIT_NAMESPACE
