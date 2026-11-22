#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "../legalize/Matrix.h"
using namespace compilerinfra;
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
llvm::Expected<std::string> taskSource(Operation *, const llvm::json::Array &);
std::string hostSource(Operation *, const llvm::json::Array &);
void dialect(DialectRegistry &);
llvm::Error emitMatrix(func::FuncOp,llvm::raw_pwrite_stream &);
llvm::Error emitOrdered(func::FuncOp, llvm::raw_pwrite_stream &);
void lower(OpPassManager &, bool,const intentdsl::CallFacts &,bool);
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
  if(auto err=intentdsl::bindPerformance(p,options,"",options.getString("calibration").value_or("").str(),*workers))return std::move(err);
  auto caps = std::make_shared<intentdsl::cpu::RowMachine>();
  caps->vectorBits = *bits;
  caps->structuredRows = true;
  caps->workers = *workers;
  int64_t v=*bits/32;auto matrix=std::make_shared<intentdsl::cpu::MatrixMachine>();
  matrix->choices={{2,v,32,v},{4,2*v,64,v},{8,4*v,128,v},{8,3*v,64,v}};
  matrix->estimate="exits/weft::native-panels";
  if(*bits==512 && options.getBoolean("grouped_i8").value_or(false))matrix->integerPacking="reduction-packets";
  matrix->records={{8,1,256,v},{16,1,256,v},{64,1,256,v}};
  if(auto e=p.services.provide<intentdsl::cpu::MatrixMachine>(matrix))return e;
  caps->wideningPanel = *bits / 32;
  caps->boundWidening = options.getString("arch") == "riscv64";
  if(auto e=p.services.provide<RecordFields>(std::make_shared<RecordFields>(RecordFields{caps->boundWidening})))return e;
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
  if (fn.front().front().getName().getStringRef() == "cpu.owner_task") {
    return emitOrdered(fn, out);
  }
  if(intentdsl::cpu::matrixTask(fn))return emitMatrix(fn,out);
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
  if(task.hasAttr("cpu.key_extent")){
    auto *symbols=fields["shape_parameters"].getAsArray();
    symbols->push_back("KL");symbols->push_back("KH");
  }
  fields["version"] = 3;
  llvm::json::Array compact;
  for(unsigned i=0;i<fn.getNumArguments();++i)if(fn.getArgAttr(i,"weft_host.contiguous"))compact.push_back(i);
  if(!compact.empty())fields["contiguous_slots"]=std::move(compact);
  if(fn->hasAttr("kir.owner_axis")){
    auto ranks=task.getAttrOfType<DenseI64ArrayAttr>("weft_host.slot_ranks");
    auto rank=ranks[intentdsl::domainSlot(fn)];
    for(int i=0;i<rank;++i)fields["shape_parameters"].getAsArray()->push_back("D"+std::to_string(i));
    fields["task_ranks"]=llvm::json::Array(ranks.asArrayRef());
  }
  fields["domain"]=intentdsl::publishDomain(fn);
  fields["abi"] = std::move(*abi);
  fields["rows"] = std::move(rows);
  out << llvm::json::Value(std::move(fields));
  return llvm::Error::success();
}
} // namespace
void nativePanelCost(Unit &);
void bind(Unit &u) {
  nativePanelCost(u);
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
                             [phase](OpPassManager &pm, const TargetProfile &p) {
                               lower(pm, phase == "callable",*p.services.get<intentdsl::CallFacts>(),p.services.get<RecordFields>()->grouped);
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
