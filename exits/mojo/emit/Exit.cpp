#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
using namespace compilerinfra;
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
void dialect(DialectRegistry &);
std::string hostSource(func::FuncOp);
void publishABI(OpPassManager &);
LogicalResult verifyLoops(ModuleOp, const TargetProfile &, bool callable);
void realize(OpPassManager &, int64_t);
llvm::Expected<std::string> mojoSource(Operation *);
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
  caps->workers = *workers;
  caps->rows = {{1, 64}, {4, *bits == 512 ? 128 : 64}};
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
  auto program = mojoSource(&task);
  if (!program)
    return program.takeError();
  llvm::json::Object fields{{"source", *program}, {"host", hostSource(fn)}};
  fields["version"] = 2;
  fields["domain"]=intentdsl::publishDomain(fn);
  fields["abi"] = std::move(*abi);
  fields["rows"] = std::move(rows);
  out << llvm::json::Value(std::move(fields));
  return llvm::Error::success();
}
} // namespace
void bind(Unit &u) {
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
                realize(pm, caps->vectorBits / 32);
            }
            return llvm::Error::success();
          },
          [phase](ModuleOp m, const TargetProfile &p) {
            if (phase == "legal")
              return intentdsl::cpu::verifyTask(m, p, "private-values");
            return verifyLoops(m, p, phase == "callable");
          }},
         COMPILERINFRA_SOURCE});
  u.exit =
      Located<ExitImplementation>{{emit, capabilities}, COMPILERINFRA_SOURCE};
}
} // namespace CI_UNIT_NAMESPACE
