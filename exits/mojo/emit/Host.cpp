#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringExtras.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
// One synchronous OpenMP region owns the row groups. The Mojo interval
// receives its exact row range and never dispatches additional workers.
std::string hostSource(func::FuncOp entry) {
  SmallVector<std::string> slots, names;
  for (auto [i, type] : llvm::enumerate(entry.getArgumentTypes())) {
    std::string n = "a" + std::to_string(i);
    auto view = dyn_cast<MemRefType>(type);
    names.push_back(n);
    slots.push_back(std::string(view                 ? "void *"
                                : type.isInteger(32) ? "int32_t "
                                                     : "float ") +
                    n);
    if (!view)
      continue;
    for (int64_t j = 0; j < view.getRank(); ++j) {
      auto s = n + "s" + std::to_string(j);
      slots.push_back("int64_t " + s);
      names.push_back(s);
    }
  }
  auto params = llvm::join(slots, ",");
  std::string out;
  llvm::raw_string_ostream c(out);
  c << "#include <stdint.h>\nextern void intent_row_interval(" << params
    << ",int64_t,int64_t,int64_t,int64_t);\n"
    << "void " << entry->getAttrOfType<StringAttr>("mojo_row.entry").getValue()
    << "(" << params
    << ",int64_t rows,int64_t cols,int64_t grain,int64_t columns,int64_t "
       "workers){\n"
    << "if (rows == 0 || cols == 0) return;\nconst int64_t "
       "groups=(rows-1)/grain+1;\n"
    << "const int64_t threads=groups<workers?groups:workers;\n"
    << "#pragma omp parallel for num_threads(threads) if(threads>1) "
       "schedule(static)\n"
    << "for(int64_t g=0;g<groups;++g){int64_t start=g*grain;int64_t "
       "stop=rows-start<grain?rows:start+grain;\n"
    << "intent_row_interval(" << llvm::join(names, ",")
    << ",start,stop,cols,columns);}\n}\n";
  return out;
}
} // namespace CI_UNIT_NAMESPACE
