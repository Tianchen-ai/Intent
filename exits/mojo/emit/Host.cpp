#include "intentdsl/levels/execution/cpu/ir/Traversal.h"
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
    slots.push_back(std::string(view                         ? "void *"
                                : type.isUnsignedInteger(64) ? "uint64_t "
                                : type.isInteger(32)         ? "int32_t "
                                                             : "float ") +
                    n);
    if (!view)
      continue;
    if(view.getRank()>2)for(int j=0;j<view.getRank()-1;++j){
      auto d=n+"d"+std::to_string(j);slots.push_back("int64_t "+d);names.push_back(d);
    }
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
    << "if (rows == 0 || cols == 0) return;\n";
  auto &task=entry.front().front();
  auto walk=task.getAttrOfType<StringAttr>("cpu.walk");
  if(walk=="destination"){
    c<<"int64_t t=rows;rows=cols;cols=t;\n";
    for(int i=0;i<2;++i)c<<"t=a"<<i<<"s0;a"<<i<<"s0=a"<<i<<"s1;a"<<i<<"s1=t;\n";
  }
  if(walk=="tile"){
    c<<intentdsl::cpu::tileLoop(task.getAttrOfType<IntegerAttr>("cpu.tile").getInt());
    auto bytes=cast<MemRefType>(entry.getArgument(0).getType()).getElementTypeBitWidth()/8;
    c<<"intent_row_interval((char*)a0+(r*a0s0+c*a0s1)*"<<bytes<<",a0s0,a0s1,(char*)a1+(c*a1s0+r*a1s1)*"<<bytes<<",a1s0,a1s1,0,nr,nc,columns);}}\n";
    return out;
  }
  c << "const int64_t "
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
