#include "intentdsl/levels/execution/cpu/ir/Traversal.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringExtras.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
// One synchronous OpenMP region owns the row groups. The Mojo interval
// receives its exact row range and never dispatches additional workers.
std::string hostSource(func::FuncOp entry) {
  SmallVector<std::string> slots, names;
  bool members=entry.front().front().hasAttr("cpu.member_tasks");
  auto keys=entry.front().front().getAttrOfType<IntegerAttr>("cpu.key_extent");
  unsigned projectedRank=0;
  if(entry->hasAttr("kir.iteration") && !entry->hasAttr("kir.owner_axis")){
    auto t=dyn_cast<MemRefType>(entry.getArgument(0).getType());
    if(t && t.getRank()>2) projectedRank=t.getRank();
  }
  for (auto [i, type] : llvm::enumerate(entry.getArgumentTypes())) {
    std::string n = "a" + std::to_string(i);
    auto view = dyn_cast<MemRefType>(type);
    if(projectedRank && view && view.getRank()==projectedRank){
      std::string offset="0";
      for(unsigned axis=0;axis+1<projectedRank;++axis)
        offset+="+c"+std::to_string(axis)+"*"+n+"s"+std::to_string(axis);
      names.push_back("(char*)"+n+"+("+offset+")*"+std::to_string(view.getElementTypeBitWidth()/8));
    }else if(members && view && i==entry->getAttrOfType<IntegerAttr>("kir.domain_arg").getInt())
      names.push_back("(char*)"+n+"+start*"+n+"s0*"+std::to_string(view.getElementTypeBitWidth()/8));
    else names.push_back(n);
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
  c << "#include <stdint.h>\n#include <omp.h>\nextern void intent_row_interval(" << params
    << ",int64_t,int64_t,int64_t,int64_t" << (keys?",int64_t,int64_t":"") << ");\n"
    << "void " << entry->getAttrOfType<StringAttr>("mojo_row.entry").getValue()
    << "(" << params
    << ",int64_t rows,int64_t cols,int64_t grain,int64_t columns,int64_t "
       "workers){\n"
    << "if (rows == 0) return;\n";
  auto &task=entry.front().front();
  if(keys){
    c<<"const int64_t keys="<<keys.getInt()<<",lines=(keys+15)/16,n=workers<lines?workers:lines;\n"
      <<"#pragma omp parallel for num_threads(n) schedule(static) if(n>1)\n"
      <<"for(int64_t b=0;b<n;++b)intent_row_interval("<<llvm::join(names,",")<<",0,rows,cols,columns,16*(lines*b/n),16*(lines*(b+1)/n));}\n";
    return out;
  }
  if(task.hasAttr("cpu.single_dispatcher"))c<<"workers=1;\n";
  if(members)c<<"rows=cols;\n";
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
    << "#pragma omp parallel num_threads(threads) if(threads>1)\n"
    << "{int64_t b=omp_get_thread_num(), nt=omp_get_num_threads();"
       "int64_t q=groups/nt, rem=groups%nt;"
       "int64_t start=(q*b+(b<rem?b:rem))*grain;"
       "int64_t stop=b+1==nt?rows:start+(q+(b<rem))*grain;\n";
  if(projectedRank) {
    c<<"while(start<stop){int64_t outer=start;\n";
    for(int axis=projectedRank-2;axis>=0;--axis)
      c<<"int64_t c"<<axis<<"=outer%a0d"<<axis<<";outer/=a0d"<<axis<<";\n";
    c<<"int64_t span=a0d"<<projectedRank-2<<"-c"<<projectedRank-2
     <<";int64_t end=stop-start<span?stop:start+span;\n"
     <<"intent_row_interval("<<llvm::join(names,",")<<",start,end,cols,columns);start=end;}\n";
  } else if(members)c<<"intent_row_interval("<<llvm::join(names,",")<<",start,start+1,stop-start,columns);\n";
  else c<<"intent_row_interval("<<llvm::join(names,",")<<",start,stop,cols,columns);\n";
  c<<"}\n}\n";
  return out;
}
} // namespace CI_UNIT_NAMESPACE
