#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include "llvm/ADT/STLExtras.h"
namespace intentdsl::dsa {
TemporaryLayout placeTemporaries(mlir::Block &body) {
  using namespace mlir;
  struct Live { unsigned end; int64_t offset, size; };
  llvm::DenseMap<Operation *,unsigned> ordinal;
  unsigned end = 0;
  for (auto &op : body) ordinal[&op] = end++;
  SmallVector<Live> live;
  TemporaryLayout layout;
  for (auto &op : body) {
    if (op.getNumResults()!=1) continue;
    auto code=op.getName().stripDialect();
    if (code=="load" || code=="copy_in" || code=="await" || code=="active") continue;
    auto value=op.getResult(0);
    auto t=dyn_cast<RankedTensorType>(value.getType());
    if (!t || t.getRank()!=2 || !t.getElementType().isIntOrFloat()) continue;
    auto index=ordinal[&op],last=index;
    for(auto *use:value.getUsers()){
      auto asynchronous=use->getAttrOfType<BoolAttr>("async");
      last=std::max(last,use->getBlock()!=&body || (asynchronous && asynchronous.getValue())?end:ordinal[use]);
    }
    llvm::erase_if(live,[index](const Live &slot){return slot.end<index;});
    auto size=(t.getElementTypeBitWidth()+7)/8;
    int64_t offset=0;
    for(auto slot:live){
      if(offset+size<=slot.offset)break;
      offset=slot.offset+slot.size;
    }
    layout.offsets[&op]=offset;
    layout.peak=std::max(layout.peak,offset+size);
    live.insert(llvm::lower_bound(live,offset,[](const Live &slot,int64_t key){return slot.offset<key;}),
                {last,offset,int64_t(size)});
  }
  return layout;
}
} // namespace intentdsl::dsa
