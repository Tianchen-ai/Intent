#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "llvm/ADT/SmallPtrSet.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
// Publish independent reductions on the second CTA axis if row ownership alone
// leaves SMs idle. Each region contains the complete producer of one output.
void partitionOwners(func::FuncOp fn,const intentdsl::CallFacts &facts){
  auto &body=fn.front();
  if(!fn->hasAttr("kir.owner_axis") || fn->getAttrOfType<StringAttr>("block.topology")!="stream")return;
  auto abi=llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi").getValue());
  if(!abi){llvm::consumeError(abi.takeError());return;}
  for(auto &slot:*abi->getAsArray())
    if(slot.getAsObject()->getString("direction")=="InOut")return;
  auto [owners,columns]=facts.geometry(fn);
  if(!owners || owners>=facts.participants)return;
  SmallVector<Operation*> stores;
  for(auto &op:body){
    auto n=op.getName().stripDialect();
    if(n=="owner_store")stores.push_back(&op);
    else if(n=="tiles"){
      if(op.getNumResults()!=1)return;
    }else if(n!="pid" && n!="rows" && n!="identity" && n!="cast" && n!="return")return;
  }
  if(stores.size()<2)return;
  SmallVector<SmallVector<Operation*>> groups;
  llvm::SmallPtrSet<Operation*,32> taken;
  for(auto *store:stores){
    auto *cast=store->getOperand(2).getDefiningOp();
    auto *loop=cast;
    if(cast && cast->getName().stripDialect()=="cast"){
      if(!cast->hasOneUse())return;
      loop=cast->getOperand(0).getDefiningOp();
    }
    if(!loop || loop->getName().stripDialect()!="tiles" || !loop->hasOneUse())return;
    // The loop may capture the common row indices, and no other sweep.
    bool dependent=false;
    loop->walk([&](Operation *inner){for(auto v:inner->getOperands()){
      auto *producer=v.getDefiningOp();
      if(producer && producer->getBlock()==&body && producer!=loop &&
         producer->getName().stripDialect()!="rows" && producer->getName().stripDialect()!="identity")dependent=true;
    }});
    if(dependent || !taken.insert(loop).second)return;
    SmallVector<Operation*> ops{loop};
    if(cast!=loop)ops.push_back(cast);
    ops.push_back(store);groups.push_back(std::move(ops));
  }
  OpBuilder b(body.getTerminator());
  for(auto [index,ops]:llvm::enumerate(groups)){
    OperationState region(fn.getLoc(),"ct_src.output_partition");
    region.addAttribute("index",b.getI32IntegerAttr(index));
    region.addRegion();
    auto *branch=b.create(region);
    auto *block=new Block;
    branch->getRegion(0).push_back(block);
    for(auto *op:ops)op->moveBefore(block,block->end());
    OpBuilder inside(fn.getContext());inside.setInsertionPointToEnd(block);
    inside.create(OperationState(fn.getLoc(),"ct_src.partition_end"));
  }
}
}
