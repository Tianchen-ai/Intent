#pragma once

#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace CI_UNIT_NAMESPACE {
inline mlir::DictionaryAttr indexedLoop(mlir::Builder b,int64_t depth){
  llvm::SmallVector<mlir::Attribute> batch{b.getStringAttr("stage_contiguous_indices"),
                                         b.getStringAttr("async_dma_per_member")};
  if(depth==1)batch.push_back(b.getStringAttr("wait_io"));
  return b.getDictionaryAttr({b.getNamedAttr("when",b.getStringAttr("batch_contiguous_u32_pitch")),
          b.getNamedAttr("batch",b.getArrayAttr(batch)),
          b.getNamedAttr("otherwise",b.getStringAttr("blocking_member_gather"))});
}
// Each iteration covers the remainder of the current innermost member axis.
// A guarded pitched instruction may consume several complete runs at once.
inline mlir::DictionaryAttr slabLoop(mlir::Operation *op, bool read,
                                     const intentdsl::CallFacts &facts) {
  using namespace mlir;
  Builder b(op->getContext());
  auto fn=op->getParentOfType<func::FuncOp>();
  auto owner=fn->getAttrOfType<IntegerAttr>("kir.owner_axis").getInt();
  auto view=cast<BlockArgument>(op->getOperand(0));
  const llvm::json::Array *strides=nullptr;
  if(!fn.getArgAttr(view.getArgNumber(),"kir.source") &&
     !fn.getArgAttr(view.getArgNumber(),"kir.indexing") && view.getArgNumber()<facts.arguments.size())
    if(auto *arg=facts.arguments[view.getArgNumber()].getAsObject())
      strides=arg->getArray("strides");
  int64_t innerSlot=-1,outerSlot=-1,extentSlot=-1,extentAxis=-1;
  for(auto [pos,coord]:llvm::enumerate(op->getOperands().drop_front(read?2:3))){
    auto *axis=coord.getDefiningOp();
    if(axis->getName().getStringRef()!="dsa.slab_axis")continue;
    auto domain=cast<BlockArgument>(axis->getOperand(1));
    int rank=cast<MemRefType>(domain.getType()).getRank();
    extentSlot=domain.getArgNumber();extentAxis=rank-1-(owner==rank-1);
    if(axis->getAttrOfType<IntegerAttr>("axis").getInt()==extentAxis)innerSlot=pos;
    else if(rank==3)outerSlot=pos;
  }
  std::optional<int64_t> stride=innerSlot<0?std::optional<int64_t>(0):std::nullopt;
  if(innerSlot>=0 && strides && unsigned(innerSlot)<strides->size())
    stride=(*strides)[innerSlot].getAsInteger();
  SmallVector<Attribute> body;
  if(outerSlot>=0 && (!stride || *stride==1))body.push_back(b.getStringAttr("pitched_full_runs"));
  if(!stride || *stride==1)body.push_back(b.getStringAttr("contiguous"));
  if(read && (!stride || *stride==0))body.push_back(b.getStringAttr("broadcast"));
  if(!stride || (*stride!=1 && (!read || *stride!=0)))body.push_back(b.getStringAttr("elementwise"));
  return b.getDictionaryAttr({b.getNamedAttr("loop",b.getStringAttr("until_live_split_on_member_wrap")),
      b.getNamedAttr("extent_slot",b.getI64IntegerAttr(extentSlot)),b.getNamedAttr("extent_axis",b.getI64IntegerAttr(extentAxis)),
      b.getNamedAttr("inner_stride",b.getI64IntegerAttr(innerSlot)),b.getNamedAttr("outer_pitch",b.getI64IntegerAttr(outerSlot)),
      b.getNamedAttr("body",b.getArrayAttr(body))});
}
}
