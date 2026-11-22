#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
namespace intentdsl::kir {
inline llvm::Expected<int64_t> keyExtent(mlir::Operation *op, const CallFacts &facts) {
  auto slot=mlir::dyn_cast<mlir::BlockArgument>(op->getOperand(0));
  if(!slot)return llvm::createStringError("key reduction requires a buffer capture");
  auto fn=op->getParentOfType<mlir::func::FuncOp>();
  unsigned i=slot.getArgNumber()-(slot.getOwner()!=&fn.front());
  if(i>=facts.arguments.size())return llvm::createStringError("missing key input binding");
  auto *d=facts.arguments[i].getAsObject();auto *shape=d?d->getArray("shape"):nullptr;
  if(!shape || shape->empty() || !shape->front().getAsInteger())return llvm::createStringError("missing reduction extent");
  int64_t extent=*shape->front().getAsInteger();
  if(auto span=op->getAttrOfType<mlir::IntegerAttr>("span")){
    auto [m,n]=facts.geometry(op);
    if(extent<0 || span.getInt()<=0 || m!=(extent?1+(extent-1)/span.getInt():0))return llvm::createStringError("partial rows must cover all key chunks");
  }
  return extent;
}
}
