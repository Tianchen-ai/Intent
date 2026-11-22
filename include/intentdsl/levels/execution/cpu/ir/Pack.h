#pragma once
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/JSON.h"

namespace intentdsl::cpu {
// A task owns one destination row and its count. Its inclusive prefix can be
// carried in a scalar register until the last accepted member is published.
inline mlir::LogicalResult packMemberLoop(mlir::Block &body) {
  using namespace mlir;
  SmallVector<Operation*> members;
  for(auto &op:body.without_terminator())members.push_back(&op);
  if(members.empty() || members.back()->getName().stripDialect()!="prefix_scatter")return success();
  auto *store=members.back(), *test=store->getOperand(3).getDefiningOp(),*value=store->getOperand(5).getDefiningOp();
  if(!test || test->getName().stripDialect()!="ne" || !value || value->getName().stripDialect()!="member_index")
    return store->emitError("CPU row scatter needs a member-index payload and a supported predicate");
  auto *load=test->getOperand(0).getDefiningOp(), *threshold=test->getOperand(1).getDefiningOp();
  if(!load || load->getName().stripDialect()!="read" || !threshold || !threshold->getAttrOfType<FloatAttr>("value"))
    return store->emitError("expected a row input and scalar comparison threshold");
  auto *scan=store->getOperand(4).getDefiningOp();
  if(!scan || scan->getName().stripDialect()!="scan_sum")return success();
  auto *cast=scan->getOperand(0).getDefiningOp();
  if(!cast || cast->getName().stripDialect()!="cast" || cast->getOperand(0)!=test->getResult(0))
    return success();
  // Erase exactly the matched expression. An unused SSA result does not
  // establish that its producer can be dropped (notably for atomic RMW).
  for(auto *op:members)
    if(op!=store && op!=test && op!=value && op!=load && op!=threshold && op!=scan && op!=cast)
      return success();
  OpBuilder b(store);OperationState loop(store->getLoc(),"cpu.pack_members");
  loop.addOperands({load->getOperand(0),store->getOperand(0),store->getOperand(1)});
  loop.addAttribute("threshold",threshold->getAttr("value"));loop.addAttribute("fill",store->getAttr("fill"));
  b.create(loop);for(auto *op:llvm::reverse(members))op->erase();
  return success();
}
inline bool packed(mlir::Block &body){
  return llvm::hasSingleElement(body.without_terminator()) && body.front().getName().getStringRef()=="cpu.pack_members";
}
inline mlir::LogicalResult verifyPack(mlir::Block &body){
  using namespace mlir;
  auto &op=body.front();auto fn=op.getParentOfType<func::FuncOp>();
  auto abi=llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi").getValue());
  if(!abi){llvm::consumeError(abi.takeError());return op.emitError("invalid packing ABI");}
  const llvm::json::Array *shape=nullptr;
  for(auto [i,v]:llvm::enumerate(op.getOperands())){
    auto arg=dyn_cast<BlockArgument>(v);
    if(!arg || arg.getOwner()!=&body || !arg.getArgNumber())return op.emitError("expected a task view");
    auto *slot=(*abi->getAsArray())[arg.getArgNumber()-1].getAsObject();
    auto dims=slot->getArray("dims");
    if(slot->getString("direction")!=(i?"Out":"In") || slot->getString("dtype")!=(i?"i32":"f32") ||
       !dims || dims->size()!=(i==2?1:2) || (i && (i==2?dims->front()!=shape->front():*dims!=*shape)))
      return op.emitError("source, packed values and count must cover the same owner rows");
    if(!i)shape=dims;
  }
  return success();
}
}
