#pragma once
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
namespace intentdsl::cpu {
struct Reservations {mlir::Operation *update{}, *load{}, *scatter{}, *offsets{};};
inline Reservations matchReservations(mlir::Block &body){
  using namespace mlir;
  Reservations m;
  for(auto &v:body.without_terminator()){
    auto n=v.getName().stripDialect();
    if(n=="atomic_add"){if(m.update)return {};m.update=&v;}
    else if(n=="unique_put"){if(m.scatter)return {};m.scatter=&v;}
    else if(n=="index_read"){if(m.offsets)return {};m.offsets=&v;}
    else if(n!="read" && n!="add" && n!="constant" && n!="ordinal")return {};
  }
  if(!m.update)return {};
  m.load=m.update->getOperand(1).getDefiningOp();auto *one=m.update->getOperand(2).getDefiningOp();
  if(!m.load || m.load->getName().stripDialect()!="read" || !one || !one->getAttrOfType<IntegerAttr>("value") ||
     one->getAttrOfType<IntegerAttr>("value").getInt()!=1)return {};
  if(!m.scatter)return m.update->use_empty() && !m.offsets?m:Reservations{};
  auto *pos=m.scatter->getOperand(1).getDefiningOp(),*ord=m.scatter->getOperand(2).getDefiningOp();
  if(!m.offsets || !pos || pos->getName().stripDialect()!="add" || pos->getOperand(0)!=m.offsets->getResult(0) ||
     pos->getOperand(1)!=m.update->getResult(0) || m.offsets->getOperand(1)!=m.load->getResult(0) ||
     !ord || ord->getName().stripDialect()!="ordinal")return {};
  return m;
}
inline bool reservations(mlir::Block &b){return !b.empty() && b.front().getName().getStringRef()=="cpu.tally_keys";}
inline bool formReservations(mlir::Block &bb) {
  using namespace mlir;
  if (reservations(bb))
    return true;
  auto keys=bb.getParentOp()->getAttrOfType<IntegerAttr>("cpu.batch_extent");
  if(!keys)return false;
  auto m=matchReservations(bb);if(!m.update)return false;
  SmallVector<Operation*> old;for(auto &v:bb.without_terminator())old.push_back(&v);
  OpBuilder b(bb.getTerminator());auto loc=m.update->getLoc();
  auto storage=MemRefType::get({keys.getInt()},b.getI32Type());
  OperationState count(loc,"cpu.tally_keys");count.addOperands(m.load->getOperand(0));count.addTypes(storage);
  auto counts=b.create(count)->getResult(0);
  OperationState reserve(loc,"cpu.reserve_tickets");reserve.addOperands({m.update->getOperand(0),counts});
  reserve.addAttribute("ordering",m.update->getAttr("ordering"));reserve.addTypes(storage);
  auto bases=b.create(reserve)->getResult(0);
  if(m.scatter){
    OperationState put(loc,"cpu.distribute_tickets");
    put.addOperands({m.load->getOperand(0),m.offsets->getOperand(0),bases,m.scatter->getOperand(0)});b.create(put);
  }
  for(auto *v:llvm::reverse(old))v->erase();
  return true;
}
inline mlir::LogicalResult verifyReservations(mlir::Block &body,const CallFacts &facts){
  using namespace mlir;
  auto &tally=body.front();auto *get=tally.getNextNode();
  auto size=body.getParentOp()->getAttrOfType<IntegerAttr>("cpu.batch_extent");
  if(!size || size.getInt()<=0 || size.getInt()>1024 || !get || get->getName().stripDialect()!="reserve_tickets" || get->getOperand(1)!=tally.getResult(0) ||
     get->getAttrOfType<StringAttr>("ordering")!="relaxed" || get->getResult(0).getType()!=tally.getResult(0).getType() ||
     cast<MemRefType>(tally.getResult(0).getType()).getShape()!=ArrayRef<int64_t>{size.getInt()})
    return tally.emitError("invalid private count and atomic reservation");
  auto counter=dyn_cast<BlockArgument>(get->getOperand(0));
  if(!counter || counter.getOwner()!=&body || !counter.getArgNumber() || counter.getArgNumber()>facts.arguments.size())
    return get->emitError("reserve requires a bound counter vector");
  auto record=facts.arguments[counter.getArgNumber()-1].getAsObject();auto shape=record?record->getArray("shape"):nullptr;
  if(!shape || shape->size()!=1 || shape->front().getAsInteger()!=size.getInt())
    return get->emitError("private histogram extent differs from the counter vector");
  auto *put=get->getNextNode();if(put==body.getTerminator())return success();
  if(put->getName().stripDialect()!="distribute_tickets" || put->getOperand(0)!=tally.getOperand(0) || put->getOperand(2)!=get->getResult(0) ||
     put->getNextNode()!=body.getTerminator())return put->emitError("ticket replay must consume its counts and original keys");
  return success();
}
inline bool atomicMembers(mlir::Block &bb){
  return llvm::hasSingleElement(bb.without_terminator()) && bb.front().getName().getStringRef()=="cpu.atomic_members";
}
inline bool formAtomicMembers(mlir::Block &bb){
  using namespace mlir;
  if (atomicMembers(bb))
    return true;
  if(!llvm::any_of(bb,[](Operation &op){auto n=op.getName().stripDialect();return n=="atomic_add" || n=="write_offsets" || n=="fill_ranges";}))return false;
  OpBuilder b(bb.getTerminator());OperationState state(bb.front().getLoc(),"cpu.atomic_members");state.addRegion();
  auto *loop=b.create(state);auto &body=loop->getRegion(0).emplaceBlock();
  while(&bb.front()!=loop)bb.front().moveBefore(&body,body.end());
  auto row=body.addArgument(b.getIndexType(),loop->getLoc());
  bb.getArgument(0).replaceUsesWithIf(row,[&](OpOperand &use){return use.getOwner()->getBlock()==&body;});
  b.setInsertionPointToEnd(&body);b.create(OperationState(loop->getLoc(),"cpu.end"));return true;
}
}
