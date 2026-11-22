#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"

namespace intentdsl::kir {
// The two arguments are absolute member and output feature. Only read-only
// view captures may cross the region boundary. All arithmetic is local.
template<class IsInput>
mlir::LogicalResult verifyMemberExpression(mlir::Operation *op, IsInput isInput) {
  using namespace mlir;
  if(op->getNumRegions()!=1 || !llvm::hasSingleElement(op->getRegion(0)))
    return op->emitError("expected one ordered member region");
  auto &bb=op->getRegion(0).front();
  if(bb.empty() || bb.getNumArguments()!=2 || !bb.getArgument(0).getType().isIndex() ||
     !bb.getArgument(1).getType().isIndex() || bb.back().getName().stripDialect()!="term")
    return op->emitError("expected member, feature and term");
  llvm::SmallDenseSet<Value> local{bb.getArgument(0),bb.getArgument(1)};
  for(auto &v:bb.without_terminator()){
    auto n=v.getName().stripDialect();
    if(v.getNumResults()!=1 || v.getNumRegions())return v.emitError("expected a scalar expression");
    if(n=="member_load"){
      auto source=v.getOperand(0);
      auto type=dyn_cast<MemRefType>(source.getType());
      if(!type || !isInput(source) || v.getNumOperands()!=type.getRank()+1 ||
         v.getResult(0).getType()!=type.getElementType())return v.emitError("invalid indexed input");
      for(auto index:v.getOperands().drop_front())
        if(!local.contains(index) || !index.getType().isSignlessIntOrIndex())return v.emitError("invalid member index");
    }else if((n!="add" && n!="mul") || v.getNumOperands()!=2 || !v.getResult(0).getType().isF32() ||
             !llvm::all_of(v.getOperands(),[&](Value a){return local.contains(a) && a.getType().isF32();}))
      return v.emitError("expected strict scalar f32 arithmetic");
    local.insert(v.getResult(0));
  }
  if(bb.back().getNumOperands()!=1 || !bb.back().getOperand(0).getType().isF32() || !local.contains(bb.back().getOperand(0)))
    return op->emitError("term must yield the local f32 expression");
  return success();
}

inline mlir::Operation *contiguousFeatureRead(mlir::Operation *fold) {
  auto &bb=fold->getRegion(0).front();
  auto *v=bb.back().getOperand(0).getDefiningOp();
  if(v && v->getName().stripDialect()=="member_load" && v->getNumOperands()==3 &&
     v->getOperand(1)==bb.getArgument(0) && v->getOperand(2)==bb.getArgument(1))return v;
  return nullptr;
}
}
