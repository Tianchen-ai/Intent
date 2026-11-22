#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"

namespace intentdsl::kir {
struct Traffic {double keys=0, segments=0;};

// A packed member region binds the absolute member and independent feature.
// Only a direct use of those coordinates establishes a packed source extent.
inline double memberReads(mlir::Block &body,const CallFacts &facts){
  double members=0,features=1,bytes=0;
  for(auto &op:body){
    if(op.getName().stripDialect()!="member_load")continue;
    auto view=op.getOperand(0);auto *shape=facts.shapeOf(view);
    if(!shape || shape->empty())continue;
    bytes+=mlir::cast<mlir::MemRefType>(view.getType()).getElementTypeBitWidth()/8.;
    if(op.getOperand(1)!=body.getArgument(0))continue;
    members=std::max(members,double(shape->front().getAsInteger().value_or(0)));
    if(op.getNumOperands()==3 && op.getOperand(2)==body.getArgument(1) && shape->size()==2)
      features=std::max(features,double(shape->back().getAsInteger().value_or(0)));
  }
  return members*features*bytes;
}
inline Traffic valueTraffic(mlir::Operation *program,const CallFacts &facts){
  Traffic traffic;
  program->walk([&](mlir::Operation *op){
    auto name=op->getName().stripDialect();
    if(name=="count_keys" || name=="column_sum")traffic.keys+=facts.viewBytes(op->getOperand(0));
    if(name=="ordered_sum")traffic.segments+=memberReads(op->getRegion(0).front(),facts);
  });
  return traffic;
}
}
