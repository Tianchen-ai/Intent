#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Traffic.h"

namespace intentdsl::block {
inline kir::Traffic realizedTraffic(mlir::Operation *program,const CallFacts &facts){
  kir::Traffic traffic;
  program->walk([&](mlir::Operation *op){
    auto name=op->getName().getStringRef();
    if(name=="block.key_counts" || name=="block.sum_columns")traffic.keys+=facts.viewBytes(op->getOperand(0));
    if(name=="block.segment_fold")traffic.segments+=kir::memberReads(op->getRegion(0).front(),facts);
  });
  return traffic;
}
}
