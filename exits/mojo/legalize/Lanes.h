#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
namespace CI_UNIT_NAMESPACE {
inline int64_t laneCount(mlir::Operation *task,int64_t vectorBits) {
  if(task->hasAttr("cpu.member_tasks"))return 1;
  // Keep the mixed-precision accumulation width. Uniform half arithmetic
  // and half copies can occupy all the lanes in the same register.
  bool half=true;
  task->walk([&](mlir::Operation *op){
    for(auto type:op->getResultTypes()){
      auto shape=mlir::dyn_cast<mlir::ShapedType>(type);
      if(shape && !mlir::isa<mlir::MemRefType>(type))
        half &= shape.getElementType().isF16();
    }
  });
  return vectorBits/(half?16:32);
}
}
