#pragma once
#include "intentdsl/levels/execution/block/ir/MatrixAccess.h"
#include "mlir/IR/Builders.h"
namespace intentdsl::block {
inline llvm::StringRef scaleMode(mlir::Operation &dot, llvm::StringRef native) {
  auto fn = dot.getParentOfType<mlir::func::FuncOp>();
  auto mode = fn->getAttrOfType<mlir::StringAttr>("kir.fp8_dot");
  return mode == "source" || mode == "fp32-product" ? "float32" : native;
}
inline void formScaleDispatch(mlir::Operation &dot,llvm::StringRef fast){
  mlir::OpBuilder b(&dot);
  mlir::OperationState s(dot.getLoc(),dot.getName().getDialectNamespace().str()+".scale_dispatch");
  s.addOperands({dot.getOperand(1),kir::RightScale(dot).view});
  s.addAttribute("interval",b.getDenseI64ArrayAttr({87,163}));
  s.addAttribute("on_match", b.getStringAttr(scaleMode(dot, fast)));
  s.addAttribute("otherwise",b.getStringAttr("float32"));b.create(s);
}
inline mlir::LogicalResult verifyScaleDispatch(mlir::func::FuncOp fn,const compilerinfra::TargetProfile &p,llvm::StringRef ns,llvm::StringRef fast){
  bool scaled=fn.walk([](mlir::Operation *o){return o->getName().stripDialect()=="scaled_contract"?mlir::WalkResult::interrupt():mlir::WalkResult::advance();}).wasInterrupted();
  if(!scaled)return verifyTiledMatrix(fn,p,ns);
  return verifyTiledMatrix(fn,p,ns,[&](mlir::Operation &s){
    auto *dot=s.getNextNode();mlir::Builder b(fn.getContext());
    if (s.getName().getStringRef() != ns.str() + ".scale_dispatch" || !dot ||
        dot->getName().stripDialect() != "scaled_contract" ||
        s.getOperand(0) != dot->getOperand(1) ||
        s.getOperand(1) != kir::RightScale(*dot).view ||
        s.getAttr("interval") != b.getDenseI64ArrayAttr({87, 163}) ||
        s.getAttr("on_match") != b.getStringAttr(scaleMode(*dot, fast)) ||
        s.getAttr("otherwise") != b.getStringAttr("float32"))
      return s.emitError("invalid E8M0 dispatch"), mlir::failure();
    return mlir::success();
  });
}
}
