#pragma once
#include "intentdsl/levels/execution/block/ir/Contraction.h"
namespace intentdsl::block {
struct MatrixAccess {
  mlir::func::FuncOp fn;
  mlir::Operation &domain,&fold,&dot;
  mlir::DenseI64ArrayAttr tile,shape;
  explicit MatrixAccess(mlir::func::FuncOp f):fn(f),domain(f.front().front()),
      fold(*llvm::find_if(domain.getRegion(0).front(),[](mlir::Operation &o){return o.getName().stripDialect()=="panel_fold";})),
      dot(*fold.getRegion(0).front().back().getOperand(0).getDefiningOp()),tile(domain.getAttrOfType<mlir::DenseI64ArrayAttr>("block.output_tile")),
      shape(domain.getAttrOfType<mlir::DenseI64ArrayAttr>("block.extents")){}
  unsigned slot(mlir::Value v)const{return mlir::cast<mlir::BlockArgument>(v).getArgNumber();}
  std::string name(mlir::Value v)const{return "a"+std::to_string(slot(v));}
  int64_t stride(mlir::Value v,int axis){return fn.getArgAttrOfType<mlir::DenseI64ArrayAttr>(slot(v),"block.strides")[axis];}
  int rank(mlir::Value v)const{return mlir::cast<mlir::MemRefType>(v.getType()).getRank();}
  int axis(llvm::StringRef key)const{return dot.getAttrOfType<mlir::IntegerAttr>(key).getInt();}
  int freeAxis(mlir::Value v,int k,int b)const{for(int i=0;i<rank(v);++i)if(i!=k && i!=b)return i;llvm_unreachable("verified contraction view");}
  bool routed()const{return dot.getName().stripDialect()=="contract" && dot.getNumOperands()==4;}
};
}
