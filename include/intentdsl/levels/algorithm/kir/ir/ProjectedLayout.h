#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "mlir/IR/Builders.h"

namespace intentdsl {
// Project exactly the binding used by this specialization. A missing fact
// leaves the ordinary strided ABI in effect.
inline mlir::DenseI64ArrayAttr projectedLayout(mlir::func::FuncOp fn,
    unsigned slot,const CallFacts &facts,bool shape,bool includeOwner=false) {
  using namespace mlir;
  auto map=fn.getArgAttrOfType<AffineMapAttr>(slot,"kir.indexing");
  auto bounds=fn->getAttrOfType<DenseI64ArrayAttr>("kir.iteration");
  auto src=fn.getArgAttrOfType<IntegerAttr>(slot,"kir.source");
  bool owned=bool(fn.getArgAttr(slot,"kir.owner_projection"));
  if(!map || !bounds || !src || (owned && !includeOwner))return {};
  SmallVector<int64_t> values(map.getValue().getNumDims(),0);
  auto get=[&](int64_t s,StringRef field,int64_t axis)->std::optional<int64_t>{
    if(s<0 || size_t(s)>=facts.arguments.size())return {};
    auto *f=facts.arguments[s].getAsObject();
    auto *a=f?f->getArray(field):nullptr;
    if(!a || axis<0 || size_t(axis)>=a->size())return {};
    return (*a)[axis].getAsInteger();
  };
  if(shape){
    for(unsigned j=0;j<values.size();++j){
      auto b=bounds.asArrayRef().slice(4*j,4);
      auto n=b[0]<0?std::optional<int64_t>(b[3]):get(b[0],"shape",b[1]);
      if(!n || *n<0 || b[2]<=0 || *n%b[2])return {};
      values[j]=b[0]<0?*n:*n/b[2];
    }
  }else for(auto [i,expr]:llvm::enumerate(map.getValue().getResults())){
    SmallVector<int64_t> factors(values.size()+1,0);
    auto stride=get(src.getInt(),"strides",i);
    if(!stride || !coefficients(expr,factors))return {};
    for(unsigned j=0;j<values.size();++j){
      int64_t term;
      if(__builtin_mul_overflow(factors[j],*stride,&term) ||
         __builtin_add_overflow(values[j],term,&values[j]))return {};
    }
  }
  if(owned){
    auto owner=fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
    if(!owner || owner.getInt()<0 || size_t(owner.getInt())>=values.size())return {};
    return Builder(fn.getContext()).getDenseI64ArrayAttr({values[owner.getInt()]});
  }
  return Builder(fn.getContext()).getDenseI64ArrayAttr(values);
}
inline mlir::LogicalResult specializeProjectedLayout(mlir::func::FuncOp fn,
    const CallFacts &facts,llvm::StringRef prefix,bool install){
  for(unsigned i=0;i<fn.getNumArguments();++i)for(bool shape:{false,true}){
    auto key=(prefix+(shape?".shape":".strides")).str();
    auto expected=projectedLayout(fn,i,facts,shape);
    if(install && expected)fn.setArgAttr(i,key,expected);
    if(fn.getArgAttr(i,key)!=expected)return fn.emitError("inconsistent bound layout for slot ")<<i;
  }
  return mlir::success();
}
}
