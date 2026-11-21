#include "intentdsl/levels/execution/block/decisions/Verify.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"
namespace intentdsl {
using namespace mlir;
llvm::Error verifyRows(ArrayAttr rows,StringAttr topology,DenseI64ArrayAttr guard,const compilerinfra::TargetProfile &p){
  auto *caps=p.services.get<RowCapacity>();
  if(!caps || !rows || rows.empty() || !topology || (topology!="resident" && topology!="stream")
      || !guard || guard.size()!=2) return llvm::createStringError("incomplete row domain");
  auto lo=guard[0],hi=guard[1];
  if(lo<0 || hi < -1 || (hi!=-1 && hi<lo) || lo>caps->observedWidth || (hi!=-1 && hi<caps->observedWidth))
    return llvm::createStringError("width guard excludes bound shape");
  llvm::DenseSet<int64_t> seen;
  for(Attribute a:rows){
    auto d=dyn_cast<DictionaryAttr>(a);
    if(!d || !d.getAs<IntegerAttr>("id") || !d.getAs<IntegerAttr>("rows") ||
       !d.getAs<IntegerAttr>("width") || !d.getAs<StringAttr>("payload"))return llvm::createStringError("incomplete parameter row");
    auto id=d.getAs<IntegerAttr>("id").getInt(), width=d.getAs<IntegerAttr>("width").getInt();
    if(!seen.insert(id).second || (topology=="resident" && (hi<0 || width<hi)) ||
       !llvm::any_of(caps->rows,[&](auto &r){return id==r.id && width==r.width &&
       d.getAs<IntegerAttr>("rows").getInt()==r.rows && d.getAs<StringAttr>("payload")==r.payload;}))
        return llvm::createStringError("invalid row id, width guard or native payload");
  }
  auto tile=cast<DictionaryAttr>(rows[0]).getAs<IntegerAttr>("width").getInt();
  int64_t limit=0,previous=0;
  for(auto &r:caps->rows){limit=std::max(limit,int64_t(r.width));if(r.width<tile)previous=std::max(previous,int64_t(r.width));}
  if(topology=="resident" && (hi!=tile || lo!=(previous?previous+1:0)))
    return llvm::createStringError("resident guard differs from target width class");
  if(topology=="stream" && (hi!=-1 || (lo!=0 && lo!=limit+1)))
    return llvm::createStringError("stream guard differs from open width class");
  return llvm::Error::success();
}

LogicalResult verifyMapping(ModuleOp m,const compilerinfra::TargetProfile &p,StringRef dialect){
  auto funcs=m.getOps<func::FuncOp>(); if(!llvm::hasSingleElement(funcs))return m.emitError("expected one row program");
  auto f=*funcs.begin();Builder b(m.getContext());
  auto topology=f->getAttrOfType<StringAttr>("block.topology");
  if(auto e=verifyRows(f->getAttrOfType<ArrayAttr>("block.configs"),topology,f->getAttrOfType<DenseI64ArrayAttr>("block.guard"),p))
    return f.emitError()<<llvm::toString(std::move(e));
  if(f->getAttr("block.grid")!=b.getStrArrayAttr({"ceildiv","M","rows"}))
    return f.emitError("grid must cover the M domain at the instance row step");
  unsigned pids=0,rowMaps=0,sweeps=0;
  auto result=f.walk([&](Operation *op){
    if(op->getName().getDialectNamespace()!=dialect)return WalkResult::advance();
    auto n=op->getName().stripDialect();
    auto def=[&](Value v,StringRef k){auto *d=v.getDefiningOp();return d && d->getName().getDialectNamespace()==dialect && d->getName().stripDialect()==k;};
    bool ok=true;
    if(n=="pid"){++pids;ok=op->getAttr("axis")==b.getI32IntegerAttr(0);}
    if(n=="rows"){++rowMaps;ok=op->getAttr("step")==b.getStringAttr("rows") && def(op->getOperand(0),"pid");}
    if(n=="tiles"){
      ++sweeps;
      if(!llvm::hasSingleElement(op->getRegion(0)))return WalkResult::interrupt();
      auto &body=op->getRegion(0).front();
      auto count=op->getNumResults();
      ok=count<=1 && op->getNumOperands()==count+1 && body.getNumArguments()==count+1 && !body.empty() &&
         body.getArgument(0).getType().isIndex() && body.back().getName().stripDialect()=="yield" && body.back().getNumOperands()==count;
      if(ok && count){
        auto init=op->getOperand(1),result=body.back().getOperand(0);
        ok=def(init,"identity") && def(result,"combine") &&
           result.getDefiningOp()->getOperand(0)==body.getArgument(1) &&
           init.getDefiningOp()->getAttr("algebra")==result.getDefiningOp()->getAttr("algebra") &&
           init.getType()==result.getType() && init.getType()==op->getResult(0).getType();
      }
    }
    if(n=="columns"){
      auto v=op->getOperand(0);
      ok=topology=="resident"?def(v,"zero"):isa<BlockArgument>(v) && cast<BlockArgument>(v).getArgNumber()==0 && v.getParentBlock()->getParentOp()->getName().stripDialect()=="tiles";
    }
    if(n=="active" || (dialect=="block" && (n=="load" || n=="store")))
      ok=def(op->getOperand(1),"rows") && def(op->getOperand(2),"columns");
    if(n=="reduce" || n=="combine" || n=="identity"){
      auto alg=op->getAttrOfType<StringAttr>("algebra");ok=alg && (alg=="max_num" || alg=="sum");
      if(n=="combine")ok &= def(op->getOperand(1),"reduce") && op->getOperand(1).getDefiningOp()->getAttr("algebra")==alg;
    }
    if(!ok){op->emitError("invalid row coordinates, instance mapping or reduction algebra");return WalkResult::interrupt();}
    return WalkResult::advance();
  });
  if(result.wasInterrupted())return failure();
  if(pids!=1 || rowMaps!=1 || ((topology=="stream")!=(sweeps!=0)))return f.emitError("instance mapping does not match topology");
  return success();
}
}
