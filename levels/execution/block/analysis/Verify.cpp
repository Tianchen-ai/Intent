#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/execution/block/ir/Access.h"
#include "intentdsl/levels/execution/block/decisions/Verify.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"
#include "intentdsl/levels/execution/block/ir/Grid.h"
namespace intentdsl {
using namespace mlir;
llvm::Error verifyRows(Operation *anchor,ArrayAttr rows,StringAttr topology,DenseI64ArrayAttr guard,const compilerinfra::TargetProfile &p){
  auto *caps=p.services.get<RowCapacity>();
  if(!caps || !rows || rows.empty() || !topology || (topology!="resident" && topology!="stream")
      || !guard || guard.size()!=2) return llvm::createStringError("incomplete row domain");
  auto lo=guard[0],hi=guard[1];
  auto extent=caps->width(anchor);
  if(!extent)return extent.takeError();
  if(lo<0 || hi < -1 || (hi!=-1 && hi<lo) || lo>*extent || (hi!=-1 && hi<*extent))
    return llvm::createStringError("width guard excludes bound shape");
  llvm::DenseSet<int64_t> seen;
  for(Attribute a:rows){
    auto d=dyn_cast<DictionaryAttr>(a);
    if(!d || !d.getAs<IntegerAttr>("id") || !d.getAs<IntegerAttr>("rows") ||
       !d.getAs<IntegerAttr>("width") || !d.getAs<StringAttr>("payload"))return llvm::createStringError("incomplete parameter row");
    auto id=d.getAs<IntegerAttr>("id").getInt(), width=d.getAs<IntegerAttr>("width").getInt();
    if(!caps->acceptsGroup(anchor, d.getAs<IntegerAttr>("rows").getInt()) ||
       !seen.insert(id).second || (topology=="resident" && (hi<0 || width<hi)) ||
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
  if(failed(intentdsl::verifyDomain(f)))return failure();
  auto topology=f->getAttrOfType<StringAttr>("block.topology");
  if(auto e=verifyRows(f,f->getAttrOfType<ArrayAttr>("block.configs"),topology,f->getAttrOfType<DenseI64ArrayAttr>("block.guard"),p))
    return f.emitError()<<llvm::toString(std::move(e));
  bool tiles=f->getAttr("block.grid")==block::grid(b,true);
  if((tiles && (p.services.get<RowCapacity>()->independentWidths.empty() || topology!="stream")) ||
     (!tiles && f->getAttr("block.grid")!=block::grid(b,false)))
    return f.emitError("grid must cover the M domain at the instance row step");
  auto domainIndex = f->getAttrOfType<IntegerAttr>("kir.domain_arg");
  if (!domainIndex || domainIndex.getInt()<0 || domainIndex.getInt()>=f.getNumArguments())
    return f.emitError("missing bound row domain argument");
  auto domain=f.getArgument(domainIndex.getInt());
  auto view=dyn_cast<MemRefType>(domain.getType());
  if (!view || view.getRank()<1 || view.getRank()>4) return f.emitError("row domain must be a matrix argument");
  llvm::SmallDenseSet<Value> instances,mapped;
  unsigned sweeps=0;
  auto result=f.walk([&](Operation *op){
    if(op->getName().getDialectNamespace()!=dialect)return WalkResult::advance();
    auto n=op->getName().stripDialect();
    auto def=[&](Value v,StringRef k){auto *d=v.getDefiningOp();return d && d->getName().getDialectNamespace()==dialect && d->getName().stripDialect()==k;};
    bool ok=true;
    if(n=="pid"){instances.insert(op->getResult(0));ok=op->getAttr("axis")==b.getI32IntegerAttr(0) || (tiles && op->getAttr("axis")==b.getI32IntegerAttr(1));}
    if(n=="rows"){mapped.insert(op->getOperand(0));ok=op->getAttr("step")==b.getStringAttr("rows") && def(op->getOperand(0),"pid") && op->getOperand(0).getDefiningOp()->getAttr("axis")==b.getI32IntegerAttr(0);}
    if(n=="tiles"){
      ++sweeps;
      if(!llvm::hasSingleElement(op->getRegion(0)))return WalkResult::interrupt();
      auto &body=op->getRegion(0).front();
      auto count=op->getNumResults();
      ok=count<=1 && op->getNumOperands()==count+1 && body.getNumArguments()==count+1 && !body.empty() &&
         body.getArgument(0).getType().isIndex() && body.back().getName().stripDialect()=="yield" && body.back().getNumOperands()==count;
      if(ok && count){
        auto init=op->getOperand(1),result=body.back().getOperand(0);
        bool tuple=isa<TupleType>(init.getType());
        ok=def(init,tuple?"summary_identity":"identity") && def(result,tuple?"summary_merge":"combine") &&
           result.getDefiningOp()->getOperand(0)==body.getArgument(1) &&
           init.getDefiningOp()->getAttr("algebra")==result.getDefiningOp()->getAttr("algebra") &&
           init.getType()==result.getType() && init.getType()==op->getResult(0).getType();
      }
    }
    if(n=="columns"){
      auto v=op->getOperand(0);
      if(tiles){mapped.insert(v);ok=def(v,"pid") && v.getDefiningOp()->getAttr("axis")==b.getI32IntegerAttr(1);}
      else ok=topology=="resident"?def(v,"zero"):isa<BlockArgument>(v) && cast<BlockArgument>(v).getArgNumber()==0 && v.getParentBlock()->getParentOp()->getName().stripDialect()=="tiles";
    }
    if(n=="active")
      ok=def(op->getOperand(1),"rows") && def(op->getOperand(2),"columns");
    if(n=="scan_sum" && topology!="resident")
      return op->emitError("scan crossing a tile boundary requires a carry"),WalkResult::interrupt();
    if(n=="indexed_tile_read"){
      auto a=cast<MemRefType>(op->getOperand(0).getType()),i=cast<MemRefType>(op->getOperand(1).getType());
      auto mask=op->getOperand(4).getDefiningOp();
      ok=a.getRank()==2 && i.getRank()==1 && i.getElementType().isSignlessIntOrIndex() &&
         def(op->getOperand(4),"active") && mask->getOperand(0)==domain && mask->getOperand(1)==op->getOperand(2) &&
         mask->getOperand(2)==op->getOperand(3) && cast<RankedTensorType>(op->getResult(0).getType()).getElementType()==a.getElementType();
    }
    if(dialect=="block" && (n=="load" || n=="store")){
      auto mask=op->getOperand(3).getDefiningOp();
      if(!mask || !def(op->getOperand(3),"active") || mask->getBlock()!=op->getBlock())return op->emitError("access requires local validity domain"),WalkResult::interrupt();
      auto r=mask->getOperand(1),c=mask->getOperand(2);
      bool reversed=op->getOperand(1)==c && op->getOperand(2)==r;
      ok=(op->getOperand(1)==r && op->getOperand(2)==c) || reversed;
      auto slot=dyn_cast<BlockArgument>(op->getOperand(0));
      if(!slot || slot.getOwner()!=&f.front())return WalkResult::interrupt();
      auto sig=f->getAttrOfType<StringAttr>("kir.abi");if(!sig)return WalkResult::interrupt();
      auto abi=llvm::json::parse(sig.getValue());
      if(!abi){llvm::consumeError(abi.takeError());return WalkResult::interrupt();}
      auto *entries=abi->getAsArray();if(!entries || entries->size()!=f.getNumArguments())return WalkResult::interrupt();
      auto dims=[&](unsigned i){auto *d=(*entries)[i].getAsObject();return d?d->getArray("dims"):nullptr;};
      auto *a=dims(slot.getArgNumber()),*d=dims(domainIndex.getInt());
      if(!a || !d || d->size()!=size_t(view.getRank()))return WalkResult::interrupt();
      if (f->hasAttr("kir.owner_axis"))
        return ok && !reversed && *a==*d ? WalkResult::advance() : WalkResult::interrupt();
      if(auto projection=op->getAttrOfType<DenseI64ArrayAttr>("block.member_axes")){
        if(!ok || reversed || size_t(projection.size())!=a->size())return WalkResult::interrupt();
        llvm::SmallDenseSet<int64_t> seen;
        for(auto [i,axis]:llvm::enumerate(projection.asArrayRef()))
          if(axis<0 || size_t(axis)>=d->size() || !seen.insert(axis).second || (*a)[i]!=(*d)[axis])return WalkResult::interrupt();
        if(n=="store" && seen.size()!=d->size())return op->emitError("noninjective projected write"),WalkResult::interrupt();
        return WalkResult::advance();
      }
      if(d->size()!=2 || (a->size()!=1 && a->size()!=2))return op->emitError("explicit axes required for this member domain"),WalkResult::interrupt();
      if(a->size()==2)ok &= ((*a)[0]==(*d)[reversed?1:0] && (*a)[1]==(*d)[reversed?0:1]);
      else ok &= !reversed && (*a)[0]==(*d)[1];
    }
    if (n == "linear_index" &&
        (op->getOperand(0) != domain || !def(op->getOperand(1), "rows") ||
         !def(op->getOperand(2), "columns")))
      return op->emitError(
                 "member ordinal requires its own domain and coordinates"),
             WalkResult::interrupt();
    if(n=="active" || n=="tiles"){
      if (op->getOperand(0)!=domain) return op->emitError("expected the bound matrix domain"), WalkResult::interrupt();
    }
    if(n=="owner_store" || n=="owner_load"){
      auto axis=f->getAttrOfType<IntegerAttr>("kir.owner_axis");
      auto slot=dyn_cast<BlockArgument>(op->getOperand(0));
      auto val=n=="owner_store"?op->getOperand(2):op->getResult(0);
      auto tile=dyn_cast<RankedTensorType>(val.getType());
      if(!axis || !slot || slot.getOwner()!=&f.front() || !def(op->getOperand(1),"rows") ||
         !tile || tile.getRank()!=1) return WalkResult::interrupt();
      auto source=cast<MemRefType>(slot.getType());
      auto signature=llvm::json::parse(f->getAttrOfType<StringAttr>("kir.abi").getValue());
      if(!signature){llvm::consumeError(signature.takeError());return WalkResult::interrupt();}
      auto array=signature->getAsArray(); if(!array || array->size()!=f.getNumArguments())return WalkResult::interrupt();
      auto param=(*array)[slot.getArgNumber()].getAsObject(), dom=(*array)[domainIndex.getInt()].getAsObject();
      if(!param || !dom || !dom->getArray("dims") || !param->getArray("dims") ||
         axis.getInt()<0 || size_t(axis.getInt())>=dom->getArray("dims")->size() ||
         param->getArray("dims")->size()!=1 || (*param->getArray("dims"))[0]!=(*dom->getArray("dims"))[axis.getInt()] ||
         source.getRank()!=1 || source.getElementType()!=tile.getElementType() ||
         (param->getString("direction")!="InOut" && param->getString("direction")!=(n=="owner_store"?"Out":"In")))
        return op->emitError("invalid row-owner access"),WalkResult::interrupt();
    }
    if(n=="active"){
      auto mask=dyn_cast<RankedTensorType>(op->getResult(0).getType());
      ok &= mask && mask.getRank()==2 && mask.getElementType().isInteger(1);
    }
    if((n=="reduce" || n=="summary_reduce") && op->getNumOperands()==2){
      Value mask=op->getOperand(1);
      auto type=dyn_cast<RankedTensorType>(mask.getType());
      auto data=dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      if(!def(mask,"active") || !type || !data || !type.getElementType().isInteger(1) ||
          type.getShape()!=data.getShape() || mask.getParentBlock()!=op->getBlock())
        return op->emitError("reduction requires its tile's boolean active mask"), WalkResult::interrupt();
    }
    if(tiles && (n=="reduce" || n=="combine" || n=="identity"))return op->emitError("column tile cannot split a row collective"),WalkResult::interrupt();
    if (n.starts_with("summary_")) {
      auto t=dyn_cast<TupleType>(n=="summary_field"?op->getOperand(0).getType():op->getResult(0).getType());
      if (!t || t.size()!=3 || tiles) return WalkResult::interrupt();
      for (unsigned i=0;i<3;++i) {
        auto v=dyn_cast<RankedTensorType>(t.getType(i));
        if (!v || v.getRank()!=1 || (i?!v.getElementType().isF32():!v.getElementType().isInteger(32)))
          return WalkResult::interrupt();
      }
      if (n=="summary_field") {
        auto i=op->getAttrOfType<IntegerAttr>("position").getInt();
        ok=i>=0 && i<3 && t.getType(i)==op->getResult(0).getType();
      } else if (n=="summary_merge")
        ok=op->getOperand(0).getType()==t && op->getOperand(1).getType()==t &&
           def(op->getOperand(1),"summary_reduce");
      else if(n=="summary_reduce") ok=cast<RankedTensorType>(op->getOperand(0).getType()).getElementType().isF32();
    }
    if(n=="reduce" || n=="combine" || n=="identity"){
      auto alg=op->getAttrOfType<StringAttr>("algebra");ok=alg && (alg=="max_num" || alg=="sum");
      if(n=="combine")ok &= def(op->getOperand(1),"reduce") && op->getOperand(1).getDefiningOp()->getAttr("algebra")==alg;
    }
    if(!ok){op->emitError("invalid row coordinates, instance mapping or reduction algebra");return WalkResult::interrupt();}
    return WalkResult::advance();
  });
  if(result.wasInterrupted())return failure();
  if(instances.size()!=(tiles?2u:1u) || instances!=mapped || ((topology=="stream" && !tiles)!=(sweeps!=0)))return f.emitError("instance mapping does not match topology");
  return success();
}
}

mlir::LogicalResult intentdsl::block::verifyRowAccess(mlir::Operation *op){
  auto view=op->getOperand(0);
  auto axes=projection(view.getType());
  if(failed(axes))return op->emitError("expected a column or matrix row view");
  auto store=op->getName().stripDialect()=="store";
  auto *active = op->getOperand(3).getDefiningOp();
  if (!active || active->getName().getStringRef() != "block.active")
    return op->emitError("missing member validity domain");
  auto domainRank =
      mlir::cast<mlir::MemRefType>(active->getOperand(0).getType()).getRank();
  if (store && domainRank != 1 && *axes != RowProjection::Matrix)
    return op->emitError("unique row writes require both axes");
  auto tile=mlir::dyn_cast<mlir::RankedTensorType>(store?op->getOperand(4).getType():op->getResult(0).getType());
  auto mask=op->getOperand(3).getDefiningOp();
  if(!tile || tile.getRank()!=2 || tile.getElementType()!=mlir::cast<mlir::MemRefType>(view.getType()).getElementType() ||
      !mask || mask->getName().getStringRef()!="block.active" || mask->getOperand(1)!=op->getOperand(1) || mask->getOperand(2)!=op->getOperand(2))
    return op->emitError("projected view, fragment or row validity domain differs");
  return mlir::success();
}
