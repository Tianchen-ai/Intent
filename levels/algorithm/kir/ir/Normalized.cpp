#include "intentdsl/levels/algorithm/kir/ir/Normalized.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
using namespace mlir;
namespace intentdsl::kir {
namespace {
Operation *producer(Value v,StringRef n){
  auto *op=v?v.getDefiningOp():nullptr;
  return op && op->getName().stripDialect()==n?op:nullptr;
}
Value input(Value v,StringRef n,unsigned i=0){
  auto *op=producer(v,n);
  return op && i<op->getNumOperands()?op->getOperand(i):Value();
}
bool literal(Value v,float x){
  auto op=v?v.getDefiningOp<arith::ConstantOp>():arith::ConstantOp();
  auto attr=op?dyn_cast<FloatAttr>(op.getValue()):FloatAttr();
  return attr && attr.getValueAsDouble()==double(x);
}
bool axisIs(Operation *o,unsigned axis){
  auto a=o->getAttrOfType<IntegerAttr>("axis");
  return a && a.getInt()==axis;
}
bool read(Value v, Value buf,Value b,Value h,Value t){
  auto *s=producer(input(v,"extf"),"slice_read");
  return s && axisIs(s,3) && s->getOperands()==ValueRange({buf,b,h,t});
}
bool bound(Value v,Value key){
  auto d=v?v.getDefiningOp<memref::DimOp>():memref::DimOp();
  return d && d.getSource()==key && d.getConstantIndex()==2;
}
bool fullFold(scf::ForOp loop,Value key){
  return loop->hasAttr("kir.reassociate") && matchPattern(loop.getLowerBound(),m_Zero()) &&
         matchPattern(loop.getStep(),m_One()) && bound(loop.getUpperBound(),key);
}
bool dotWithScale(Value score,Value scale,Value q,Value k,Value b,Value h,Value kv,Value row,Value t){
  auto dot=input(input(score,"mulf"),"lane_sum");
  auto gain=input(score,"mulf",1);
  return input(gain,"mulf")==scale && literal(input(gain,"mulf",1),1.4426950408889634f) &&
         read(input(dot,"mulf"),q,b,h,row) && read(input(dot,"mulf",1),k,b,kv,t);
}
scf::IfOp causal(Value value,Value row,Value t,float masked){
  auto branch=value?value.getDefiningOp<scf::IfOp>():scf::IfOp();
  if(!branch || !literal(branch.elseYield().getOperand(0),masked))return {};
  auto cond=branch.getCondition().getDefiningOp<arith::CmpIOp>();
  return cond && cond.getPredicate()==arith::CmpIPredicate::sge && cond.getLhs()==row && cond.getRhs()==t ? branch : scf::IfOp();
}
} // namespace
bool scoreShape(ScorePanel &s,const CallFacts &f) {
  auto dims=[&](Value v){SmallVector<int64_t> ds;if(auto *arr=f.shapeOf(v))
    for(auto &n:*arr)ds.push_back(n.getAsInteger().value_or(-1));return ds;};
  auto q=dims(s.q),k=dims(s.k),v=dims(s.v),y=dims(s.y);
  for(auto view:{s.q,s.k,s.v,s.y})
    if(!isa<BlockArgument>(view) || !isa<MemRefType>(view.getType()) ||
       !cast<MemRefType>(view.getType()).getElementType().isBF16())return false;
  if(q.size()!=4 || k.size()!=4 || v.size()!=4 || y.size()!=4 || s.group<=0 ||
     q[1]%s.group || q[1]/s.group!=k[1] || q[0]!=k[0] || q[3]!=k[3] ||
     ArrayRef(k).take_front(3)!=ArrayRef(v).take_front(3) ||
     ArrayRef(q).take_front(3)!=ArrayRef(y).take_front(3) || y[3]!=v[3])return false;
  s.queries=q[2];s.keys=k[2];s.depth=q[3];s.features=v[3];
  return s.keys>0 && s.depth>0 && s.features>0;
}
std::optional<ScorePanel> normalizedContractions(Operation *o,const CallFacts &facts){
  auto fn=o->getParentOfType<func::FuncOp>();auto &bb=o->getRegion(0).front();
  if(fn.getNumArguments()!=5 || bb.getNumArguments()!=3)return {};
  ScorePanel s{fn.getArgument(0),fn.getArgument(1),fn.getArgument(2),fn.getArgument(3),fn.getArgument(4),0,0,0,0,0};
  if(o->getOperands()!=ValueRange({s.q,s.q,s.q}) || o->getAttrOfType<DenseI64ArrayAttr>("axes").asArrayRef()!=ArrayRef<int64_t>({0,1,2}))return {};
  auto b=bb.getArgument(0),h=bb.getArgument(1),row=bb.getArgument(2);
  auto loops=llvm::to_vector(bb.getOps<scf::ForOp>());
  if(loops.size()!=2 || !fullFold(loops[0],s.k) || !fullFold(loops[1],s.k))return {};
  auto peak=loops[0],sum=loops[1];
  if(peak.getNumResults()!=1 || sum.getNumResults()!=2 || !literal(peak.getInitArgs()[0],-INFINITY) ||
     !literal(sum.getInitArgs()[0],0) || !literal(input(sum.getInitArgs()[1],"lane_splat"),0))return {};
  auto initial=sum.getInitArgs()[1].getDefiningOp();
  if(initial->getOperand(1)!=s.v || !axisIs(initial,3))return {};
  auto max=peak.getBody()->getTerminator()->getOperand(0);
  if(input(max,"maxnumf")!=peak.getRegionIterArgs()[0])return {};
  auto top=causal(input(max,"maxnumf",1),row,peak.getInductionVar(),-INFINITY);
  if(!top)return {};
  auto score=top.thenYield().getOperand(0);
  auto prod=input(input(score,"mulf"),"lane_sum");
  auto *slice=producer(input(input(prod,"mulf",1),"extf"),"slice_read");
  if(!slice || slice->getNumOperands()!=4)return {};
  auto kv=slice->getOperand(2);
  auto group=input(kv,"floordivsi",1);
  auto factor=group?group.getDefiningOp<arith::ConstantIndexOp>():arith::ConstantIndexOp();
  if(!factor || input(kv,"floordivsi")!=h)return {};
  s.group=factor.value();
  if(!scoreShape(s,facts) || !dotWithScale(score,s.scale,s.q,s.k,b,h,kv,row,peak.getInductionVar()))return {};
  auto ys=sum.getBody()->getTerminator()->getOperands();
  if(input(ys[0],"addf")!=sum.getRegionIterArgs()[0] || input(ys[1],"addf")!=sum.getRegionIterArgs()[1])return {};
  auto w=input(ys[0],"addf",1);
  auto branch=causal(w,row,sum.getInductionVar(),0);
  if(!branch)return {};
  auto diff=input(branch.thenYield().getOperand(0),"exp2");
  if(input(diff,"subf",1)!=peak.getResult(0) ||
     !dotWithScale(input(diff,"subf"),s.scale,s.q,s.k,b,h,kv,row,sum.getInductionVar()))return {};
  auto weighted=input(ys[1],"addf",1);
  auto round=input(input(input(weighted,"mulf",1),"lane_splat"),"extf");
  if(!read(input(weighted,"mulf"),s.v,b,kv,sum.getInductionVar()) ||
     !round || !round.getType().isBF16() || input(round,"truncf")!=w)return {};
  Operation *store=nullptr;
  auto effects=o->getRegion(0).walk([&](Operation *op){
    if(op->getName().stripDialect()=="slice_write"){
      if(store || op->getBlock()!=&bb)return WalkResult::interrupt();
      store=op;
    }else if(op->getName().stripDialect()!="slice_read" && !isa<scf::ForOp,scf::IfOp>(op) &&
             !isMemoryEffectFree(op) && op->getName().stripDialect()!="end")return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if(effects.wasInterrupted() || !store || !axisIs(store,3) || store->getOperands().drop_front()!=ValueRange({s.y,b,h,row}))return {};
  auto output=input(store->getOperand(0),"truncf");
  auto inv=input(input(output,"mulf",1),"lane_splat");
  auto safe=inv?inv.getDefiningOp<scf::IfOp>():scf::IfOp();
  if(!safe || input(output,"mulf")!=sum.getResult(1) || !literal(safe.thenYield().getOperand(0),0))return {};
  auto empty=safe.getCondition().getDefiningOp<arith::CmpIOp>();
  auto reciprocal=safe.elseYield().getOperand(0);
  if(!empty || empty.getPredicate()!=arith::CmpIPredicate::eq || !bound(empty.getLhs(),s.k) ||
     !matchPattern(empty.getRhs(),m_Zero()) || !literal(input(reciprocal,"divf"),1) ||
     input(reciprocal,"divf",1)!=sum.getResult(0))return {};
  s.maxScore=score;
  s.weightScore=input(diff,"subf");
  return s;
}

} // namespace intentdsl::kir
