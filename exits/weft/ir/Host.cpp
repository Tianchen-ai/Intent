#include "../legalize/Matrix.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/algorithm/kir/ir/RowSemantics.h"
#include "intentdsl/levels/execution/cpu/ir/IndexedEffects.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "intentdsl/levels/execution/cpu/ir/Owners.h"
#include "intentdsl/levels/execution/cpu/ir/Pack.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"

#include "Dialect.h.inc"
#include "Dialect.cpp.inc"
#define GET_OP_CLASSES
#include "Ops.h.inc"
#define GET_OP_CLASSES
#include "Ops.cpp.inc"

using namespace mlir;
void intentdsl::wefthost::WeftHostDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
}
namespace CI_UNIT_NAMESPACE {
void buildPanels(Operation *);
LogicalResult verifyPanels(Operation *);
void dialect(DialectRegistry &r) {
  r.insert<intentdsl::wefthost::WeftHostDialect>();
}
namespace {
bool nativeRecords(func::FuncOp f){
  auto &domain=f.front().front().getRegion(0).front().front();auto &dot=intentdsl::cpu::calculation(domain);
  if(dot.getName().stripDialect()!="q4k_dot")return false;
  auto slot=cast<BlockArgument>(dot.getOperand(0)).getArgNumber();auto s=f.getArgAttrOfType<DenseI64ArrayAttr>(slot,"cpu.strides");
  auto shape=f.getArgAttrOfType<DenseI64ArrayAttr>(slot,"cpu.shape");
  return s[2]==1 && s[1]==144 && s[0]==144*shape[1];
}
int64_t registerRows(func::FuncOp fn,bool rvv){
  auto &domain=fn.front().front().getRegion(0).front().front();auto &op=intentdsl::cpu::calculation(domain);
  auto tile=domain.getAttrOfType<DenseI64ArrayAttr>("cpu.output_tile"),dims=domain.getAttrOfType<DenseI64ArrayAttr>("cpu.extents");
  if (rvv || op.getName().stripDialect() == "q4k_dot" ||
      (op.getName().stripDialect() == "contract" && op.getNumOperands() > 2) ||
      dims[0] % tile[0] || (dims[1] > tile[1] && dims[1] % tile[1]))
    return 1;
  return std::min<int64_t>(tile[0],4);
}
struct Lower : PassWrapper<Lower, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Lower)
  bool callable;
  intentdsl::CallFacts facts;
  bool grouped;
  Lower(bool callable,const intentdsl::CallFacts &f,bool grouped) : callable(callable),facts(f),grouped(grouped) {}
  void runOnOperation() final {
    auto f = *getOperation().getOps<func::FuncOp>().begin();
    auto &task = f.front().front();
    OpBuilder b(&task);
    if (task.getName().getStringRef() == "cpu.owner_task") {
      task.setAttr("weft_host.schedule",
                   b.getStringAttr("ordered-owner-range"));
      if (callable)
        task.setAttr("task_symbol", b.getStringAttr("owner_task"));
      return;
    }
    if(task.getName().getStringRef()=="cpu.matrix_task"){
      bool empty=task.getRegion(0).front().front().getAttrOfType<DenseI64ArrayAttr>("cpu.extents")[2]==0;
      if(empty)task.setAttr("weft_host.empty_reduction",b.getUnitAttr());
      if(!empty && grouped && nativeRecords(f))task.setAttr("weft_host.record_fields",b.getUnitAttr());
      task.setAttr("weft_host.register_rows",b.getI64IntegerAttr(registerRows(f,grouped)));
      task.setAttr("weft_host.schedule",b.getStringAttr("single-instance-output-panel"));
      if(callable)task.setAttr("task_symbol",b.getStringAttr("matrix_panel"));
      return;
    }
    if (callable) {
      SmallVector<int64_t> ranks;
      for (Type t : task.getOperandTypes()) {
        auto view = dyn_cast<MemRefType>(t);
        ranks.push_back(view ? view.getRank() : 0);
      }
      bool absolute = false;
      task.walk([&](Operation *o) {
        absolute |= o->getName().getStringRef() == "cpu.ordinal";
        auto n=o->getName().stripDialect();
        if(n=="write_offsets" || n=="fill_ranges"){
          for(auto v:o->getOperands().take_front(n=="write_offsets"?2:1)){
            auto i=cast<BlockArgument>(v).getArgNumber()-1;
            auto *shape=facts.arguments[i].getAsObject()->getArray("shape");
            f.setArgAttr(i,"weft_host.indirect_extents",b.getDenseI64ArrayAttr({*shape->front().getAsInteger()}));
          }
        }
        if(n=="reserve_tickets" || n=="distribute_tickets"){
          SmallVector<Value> shared;
          if(n=="reserve_tickets")shared.push_back(o->getOperand(0));
          else {shared.push_back(o->getOperand(1));shared.push_back(o->getOperand(3));absolute=true;}
          for(auto v:shared){
            auto i=cast<BlockArgument>(v).getArgNumber()-1;
            auto *shape=facts.arguments[i].getAsObject()->getArray("shape");
            f.setArgAttr(i,"weft_host.indirect_extents",b.getDenseI64ArrayAttr({*shape->front().getAsInteger()}));
          }
        }
        if(n=="pack_members" || n=="prefix_scatter"){
          auto slot=cast<BlockArgument>(o->getOperand(n=="pack_members"?2:1)).getArgNumber()-1;
          f.setArgAttr(slot,"weft_host.task_indices",b.getUnitAttr());
        }
        if(n=="ordered_sum" || n=="segment_length" || n=="member_load" || n=="count_keys" || n=="column_sum" || n=="atomic_add" || n=="unique_put" || n=="index_read"){
          auto argument=dyn_cast<BlockArgument>(o->getOperand(0));
          if(!argument || argument.getOwner()!=&task.getRegion(0).front() || !argument.getArgNumber()){
            o->emitError("segment read requires a captured input");signalPassFailure();return;
          }
          auto slot=argument.getArgNumber()-1;
          auto *record=facts.arguments[slot].getAsObject();auto *shape=record?record->getArray("shape"):nullptr;
          SmallVector<int64_t> dims;
          if(shape)for(auto &d:*shape)if(auto extent=d.getAsInteger())dims.push_back(*extent);
          if(dims.size()!=cast<MemRefType>(argument.getType()).getRank()){o->emitError("missing indexed input extents");signalPassFailure();return;}
          f.setArgAttr(slot,"weft_host.indirect_extents",b.getDenseI64ArrayAttr(dims));
          absolute|=n=="ordered_sum" || n=="segment_length" || n=="member_load" || n=="count_keys";
        }
        if(o->getName().getStringRef()=="cpu.gather_row"){
          auto table=cast<BlockArgument>(o->getOperand(0)).getArgNumber()-1;
          auto ids=cast<BlockArgument>(o->getOperand(1)).getArgNumber()-1;
          auto *slot=facts.arguments[table].getAsObject();auto *shape=slot?slot->getArray("shape"):nullptr;
          if(!shape || shape->size()!=2 || !shape->front().getAsInteger()){o->emitError("missing indexed table height");signalPassFailure();return;}
          f.setArgAttr(table,"weft_host.table_rows",b.getI64IntegerAttr(*shape->front().getAsInteger()));
          f.setArgAttr(ids,"weft_host.task_indices",b.getUnitAttr());
        }
      });
      task.setAttr("weft_host.absolute_origin", b.getBoolAttr(absolute));
      if(task.walk([](Operation *v){return v->getName().stripDialect()=="ordered_sum"?WalkResult::interrupt():WalkResult::advance();}).wasInterrupted()){
        for(unsigned i=0;i<f.getNumArguments();++i){
          if(!isa<MemRefType>(f.getArgument(i).getType()))continue;
          auto *a=facts.arguments[i].getAsObject();
          auto *shape=a->getArray("shape"),*strides=a->getArray("strides");
          bool dense=shape && strides && shape->size()==strides->size();int64_t step=1;
          if(dense)for(int k=shape->size()-1;k>=0;--k){
            auto n=(*shape)[k].getAsInteger(),s=(*strides)[k].getAsInteger();
            if(!n || !s || *n<0 || (*n>1 && *s!=step) || (*n && step>INT64_MAX / *n)){dense=false;break;}
            step*=*n;
          }
          if(dense)f.setArgAttr(i,"weft_host.contiguous",b.getUnitAttr());
        }
      }
      task.setAttr("weft_host.slot_ranks", b.getDenseI64ArrayAttr(ranks));
      return;
    }
    getContext().getOrLoadDialect<intentdsl::wefthost::WeftHostDialect>();
    OperationState s(task.getLoc(), "weft_host.dispatch_rows");
    s.addOperands(task.getOperands());
    s.addTypes(task.getResultTypes());
    s.addAttributes(task.getAttrs());
    s.addAttribute("task_symbol", b.getStringAttr("row_task"));
    s.addRegion()->takeBody(task.getRegion(0));
    auto *dispatch = b.create(s);
    task.replaceAllUsesWith(dispatch);
    auto &body=dispatch->getRegion(0).front();
    if(failed(intentdsl::cpu::packMemberLoop(body))){signalPassFailure();return;}
    bool moments=llvm::any_of(body,[](Operation &op){return op.getName().stripDialect()=="welford";});
    if ((moments || f->hasAttr("kir.owner_axis") || cast<DictionaryAttr>(dispatch->getAttrOfType<ArrayAttr>("rows")[0])
            .getAs<IntegerAttr>("columns")
                     .getInt() != 0) && !intentdsl::cpu::packed(body) && !intentdsl::cpu::formReservations(body) && !intentdsl::cpu::formAtomicMembers(body))
      buildPanels(dispatch);
    task.erase();
  }
};
} // namespace
void lower(OpPassManager &pm, bool callable,const intentdsl::CallFacts &facts,bool grouped) {
  pm.addPass(std::make_unique<Lower>(callable,facts,grouped));
}
LogicalResult verifyHost(ModuleOp m, const compilerinfra::TargetProfile &p,
                         bool callable) {
  if (failed(intentdsl::cpu::verifyTask(m, p, "weft-task-ssa",
                                        "weft_host.dispatch_rows", false)))
    return failure();
  auto &task = (*m.getOps<func::FuncOp>().begin()).front().front();
  if (task.getName().getStringRef() == "cpu.owner_task") {
    auto fn = *m.getOps<func::FuncOp>().begin();
    if (failed(intentdsl::cpu::verifyOwnerTask(fn, p, "weft-task-ssa")))
      return failure();
    return success(task.getAttrOfType<StringAttr>("weft_host.schedule") ==
                       "ordered-owner-range" &&
                   (!callable || task.getAttrOfType<StringAttr>(
                                     "task_symbol") == "owner_task"));
  }
  if(task.getName().getStringRef()=="cpu.matrix_task"){
    bool empty=task.getRegion(0).front().front().getAttrOfType<DenseI64ArrayAttr>("cpu.extents")[2]==0;
    if(empty!=task.hasAttr("weft_host.empty_reduction"))return task.emitError("zero reduction differs from bound K extent");
    if((!empty && p.services.get<RecordFields>()->grouped && nativeRecords(*m.getOps<func::FuncOp>().begin()))!=task.hasAttr("weft_host.record_fields"))
      return task.emitError("record contraction requires packed group and row strides");
    auto width=task.getAttrOfType<IntegerAttr>("weft_host.register_rows");
    if(!width || width.getInt()!=registerRows(*m.getOps<func::FuncOp>().begin(),p.services.get<RecordFields>()->grouped))
      return task.emitError("native accumulator rows exceed the task extent");
    if(task.getAttrOfType<StringAttr>("weft_host.schedule")!="single-instance-output-panel" ||
       (callable && task.getAttrOfType<StringAttr>("task_symbol")!="matrix_panel"))
      return task.emitError("incomplete native matrix task binding");
    return success();
  }
  bool whole = cast<DictionaryAttr>(task.getAttrOfType<ArrayAttr>("rows")[0])
                   .getAs<IntegerAttr>("columns")
                   .getInt() == 0;
  auto scalar=task.hasAttr("weft_host.scalar_members");
  bool momentSweep=llvm::any_of(task.getRegion(0).front(),[](Operation &op){
    return op.getName().getStringRef()=="weft_host.sweep" &&
           op.getAttrOfType<StringAttr>("combine")=="welford";
  });
  if(scalar!=(task.getParentOfType<func::FuncOp>()->hasAttr("kir.owner_axis") || momentSweep))
    return task.emitError("member traversal must match the projection or tuple reduction");
  if (whole && !scalar && failed(intentdsl::cpu::verifyTask(
                   m, p, "weft-task-ssa", "weft_host.dispatch_rows", true)))
    return failure();
  auto &body=task.getRegion(0).front();
  if(intentdsl::cpu::reservations(body) && failed(intentdsl::cpu::verifyReservations(body,*p.services.get<intentdsl::CallFacts>())))return failure();
  if(intentdsl::cpu::atomicMembers(body) && failed(intentdsl::verifyRowComputation(*m.getOps<func::FuncOp>().begin(),
     body.front().getRegion(0).front(),body.getArguments().drop_front(),task.getAttrOfType<IntegerAttr>("domain_arg").getInt(),"cpu",true)))return failure();
  if ((!whole || scalar) && !intentdsl::cpu::reservations(body) && !intentdsl::cpu::packed(body) && !intentdsl::cpu::atomicMembers(body) && failed(verifyPanels(&task)))
    return failure();
  if(intentdsl::cpu::packed(task.getRegion(0).front()) && failed(intentdsl::cpu::verifyPack(task.getRegion(0).front())))return failure();
  if (task.getAttrOfType<StringAttr>("task_symbol") != "row_task")
    return task.emitError("unbound native row task");
  if (callable) {
    auto slots = task.getAttrOfType<DenseI64ArrayAttr>("weft_host.slot_ranks");
    bool absolute = false;
    task.walk([&](Operation *o) {
      auto n=o->getName().stripDialect();
      absolute |= n=="ordinal" || n=="ordered_sum" || n=="segment_length" || n=="count_keys" || n=="distribute_tickets";
    });
    if (task.getAttrOfType<BoolAttr>("weft_host.absolute_origin") !=
        BoolAttr::get(m.getContext(), absolute))
      return task.emitError(
          "task absolute index requires its logical interval origin");
    if (!slots || slots.size() != int64_t(task.getNumOperands()))
      return task.emitError("incomplete bridge slots");
    for (auto [rank, t] :
         llvm::zip(slots.asArrayRef(), task.getOperandTypes())) {
      auto view = dyn_cast<MemRefType>(t);
      if (rank != (view ? view.getRank() : 0))
        return task.emitError("task bridge rank differs from capture");
    }
  }
  return success();
}
} // namespace CI_UNIT_NAMESPACE
