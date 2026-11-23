#include "MemberAccess.h"
#include "intentdsl/levels/algorithm/kir/ir/ProjectedLayout.h"
#include "intentdsl/levels/algorithm/kir/ir/RowSemantics.h"
#include "intentdsl/levels/algorithm/kir/ir/Segment.h"
#include "intentdsl/levels/execution/cpu/ir/IndexedEffects.h"
#include "intentdsl/levels/execution/cpu/ir/Pack.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"
using namespace compilerinfra;
using namespace mlir;
#include "Lanes.h"
namespace CI_UNIT_NAMESPACE {
namespace {
struct Publish : PassWrapper<Publish, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Publish)
  void runOnOperation() final {
    auto fn = *getOperation().getOps<func::FuncOp>().begin();
    fn->setAttr("mojo_row.entry",
                StringAttr::get(&getContext(), "intent_entry"));
  }
};
} // namespace
void publishABI(OpPassManager &pm) { pm.addPass(std::make_unique<Publish>()); }
LogicalResult verifyLoops(ModuleOp m, const TargetProfile &p, bool callable) {
  if (failed(intentdsl::cpu::verifyTask(m, p, "private-values", "cpu.task",
                                        false)))
    return failure();
  auto fn = *m.getOps<func::FuncOp>().begin();
  if (fn.front().front().getName().getStringRef() == "cpu.owner_task")
    return success(fn.front().front().getAttrOfType<StringAttr>(
                       "mojo_row.schedule") == "ordered-owner-range" &&
                   succeeded(memberAccesses(fn, false)));
  if(fn.front().front().getName().getStringRef()=="cpu.matrix_task"){
    auto &task=fn.front().front();
    auto lanes=task.getAttrOfType<IntegerAttr>("mojo_row.column_lanes");
    if(!lanes || lanes.getInt()!=task.getRegion(0).front().front().getAttrOfType<DenseI64ArrayAttr>("cpu.output_tile")[3] ||
       task.getAttrOfType<StringAttr>("mojo_row.schedule")!="static-output-blocks" ||
       (callable && fn->getAttrOfType<StringAttr>("mojo_row.entry")!="intent_entry"))
      return task.emitError("invalid Mojo matrix vector or host mapping");
    return success();
  }
  if(failed(intentdsl::specializeProjectedLayout(fn,*p.services.get<intentdsl::CallFacts>(),"mojo_row",false)))return failure();
  auto &task = fn.front().front();
  if (task.getAttrOfType<StringAttr>("mojo_row.schedule") !=
      "static-row-groups")
    return task.emitError("unbound synchronous Mojo row schedule");
  auto lanes = task.getAttrOfType<IntegerAttr>("mojo_row.lanes");
  if (!lanes ||
      lanes.getInt() !=
          laneCount(&task,p.services.get<intentdsl::cpu::RowMachine>()->vectorBits))
    return task.emitError(
        "sweep width differs from the effective vector target");
  if (callable &&
      fn->getAttrOfType<StringAttr>("mojo_row.entry") != "intent_entry")
    return fn.emitError("missing synchronous host entry");
  Operation *full = nullptr;
  if(intentdsl::cpu::packed(task.getRegion(0).front()))return intentdsl::cpu::verifyPack(task.getRegion(0).front());
  if(intentdsl::cpu::reservations(task.getRegion(0).front()))return intentdsl::cpu::verifyReservations(task.getRegion(0).front(),*p.services.get<intentdsl::CallFacts>());
  if(intentdsl::cpu::atomicMembers(task.getRegion(0).front())){
    auto &body=task.getRegion(0).front();
    return intentdsl::verifyRowComputation(fn,body.front().getRegion(0).front(),body.getArguments().drop_front(),
           task.getAttrOfType<IntegerAttr>("domain_arg").getInt(),"cpu",true);
  }
  for (auto &s : task.getRegion(0).front().without_terminator()) {
    if (s.getName().getStringRef() != "mojo_row.sweep")
      return s.emitError("expected a realized column sweep");
    auto alg = s.getAttrOfType<StringAttr>("algebra");
    bool store = alg == "store";
    auto w = s.getAttrOfType<IntegerAttr>("width").getInt();
    bool tail = full != nullptr;
    bool indexed = s.getAttrOfType<StringAttr>("range")=="indexed_copy";
    bool key=s.getAttrOfType<StringAttr>("range")=="private_bins";
    bool copy = s.getAttrOfType<StringAttr>("range")=="copy" || indexed || key;
    bool owners = s.getAttrOfType<StringAttr>("range")=="owners";
    if(owners && (!store || full || w!=1 || !fn->hasAttr("kir.owner_axis")))return s.emitError("invalid owner publication");
    if(copy && (!store || !llvm::hasSingleElement(task.getRegion(0).front().without_terminator())))
      return s.emitError("a linear copy must cover the whole task");
    if ((!store && alg != "max_num" && alg != "sum" && alg!="welford") ||
        (!copy && !owners && s.getAttrOfType<StringAttr>("range") != (tail ? "tail" : "full")) ||
        w != (owners || copy || tail ? 1 : lanes.getInt()) ||
        s.getNumResults() != unsigned(!store) ||
        s.getNumOperands() != unsigned(tail && !store) ||
        (!store && !(alg=="welford"?isa<TupleType>(s.getResult(0).getType()):s.getResult(0).getType().isF32())) ||
        (tail && (alg != full->getAttr("algebra") ||
                  (!store && s.getOperand(0) != full->getResult(0)))))
      return s.emitError("invalid full sweep or scalar tail continuation");
    auto &bb = s.getRegion(0).front();
    if (bb.getNumArguments() != 1 || !bb.getArgument(0).getType().isIndex() ||
        bb.empty() || bb.back().getName().getStringRef() != "mojo_row.yield" ||
        bb.back().getNumOperands() != unsigned(!store))
      return s.emitError("invalid column induction or sweep yield");
    if (!store && bb.back().getOperand(0).getType() !=
                      VectorType::get({w}, Float32Type::get(m.getContext())))
      return s.emitError("expected f32 reduction lanes");
    unsigned writes = 0;
    for (auto &op : bb.without_terminator()) {
      auto code = op.getName().getStringRef();
      if(code=="mojo_row.key_counts" || code=="mojo_row.sum_columns"){
        if(op.getOperand(1)!=task.getRegion(0).front().getArgument(0) || op.getOperand(2)!=bb.getArgument(0) ||
            op.getResult(0).getType()!=VectorType::get({w},IntegerType::get(m.getContext(),32)) ||
            op.getAttrOfType<IntegerAttr>("extent").getInt()<0 || (code=="mojo_row.key_counts" && !key))
          return op.emitError("key reduction must preserve its output tile");
        if(key && (op.getNextNode()==nullptr || op.getNextNode()->getName().stripDialect()!="write" ||
                   op.getNextNode()->getOperand(3)!=op.getResult(0)))return op.emitError("bins require private output storage");
      }else if(code=="mojo_row.owner_read" || code=="mojo_row.owner_write"){
        bool write=code=="mojo_row.owner_write";
        auto view=cast<MemRefType>(op.getOperand(0).getType());
        auto val=write?op.getOperand(2):op.getResult(0);
        if((write && !owners) || !fn->hasAttr("kir.owner_axis") || view.getRank()!=1 ||
           view.getElementType()!=val.getType() || op.getOperand(1)!=task.getRegion(0).front().getArgument(0))
          return op.emitError("invalid task-owned scalar view");
        writes+=write;
      } else if(code=="mojo_row.segment_fold" || code=="mojo_row.segment_size"){
        if(code=="mojo_row.segment_fold" && op.hasAttr("feature_vector")!=bool(intentdsl::kir::contiguousFeatureRead(&op)))
          return op.emitError("feature SIMD must consume contiguous feature coordinates");
        auto view=cast<MemRefType>(op.getOperand(0).getType());
        if(view.getRank()!=1 || !view.getElementType().isInteger(32) ||
           op.getOperand(1)!=task.getRegion(0).front().getArgument(0) || op.getOperand(2)!=bb.getArgument(0) ||
           op.getResult(0).getType()!=VectorType::get({w},Float32Type::get(m.getContext())))
          return op.emitError("invalid segment endpoints or SIMD output ownership");
      } else if(code=="mojo_row.indexed_read"){
        auto table=cast<MemRefType>(op.getOperand(0).getType()),ids=cast<MemRefType>(op.getOperand(1).getType());
        if(table.getRank()!=2 || ids.getRank()!=1 || !ids.getElementType().isSignlessIntOrIndex() ||
           op.getOperand(2)!=task.getRegion(0).front().getArgument(0) || op.getOperand(3)!=bb.getArgument(0) ||
           op.getResult(0).getType()!=VectorType::get({w},table.getElementType()))return op.emitError("invalid indexed SIMD read");
      } else if (code == "mojo_row.read" || code == "mojo_row.write") {
        bool write = code == "mojo_row.write";
        writes += write;
        auto view = cast<MemRefType>(op.getOperand(0).getType());
        if ((view.getRank()<1 || view.getRank()>4) ||
            op.getOperand(1) != task.getRegion(0).front().getArgument(0) ||
            op.getOperand(2) != bb.getArgument(0) ||
            (write ? op.getOperand(3) : op.getResult(0)).getType() !=
                VectorType::get({w}, view.getElementType()))
          return op.emitError("invalid strided lane access");
      } else if (code == "mojo_row.apply") {
        auto k = op.getAttrOfType<StringAttr>("kind").getValue();
        auto nargs = llvm::StringSwitch<unsigned>(k)
                         .Cases("add", "sub", "mul", "div", 2)
                         .Cases("exp", "rsqrt", "cast", 1)
                         .Cases("and", "xor", "shl", "shr", "gt", 2)
                         .Cases("uniform", "ordinal", 2)
                         .Case("select", 3)
                         .Case("sqrt", 1)
                         .Case("maximum", 2)
                         .Cases("field","scan_sum", 1).Case("constant", 0)
                         .Default(-1);
        if (k == "uniform" && op.getNumOperands() == 2) {
          auto position = dyn_cast<VectorType>(op.getOperand(1).getType());
          auto samples = dyn_cast<VectorType>(op.getResult(0).getType());
          if (!op.getOperand(0).getType().isUnsignedInteger(64) || !position ||
              !samples || !position.getElementType().isUnsignedInteger(64) ||
              !samples.getElementType().isF32() ||
              position.getShape() != samples.getShape())
            return op.emitError(
                "invalid Philox seed, member counters or sample lanes");
        }
        if(k=="scan_sum" && (!op.getAttrOfType<IntegerAttr>("carry") ||
            op.getAttrOfType<IntegerAttr>("carry").getInt()<0 || op.getOperand(0).getType()!=op.getResult(0).getType()))
          return op.emitError("scan requires a typed interval carry");
        if (op.getNumOperands() != nargs ||
            (k == "constant" && !op.getAttrOfType<TypedAttr>("value")))
          return op.emitError("unknown or ill-formed lane expression");
      } else
        return op.emitError("unexpected sweep instruction");
    }
    if (store ? !writes : bool(writes))
      return s.emitError("sweep write count differs from its algebra");
    if(copy){
      auto &read=bb.front();
      auto *write=read.getNextNode();
      if(bb.getOperations().size()!=3 || read.getName().getStringRef()!=(key?"mojo_row.key_counts":indexed?"mojo_row.indexed_read":"mojo_row.read") ||
         !write || write->getName().getStringRef()!="mojo_row.write" ||
         write->getOperand(3)!=read.getResult(0) ||
         cast<MemRefType>(read.getOperand(0).getType()).getRank()!=(key?1:2) ||
         cast<MemRefType>(write->getOperand(0).getType()).getRank()!=2)
        return s.emitError("expected a typed matrix read followed by its matching write");
    }
    full = tail || copy || owners ? nullptr : &s;
  }
  if (full)
    return full->emitError("missing matching tail sweep");
  return success();
}
} // namespace CI_UNIT_NAMESPACE
