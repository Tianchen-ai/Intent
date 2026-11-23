#include "MemberAccess.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/KeyReduction.h"
#include "intentdsl/levels/algorithm/kir/ir/ProjectedLayout.h"
#include "intentdsl/levels/algorithm/kir/ir/Segment.h"
#include "intentdsl/levels/execution/cpu/ir/IndexedEffects.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "intentdsl/levels/execution/cpu/ir/Pack.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include "Dialect.h.inc"
using namespace mlir;
#include "Lanes.h"
namespace CI_UNIT_NAMESPACE {
void dialect(DialectRegistry &r) {
  r.insert<intentdsl::rowmojo::RowMojoDialect>();
}
namespace {
class Columns {
  OpBuilder b;
  Operation *task;
  Location loc;
  IRMapping reduced, tile;
  Value col;
  int64_t width;
  SmallVector<Operation *> stores, ownerStores;
  Operation *make(StringRef name, ValueRange args, TypeRange types,
                  ArrayRef<NamedAttribute> attrs = {}) {
    OperationState s(loc, ("mojo_row." + name).str());
    s.addOperands(args);
    s.addTypes(types);
    s.addAttributes(attrs);
    return b.create(s);
  }
  Value expression(Value v) {
    if (auto known = reduced.lookupOrNull(v))
      return known;
    if (auto known = tile.lookupOrNull(v))
      return known;
    auto *op = v.getDefiningOp();
    if (!op)
      return v;
    auto n = op->getName().stripDialect();
    Type t = v.getType();
    if (auto ranked = dyn_cast<RankedTensorType>(t))
      t = VectorType::get({width}, ranked.getElementType());
    Value result;
    if(n=="ordered_sum" || n=="segment_length"){
      bool fold=n=="ordered_sum";
      OperationState s(loc,fold?"mojo_row.segment_fold":"mojo_row.segment_size");
      s.addOperands({op->getOperand(0),op->getOperand(1),col});
      s.addTypes(t);
      if(fold){IRMapping mapping;op->getRegion(0).cloneInto(s.addRegion(),mapping);}
      auto *physical=b.create(s);
      if(fold && intentdsl::kir::contiguousFeatureRead(physical))physical->setAttr("feature_vector",b.getUnitAttr());
      result=physical->getResult(0);
    }else if(n=="count_keys" || n=="column_sum"){
      result=make(n=="count_keys"?"key_counts":"sum_columns",{op->getOperand(0),op->getOperand(1),col},t,op->getAttrs())->getResult(0);
    }else if(n=="read_owner")
      result=make("owner_read",op->getOperands(),t)->getResult(0);
    else if(n=="gather_row")result=make("indexed_read",{op->getOperand(0),op->getOperand(1),op->getOperand(2),col},t)->getResult(0);
    else if (n == "ordinal") {
      result = make("apply", {task->getRegion(0).front().getArgument(0), col},
                    t, b.getNamedAttr("kind", b.getStringAttr("ordinal")))
                   ->getResult(0);
    } else if (n == "read")
      result = make("read",
                    {op->getOperand(0),
                     task->getRegion(0).front().getArgument(0), col},
                    t)
                   ->getResult(0);
    else {
      SmallVector<Value> args;
      for (auto value : op->getOperands())
        args.push_back(expression(value));
      result =
          make("apply", args, t, b.getNamedAttr("kind", b.getStringAttr(n)))
              ->getResult(0);
      if(n=="scan_sum"){
        unsigned index=0;
        for(auto &prior:*op->getBlock()){
          if(&prior==op)break;
          index+=prior.getName().stripDialect()=="scan_sum";
        }
        result.getDefiningOp()->setAttr("carry",b.getI64IntegerAttr(index));
      }
    }
    if (auto c = op->getAttr("value"))
      result.getDefiningOp()->setAttr("value", c);
    if(auto pos=op->getAttr("position")) result.getDefiningOp()->setAttr("position",pos);
    tile.map(v, result);
    return result;
  }
  Value sweep(Operation *op, StringRef range, Value prev, int64_t lanes) {
    OpBuilder::InsertionGuard restore(b);
    width = lanes;
    tile.clear();
    auto code=op->getName().stripDialect();
    bool owned=code=="write_owner";
    bool store = code=="write" || owned;
    OperationState s(loc, "mojo_row.sweep");
    if (prev)
      s.addOperands(prev);
    if (!store)
      s.addTypes(op->getResult(0).getType());
    s.addAttribute("width", b.getI64IntegerAttr(width));
    s.addAttribute("range", b.getStringAttr(range));
    s.addAttribute(
        "algebra",
        b.getStringAttr(store ? "store" : op->getName().stripDialect()));
    s.addRegion();
    auto *loop = b.create(s);
    auto &bb = loop->getRegion(0).emplaceBlock();
    col = bb.addArgument(b.getIndexType(), loc);
    b.setInsertionPointToEnd(&bb);
    if (store) {
      auto &writes=owned?ownerStores:stores;
      for (auto *w : writes)
        expression(w->getOperand(2));
      for (auto *w : writes)
        if(owned) make("owner_write",{w->getOperand(0),w->getOperand(1),expression(w->getOperand(2))},{});
        else make("write",
             {w->getOperand(0), w->getOperand(1), col,
              expression(w->getOperand(2))},
             {}, w->getAttrs());
      make("yield", {}, {});
      return {};
    }
    make("yield", expression(op->getOperand(0)), {});
    return loop->getResult(0);
  }

public:
  static bool storeInterval(ArrayRef<Operation *> body, Operation *write) {
    if(body.size()!=2 || body[1]!=write || body[0]->getName().stripDialect()!="read" ||
       write->getName().stripDialect()!="write" || write->getOperand(2)!=body[0]->getResult(0))
      return false;
    auto src=cast<MemRefType>(body[0]->getOperand(0).getType());
    auto dst=cast<MemRefType>(write->getOperand(0).getType());
    return src.getRank()==2 && dst.getRank()==2 && src.getElementType()==dst.getElementType();
  }
  Columns(Operation *op) : b(op), task(op), loc(op->getLoc()) {}
  void run(int64_t lanes) {
    auto &bb = task->getRegion(0).front();
    SmallVector<Operation *> original;
    for (auto &op : bb.without_terminator())
      original.push_back(&op);
    for (auto *op : original)
      if (op->getName().stripDialect() == "write")
        stores.push_back(op);
      else if(op->getName().stripDialect()=="write_owner") ownerStores.push_back(op);
    b.setInsertionPoint(&bb.back());
    auto ordered = original;
    std::stable_partition(ordered.begin(), ordered.end(), [](Operation *op) {
      auto n=op->getName().stripDialect();return n!="write" && n!="write_owner";
    });
    for (auto *op : ordered) {
      auto n = op->getName().stripDialect();
      if (n == "write" && op != stores.front())
        continue;
      if(n=="write_owner") continue;
      if (n != "sum" && n != "max_num" && n!="welford" && n != "write")
        continue;
      if(n=="write" && original.size()==2 && original.front()->getName().stripDialect()=="count_keys" &&
         op->getOperand(2)==original.front()->getResult(0)){sweep(op,"private_bins",{},1);continue;}
      if(n=="write" && original.size()==2 && original.front()->getName().stripDialect()=="gather_row" &&
         op->getOperand(2)==original.front()->getResult(0)){
        sweep(op,"indexed_copy",{},1);continue;
      }
      if (storeInterval(original,op)) {
        sweep(op,"copy",{},1);
        continue;
      }
      auto full = sweep(op, "full", {}, lanes);
      auto done = sweep(op, "tail", full, 1);
      if (done)
        reduced.map(op->getResult(0), done);
    }
    if(!ownerStores.empty())sweep(ownerStores.front(),"owners",{},1);
    for (auto *op : llvm::reverse(original))
      op->erase();
    task->setAttr("mojo_row.schedule", b.getStringAttr("static-row-groups"));
    task->setAttr("mojo_row.lanes", b.getI64IntegerAttr(lanes));
  }
};
struct Realize : PassWrapper<Realize, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Realize)
  int64_t lanes;
  intentdsl::CallFacts facts;
  Realize(int64_t lanes,const intentdsl::CallFacts &f) : lanes(lanes),facts(f) {}
  void runOnOperation() final {
    auto &first=(*getOperation().getOps<func::FuncOp>().begin()).front().front();
    if (first.getName().getStringRef() == "cpu.owner_task") {
      if (failed(memberAccesses(first.getParentOfType<func::FuncOp>(), true))) {
        signalPassFailure();
        return;
      }
      first.setAttr("mojo_row.schedule",
                    StringAttr::get(&getContext(), "ordered-owner-range"));
      return;
    }
    if (auto *mt = intentdsl::cpu::findMatrixTask(
            first.getParentOfType<func::FuncOp>())) {
      auto &region = mt->getRegion(0).front().front();
      mt->setAttr("mojo_row.column_lanes",
                  IntegerAttr::get(
                      IntegerType::get(&getContext(), 64),
                      region.getAttrOfType<DenseI64ArrayAttr>(
                          "cpu.output_tile")[3]));
      mt->setAttr("mojo_row.schedule",
                  StringAttr::get(&getContext(), "static-output-blocks"));
      return;
    }
    for(auto fn:getOperation().getOps<func::FuncOp>())fn.walk([&](Operation *op){
      auto n=op->getName().stripDialect();if(n!="count_keys" && n!="column_sum")return;
      auto extent=intentdsl::kir::keyExtent(op,facts);
      if(!extent){op->emitError()<<llvm::toString(extent.takeError());signalPassFailure();return;}
      op->setAttr("extent",IntegerAttr::get(IntegerType::get(&getContext(),64),*extent));
    });
    if(failed(intentdsl::specializeProjectedLayout(*getOperation().getOps<func::FuncOp>().begin(),facts,"mojo_row",true))){signalPassFailure();return;}
    getContext().getOrLoadDialect<intentdsl::rowmojo::RowMojoDialect>();
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "cpu.task"){
        auto &body=op->getRegion(0).front();
        if(failed(intentdsl::cpu::packMemberLoop(body))){signalPassFailure();return;}
        if(intentdsl::cpu::packed(body) || intentdsl::cpu::formReservations(body) || intentdsl::cpu::formAtomicMembers(body)){
          Builder b(&getContext());op->setAttr("mojo_row.schedule",b.getStringAttr("static-row-groups"));
          op->setAttr("mojo_row.lanes",b.getI64IntegerAttr(laneCount(op,lanes)));
        }else Columns(op).run(laneCount(op,lanes));
      }
    });
  }
};
} // namespace
void realize(OpPassManager &pm, int64_t lanes,const intentdsl::CallFacts &f) {
  pm.addPass(std::make_unique<Realize>(lanes,f));
}
} // namespace CI_UNIT_NAMESPACE
