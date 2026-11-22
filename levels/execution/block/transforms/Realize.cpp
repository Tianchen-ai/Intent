#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/KeyReduction.h"
#include "intentdsl/levels/execution/block/ir/Grid.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
namespace {
class RowTile {
public:
  RowTile(Operation *anchor, DictionaryAttr choice, RewriterBase &rw)
      : anchor(anchor), rw(rw), loc(anchor->getLoc()),
        domain(anchor->getOperand(0)),
        resident(choice.getAs<StringAttr>("topology") == "resident") {
    auto fn = anchor->getParentOfType<func::FuncOp>();
    if((choice.get("walk") && choice.getAs<StringAttr>("walk")=="destination")){
      auto json=llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi").getValue());
      fn->setAttr("kir.public_abi",fn->getAttr("kir.abi"));
      fn->setAttr("kir.iteration",rw.getDenseI64ArrayAttr({0,1,1,0,0,0,1,0}));
      auto i=rw.getAffineDimExpr(0),j=rw.getAffineDimExpr(1);
      for(unsigned k=0;k<2;++k){
        (*(*json->getAsArray())[k].getAsObject())["dims"]=llvm::json::Array{"outer","inner"};
        fn.setArgAttr(k,"kir.source",rw.getI64IntegerAttr(k));
        fn.setArgAttr(k,"kir.indexing",AffineMapAttr::get(AffineMap::get(2,0,k?ArrayRef<AffineExpr>({i,j}):ArrayRef<AffineExpr>({j,i}),rw.getContext())));
      }
      std::string abi;llvm::raw_string_ostream(abi)<<*json;
      fn->setAttr("kir.abi",rw.getStringAttr(abi));
      anchor->walk([](Operation *op){if(op->getName().stripDialect()=="write")op->removeAttr("axes");});
    }
    fn->setAttr("block.guard",choice.get("guard"));
    auto walk = choice.getAs<StringAttr>("walk");
    gridTiles = walk && (walk=="grid-tile" || walk=="segment-tile" || walk=="column-panels");
    fn->setAttr("block.grid", intentdsl::block::grid(rw,gridTiles));
    fn->setAttr("block.configs", choice.get("rows"));
    fn->setAttr("block.topology", choice.get("topology"));
    batch=choice.getAs<IntegerAttr>("member_batch");
    binMode=choice.getAs<StringAttr>("bins");
  }
  void build() {
    rw.setInsertionPoint(anchor);
    auto pid = make("pid", {}, rw.getIndexType(), rw.getNamedAttr("axis",rw.getI32IntegerAttr(0)))->getResult(0);
    rows = make("rows", pid, vector(rw.getIndexType(), 1), rw.getNamedAttr("step",rw.getStringAttr("rows")))->getResult(0);
    auto &body = anchor->getRegion(0).front();
    if(gridTiles)
      tile(make("pid", {}, rw.getIndexType(),rw.getNamedAttr("axis",rw.getI32IntegerAttr(1)))->getResult(0));
    else if (resident)
      tile(make("zero", {}, rw.getIndexType())->getResult(0));
    SmallVector<Operation*> ownerStores;
    for (auto &operation : body.without_terminator()) {
      auto code=operation.getName().stripDialect();
      if(code=="fill_ranges"){
        make("fill_ranges_at",{operation.getOperand(0),operation.getOperand(1),cols,active},{},operation.getAttrs());continue;
      }
      if(code=="write_offsets"){
        make("publish_offsets",{operation.getOperand(0),operation.getOperand(1),cols,active,expr(operation.getOperand(3))},{});
        continue;
      }
      if(code=="atomic_add"){expr(operation.getResult(0));continue;}
      if(code=="unique_put"){
        make("unique_put_at",{operation.getOperand(0),expr(operation.getOperand(1)),expr(operation.getOperand(2)),active},{},operation.getAttrs());
        continue;
      }
      if(code=="prefix_scatter"){
        if(!resident)llvm_unreachable("prefix scatter requires one complete row");
        make("scatter_fragment",{operation.getOperand(0),operation.getOperand(1),rows,cols,active,
             expr(operation.getOperand(3)),expr(operation.getOperand(4)),expr(operation.getOperand(5))},{},operation.getAttrs());
        continue;
      }
      if(code=="read_owner") {
        auto v=operation.getResult(0);
        fixed.map(v,make("owner_load",{operation.getOperand(0),rows},vector(v.getType(),1))->getResult(0));
        continue;
      }
      if(code=="write_owner"){ownerStores.push_back(&operation);continue;}
      if (operation.getName().stripDialect() == "write")
        writes.push_back(&operation);
      else if (isReduction(&operation)) {
        if (resident)
          fixed.map(operation.getResult(0), reduce(&operation));
        else
          sweep(&operation);
      }
    }
    if (!writes.empty()) {
      if (resident || gridTiles) {
        for (auto *write : writes)
          expr(write->getOperand(2));
        for (auto *write : writes)
          store(write);
      } else
        sweep(writes.front());
    }
    local.clear();
    for(auto *w:ownerStores)
      make("owner_store",{w->getOperand(0),rows,expr(w->getOperand(2))},{});
    rw.eraseOp(anchor);
  }

private:
  Operation *anchor;
  RewriterBase &rw;
  Location loc;
  Value domain, rows, cols, active;
  bool resident, gridTiles;
  IntegerAttr batch;
  StringAttr binMode;
  SmallVector<Operation *> writes;
  IRMapping fixed, local;
  Type vector(Type element, unsigned rank) {
    return RankedTensorType::get(
        SmallVector<int64_t>(rank, ShapedType::kDynamic), element);
  }
  Operation *make(StringRef code, ValueRange args, TypeRange results,
                  ArrayRef<NamedAttribute> attrs = {}) {
    OperationState state(loc, ("block." + code).str());
    state.addOperands(args);
    state.addTypes(results);
    state.addAttributes(attrs);
    return rw.create(state);
  }
  bool isReduction(Operation *op) {
    auto n = op->getName().stripDialect();
    return n == "sum" || n == "max_num" || n == "welford";
  }
  Value segment(Operation *op) {
    auto n=op->getName().stripDialect();
    if (n=="ordered_sum" || n=="segment_length") {
      bool fold=n=="ordered_sum";
      OperationState state(loc,fold?"block.segment_fold":"block.segment_size");
      state.addOperands({op->getOperand(0),rows,cols,active});
      state.addTypes(vector(rw.getF32Type(),2));
      if (fold) {
        state.addAttribute("strict",rw.getUnitAttr());
        if(batch)state.addAttribute("member_batch",batch);
        IRMapping capture;
        op->getRegion(0).cloneInto(state.addRegion(),capture);
      }
      auto fragment=rw.create(state)->getResult(0);
      return fragment;
    }
    llvm_unreachable("expected an offset-defined segment");
  }
  void tile(Value i) {
    cols = make("columns", i, vector(rw.getIndexType(), 1))->getResult(0);
    active = make("active", {domain, rows, cols}, vector(rw.getI1Type(), 2))
                 ->getResult(0);
    local.clear();
  }
  Value expr(Value v) {
    if (auto known = fixed.lookupOrNull(v))
      return known;
    if (auto known = local.lookupOrNull(v))
      return known;
    auto *op = v.getDefiningOp();
    if (!op)
      return v;
    auto n = op->getName().stripDialect();
    if(n=="ordered_sum" || n=="segment_length"){
      auto fragment=segment(op);
      local.map(v,fragment);
      return fragment;
    }
    if(n=="count_keys" || n=="column_sum"){
      auto fragment=make(n=="count_keys"?"key_counts":"sum_columns",{op->getOperand(0),rows,cols,active},
                         vector(rw.getI32Type(),2),op->getAttrs())->getResult(0);
      if(n=="count_keys")fragment.getDefiningOp()->setAttr("bin_mode",binMode);
      local.map(v,fragment);return fragment;
    }
    if (n == "field") {
      auto state=expr(op->getOperand(0));
      auto result=make("summary_field",state,vector(v.getType(),1),op->getAttrs())->getResult(0);
      local.map(v,result);
      return result;
    }
    if(n=="gather_row"){
      auto elem=cast<MemRefType>(op->getOperand(0).getType()).getElementType();
      auto result=make("indexed_tile_read",{op->getOperand(0),op->getOperand(1),rows,cols,active},vector(elem,2))->getResult(0);
      local.map(v,result);return result;
    }
    if (n == "ordinal") {
      auto ordinal =
          make("linear_index", {domain, rows, cols},
               vector(cast<RankedTensorType>(v.getType()).getElementType(), 2))
              ->getResult(0);
      local.map(v, ordinal);
      return ordinal;
    }
    if(n=="member_index"){
      auto index=make("member_tile",{cols,active},vector(rw.getI32Type(),2))->getResult(0);
      local.map(v,index);return index;
    }
    if(n=="atomic_add" || n=="index_read"){
      SmallVector<Value> inputs{op->getOperand(0),expr(op->getOperand(1))};
      if(n=="atomic_add")inputs.push_back(expr(op->getOperand(2)));
      inputs.push_back(active);
      auto fragment=make((n+"_at").str(),inputs,vector(rw.getI32Type(),2),op->getAttrs())->getResult(0);
      local.map(v,fragment);return fragment;
    }
    if (n == "read") {
      auto view = op->getOperand(0);
      auto elem = cast<MemRefType>(view.getType()).getElementType();
      auto x = make("load", {view, rows, cols, active}, vector(elem, 2))
                   ->getResult(0);
      if(anchor->getParentOfType<func::FuncOp>()->hasAttr("kir.iteration")){
        SmallVector<int64_t> axes;
        for(int i=0;i<cast<MemRefType>(domain.getType()).getRank();++i)axes.push_back(i);
        x.getDefiningOp()->setAttr("block.member_axes",rw.getDenseI64ArrayAttr(axes));
      } else if (cast<MemRefType>(domain.getType()).getRank() == 1)
        x.getDefiningOp()->setAttr("block.member_axes",
                                   rw.getDenseI64ArrayAttr({0}));
      local.map(v, x);
      return x;
    }
    SmallVector<Value> args;
    unsigned rank = 0;
    for (Value operand : op->getOperands()) {
      auto input = expr(operand);
      args.push_back(input);
      if (auto shaped = dyn_cast<RankedTensorType>(input.getType()))
        rank = std::max(rank, unsigned(shaped.getRank()));
    }
    auto element = v.getType();
    if (auto shaped = dyn_cast<RankedTensorType>(element))
      element = shaped.getElementType();
    auto x =
        make(n, args, rank ? vector(element, rank) : element, op->getAttrs())
            ->getResult(0);
    local.map(v, x);
    return x;
  }
  NamedAttribute algebra(Operation *op) {
    return rw.getNamedAttr("algebra",
                           rw.getStringAttr(op->getName().stripDialect()));
  }
  Value reduce(Operation *op) {
    if (op->getName().stripDialect()=="welford")
      return make("summary_reduce",{expr(op->getOperand(0)),active},summaryType())->getResult(0);
    return make("reduce", {expr(op->getOperand(0)), active},
                vector(op->getResult(0).getType(), 1), algebra(op))
        ->getResult(0);
  }
  Type summaryType() {
    return TupleType::get(rw.getContext(),{vector(rw.getI32Type(),1),vector(rw.getF32Type(),1),vector(rw.getF32Type(),1)});
  }
  void store(Operation *op) {
    auto *store = make(
        "store",
        {op->getOperand(0), rows, cols, active, expr(op->getOperand(2))}, {});
    if (auto axes = op->getAttr("axes"))
      store->setAttr("block.member_axes", axes);
    else if(anchor->getParentOfType<func::FuncOp>()->hasAttr("kir.iteration")){
        SmallVector<int64_t> axes;
        for(int i=0;i<cast<MemRefType>(domain.getType()).getRank();++i)axes.push_back(i);
        store->setAttr("block.member_axes",rw.getDenseI64ArrayAttr(axes));
      } else if (cast<MemRefType>(domain.getType()).getRank() == 1)
      store->setAttr("block.member_axes", rw.getDenseI64ArrayAttr({0}));
  }
  void sweep(Operation *op) {
    bool accum = isReduction(op);
    OperationState state(loc, "block.tiles");
    state.addOperands(domain);
    if (accum) {
      bool tuple=op->getName().stripDialect()=="welford";
      auto t = tuple?summaryType():vector(op->getResult(0).getType(), 1);
      state.addOperands(make(tuple?"summary_identity":"identity", {}, t, algebra(op))->getResult(0));
      state.addTypes(t);
    }
    state.addRegion();
    auto *loop = rw.create(state);
    {
      OpBuilder::InsertionGuard guard(rw);
      auto *bb = rw.createBlock(&loop->getRegion(0));
      tile(bb->addArgument(rw.getIndexType(), loc));
      if (accum) {
        auto prev = bb->addArgument(loop->getResult(0).getType(), loc);
        auto combined =
            make(op->getName().stripDialect()=="welford"?"summary_merge":"combine", {prev, reduce(op)}, prev.getType(), algebra(op));
        make("yield", combined->getResults(), {});
      } else {
        for (auto *w : writes)
          expr(w->getOperand(2));
        for (auto *w : writes)
          store(w);
        make("yield", {}, {});
      }
    }
    if (accum)
      fixed.map(op->getResult(0), loop->getResult(0));
  }
};
} // namespace
void registerRealizer(Unit &unit) {
  Realizer r;
  r.name = "tile-rows";
  r.family = unit.qualify("structure");
  r.apply = [](Operation *op, Attribute a, const TargetProfile &p,
               RewriterBase &rw) {
    llvm::Error problem=llvm::Error::success();
    op->walk([&](Operation *v){
      if(problem)return;
      auto n=v->getName().stripDialect();if(n!="count_keys" && n!="column_sum")return;
      auto e=intentdsl::kir::keyExtent(v,*p.services.get<intentdsl::CallFacts>());
      if(!e)problem=e.takeError();else v->setAttr("extent",rw.getI64IntegerAttr(*e));
    });
    if(problem)return problem;
    RowTile(op, cast<DictionaryAttr>(a), rw).build();
    return llvm::Error(llvm::Error::success());
  };
  unit.realizers.push_back({std::move(r), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
