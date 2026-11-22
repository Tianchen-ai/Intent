#include "compilerinfra/Assembly.h"
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
    gridTiles = choice.get("walk") && choice.getAs<StringAttr>("walk")=="grid-tile";
    fn->setAttr("block.grid", intentdsl::block::grid(rw,gridTiles));
    fn->setAttr("block.configs", choice.get("rows"));
    fn->setAttr("block.topology", choice.get("topology"));
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
  r.apply = [](Operation *op, Attribute a, const TargetProfile &,
               RewriterBase &rw) {
    RowTile(op, cast<DictionaryAttr>(a), rw).build();
    return llvm::Error::success();
  };
  unit.realizers.push_back({std::move(r), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
