#include "intentdsl/levels/algorithm/kir/ir/Permutation.h"
#include "intentdsl/levels/execution/block/ir/Access.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/RowSemantics.h"
#include "intentdsl/levels/execution/block/decisions/Verify.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "intentdsl/levels/execution/block/ir/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace compilerinfra;
namespace CI_UNIT_NAMESPACE {
#include "Strategies.inc"
void normalize(OpPassManager &);
void registerRealizer(Unit &);
namespace {
struct Form : PassWrapper<Form, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Form)
  void runOnOperation() final {
    getContext().getOrLoadDialect<intentdsl::iblock::IblockDialect>();
    getOperation().walk<WalkOrder::PostOrder>([](Operation *op) {
      if (op->getName().getDialectNamespace() != "kir")
        return;
      OpBuilder b(op);
      OperationState state(op->getLoc(),
                           ("block." + op->getName().stripDialect()).str());
      state.addOperands(op->getOperands());
      state.addTypes(op->getResultTypes());
      state.addAttributes(op->getAttrs());
      for (auto &region : op->getRegions())
        state.addRegion()->takeBody(region);
      auto *next = b.create(state);
      op->replaceAllUsesWith(next);
      op->erase();
    });
  }
};
} // namespace
void registerCost(Unit &);
void bind(Unit &unit) {
  registerCost(unit);
  unit.dialects = [](DialectRegistry &r) {
    r.insert<intentdsl::iblock::IblockDialect>();
  };
  auto family = unit.qualify("structure");
  DecisionFamily f;
  f.name = "structure";
  f.applies = [](Operation *op) {
    return op->getName().getStringRef() == "block.parallel";
  };
  f.read = [](Operation *op) { return op->getAttr("block.structure"); };
  f.write = [](Operation *op, Attribute v) {
    op->setAttr("block.structure", v);
    return llvm::Error::success();
  };
  f.verify = [](Operation *op, Attribute value,
                const TargetProfile &profile) -> llvm::Error {
    auto d = dyn_cast<DictionaryAttr>(value);
    if(!d)return llvm::createStringError("missing block decision");
    return intentdsl::verifyRows(op,d.getAs<ArrayAttr>("rows"),d.getAs<StringAttr>("topology"),d.getAs<DenseI64ArrayAttr>("guard"),profile);
  };
  f.verifyCandidate=[](Operation *op,const StructuralCandidate &c,const TargetProfile &)->llvm::Error{
    auto d=cast<DictionaryAttr>(c.value);auto topology=d.getAs<StringAttr>("topology").str();
    auto rows=d.getAs<ArrayAttr>("rows");auto group=cast<DictionaryAttr>(rows[0]).getAs<IntegerAttr>("rows");
    if(auto walk=d.getAs<StringAttr>("walk")){
      if(c.topology!=walk.str() || c.id!=walk.str() || c.preconditions!=std::vector<std::string>{walk=="source"?"independent-members":"bijective-rectangular-copy"} ||
         (walk!="source" && (!intentdsl::rectangularCopy(op) || (walk!="destination" && walk!="tile") || topology!="stream")))
        return llvm::createStringError("invalid traversal for the access domain");
      return llvm::Error::success();
    }
    if(c.preconditions.empty() && c.id==c.topology && c.id.find("pointwise")!=std::string::npos)return llvm::Error::success();
    if(c.topology!=topology || c.id!=topology+"-"+std::to_string(group.getInt()) ||
       c.preconditions!=std::vector<std::string>{"legal-width-group"} ||
       !llvm::all_of(rows,[&](Attribute a){return cast<DictionaryAttr>(a).get("rows")==group;}))return llvm::createStringError("block topology and row domain disagree");
    return llvm::Error::success();};
  unit.families.push_back({{std::move(f), "block@decided", "block@realized"},
                           COMPILERINFRA_SOURCE});
  registerStrategies(unit);
  registerRealizer(unit);
  for (StringRef phase : {"formed", "decided", "realized", "normalized"}) {
    unit.boundaries.push_back(
        {{phase.str(),
          [phase](OpPassManager &pm, const TargetProfile &) {
            if (phase == "normalized")
              normalize(pm);
            if (phase == "formed")
              pm.addPass(std::make_unique<Form>());
            return llvm::Error::success();
          },
          [phase](ModuleOp m, const TargetProfile &profile) {
            unsigned loops = 0, pids = 0, pending = 0;
            m.walk([&](Operation *op) {
              loops += op->getName().getStringRef() == "block.parallel";
              pids += op->getName().getStringRef() == "block.pid";
              pending += op->hasAttr("block.structure");
            });
            bool ready = phase == "realized" || phase == "normalized";
            if (loops != unsigned(!ready) || (pids != 0) != ready ||
                pending != unsigned(phase == "decided"))
              return m.emitError("invalid row structure at ") << phase,
                     failure();
            auto funcs = m.getOps<func::FuncOp>();
            if (!llvm::hasSingleElement(funcs))
              return failure();
            auto fn = *funcs.begin();
            if (ready && (!fn->getAttrOfType<ArrayAttr>("block.configs") ||
                          !fn->getAttrOfType<StringAttr>("block.topology")))
              return failure();
            if(!ready && failed(intentdsl::verifyLogicalRows(m,"block",true))) return failure();
            if(ready && failed(intentdsl::verifyMapping(m,profile,"block"))) return failure();
            auto checked = m.walk([&](Operation *op) {
              if (op->getName().getDialectNamespace() == "func" ||
                  op == m.getOperation())
                return WalkResult::advance();
              if (op->getName().getDialectNamespace() != "block")
                return WalkResult::interrupt();
              auto n = op->getName().stripDialect();
              if (phase=="normalized" && (n=="load" || n=="store") && failed(intentdsl::block::verifyRowAccess(op)))return WalkResult::interrupt();
              bool logical = n == "parallel" || n == "end" || n == "read" ||
                             n == "write" || n == "sum" || n == "max_num";
              if (ready && logical)
                return WalkResult::interrupt();
              if (phase == "normalized" && op->getNumOperands() == 2 &&
                  (n == "add" || n == "sub" || n == "mul" || n == "div")) {
                auto ty =
                    dyn_cast<RankedTensorType>(op->getResult(0).getType());
                for (auto val : op->getOperands()) {
                  auto t = dyn_cast<RankedTensorType>(val.getType());
                  if (ty && ty.getRank() == 2 && t && t.getRank() == 1)
                    return WalkResult::interrupt();
                }
              }
              return WalkResult::advance();
            });
            if (checked.wasInterrupted())
              return m.emitError("invalid operation or broadcast at ") << phase,
                     failure();
            return success();
          }},
         COMPILERINFRA_SOURCE});
  }
}
} // namespace CI_UNIT_NAMESPACE
