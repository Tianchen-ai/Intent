#include "../capabilities/Index.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/execution/block/ir/Access.h"
#include "intentdsl/levels/execution/block/decisions/Verify.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "Dialect.h.inc"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void registerIR(DialectRegistry &);
llvm::Expected<TargetProfile> capabilities(const llvm::json::Object &);
llvm::Error emit(ModuleOp, llvm::raw_pwrite_stream &);
namespace {
struct Lower : PassWrapper<Lower, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Lower)
  bool tma;
  explicit Lower(bool permit) : tma(permit) {}
  void runOnOperation() final {
    getContext().getOrLoadDialect<intentdsl::cutile::TileSource>();
    getOperation().walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (op->getName().getDialectNamespace() != "block")
        return;
      OpBuilder b(op);
      auto code = op->getName().stripDialect();
      bool access = code == "load" || code == "store";
      OperationState s(op->getLoc(),
                       "ct_src." +
                           (access ? ("tile_" + code).str() : code.str()));
      s.addAttributes(op->getAttrs());
      if (access) {
        auto view = op->getOperand(0);
        auto axes = intentdsl::block::projection(view.getType());
        auto *r = op->getOperand(1).getDefiningOp(),
             *c = op->getOperand(2).getDefiningOp();
        // normalized verified a rectangular active domain and both coordinates.
        s.addOperands({view, r->getOperand(0), c->getOperand(0)});
        s.addAttribute("allow_tma", b.getBoolAttr(tma));
        s.addAttribute(
            "matrix",
            b.getBoolAttr(*axes == intentdsl::block::RowProjection::Matrix));
        if (code == "store")
          s.addOperands(op->getOperand(4));
      } else
        s.addOperands(op->getOperands());
      s.addTypes(op->getResultTypes());
      for (auto &region : op->getRegions())
        s.addRegion()->takeBody(region);
      op->replaceAllUsesWith(b.create(s));
      op->erase();
    });
  }
};
struct Signature : PassWrapper<Signature, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Signature)
  unsigned bits;
  explicit Signature(unsigned bits) : bits(bits) {}
  void runOnOperation() final {
    auto fn = *getOperation().getOps<func::FuncOp>().begin();
    Builder b(&getContext());
    fn->setAttr("ct_src.entry", b.getStringAttr("kernel"));
    fn->setAttr("ct_src.index_bits", b.getI32IntegerAttr(bits));
  }
};
LogicalResult verify(ModuleOp module, const TargetProfile &profile,
                     bool callable) {
  if (failed(intentdsl::verifyMapping(module, profile, "ct_src")))
    return failure();
  auto fn = *module.getOps<func::FuncOp>().begin();
  auto indices = profile.services.get<intentdsl::cutile::ArrayIndices>();
  if (!indices)
    return fn.emitError("missing cuTile array index profile");
  if (callable &&
      (fn->getAttrOfType<StringAttr>("ct_src.entry") != "kernel" ||
       fn->getAttr("ct_src.index_bits") !=
           Builder(module.getContext()).getI32IntegerAttr(indices->bits)))
    return fn.emitError("missing cuTile callable ABI");
  for (auto row : fn->getAttrOfType<ArrayAttr>("block.configs")) {
    auto v = llvm::json::parse(
        cast<DictionaryAttr>(row).getAs<StringAttr>("payload").getValue());
    if (!v) {
      llvm::consumeError(v.takeError());
      return failure();
    }
    auto *o = v->getAsObject();
    if (!o || o->size() != 3 || o->getInteger("num_ctas") != 1 ||
        o->getInteger("opt_level") != 3 ||
        (o->getInteger("occupancy") != 1 && o->getInteger("occupancy") != 2))
      return fn.emitError("invalid native hints for row");
  }
  auto res = module.walk([&](Operation *op) {
    if (op == module || op->getName().getDialectNamespace() == "func")
      return WalkResult::advance();
    auto n = op->getName().stripDialect();
    bool ok = op->getName().getDialectNamespace() == "ct_src";
    if (ok && (n == "tile_store" || n == "tile_load")) {
      auto axes = intentdsl::block::projection(op->getOperand(0).getType());
      auto *pid = op->getOperand(1).getDefiningOp(),
           *col = op->getOperand(2).getDefiningOp();
      auto ind = dyn_cast<BlockArgument>(op->getOperand(2));
      ok = op->getAttrOfType<BoolAttr>("allow_tma").getValue() ==
               indices->permitTma &&
           succeeded(axes) &&
           op->getAttrOfType<BoolAttr>("matrix").getValue() ==
               (*axes == intentdsl::block::RowProjection::Matrix) &&
           (n != "tile_store" ||
            *axes == intentdsl::block::RowProjection::Matrix) &&
           pid && pid->getName().getStringRef() == "ct_src.pid" &&
           ((col && col->getName().getStringRef() == "ct_src.zero") ||
            (ind && ind.getArgNumber() == 0 &&
             ind.getOwner()->getParentOp()->getName().getStringRef() ==
                 "ct_src.tiles"));
    }
    if (!ok) {
      op->emitError("invalid cuTile native projection or tile index");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success(!res.wasInterrupted());
}
} // namespace
void bind(Unit &unit) {
  unit.dialects = registerIR;
  for (bool callable : {false, true})
    unit.boundaries.push_back(
        {{callable ? "callable" : "legal",
          [callable](OpPassManager &pm, const TargetProfile &p) -> llvm::Error {
            if (callable) {
              auto indices = p.services.get<intentdsl::cutile::ArrayIndices>();
              if (!indices)
                return llvm::createStringError(
                    "cuTile needs bound device options");
              pm.addPass(std::make_unique<Signature>(indices->bits));
            } else {
              auto indices = p.services.get<intentdsl::cutile::ArrayIndices>();
              if (!indices)
                return llvm::createStringError("missing array profile");
              pm.addPass(std::make_unique<Lower>(indices->permitTma));
            }
            return llvm::Error::success();
          },
          [callable](ModuleOp m, const TargetProfile &p) {
            return verify(m, p, callable);
          }},
         COMPILERINFRA_SOURCE});
  unit.exit =
      Located<ExitImplementation>{{emit, capabilities}, COMPILERINFRA_SOURCE};
}
} // namespace CI_UNIT_NAMESPACE
