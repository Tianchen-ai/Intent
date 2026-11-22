#include "intentdsl/levels/execution/block/ir/Access.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/exits/triton/ir/Dialect.h"
#include "mlir/Pass/Pass.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
struct Legalize : PassWrapper<Legalize, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Legalize)
  void runOnOperation() final {
    getContext().getOrLoadDialect<intentdsl::itsrc::ItsrcDialect>();
    getOperation().walk<WalkOrder::PostOrder>([](Operation *op) {
      if (op->getName().getDialectNamespace() != "block")
        return;
      OpBuilder b(op);
      OperationState state(
          op->getLoc(), ("triton_src." + op->getName().stripDialect()).str());
      auto code = op->getName().stripDialect();
      auto args = op->getOperands();
      if(code=="key_counts" || code=="sum_columns"){
        auto count=op->getAttrOfType<IntegerAttr>(code=="key_counts"?"span":"extent").getInt();
        uint64_t width=std::min<uint64_t>(code=="key_counts"?8192:128,llvm::PowerOf2Ceil(std::max<int64_t>(1,count)));
        state.addAttribute("native_panel",b.getI64IntegerAttr(width));
        state.addAttribute("address_bits",b.getI32IntegerAttr(64));
      }
      if (code == "load" || code == "store") {
        OperationState offset(op->getLoc(), "triton_src.offsets");
        auto axes=*intentdsl::block::projection(args.front().getType());
        if (auto axes =
                op->getAttrOfType<DenseI64ArrayAttr>("block.member_axes"))
          offset.addAttribute("permuted",
                              b.getBoolAttr(axes.size() == 2 && axes[0] == 1));
        offset.addAttribute("matrix",b.getBoolAttr(axes==intentdsl::block::RowProjection::Matrix));
        offset.addOperands(args.take_front(3));
        offset.addTypes(RankedTensorType::get(
            {ShapedType::kDynamic, ShapedType::kDynamic}, b.getIndexType()));
        state.addOperands(args.front());
        state.addOperands(b.create(offset)->getResults());
        state.addOperands(args.drop_front(3));
      } else if (code == "reduce") {
        OperationState mask(op->getLoc(), "triton_src.neutralize");
        mask.addOperands(args);
        mask.addTypes(args[0].getType());
        mask.addAttributes(op->getAttrs());
        state.addOperands(b.create(mask)->getResults());
      } else
        state.addOperands(args);
      state.addTypes(op->getResultTypes());
      state.addAttributes(op->getAttrs());
      for (auto &region : op->getRegions())
        state.addRegion()->takeBody(region);
      auto *newOp = b.create(state);
      op->replaceAllUsesWith(newOp);
      op->erase();
    });
  }
};
void legalize(OpPassManager &pm) { pm.addPass(std::make_unique<Legalize>()); }
} // namespace CI_UNIT_NAMESPACE
