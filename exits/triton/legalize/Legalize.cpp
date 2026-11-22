#include "intentdsl/levels/execution/block/ir/Access.h"
#include "intentdsl/levels/execution/block/ir/MatrixAccess.h"
#include "intentdsl/levels/execution/block/ir/ScaleDispatch.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/exits/triton/ir/Dialect.h"
#include "mlir/Pass/Pass.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
bool neutralScaleDispatch(intentdsl::block::MatrixAccess &p) {
  auto groups=p.shape[2]/32;
  auto *publish=p.fold.getNextNode();
  return groups>0 && groups<=64 && llvm::isPowerOf2_64(groups) && publish &&
      publish->getName().stripDialect()=="publish_matrix" && publish->getOperand(1)==p.fold.getResult(0);
}
LogicalResult verifyMatrixExit(func::FuncOp fn,const compilerinfra::TargetProfile &profile){
  if(failed(intentdsl::block::verifyScaleDispatch(fn,profile,"triton_src","bf16")))return failure();
  intentdsl::block::MatrixAccess p(fn);
  if(p.dot.getName().stripDialect()=="scaled_contract" &&
     p.dot.getAttr("triton_src.neutral_scale_dispatch")!=Builder(fn.getContext()).getBoolAttr(neutralScaleDispatch(p)))
    return p.dot.emitError("unit-scale FP8 dispatch disagrees with reduction geometry or output use");
  return success();
}
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
    getOperation().walk([](Operation *store){
      if(store->getName().getStringRef()!="triton_src.store")return;
      auto *count=store->getOperand(3).getDefiningOp();
      if(!count || !count->hasOneUse() || count->getName().getStringRef()!="triton_src.key_counts" ||
         count->getAttrOfType<StringAttr>("bin_mode")!="global-atomic")return;
      OpBuilder b(store);
      OperationState state(store->getLoc(),"triton_src.atomic_bins");
      state.addOperands({count->getOperand(0),store->getOperand(0),count->getOperand(2)});
      auto *offset=store->getOperand(1).getDefiningOp();
      state.addAttribute("transposed",offset->getAttr("permuted")?offset->getAttr("permuted"):b.getBoolAttr(false));
      state.addAttributes(count->getAttrs());state.addRegion();
      auto *bins=b.create(state);b.createBlock(&bins->getRegion(0));
      b.create(OperationState(store->getLoc(),"triton_src.clear_bins"));
      b.create(OperationState(store->getLoc(),"triton_src.cta_barrier"));
      OperationState loop(store->getLoc(),"triton_src.bin_panels");
      auto span=count->getAttrOfType<IntegerAttr>("span").getInt();
      auto width=count->getAttrOfType<IntegerAttr>("native_panel").getInt();
      loop.addAttribute("count",b.getI64IntegerAttr(llvm::divideCeil(span,width)));loop.addRegion();
      auto *body=b.createBlock(&b.create(loop)->getRegion(0),{},b.getIndexType(),{store->getLoc()});
      OperationState add(store->getLoc(),"triton_src.bin_add");add.addOperands(body->getArgument(0));b.create(add);
      store->erase();count->erase();
    });
    for(auto fn:getOperation().getOps<func::FuncOp>())if(intentdsl::kir::matrixDomain(fn)){
      intentdsl::block::MatrixAccess p(fn);
      if(p.dot.getName().stripDialect()=="scaled_contract"){
        intentdsl::block::formScaleDispatch(p.dot,"bf16");
        p.dot.setAttr("triton_src.neutral_scale_dispatch",Builder(fn.getContext()).getBoolAttr(neutralScaleDispatch(p)));
      }
    }
  }
};
void legalize(OpPassManager &pm) { pm.addPass(std::make_unique<Legalize>()); }
} // namespace CI_UNIT_NAMESPACE
