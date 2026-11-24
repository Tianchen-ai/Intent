#include "../capabilities/Target.h"
#include "intentdsl/levels/execution/block/ir/Access.h"
#include "intentdsl/levels/execution/block/ir/MatrixAccess.h"
#include "intentdsl/levels/execution/block/ir/ScaleDispatch.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/exits/triton/ir/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
std::optional<int64_t> tritonOwnerRectangleAxis(Operation *, func::FuncOp);
bool tritonOwnerRectPermitsTma(Operation *, func::FuncOp, bool);
bool tritonPanelPermitsTma(Operation *, func::FuncOp, bool);
bool tritonPagePermitsTma(Operation *, func::FuncOp, bool);

LogicalResult verifyMatrixExit(func::FuncOp fn,const compilerinfra::TargetProfile &profile){
  auto cap = profile.services.get<intentdsl::itsrc::TritonTarget>();
  if (!cap || fn->hasAttr("triton_src.tma") != cap->permitTma)
    return fn.emitError("invalid Triton transfer capability");
  return intentdsl::block::verifyScaleDispatch(
      fn, profile, "triton_src", cap->scaledMma ? "mx" : "bf16");
}
LogicalResult verifyColumnViews(func::FuncOp fn) {
  if (!fn.empty() && !fn.front().empty() &&
      fn.front().front().hasAttr("block.column_prefixes")) {
    auto rnWalk = fn.walk([&](Operation *op) {
      if (isa<arith::AddFOp, arith::SubFOp, math::SqrtOp>(op) &&
          !op->hasAttr("triton_src.round_nearest")) {
        op->emitError("Cholesky column update requires round-to-nearest");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (rnWalk.wasInterrupted())
      return failure();
  }
  auto walk = fn.walk([&](Operation *op) {
    auto code = op->getName().getStringRef();
    if (code != "triton_src.read_owner_matrix" && code != "triton_src.store_column")
      return WalkResult::advance();
    auto view = dyn_cast<BlockArgument>(op->getOperand(0));
    if (view && view.getOwner() == &fn.front()) {
      auto s = view.getArgNumber();
      auto shape = fn.getArgAttrOfType<DenseI64ArrayAttr>(s, "block.shape");
      auto strides = fn.getArgAttrOfType<DenseI64ArrayAttr>(s, "block.strides");
      Builder b(fn.getContext());
      if (shape && shape.size() == 3 && shape[1] == shape[2] &&
          shape[1] > 0 && llvm::isPowerOf2_64(shape[1]) &&
          shape[1] <= INT32_MAX && strides && strides.size() == 3 &&
          op->getAttr("triton_src.view_strides") == strides &&
          op->getAttr("triton_src.side") == b.getI64IntegerAttr(shape[1]) &&
          op->getAttr("address_bits") == b.getI32IntegerAttr(64))
        return WalkResult::advance();
    }
    op->emitError("requires a square column tile with ABI strides and i64 addressing");
    return WalkResult::interrupt();
  });
  return success(!walk.wasInterrupted());
}
struct Legalize : PassWrapper<Legalize, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Legalize)
  bool tma = false, scaledMma = false;
  Legalize() = default;
  Legalize(bool tma, bool scaledMma) : tma(tma), scaledMma(scaledMma) {}
  void runOnOperation() final {
    getContext().getOrLoadDialect<intentdsl::itsrc::ItsrcDialect>();
    auto entryFn = *getOperation().getOps<func::FuncOp>().begin();
    if (tma)
      entryFn->setAttr("triton_src.tma", UnitAttr::get(&getContext()));
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "block.owner_space" &&
          op->hasAttr("block.pair_axis") &&
          op->getAttrOfType<IntegerAttr>("block.owner_tile").getInt() < 16)
        op->setAttr("triton_src.matrix_rows", Builder(&getContext()).getI64IntegerAttr(16));
      if (op->getName().getStringRef() == "block.owner_space" &&
          op->hasAttr("block.column_prefixes")) {
        Builder b(&getContext());
        op->walk([&](Operation *inner) {
          if (isa<arith::AddFOp, arith::SubFOp, math::SqrtOp>(inner))
            inner->setAttr("triton_src.round_nearest", b.getUnitAttr());
        });
      }
    });
    getOperation().walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (op->getName().getDialectNamespace() != "block")
        return;
      OpBuilder b(op);
      OperationState state(
          op->getLoc(), ("triton_src." + op->getName().stripDialect()).str());
      auto code = op->getName().stripDialect();
      auto args = op->getOperands();
      if (code == "read_owner_matrix" || code == "store_column") {
        auto slot = cast<BlockArgument>(args[0]).getArgNumber();
        auto fn = op->getParentOfType<func::FuncOp>();
        auto shape = fn.getArgAttrOfType<DenseI64ArrayAttr>(slot, "block.shape");
        state.addAttribute("triton_src.side", b.getI64IntegerAttr(shape[1]));
        state.addAttribute("triton_src.view_strides", fn.getArgAttr(slot, "block.strides"));
        state.addAttribute("address_bits", b.getI32IntegerAttr(64));
      }
      if (code == "key_counts" || code == "sum_columns") {
        auto count = op->getAttrOfType<IntegerAttr>(
                           code == "key_counts" ? "span" : "extent")
                         .getInt();
        uint64_t width =
            std::min<uint64_t>(code == "key_counts" ? 8192 : 256,
                               llvm::PowerOf2Ceil(std::max<int64_t>(1, count)));
        state.addAttribute("native_panel", b.getI64IntegerAttr(width));
        state.addAttribute("address_bits", b.getI32IntegerAttr(64));
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
      auto fn = op->getParentOfType<func::FuncOp>();
      if (tritonPagePermitsTma(op, fn, tma))
        state.addAttribute("triton_src.page_tma", b.getUnitAttr());
      if (tritonPanelPermitsTma(op, fn, tma))
        state.addAttribute("triton_src.panel_tma", b.getUnitAttr());
      if (auto rectAx = tritonOwnerRectangleAxis(op, fn)) {
        state.addAttribute("triton_src.owner_rectangle",
                           b.getI64IntegerAttr(*rectAx));
        if (tritonOwnerRectPermitsTma(op, fn, tma))
          state.addAttribute("triton_src.owner_tma", b.getUnitAttr());
      }
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
        intentdsl::block::formScaleDispatch(p.dot, scaledMma ? "mx" : "bf16");
      }
    }
  }
};
void legalize(OpPassManager &pm, const compilerinfra::TargetProfile &profile) {
  auto cap = profile.services.get<intentdsl::itsrc::TritonTarget>();
  pm.addPass(std::make_unique<Legalize>(cap && cap->permitTma,
                                        cap && cap->scaledMma));
}
} // namespace CI_UNIT_NAMESPACE
