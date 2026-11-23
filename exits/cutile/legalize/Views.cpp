#include "intentdsl/levels/execution/block/ir/RowProgram.h"
#include "intentdsl/levels/execution/block/ir/MatrixProgram.h"
#include "intentdsl/levels/execution/block/ir/OwnerProgram.h"
#include "../capabilities/Index.h"
#include "intentdsl/levels/execution/block/ir/ScaleDispatch.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/ProjectedLayout.h"
#include "intentdsl/levels/execution/block/decisions/Verify.h"
#include "intentdsl/levels/execution/block/ir/Access.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "Dialect.h.inc"
namespace CI_UNIT_NAMESPACE {
mlir::Value scaledPanels(intentdsl::block::Tiles &,intentdsl::block::MatrixAccess &);
using namespace mlir;
using namespace compilerinfra;
void registerIR(DialectRegistry &);
void partitionOwners(func::FuncOp,const intentdsl::CallFacts &);
llvm::Expected<TargetProfile> capabilities(const llvm::json::Object &);
llvm::Error describeArtifact(ModuleOp, llvm::raw_pwrite_stream &);
TerminalPrinter cutilePrinter();
void registerContracts(Unit &);
LogicalResult verifyArrayGeometry(func::FuncOp,const TargetProfile &);
intentdsl::block::OwnerAccess nativeOwnerViews();
namespace {
bool ownerRectangle(Operation *op, func::FuncOp fn) {
  auto kind = op->getName().stripDialect();
  if (kind != "slice_read" && kind != "slice_write")
    return false;
  auto &space = fn.front().front();
  auto pair = space.getAttrOfType<IntegerAttr>("block.pair_axis");
  if (!pair || space.hasAttr("ct_src.matrix_rows") ||
      op->getParentOp() != &space)
    return false;
  auto &bb = space.getRegion(0).front();
  auto coords = op->getOperands().drop_front(kind == "slice_write" ? 2 : 1);
  if (coords.empty() ||
      op->getAttrOfType<IntegerAttr>("axis").getInt() !=
          int64_t(coords.size()) ||
      coords.back() != bb.getArgument(pair.getInt()))
    return false;
  auto view =
      cast<BlockArgument>(op->getOperand(kind == "slice_write" ? 1 : 0));
  auto shape = fn.getArgAttrOfType<DenseI64ArrayAttr>(view.getArgNumber(),
                                                      "block.shape");
  return shape &&
         shape[shape.size() - 2] %
                 space.getAttrOfType<IntegerAttr>("block.owner_tile")
                     .getInt() ==
             0 &&
         llvm::isPowerOf2_64(shape.asArrayRef().back()) &&
         llvm::all_of(coords.drop_back(), [&](Value v) {
           auto arg = dyn_cast<BlockArgument>(v);
           return arg && arg.getOwner() == &bb && arg != coords.back();
         });
}
DenseI64ArrayAttr adjacentScaleGroups(Operation *op,func::FuncOp fn){
  SmallVector<int64_t> slots;
  for(unsigned pos:{0u,2u}){
    auto i=cast<BlockArgument>(op->getOperand(pos)).getArgNumber();
    auto s=fn.getArgAttrOfType<DenseI64ArrayAttr>(i,"block.strides");
    int axis=pos==0?1:0;
    if(s && s.size()==3 && s[axis]==32*s[axis+1])slots.push_back(i);
  }
  return Builder(fn.getContext()).getDenseI64ArrayAttr(slots);
}
bool rectangularSlab(Operation *op,func::FuncOp fn){
  auto arg=dyn_cast<BlockArgument>(op->getOperand(0));
  auto owner=fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
  if(!owner || owner.getInt()>1 || !arg || arg.getOwner()!=&fn.front())return false;
  auto shape=fn.getArgAttrOfType<DenseI64ArrayAttr>(arg.getArgNumber(),"ct_src.shape");
  return shape && shape.size()==3 && llvm::isPowerOf2_64(shape[2]) &&
    llvm::all_of(fn->getAttrOfType<ArrayAttr>("block.configs"),[](Attribute row){
      return llvm::isPowerOf2_64(cast<DictionaryAttr>(row).getAs<IntegerAttr>("width").getInt());
    });
}
struct Lower : PassWrapper<Lower, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Lower)
  bool tma, scaledMma;
  intentdsl::CallFacts facts;
  Lower(const intentdsl::cutile::ArrayIndices &indices,const intentdsl::CallFacts &f) : tma(indices.permitTma),scaledMma(indices.scaledMma),facts(f) {}
  void runOnOperation() final {
    auto fn=*getOperation().getOps<func::FuncOp>().begin();
    if(tma)fn->setAttr("ct_src.tma",UnitAttr::get(fn.getContext()));
    if(failed(intentdsl::specializeProjectedLayout(fn,facts,"ct_src",true))){signalPassFailure();return;}
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "block.owner_space" &&
          op->hasAttr("block.pair_axis") &&
          op->getAttrOfType<IntegerAttr>("block.owner_tile").getInt() < 16)
        op->setAttr("ct_src.matrix_rows", Builder(&getContext()).getI64IntegerAttr(16));
    });
    getContext().getOrLoadDialect<intentdsl::cutile::TileSource>();
    getOperation().walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (op->getName().getDialectNamespace() != "block")
        return;
      OpBuilder b(op);
      auto code = op->getName().stripDialect();
      bool access = code == "load" || code == "store";
      bool slab = access && op->getParentOfType<func::FuncOp>()->hasAttr("kir.owner_axis");
      OperationState s(op->getLoc(),
                       "ct_src." +
                           (slab ? (code=="load"?"gather_slab":"scatter_slab") : access ? ("tile_" + code).str() : code.str()));
      s.addAttributes(op->getAttrs());
      if (tma && op->hasAttr("block.page_extent"))
        s.addAttribute("ct_src.page_tma", b.getUnitAttr());
      if (tma && op->hasAttr("block.rectangular_tokens") &&
          op->getParentOp()->hasAttr("block.uniform_tokens"))
        s.addAttribute("ct_src.panel_tma", b.getUnitAttr());
      if (ownerRectangle(op, fn))
        s.addAttribute("ct_src.owner_rectangle", b.getUnitAttr());
      if(code=="scaled_contract"){
        s.addAttribute("ct_src.flat_groups",adjacentScaleGroups(op,fn));
      }
      if(code=="sum_columns"){
        auto extent=op->getAttrOfType<IntegerAttr>("extent").getInt();
        s.addAttribute("native_panel",b.getI64IntegerAttr(std::min<uint64_t>(128,llvm::PowerOf2Ceil(std::max<int64_t>(1,extent)))));
      }
      if(slab && rectangularSlab(op,fn))s.addAttribute("ct_src.rectangular",b.getUnitAttr());
      if (access && !slab) {
        auto view = op->getOperand(0);
        auto axes = intentdsl::block::projection(view.getType());
        auto *r = op->getOperand(1).getDefiningOp(),
             *c = op->getOperand(2).getDefiningOp();
        // normalized verified a rectangular active domain and both coordinates.
        s.addOperands({view, r->getOperand(0), c->getOperand(0)});
        s.addAttribute("allow_tma", b.getBoolAttr(tma && cast<MemRefType>(view.getType()).getRank() <= 2));
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
      auto *lowered=b.create(s);
      if(code=="scaled_contract")intentdsl::block::formScaleDispatch(*lowered,scaledMma?"mx":"bf16");
      op->replaceAllUsesWith(lowered);
      op->erase();
    });
    fn.walk([&](Operation *store){
      if(store->getName().getStringRef()!="ct_src.tile_store")return;
      auto *value=store->getOperand(3).getDefiningOp();
      if(!value || value->getName().getStringRef()!="ct_src.key_counts" || !value->hasOneUse())return;
      if(value->getAttrOfType<StringAttr>("bin_mode")!="global-atomic")return;
      OpBuilder b(store);
      auto span=value->getAttrOfType<IntegerAttr>("span").getInt();
      int64_t step=std::min<int64_t>(2048,uint64_t(span)&-uint64_t(span));
      OperationState s(store->getLoc(),"ct_src.private_bin_store");
      auto axes=store->getAttrOfType<DenseI64ArrayAttr>("block.member_axes");
      s.addAttribute("transposed",b.getBoolAttr(axes && axes[0]==1));
      s.addAttribute("source_step",b.getI64IntegerAttr(step));
      s.addOperands(value->getOperand(0));s.addOperands(store->getOperand(0));s.addOperands(value->getOperands().drop_front());
      s.addAttributes(value->getAttrs());s.addRegion();
      auto *guard=b.create(s);b.createBlock(&guard->getRegion(0));
      b.create(OperationState(store->getLoc(),"ct_src.zero_bins"));
      OperationState panels(store->getLoc(),"ct_src.bin_panels");panels.addRegion();
      auto *body=b.createBlock(&b.create(panels)->getRegion(0),{},b.getIndexType(),{store->getLoc()});
      OperationState read(store->getLoc(),"ct_src.read_bin_keys");
      read.addOperands(body->getArgument(0));read.addTypes(RankedTensorType::get({step},cast<MemRefType>(value->getOperand(0).getType()).getElementType()));
      auto keys=b.create(read)->getResult(0);
      OperationState update(store->getLoc(),"ct_src.atomic_bins"); update.addOperands({body->getArgument(0),keys});b.create(update);
      store->erase();value->erase();
    });
    partitionOwners(fn,facts);
  }
};
struct Signature : PassWrapper<Signature, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Signature)
  unsigned bits;
  explicit Signature(unsigned bits) : bits(bits) {}
  void runOnOperation() final {
    auto fn = *getOperation().getOps<func::FuncOp>().begin();
    Builder b(&getContext());
    getContext().loadDialect<arith::ArithDialect,scf::SCFDialect>();
    if(intentdsl::kir::orderedOwners(fn)) intentdsl::block::expandOwnerProgram(fn,"exits/cutile",nativeOwnerViews());
    else if(intentdsl::kir::matrixDomain(fn)){
      intentdsl::block::MatrixAccess p(fn);
      if(auto flat=p.dot.getAttrOfType<DenseI64ArrayAttr>("ct_src.flat_groups")){
        llvm::json::Object views;
        for(auto slot:flat.asArrayRef())
          views[std::to_string(slot)]=slot==p.slot(p.dot.getOperand(0))?llvm::json::Array{p.shape[0],p.shape[2]}:llvm::json::Array{p.shape[2],p.shape[1]};
        fn->setAttr("ct_src.native_views",b.getStringAttr(llvm::formatv("{0}",llvm::json::Value(std::move(views))).str()));
      }
      intentdsl::block::expandMatrixProgram(fn,"exits/cutile",scaledPanels);
    }
    else fn.walk<WalkOrder::PostOrder>([](Operation *op){
      if(op->getName().getDialectNamespace()!="ct_src") return;
      auto n=op->getName().stripDialect();
      if(n=="key_counts")op->setAttr("native_panel",Builder(op->getContext()).getI64IntegerAttr(64));
      if(n=="indexed_tile_read" || n=="fill_ranges_at" || n=="segment_fold" || n=="segment_size" || n=="scatter_fragment" ||
         n=="sum_columns" || n=="key_counts" || n=="publish_offsets")
        intentdsl::block::expandRowOperation(*op,"exits/cutile",false);
    });
    fn->setAttr("ct_src.entry", b.getStringAttr("kernel"));
    fn->setAttr("ct_src.index_bits", b.getI32IntegerAttr(bits));
    fn.walk([&](Operation *op){
      if(op->getName().getDialectNamespace()!="ct_src")return;
      auto n=op->getName().stripDialect();
      auto id=llvm::StringSwitch<StringRef>(n).Case("summary_reduce","moments-reduction").Case("uniform","random")
        .Case("scan_sum","inclusive-scan").Case("reduce","native-row-reduce").Default("");
      if(!id.empty())op->setAttr("ci.implementation",b.getStringAttr(("exits/cutile::"+id).str()));
      if(n=="summary_reduce" || n=="uniform")fn->setAttr(("ct_src.has_"+n).str(),b.getUnitAttr());
    });
    SmallVector<char> data;llvm::raw_svector_ostream os(data);
    if(auto err=describeArtifact(getOperation(),os)){fn.emitError()<<llvm::toString(std::move(err));signalPassFailure();return;}
    getOperation()->setAttr("ct_src.package",b.getStringAttr(os.str()));
  }
};
LogicalResult verify(ModuleOp module, const TargetProfile &profile,
                     bool callable) {
  auto matrix=*module.getOps<func::FuncOp>().begin();
  auto arrayCap=profile.services.get<intentdsl::cutile::ArrayIndices>();
  if(!arrayCap || matrix->hasAttr("ct_src.tma")!=arrayCap->permitTma)return matrix.emitError("invalid array transfer capability");
  if(callable){
    auto attached=module->getAttrOfType<StringAttr>("ct_src.package");
    if(!attached)return matrix.emitError("missing cuTile callable package");
  }
  if(matrix.front().front().getName().getStringRef()=="ct_src.tile_program"){
    if(!callable || failed(verifyArrayGeometry(matrix,profile)))return failure();
    return intentdsl::block::verifyTileProgram(matrix,profile,[](Operation &){return false;});
  }
  if (intentdsl::kir::orderedOwners(matrix)) {
    {
      auto &owner = matrix.front().front();
      auto rows = owner.getAttrOfType<IntegerAttr>("block.owner_tile");
      Attribute expected;
      if (owner.hasAttr("block.pair_axis") && rows && rows.getInt() < 16)
        expected = Builder(matrix.getContext()).getI64IntegerAttr(16);
      if (owner.getAttr("ct_src.matrix_rows") != expected)
        return matrix.emitError("invalid native matrix fragment height"), failure();
    }
    if (failed(intentdsl::block::verifyOwnerMap(matrix, profile, "ct_src",
                                                "realized")))
      return failure();
    auto result = matrix.walk([&](Operation *op) {
      auto cap = profile.services.get<intentdsl::cutile::ArrayIndices>();
      if (op->hasAttr("ct_src.panel_tma") !=
          (op->hasAttr("block.rectangular_tokens") && cap->permitTma &&
           op->getParentOp()->hasAttr("block.uniform_tokens"))) {
        op->emitError(
            "rectangular transfer requires a compatible array layout");
        return WalkResult::interrupt();
      }
      if (op->hasAttr("ct_src.page_tma") !=
          (op->hasAttr("block.page_extent") && cap->permitTma)) {
        op->emitError("page transfer requires bounded page coordinates and a "
                      "compatible array layout");
        return WalkResult::interrupt();
      }
      if (op->hasAttr("ct_src.owner_rectangle") != ownerRectangle(op, matrix)) {
        op->emitError("owner rectangle requires an aligned full group and "
                      "contiguous logical axes");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return success(!result.wasInterrupted());
  }
  if(intentdsl::kir::matrixDomain(matrix)){
    if(failed(intentdsl::block::verifyScaleDispatch(matrix,profile,"ct_src",profile.services.get<intentdsl::cutile::ArrayIndices>()->scaledMma?"mx":"bf16")))return failure();
    if(callable && matrix->getAttr("ct_src.index_bits")!=Builder(matrix.getContext()).getI32IntegerAttr(profile.services.get<intentdsl::cutile::ArrayIndices>()->bits))
      return matrix.emitError("matrix index width disagrees with the target");
    return success(!matrix.walk([&](Operation *op){
      return op->getName().getStringRef()=="ct_src.scaled_contract" &&
        op->getAttr("ct_src.flat_groups")!=adjacentScaleGroups(op,matrix)?
        WalkResult::interrupt():WalkResult::advance();}).wasInterrupted());
  }
  if (failed(intentdsl::verifyMapping(module, profile, "ct_src")))
    return failure();
  auto fn = *module.getOps<func::FuncOp>().begin();
  unsigned branch=0;
  llvm::SmallDenseSet<Value> published;
  for(auto &op:fn.front())if(op.getName().getStringRef()=="ct_src.output_partition"){
    if(op.getAttrOfType<IntegerAttr>("index").getInt()!=branch++ || !llvm::hasSingleElement(op.getRegion(0)))
      return op.emitError("expected distinct contiguous output partitions");
    auto &bb=op.getRegion(0).front();
    if(bb.empty() || bb.back().getName().getStringRef()!="ct_src.partition_end")
      return op.emitError("unterminated output partition");
    unsigned writes=0;
    auto checked=op.walk([&](Operation *inner){
      auto n=inner->getName().stripDialect();
      if(n=="tile_store" || n=="scatter_slab")return WalkResult::interrupt();
      if(n=="owner_store"){
        if(inner->getBlock()!=&bb || !published.insert(inner->getOperand(0)).second)return WalkResult::interrupt();
        ++writes;
      }
      return WalkResult::advance();
    });
    if(checked.wasInterrupted() || writes!=1)
      return op.emitError("expected one owner publication per partition");
  }
  if(branch)for(auto &op:fn.front())
    if(op.getName().stripDialect()!="output_partition" && op.getName().stripDialect()!="identity" &&
       op.getName().stripDialect()!="rows" && op.getName().stripDialect()!="pid" &&
       op.getName().stripDialect()!="return")return op.emitError("unguarded work in partitioned grid");
  if(failed(intentdsl::specializeProjectedLayout(fn,*profile.services.get<intentdsl::CallFacts>(),"ct_src",false)))return failure();
  if (fn.empty() || !llvm::hasSingleElement(fn.getBody()) || fn.getNumResults())
    return fn.emitError("expected a defined output-buffer kernel");
  auto abi = fn->getAttrOfType<StringAttr>("kir.abi");
  if (!abi)
    return fn.emitError("missing cuTile argument slots");
  auto signature = llvm::json::parse(abi.getValue());
  if (!signature) {
    llvm::consumeError(signature.takeError());
    return fn.emitError("malformed cuTile argument slots");
  }
  auto *slots = signature->getAsArray();
  if (!slots || slots->size() != fn.getNumArguments())
    return fn.emitError("cuTile argument count mismatch");
  for (auto [arg, slot] : llvm::zip(fn.getArguments(), *slots)) {
    auto *record = slot.getAsObject();
    auto view = dyn_cast<MemRefType>(arg.getType());
    Type scalar = view ? view.getElementType() : arg.getType();
    if (!record || !record->getString("name") || !record->getArray("dims") ||
        record->getArray("dims")->size() !=
            unsigned(view ? view.getRank() : 0) ||
        (!scalar.isF16() && !scalar.isF32() && !scalar.isBF16() &&
         !scalar.isInteger(32) && !(view && scalar.isSignlessInteger(64)) && (view || !scalar.isUnsignedInteger(64))) ||
        record->getString("dtype") != (scalar.isF16()         ? "f16"
                                       : scalar.isBF16()      ? "bf16"
                                       : scalar.isInteger(32) ? "i32"
                                       : scalar.isSignlessInteger(64) ? "i64"
                                       : scalar.isUnsignedInteger(64) ? "u64"
                                                              : "f32") ||
        (view ? (record->getString("direction") != "In" &&
                 record->getString("direction") != "Out" &&
                 record->getString("direction") != "InOut")
              : record->getString("direction") != "Scalar"))
      return fn.emitError(
          "cuTile array or scalar slot does not match function type");
  }
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
  if(callable && failed(intentdsl::block::verifyRowProgram(fn,"ct_src","ct_src.has_summary_reduce")))return failure();
  auto res = module.walk([&](Operation *op) {
    if(callable && isa<arith::ConstantOp,arith::IndexCastOp,scf::ForOp,scf::YieldOp>(op))return WalkResult::advance();
    if (op == module || op->getName().getDialectNamespace() == "func")
      return WalkResult::advance();
    auto n = op->getName().stripDialect();
    bool ok = op->getName().getDialectNamespace() == "ct_src";
    if(ok && n=="private_bin_store"){
      auto src=dyn_cast<BlockArgument>(op->getOperand(0)),dst=dyn_cast<BlockArgument>(op->getOperand(1));
      auto *mask=op->getOperand(4).getDefiningOp();
      ok=src && dst && src.getOwner()==&fn.front() && dst.getOwner()==&fn.front() &&
         op->getAttrOfType<StringAttr>("bin_mode")=="global-atomic" &&
         (*slots)[src.getArgNumber()].getAsObject()->getString("direction")=="In" &&
         (*slots)[dst.getArgNumber()].getAsObject()->getString("direction")=="Out" &&
         cast<MemRefType>(src.getType()).getRank()==1 && cast<MemRefType>(dst.getType()).getRank()==2 &&
         cast<MemRefType>(dst.getType()).getElementType().isInteger(32) && mask && mask->getName().stripDialect()=="active" &&
         mask->getOperand(1)==op->getOperand(2) && mask->getOperand(2)==op->getOperand(3) &&
         op->getAttrOfType<IntegerAttr>("span").getInt()>0 && fn->getAttrOfType<StringAttr>("block.topology")=="resident" &&
         llvm::all_of(fn->getAttrOfType<ArrayAttr>("block.configs"),[](Attribute row){return cast<DictionaryAttr>(row).getAs<IntegerAttr>("rows").getInt()==1;});
      auto step=op->getAttrOfType<IntegerAttr>("source_step");
      if(!llvm::hasSingleElement(op->getRegion(0)))return WalkResult::interrupt();
      auto &guard=op->getRegion(0).front();
      if(guard.getOperations().size()!=2 || guard.front().getName().getStringRef()!="ct_src.zero_bins" ||
         guard.back().getName().getStringRef()!="ct_src.bin_panels" || !llvm::hasSingleElement(guard.back().getRegion(0)))
        return op->emitError("expected guarded zero then count panels"),WalkResult::interrupt();
      auto &loop=guard.back().getRegion(0).front();
      if(!step || loop.getNumArguments()!=1 || !loop.getArgument(0).getType().isIndex() || loop.getOperations().size()!=2 ||
         loop.front().getName().getStringRef()!="ct_src.read_bin_keys" || loop.back().getName().getStringRef()!="ct_src.atomic_bins" ||
         loop.front().getOperand(0)!=loop.getArgument(0) || loop.back().getOperand(0)!=loop.getArgument(0) ||
         loop.back().getOperand(1)!=loop.front().getResult(0) || !src ||
         loop.front().getResult(0).getType()!=RankedTensorType::get({step.getInt()},cast<MemRefType>(src.getType()).getElementType()))
        return op->emitError("invalid count panel or atomic update"),WalkResult::interrupt();
      ok=ok && step && step.getInt()>0 && step.getInt()<=2048 && llvm::isPowerOf2_64(step.getInt()) && op->getAttrOfType<IntegerAttr>("span").getInt()%step.getInt()==0;
    }
    if(ok && (n=="gather_slab" || n=="scatter_slab"))
      ok=op->hasAttr("ct_src.rectangular")==rectangularSlab(op,fn);
    if (ok && (n == "tile_store" || n == "tile_load")) {
      auto arg = dyn_cast<BlockArgument>(op->getOperand(0));
      if (!arg || arg.getOwner() != &fn.front() ||
          ((*slots)[arg.getArgNumber()].getAsObject()->getString("direction") !=
               "InOut" &&
           (*slots)[arg.getArgNumber()].getAsObject()->getString("direction") !=
               (n == "tile_load" ? "In" : "Out"))) {
        op->emitError("tile access conflicts with public buffer direction");
        return WalkResult::interrupt();
      }
      auto axes = intentdsl::block::projection(op->getOperand(0).getType());
      auto *pid = op->getOperand(1).getDefiningOp(),
           *col = op->getOperand(2).getDefiningOp();
      auto ind = dyn_cast<BlockArgument>(op->getOperand(2));
      ok = op->getAttrOfType<BoolAttr>("allow_tma").getValue() ==
               (indices->permitTma && cast<MemRefType>(arg.getType()).getRank() <= 2) &&
           succeeded(axes) &&
           op->getAttrOfType<BoolAttr>("matrix").getValue() ==
               (*axes == intentdsl::block::RowProjection::Matrix) &&
           (n != "tile_store" ||
            *axes == intentdsl::block::RowProjection::Matrix ||
            (cast<MemRefType>(fn.getArgument(fn->getAttrOfType<IntegerAttr>(
                                                   "kir.domain_arg")
                                                 .getInt())
                                  .getType())
                     .getRank() == 1 &&
             op->getAttrOfType<DenseI64ArrayAttr>("block.member_axes") ==
                 DenseI64ArrayAttr::get(fn.getContext(), {0}))) &&
           pid && pid->getName().getStringRef() == "ct_src.pid" &&
           ((col && col->getName().getStringRef() == "ct_src.zero") ||
            (col && col->getName().getStringRef()=="ct_src.pid" &&
             col->getAttrOfType<IntegerAttr>("axis").getInt()==1 &&
             pid->getAttrOfType<IntegerAttr>("axis").getInt()==0 &&
             isa<ArrayAttr>(fn->getAttrOfType<ArrayAttr>("block.grid")[1])) ||
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
void matrixIssues(Unit &);
void bind(Unit &unit) {
  matrixIssues(unit);
  registerContracts(unit);
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
              pm.addPass(std::make_unique<Lower>(*indices,*p.services.get<intentdsl::CallFacts>()));
            }
            return llvm::Error::success();
          },
          [callable](ModuleOp m, const TargetProfile &p) {
            return verify(m, p, callable);
          }},
         COMPILERINFRA_SOURCE});
  ExitImplementation exit;exit.capabilities=capabilities;exit.terminal=cutilePrinter();
  unit.exit=Located<ExitImplementation>{std::move(exit),COMPILERINFRA_SOURCE};
}
} // namespace CI_UNIT_NAMESPACE
