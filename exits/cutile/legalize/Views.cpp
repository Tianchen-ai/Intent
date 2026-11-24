#include "intentdsl/levels/execution/block/ir/RowProgram.h"
#include "intentdsl/levels/execution/block/ir/MatrixProgram.h"
#include "intentdsl/levels/execution/block/ir/OwnerProgram.h"
#include "../capabilities/Index.h"
#include "intentdsl/levels/execution/block/ir/ScaleDispatch.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/KeyReduction.h"
#include "intentdsl/levels/algorithm/kir/ir/ProjectedLayout.h"
#include "intentdsl/levels/execution/block/decisions/Verify.h"
#include "intentdsl/levels/execution/block/ir/Access.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <set>

#include "Dialect.h.inc"
namespace CI_UNIT_NAMESPACE {
mlir::Value scaledPanels(intentdsl::block::Tiles &,intentdsl::block::MatrixAccess &);
using namespace mlir;
using namespace compilerinfra;
void registerIR(DialectRegistry &);
llvm::Expected<TargetProfile> capabilities(const llvm::json::Object &);
llvm::Error describeArtifact(ModuleOp, llvm::raw_pwrite_stream &);
TerminalPrinter cutilePrinter();
void registerContracts(Unit &);
Value nativeMatrixRead(intentdsl::block::Tiles &,
                       intentdsl::block::MatrixAccess &, bool, Value, Value);
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
bool broadcastInner(Value viewArg, func::FuncOp fn) {
  auto arg = dyn_cast<BlockArgument>(viewArg);
  if (!arg || arg.getOwner() != &fn.front())
    return false;
  auto strides =
      fn.getArgAttrOfType<DenseI64ArrayAttr>(arg.getArgNumber(), "ct_src.strides");
  auto shape =
      fn.getArgAttrOfType<DenseI64ArrayAttr>(arg.getArgNumber(), "ct_src.shape");
  return strides && shape && !strides.empty() &&
         strides.size() == shape.size() && strides.asArrayRef().back() == 0 &&
         shape.asArrayRef().back() > 0;
}
bool alignedProjectedView(Value viewArg, func::FuncOp fn,
                          const intentdsl::CallFacts &facts, bool permitTma) {
  if (!permitTma)
    return false;
  auto arg = dyn_cast<BlockArgument>(viewArg);
  if (!arg || arg.getOwner() != &fn.front())
    return false;
  auto view = cast<MemRefType>(arg.getType());
  int64_t elemBytes = llvm::divideCeil(view.getElementTypeBitWidth(), 8);
  unsigned slot = arg.getArgNumber();
  if (auto map = fn.getArgAttrOfType<AffineMapAttr>(slot, "kir.indexing")) {
    auto strides = fn.getArgAttrOfType<DenseI64ArrayAttr>(slot, "ct_src.strides");
    auto shape = fn.getArgAttrOfType<DenseI64ArrayAttr>(slot, "ct_src.shape");
    auto src = fn.getArgAttrOfType<IntegerAttr>(slot, "kir.source");
    if (!strides || !shape || strides.size() != size_t(view.getRank()) ||
        shape.size() != strides.size() || strides.empty() ||
        strides.asArrayRef().back() != 1 || !src || src.getInt() < 0 ||
        size_t(src.getInt()) >= facts.arguments.size())
      return false;
    for (int64_t st : strides.asArrayRef().drop_back())
      if (st <= 0 || (st * elemBytes) % 16 != 0)
        return false;
    auto *srcObj = facts.arguments[src.getInt()].getAsObject();
    auto *srcStrides = srcObj ? srcObj->getArray("strides") : nullptr;
    if (!srcStrides || srcStrides->size() != map.getValue().getNumResults())
      return false;
    int64_t offsetElems = 0;
    for (auto [i, expr] : llvm::enumerate(map.getValue().getResults())) {
      SmallVector<int64_t> factors(map.getValue().getNumDims() + 1, 0);
      auto st = (*srcStrides)[i].getAsInteger();
      if (!st || !intentdsl::coefficients(expr, factors))
        return false;
      int64_t term = 0;
      if (__builtin_mul_overflow(factors.back(), *st, &term) ||
          __builtin_add_overflow(offsetElems, term, &offsetElems))
        return false;
    }
    return offsetElems >= 0 && (offsetElems * elemBytes) % 16 == 0;
  }
  if (slot >= facts.arguments.size())
    return false;
  auto *obj = facts.arguments[slot].getAsObject();
  auto *strides = obj ? obj->getArray("strides") : nullptr;
  if (!strides || strides->size() != size_t(view.getRank()) || strides->empty())
    return false;
  if ((*strides)[strides->size() - 1].getAsInteger() != 1)
    return false;
  for (size_t i = 0; i + 1 < strides->size(); ++i) {
    auto st = (*strides)[i].getAsInteger();
    if (!st || *st <= 0 || (*st * elemBytes) % 16 != 0)
      return false;
  }
  return true;
}
bool tileLoadStorePermitsTma(Value viewArg, func::FuncOp fn,
                             const intentdsl::CallFacts &facts, bool permitTma) {
  if (!permitTma)
    return false;
  auto arg = dyn_cast<BlockArgument>(viewArg);
  if (!arg || arg.getOwner() != &fn.front())
    return false;
  int rank = cast<MemRefType>(arg.getType()).getRank();
  unsigned slot = arg.getArgNumber();
  if (rank <= 2 && !fn.getArgAttr(slot, "kir.indexing"))
    return true;
  if (!alignedProjectedView(viewArg, fn, facts, permitTma))
    return false;
  if (rank > 2) {
    std::optional<int64_t> rowDim;
    if (auto shape = fn.getArgAttrOfType<DenseI64ArrayAttr>(slot, "ct_src.shape");
        shape && shape.size() == size_t(rank))
      rowDim = shape[rank - 2];
    else if (auto *sh = facts.shapeOf(arg); sh && sh->size() == size_t(rank))
      rowDim = (*sh)[rank - 2].getAsInteger();
    auto configs = fn->getAttrOfType<ArrayAttr>("block.configs");
    if (!rowDim || *rowDim <= 0 || !configs)
      return false;
    for (Attribute row : configs)
      if (*rowDim %
              cast<DictionaryAttr>(row).getAs<IntegerAttr>("rows").getInt() !=
          0)
        return false;
  }
  return true;
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
      if (code == "stripe_load" || code == "stripe_store") {
        auto view = op->getOperand(0);
        int rank = cast<MemRefType>(view.getType()).getRank();
        if (code == "stripe_load" && rank == 1) {
          OperationState so(op->getLoc(), "ct_src.stripe_owner_load");
          so.addOperands({view, op->getOperand(1), op->getOperand(2)});
          so.addAttribute("stride", op->getAttr("stride"));
          so.addTypes(op->getResultTypes());
          op->replaceAllUsesWith(b.create(so));
        } else {
          auto *r = op->getOperand(2).getDefiningOp(),
               *c = op->getOperand(3).getDefiningOp();
          OperationState st(op->getLoc(), code == "stripe_load"
                                              ? "ct_src.stripe_tile_load"
                                              : "ct_src.stripe_tile_store");
          st.addOperands(
              {view, op->getOperand(1), r->getOperand(0), c->getOperand(0)});
          if (code == "stripe_store")
            st.addOperands(op->getOperand(5));
          st.addAttribute("stride", op->getAttr("stride"));
          st.addAttribute("allow_tma", b.getBoolAttr(tma));
          st.addTypes(op->getResultTypes());
          op->replaceAllUsesWith(b.create(st));
        }
        op->erase();
        return;
      }
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
      if (slab && rectangularSlab(op, fn)) {
        s.addAttribute("ct_src.rectangular", b.getUnitAttr());
        bool bcast = code == "load" && broadcastInner(op->getOperand(0), fn);
        if (bcast)
          s.addAttribute("ct_src.broadcast_inner", b.getUnitAttr());
        else if (alignedProjectedView(op->getOperand(0), fn, facts, tma))
          s.addAttribute("ct_src.slab_tma", b.getUnitAttr());
      }
      if (access && !slab) {
        auto view = op->getOperand(0);
        auto axes = intentdsl::block::projection(view.getType());
        auto *r = op->getOperand(1).getDefiningOp(),
             *c = op->getOperand(2).getDefiningOp();
        bool bcast = code == "load" && broadcastInner(view, fn);
        if (bcast)
          s.addAttribute("ct_src.broadcast_inner", b.getUnitAttr());
        // normalized verified a rectangular active domain and both coordinates.
        s.addOperands({view, r->getOperand(0), c->getOperand(0)});
        s.addAttribute(
            "allow_tma",
            b.getBoolAttr(!bcast &&
                          tileLoadStorePermitsTma(view, fn, facts, tma)));
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
      Location loc = store->getLoc();
      auto extent = value->getAttrOfType<IntegerAttr>("extent").getInt();
      auto span = value->getAttrOfType<IntegerAttr>("span").getInt();
      int64_t step = std::min<int64_t>(2048, uint64_t(span) & -uint64_t(span));
      int64_t panels = span / step;
      auto axes = store->getAttrOfType<DenseI64ArrayAttr>("block.member_axes");
      bool swap = axes && axes[0] == 1;
      Value src = value->getOperand(0), dst = store->getOperand(0);
      auto cInt = [&](int64_t v, Type ty) -> Value {
        OperationState cs(loc, "ct_src.constant");
        cs.addAttribute("value", b.getIntegerAttr(ty, v));
        cs.addTypes(ty);
        return b.create(cs)->getResult(0);
      };
      auto rowMap = [&](StringRef kind, ValueRange args, Type resTy) -> Value {
        OperationState ms(loc, "ct_src.map");
        ms.addOperands(args);
        ms.addAttribute("kind", b.getStringAttr(kind));
        ms.addTypes(resTy);
        return b.create(ms)->getResult(0);
      };
      OperationState dimSt(loc, "ct_src.array_dim");
      dimSt.addOperands(dst);
      dimSt.addAttribute("axis", b.getI64IntegerAttr(swap ? 0 : 1));
      dimSt.addTypes(b.getI64Type());
      Value bins = b.create(dimSt)->getResult(0);
      Value c0 = cInt(0, b.getI64Type());
      Value hasBins = rowMap("lt", {c0, bins}, b.getI1Type());

      OperationState s(loc, "ct_src.private_bin_store");
      s.addAttribute("transposed", b.getBoolAttr(swap));
      s.addAttribute("source_step", b.getI64IntegerAttr(step));
      s.addOperands(src);
      s.addOperands(dst);
      s.addOperands(value->getOperands().drop_front());
      s.addOperands(hasBins);
      s.addAttributes(value->getAttrs());
      s.addRegion();
      auto *guard = b.create(s);
      b.setInsertionPointToStart(b.createBlock(&guard->getRegion(0)));

      OperationState zb(loc, "ct_src.zero_bins");
      zb.addOperands(dst);
      zb.addAttribute("transposed", b.getBoolAttr(swap));
      b.create(zb);

      OperationState pidSt(loc, "ct_src.tile_pid");
      pidSt.addAttribute("axis", b.getI64IntegerAttr(0));
      pidSt.addTypes(b.getI64Type());
      Value pid = b.create(pidSt)->getResult(0);
      Value cPanels = cInt(panels, b.getI64Type());
      Value cSpan = cInt(span, b.getI64Type());
      Value cExt = cInt(extent, b.getI64Type());
      Value cStep = cInt(step, b.getI64Type());
      Value limit;
      if (extent % span == 0) {
        limit = cInt(panels, b.getI32Type());
      } else {
        Value partBase = rowMap("mul", {pid, cSpan}, b.getI64Type());
        Value rem = rowMap("sub", {cExt, partBase}, b.getI64Type());
        Value remPanels = rowMap("ceildiv", {rem, cStep}, b.getI64Type());
        Value clamped = rowMap("min", {cPanels, remPanels}, b.getI64Type());
        limit = rowMap("i32", {clamped}, b.getI32Type());
      }

      OperationState panelsSt(loc, "ct_src.bin_panels");
      panelsSt.addOperands(limit);
      panelsSt.addRegion();
      auto *panelsOp = b.create(panelsSt);
      auto *body = b.createBlock(&panelsOp->getRegion(0), {}, b.getI32Type(),
                                 {loc});
      b.setInsertionPointToStart(body);
      Value iv = body->getArgument(0);
      Value iv64 = rowMap("i64", {iv}, b.getI64Type());
      Value panelBase = rowMap("mul", {pid, cPanels}, b.getI64Type());
      Value panelIdx = rowMap("add", {panelBase, iv64}, b.getI64Type());

      Type srcElemTy = cast<MemRefType>(src.getType()).getElementType();
      OperationState read(loc, "ct_src.read_bin_keys");
      read.addOperands({src, panelIdx});
      read.addTypes(RankedTensorType::get({step}, srcElemTy));
      Value raw = b.create(read)->getResult(0);
      Type keyElemTy = srcElemTy.isInteger(64) ? b.getI64Type() : b.getI32Type();
      auto keyVecTy = RankedTensorType::get({step}, keyElemTy);
      auto i64VecTy = RankedTensorType::get({step}, b.getI64Type());
      auto i32VecTy = RankedTensorType::get({step}, b.getI32Type());
      auto i1VecTy = RankedTensorType::get({step}, b.getI1Type());
      OperationState convert(loc, "ct_src.bin_index");
      convert.addOperands(raw);
      convert.addTypes(keyVecTy);
      Value keys = b.create(convert)->getResult(0);

      Value partOff = rowMap("mul", {pid, cSpan}, b.getI64Type());
      Value stepOff = rowMap("mul", {iv64, cStep}, b.getI64Type());
      Value basePos = rowMap("add", {partOff, stepOff}, b.getI64Type());
      OperationState rangeSt(loc, "ct_src.range");
      rangeSt.addAttribute("size", b.getI64IntegerAttr(step));
      rangeSt.addTypes(i64VecTy);
      Value lanes = b.create(rangeSt)->getResult(0);
      Value pos = rowMap("add", {basePos, lanes}, i64VecTy);
      Value inExtent = rowMap("lt", {pos, cExt}, i1VecTy);
      Value cZeroKey = cInt(0, keyElemTy);
      Value ge0 = rowMap("ge", {keys, cZeroKey}, i1VecTy);
      Value binsKey =
          keyElemTy.isInteger(64) ? bins : rowMap("i32", {bins}, b.getI32Type());
      Value ltBins = rowMap("lt", {keys, binsKey}, i1VecTy);
      Value inBins = rowMap("and", {ge0, ltBins}, i1VecTy);
      Value valid = rowMap("and", {inExtent, inBins}, i1VecTy);
      Value safeKey = rowMap("select", {valid, keys, cZeroKey}, keyVecTy);
      Value inc = rowMap("i32", {valid}, i32VecTy);

      OperationState update(loc, "ct_src.atomic_bins");
      update.addOperands({dst, safeKey, inc});
      update.addAttribute("transposed", b.getBoolAttr(swap));
      update.addAttribute("scope", b.getStringAttr("workgroup"));
      b.create(update);

      store->erase();
      value->erase();
    });
  }
};
struct Signature : PassWrapper<Signature, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Signature)
  unsigned bits;
  TargetProfile profile;
  Signature(unsigned bits, const TargetProfile &p) : bits(bits), profile(p) {}
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
      intentdsl::block::expandMatrixProgram(fn, "exits/cutile", scaledPanels,
                                            nativeMatrixRead);
    }
    else fn.walk<WalkOrder::PostOrder>([](Operation *op){
      if(op->getName().getDialectNamespace()!="ct_src") return;
      auto n=op->getName().stripDialect();
      if(n=="key_counts")op->setAttr("native_panel",Builder(op->getContext()).getI64IntegerAttr(64));
      if(n=="indexed_tile_read" || n=="fill_ranges_at" || n=="segment_fold" || n=="segment_size" || n=="scatter_fragment" ||
         n=="sum_columns" || n=="key_counts" || n=="publish_offsets" || n=="stripe_loop")
        intentdsl::block::expandRowOperation(*op,"exits/cutile",false);
    });
    fn->setAttr("ct_src.entry", b.getStringAttr("kernel"));
    fn->setAttr("ct_src.index_bits", b.getI32IntegerAttr(bits));
    const auto *facts = profile.services.get<intentdsl::CallFacts>();
    fn.walk([&](Operation *op){
      if(op->getName().getDialectNamespace()!="ct_src")return;
      auto n=op->getName().stripDialect();
      if (n == "uniform") {
        bool quad = intentdsl::cutile::quadAlignedUniform(op, fn, facts);
        if (quad)
          op->setAttr("quad_aligned", b.getUnitAttr());
        op->setAttr("ci.implementation",
                    b.getStringAttr(quad ? "exits/cutile::random-quad"
                                         : "exits/cutile::random"));
        return;
      }
      auto id=llvm::StringSwitch<StringRef>(n).Case("summary_reduce","moments-reduction").Case("summary_merge","moments-merge")
        .Case("scan_sum","inclusive-scan").Case("reduce","native-row-reduce").Default("");
      if(!id.empty())op->setAttr("ci.implementation",b.getStringAttr(("exits/cutile::"+id).str()));
    });
    std::set<std::string> seenResources;
    SmallVector<std::string> runtimeImpls;
    fn.walk([&](Operation *op) {
      if (auto attr = op->getAttrOfType<StringAttr>("ci.implementation")) {
        auto key = attr.getValue().str();
        auto it = profile.implementations.find(key);
        if (it != profile.implementations.end() &&
            it->second.source == ImplementationSource::Runtime &&
            seenResources.insert(it->second.resource).second)
          runtimeImpls.push_back(key);
      }
    });
    OpBuilder mb(fn);
    for (auto &impl : runtimeImpls) {
      OperationState st(fn.getLoc(), "ct_src.runtime_source");
      st.addAttribute("implementation", mb.getStringAttr(impl));
      mb.create(st);
    }
    SmallVector<char> data;llvm::raw_svector_ostream os(data);
    if(auto err=describeArtifact(getOperation(),os)){fn.emitError()<<llvm::toString(std::move(err));signalPassFailure();return;}
    getOperation()->setAttr("ct_src.package",b.getStringAttr(os.str()));
  }
};
LogicalResult verify(ModuleOp module, const TargetProfile &profile,
                     bool callable) {
  auto funcs = module.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs))
    return failure();
  auto matrix=*funcs.begin();
  if (matrix.empty() || matrix.front().empty())
    return failure();
  if (!callable) {
    if (!llvm::hasSingleElement(module.getBody()->getOperations()))
      return failure();
  } else {
    auto attached=module->getAttrOfType<StringAttr>("ct_src.package");
    if(!attached)return matrix.emitError("missing cuTile callable package");
    if (&module.getBody()->back() != matrix.getOperation())
      return matrix.emitError("expected function after runtime_source declarations");
    std::set<std::string> declaredResources;
    for (auto &top : module.getBody()->without_terminator()) {
      if (&top == matrix.getOperation())
        break;
      if (top.getName().getStringRef() != "ct_src.runtime_source")
        return top.emitError("unexpected module-level operation");
      auto impl = top.getAttrOfType<StringAttr>("implementation");
      if (!impl)
        return top.emitError("missing runtime_source implementation");
      if (auto err = profile.checkImplementation(impl.getValue()))
        return top.emitError() << llvm::toString(std::move(err));
      auto &c = profile.implementations.at(impl.getValue().str());
      if (c.source != ImplementationSource::Runtime ||
          !declaredResources.insert(c.resource).second)
        return top.emitError("invalid or duplicate runtime_source");
    }
    std::set<std::string> usedResources;
    matrix.walk([&](Operation *op) {
      if (auto attr = op->getAttrOfType<StringAttr>("ci.implementation")) {
        auto it = profile.implementations.find(attr.getValue().str());
        if (it != profile.implementations.end() &&
            it->second.source == ImplementationSource::Runtime)
          usedResources.insert(it->second.resource);
      }
    });
    if (declaredResources != usedResources)
      return matrix.emitError("runtime_source declarations disagree with bound operations");
  }
  auto arrayCap=profile.services.get<intentdsl::cutile::ArrayIndices>();
  if(!arrayCap || matrix->hasAttr("ct_src.tma")!=arrayCap->permitTma)return matrix.emitError("invalid array transfer capability");
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
        (o->getInteger("occupancy") != 1 && o->getInteger("occupancy") != 2 &&
         o->getInteger("occupancy") != 4))
      return fn.emitError("invalid native hints for row");
  }
  if(callable && failed(intentdsl::block::verifyRowProgram(fn,"ct_src")))return failure();
  const auto *factsPtr = profile.services.get<intentdsl::CallFacts>();
  auto res = module.walk([&](Operation *op) {
    if(callable && isa<arith::ConstantOp,arith::IndexCastOp,scf::ForOp,scf::YieldOp>(op))return WalkResult::advance();
    if (op == module || op->getName().getDialectNamespace() == "func" ||
        (callable && op->getName().getStringRef() == "ct_src.runtime_source"))
      return WalkResult::advance();
    auto n = op->getName().stripDialect();
    bool ok = op->getName().getDialectNamespace() == "ct_src";
    if (ok && n == "uniform") {
      bool expectedQuad =
          callable && intentdsl::cutile::quadAlignedUniform(op, fn, factsPtr);
      if (op->hasAttr("quad_aligned") != expectedQuad)
        return op->emitError("uniform quad alignment disagrees with row layout"),
               WalkResult::interrupt();
      if (callable &&
          op->getAttrOfType<StringAttr>("ci.implementation") !=
              (expectedQuad ? "exits/cutile::random-quad"
                            : "exits/cutile::random"))
        return op->emitError("uniform implementation contract mismatch"),
               WalkResult::interrupt();
    }
    if (ok && n == "array_dim") {
      auto arr = dyn_cast<BlockArgument>(op->getOperand(0));
      auto ax = op->getAttrOfType<IntegerAttr>("axis");
      ok = arr && arr.getOwner() == &fn.front() && isa<MemRefType>(arr.getType()) &&
           ax && ax.getInt() >= 0 && ax.getInt() < cast<MemRefType>(arr.getType()).getRank();
      if (!ok)
        return op->emitError("invalid array_dim operand or axis"), WalkResult::interrupt();
    }
    if (ok && (n=="zero_bins" || n=="bin_panels" || n=="read_bin_keys" || n=="bin_index" || n=="atomic_bins")) {
      auto parent=op->getParentOp();
      StringRef required=n=="zero_bins" || n=="bin_panels"?"ct_src.private_bin_store":"ct_src.bin_panels";
      if(!parent || parent->getName().getStringRef()!=required)
        return op->emitError("histogram operation requires its enclosing count region"),WalkResult::interrupt();
    }
    if(ok && n=="private_bin_store"){
      auto src=dyn_cast<BlockArgument>(op->getOperand(0)),dst=dyn_cast<BlockArgument>(op->getOperand(1));
      auto *mask=op->getOperand(4).getDefiningOp();
      auto actualExtent = intentdsl::kir::keyExtent(
          op, *profile.services.get<intentdsl::CallFacts>());
      if (!actualExtent ||
          op->getAttrOfType<IntegerAttr>("extent").getInt() != *actualExtent)
        return op->emitError(
                   "private_bin_store extent disagrees with source view"),
               WalkResult::interrupt();
      int64_t extentVal = *actualExtent;
      int64_t spanVal = op->getAttrOfType<IntegerAttr>("span").getInt();
      bool swapVal = op->getAttrOfType<BoolAttr>("transposed").getValue();
      auto isMap = [](Value v, StringRef kind) -> Operation * {
        auto *d = v.getDefiningOp();
        return d && d->getName().getStringRef() == "ct_src.map" &&
                       d->getAttrOfType<StringAttr>("kind") == kind
                   ? d
                   : nullptr;
      };
      auto isConst = [](Value v, int64_t expected) -> bool {
        auto *d = v.getDefiningOp();
        if (!d || d->getName().getStringRef() != "ct_src.constant")
          return false;
        auto a = d->getAttrOfType<IntegerAttr>("value");
        return a && a.getValue().getSExtValue() == expected;
      };
      auto *nonEmpty = isMap(op->getOperand(5), "lt");
      auto *binsDef = nonEmpty ? nonEmpty->getOperand(1).getDefiningOp() : nullptr;
      ok=src && dst && src.getOwner()==&fn.front() && dst.getOwner()==&fn.front() &&
         op->getAttrOfType<StringAttr>("bin_mode")=="global-atomic" &&
         (*slots)[src.getArgNumber()].getAsObject()->getString("direction")=="In" &&
         (*slots)[dst.getArgNumber()].getAsObject()->getString("direction")=="Out" &&
         cast<MemRefType>(src.getType()).getRank()==1 && cast<MemRefType>(dst.getType()).getRank()==2 &&
         cast<MemRefType>(dst.getType()).getElementType().isInteger(32) && mask && mask->getName().stripDialect()=="active" &&
         mask->getOperand(1)==op->getOperand(2) && mask->getOperand(2)==op->getOperand(3) &&
         nonEmpty && isConst(nonEmpty->getOperand(0), 0) && binsDef &&
         binsDef->getName().getStringRef() == "ct_src.array_dim" &&
         binsDef->getOperand(0) == dst &&
         binsDef->getAttrOfType<IntegerAttr>("axis").getInt() == (swapVal ? 0 : 1) &&
         spanVal>0 && fn->getAttrOfType<StringAttr>("block.topology")=="resident" &&
         llvm::all_of(fn->getAttrOfType<ArrayAttr>("block.configs"),[](Attribute row){return cast<DictionaryAttr>(row).getAs<IntegerAttr>("rows").getInt()==1;});
      auto step=op->getAttrOfType<IntegerAttr>("source_step");
      if(!step || step.getInt()<=0 || step.getInt()>2048 || !llvm::isPowerOf2_64(step.getInt()) || spanVal%step.getInt()!=0 || !llvm::hasSingleElement(op->getRegion(0)))
        return op->emitError("invalid private_bin_store step or region"),WalkResult::interrupt();
      int64_t stepVal = step.getInt(), panelsVal = spanVal / stepVal;
      auto &guard=op->getRegion(0).front();
      if(guard.empty() || guard.front().getName().getStringRef()!="ct_src.zero_bins" ||
         guard.front().getOperand(0)!=dst ||
         guard.front().getAttrOfType<BoolAttr>("transposed").getValue()!=swapVal ||
         guard.back().getName().getStringRef()!="ct_src.bin_panels" ||
         !llvm::hasSingleElement(guard.back().getRegion(0)))
        return op->emitError("expected guarded zero then count panels"),WalkResult::interrupt();
      Value limitVal = guard.back().getOperand(0);
      bool limitOk = false;
      if (extentVal % spanVal == 0) {
        limitOk = isConst(limitVal, panelsVal);
      } else if (auto *toI32 = isMap(limitVal, "i32")) {
        if (auto *mn = isMap(toI32->getOperand(0), "min"))
          if (isConst(mn->getOperand(0), panelsVal))
            if (auto *cd = isMap(mn->getOperand(1), "ceildiv"))
              if (isConst(cd->getOperand(1), stepVal))
                if (auto *sb = isMap(cd->getOperand(0), "sub"))
                  if (isConst(sb->getOperand(0), extentVal))
                    if (auto *ml = isMap(sb->getOperand(1), "mul"))
                      limitOk = ml->getOperand(0).getDefiningOp() &&
                                ml->getOperand(0).getDefiningOp()->getName().getStringRef() == "ct_src.tile_pid" &&
                                isConst(ml->getOperand(1), spanVal);
      }
      if (!limitOk)
        return guard.back().emitError("bin_panels upper bound disagrees with partition geometry"), WalkResult::interrupt();
      auto &loop=guard.back().getRegion(0).front();
      Operation *readOp = nullptr, *convOp = nullptr, *atomOp = &loop.back();
      for (auto &inst : loop) {
        if (inst.getName().getStringRef() == "ct_src.read_bin_keys")
          readOp = &inst;
        else if (inst.getName().getStringRef() == "ct_src.bin_index")
          convOp = &inst;
      }
      if(loop.getNumArguments()!=1 || !loop.getArgument(0).getType().isInteger(32) ||
         !readOp || !convOp || atomOp->getName().getStringRef()!="ct_src.atomic_bins" ||
         readOp->getOperand(0)!=src || convOp->getOperand(0)!=readOp->getResult(0) ||
         atomOp->getOperand(0)!=dst ||
         atomOp->getAttrOfType<BoolAttr>("transposed").getValue()!=swapVal ||
         atomOp->getAttrOfType<StringAttr>("scope")!="workgroup" || !src ||
         convOp->getResult(0).getType()!=RankedTensorType::get({stepVal},cast<MemRefType>(src.getType()).getElementType().isInteger(64)?IntegerType::get(fn.getContext(),64):IntegerType::get(fn.getContext(),32)) ||
         readOp->getResult(0).getType()!=RankedTensorType::get({stepVal},cast<MemRefType>(src.getType()).getElementType()))
        return op->emitError("invalid count panel or atomic update"),WalkResult::interrupt();
      auto isPid0 = [](Value v) -> bool {
        auto *d = v.getDefiningOp();
        return d && d->getName().getStringRef() == "ct_src.tile_pid" &&
               d->getAttrOfType<IntegerAttr>("axis").getInt() == 0;
      };
      auto isIv64 = [&](Value v) -> bool {
        auto *d = isMap(v, "i64");
        return d && d->getOperand(0) == loop.getArgument(0);
      };
      auto isBinsDim = [&](Value v) -> bool {
        auto *d = v.getDefiningOp();
        return d && d->getName().getStringRef() == "ct_src.array_dim" &&
               d->getOperand(0) == dst &&
               d->getAttrOfType<IntegerAttr>("axis").getInt() ==
                   (swapVal ? 0 : 1);
      };
      auto *panelAdd = isMap(readOp->getOperand(1), "add");
      auto *panelMul =
          panelAdd ? isMap(panelAdd->getOperand(0), "mul") : nullptr;
      bool panelIdxOk = panelAdd && panelMul && isPid0(panelMul->getOperand(0)) &&
                        isConst(panelMul->getOperand(1), panelsVal) &&
                        isIv64(panelAdd->getOperand(1));
      auto *sel = isMap(atomOp->getOperand(1), "select");
      auto *incCast = isMap(atomOp->getOperand(2), "i32");
      auto *validOp = (sel && incCast && sel->getOperand(0) == incCast->getOperand(0))
                          ? isMap(sel->getOperand(0), "and")
                          : nullptr;
      auto *inExtOp = validOp ? isMap(validOp->getOperand(0), "lt") : nullptr;
      auto *posAdd = inExtOp ? isMap(inExtOp->getOperand(0), "add") : nullptr;
      auto *baseAdd = posAdd ? isMap(posAdd->getOperand(0), "add") : nullptr;
      auto *partMul = baseAdd ? isMap(baseAdd->getOperand(0), "mul") : nullptr;
      auto *stepMul = baseAdd ? isMap(baseAdd->getOperand(1), "mul") : nullptr;
      auto *lanesOp = posAdd ? posAdd->getOperand(1).getDefiningOp() : nullptr;
      bool posOk = posAdd && baseAdd && partMul && stepMul && lanesOp &&
                   isPid0(partMul->getOperand(0)) &&
                   isConst(partMul->getOperand(1), spanVal) &&
                   isIv64(stepMul->getOperand(0)) &&
                   isConst(stepMul->getOperand(1), stepVal) &&
                   lanesOp->getName().getStringRef() == "ct_src.range" &&
                   lanesOp->getAttrOfType<IntegerAttr>("size").getInt() == stepVal;
      auto *inBinsOp = validOp ? isMap(validOp->getOperand(1), "and") : nullptr;
      auto *ge0Op = inBinsOp ? isMap(inBinsOp->getOperand(0), "ge") : nullptr;
      auto *ltBinsOp = inBinsOp ? isMap(inBinsOp->getOperand(1), "lt") : nullptr;
      bool wideKey =
          cast<MemRefType>(src.getType()).getElementType().isInteger(64);
      bool binsBoundOk = false;
      if (ltBinsOp) {
        if (wideKey) {
          binsBoundOk = isBinsDim(ltBinsOp->getOperand(1));
        } else if (auto *binsI32 = isMap(ltBinsOp->getOperand(1), "i32")) {
          binsBoundOk = isBinsDim(binsI32->getOperand(0));
        }
      }
      bool inBinsOk = ge0Op && ltBinsOp &&
                      ge0Op->getOperand(0) == convOp->getResult(0) &&
                      isConst(ge0Op->getOperand(1), 0) &&
                      ltBinsOp->getOperand(0) == convOp->getResult(0) &&
                      binsBoundOk;
      if (!panelIdxOk || !sel || sel->getOperand(1) != convOp->getResult(0) ||
          !isConst(sel->getOperand(2), 0) || !inExtOp || !posOk ||
          !isConst(inExtOp->getOperand(1), extentVal) || !inBinsOk)
        return atomOp->emitError("atomic_bins requires verified extent and bin range masks"), WalkResult::interrupt();
    }
    if (ok && op->hasAttr("ct_src.slab_tma") && n != "gather_slab" &&
        n != "scatter_slab")
      ok = false;
    if (ok && op->hasAttr("ct_src.broadcast_inner") && n != "gather_slab" &&
        n != "tile_load")
      ok = false;
    if (ok && (n == "gather_slab" || n == "scatter_slab")) {
      bool rect = rectangularSlab(op, fn);
      bool bcast =
          n == "gather_slab" && rect && broadcastInner(op->getOperand(0), fn);
      bool slabTma = rect && !bcast && factsPtr &&
                     alignedProjectedView(op->getOperand(0), fn, *factsPtr,
                                          indices->permitTma);
      ok = op->hasAttr("ct_src.rectangular") == rect &&
           op->hasAttr("ct_src.broadcast_inner") == bcast &&
           op->hasAttr("ct_src.slab_tma") == slabTma;
    }
    if (ok && n == "stripe_loop")
      ok = !callable;
    if (ok && (n == "stripe_tile_load" || n == "stripe_tile_store" ||
               n == "stripe_owner_load")) {
      bool store = n == "stripe_tile_store";
      bool ownerLd = n == "stripe_owner_load";
      auto arg = dyn_cast<BlockArgument>(op->getOperand(0));
      auto round = dyn_cast<BlockArgument>(op->getOperand(1));
      auto *parent = op->getParentOp();
      auto st = op->getAttrOfType<IntegerAttr>("stride");
      bool validRound =
          round && parent && round.getOwner() == &parent->getRegion(0).front() &&
          round.getArgNumber() == 0 &&
          (callable ? isa<scf::ForOp>(parent)
                    : parent->getName().getStringRef() == "ct_src.stripe_loop");
      bool validDir =
          arg && arg.getOwner() == &fn.front() &&
          ((*slots)[arg.getArgNumber()].getAsObject()->getString("direction") ==
               "InOut" ||
           (*slots)[arg.getArgNumber()].getAsObject()->getString("direction") ==
               (store ? "Out" : "In"));
      if (!validRound || !validDir || !st || st.getInt() <= 0) {
        op->emitError("invalid stripe access binding");
        return WalkResult::interrupt();
      }
      int rank = cast<MemRefType>(arg.getType()).getRank();
      if (ownerLd) {
        auto *rowsOp = op->getOperand(2).getDefiningOp();
        ok = rank == 1 && rowsOp &&
             rowsOp->getName().getStringRef() == "ct_src.rows";
      } else {
        auto *pid = op->getOperand(2).getDefiningOp(),
             *col = op->getOperand(3).getDefiningOp();
        ok = rank == 2 &&
             op->getAttrOfType<BoolAttr>("allow_tma").getValue() ==
                 indices->permitTma &&
             pid && pid->getName().getStringRef() == "ct_src.pid" && col &&
             col->getName().getStringRef() == "ct_src.zero" &&
             llvm::all_of(fn->getAttrOfType<ArrayAttr>("block.configs"),
                          [&](Attribute row) {
                            return st.getInt() %
                                       cast<DictionaryAttr>(row)
                                           .getAs<IntegerAttr>("rows")
                                           .getInt() ==
                                   0;
                          });
      }
    }
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
      bool bcast = n == "tile_load" && broadcastInner(arg, fn);
      bool allowTma =
          !bcast && factsPtr &&
          tileLoadStorePermitsTma(arg, fn, *factsPtr, indices->permitTma);
      ok = op->hasAttr("ct_src.broadcast_inner") == bcast &&
           op->getAttrOfType<BoolAttr>("allow_tma").getValue() == allowTma &&
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
              pm.addPass(std::make_unique<Signature>(indices->bits, p));
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
