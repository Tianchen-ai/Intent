#include "Matrix.h"
#include "OwnerVectors.h"
#include "Rows.h"
#include "Slab.h"
#include "intentdsl/exits/bangc/ir/Dialect.h"
#include "intentdsl/levels/algorithm/kir/ir/Segment.h"
#include "intentdsl/levels/storage/dsa/ir/Owners.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
llvm::Expected<TargetProfile> capabilities(const llvm::json::Object &);
namespace {
// A preceding collective releases its scratch before the division. A second
// consumer of the FP32 quotient would require a separately rounded FP32 value.
Operation *availableCollective(Operation *op) {
  for (auto *prev = op->getPrevNode(); prev; prev = prev->getPrevNode())
    if (prev->getName().getStringRef() == "dsa.reduce")
      return prev;
  return nullptr;
}
bool narrowedQuotient(Operation *div) {
  if (!div || div->getName().getStringRef() != "block.div" ||
      !div->hasOneUse() || !availableCollective(div))
    return false;
  auto *castOp = *div->getUsers().begin();
  if (castOp->getName().getStringRef() != "block.cast" ||
      castOp->getBlock() != div->getBlock())
    return false;
  auto lhs = dyn_cast<RankedTensorType>(div->getOperand(0).getType());
  auto rhs = div->getOperand(1).getType();
  auto scalar = dyn_cast<RankedTensorType>(rhs);
  auto out = dyn_cast<RankedTensorType>(castOp->getResult(0).getType());
  return lhs && lhs.getRank() == 2 && lhs.getElementType().isF32() &&
         (rhs.isF32() || (scalar && scalar.getRank() == 1 &&
                          scalar.getElementType().isF32())) &&
         out && out.getElementType().isF16();
}
StringRef primitive(Operation *op, int64_t depth) {
  auto n = op->getName().getStringRef();
  if(n=="block.segment_size") return "segment_size";
  if(n=="block.segment_fold"){
    if(intentdsl::kir::contiguousFeatureRead(op))return "feature_panel_sum";
    return "ordered_members";
  }
  if(n=="block.member_load" || n=="block.term")return op->getName().stripDialect();
  if(n=="block.scan_sum")return "row_inclusive_sum";
  if(n=="dsa.publish_offsets")return "prefix_endpoints";
  if(n=="dsa.fill_ranges_at")return "mark_intervals";
  if(n=="block.atomic_add_at")return "atomic_fetch_add";
  if(n=="block.index_read_at" || n=="dsa.unique_put")return op->getName().stripDialect();
  if(n=="dsa.scatter_fragment")return "pack_scatter";
  if(n=="block.member_tile" || n=="block.ne")return op->getName().stripDialect();
  if(n=="block.key_counts")return "private_bin_count";
  if(n=="block.sum_columns")return "vector_sum_columns";
  if(n=="block.summary_reduce")return "welford_f32";
  if(n=="block.summary_field" || n=="block.summary_merge" || n=="block.summary_identity" ||
     n=="dsa.owner_rows" || n=="dsa.owner_read" || n=="dsa.owner_write" || n=="dsa.slab_axis" || n=="dsa.lookup_axis")return op->getName().stripDialect();
  if(n=="dsa.compute_members"){
    auto layout=op->getParentOfType<func::FuncOp>()->getAttrOfType<DictionaryAttr>("dsa.layout");
    if(layout.getAs<BoolAttr>("group_temporaries").getValue())return "packed_pointwise";
  }
  if (n == "block.mul" || n == "block.cast") {
    auto elem = [](Type t) {
      if (auto s = dyn_cast<ShapedType>(t))
        return s.getElementType();
      return t;
    };
    if (elem(op->getResult(0).getType()).isBF16())
      return n == "block.mul" ? "bf16_product" : "bf16_round";
    if (n == "block.cast" && elem(op->getOperand(0).getType()).isBF16())
      return "bf16_widen";
  }
  if (n == "block.div" && narrowedQuotient(op))
    return "quotient_for_narrow";
  if (n == "block.cast" && narrowedQuotient(op->getOperand(0).getDefiningOp()))
    return "certified_half_quotient";
  if (n == "dsa.load_shared" || n == "dsa.member_axis" || n == "dsa.window" || n == "dsa.task_groups" || n == "dsa.range" ||
      n == "dsa.slot_zero" || n == "dsa.next" || n == "dsa.other_slot" ||
      n == "dsa.issue_if_in_bounds" || n == "dsa.wait_slot" ||
      n == "dsa.compute_members" || n == "dsa.iterate" || n == "dsa.advance" ||
      n == "dsa.end" || n == "dsa.complete")
    return op->getName().stripDialect();
  if (n == "dsa.phase")
    return "block_tasks";
  if (n == "dsa.copy_in") {
    if(op->getNumOperands()==4 && op->getOperand(2).getDefiningOp()->getName().getStringRef()=="dsa.lookup_axis")
      return "indexed_dma";
    if(op->getParentOfType<func::FuncOp>()->hasAttr("kir.owner_axis"))return "segmented_slab_read";
    // The grouped transfer increments the view's penultimate stride. Prove
    // that it is precisely the fastest logical row axis; earlier coordinates
    // are invariant until that axis wraps. The emitted guard excludes wraps.
    if (intentdsl::dsa::projectsConsecutiveRows(op, 1))
      return depth==2 ? "group_dma_async" : "group_dma";
    return depth == 2 ? "dma_async" : "dma";
  }
  if (n == "dsa.await")
    return depth == 2 ? "wait_slot" : "ready";
  if (n == "dsa.write_back") {
    if(op->getParentOfType<func::FuncOp>()->hasAttr("kir.owner_axis"))return "segmented_slab_write";
    auto layout =
        op->getParentOfType<func::FuncOp>()->getAttrOfType<DictionaryAttr>(
            "dsa.layout");
    if (intentdsl::dsa::projectsTransposed2D(op) && layout &&
        layout.getAs<BoolAttr>("group_temporaries") &&
        layout.getAs<BoolAttr>("group_temporaries").getValue())
      return op->getAttrOfType<BoolAttr>("async").getValue()
                 ? "transpose_store_async"
                 : "transpose_store";
    if (intentdsl::dsa::projectsConsecutiveRows(op, 2))
      return op->getAttrOfType<BoolAttr>("async").getValue()
                 ? "group_store_async" : "group_store";
    return op->getAttrOfType<BoolAttr>("async").getValue()
               ? "async_store_extent"
               : "store_extent";
  }
  if (n == "dsa.drain")
    return "drain_writes";
  if (n == "dsa.issued" || n == "dsa.yield")
    return "forward";
  if (n == "dsa.reduce" || n == "block.combine" || n == "block.identity") {
    auto alg = op->getAttrOfType<StringAttr>("algebra");
    if (!alg)
      return {};
    if (n == "dsa.reduce")
      return alg == "sum" ? "row_sum" : alg == "max_num" ? "row_max_num" : "";
    return alg == "sum" ? "sum" : alg == "max_num" ? "max_num" : "";
  }

  if (n == "block.maximum") {
    auto *constant = op->getOperand(1).getDefiningOp();
    auto zero =
        constant ? constant->getAttrOfType<FloatAttr>("value") : FloatAttr{};
    auto tile = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (tile && tile.getRank() == 2 && tile.getElementType().isF16() && zero &&
        zero.getType().isF16() && zero.getValue().isPosZero())
      return "positive_half_or_nan";
  }
  if (n == "block.uniform") return "philox4x32_uniform";
  if (n == "block.linear_index") {
    auto element=cast<RankedTensorType>(op->getResult(0).getType()).getElementType();
    if (!element.isInteger(32) && !element.isUnsignedInteger(64)) return {};
  }
  if (n == "block.sqrt" || n == "block.linear_index" || n == "block.xor" ||
      n == "block.and" || n == "block.shl" || n == "block.shr" ||
      n == "block.gt" || n == "block.select")
    return op->getName().stripDialect();
  if (n == "block.constant" || n == "block.maximum")
    return op->getName().stripDialect();
  if (n == "block.add" || n == "block.sub" || n == "block.mul" ||
      n == "block.div" || n == "block.rsqrt" || n == "block.exp" ||
      n == "block.cast")
    return op->getName().stripDialect();
  return {};
}
// Offset in bytes is pitch * (perGroup * R * slots + perMember).
// The same formula describes every admitted specialization of this phase.
LogicalResult offsets(func::FuncOp f, bool install) {
  Builder b(f.getContext());
  auto phases = f.walk([&](Operation *p) {
    if (p->getName().getStringRef() != "dsa.phase")
      return WalkResult::advance();
    int64_t grouped = 0, local = 0;
    auto attach = [&](Operation &o, int64_t g, int64_t l) {
      auto address = b.getDenseI64ArrayAttr({g, l});
      if (install)
        o.setAttr("bangc.offset", address);
      return o.getAttr("bangc.offset") == address;
    };
    for (auto &o : p->getRegion(0).front().without_terminator()) {
      if (o.getName().getStringRef() == "dsa.window" || o.getName().getStringRef()=="dsa.member_axis" || o.getName().getStringRef()=="dsa.slab_axis" || o.getName().getStringRef()=="dsa.lookup_axis")
        continue;
      if (o.getName().getStringRef() != "dsa.copy_in" || !attach(o, grouped, 0))
        return WalkResult::interrupt();
      grouped +=
          cast<MemRefType>(o.getOperand(0).getType()).getElementTypeBitWidth() /
          8;
    }
    auto &compute = p->getRegion(1).front();
    bool reuse=f->getAttrOfType<DictionaryAttr>("dsa.layout").contains("reuse_temporaries");
    auto temporaries=reuse?intentdsl::dsa::placeTemporaries(compute):intentdsl::dsa::TemporaryLayout{};
    for(auto &o:compute){
      if(o.getNumResults()!=1 || o.getName().getStringRef()=="dsa.await")continue;
      auto t=dyn_cast<RankedTensorType>(o.getResult(0).getType());
      if(!t || t.getRank()!=2)continue;
      if (reuse && !temporaries.offsets.count(&o))return WalkResult::interrupt();
      if (!attach(o, grouped, reuse?temporaries.offsets.lookup(&o):local))
        return WalkResult::interrupt();
      local+=(t.getElementTypeBitWidth()+7)/8;
    }
    if(reuse)local=temporaries.peak;
    for (auto &o : compute)
      if (o.getName().getStringRef() == "dsa.reduce" &&
          !attach(o, grouped, local))
        return WalkResult::interrupt();
    bool wideGroup =
        f->getAttrOfType<DictionaryAttr>("dsa.layout")
            .getAs<BoolAttr>("group_temporaries")
            .getValue();
    for (auto &o : compute) {
      auto code = o.getName().getStringRef();
      auto resTy = o.getNumResults() == 1
                       ? dyn_cast<RankedTensorType>(o.getResult(0).getType())
                       : RankedTensorType{};
      if (code == "block.scan_sum") {
        auto addr = b.getDenseI64ArrayAttr({grouped, local});
        if (install)
          o.setAttr("bangc.scan_scratch", addr);
        if (o.getAttr("bangc.scan_scratch") != addr)
          return WalkResult::interrupt();
      }
      if ((code == "block.mul" && resTy && resTy.getRank() == 2 &&
           resTy.getElementType().isBF16()) ||
          (code == "block.div" && resTy && resTy.getRank() == 2 &&
           resTy.getElementType().isF32() && !narrowedQuotient(&o)) ||
          (code == "block.exp" && resTy && resTy.getRank() == 2 &&
           resTy.getElementType().isF32()) ||
          code == "block.summary_reduce" ||
          (wideGroup && intentdsl::dsa::projectsTransposed2D(&o))) {
        auto addr = b.getDenseI64ArrayAttr({grouped, local});
        if (install)
          o.setAttr("bangc.scratch", addr);
        if (o.getAttr("bangc.scratch") != addr)
          return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (phases.wasInterrupted())
    return f.emitError("private slot byte binding disagrees with its phase");
  return success();
}
IntegerAttr emptyCountSlot(Operation *phase) {
  if(phase->getName().getStringRef()!="dsa.phase" || phase->getNumResults())return {};
  BlockArgument count;
  for(auto &v:phase->getRegion(1).front()){
    auto n=v.getName().getStringRef();
    if(n=="dsa.scatter_fragment"){
      if(count)return {};
      count=dyn_cast<BlockArgument>(v.getOperand(1));
    }else if(!llvm::is_contained(ArrayRef<StringRef>{"dsa.await","block.constant","block.ne","block.cast","block.scan_sum",
        "block.member_tile","dsa.active","dsa.window","dsa.complete","dsa.drain","dsa.yield","block.active",
        "block.rows","block.columns","dsa.end"},n))return {};
  }
  return count?Builder(phase->getContext()).getI32IntegerAttr(count.getArgNumber()):IntegerAttr{};
}
struct Legalize : PassWrapper<Legalize, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Legalize)
  unsigned phase;
  compilerinfra::TargetProfile target;
  intentdsl::CallFacts facts;
  intentdsl::dsa::LocalSupply supply;
  Legalize(unsigned step, const compilerinfra::TargetProfile &p,
           const intentdsl::CallFacts &f, const intentdsl::dsa::LocalSupply &s)
      : phase(step), target(p), facts(f), supply(s) {}
  void runOnOperation() final {
    auto f = *getOperation().getOps<func::FuncOp>().begin();
    if(intentdsl::kir::matrixDomain(f)){
      if (phase == 2) {
        if (failed(lowerMatrixToCallable(getOperation(), target)))
          signalPassFailure();
        return;
      }
      if(failed(bindMatrix(f,phase,true)))signalPassFailure();
      return;
    }
    if (intentdsl::kir::orderedOwners(f)) {
      if (phase == 2) {
        if (failed(lowerOrderedToCallable(getOperation(), target)))
          signalPassFailure();
        return;
      }
      auto &space = f.front().front();
      space.setAttr(
          "bangc.workers",
          Builder(f.getContext()).getI64IntegerAttr(supply.consumers));
      space.setAttr(
          "bangc.owner_memory",
          StringAttr::get(
              f.getContext(),
              space.getAttrOfType<IntegerAttr>("dsa.owner_bytes").getInt()
                  ? "private-nram"
                  : "global"));
      if (phase >= 1 && failed(ownerVectors(
                            f, true, std::min(supply.budget, supply.capacity))))
        signalPassFailure();
      return;
    }
    Builder b(&getContext());
    if (phase == 0) {
      if (failed(offsets(f, true))) {
        signalPassFailure();
        return;
      }
      f->setAttr("bangc.space", b.getStringAttr("private.nram"));
      f->setAttr("bangc.participant", b.getStringAttr("block"));
      return;
    }
    if (phase == 2) {
      if (failed(lowerRowsToCallable(getOperation(), target)))
        signalPassFailure();
      return;
    }
    auto layout = f->getAttrOfType<DictionaryAttr>("dsa.layout");
    auto depth = layout.getAs<IntegerAttr>("depth").getInt();
    f.walk([&](Operation *o) {
      if (o->getName().getDialectNamespace() == "func")
        return;
      auto p = primitive(o, depth);
      if (p.empty()) {
        o->emitError("no typed row microkernel");
        signalPassFailure();
        return;
      }
      o->setAttr("bangc.primitive", b.getStringAttr(p));
      if(p=="feature_panel_sum")o->setAttr("bangc.panel_members",b.getI64IntegerAttr(supply.featurePanelMembers));
      if(p=="indexed_dma")o->setAttr("bangc.indexed_loop",indexedLoop(b,depth,supply.indexScratchPerMember,supply.alignment));
      if(auto slot=emptyCountSlot(o))o->setAttr("bangc.empty_count_slot",slot);
      if(p.starts_with("segmented_slab_"))
        o->setAttr("bangc.slab_loop",slabLoop(o,p.ends_with("_read"),facts));
      if (p == "certified_half_quotient")
        o->setAttr("bangc.scratch",
                   availableCollective(o->getOperand(0).getDefiningOp())
                       ->getAttr("bangc.offset"));
    });
  }
};
LogicalResult callable(func::FuncOp fn) {
  auto spec = fn->getAttrOfType<StringAttr>("kir.abi");
  if (!spec || fn->getAttrOfType<StringAttr>("bangc.entry") != "intent_enqueue")
    return fn.emitError("missing published queue entry or buffer slots");
  auto parsed = llvm::json::parse(spec.getValue());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return fn.emitError("invalid argument descriptor");
  }
  auto *args = parsed->getAsArray();
  if (!args || args->size() != fn.getNumArguments())
    return fn.emitError("native arity differs from author slots");
  for (auto [value, desc] : llvm::zip(fn.getArguments(), *args)) {
    auto *slot = desc.getAsObject();
    auto view = dyn_cast<MemRefType>(value.getType());
    auto t = view ? view.getElementType() : value.getType();
    int rank = view ? view.getRank() : 0;
    if (!slot || !slot->getArray("dims") ||
        slot->getArray("dims")->size() != unsigned(rank) || rank > 4 ||
        (!t.isF32() && !t.isF16() && !t.isBF16() && !t.isInteger(32) &&
         !(view && t.isSignlessInteger(64)) && (view || !t.isUnsignedInteger(64))) ||
        slot->getString("dtype") != (t.isF16()         ? "f16"
                                     : t.isBF16()      ? "bf16"
                                     : t.isInteger(32) ? "i32"
                                     : t.isSignlessInteger(64) ? "i64"
                                        : t.isUnsignedInteger(64) ? "u64" : "f32") ||
        (view ? (slot->getString("direction") != "In" &&
                 slot->getString("direction") != "Out" &&
                 slot->getString("direction") != "InOut")
              : (slot->getString("direction") != "Scalar" ||
                 (!t.isF32() && !t.isInteger(32) && !t.isUnsignedInteger(64)))))
      return fn.emitError(
          "row ABI dtype, extent or direction inconsistent with function");
  }
  auto access = fn.walk([&](Operation *o) {
    bool read = o->getName().getStringRef() == "dsa.copy_in";
    if (!read && o->getName().getStringRef() != "dsa.write_back")
      return WalkResult::advance();
    auto v = dyn_cast<BlockArgument>(o->getOperand(0));
    if (!v || v.getOwner() != &fn.front() ||
        ((*args)[v.getArgNumber()].getAsObject()->getString("direction") !=
             "InOut" &&
         (*args)[v.getArgNumber()].getAsObject()->getString("direction") !=
             (read ? "In" : "Out"))) {
      o->emitError("native transfer conflicts with public slot direction");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success(!access.wasInterrupted());
}
LogicalResult verify(ModuleOp m, const TargetProfile &p, unsigned stage) {
  auto funcs = m.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs))
    return failure();
  auto fn = *funcs.begin();
  if (stage == 2 && !fn.empty() && !fn.front().empty()) {
    auto top = fn.front().front().getName().getStringRef();
    if (top == "bangc_src.matrix_program")
      return verifyCallableMatrix(m, p);
    if (top == "bangc_src.row_program")
      return verifyCallableRows(m, p);
    if (top == "bangc_src.ordered_program")
      return verifyCallableOrdered(m, p);
    return failure();
  }
  if (stage == 2)
    return failure();
  if (failed(intentdsl::dsa::verifyProgram(m, p, true)))
    return failure();
  if(intentdsl::kir::matrixDomain(fn))return bindMatrix(fn,stage,false);
  if (intentdsl::kir::orderedOwners(fn)) {
    if (failed(intentdsl::dsa::verifyOwners(fn, p, true)))
      return failure();
    auto supply = p.services.get<intentdsl::dsa::LocalSupply>();
    if (stage >= 1 &&
        failed(ownerVectors(fn, false,
                            std::min(supply->budget, supply->capacity))))
      return failure();
    auto &space = fn.front().front();
    auto n = p.services.get<intentdsl::dsa::LocalSupply>()->consumers;
    if (space.getAttrOfType<StringAttr>("bangc.owner_memory") !=
        (space.getAttrOfType<IntegerAttr>("dsa.owner_bytes").getInt()
             ? "private-nram"
             : "global"))
      return fn.emitError("invalid owner storage binding");
    return success(n > 0 && space.getAttr("bangc.workers") ==
                                Builder(fn.getContext()).getI64IntegerAttr(n));
  }
  auto layout = fn->getAttrOfType<DictionaryAttr>("dsa.layout");
  auto depth = layout.getAs<IntegerAttr>("depth").getInt();
  if (fn->getAttrOfType<StringAttr>("bangc.space") != "private.nram" ||
      fn->getAttrOfType<StringAttr>("bangc.participant") != "block")
    return fn.emitError("missing private address or participant mapping");
  auto capacity = p.services.get<intentdsl::dsa::LocalSupply>();
  for (auto a : fn->getAttrOfType<ArrayAttr>("block.configs")) {
    unsigned tasks = 0;
    if (cast<DictionaryAttr>(a)
            .getAs<StringAttr>("payload")
            .getValue()
            .getAsInteger(10, tasks) ||
        !tasks || tasks > capacity->consumers)
      return fn.emitError("row exceeds available physical participants");
  }
  if (failed(offsets(fn, false)))
    return failure();
  if (stage == 2 && failed(callable(fn)))
    return failure();
  if (stage > 0 &&
      fn.walk([&](Operation *o) {
          if (o->getName().getDialectNamespace() == "func")
            return WalkResult::advance();
          auto k = primitive(o, depth);
          if(o->getAttr("bangc.empty_count_slot")!=emptyCountSlot(o))
            return o->emitError("empty owner publication disagrees with phase effects"),WalkResult::interrupt();
          if(k=="feature_panel_sum" && o->getAttr("bangc.panel_members")!=Builder(m.getContext()).getI64IntegerAttr(capacity->featurePanelMembers))
            return o->emitError("feature staging disagrees with reserved storage"), WalkResult::interrupt();
          if(k=="indexed_dma" && o->getAttr("bangc.indexed_loop")!=indexedLoop(Builder(m.getContext()),depth,capacity->indexScratchPerMember,capacity->alignment))
            return o->emitError("invalid index prefetch, member copies or completion"), WalkResult::interrupt();
          DictionaryAttr loop;
          if(k.starts_with("segmented_slab_"))
            loop=slabLoop(o,k.ends_with("_read"),*p.services.get<intentdsl::CallFacts>());
          if(o->getAttr("bangc.slab_loop")!=loop){
            o->emitError("segmented transfer loop differs from the projected strides");
            return WalkResult::interrupt();
          }
          if (k == "certified_half_quotient" &&
              o->getAttr("bangc.scratch") !=
                  availableCollective(o->getOperand(0).getDefiningOp())
                      ->getAttr("bangc.offset"))
            return WalkResult::interrupt();
          if (k.empty() ||
              o->getAttrOfType<StringAttr>("bangc.primitive") != k) {
            o->emitError(
                "native primitive differs from transfer or calculation");
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        }).wasInterrupted())
    return failure();
  return success();
}
} // namespace
void dmaCost(Unit &);
void panelCost(Unit &);
void bind(Unit &u) {
  dmaCost(u);
  panelCost(u);
  rowContracts(u);
  addressOptimization(u);
  u.dialects = [](DialectRegistry &r) {
    r.insert<intentdsl::bangc_src::BangcSrcDialect, scf::SCFDialect,
             arith::ArithDialect>();
  };
  unsigned stage = 0;
  for (StringRef name : {"bound", "legal", "callable"}) {
    u.boundaries.push_back(
        {{name.str(),
          [stage](OpPassManager &pm, const TargetProfile &p) {
            pm.addPass(std::make_unique<Legalize>(
                stage, p, *p.services.get<intentdsl::CallFacts>(),
                *p.services.get<intentdsl::dsa::LocalSupply>()));
            return llvm::Error::success();
          },
          [stage](ModuleOp m, const TargetProfile &p) {
            return verify(m, p, stage);
          }},
         COMPILERINFRA_SOURCE});
    ++stage;
  }
  ExitImplementation exit;
  exit.capabilities = capabilities;
  exit.terminal = rowPrinter();
  u.exit = Located<ExitImplementation>{std::move(exit), COMPILERINFRA_SOURCE};
}
} // namespace CI_UNIT_NAMESPACE
