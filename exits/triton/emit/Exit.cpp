#include "../capabilities/Target.h"
#include "intentdsl/levels/execution/block/ir/MatrixProgram.h"
#include "intentdsl/levels/execution/block/ir/OwnerProgram.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/exits/triton/ir/Dialect.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "intentdsl/levels/execution/block/decisions/Verify.h"
#include "intentdsl/levels/execution/block/ir/Access.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Grid.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include <set>

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void legalize(OpPassManager &, const TargetProfile &);
Value tritonScaledPanels(intentdsl::block::Tiles &,
                         intentdsl::block::MatrixAccess &);
Value tritonMatrixRead(intentdsl::block::Tiles &,
                       intentdsl::block::MatrixAccess &, bool, Value, Value);
Value tritonWindowPanels(intentdsl::block::Tiles &,
                         intentdsl::block::MatrixAccess &, Value &, Value &,
                         Value &);
bool tritonMatrixWrite(intentdsl::block::Tiles &,
                       intentdsl::block::MatrixAccess &, Value, Value,
                       ArrayRef<Value>);
intentdsl::block::OwnerAccess tritonOwnerViews();
std::optional<int64_t> tritonOwnerRectangleAxis(Operation *, func::FuncOp);
bool tritonOwnerRectPermitsTma(Operation *, func::FuncOp, bool);
bool tritonPanelPermitsTma(Operation *, func::FuncOp, bool);
bool tritonPagePermitsTma(Operation *, func::FuncOp, bool);

LogicalResult verifyTileProgram(func::FuncOp, const TargetProfile &);
TerminalPrinter tritonPrinter();
void lowerRows(func::FuncOp, const TargetProfile &);
LogicalResult verifyDynamicRows(func::FuncOp, const TargetProfile &);
void tileContracts(Unit &);
LogicalResult verifyColumnViews(func::FuncOp);
LogicalResult verifyMatrixExit(func::FuncOp,const TargetProfile &);
llvm::Expected<TargetProfile> capabilities(const llvm::json::Object &);
namespace {
ArrayAttr computeTritonRangeBounds(scf::ForOp loop) {
  Builder b(loop.getContext());
  SmallVector<Attribute, 3> bounds;
  for (Value v : {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()}) {
    if (auto i = v.getDefiningOp<arith::ConstantIndexOp>())
      bounds.push_back(b.getI64IntegerAttr(i.value()));
    else if (auto c = v.getDefiningOp<arith::ConstantIntOp>())
      bounds.push_back(b.getI64IntegerAttr(c.value()));
    else
      bounds.push_back(b.getUnitAttr());
  }
  return b.getArrayAttr(bounds);
}

bool isTritonCarryOwner(Operation *op) {
  return isa<scf::ForOp, scf::IfOp>(op) ||
         op->getName().getStringRef() == "triton_src.tiles";
}

bool isTritonCarryYield(Operation *op) {
  return isa<scf::YieldOp>(op) ||
         op->getName().getStringRef() == "triton_src.yield";
}

void stampTritonCarryAndLoops(func::FuncOp fn) {
  Builder b(fn.getContext());
  int64_t nextId = 0;
  fn.walk([&](Operation *op) {
    if (auto loop = dyn_cast<scf::ForOp>(op))
      loop->setAttr("triton_src.range_bounds", computeTritonRangeBounds(loop));
    if (isTritonCarryOwner(op) && op->getNumResults() > 0) {
      auto idAttr = b.getI64IntegerAttr(nextId++);
      op->setAttr("triton_src.carry_id", idAttr);
      for (Region &reg : op->getRegions()) {
        for (Block &blk : reg) {
          if (blk.empty())
            continue;
          Operation &term = blk.back();
          if (isTritonCarryYield(&term) && term.getNumOperands() > 0)
            term.setAttr("triton_src.carry_id", idAttr);
        }
      }
    }
  });
}

LogicalResult verifyTritonCarryAndLoops(func::FuncOp fn) {
  llvm::DenseSet<int64_t> seenIds;
  auto res = fn.walk([&](Operation *op) {
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (loop->getAttr("triton_src.range_bounds") !=
          computeTritonRangeBounds(loop)) {
        loop.emitError("missing or mismatched triton_src.range_bounds on scf.for");
        return WalkResult::interrupt();
      }
    }
    if (isTritonCarryOwner(op) && op->getNumResults() > 0) {
      auto idAttr = op->getAttrOfType<IntegerAttr>("triton_src.carry_id");
      if (!idAttr || !seenIds.insert(idAttr.getInt()).second) {
        op->emitError("missing or duplicate triton_src.carry_id");
        return WalkResult::interrupt();
      }
    }
    if (isTritonCarryYield(op) && op->getNumOperands() > 0) {
      Operation *parent = op->getParentOp();
      auto idAttr = op->getAttrOfType<IntegerAttr>("triton_src.carry_id");
      if (!parent || !isTritonCarryOwner(parent) || !idAttr ||
          parent->getAttr("triton_src.carry_id") != idAttr ||
          parent->getNumResults() != op->getNumOperands()) {
        op->emitError("triton_src.carry_id on yield disagrees with enclosing op");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return success(!res.wasInterrupted());
}

llvm::Error describeArtifact(func::FuncOp, llvm::raw_pwrite_stream &);
struct BindCallable : PassWrapper<BindCallable, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BindCallable)
  TargetProfile profile;
  explicit BindCallable(const TargetProfile &p) : profile(p) {}
  void runOnOperation() final {
    auto module = getOperation();
    auto fn = *module.getOps<func::FuncOp>().begin();
    if (intentdsl::kir::orderedOwners(fn))
      intentdsl::block::expandOwnerProgram(fn, "exits/triton",
                                           tritonOwnerViews());
    else {
      auto mt = profile.services.get<intentdsl::block::MatrixTarget>();
      intentdsl::block::expandMatrixProgram(
          fn, "exits/triton", tritonScaledPanels, tritonMatrixRead,
          tritonWindowPanels, tritonMatrixWrite,
          mt && mt->flatRecordBitPlanes);
    }
    Builder b(fn.getContext());
    if (fn.front().front().getName().getStringRef() !=
            "triton_src.tile_program" &&
        !intentdsl::kir::orderedOwners(fn)) {
      auto guard = fn->getAttrOfType<DenseI64ArrayAttr>("block.guard");
      fn->setAttr("triton_src.min_width", b.getI64IntegerAttr(guard[0]));
      fn->setAttr("triton_src.max_width", b.getI64IntegerAttr(guard[1]));
      lowerRows(fn, profile);
    }
    stampTritonCarryAndLoops(fn);
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
      OperationState st(fn.getLoc(), "triton_src.runtime_source");
      st.addAttribute("implementation", mb.getStringAttr(impl));
      mb.create(st);
    }
    llvm::SmallVector<char> metadata;
    llvm::raw_svector_ostream os(metadata);
    if (auto error = describeArtifact(fn, os)) {
      fn.emitError() << llvm::toString(std::move(error));
      signalPassFailure();
      return;
    }
    module->setAttr(
        "triton_src.package",
        b.getStringAttr(StringRef(metadata.data(), metadata.size())));
  }
};
llvm::Error describeArtifact(func::FuncOp fn, llvm::raw_pwrite_stream &out) {
  auto &entry = fn.front().front();
  if (entry.getName().getStringRef() == "triton_src.tile_program") {
    if (entry.getAttrOfType<StringAttr>("calling_convention") == "matrix")
      return intentdsl::block::emitMatrixPackage(fn, "", out, 2);
    if (entry.hasAttr("block.pair_axis")) {
      auto ownerTile = entry.getAttrOfType<IntegerAttr>("block.owner_tile");
      if (ownerTile && ownerTile.getInt() >= 128)
        return intentdsl::block::emitOwnerArtifact(
            fn, "",
            ArrayRef<llvm::json::Object>{
                llvm::json::Object{{"num_warps", 8}, {"num_stages", 2}},
                llvm::json::Object{{"num_warps", 8}, {"num_stages", 3}}},
            2, out);
      return intentdsl::block::emitOwnerArtifact(
          fn, "",
          ArrayRef<llvm::json::Object>{
              llvm::json::Object{{"num_warps", 4}, {"num_stages", 1}},
              llvm::json::Object{{"num_warps", 4}, {"num_stages", 2}},
              llvm::json::Object{{"num_warps", 4}, {"num_stages", 3}},
              llvm::json::Object{{"num_warps", 8}, {"num_stages", 2}},
              llvm::json::Object{{"num_warps", 8}, {"num_stages", 3}}},
          2, out);
    }
    int64_t warps = 4;
    int64_t ownerRows = 1;
    if (auto ot = entry.getAttrOfType<IntegerAttr>("block.owner_tile"))
      ownerRows = std::max<int64_t>(1, ot.getInt());
    if (auto ts = entry.getAttrOfType<IntegerAttr>("block.token_span");
        ts && ts.getInt() >= 1024)
      warps = 8;
    else {
      bool hasHistogram = false;
      int64_t maxAxis = 0;
      entry.walk([&](Operation *op) {
        auto k = op->getName().stripDialect();
        if (k == "tile_histogram")
          hasHistogram = true;
        if (k == "tile_axis")
          if (auto rt = dyn_cast<RankedTensorType>(op->getResult(0).getType());
              rt && rt.getRank() == 1)
            maxAxis = std::max(maxAxis, rt.getDimSize(0));
      });
      if ((maxAxis >= 256 || ownerRows * maxAxis >= 2048) &&
          !(hasHistogram && maxAxis <= 512))
        warps = 8;
      if (ownerRows == 1 && maxAxis <= 64 && !hasHistogram &&
          !entry.hasAttr("block.token_span"))
        return intentdsl::block::emitOwnerArtifact(
            fn, "",
            ArrayRef<llvm::json::Object>{
                llvm::json::Object{{"num_warps", 1}, {"num_stages", 1}},
                llvm::json::Object{{"num_warps", 2}, {"num_stages", 1}},
                llvm::json::Object{{"num_warps", 4}, {"num_stages", 1}}},
            2, out);
    }
    return intentdsl::block::emitOwnerArtifact(
        fn, "", llvm::json::Object{{"num_warps", warps}, {"num_stages", 1}}, 2,
        out);
  }
  if (intentdsl::kir::orderedOwners(fn)) {
    int64_t warps = 4;
    int64_t ownerRows = 1;
    if (auto ot = entry.getAttrOfType<IntegerAttr>("block.owner_tile"))
      ownerRows = std::max<int64_t>(1, ot.getInt());
    if (auto ts = entry.getAttrOfType<IntegerAttr>("block.token_span");
        ts && ts.getInt() >= 1024)
      warps = 8;
    else {
      bool hasHistogram = false;
      int64_t maxAxis = 0;
      entry.walk([&](Operation *op) {
        auto k = op->getName().stripDialect();
        if (k == "tile_histogram")
          hasHistogram = true;
        if (k == "tile_axis")
          if (auto rt = dyn_cast<RankedTensorType>(op->getResult(0).getType());
              rt && rt.getRank() == 1)
            maxAxis = std::max(maxAxis, rt.getDimSize(0));
      });
      if ((maxAxis >= 256 || ownerRows * maxAxis >= 2048) &&
          !(hasHistogram && maxAxis <= 512))
        warps = 8;
      if (ownerRows == 1 && maxAxis <= 64 && !hasHistogram &&
          !entry.hasAttr("block.token_span"))
        return intentdsl::block::emitOwnerArtifact(
            fn, "",
            ArrayRef<llvm::json::Object>{
                llvm::json::Object{{"num_warps", 1}, {"num_stages", 1}},
                llvm::json::Object{{"num_warps", 2}, {"num_stages", 1}},
                llvm::json::Object{{"num_warps", 4}, {"num_stages", 1}}},
            2, out);
    }
    return intentdsl::block::emitOwnerArtifact(
        fn, "", llvm::json::Object{{"num_warps", warps}, {"num_stages", 1}}, 2,
        out);
  }
  auto args = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!args)
    return args.takeError();
  llvm::json::Array configs;
  for (Attribute a : fn->getAttrOfType<ArrayAttr>("block.configs")) {
    auto d = cast<DictionaryAttr>(a);
    llvm::json::Object row;
    for (StringRef k : {"id", "rows", "width"})
      row[k] = d.getAs<IntegerAttr>(k).getInt();
    auto native = llvm::json::parse(d.getAs<StringAttr>("payload").getValue());
    if (!native)
      return native.takeError();
    row["native"] = std::move(*native);
    configs.push_back(std::move(row));
  }
  out << llvm::json::Value(llvm::json::Object{
      {"version", 2},
      {"grid",
       llvm::json::Array{
           intentdsl::block::gridExpression(fn->getAttr("block.grid")),
           std::max<int64_t>(
               1, llvm::count_if(fn.front(),
                                 [](Operation &op) {
                                   return op.getName().getStringRef() ==
                                          "triton_src.output_partition";
                                 })),
           1}},
      {"entry", "kernel"},
      {"source", ""},
      {"domain", intentdsl::publishDomain(fn)},
      {"abi", std::move(*args)},
      {"rows", std::move(configs)},
      {"topology", fn->getAttrOfType<StringAttr>("block.topology").str()},
      {"guard",
       llvm::json::Object{
           {"min_width",
            fn->getAttrOfType<IntegerAttr>("triton_src.min_width").getInt()},
           {"max_width",
            fn->getAttrOfType<IntegerAttr>("triton_src.max_width").getInt() >= 0
                ? llvm::json::Value(
                      fn->getAttrOfType<IntegerAttr>("triton_src.max_width")
                          .getInt())
                : llvm::json::Value(nullptr)}}}});
  return llvm::Error::success();
}
} // namespace
static LogicalResult verifyBoundary(ModuleOp m, const TargetProfile &profile,
                                    StringRef phase) {

  auto funcs = m.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs))
    return failure();
  auto fn = *funcs.begin();
  if (fn.empty() || fn.front().empty())
    return failure();
  if (phase != "callable") {
    if (!llvm::hasSingleElement(m.getBody()->getOperations()))
      return failure();
  } else {
    if (!m->getAttrOfType<StringAttr>("triton_src.package"))
      return failure();
    if (&m.getBody()->back() != fn.getOperation())
      return fn.emitError("expected function after runtime_source declarations"),
             failure();
    std::set<std::string> declaredResources;
    for (auto &top : m.getBody()->without_terminator()) {
      if (&top == fn.getOperation())
        break;
      if (top.getName().getStringRef() != "triton_src.runtime_source")
        return top.emitError("unexpected module-level operation"), failure();
      auto impl = top.getAttrOfType<StringAttr>("implementation");
      if (!impl)
        return top.emitError("missing runtime_source implementation"),
               failure();
      if (auto err = profile.checkImplementation(impl.getValue()))
        return top.emitError() << llvm::toString(std::move(err)), failure();
      auto &c = profile.implementations.at(impl.getValue().str());
      if (c.source != ImplementationSource::Runtime ||
          !declaredResources.insert(c.resource).second)
        return top.emitError("invalid or duplicate runtime_source"), failure();
    }
    std::set<std::string> usedResources;
    fn.walk([&](Operation *op) {
      if (auto attr = op->getAttrOfType<StringAttr>("ci.implementation")) {
        auto it = profile.implementations.find(attr.getValue().str());
        if (it != profile.implementations.end() &&
            it->second.source == ImplementationSource::Runtime)
          usedResources.insert(it->second.resource);
      }
    });
    if (declaredResources != usedResources)
      return fn.emitError(
                 "runtime_source declarations disagree with bound operations"),
             failure();
  }
  auto tritonCap = profile.services.get<intentdsl::itsrc::TritonTarget>();
  if (!tritonCap || fn->hasAttr("triton_src.tma") != tritonCap->permitTma)
    return fn.emitError("invalid Triton TMA transfer capability"), failure();
  if (phase == "callable" && failed(verifyTritonCarryAndLoops(fn)))
    return failure();
  if (fn.front().front().getName().getStringRef() == "triton_src.tile_program")
    return phase == "callable" ? verifyTileProgram(fn, profile) : failure();
  if (failed(verifyColumnViews(fn)))
    return failure();
  if (intentdsl::kir::orderedOwners(fn)) {
    auto &owner = fn.front().front();
    auto rows = owner.getAttrOfType<IntegerAttr>("block.owner_tile");
    Attribute expected;
    if (owner.hasAttr("block.pair_axis") && rows && rows.getInt() < 16)
      expected = Builder(fn.getContext()).getI64IntegerAttr(16);
    if (owner.getAttr("triton_src.matrix_rows") != expected)
      return fn.emitError("invalid native matrix fragment height"), failure();
    if (failed(intentdsl::block::verifyOwnerMap(fn, profile, "triton_src",
                                                "realized")))
      return failure();
    auto tmaWalk = fn.walk([&](Operation *op) {
      if (op->hasAttr("triton_src.panel_tma") !=
          tritonPanelPermitsTma(op, fn, tritonCap->permitTma)) {
        op->emitError("Triton panel TMA disagrees with layout");
        return WalkResult::interrupt();
      }
      if (op->hasAttr("triton_src.page_tma") !=
          tritonPagePermitsTma(op, fn, tritonCap->permitTma)) {
        op->emitError("Triton page TMA disagrees with layout");
        return WalkResult::interrupt();
      }
      Attribute expectedRect;
      if (auto rectAx = tritonOwnerRectangleAxis(op, fn))
        expectedRect = Builder(fn.getContext()).getI64IntegerAttr(*rectAx);
      if (op->getAttr("triton_src.owner_rectangle") != expectedRect ||
          op->hasAttr("triton_src.owner_tma") !=
              tritonOwnerRectPermitsTma(op, fn, tritonCap->permitTma)) {
        op->emitError("Triton owner rectangle TMA disagrees with layout");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return success(!tmaWalk.wasInterrupted());
  }
  if (intentdsl::kir::matrixDomain(fn))
    return verifyMatrixExit(fn, profile);
  if (!fn->getAttrOfType<StringAttr>("kir.abi") ||
      !fn->getAttrOfType<ArrayAttr>("block.configs") ||
      !fn->getAttrOfType<StringAttr>("block.topology"))
    return failure();
  if (phase == "callable" &&
      (!fn->getAttrOfType<IntegerAttr>("triton_src.min_width") ||
       !fn->getAttrOfType<IntegerAttr>("triton_src.max_width")))
    return failure();
  if (failed(intentdsl::verifyMapping(m, profile, "triton_src")))
    return failure();
  if (m.walk([](Operation *op) {
         auto n = op->getName().getStringRef();
         if (n == "triton_src.atomic_bins") {
           auto src = dyn_cast<BlockArgument>(op->getOperand(0)),
                dst = dyn_cast<BlockArgument>(op->getOperand(1));
           auto span = op->getAttrOfType<IntegerAttr>("span"),
                panel = op->getAttrOfType<IntegerAttr>("native_panel");
           if (!src || !dst || src == dst ||
               cast<MemRefType>(src.getType()).getRank() != 1 ||
               cast<MemRefType>(dst.getType()).getRank() != 2 ||
               op->getAttrOfType<StringAttr>("bin_mode") != "global-atomic" ||
               !span || span.getInt() < 1 || !panel ||
               !llvm::isPowerOf2_64(panel.getInt())) {
             op->emitError("invalid destination, partition or native panel for "
                           "global atomic bins");
             return WalkResult::interrupt();
           }
           auto &seq = op->getRegion(0).front();
           if (seq.getNumArguments() || seq.getOperations().size() != 3 ||
               seq.front().getName().stripDialect() != "clear_bins" ||
               seq.front().getNextNode()->getName().stripDialect() !=
                   "cta_barrier" ||
               seq.back().getName().stripDialect() != "bin_panels" ||
               seq.back().getAttrOfType<IntegerAttr>("count").getInt() !=
                   int64_t(llvm::divideCeil(span.getInt(), panel.getInt()))) {
             op->emitError("expected zero initialization, CTA synchronization "
                           "and complete key panels");
             return WalkResult::interrupt();
           }
           auto &loop = seq.back().getRegion(0).front();
           if (loop.getNumArguments() != 1 ||
               !loop.getArgument(0).getType().isIndex() ||
               !llvm::hasSingleElement(loop) ||
               loop.front().getName().stripDialect() != "bin_add" ||
               loop.front().getOperand(0) != loop.getArgument(0)) {
             op->emitError(
                 "expected one counted panel update per induction index");
             return WalkResult::interrupt();
           }
         }
         if (n == "triton_src.key_counts" || n == "triton_src.sum_columns" ||
             n == "triton_src.atomic_bins") {
           auto bits = op->getAttrOfType<IntegerAttr>("address_bits");
           if (!bits || bits.getInt() != 64) {
             op->emitError("key partition addressing requires i64");
             return WalkResult::interrupt();
           }
           if (n == "triton_src.sum_columns") {
             auto panel = op->getAttrOfType<IntegerAttr>("native_panel");
             auto extent = op->getAttrOfType<IntegerAttr>("extent");
             int64_t expected =
                 extent ? int64_t(std::min<uint64_t>(
                              256, llvm::PowerOf2Ceil(
                                       std::max<int64_t>(1, extent.getInt()))))
                        : 0;
             if (!panel || panel.getInt() != expected) {
               op->emitError("invalid sum_columns native panel width");
               return WalkResult::interrupt();
             }
           }
         }
         return WalkResult::advance();
       }).wasInterrupted())
    return failure();
  if (phase == "callable") {
    auto g = fn->getAttrOfType<DenseI64ArrayAttr>("block.guard");
    if (g[0] !=
            fn->getAttrOfType<IntegerAttr>("triton_src.min_width").getInt() ||
        g[1] != fn->getAttrOfType<IntegerAttr>("triton_src.max_width").getInt())
      return fn.emitError("call guard differs from verified block class"),
             failure();
    for (auto a : fn->getAttrOfType<ArrayAttr>("block.configs")) {
      auto payload = llvm::json::parse(
          cast<DictionaryAttr>(a).getAs<StringAttr>("payload").getValue());
      if (!payload) {
        llvm::consumeError(payload.takeError());
        return failure();
      }
      auto *opts = payload->getAsObject();
      if (!opts || opts->size() != 2 ||
          opts->getInteger("num_warps").value_or(0) <= 0 ||
          opts->getInteger("num_stages").value_or(0) <= 0)
        return fn.emitError("illegal native configuration"), failure();
    }
  }
  bool valid = true;
  m.walk([&](Operation *op) {
    auto ns = op->getName().getDialectNamespace();
    if (ns == "triton_src") {
      auto name = op->getName().stripDialect();
      auto matches = [&](unsigned i, StringRef opName) {
        auto *v = op->getOperand(i).getDefiningOp();
        return v && v->getName().getStringRef() == opName;
      };
      if (name == "offsets") {
        auto axes = intentdsl::block::projection(op->getOperand(0).getType());
        valid &= succeeded(axes) &&
                 op->getAttrOfType<BoolAttr>("matrix").getValue() ==
                     (*axes == intentdsl::block::RowProjection::Matrix) &&
                 matches(1, "triton_src.rows") &&
                 matches(2, "triton_src.columns");
      }
      if (name == "neutralize") {
        auto a = op->getAttrOfType<StringAttr>("algebra");
        valid &= (a == "sum" || a == "max_num") &&
                 matches(1, "triton_src.active") &&
                 op->getOperand(1).getParentBlock() == op->getBlock();
      }
      if (name == "reduce")
        valid &= matches(0, "triton_src.neutralize") &&
                 op->getOperand(0).getDefiningOp()->getAttr("algebra") ==
                     op->getAttr("algebra");
    }
    valid &= ns == "builtin" || ns == "func" || ns == "triton_src" ||
             (phase == "callable" && (ns == "scf" || ns == "arith"));
  });
  if (!valid)
    return failure();
  return phase == "callable" ? verifyDynamicRows(fn, profile) : success();
}
void matrixIssues(Unit &);
void addressOptimization(Unit &);
void bind(Unit &unit) {
  matrixIssues(unit);
  addressOptimization(unit);
  tileContracts(unit);
  auto addParity = [&](ParityDeclaration p) {
    unit.parity.push_back({std::move(p), COMPILERINFRA_SOURCE});
  };
  addParity({"levels/execution/block::matrix-contraction",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/block::contraction", "exits/triton::native-dot"},
             "Tensor Core tl.dot contraction across SM90 and SM120"});
  addParity({"levels/execution/block::block-scaled-fp8-contraction",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/block::contraction", "exits/triton::scaled-mma"},
             "Guarded MX8/E8M0 block-scaled FP8 Tensor Core contraction via tl.dot_scaled across SM90 and SM120"});
  addParity({"levels/execution/block::q4k-packed-contraction",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/block::contraction", "exits/triton::row-collective"},
             "Packed Q4_K x Q8_K super-block integer subgroup dot and FP32 scale accumulation"});
  addParity({"levels/execution/block::ordered-owner-pairing",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/block::ordered-owners", "exits/triton::native-dot",
              "exits/triton::row-collective"},
             "Paired QK/PV Tensor Core owner tiles and single-pass online rescaling"});
  addParity({"levels/execution/block::row-reduction-and-scan",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/block::structure",
              "exits/triton::native-row-reduce", "exits/triton::inclusive-scan",
              "exits/triton::moments-reduction", "exits/triton::moments-merge"},
             "Native tl.sum/tl.max, tl.cumsum, and Welford moments reduction"});
  addParity({"levels/execution/block::shared-private-histogram",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/block::structure", "exits/triton::histogram"},
             "CTA-local tl.histogram private bin accumulation before global merge"});
  addParity({"levels/execution/block::counter-philox-rng",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/block::structure"},
             "Builtin tl.randint4x counter-based Philox uniform generation"});
  addParity({"levels/execution/block::bulk-tma-transfer",
             "*",
             ParityStatus::Implemented,
             {"levels/execution/block::contraction",
              "levels/execution/block::ordered-owners"},
             "Hardware TMA 2D/ND descriptor loads and stores via tl.make_tensor_descriptor"});
  addParity({"levels/execution/block::narrow-i32-addressing",
             "*",
             ParityStatus::Implemented,
             {"exits/triton::narrow-proven-addresses"},
             "proveNarrowAddress-verified 32-bit signed tile_address narrowing"});
  unit.dialects = [](DialectRegistry &r) {
    r.insert<intentdsl::itsrc::ItsrcDialect>();
  };
  for (StringRef phase : {"legal", "callable"})
    unit.boundaries.push_back(
        {{phase.str(),
          [phase](OpPassManager &pm,
                  const TargetProfile &profile) -> llvm::Error {
            if (phase == "legal")
              legalize(pm, profile);
            else {
              pm.addPass(std::make_unique<BindCallable>(profile));
            }
            return llvm::Error::success();
          },
          [phase](ModuleOp m, const TargetProfile &profile) {
            if (failed(verifyBoundary(m, profile, phase)))
              return failure();
            if (phase == "legal")
              return success();
            auto fn = *m.getOps<func::FuncOp>().begin();
            llvm::SmallVector<char> expected;
            llvm::raw_svector_ostream os(expected);
            if (auto err = describeArtifact(fn, os))
              return fn.emitError() << llvm::toString(std::move(err)),
                     failure();
            auto package = m->getAttrOfType<StringAttr>("triton_src.package");
            if (!package || package.getValue() !=
                                StringRef(expected.data(), expected.size()))
              return fn.emitError("Triton package disagrees with the verified "
                                  "entry ABI or launch rows"),
                     failure();
            return success();
          }},
         COMPILERINFRA_SOURCE});
  ExitImplementation exit;
  exit.capabilities = capabilities;
  exit.terminal = tritonPrinter();
  unit.exit =
      Located<ExitImplementation>{std::move(exit), COMPILERINFRA_SOURCE};
}
} // namespace CI_UNIT_NAMESPACE
