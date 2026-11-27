#include "../capabilities/Target.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/exits/triton/ir/Dialect.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/DotSemantics.h"
#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "intentdsl/levels/execution/block/decisions/Verify.h"
#include "intentdsl/levels/execution/block/ir/Access.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Grid.h"
#include "intentdsl/levels/execution/block/ir/MatrixProgram.h"
#include "intentdsl/levels/execution/block/ir/OwnerProgram.h"
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
intentdsl::block::OwnerAccess
tritonOwnerViews(const intentdsl::block::OwnerMachine *);
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

void promoteSteppedTmaLoops(func::FuncOp fn) {
  auto constInt = [](Value v) -> std::optional<int64_t> {
    if (auto ci = v.getDefiningOp<arith::ConstantIndexOp>())
      return ci.value();
    if (auto c = v.getDefiningOp<arith::ConstantIntOp>())
      return c.value();
    return std::nullopt;
  };
  SmallVector<scf::ForOp> candidates;
  fn.walk([&](scf::ForOp loop) { candidates.push_back(loop); });
  for (scf::ForOp loop : candidates) {
    auto lo = constInt(loop.getLowerBound());
    auto hi = constInt(loop.getUpperBound());
    auto st = constInt(loop.getStep());
    if (!lo || !hi || !st || *lo != 0 || *st != 1 || *hi <= 1)
      continue;
    Value iv = loop.getInductionVar();
    Value ivI64 = iv;
    arith::IndexCastOp ivCast;
    if (iv.getType().isIndex()) {
      for (Operation *u : iv.getUsers()) {
        if (auto ic = dyn_cast<arith::IndexCastOp>(u);
            ic && ic.getType().isInteger(64) && !ivCast)
          ivCast = ic;
        else {
          ivCast = nullptr;
          break;
        }
      }
      if (!ivCast)
        continue;
      ivI64 = ivCast.getResult();
    } else if (!iv.getType().isInteger(64)) {
      continue;
    }

    int64_t stride = 0;
    bool ok = true;
    unsigned descReadCount = 0;
    SmallVector<arith::MulIOp, 4> strideMuls;
    for (Operation &inst : loop.getBody()->without_terminator()) {
      if (inst.getName().getStringRef() != "triton_src.desc_read")
        continue;
      ++descReadCount;
      for (Value off : inst.getOperands().drop_front()) {
        auto mul = off.getDefiningOp<arith::MulIOp>();
        if (!mul) {
          if (off == ivI64)
            ok = false;
          continue;
        }
        std::optional<int64_t> factor;
        if (mul.getLhs() == ivI64)
          factor = constInt(mul.getRhs());
        else if (mul.getRhs() == ivI64)
          factor = constInt(mul.getLhs());
        if (!factor)
          continue;
        if (*factor < 16 || !llvm::isPowerOf2_64(*factor) ||
            (stride != 0 && stride != *factor)) {
          ok = false;
          break;
        }
        stride = *factor;
      }
      if (!ok)
        break;
    }
    if (!ok || descReadCount == 0 || stride == 0 ||
        *hi > INT32_MAX / stride)
      continue;

    for (Operation *u : ivI64.getUsers()) {
      if (auto mul = dyn_cast<arith::MulIOp>(u)) {
        std::optional<int64_t> f;
        if (mul.getLhs() == ivI64)
          f = constInt(mul.getRhs());
        else if (mul.getRhs() == ivI64)
          f = constInt(mul.getLhs());
        if (f && *f == stride)
          strideMuls.push_back(mul);
      }
    }

    OpBuilder b(loop);
    Location loc = loop.getLoc();
    auto makeBound = [&](int64_t val) -> Value {
      if (iv.getType().isIndex())
        return b.create<arith::ConstantIndexOp>(loc, val);
      return b.create<arith::ConstantIntOp>(loc, val, 64);
    };
    auto newLoop = b.create<scf::ForOp>(loc, makeBound(0),
                                        makeBound(*hi * stride),
                                        makeBound(stride), loop.getInitArgs());
    if (loop->hasAttr("triton_src.runtime_range"))
      newLoop->setAttr("triton_src.runtime_range", b.getUnitAttr());
    Block &oldBody = *loop.getBody();
    Block &newBody = *newLoop.getBody();
    b.setInsertionPointToStart(&newBody);
    Value newIv = newLoop.getInductionVar();
    Value newIvI64 =
        newIv.getType().isIndex()
            ? Value(b.create<arith::IndexCastOp>(loc, b.getI64Type(), newIv))
            : newIv;

    for (auto mul : strideMuls) {
      mul.replaceAllUsesWith(newIvI64);
      mul.erase();
    }
    if (ivCast && ivCast->use_empty())
      ivCast.erase();
    else if (!ivI64.use_empty()) {
      Value panelIdx = b.create<arith::DivUIOp>(
          loc, newIvI64, b.create<arith::ConstantIntOp>(loc, stride, 64));
      ivI64.replaceAllUsesWith(panelIdx);
      if (ivCast)
        ivCast.erase();
    }
    for (auto [oldArg, newArg] :
         llvm::zip(oldBody.getArguments().drop_front(),
                   newBody.getArguments().drop_front()))
      oldArg.replaceAllUsesWith(newArg);
    newBody.getOperations().splice(newBody.end(), oldBody.getOperations());
    loop.replaceAllUsesWith(newLoop.getResults());
    loop.erase();
  }
}

llvm::Error describeArtifact(func::FuncOp, const TargetProfile &,
                             llvm::raw_pwrite_stream &);
static bool canContract(Operation *op) {
  return !op->hasAttr("triton_src.round_nearest") &&
         compilerinfra::hasFastMath(op, arith::FastMathFlags::contract);
}

static void bindContraction(func::FuncOp f, const TargetProfile &profile) {
  bool uniform = f->getAttrOfType<StringAttr>("kir.fp_contract") == "on";
  f.walk([&](Operation *op) {
    auto n = op->getName().stripDialect();
    if (op->getNumResults() == 1 && isa<FloatType>(getElementTypeOrSelf(op->getResult(0).getType())) &&
        llvm::is_contained(ArrayRef<StringRef>{"add", "sub", "mul", "addf", "subf", "mulf"}, n))
      uniform &= canContract(op);
    if (auto id = op->getAttrOfType<StringAttr>("ci.implementation")) {
      auto it = profile.implementations.find(id.str());
      if (it != profile.implementations.end() && it->second.source == ImplementationSource::Runtime)
        uniform = false;
    }
    // Map operations and tile programs also contain generated arithmetic.
    if (n == "map" || n == "tile_program")
      uniform = false;
  });
  f->setAttr("triton_src.enable_fp_fusion", BoolAttr::get(f.getContext(), uniform));
  if (uniform)
    return;
  f.walk([&](Operation *add) {
    auto name = add->getName().getStringRef();
    if ((name != "arith.addf" && name != "triton_src.add") || !canContract(add))
      return;
    for (unsigned i = 0; i < 2; ++i) {
      auto *mul = add->getOperand(i).getDefiningOp();
      if (!mul || !canContract(mul) || (mul->getName().getStringRef() != "triton_src.mul" &&
                                        !isa<arith::MulFOp>(mul)))
        continue;
      OpBuilder b(add);
      OperationState s(add->getLoc(), "triton_src.fma");
      s.addOperands({mul->getOperand(0), mul->getOperand(1), add->getOperand(1-i)});
      s.addTypes(add->getResultTypes());
      auto *fma = b.create(s);
      add->replaceAllUsesWith(fma);
      add->erase();
      if (mul->use_empty())
        mul->erase();
      break;
    }
  });
}

struct BindCallable : PassWrapper<BindCallable, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BindCallable)
  TargetProfile profile;
  explicit BindCallable(const TargetProfile &p) : profile(p) {}
  void runOnOperation() final {
    auto module = getOperation();
    auto fn = *module.getOps<func::FuncOp>().begin();
    auto tritonCap = profile.services.get<intentdsl::itsrc::TritonTarget>();
    if (intentdsl::kir::orderedOwners(fn)) {
      auto *om = profile.services.get<intentdsl::block::OwnerMachine>();
      intentdsl::block::expandOwnerProgram(fn, "exits/triton",
                                           tritonOwnerViews(om));
    } else {
      auto mt = profile.services.get<intentdsl::block::MatrixTarget>();
      intentdsl::block::expandMatrixProgram(
          fn, "exits/triton", tritonScaledPanels, tritonMatrixRead,
          tritonWindowPanels, tritonMatrixWrite,
          mt && mt->flatRecordBitPlanes, mt && mt->flatSplitDequant,
          tritonCap && tritonCap->permitTma,
          mt ? StringRef(mt->libraryGemmImpl) : StringRef(""));
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
    for (auto f : module.getOps<func::FuncOp>()) {
      if (tritonCap && tritonCap->permitTma)
        f->setAttr("triton_src.tma", b.getUnitAttr());
      promoteSteppedTmaLoops(f);
      bindContraction(f, profile);
      stampTritonCarryAndLoops(f);
      f.walk([&](Operation *op) {
        auto name = op->getName().getStringRef();
        if (name == "triton_src.div" || isa<arith::DivFOp>(op)) {
          bool reciprocal = compilerinfra::hasFastMath(op, arith::FastMathFlags::arcp) &&
                            intentdsl::kir::allowsSignedFtz(op);
          op->setAttr("triton_src.division", b.getStringAttr(reciprocal ? "reciprocal" : "rn"));
        }
        if (name == "triton_src.tile_dot") {
          bool tf32 = getElementTypeOrSelf(op->getOperand(0).getType()).isF32() &&
                      f->getAttrOfType<StringAttr>("kir.matmul_precision") == "high";
          op->setAttr("input_precision", b.getStringAttr(tf32 ? "tf32" : "ieee"));
        }
      });
    }
    std::set<std::string> seenResources;
    SmallVector<std::string> runtimeImpls;
    module.walk([&](Operation *op) {
      if (auto attr = op->getAttrOfType<StringAttr>("ci.implementation")) {
        auto key = attr.getValue().str();
        auto it = profile.implementations.find(key);
        if (it != profile.implementations.end() &&
            it->second.source == ImplementationSource::Runtime &&
            seenResources.insert(it->second.resource).second)
          runtimeImpls.push_back(key);
      }
    });
    auto firstFn = *module.getOps<func::FuncOp>().begin();
    OpBuilder mb(firstFn);
    for (auto &impl : runtimeImpls) {
      OperationState st(firstFn.getLoc(), "triton_src.runtime_source");
      st.addAttribute("implementation", mb.getStringAttr(impl));
      mb.create(st);
    }
    llvm::SmallVector<char> metadata;
    llvm::raw_svector_ostream os(metadata);
    if (auto error = describeArtifact(fn, profile, os)) {
      fn.emitError() << llvm::toString(std::move(error));
      signalPassFailure();
      return;
    }
    module->setAttr(
        "triton_src.package",
        b.getStringAttr(StringRef(metadata.data(), metadata.size())));
  }
};
llvm::Error emitTritonOwnerDefaults(func::FuncOp fn, Operation &entry,
                                    const TargetProfile &profile,
                                    llvm::raw_pwrite_stream &out) {
  const auto *om = profile.services.get<intentdsl::block::OwnerMachine>();
  int64_t warpWidth = (om && om->warpSize > 0) ? om->warpSize : 32;
  int64_t residentEl =
      (om && om->residentElements > 0) ? om->residentElements : 256;
  int64_t histMax =
      (om && om->histogramPanelWidth > 0) ? om->histogramPanelWidth : 512;
  int64_t histMin =
      (om && om->minHistogramPanelWidth > 0) ? om->minHistogramPanelWidth : 64;
  int64_t warps = 4;
  int64_t ownerRows = 1;
  if (auto ot = entry.getAttrOfType<IntegerAttr>("block.owner_tile"))
    ownerRows = std::max<int64_t>(1, ot.getInt());
  if (auto ts = entry.getAttrOfType<IntegerAttr>("block.token_span");
      ts && ts.getInt() >= 8 * warpWidth * 4)
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
    if ((maxAxis >= residentEl || ownerRows * maxAxis >= residentEl * 8) &&
        !(hasHistogram && maxAxis <= histMax))
      warps = 8;
    if (ownerRows == 1 && hasHistogram && !entry.hasAttr("block.token_span"))
      return intentdsl::block::emitOwnerArtifact(
          fn, "",
          ArrayRef<llvm::json::Object>{
              llvm::json::Object{{"num_warps", 2}, {"num_stages", 1}},
              llvm::json::Object{{"num_warps", 4}, {"num_stages", 1}},
              llvm::json::Object{{"num_warps", 8}, {"num_stages", 1}}},
          2, out);
    if (ownerRows == 1 && maxAxis <= histMin && !hasHistogram &&
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
llvm::Error describeKernel(func::FuncOp fn, const TargetProfile &profile,
                           llvm::raw_pwrite_stream &out) {
  auto &entry = fn.front().front();
  if (entry.getName().getStringRef() == "triton_src.tile_program") {
    if (entry.getAttrOfType<StringAttr>("calling_convention") == "matrix")
      return intentdsl::block::emitMatrixPackage(fn, "", out, 2);
    if (entry.hasAttr("block.pair_axis")) {
      const auto *om = profile.services.get<intentdsl::block::OwnerMachine>();
      int64_t wideOwnerRows =
          (om && om->parallelMembers > 0) ? om->parallelMembers : 128;
      auto ownerTile = entry.getAttrOfType<IntegerAttr>("block.owner_tile");
      if (ownerTile && ownerTile.getInt() >= wideOwnerRows)
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
    return emitTritonOwnerDefaults(fn, entry, profile, out);
  }
  if (intentdsl::kir::orderedOwners(fn))
    return emitTritonOwnerDefaults(fn, entry, profile, out);
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
llvm::Error describeArtifact(func::FuncOp fn, const TargetProfile &profile,
                             llvm::raw_pwrite_stream &out) {
  SmallVector<char> buffer;
  llvm::raw_svector_ostream os(buffer);
  if (auto err = describeKernel(fn, profile, os))
    return err;
  auto pkg = llvm::json::parse(StringRef(buffer.data(), buffer.size()));
  if (!pkg)
    return pkg.takeError();
  auto enabled = fn->getAttrOfType<BoolAttr>("triton_src.enable_fp_fusion");
  if (!enabled)
    return llvm::createStringError("missing Triton FP fusion binding");
  (*pkg->getAsObject())["enable_fp_fusion"] = enabled.getValue();
  if (auto *pre = pkg->getAsObject()->getObject("pre_stage")) {
    auto first = *fn->getParentOfType<ModuleOp>().getOps<func::FuncOp>().begin();
    (*pre)["enable_fp_fusion"] = first->getAttrOfType<BoolAttr>("triton_src.enable_fp_fusion").getValue();
  }
  out << *pkg;
  return llvm::Error::success();
}
} // namespace
static LogicalResult verifyBoundary(ModuleOp m, const TargetProfile &profile,
                                    StringRef phase) {

  SmallVector<func::FuncOp, 2> fns(m.getOps<func::FuncOp>());
  if (fns.empty() || fns.size() > (phase == "callable" ? 2u : 1u))
    return failure();
  for (auto f : fns)
    if (f.empty() || f.front().empty())
      return failure();
  bool isSplitModule = fns.size() == 2;
  auto fn = fns.back();
  if (isSplitModule) {
    auto dqFn = fns.front();
    if (dqFn->getAttrOfType<StringAttr>("block.entry") != "dequant_rhs" ||
        dqFn.front().front().getName().getStringRef() !=
            "triton_src.tile_program" ||
        dqFn.front().front().getAttrOfType<StringAttr>("calling_convention") !=
            "split_dequant" ||
        !fn->hasAttr("block.split_dequant"))
      return dqFn.emitError("invalid split dequantization module structure"),
             failure();
  }
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
    bool seenFunc = false;
    for (auto &top : m.getBody()->without_terminator()) {
      if (isa<func::FuncOp>(top)) {
        seenFunc = true;
        continue;
      }
      if (seenFunc || top.getName().getStringRef() != "triton_src.runtime_source")
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
    m.walk([&](Operation *op) {
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
  for (auto f : fns) {
    if (!tritonCap || f->hasAttr("triton_src.tma") != tritonCap->permitTma)
      return f.emitError("invalid Triton TMA transfer capability"), failure();
    if (phase == "callable" && failed(verifyTritonCarryAndLoops(f)))
      return failure();
  }
  if (isSplitModule) {
    for (auto f : fns)
      if (failed(verifyTileProgram(f, profile)))
        return failure();
    return success();
  }
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
  intentdsl::kir::registerDotSemantics(
      unit, "triton", "bf16", "NVIDIA PTX matrix multiply-accumulate",
      "unspecified",
      "PTX ISA, Warp Level Matrix Multiply-Accumulate Instructions, mma "
      "Precision and rounding: accumulation order and handling of subnormal "
      "inputs unspecified; CUDA Tile mma use_fast_acc is a separate "
      "accumulation option.");
  intentdsl::kir::registerDotSemantics(
      unit, "triton", "f16", "NVIDIA PTX matrix multiply-accumulate",
      "unspecified",
      "PTX ISA, Warp Level Matrix Multiply-Accumulate Instructions, mma "
      "Precision and rounding: accumulation order and handling of subnormal "
      "inputs unspecified; CUDA Tile mma use_fast_acc is a separate "
      "accumulation option.");
  intentdsl::kir::registerDotSemantics(
      unit, "triton", "fp8", "NVIDIA PTX matrix multiply-accumulate",
      "unspecified",
      "PTX ISA, Warp Level Matrix Multiply-Accumulate Instructions, mma "
      "Precision and rounding: accumulation order and handling of subnormal "
      "inputs unspecified; CUDA Tile mma use_fast_acc is a separate "
      "accumulation option.");
  unit.semantics.push_back({{"fp32-dot", "NVIDIA f32 dot operands", "tl.dot(input_precision)", "tf32",
      "Triton language dot: default NVIDIA input_precision=tf32",
      [](const TargetProfile &, const llvm::json::Object &o) -> std::optional<SemanticChoice> {
        bool high = o.getString("float32_matmul_precision") == "high";
        return SemanticChoice{high ? "tf32" : "ieee", high ? "configuration" : "author-language default",
                              "tl.dot(input_precision)", high ? "Explicit high precision mode" :
                              "PyTorch highest requires full f32 inputs"};
      }}, COMPILERINFRA_SOURCE});
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
            SmallVector<func::FuncOp, 2> fns(m.getOps<func::FuncOp>());
            auto fn = fns.back();
            llvm::SmallVector<char> expected;
            llvm::raw_svector_ostream os(expected);
            if (auto err = describeArtifact(fn, profile, os))
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
