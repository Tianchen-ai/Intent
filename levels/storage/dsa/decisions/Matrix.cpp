#include "intentdsl/levels/storage/dsa/ir/Matrix.h"
#include "mlir/IR/PatternMatch.h"
#include <climits>
using namespace mlir;
using namespace compilerinfra;
namespace intentdsl::dsa {
std::string PanelPlan::id() const {
  return (placement + "-" + std::to_string(participants)).str();
}

DictionaryAttr PanelPlan::attribute(Builder &b) const {
  return b.getDictionaryAttr({
      b.getNamedAttr("participants", b.getI64IntegerAttr(participants)),
      b.getNamedAttr("placement", b.getStringAttr(placement))});
}

std::optional<PanelPlan> PanelPlan::read(Attribute value, block::MatrixAccess &a,
                                         const MatrixStorage &ms) {
  auto dict = dyn_cast_or_null<DictionaryAttr>(value);
  auto participants = dict ? dict.getAs<IntegerAttr>("participants")
                           : dyn_cast_or_null<IntegerAttr>(value);
  if (!participants || !participants.getType().isInteger(64))
    return std::nullopt;
  int64_t n = participants.getInt();
  auto supported = ms.panelPlacement(a, n, ms);
  if (supported.empty())
    return std::nullopt;
  // Older native checkpoints carried only the participant count.
  if (!dict)
    return PanelPlan{n, supported};
  auto placement = dict.getAs<StringAttr>("placement");
  if (!placement || (placement != supported &&
                     !(placement == "private" && supported == "shared-lhs")))
    return std::nullopt;
  return PanelPlan{n, placement.getValue()};
}

std::optional<LibraryMatrixPlan> matchLibraryMatrix(block::MatrixAccess &a,
                                                    const TargetProfile &p) {
  auto ms = p.services.get<MatrixStorage>();
  if (!ms || ms->libraryEstimator.empty())
    return std::nullopt;
  if (a.dot.getName().stripDialect() != "contract" ||
      a.dot.getNumOperands() != 2 || a.routed() ||
      a.domain.getNumOperands() != 1)
    return std::nullopt;
  auto lhsArg = dyn_cast<BlockArgument>(a.dot.getOperand(0));
  auto rhsArg = dyn_cast<BlockArgument>(a.dot.getOperand(1));
  auto dstArg = dyn_cast<BlockArgument>(a.domain.getOperand(0));
  if (!lhsArg || !rhsArg || !dstArg || lhsArg.getOwner() != &a.fn.front() ||
      rhsArg.getOwner() != &a.fn.front() ||
      dstArg.getOwner() != &a.fn.front() || dstArg == lhsArg ||
      dstArg == rhsArg)
    return std::nullopt;

  auto lhsTy = dyn_cast<MemRefType>(lhsArg.getType());
  auto rhsTy = dyn_cast<MemRefType>(rhsArg.getType());
  auto dstTy = dyn_cast<MemRefType>(dstArg.getType());
  if (!lhsTy || !rhsTy || !dstTy)
    return std::nullopt;
  Type elem = lhsTy.getElementType();
  if (elem != rhsTy.getElementType() || elem != dstTy.getElementType())
    return std::nullopt;

  int32_t dtypeCode = elem.isF16()    ? 1
                      : elem.isF32()  ? 2
                      : elem.isBF16() ? 17
                                      : 0;
  if (!dtypeCode)
    return std::nullopt;
  std::string impl = elem.isF16() ? ms->libraryF16Impl : ms->libraryFtzImpl;
  if (impl.empty())
    return std::nullopt;

  if (&a.domain.getRegion(0).front().front() != &a.fold)
    return std::nullopt;
  Operation *afterFold = a.fold.getNextNode();
  if (!afterFold)
    return std::nullopt;
  auto isEndOrNull = [](Operation *op) {
    return !op || op->hasTrait<OpTrait::IsTerminator>();
  };
  if (elem.isF32()) {
    if (afterFold->getName().stripDialect() != "publish_matrix" ||
        afterFold->getNumOperands() != 2 ||
        afterFold->getOperand(0) != dstArg ||
        afterFold->getOperand(1) != a.fold.getResult(0) ||
        !isEndOrNull(afterFold->getNextNode()))
      return std::nullopt;
  } else {
    Operation *pub = afterFold->getNextNode();
    if (afterFold->getName().stripDialect() != "convert_matrix" ||
        afterFold->getOperand(0) != a.fold.getResult(0) || !pub ||
        pub->getName().stripDialect() != "publish_matrix" ||
        pub->getNumOperands() != 2 || pub->getOperand(0) != dstArg ||
        pub->getOperand(1) != afterFold->getResult(0) ||
        !isEndOrNull(pub->getNextNode()))
      return std::nullopt;
  }

  int64_t M = a.shape[0], N = a.shape[1], K = a.shape[2], B = a.shape[3];
  if (M <= 0 || N <= 0 || K <= 0 || B <= 0 || M > INT32_MAX || N > INT32_MAX ||
      K > INT32_MAX)
    return std::nullopt;

  auto checkOperand = [&](BlockArgument v, bool isLhs, int32_t &trans,
                          int64_t &batchStride) -> bool {
    int kAx = a.axis(isLhs ? "lhs_k" : "rhs_k");
    int bAx = a.axis(isLhs ? "lhs_batch" : "rhs_batch");
    int expectedRank = bAx < 0 ? 2 : 3;
    if (a.rank(v) != expectedRank)
      return false;
    int fAx = a.freeAxis(v, kAx, bAx);
    auto vShape =
        a.fn.getArgAttrOfType<DenseI64ArrayAttr>(a.slot(v), "block.shape");
    if (!vShape || static_cast<int>(vShape.size()) != expectedRank)
      return false;
    int64_t freeDim = isLhs ? M : N;
    if (vShape[fAx] != freeDim || vShape[kAx] != K)
      return false;
    if (bAx < 0) {
      batchStride = 0;
    } else if (vShape[bAx] == 1) {
      batchStride = 0;
    } else if (vShape[bAx] == B) {
      batchStride = a.stride(v, bAx);
      if (batchStride < 0)
        return false;
    } else {
      return false;
    }
    int64_t stF = a.stride(v, fAx), stK = a.stride(v, kAx);
    if (isLhs) {
      if (stK == 1 && (M == 1 || stF == K)) {
        trans = 0;
        return true;
      }
      if (stF == 1 && (K == 1 || stK == M)) {
        trans = 1;
        return true;
      }
      return false;
    }
    if (stF == 1 && (K == 1 || stK == N)) {
      trans = 0;
      return true;
    }
    if (stK == 1 && (N == 1 || stF == K)) {
      trans = 1;
      return true;
    }
    return false;
  };

  LibraryMatrixPlan plan;
  plan.impl = std::move(impl);
  plan.lhsSlot = lhsArg.getArgNumber();
  plan.rhsSlot = rhsArg.getArgNumber();
  plan.dstSlot = dstArg.getArgNumber();
  plan.batch = B;
  plan.m = M;
  plan.n = N;
  plan.k = K;
  plan.inDtype = dtypeCode;
  plan.outDtype = dtypeCode;
  if (!checkOperand(lhsArg, true, plan.transA, plan.lhsBatchStride) ||
      !checkOperand(rhsArg, false, plan.transB, plan.rhsBatchStride))
    return std::nullopt;

  auto dstShape =
      a.fn.getArgAttrOfType<DenseI64ArrayAttr>(a.slot(dstArg), "block.shape");
  int dstRank = a.rank(dstArg);
  if (!dstShape || static_cast<int>(dstShape.size()) != dstRank)
    return std::nullopt;
  if (dstRank == 2) {
    if (B != 1 || dstShape[0] != M || dstShape[1] != N ||
        (N > 1 && a.stride(dstArg, 1) != 1) ||
        (M > 1 && a.stride(dstArg, 0) != N))
      return std::nullopt;
    plan.dstBatchStride = 0;
  } else if (dstRank == 3) {
    if (dstShape[0] != B || dstShape[1] != M || dstShape[2] != N ||
        (N > 1 && a.stride(dstArg, 2) != 1) ||
        (M > 1 && a.stride(dstArg, 1) != N) ||
        (B > 1 && a.stride(dstArg, 0) < M * N))
      return std::nullopt;
    plan.dstBatchStride = B > 1 ? a.stride(dstArg, 0) : 0;
  } else {
    return std::nullopt;
  }
  return plan;
}

namespace {
// The exit supplies one physical implementation of the two input panels and
// its accumulator. All sizes include the consumer's padded native footprint.
int64_t bytes(const block::MatrixAccess &a, const TargetProfile &profile) {
  auto t = a.tile;
  return profile.services.get<MatrixStorage>()->privateBytes(
      t[0], t[1], t[2], a.dot.getName().stripDialect());
}
SmallVector<std::pair<Value, std::string>> feeds(block::MatrixAccess &a) {
  auto &dot = a.dot;
  auto n = dot.getName().stripDialect();
  if (n == "window_contract")
    return {{dot.getOperand(0), "window-left"},
            {dot.getOperand(1), "window-right"},
            {dot.getOperand(2), "row-coefficient"},
            {dot.getOperand(3), "row-prefix"}};
  if (n == "q4k_dot")
    return {{dot.getOperand(0), "packed-weight"},
            dot.getOperand(1).getDefiningOp()
                ? std::pair<Value, std::string>{
                      dot.getOperand(1).getDefiningOp()->getOperand(0),
                      "encode-activation"}
                : std::pair<Value, std::string>{dot.getOperand(1),
                                                "packed-weight"}};
  if (n == "scaled_contract")
    return {{dot.getOperand(0), "left-groups"},
            {dot.getOperand(1), "left-scales"},
            {dot.getOperand(2), "right-groups"},
            {kir::RightScale(dot).view, "right-scales"}};
  if (n == "quant_contract") {
    auto fmtAttr = dot.getAttrOfType<StringAttr>("format");
    const auto *spec =
        fmtAttr ? kir::lookupQuantFormat(fmtAttr.getValue()) : nullptr;
    if (spec && spec->roleLayout == kir::QuantRoleLayout::RecordGemv) {
      Value act = spec->q8kActivation
                      ? dot.getOperand(1).getDefiningOp()->getOperand(0)
                      : dot.getOperand(1);
      return {{dot.getOperand(0), "packed-weight"}, {act, "encode-activation"}};
    }
    if (spec && spec->roleLayout == kir::QuantRoleLayout::BlockScaledPair)
      return {{dot.getOperand(0), "left-groups"},
              {dot.getOperand(1), "left-scales"},
              {dot.getOperand(2), "right-groups"},
              {kir::RightScale(dot).view, "right-scales"}};
    SmallVector<std::pair<Value, std::string>> res{
        {dot.getOperand(0), "quant-left"},
        {dot.getOperand(1), "packed-weight"},
        {dot.getOperand(2), "right-scales"}};
    if (spec && spec->hasZeroView)
      res.push_back({dot.getOperand(3), "right-zeros"});
    return res;
  }
  return {{dot.getOperand(0), "left-panel"},
          {dot.getOperand(1), "right-panel"}};
}
bool supportsSupply(block::MatrixAccess &a, Attribute v,
                    const TargetProfile &p) {
  if (auto lib = dyn_cast_or_null<StringAttr>(v)) {
    auto plan = matchLibraryMatrix(a, p);
    if (!plan || plan->impl != lib.getValue())
      return false;
    if (auto err = p.checkImplementationForOp(lib.getValue(), &a.dot)) {
      llvm::consumeError(std::move(err));
      return false;
    }
    return true;
  }
  auto cap = p.services.get<LocalSupply>();
  auto ms = p.services.get<MatrixStorage>();
  auto plan = ms ? PanelPlan::read(v, a, *ms) : std::nullopt;
  if (!cap || !plan || plan->participants < 1 || plan->participants > cap->consumers)
    return false;
  return bytes(a, p) <= std::min(cap->capacity, cap->budget);
}
} // namespace
LogicalResult verifyMatrix(func::FuncOp fn, const TargetProfile &p,
                           bool supplied) {
  if (!supplied) {
    if (failed(block::verifyTiledMatrix(fn, p, "block")))
      return failure();
    block::MatrixAccess a(fn);
    if (!supportsSupply(a, a.dot.getAttr("dsa.matrix_choice"), p))
      return fn.emitError(
          "matrix choice exceeds worker count or private storage");
    return success();
  }
  auto verify = [&](Operation &setup) -> LogicalResult {
    if (setup.getName().getStringRef() != "dsa.panel_supply" ||
        !llvm::hasSingleElement(setup.getRegion(0)))
      return failure();
    block::MatrixAccess a(fn);
    auto cap = p.services.get<LocalSupply>();
    auto tasks = setup.getAttrOfType<IntegerAttr>("participants").getInt();
    auto scratch = setup.getAttrOfType<IntegerAttr>("scratch_bytes").getInt();
    if (auto lib = setup.getAttrOfType<StringAttr>("library_impl")) {
      auto plan = matchLibraryMatrix(a, p);
      if (!cap || !plan || plan->impl != lib.getValue() || tasks != 1 ||
          scratch != 0)
        return setup.emitError("invalid external library matrix supply");
      if (auto err = p.checkImplementationForOp(lib.getValue(), &a.dot))
        return setup.emitError() << llvm::toString(std::move(err));
      if (setup.hasAttr("placement"))
        return setup.emitError("external library cannot use native panel storage");
    } else {
      Builder b(fn.getContext());
      Attribute plan = setup.getAttr("participants");
      if (auto placement = setup.getAttrOfType<StringAttr>("placement"))
        plan = PanelPlan{tasks, placement.getValue()}.attribute(b);
      if (!supportsSupply(a, plan, p) || scratch != bytes(a, p))
        return setup.emitError("invalid native matrix placement or footprint");
    }
    auto sources = feeds(a);
    auto &body = setup.getRegion(0).front();
    if (body.getOperations().size() != sources.size() + 1 ||
        body.back().getName().getStringRef() != "dsa.panel_ready" ||
        body.back().getNumOperands() != sources.size())
      return setup.emitError("panel completion arity");
    for (auto [i, entry] : llvm::enumerate(body.without_terminator())) {
      if (entry.getName().getStringRef() != "dsa.stage_operand" ||
          entry.getOperand(0) != sources[i].first ||
          entry.getAttrOfType<StringAttr>("access") != sources[i].second ||
          entry.getAttrOfType<IntegerAttr>("slot").getInt() != int64_t(i) ||
          body.back().getOperand(i) != entry.getResult(0))
        return entry.emitError(
            "incorrect matrix supply or completion dependency");
    }
    return success();
  };
  return block::verifyTiledMatrix(fn, p, "block", verify);
}
} // namespace intentdsl::dsa
namespace CI_UNIT_NAMESPACE {
using namespace intentdsl;
void matrixSupply(Unit &u) {
  DecisionFamily f;
  f.name = "panel-supply";
  f.applies = [](Operation *op) {
    return kir::isContraction(op->getName().stripDialect()) &&
           kir::matrixDomain(op->getParentOfType<func::FuncOp>());
  };
  f.read = [](Operation *op) { return op->getAttr("dsa.matrix_choice"); };
  f.write = [](Operation *op, Attribute v) {
    op->setAttr("dsa.matrix_choice", v);
    return llvm::Error::success();
  };
  f.verify = [](Operation *op, Attribute v,
                const TargetProfile &p) -> llvm::Error {
    block::MatrixAccess a(op->getParentOfType<func::FuncOp>());
    if (auto lib = dyn_cast_or_null<StringAttr>(v)) {
      auto plan = dsa::matchLibraryMatrix(a, p);
      if (!plan || plan->impl != lib.getValue())
        return llvm::createStringError("unsupported library matrix ABI");
      if (auto err = p.checkImplementationForOp(lib.getValue(), &a.dot))
        return err;
      return llvm::Error::success();
    }
    if (!dsa::supportsSupply(a, v, p))
      return llvm::createStringError(
          "illegal participant or private panel footprint");
    return llvm::Error::success();
  };
  f.verifyCandidate = [](Operation *op, const StructuralCandidate &c,
                         const TargetProfile &p) -> llvm::Error {
    if (auto lib = dyn_cast<StringAttr>(c.value)) {
      if (c.id != "library-cnnl" || c.topology != "external-library" ||
          c.preconditions != std::vector<std::string>{"exact-library-abi"} ||
          c.implementations != std::vector<std::string>{lib.getValue().str()})
        return llvm::createStringError("invalid library matrix candidate");
      return llvm::Error::success();
    }
    block::MatrixAccess a(op->getParentOfType<func::FuncOp>());
    auto ms = p.services.get<dsa::MatrixStorage>();
    auto plan = ms ? dsa::PanelPlan::read(c.value, a, *ms) : std::nullopt;
    if (!plan || c.id != plan->id() ||
        c.topology != (plan->placement + "-panels").str() ||
        c.preconditions != std::vector<std::string>{"legal-panel-placement"})
      return llvm::createStringError("invalid matrix supply candidate");
    return llvm::Error::success();
  };
  u.families.push_back(
      {{std::move(f), "dsa@formed", "dsa@supplied"}, COMPILERINFRA_SOURCE});
  Strategy s;
  s.name = "task-panels";
  s.family = u.qualify("panel-supply");
  s.enumerate = [](Operation *op, const TargetProfile &p, AnalysisManager &)
      -> llvm::Expected<std::vector<StructuralCandidate>> {
    auto cap = p.services.get<dsa::LocalSupply>();
    block::MatrixAccess a(op->getParentOfType<func::FuncOp>());
    std::vector<StructuralCandidate> cs;
    Builder b(op->getContext());
    auto ms = p.services.get<dsa::MatrixStorage>();
    for (auto n = cap->consumers; n > 0; n /= 2) {
      auto placement = ms->panelPlacement(a, n, *ms);
      if (placement.empty())
        continue;
      auto propose = [&](StringRef kind) {
        dsa::PanelPlan plan{n, kind};
        auto attr = plan.attribute(b);
        if (dsa::supportsSupply(a, attr, p))
          cs.push_back({plan.id(), (kind + "-panels").str(), attr,
                        {"legal-panel-placement"}});
      };
      propose(placement);
      if (placement == "shared-lhs")
        propose("private");
    }
    return cs;
  };
  u.strategies.push_back({std::move(s), COMPILERINFRA_SOURCE});
  Strategy libStrat;
  libStrat.name = "library-panels";
  libStrat.family = u.qualify("panel-supply");
  libStrat.enumerate = [](Operation *op, const TargetProfile &p,
                          AnalysisManager &)
      -> llvm::Expected<std::vector<StructuralCandidate>> {
    std::vector<StructuralCandidate> cs;
    block::MatrixAccess a(op->getParentOfType<func::FuncOp>());
    if (auto plan = dsa::matchLibraryMatrix(a, p)) {
      cs.push_back({"library-cnnl",
                    "external-library",
                    Builder(op->getContext()).getStringAttr(plan->impl),
                    {"exact-library-abi"},
                    {plan->impl}});
    }
    return cs;
  };
  u.strategies.push_back({std::move(libStrat), COMPILERINFRA_SOURCE});
  Realizer r;
  r.name = "place-panels";
  r.family = u.qualify("panel-supply");
  r.apply = [](Operation *op, Attribute choice, const TargetProfile &p,
               RewriterBase &b) -> llvm::Error {
    op->getContext()->getOrLoadDialect("dsa");
    block::MatrixAccess a(op->getParentOfType<func::FuncOp>());
    b.setInsertionPoint(&a.dot);
    OperationState setup(op->getLoc(), "dsa.panel_supply");
    if (auto lib = dyn_cast<StringAttr>(choice)) {
      setup.addAttribute("participants", b.getI64IntegerAttr(1));
      setup.addAttribute("scratch_bytes", b.getI64IntegerAttr(0));
      setup.addAttribute("library_impl", lib);
    } else {
      auto plan = dsa::PanelPlan::read(choice, a, *p.services.get<dsa::MatrixStorage>());
      if (!plan)
        return llvm::createStringError("cannot realize native panel placement");
      setup.addAttribute("participants", b.getI64IntegerAttr(plan->participants));
      setup.addAttribute("placement", b.getStringAttr(plan->placement));
      setup.addAttribute("scratch_bytes", b.getI64IntegerAttr(dsa::bytes(a,p)));
    }
    setup.addRegion();
    auto *panel = b.create(setup);
    b.createBlock(&panel->getRegion(0));
    SmallVector<Value> completions;
    for (auto [v, mode] : dsa::feeds(a)) {
      OperationState stage(op->getLoc(), "dsa.stage_operand");
      stage.addOperands(v);
      stage.addTypes(b.getIndexType());
      stage.addAttribute("slot", b.getI64IntegerAttr(completions.size()));
      stage.addAttribute("access", b.getStringAttr(mode));
      completions.push_back(b.create(stage)->getResult(0));
    }
    OperationState ready(op->getLoc(), "dsa.panel_ready");
    ready.addOperands(completions);
    b.create(ready);
    op->removeAttr("dsa.matrix_choice");
    return llvm::Error::success();
  };
  u.realizers.push_back({std::move(r), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
