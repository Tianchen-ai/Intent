#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
namespace intentdsl::block {
static int64_t contractionOperandBytes(Operation *domain) {
  int64_t maxBytes = 0;
  domain->walk([&](Operation *op) {
    auto kind = op->getName().stripDialect();
    if (!kir::isContraction(kind))
      return;
    unsigned limit = std::min(2u, op->getNumOperands());
    if (kind == "quant_contract") {
      if (const auto *spec = kir::quantFormatSpec(*op)) {
        if (spec->roleLayout == kir::QuantRoleLayout::GroupedWeightRhs)
          limit = std::min(1u, op->getNumOperands());
      }
    }
    for (unsigned i = 0; i < limit; ++i) {
      Type elem = getElementTypeOrSelf(op->getOperand(i).getType());
      if (elem.isIntOrFloat())
        maxBytes = std::max<int64_t>(
            maxBytes, llvm::divideCeil(elem.getIntOrFloatBitWidth(), 8));
    }
  });
  return maxBytes;
}
static bool domainMatchesTileOperation(Operation *domain, StringRef requiredOp,
                                       bool explicitRecordTiles) {
  if (explicitRecordTiles) {
    bool isRecord =
        domain
            ->walk([&](Operation *op) {
              auto kind = op->getName().stripDialect();
              if (kind == "q4k_dot")
                return WalkResult::interrupt();
              if (kind == "quant_contract") {
                if (const auto *spec = kir::quantFormatSpec(*op);
                    spec &&
                    spec->roleLayout == kir::QuantRoleLayout::RecordGemv)
                  return WalkResult::interrupt();
              }
              return WalkResult::advance();
            })
            .wasInterrupted();
    if (isRecord)
      return requiredOp == "q4k_dot";
  }
  if (requiredOp.empty())
    return true;
  return domain
      ->walk([&](Operation *op) {
        auto kind = op->getName().stripDialect();
        if (kind == "quant_contract") {
          if (const auto *spec = kir::quantFormatSpec(*op)) {
            if (spec->roleLayout == kir::QuantRoleLayout::RecordGemv)
              return requiredOp == "q4k_dot" ? WalkResult::interrupt()
                                             : WalkResult::advance();
            if (spec->roleLayout == kir::QuantRoleLayout::BlockScaledPair)
              return requiredOp == "scaled_contract" ? WalkResult::interrupt()
                                                     : WalkResult::advance();
            if (spec->roleLayout == kir::QuantRoleLayout::GroupedWeightRhs)
              return requiredOp == "quant_contract" ? WalkResult::interrupt()
                                                    : WalkResult::advance();
          }
        }
        if (kind == requiredOp)
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}
bool canSplitGroupedRhs(Operation *domain,
                        const compilerinfra::TargetProfile &profile) {
  auto target = profile.services.get<MatrixTarget>();
  auto facts = profile.services.get<CallFacts>();
  if (!target || !target->supportsSplitGroupedRhs || !facts)
    return false;
  auto fn = domain->getParentOfType<func::FuncOp>();
  if (!fn)
    return false;
  Operation *quantOp = nullptr;
  const kir::QuantFormatSpec *qSpec = nullptr;
  for (Operation &entry : domain->getRegion(0).front()) {
    Operation *op =
        entry.getName().stripDialect() == "panel_fold"
            ? entry.getRegion(0).front().back().getOperand(0).getDefiningOp()
            : &entry;
    if (!op)
      continue;
    auto k = op->getName().stripDialect();
    if (k == "quant_contract") {
      quantOp = op;
      qSpec = kir::quantFormatSpec(*op);
    } else if (kir::isContraction(k)) {
      return false;
    }
  }
  if (!quantOp || !qSpec ||
      qSpec->roleLayout != kir::QuantRoleLayout::GroupedWeightRhs ||
      qSpec->packFactor <= 1 || qSpec->blockElements <= 0)
    return false;
  auto shape = matrixShape(domain, *facts);
  if (!shape) {
    llvm::consumeError(shape.takeError());
    return false;
  }
  if (shape->batches != 1 || shape->m <= 0 || shape->n <= 0 || shape->k <= 0 ||
      shape->k % qSpec->blockElements != 0 ||
      shape->n % qSpec->packFactor != 0)
    return false;
  auto checkContig2D = [&](Value v) -> bool {
    auto arg = dyn_cast<BlockArgument>(v);
    if (!arg || arg.getOwner() != &fn.front())
      return false;
    auto mem = dyn_cast<MemRefType>(arg.getType());
    if (!mem || mem.getRank() != 2)
      return false;
    auto *obj = facts->arguments[arg.getArgNumber()].getAsObject();
    auto *sh = obj ? obj->getArray("shape") : nullptr;
    auto *st = obj ? obj->getArray("strides") : nullptr;
    if (!sh || !st || sh->size() != 2 || st->size() != 2)
      return false;
    auto d1 = (*sh)[1].getAsInteger();
    auto s0 = (*st)[0].getAsInteger(), s1 = (*st)[1].getAsInteger();
    return d1 && s0 && s1 && *s1 == 1 && *s0 == *d1;
  };
  if (!checkContig2D(quantOp->getOperand(0)) ||
      !checkContig2D(quantOp->getOperand(1)) ||
      !checkContig2D(quantOp->getOperand(2)) ||
      (qSpec->hasZeroView && !checkContig2D(quantOp->getOperand(3))) ||
      !checkContig2D(domain->getOperand(0)))
    return false;
  Type lhsElem =
      cast<MemRefType>(quantOp->getOperand(0).getType()).getElementType();
  return lhsElem.isF16() || lhsElem.isBF16() || lhsElem.isF32();
}
const MatrixTile *findMatrixTile(Operation *domain, ArrayRef<int64_t> dims,
                                 const compilerinfra::TargetProfile &profile,
                                 std::optional<StringRef> native, bool split) {
  auto target = profile.services.get<MatrixTarget>();
  if (!target)
    return nullptr;
  if (dims.size() == 4) {
    if (dims[3] == 2) {
      static const MatrixTile kLibraryTile{1, 1, 1, R"({"library_gemm":true})",
                                           "contract", 0};
      if (dims[0] == 1 && dims[1] == 1 && dims[2] == 1 &&
          (!native || *native == kLibraryTile.native) &&
          !target->libraryGemmImpl.empty() &&
          canSplitGroupedRhs(domain, profile)) {
        Type outElem =
            cast<MemRefType>(domain->getOperand(0).getType()).getElementType();
        Type lhsElem;
        domain->walk([&](Operation *op) {
          if (op->getName().stripDialect() == "quant_contract")
            lhsElem =
                cast<MemRefType>(op->getOperand(0).getType()).getElementType();
        });
        if (lhsElem && lhsElem == outElem)
          return &kLibraryTile;
      }
      return nullptr;
    }
    if (dims[3] != 1)
      return nullptr;
    split = true;
    dims = dims.take_front(3);
  }
  if (dims.size() != 3)
    return nullptr;
  int64_t operandBytes = contractionOperandBytes(domain);
  auto facts = profile.services.get<CallFacts>();
  if (split) {
    if (!canSplitGroupedRhs(domain, profile))
      return nullptr;
    for (const MatrixTile &t : target->tiles) {
      if (dims == ArrayRef<int64_t>{t.m, t.n, t.k} &&
          (!native || *native == t.native) &&
          (!t.maxOperandBytes || operandBytes <= t.maxOperandBytes) &&
          (t.operation.empty() || t.operation == "contract") &&
          (!t.accepts || (facts && t.accepts(domain, t.k, *facts))))
        return &t;
    }
    return nullptr;
  }
  bool explicitRecordTiles = target->implementations.count("q4k_dot") != 0;
  for (const MatrixTile &t : target->tiles) {
    if (dims == ArrayRef<int64_t>{t.m, t.n, t.k} &&
        (!native || *native == t.native) &&
        (!t.maxOperandBytes || operandBytes <= t.maxOperandBytes) &&
        domainMatchesTileOperation(domain, t.operation, explicitRecordTiles) &&
        (!t.accepts || (facts && t.accepts(domain, t.k, *facts))))
      return &t;
  }
  return nullptr;
}
bool supportsMatrixTile(Operation *domain, Attribute choice,
                        const compilerinfra::TargetProfile &profile) {
  auto dims = dyn_cast_or_null<DenseI64ArrayAttr>(choice);
  return dims && findMatrixTile(domain, dims.asArrayRef(), profile) != nullptr;
}
llvm::Expected<kir::ContractionShape> matrixShape(Operation *region,
                                                  const CallFacts &f) {
  for (auto &entry : region->getRegion(0).front()) {
    auto &op =
        entry.getName().stripDialect() == "panel_fold"
            ? *entry.getRegion(0).front().back().getOperand(0).getDefiningOp()
            : entry;
    auto n = op.getName().stripDialect();
    if (kir::isContraction(n))
      return kir::contractionShape(op, region->getOperand(0), f);
  }
  return llvm::createStringError("missing matrix reduction");
}
LogicalResult verifyTiledMatrix(func::FuncOp fn,
                                const compilerinfra::TargetProfile &p,
                                StringRef dialect, VerifyPanelSetup setup) {
  auto op = &fn.front().front();
  auto tile = op->getAttrOfType<DenseI64ArrayAttr>("block.output_tile"),
       dims = op->getAttrOfType<DenseI64ArrayAttr>("block.extents");
  if (!tile || tile.size() != 3 || !dims || dims.size() != 4)
    return op->emitError("missing matrix bounds");
  unsigned folds = 0;
  Operation *setupOp = nullptr;
  if (failed(kir::verifyMatrix(
          fn, dialect, true,
          [&](Operation &entry) -> Operation * {
            if (entry.getName().stripDialect() != "panel_fold")
              return &entry;
            if (entry.getName().getDialectNamespace() != dialect ||
                !llvm::hasSingleElement(entry.getRegion(0)))
              return nullptr;
            auto &body = entry.getRegion(0).front();
            if (body.getOperations().size() != (setup ? 3u : 2u) ||
                body.back().getName().stripDialect() != "panel_yield")
              return nullptr;
            if (setup)
              setupOp = &body.front();
            auto *math = body.back().getPrevNode();
            if (math->getNumResults() != 1 ||
                body.back().getOperand(0) != math->getResult(0) ||
                entry.getResult(0).getType() != math->getResult(0).getType() ||
                entry.getAttrOfType<IntegerAttr>("bound").getInt() != dims[2] ||
                entry.getAttrOfType<IntegerAttr>("step").getInt() != tile[2])
              return nullptr;
            ++folds;
            return math;
          })) ||
      folds != 1)
    return op->emitError("expected one complete reduction panel loop");
  if (setup && (!setupOp || failed(setup(*setupOp))))
    return op->emitError("invalid panel supply");
  auto actual = matrixShape(op, *p.services.get<CallFacts>());
  if (!actual)
    return fn.emitError() << llvm::toString(actual.takeError());
  auto native = op->getAttrOfType<StringAttr>("block.native_row");
  bool isSplit = op->hasAttr("block.split_dequant");
  bool isSplitLib = op->hasAttr("block.split_library");
  bool tileOk = false;
  if (native) {
    if (isSplitLib) {
      SmallVector<int64_t, 4> libDims(tile.asArrayRef().begin(),
                                      tile.asArrayRef().end());
      libDims.push_back(2);
      tileOk = isSplit && findMatrixTile(op, libDims, p, native.getValue());
    } else {
      tileOk =
          findMatrixTile(op, tile.asArrayRef(), p, native.getValue(), isSplit);
    }
  }
  if (!tileOk)
    return op->emitError("tile is not supported on this target");
  if (!dims ||
      dims.asArrayRef() !=
          ArrayRef<int64_t>{actual->m, actual->n, actual->k, actual->batches})
    return op->emitError("iteration bounds do not match contraction operands");
  auto facts = p.services.get<CallFacts>();
  for (auto [i, arg] : llvm::enumerate(fn.getArguments())) {
    auto shape = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "block.shape");
    auto *extents = facts->shapeOf(arg);
    if (!extents || !shape || shape.size() != extents->size())
      return fn.emitError("missing matrix shape");
    for (auto [a, b] : llvm::zip(shape.asArrayRef(), *extents))
      if (b.getAsInteger() != a)
        return fn.emitError("incorrect matrix shape");
    auto strides = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "block.strides");
    auto *s = facts->arguments[i].getAsObject()->getArray("strides");
    if (!strides || !s || strides.size() != s->size())
      return fn.emitError("incomplete matrix addressing");
    for (auto [a, b] : llvm::zip(strides.asArrayRef(), *s))
      if (b.getAsInteger() != a)
        return fn.emitError("incorrect matrix access stride");
  }
  return success();
}
LogicalResult verifyTiledMatrix(func::FuncOp fn,
                                const compilerinfra::TargetProfile &p,
                                StringRef dialect) {
  return verifyTiledMatrix(fn, p, dialect, nullptr);
}
llvm::Error emitMatrixPackage(func::FuncOp fn, StringRef src,
                              llvm::raw_pwrite_stream &out, unsigned version,
                              llvm::json::Object nativeViews) {
  auto &op = fn.front().front();
  auto dims = op.getAttrOfType<DenseI64ArrayAttr>("block.extents");
  auto tile = op.getAttrOfType<DenseI64ArrayAttr>("block.output_tile");
  auto abi = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  auto native =
      llvm::json::parse(op.getAttrOfType<StringAttr>("block.native_row"));
  if (!abi)
    return abi.takeError();
  if (!native)
    return native.takeError();
  auto ceil = [](int64_t a, int64_t b) { return a / b + (a % b != 0); };
  llvm::json::Array rows{llvm::json::Object{{"id", 0},
                                            {"rows", tile[0]},
                                            {"width", tile[1]},
                                            {"native", std::move(*native)}}};
  llvm::json::Object pkg{
      {"version", version},
      {"domain", publishDomain(fn)},
      {"abi", std::move(*abi)},
      {"rows", std::move(rows)},
      {"guard", llvm::json::Object{{"min_width", 0}, {"max_width", nullptr}}},
      {"grid", llvm::json::Array{ceil(dims[0], tile[0]), ceil(dims[1], tile[1]),
                                 dims[3]}},
      {"entry", "kernel"},
      {"matrix_views", true},
      {"native_views", std::move(nativeViews)},
      {"source", src.str()}};
  if (auto preAttr = fn->getAttrOfType<StringAttr>("block.pre_stage")) {
    auto pre = llvm::json::parse(preAttr.getValue());
    if (!pre)
      return pre.takeError();
    pkg["pre_stage"] = std::move(*pre);
  }
  if (fn->hasAttr("block.library_gemm"))
    pkg["library_gemm"] = true;
  out << llvm::json::Value(std::move(pkg));
  return llvm::Error::success();
}
} // namespace intentdsl::block
