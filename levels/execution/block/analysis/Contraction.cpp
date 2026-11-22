#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
namespace intentdsl::block {
bool supportsMatrixTile(Operation *domain, Attribute choice,
                        const compilerinfra::TargetProfile &profile) {
  auto dims = dyn_cast_or_null<DenseI64ArrayAttr>(choice);
  auto target = profile.services.get<MatrixTarget>();
  return dims && dims.size() == 3 && target &&
         llvm::any_of(target->tiles, [&](const MatrixTile &t) {
           return dims.asArrayRef() == ArrayRef<int64_t>{t.m, t.n, t.k} &&
                  (t.operation.empty() ||
                   domain
                       ->walk([&](Operation *op) {
                         return op->getName().stripDialect() == t.operation
                                    ? WalkResult::interrupt()
                                    : WalkResult::advance();
                       })
                       .wasInterrupted());
         });
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
  auto target = p.services.get<MatrixTarget>();
  auto native = op->getAttrOfType<StringAttr>("block.native_row");
  if (!tile || tile.size() != 3 || !target || !native ||
      !llvm::any_of(target->tiles, [&](auto &t) {
        if (tile.asArrayRef() != ArrayRef<int64_t>{t.m, t.n, t.k} ||
            native.getValue() != t.native)
          return false;
        return t.operation.empty() ||
               op->walk([&](Operation *child) {
                   return child->getName().stripDialect() == t.operation
                              ? WalkResult::interrupt()
                              : WalkResult::advance();
                 }).wasInterrupted();
      }))
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
  out << llvm::json::Value(llvm::json::Object{
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
      {"source", src.str()}});
  return llvm::Error::success();
}
} // namespace intentdsl::block
