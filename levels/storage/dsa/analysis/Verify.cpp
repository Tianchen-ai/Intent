#include "../decisions/Plan.h"
#include "intentdsl/levels/execution/block/ir/Access.h"
#include "intentdsl/levels/execution/block/decisions/Verify.h"
using namespace mlir;
namespace intentdsl::dsa {
LogicalResult verifyProgram(ModuleOp m, const compilerinfra::TargetProfile &p,
                            bool supplied) {
  auto fs = m.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(fs))
    return m.emitError("requires one tile function");
  auto fn = *fs.begin();
  auto d = fn->getAttrOfType<DictionaryAttr>(supplied ? "dsa.layout"
                                                      : "dsa.pending");
  auto caps = p.services.get<LocalSupply>();
  if (fn.empty() || !llvm::hasSingleElement(fn.getBody()))
    return fn.emitError("expected one defined tile region");
  if (auto err = intentdsl::verifyRows(
          fn->getAttrOfType<ArrayAttr>("block.configs"),
          fn->getAttrOfType<StringAttr>("block.topology"),
          fn->getAttrOfType<DenseI64ArrayAttr>("block.guard"), p))
    return fn.emitError() << llvm::toString(std::move(err));
  Builder builder(m.getContext());
  if (fn->getAttr("block.grid") !=
      builder.getStrArrayAttr({"ceildiv", "M", "rows"}))
    return fn.emitError("row ownership requires the parent's instance grid");
  if (!supplied && failed(intentdsl::verifyMapping(m, p, "block")))
    return failure();
  if (!d || !caps || !d.getAs<ArrayAttr>("rows") ||
      d.getAs<ArrayAttr>("rows").empty() || !d.getAs<IntegerAttr>("depth"))
    return fn.emitError("missing supply rows or depth");
  auto verifyLayout = [&]() -> LogicalResult {
    auto expected = plan(fn, *caps, d.getAs<IntegerAttr>("depth").getInt());
    if (!expected) {
      return fn.emitError() << llvm::toString(expected.takeError());
    }
    if (*expected != d)
      return fn.emitError("parameter rows or peak private allocation disagree "
                          "with live stages");
    return success();
  };
  if (!supplied)
    return verifyLayout();
  unsigned count = 0;
  auto walk = fn.walk([&](Operation *o) {
    if (o->getName().getStringRef() != "dsa.phase")
      return WalkResult::advance();
    ++count;
    Builder b(m.getContext());
    auto depth = d.getAs<IntegerAttr>("depth").getInt();
    if (o->getAttr("slots") != b.getI32IntegerAttr(depth) ||
        o->getAttr("lookahead") != b.getI32IntegerAttr(depth - 1) ||
        o->getAttrOfType<StringAttr>("visibility") != "participant_private")
      return WalkResult::interrupt();
    for (auto &r : o->getRegions())
      if (!llvm::hasSingleElement(r) || r.front().empty())
        return WalkResult::interrupt();
    auto &fetch = o->getRegion(0).front(), &body = o->getRegion(1).front();
    auto axis = o->getAttrOfType<StringAttr>("axis");
    if (axis != (fn->getAttrOfType<StringAttr>("block.topology") == "resident"
                     ? "instances"
                     : "columns"))
      return WalkResult::interrupt();
    auto &issued = fetch.back();
    if (issued.getName().getStringRef() != "dsa.issued" ||
        body.back().getName().getStringRef() != "dsa.yield" ||
        !body.back().getPrevNode() ||
        body.back().getPrevNode()->getName().getStringRef() != "dsa.drain" ||
        body.getNumArguments() != issued.getNumOperands() + o->getNumResults())
      return WalkResult::interrupt();
    if (std::distance(fetch.begin(), fetch.end()) !=
            issued.getNumOperands() + 1 ||
        o->getNumOperands() != o->getNumResults() ||
        !llvm::equal(o->getResultTypes(), body.back().getOperandTypes()))
      return WalkResult::interrupt();
    auto transfer = fetch.begin();
    for (auto [i, arg] : llvm::enumerate(issued.getOperands())) {
      auto *def = arg.getDefiningOp();
      auto in = body.getArgument(i);
      if (!def || def != &*transfer++ || def->getBlock() != &fetch ||
          def->getName().getStringRef() != "dsa.copy_in" ||
          arg.getType() != in.getType() || !in.hasOneUse() ||
          in.use_begin()->getOwner()->getName().getStringRef() != "dsa.await")
        return WalkResult::interrupt();
      auto projection =
          intentdsl::block::projection(def->getOperand(0).getType());
      if (failed(projection) ||
          def->getAttr("matrix") !=
              b.getBoolAttr(*projection ==
                            intentdsl::block::RowProjection::Matrix))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (count == 0 || walk.wasInterrupted())
    return fn.emitError(
        "phase completion and staged fragment lifetimes differ");
  return verifyLayout();
}
} // namespace intentdsl::dsa
