#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Grid.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringExtras.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
llvm::Error describeArtifact(ModuleOp m, llvm::raw_pwrite_stream &out) {
  SmallVector<func::FuncOp, 2> fns(m.getOps<func::FuncOp>());
  auto fn = fns.back();
  if (fn.front().front().getName().getStringRef() == "ct_src.tile_program" ||
      intentdsl::kir::orderedOwners(fn)) {
    if (auto conv =
            fn.front().front().getAttrOfType<StringAttr>("calling_convention");
        conv && conv.getValue() == "matrix") {
      llvm::json::Object views;
      if (auto attr = fn->getAttrOfType<StringAttr>("ct_src.native_views")) {
        auto value = llvm::json::parse(attr.getValue());
        if (!value)
          return value.takeError();
        if (!value->getAsObject())
          return llvm::createStringError("invalid native array views");
        views = std::move(*value->getAsObject());
      }
      return intentdsl::block::emitMatrixPackage(fn, "", out, 3,
                                                 std::move(views));
    }
    if (fn.front().front().hasAttr("block.pair_axis"))
      return intentdsl::block::emitOwnerArtifact(
          fn, "",
          ArrayRef<llvm::json::Object>{
              llvm::json::Object{{"num_ctas", 1}, {"occupancy", 1}, {"opt_level", 3}},
              llvm::json::Object{{"num_ctas", 1}, {"occupancy", 2}, {"opt_level", 3}},
              llvm::json::Object{{"num_ctas", 1}, {"occupancy", 4}, {"opt_level", 3}}},
          3, out);
    bool hasFetchAdd = false, hasAtomicAdd = false, hasPanelExtract = false;
    fn.walk([&](Operation *op) {
      auto name = op->getName().getStringRef();
      if (name == "ct_src.tile_atomic_fetch_add")
        hasFetchAdd = true;
      else if (name == "ct_src.tile_atomic_add")
        hasAtomicAdd = true;
      else if (name == "ct_src.panel_extract")
        hasPanelExtract = true;
    });
    if (hasFetchAdd)
      return intentdsl::block::emitOwnerArtifact(
          fn, "",
          llvm::json::Object{
              {"num_ctas", 1}, {"occupancy", 4}, {"opt_level", 3}},
          3, out);
    if (hasAtomicAdd)
      return intentdsl::block::emitOwnerArtifact(
          fn, "",
          ArrayRef<llvm::json::Object>{
              llvm::json::Object{{"num_ctas", 1}, {"occupancy", 2}, {"opt_level", 3}},
              llvm::json::Object{{"num_ctas", 1}, {"occupancy", 4}, {"opt_level", 3}}},
          3, out);
    if (fn.front().front().hasAttr("block.token_span") || hasPanelExtract)
      return intentdsl::block::emitOwnerArtifact(
          fn, "",
          ArrayRef<llvm::json::Object>{
              llvm::json::Object{{"num_ctas", 1}, {"occupancy", 2}, {"opt_level", 3}},
              llvm::json::Object{{"num_ctas", 1}, {"occupancy", 4}, {"opt_level", 3}}},
          3, out);
    return intentdsl::block::emitOwnerArtifact(fn, "", llvm::json::Object{}, 3, out);
  }

  auto abi = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!abi)
    return abi.takeError();
  llvm::json::Array rows;
  for (auto a : fn->getAttrOfType<ArrayAttr>("block.configs")) {
    auto d = cast<DictionaryAttr>(a);
    auto hints = llvm::json::parse(d.getAs<StringAttr>("payload").getValue());
    if (!hints)
      return hints.takeError();
    rows.push_back(
        llvm::json::Object{{"id", d.getAs<IntegerAttr>("id").getInt()},
                           {"rows", d.getAs<IntegerAttr>("rows").getInt()},
                           {"width", d.getAs<IntegerAttr>("width").getInt()},
                           {"native", std::move(*hints)}});
  }
  Builder b(fn.getContext());
  auto formula = intentdsl::block::gridExpression(fn->getAttr("block.grid"));
  auto guard = fn->getAttrOfType<DenseI64ArrayAttr>("block.guard");
  out << llvm::json::Value(llvm::json::Object{
      {"domain", intentdsl::publishDomain(fn)},
      {"abi", std::move(*abi)},
      {"version", 3},
      {"column_tiles_parameter",
       fn->getAttr("block.grid") == intentdsl::block::grid(b, true)},
      {"outer_shape_slot",
       cast<MemRefType>(fn.getArgument(intentdsl::domainSlot(fn)).getType())
                   .getRank() > 2
           ? intentdsl::domainSlot(fn)
           : -1},
      {"source", ""},
      {"entry", fn->getAttrOfType<StringAttr>("ct_src.entry").str()},
      {"grid",
       llvm::json::Array{
           std::move(formula),
           std::max<int64_t>(
               1, llvm::count_if(fn.front(),
                                 [](Operation &op) {
                                   return op.getName().getStringRef() ==
                                          "ct_src.output_partition";
                                 })),
           1}},
      {"rows", std::move(rows)},
      {"guard",
       llvm::json::Object{{"min_width", guard[0]},
                          {"max_width", guard[1] < 0
                                            ? llvm::json::Value(nullptr)
                                            : llvm::json::Value(guard[1])}}}});
  return llvm::Error::success();
}
} // namespace CI_UNIT_NAMESPACE
