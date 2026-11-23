#include "Rows.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/execution/block/decisions/Verify.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include <set>

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;

namespace {
LogicalResult verifyAbi(func::FuncOp fn) {
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
         !(view && t.isSignlessInteger(64)) &&
         (view || !t.isUnsignedInteger(64))) ||
        slot->getString("dtype") !=
            (t.isF16()                 ? "f16"
             : t.isBF16()              ? "bf16"
             : t.isInteger(32)         ? "i32"
             : t.isSignlessInteger(64) ? "i64"
             : t.isUnsignedInteger(64) ? "u64"
                                       : "f32") ||
        (view ? (slot->getString("direction") != "In" &&
                 slot->getString("direction") != "Out" &&
                 slot->getString("direction") != "InOut")
              : (slot->getString("direction") != "Scalar" ||
                 (!t.isF32() && !t.isInteger(32) &&
                  !t.isUnsignedInteger(64)))))
      return fn.emitError(
          "row ABI dtype, extent or direction inconsistent with function");
  }
  return success();
}
} // namespace

LogicalResult verifyCallableRows(ModuleOp m, const TargetProfile &profile) {
  auto funcs = m.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs))
    return m.emitError("requires one tile function");
  auto fn = *funcs.begin();
  if (failed(intentdsl::verifyDomain(fn)) || failed(verifyAbi(fn)))
    return failure();
  if (auto err = intentdsl::verifyRows(
          fn, fn->getAttrOfType<ArrayAttr>("block.configs"),
          fn->getAttrOfType<StringAttr>("block.topology"),
          fn->getAttrOfType<DenseI64ArrayAttr>("block.guard"), profile))
    return fn.emitError() << llvm::toString(std::move(err));
  auto capacity = profile.services.get<intentdsl::dsa::LocalSupply>();
  auto layout = fn->getAttrOfType<DictionaryAttr>("dsa.layout");
  if (!capacity || !layout || !layout.getAs<ArrayAttr>("rows") ||
      layout.getAs<ArrayAttr>("rows").empty() ||
      fn->getAttrOfType<StringAttr>("bangc.space") != "private.nram" ||
      fn->getAttrOfType<StringAttr>("bangc.participant") != "block")
    return fn.emitError("missing private NRAM layout or participant mapping");
  for (auto a : fn->getAttrOfType<ArrayAttr>("block.configs")) {
    unsigned tasks = 0;
    if (cast<DictionaryAttr>(a)
            .getAs<StringAttr>("payload")
            .getValue()
            .getAsInteger(10, tasks) ||
        !tasks || tasks > capacity->consumers)
      return fn.emitError("row exceeds available physical participants");
  }
  if (fn.empty() || !llvm::hasSingleElement(fn.getBody()) ||
      fn.front().getOperations().size() != 2)
    return fn.emitError("expected lowered bangc_src.row_program and return");
  auto &prog = fn.front().front();
  if (prog.getName().getStringRef() != "bangc_src.row_program" ||
      fn.front().back().getName().getStringRef() != "func.return")
    return fn.emitError("expected terminal bangc_src.row_program");

  llvm::SmallVector<char> expected;
  llvm::raw_svector_ostream os(expected);
  if (auto err = describeRowArtifact(fn, os))
    return fn.emitError() << llvm::toString(std::move(err)), failure();
  auto package = m->getAttrOfType<StringAttr>("bangc_src.package");
  if (!package ||
      package.getValue() != StringRef(expected.data(), expected.size()))
    return fn.emitError(
        "BANG C package disagrees with verified ABI or specialization rows");

  std::set<std::string> declaredResources, requiredResources;
  bool walkFailed = m.walk([&](Operation *op) {
                       auto ns = op->getName().getDialectNamespace();
                       if (ns != "builtin" && ns != "func" &&
                           ns != "bangc_src" && ns != "scf" && ns != "arith") {
                         op->emitError("unlowered non-terminal operation in "
                                       "bangc@callable");
                         return WalkResult::interrupt();
                       }
                       auto fullName = op->getName().getStringRef();
                       if (fullName == "bangc_src.address") {
                         auto bits = op->getAttrOfType<IntegerAttr>("index_bits");
                         if (!bits ||
                             (bits.getInt() != 64 &&
                              (bits.getInt() != 32 ||
                               !canNarrowAddress(*op, fn)))) {
                           op->emitError("address requires i64 or proven i32 "
                                         "signed bounds");
                           return WalkResult::interrupt();
                         }
                       }
                       if (fullName == "bangc_src.runtime_source") {
                         auto key =
                             op->getAttrOfType<StringAttr>("implementation");
                         if (!key)
                           return WalkResult::interrupt();
                         auto it = profile.implementations.find(
                             key.getValue().str());
                         if (it == profile.implementations.end() ||
                             it->second.source !=
                                 ImplementationSource::Runtime ||
                             !declaredResources.insert(it->second.resource)
                                  .second) {
                           op->emitError(
                               "invalid or duplicate runtime_source binding");
                           return WalkResult::interrupt();
                         }
                       }
                       if (auto impl =
                               op->getAttrOfType<StringAttr>("ci.implementation")) {
                         auto it = profile.implementations.find(
                             impl.getValue().str());
                         if (it != profile.implementations.end() &&
                             it->second.source == ImplementationSource::Runtime)
                           requiredResources.insert(it->second.resource);
                       }
                       return WalkResult::advance();
                     }).wasInterrupted();
  if (walkFailed)
    return failure();

  if (declaredResources != requiredResources)
    return prog.emitError(
        "runtime_source declarations disagree with bound implementations");

  return rowPrinter().verify(m, profile);
}
} // namespace CI_UNIT_NAMESPACE
