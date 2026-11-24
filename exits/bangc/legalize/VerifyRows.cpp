#include "Rows.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/execution/block/decisions/Verify.h"
#include "intentdsl/levels/storage/dsa/ir/Matrix.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <set>

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;

void stampScfCarryIds(ModuleOp m) {
  int64_t nextId = 0;
  Builder b(m.getContext());
  m.walk([&](Operation *op) {
    if (!isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op) ||
        op->getNumResults() == 0)
      return;
    auto idAttr = b.getI64IntegerAttr(nextId++);
    op->setAttr("bangc.carry_id", idAttr);
    for (Region &reg : op->getRegions())
      for (Block &blk : reg)
        if (!blk.empty())
          blk.back().setAttr("bangc.carry_id", idAttr);
  });
}

LogicalResult verifyScfCarryIds(ModuleOp m) {
  std::set<int64_t> seenIds;
  bool bad = m.walk([&](Operation *op) {
                if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op)) {
                  unsigned nRes = op->getNumResults();
                  auto idAttr = op->getAttrOfType<IntegerAttr>("bangc.carry_id");
                  if (nRes == 0) {
                    if (idAttr) {
                      op->emitError("unexpected bangc.carry_id on void scf op");
                      return WalkResult::interrupt();
                    }
                    return WalkResult::advance();
                  }
                  if (!idAttr || idAttr.getInt() < 0 ||
                      !seenIds.insert(idAttr.getInt()).second) {
                    op->emitError(
                        "missing or duplicate bangc.carry_id on value-carrying "
                        "scf op");
                    return WalkResult::interrupt();
                  }
                  for (Region &reg : op->getRegions()) {
                    for (Block &blk : reg) {
                      if (blk.empty() ||
                          blk.back().getAttr("bangc.carry_id") != idAttr) {
                        op->emitError(
                            "scf region terminator disagrees with parent "
                            "bangc.carry_id");
                        return WalkResult::interrupt();
                      }
                    }
                  }
                  return WalkResult::advance();
                }
                if (isa<scf::YieldOp>(op) && op->getNumOperands() > 0) {
                  auto *parent = op->getParentOp();
                  if (!parent ||
                      op->getAttr("bangc.carry_id") !=
                          parent->getAttr("bangc.carry_id") ||
                      !op->getAttrOfType<IntegerAttr>("bangc.carry_id")) {
                    op->emitError(
                        "value-carrying scf.yield missing verified "
                        "bangc.carry_id");
                    return WalkResult::interrupt();
                  }
                }
                if (isa<scf::ConditionOp>(op) && op->getNumOperands() > 1) {
                  auto *parent = op->getParentOp();
                  if (!parent ||
                      op->getAttr("bangc.carry_id") !=
                          parent->getAttr("bangc.carry_id") ||
                      !op->getAttrOfType<IntegerAttr>("bangc.carry_id")) {
                    op->emitError(
                        "value-carrying scf.condition missing verified "
                        "bangc.carry_id");
                    return WalkResult::interrupt();
                  }
                }
                return WalkResult::advance();
              }).wasInterrupted();
  return success(!bad);
}

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

  if (failed(verifyScfCarryIds(m)))
    return failure();

  return rowPrinter().verify(m, profile);
}

LogicalResult verifyCallableMatrix(ModuleOp m, const TargetProfile &profile) {
  auto funcs = m.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs))
    return m.emitError("requires one matrix function");
  auto fn = *funcs.begin();
  if (failed(intentdsl::verifyDomain(fn)))
    return failure();
  if (fn->getAttrOfType<StringAttr>("bangc.entry") != "intent_enqueue")
    return fn.emitError("missing published matrix queue entry");
  if (fn.empty() || !llvm::hasSingleElement(fn.getBody()) ||
      fn.front().getOperations().size() != 2)
    return fn.emitError("expected lowered bangc_src.matrix_program and return");
  auto &prog = fn.front().front();
  if (prog.getName().getStringRef() != "bangc_src.matrix_program" ||
      fn.front().back().getName().getStringRef() != "func.return")
    return fn.emitError("expected terminal bangc_src.matrix_program");

  auto bm = prog.getAttrOfType<IntegerAttr>("tile_m");
  auto bn = prog.getAttrOfType<IntegerAttr>("tile_n");
  auto bk = prog.getAttrOfType<IntegerAttr>("tile_k");
  auto extents = prog.getAttrOfType<DenseI64ArrayAttr>("extents");
  auto tasks = prog.getAttrOfType<IntegerAttr>("participants");
  auto scratch = prog.getAttrOfType<IntegerAttr>("scratch_bytes");
  auto capacity = profile.services.get<intentdsl::dsa::LocalSupply>();
  if (!bm || !bn || !bk || !extents || extents.size() != 4 || !tasks ||
      !scratch || !capacity || tasks.getInt() < 1 ||
      tasks.getInt() > capacity->consumers ||
      scratch.getInt() > std::min(capacity->budget, capacity->capacity))
    return prog.emitError("invalid matrix tile, participants or NRAM budget");

  auto ms = profile.services.get<intentdsl::dsa::MatrixStorage>();
  int64_t wramLimit = ms ? ms->wramBytes : 0;

  int64_t totalNram = 0, totalWram = 0;
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
                       if (fullName == "bangc_src.nram_buffer") {
                         auto elems =
                             op->getAttrOfType<IntegerAttr>("elements");
                         auto ty =
                             dyn_cast<MemRefType>(op->getResult(0).getType());
                         if (!elems || !ty)
                           return WalkResult::interrupt();
                         int64_t elemB =
                             llvm::divideCeil(ty.getElementTypeBitWidth(), 8u);
                         totalNram +=
                             llvm::alignTo(elems.getInt() * elemB, 128);
                       }
                       if (fullName == "bangc_src.wram_buffer") {
                         auto elems =
                             op->getAttrOfType<IntegerAttr>("elements");
                         auto ty =
                             dyn_cast<MemRefType>(op->getResult(0).getType());
                         if (!elems || !ty)
                           return WalkResult::interrupt();
                         int64_t elemB =
                             llvm::divideCeil(ty.getElementTypeBitWidth(), 8u);
                         totalWram += elems.getInt() * elemB;
                       }
                       if (fullName == "bangc_src.address") {
                         auto bits =
                             op->getAttrOfType<IntegerAttr>("index_bits");
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

  if (totalNram > std::min(capacity->budget, capacity->capacity))
    return prog.emitError("matrix NRAM buffers exceed private capacity");
  if (totalWram > 0 && wramLimit > 0 && totalWram > wramLimit)
    return prog.emitError("matrix WRAM buffers exceed filter capacity");
  if (declaredResources != requiredResources)
    return prog.emitError(
        "runtime_source declarations disagree with bound implementations");

  auto abiJson =
      llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi").getValue());
  if (!abiJson)
    return llvm::consumeError(abiJson.takeError()),
           fn.emitError("invalid matrix ABI JSON");
  uint64_t grid =
      llvm::divideCeil(uint64_t(extents[0]), uint64_t(bm.getInt())) *
      llvm::divideCeil(uint64_t(extents[1]), uint64_t(bn.getInt())) *
      uint64_t(extents[3]);
  llvm::SmallVector<char> expected;
  llvm::raw_svector_ostream os(expected);
  os << llvm::json::Value(llvm::json::Object{
      {"version", 2},
      {"source", ""},
      {"abi", std::move(*abiJson)},
      {"domain", intentdsl::publishDomain(fn)},
      {"grid", llvm::json::Array{int64_t(grid), 1, 1}},
      {"guard", llvm::json::Object{{"min_width", 0}, {"max_width", nullptr}}},
      {"rows",
       llvm::json::Array{llvm::json::Object{{"id", 0},
                                            {"rows", bm.getInt()},
                                            {"width", bn.getInt()},
                                            {"pitch", bk.getInt()},
                                            {"bytes", scratch.getInt()},
                                            {"tasks", tasks.getInt()}}}}});
  auto package = m->getAttrOfType<StringAttr>("bangc_src.package");
  if (!package ||
      package.getValue() != StringRef(expected.data(), expected.size()))
    return fn.emitError(
        "BANG C matrix package disagrees with verified ABI or tile row");

  if (failed(verifyScfCarryIds(m)))
    return failure();

  return rowPrinter().verify(m, profile);
}

LogicalResult verifyCallableOrdered(ModuleOp m, const TargetProfile &profile) {
  auto funcs = m.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs))
    return m.emitError("requires one ordered function");
  auto fn = *funcs.begin();
  if (failed(intentdsl::verifyDomain(fn)))
    return failure();
  if (fn->getAttrOfType<StringAttr>("bangc.entry") != "intent_enqueue")
    return fn.emitError("missing published ordered queue entry");
  if (fn.empty() || !llvm::hasSingleElement(fn.getBody()) ||
      fn.front().getOperations().size() != 2)
    return fn.emitError("expected lowered bangc_src.ordered_program and return");
  auto &prog = fn.front().front();
  if (prog.getName().getStringRef() != "bangc_src.ordered_program" ||
      fn.front().back().getName().getStringRef() != "func.return")
    return fn.emitError("expected terminal bangc_src.ordered_program");

  auto tile = prog.getAttrOfType<IntegerAttr>("owner_tile");
  auto owners = prog.getAttrOfType<IntegerAttr>("owners");
  auto tasks = prog.getAttrOfType<IntegerAttr>("participants");
  auto ownerBytes = prog.getAttrOfType<IntegerAttr>("owner_bytes");
  auto capacity = profile.services.get<intentdsl::dsa::LocalSupply>();
  if (!tile || !owners || !tasks || !ownerBytes || !capacity ||
      tile.getInt() < 1 || owners.getInt() < 0 || tasks.getInt() < 1 ||
      tasks.getInt() > capacity->consumers || ownerBytes.getInt() < 0 ||
      ownerBytes.getInt() > std::min(capacity->budget, capacity->capacity))
    return prog.emitError(
        "invalid ordered owner tile, participants or slab capacity");

  auto ms = profile.services.get<intentdsl::dsa::MatrixStorage>();
  int64_t wramLimit = ms ? ms->wramBytes : 0;

  int64_t totalNram = 0, totalWram = 0;
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
                       if (fullName == "bangc_src.nram_buffer") {
                         auto elems =
                             op->getAttrOfType<IntegerAttr>("elements");
                         auto ty =
                             dyn_cast<MemRefType>(op->getResult(0).getType());
                         if (!elems || !ty)
                           return WalkResult::interrupt();
                         int64_t elemB =
                             llvm::divideCeil(ty.getElementTypeBitWidth(), 8u);
                         totalNram +=
                             llvm::alignTo(elems.getInt() * elemB, 128);
                       }
                       if (fullName == "bangc_src.wram_buffer") {
                         auto elems =
                             op->getAttrOfType<IntegerAttr>("elements");
                         auto ty =
                             dyn_cast<MemRefType>(op->getResult(0).getType());
                         if (!elems || !ty)
                           return WalkResult::interrupt();
                         int64_t elemB =
                             llvm::divideCeil(ty.getElementTypeBitWidth(), 8u);
                         totalWram += elems.getInt() * elemB;
                       }
                       if (fullName == "bangc_src.address") {
                         auto bits =
                             op->getAttrOfType<IntegerAttr>("index_bits");
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

  if (totalNram > std::min(capacity->budget, capacity->capacity))
    return prog.emitError("ordered NRAM buffers exceed private capacity");
  if (totalWram > 0 && wramLimit > 0 && totalWram > wramLimit)
    return prog.emitError("ordered WRAM buffers exceed filter capacity");
  if (declaredResources != requiredResources)
    return prog.emitError(
        "runtime_source declarations disagree with bound implementations");

  auto abiJson =
      llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi").getValue());
  if (!abiJson)
    return llvm::consumeError(abiJson.takeError()),
           fn.emitError("invalid ordered ABI JSON");
  llvm::SmallVector<char> expected;
  llvm::raw_svector_ostream os(expected);
  os << llvm::json::Value(llvm::json::Object{
      {"version", 2},
      {"source", ""},
      {"abi", std::move(*abiJson)},
      {"domain", intentdsl::publishDomain(fn)},
      {"grid",
       llvm::json::Array{
           int64_t(llvm::divideCeil(owners.getInt(), tile.getInt())), 1, 1}},
      {"guard", llvm::json::Object{{"min_width", 0}, {"max_width", nullptr}}},
      {"rows",
       llvm::json::Array{llvm::json::Object{{"id", 0},
                                            {"rows", 1},
                                            {"width", 1},
                                            {"pitch", 1},
                                            {"bytes", ownerBytes.getInt()},
                                            {"tasks", tasks.getInt()}}}}});
  auto package = m->getAttrOfType<StringAttr>("bangc_src.package");
  if (!package ||
      package.getValue() != StringRef(expected.data(), expected.size()))
    return fn.emitError(
        "BANG C ordered package disagrees with verified ABI or launch row");

  if (failed(verifyScfCarryIds(m)))
    return failure();

  return rowPrinter().verify(m, profile);
}
} // namespace CI_UNIT_NAMESPACE
