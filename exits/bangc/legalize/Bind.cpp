#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
llvm::Expected<TargetProfile> capabilities(const llvm::json::Object &);
llvm::Error emit(ModuleOp, llvm::raw_pwrite_stream &);
namespace {
StringRef primitive(Operation *op, int64_t depth) {
  auto n = op->getName().getStringRef();
  if (n == "dsa.phase")
    return "block_tasks";
  if (n == "dsa.copy_in")
    return depth == 2 ? "dma_async" : "dma";
  if (n == "dsa.await")
    return depth == 2 ? "wait_slot" : "ready";
  if (n == "dsa.write_back")
    return "store_extent";
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
      if (o.getName().getStringRef() != "dsa.copy_in" || !attach(o, grouped, 0))
        return WalkResult::interrupt();
      grouped +=
          cast<MemRefType>(o.getOperand(0).getType()).getElementTypeBitWidth() /
          8;
    }
    auto &compute = p->getRegion(1).front();
    for (auto &o : compute) {
      if (o.getNumResults() != 1 || o.getName().getStringRef() == "dsa.await")
        continue;
      auto t = dyn_cast<RankedTensorType>(o.getResult(0).getType());
      if (!t || t.getRank() != 2)
        continue;
      if (!attach(o, grouped, local))
        return WalkResult::interrupt();
      local += t.getElementTypeBitWidth() / 8;
    }
    for (auto &o : compute)
      if (o.getName().getStringRef() == "dsa.reduce" &&
          !attach(o, grouped, local))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (phases.wasInterrupted())
    return f.emitError("private slot byte binding disagrees with its phase");
  return success();
}
struct Legalize : PassWrapper<Legalize, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Legalize)
  unsigned phase;
  explicit Legalize(unsigned step) : phase(step) {}
  void runOnOperation() final {
    auto f = *getOperation().getOps<func::FuncOp>().begin();
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
      f->setAttr("bangc.entry", b.getStringAttr("intent_enqueue"));
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
        slot->getArray("dims")->size() != unsigned(rank) || rank > 2 ||
        (!t.isF32() && !t.isF16()) ||
        slot->getString("dtype") != (t.isF16() ? "f16" : "f32") ||
        (view ? (slot->getString("direction") != "In" &&
                 slot->getString("direction") != "Out")
              : (slot->getString("direction") != "Scalar" || !t.isF32())))
      return fn.emitError(
          "row ABI dtype, extent or direction inconsistent with function");
  }
  auto access = fn.walk([&](Operation *o) {
    bool read = o->getName().getStringRef() == "dsa.copy_in";
    if (!read && o->getName().getStringRef() != "dsa.write_back")
      return WalkResult::advance();
    auto v = dyn_cast<BlockArgument>(o->getOperand(0));
    if (!v || v.getOwner() != &fn.front() ||
        (*args)[v.getArgNumber()].getAsObject()->getString("direction") !=
            (read ? "In" : "Out")) {
      o->emitError("native transfer conflicts with public slot direction");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success(!access.wasInterrupted());
}
LogicalResult verify(ModuleOp m, const TargetProfile &p, unsigned stage) {
  if (failed(intentdsl::dsa::verifyProgram(m, p, true)))
    return failure();
  auto fn = *m.getOps<func::FuncOp>().begin();
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
void bind(Unit &u) {
  unsigned stage = 0;
  for (StringRef name : {"bound", "legal", "callable"}) {
    u.boundaries.push_back({{name.str(),
                             [stage](OpPassManager &pm, const TargetProfile &) {
                               pm.addPass(std::make_unique<Legalize>(stage));
                               return llvm::Error::success();
                             },
                             [stage](ModuleOp m, const TargetProfile &p) {
                               return verify(m, p, stage);
                             }},
                            COMPILERINFRA_SOURCE});
    ++stage;
  }
  u.exit =
      Located<ExitImplementation>{{emit, capabilities}, COMPILERINFRA_SOURCE};
}
} // namespace CI_UNIT_NAMESPACE
