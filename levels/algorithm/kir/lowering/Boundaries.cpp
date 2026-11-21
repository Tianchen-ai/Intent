#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace compilerinfra;
using namespace mlir;
namespace intentdsl {
LogicalResult verifyLogicalRows(ModuleOp module,StringRef dialect,bool canonical) {
  auto funcs = module.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs))
    return module.emitError("expected one author kernel");
  auto fn = *funcs.begin();
  if (fn.empty() || !llvm::hasSingleElement(fn.getBody()) || fn.getNumResults())
    return fn.emitError("expected a defined output-buffer kernel");
  auto abi = fn->getAttrOfType<StringAttr>("kir.abi");
  if (!abi)
    return fn.emitError("missing author ABI");
  auto parsed = llvm::json::parse(abi);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return fn.emitError("invalid ABI JSON");
  }
  auto *args = parsed->getAsArray();
  if (!args || args->size() != fn.getNumArguments())
    return fn.emitError("signature length mismatch");
  for (auto [arg, entry] : llvm::zip(fn.getArguments(), *args)) {
    auto *desc = entry.getAsObject();
    if (!desc || !desc->getArray("dims") || !desc->getString("direction") ||
        !desc->getString("name"))
      return fn.emitError("invalid parameter descriptor");
    Type elt = arg.getType();
    size_t rank = 0;
    if (auto view = dyn_cast<MemRefType>(elt)) {
      rank = view.getRank();
      elt = view.getElementType();
    }
    if (rank != desc->getArray("dims")->size() ||
        (!elt.isF16() && !elt.isF32()) ||
        desc->getString("dtype") != (elt.isF16() ? "f16" : "f32"))
      return fn.emitError("parameter descriptor does not match its IR type");
  }
  auto &body = fn.front();
  if (body.getOperations().size() != 2 ||
      body.front().getName().getStringRef() != (dialect+".parallel").str())
    return fn.emitError("expected a parallel row domain and return");
  auto &loop = body.front();
  auto domain = dyn_cast<MemRefType>(loop.getOperand(0).getType());
  if (!domain || domain.getRank() != 2 ||
      !llvm::hasSingleElement(loop.getRegion(0)))
    return loop.emitError("expected a rank-two domain");
  auto &row = loop.getRegion(0).front();
  if (row.getNumArguments() != 1 || !row.getArgument(0).getType().isIndex() ||
      row.empty() || row.back().getName().getStringRef() != (dialect+".end").str())
    return loop.emitError("invalid row region");
  auto domainArg = dyn_cast<BlockArgument>(loop.getOperand(0));
  if (!domainArg || domainArg.getOwner() != &body)
    return loop.emitError("expected domain argument");
  auto &axes =
      *(*args)[domainArg.getArgNumber()].getAsObject()->getArray("dims");
  llvm::SmallDenseSet<unsigned> written;
  for (auto &op : row) {
    auto name = op.getName().stripDialect();
    if (op.getName().getDialectNamespace() != dialect)
      return op.emitError("unexpected row operation");
    if(name!="read" && name!="write" && name!="end" && name!="sum" && name!="max_num" &&
       name!="add" && name!="sub" && name!="mul" && name!="div" && name!="cast" && name!="exp" && name!="rsqrt")
      return op.emitError("operation outside logical row representation");
    if (name == "read" || name == "write") {
      auto view = cast<MemRefType>(op.getOperand(0).getType());
      auto arg = dyn_cast<BlockArgument>(op.getOperand(0));
      if (!arg || arg.getOwner() != &body)
        return op.emitError("expected an external view");
      bool write = name == "write";
      auto *info = (*args)[arg.getArgNumber()].getAsObject();
      if (info->getString("direction") != (write ? "Out" : "In") ||
          view.getRank() < 1 || view.getRank() > 2 ||
          op.getNumOperands() != unsigned(view.getRank() + write) ||
          (view.getRank() == 2 && op.getOperand(1) != row.getArgument(0)))
        return op.emitError("invalid row access");
      const auto &dims = *info->getArray("dims");
      if (dims.back() != axes.back() ||
          (view.getRank() == 2 && dims.front() != axes.front()))
        return op.emitError("view does not share the row domain");
      if (write)
        written.insert(arg.getArgNumber());
      auto type = write ? op.getOperand(op.getNumOperands() - 1).getType()
                        : op.getResult(0).getType();
      if (type !=
          RankedTensorType::get({ShapedType::kDynamic}, view.getElementType()))
        return op.emitError("row element type mismatch");
    } else if (name == "sum" || name == "max_num") {
      auto t = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
      if (!t || t.getRank() != 1 || !t.getElementType().isF32() ||
          !op.getResult(0).getType().isF32())
        return op.emitError("expected f32 row to scalar reduction");
    } else if (name == "end")
      continue;
    else {
      Type result = op.getResult(0).getType();
      auto element = [](Type t) {
        auto r = dyn_cast<RankedTensorType>(t);
        return r ? r.getElementType() : t;
      };
      if (!element(result).isF32() && !element(result).isF16())
        return op.emitError("expected floating point result");
      bool vector = false;
      for (Value x : op.getOperands()) {
        auto tensor = dyn_cast<RankedTensorType>(x.getType());
        vector |= bool(tensor);
        if ((tensor && tensor.getRank() != 1) ||
            (!element(x.getType()).isF32() && !element(x.getType()).isF16()) ||
            (name != "cast" && element(x.getType()) != element(result)))
          return op.emitError("incompatible operand dtype or rank");
      }
      if (vector != (isa<RankedTensorType>(result)) ||
          (vector && cast<RankedTensorType>(result).getRank() != 1))
        return op.emitError("result does not preserve the row shape");
    }
    if (canonical && name == "cast" &&
        op.getOperand(0).getType() == op.getResult(0).getType())
      return op.emitError("uncanonicalized identity cast");
  }
  for (auto [i, arg] : llvm::enumerate(*args))
    if (arg.getAsObject()->getString("direction") == "Out" &&
        !written.contains(i))
      return fn.emitError("declared output has no write");
  return success();
}

} // namespace intentdsl
namespace {
struct Canonicalize : PassWrapper<Canonicalize, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Canonicalize)
  void runOnOperation() final {
    getOperation().walk([](Operation *op) {
      if (op->getName().getStringRef() == "kir.cast" &&
          op->getOperand(0).getType() == op->getResult(0).getType()) {
        op->getResult(0).replaceAllUsesWith(op->getOperand(0));
        op->erase();
      }
    });
  }
};
} // namespace
namespace CI_UNIT_NAMESPACE {
void bind(Unit &unit) {
  unit.dialects = [](DialectRegistry &r) {
    r.insert<intentdsl::kir::KirDialect, func::FuncDialect>();
  };
  for (bool canonical : {false, true})
    unit.boundaries.push_back(
        {{canonical ? "canonical" : "imported",
          [canonical](OpPassManager &pm, const TargetProfile &) {
            if (canonical)
              pm.addPass(std::make_unique<Canonicalize>());
            return llvm::Error::success();
          },
          [canonical](ModuleOp m, const TargetProfile &) {
            return intentdsl::verifyLogicalRows(m,"kir",canonical);
          }},
         COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
