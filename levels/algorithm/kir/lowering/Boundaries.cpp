#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace compilerinfra;
using namespace mlir;
namespace intentdsl {
LogicalResult verifyRowComputation(func::FuncOp fn, Block &row, ValueRange captures, unsigned domainIndex, StringRef dialect, bool canonical) {
  auto abi = fn->getAttrOfType<StringAttr>("kir.abi");
  if (!abi)
    return fn.emitError("missing author ABI");
  auto parsed = llvm::json::parse(abi);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return fn.emitError("invalid ABI JSON");
  }
  auto *args = parsed->getAsArray();
  if (!args || args->size() != fn.getNumArguments() || captures.size()!=args->size() || domainIndex>=args->size())
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
    auto direction = *desc->getString("direction");
    if ((rank && direction != "In" && direction != "Out" &&
         direction != "InOut") ||
        (!rank && direction != "Scalar"))
      return fn.emitError("direction disagrees with the scalar or view type");
    if (rank != desc->getArray("dims")->size() ||
        (!elt.isF16() && !elt.isF32() && !elt.isBF16() && !elt.isInteger(32)) ||
        desc->getString("dtype") != (elt.isF16()         ? "f16"
                                     : elt.isBF16()      ? "bf16"
                                     : elt.isInteger(32) ? "i32"
                                                         : "f32"))
      return fn.emitError("parameter descriptor does not match its IR type");
  }
  auto domainType=dyn_cast<MemRefType>(captures[domainIndex].getType());
  if (!domainType || (domainType.getRank() != 2 && domainType.getRank() != 1))
    return fn.emitError("expected a matrix row domain");
  auto &axes =
      *(*args)[domainIndex].getAsObject()->getArray("dims");
  llvm::SmallDenseSet<unsigned> written;
  bool sawWrite = false;
  for (auto &op : row) {
    auto name = op.getName().stripDialect();
    if (op.getName().getDialectNamespace() != dialect)
      return op.emitError("unexpected row operation");
    if (name == "constant") {
      auto v = op.getAttrOfType<TypedAttr>("value");
      if (!v || v.getType() != op.getResult(0).getType())
        return op.emitError("typed literal mismatch");
      continue;
    }
    if (name == "ordinal") {
      if (op.getOperand(0) != captures[domainIndex] ||
          op.getOperand(1) != row.getArgument(0) ||
          op.getResult(0).getType() !=
              RankedTensorType::get({ShapedType::kDynamic},
                                    IntegerType::get(fn.getContext(), 32)))
        return op.emitError(
            "ordinal must enumerate the declared row in the owning domain");
      continue;
    }
    if (name != "sqrt" && name != "and" && name != "xor" && name != "shl" &&
        name != "shr" && name != "gt" && name != "select" &&
        name != "maximum" && name != "read" && name != "write" &&
        name != "end" && name != "sum" && name != "max_num" && name != "add" &&
        name != "sub" && name != "mul" && name != "div" && name != "cast" &&
        name != "exp" && name != "rsqrt")
      return op.emitError("operation outside logical row representation");
    if (name == "read" || name == "write") {
      auto view = cast<MemRefType>(op.getOperand(0).getType());
      auto arg = dyn_cast<BlockArgument>(op.getOperand(0));
      auto slot = llvm::find(captures, op.getOperand(0));
      if (!arg || slot == captures.end())
        return op.emitError("expected an external view");
      bool write = name == "write";
      if (!write && sawWrite &&
          llvm::any_of(*args, [](const llvm::json::Value &v) {
            return v.getAsObject()->getString("direction") == "InOut";
          }))
        return op.emitError(
            "dependent read after a write requires an ordered region");
      sawWrite |= write;
      size_t index = slot - captures.begin();
      auto *info = (*args)[index].getAsObject();
      if ((info->getString("direction") != (write ? "Out" : "In") &&
           info->getString("direction") != "InOut") ||
          view.getRank() < 1 || view.getRank() > 2 ||
          op.getNumOperands() !=
              unsigned((domainType.getRank() == 1 ? 2 : view.getRank()) +
                       write) ||
          ((view.getRank() == 2 || domainType.getRank() == 1) &&
           op.getOperand(1) != row.getArgument(0)))
        return op.emitError("invalid row access");
      const auto &dims = *info->getArray("dims");
      if (auto perm = op.getAttrOfType<DenseI64ArrayAttr>("axes")) {
        if (!write || view.getRank() != 2 ||
            perm.asArrayRef() != ArrayRef<int64_t>({1, 0}) ||
            dims[0] != axes[1] || dims[1] != axes[0])
          return op.emitError("invalid transposed member access");
      } else if (dims.back() != axes.back() ||
                 (view.getRank() == 2 && dims.front() != axes.front()))
        return op.emitError("view does not share the row domain");
      if (write)
        written.insert(index);
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
      auto valueType = element(name == "select" ? op.getOperand(1).getType()
                                                : op.getOperand(0).getType());
      auto allowed = [](Type t) {
        return t.isF32() || t.isF16() || t.isBF16() || t.isInteger(32);
      };
      if (!allowed(valueType) ||
          (name == "gt" ? !element(result).isInteger(1)
                        : !allowed(element(result))) ||
          ((name == "xor" || name == "and" || name == "shr" || name == "shl") &&
           !valueType.isInteger(32)) ||
          (name != "gt" && name != "cast" && element(result) != valueType) ||
          ((name == "sqrt" || name == "rsqrt" || name == "exp" ||
            name == "div") &&
           !isa<FloatType>(valueType)))
        return op.emitError("invalid scalar algebra or result dtype");
      bool vector = false;
      for (auto [i, x] : llvm::enumerate(op.getOperands())) {
        auto tensor = dyn_cast<RankedTensorType>(x.getType());
        vector |= bool(tensor);
        if ((tensor && tensor.getRank() != 1) ||
            (name == "select" && !i ? !element(x.getType()).isInteger(1)
                                    : element(x.getType()) != valueType))
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
LogicalResult verifyLogicalRows(ModuleOp module,StringRef dialect,bool canonical) {
  auto funcs = module.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs))
    return module.emitError("expected one author kernel");
  auto fn = *funcs.begin();
  if (fn.empty() || !llvm::hasSingleElement(fn.getBody()) || fn.getNumResults())
    return fn.emitError("expected a defined output-buffer kernel");
  auto &body = fn.front();
  if (body.getOperations().size() != 2 ||
      body.front().getName().getStringRef() != (dialect+".parallel").str())
    return fn.emitError("expected a parallel row domain and return");
  auto &loop = body.front();
  auto domain = dyn_cast<MemRefType>(loop.getOperand(0).getType());
  if (!domain || (domain.getRank() != 2 && domain.getRank() != 1) ||
      !llvm::hasSingleElement(loop.getRegion(0)))
    return loop.emitError("expected a rank-two domain");
  auto &row = loop.getRegion(0).front();
  if (row.getNumArguments() != 1 || !row.getArgument(0).getType().isIndex() ||
      row.empty() || row.back().getName().getStringRef() != (dialect+".end").str())
    return loop.emitError("invalid row region");
  auto domainArg = dyn_cast<BlockArgument>(loop.getOperand(0));
  if (!domainArg || domainArg.getOwner() != &body)
    return loop.emitError("expected domain argument");
  return verifyRowComputation(fn,row,fn.getArguments(),domainArg.getArgNumber(),dialect,canonical);
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
