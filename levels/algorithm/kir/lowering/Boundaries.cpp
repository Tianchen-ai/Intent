#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Dialect.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"
#include "intentdsl/levels/algorithm/kir/ir/Matrix.h"
#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "intentdsl/levels/algorithm/kir/ir/Segment.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
        (!elt.isF16() && !elt.isF32() && !elt.isBF16() && !elt.isInteger(32) &&
         !(rank == 0 && elt.isUnsignedInteger(64)) && !(rank && elt.isSignlessInteger(64))) ||
        desc->getString("dtype") != (elt.isF16()                 ? "f16"
                                     : elt.isBF16()              ? "bf16"
                                     : elt.isUnsignedInteger(64) ? "u64"
                                     : elt.isSignlessInteger(64) ? "i64"
                                     : elt.isInteger(32)         ? "i32"
                                                                 : "f32"))
      return fn.emitError("parameter descriptor does not match its IR type");
  }
  bool indexed=fn->hasAttr("kir.iteration");
  if(failed(verifyDomain(fn)) || domainSlot(fn)!=domainIndex)return fn.emitError("logical domain slot differs from ownership");
  auto domainType=dyn_cast<MemRefType>(captures[domainIndex].getType());
  auto ownerAxis=fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
  if (!domainType || (ownerAxis && (ownerAxis.getInt()<0 || ownerAxis.getInt()>=domainType.getRank())) ||
      (!ownerAxis && !indexed && domainType.getRank() != 2 && domainType.getRank() != 1))
    return fn.emitError("expected a matrix row domain");
  auto &axes =
      *(*args)[domainIndex].getAsObject()->getArray("dims");
  llvm::SmallDenseSet<unsigned> written;
  bool sawWrite = false;
  for (auto &op : row) {
    auto name = op.getName().stripDialect();
    if (op.getName().getDialectNamespace() != dialect)
      return op.emitError("unexpected row operation");
    if (name == "welford" || name == "field") {
      auto operand = op.getOperand(0).getType();
      auto result = op.getResult(0).getType();
      auto state = dyn_cast<TupleType>(name=="field" ? operand : result);
      if (!state || state.size()!=3 || !state.getType(0).isInteger(32) ||
          !state.getType(1).isF32() || !state.getType(2).isF32())
        return op.emitError("requires (i32 count, f32 mean, f32 M2)");
      if (name=="field") {
        auto pos=op.getAttrOfType<IntegerAttr>("position");
        if (!pos || pos.getInt()<0 || pos.getInt()>=3 || result!=state.getType(pos.getInt()))
          return op.emitError("invalid logical tuple projection");
      } else {
        auto samples=dyn_cast<RankedTensorType>(operand);
        if (!samples || samples.getRank()!=1 || !samples.getElementType().isF32())
          return op.emitError("expected row of f32 samples");
      }
      continue;
    }
    if (name == "constant") {
      auto v = op.getAttrOfType<TypedAttr>("value");
      if (!v || v.getType() != op.getResult(0).getType())
        return op.emitError("typed literal mismatch");
      continue;
    }
    if (name == "uniform") {
      auto counter = dyn_cast<RankedTensorType>(op.getOperand(1).getType());
      auto result = dyn_cast<RankedTensorType>(op.getResult(0).getType());
      if (!counter || !result || counter.getRank() != 1 ||
          !counter.getElementType().isUnsignedInteger(64) ||
          result.getShape() != counter.getShape() ||
          !result.getElementType().isF32())
        return op.emitError(
            "uniform requires one unsigned logical counter per f32 sample");
      continue;
    }
    if (name == "scan_sum") {
      auto t=dyn_cast<RankedTensorType>(op.getOperand(0).getType());
      if(!t || t.getRank()!=1 || (!t.getElementType().isF32() && !t.getElementType().isInteger(32)) ||
         op.getResult(0).getType()!=t)return op.emitError("expected inclusive prefix sum of one member vector");
      continue;
    }
    if(name=="count_keys" || name=="column_sum"){
      bool counting=name=="count_keys";
      auto slot=llvm::find(captures,op.getOperand(0));
      if(slot==captures.end() || op.getOperand(1)!=row.getArgument(0))return op.emitError("expected owned output and captured input");
      auto *input=(*args)[slot-captures.begin()].getAsObject();auto dims=input->getArray("dims");
      if(input->getString("direction")!="In" || !dims || axes.size()!=(counting?2:1) || dims->size()!=(counting?1:2) ||
         (counting ? (op.getAttrOfType<IntegerAttr>("span").getInt()<=0 || (input->getString("dtype")!="f32" && input->getString("dtype")!="i64" && input->getString("dtype")!="i32")) :
                     (input->getString("dtype")!="i32" || dims->back()!=axes.back())))
        return op.emitError("invalid key partition or column reduction domain");
      continue;
    }
    if(name=="ordered_sum" || name=="segment_length"){
      auto slot=llvm::find(captures,op.getOperand(0));
      if(slot==captures.end() || op.getOperand(1)!=row.getArgument(0) || axes.empty() || axes.size()>2)
        return op.emitError("segmented computation requires a row owner");
      const auto *desc=(*args)[slot-captures.begin()].getAsObject();
      auto od=desc->getArray("dims");
      if(!od || od->size()!=1 || desc->getString("dtype")!="i32" || desc->getString("direction")!="In" ||
         !axes.front().getAsInteger() || !od->front().getAsInteger() || *od->front().getAsInteger()<=*axes.front().getAsInteger())
        return op.emitError("offsets must provide both endpoints for each row");
      if(name=="segment_length")continue;
      if(failed(kir::verifyMemberExpression(&op,[&](Value v){
        auto slot=llvm::find(captures,v);
        return slot!=captures.end() && (*args)[slot-captures.begin()].getAsObject()->getString("direction")=="In";
      })))return failure();
      continue;
    }
    if(name=="gather_row"){
      auto src=llvm::find(captures,op.getOperand(0)),idx=llvm::find(captures,op.getOperand(1));
      if(src==captures.end() || idx==captures.end() || op.getOperand(2)!=row.getArgument(0) || axes.size()!=2)
        return op.emitError("gather must index a captured table from a row owner");
      auto *table=(*args)[src-captures.begin()].getAsObject(),*indices=(*args)[idx-captures.begin()].getAsObject();
      auto &ts=*table->getArray("dims"),&is=*indices->getArray("dims");
      if(ts.size()!=2 || is.size()!=1 || ts[1]!=axes[1] || is[0]!=axes[0] || table->getString("direction")!="In" ||
         indices->getString("direction")!="In" || (indices->getString("dtype")!="i32" && indices->getString("dtype")!="i64") ||
         !op.hasAttr("in_bounds") || op.getResult(0).getType()!=RankedTensorType::get({ShapedType::kDynamic},cast<MemRefType>(op.getOperand(0).getType()).getElementType()))
        return op.emitError("invalid table gather contract");
      continue;
    }
    if(name=="prefix_scatter"){
      auto out=llvm::find(captures,op.getOperand(0)),cnt=llvm::find(captures,op.getOperand(1));
      if(out==captures.end() || cnt==captures.end() || out==cnt || op.getOperand(2)!=row.getArgument(0) || axes.size()!=2)
        return op.emitError("scatter requires two owned output views");
      auto o=out-captures.begin(),c=cnt-captures.begin();
      auto *dv=(*args)[o].getAsObject(),*cv=(*args)[c].getAsObject();
      auto *prefix=op.getOperand(4).getDefiningOp();
      auto *toInt=prefix && prefix->getNumOperands()==1?prefix->getOperand(0).getDefiningOp():nullptr;
      if(dv->getString("direction")!="Out" || cv->getString("direction")!="Out" ||dv->getString("dtype")!="i32" ||cv->getString("dtype")!="i32" ||
         *dv->getArray("dims")!=axes || *cv->getArray("dims")!=llvm::json::Array{axes[0]} ||
         !prefix || prefix->getName().stripDialect()!="scan_sum" || !toInt || toInt->getName().stripDialect()!="cast" ||
         toInt->getOperand(0)!=op.getOperand(3) || !written.insert(o).second || !written.insert(c).second)
        return op.emitError("unique row destinations require the inclusive count of the same predicate");
      sawWrite=true;continue;
    }
    if(name=="write_offsets"){
      auto out=llvm::find(captures,op.getOperand(0)),total=llvm::find(captures,op.getOperand(1));
      if(out==captures.end() || total==captures.end() || axes.size()!=1 || !axes.front().getAsInteger() ||
         op.getOperand(2)!=row.getArgument(0))return op.emitError("expected one flat prefix");
      auto o=out-captures.begin(),t=total-captures.begin();
      auto valid=[&](unsigned i,int64_t len){
        auto *a=(*args)[i].getAsObject();
        return a && a->getString("direction")=="Out" && a->getString("dtype")=="i32" &&
               *a->getArray("dims")==llvm::json::Array{len};
      };
      auto *scan=op.getOperand(3).getDefiningOp();auto n=*axes.front().getAsInteger();
      if(n<=0 || n==INT64_MAX || !valid(o,n+1) || !valid(t,1) || !scan || scan->getName().stripDialect()!="scan_sum" ||
         !written.insert(o).second || !written.insert(t).second)return op.emitError("prefix must publish zero, all endpoints and its terminal count");
      sawWrite=true;continue;
    }
    if(name=="fill_ranges"){
      auto dest=llvm::find(captures,op.getOperand(0));
      if(dest==captures.end() || axes.size()!=1 || op.getOperand(1)!=captures[domainIndex] || op.getOperand(2)!=row.getArgument(0) ||
         !cast<MemRefType>(op.getOperand(1).getType()).getElementType().isInteger(32) || op.getAttrOfType<IntegerAttr>("unit").getInt()<=0)return op.emitError("invalid disjoint offset intervals");
      auto *slot=(*args)[dest-captures.begin()].getAsObject();
      if(slot->getString("direction")!="InOut" || slot->getString("dtype")!="i32" || slot->getArray("dims")->size()!=1)
        return op.emitError("sparse interval updates require initialized storage");
      continue;
    }
    if(name=="atomic_add" || name=="index_read" || name=="unique_put"){
      auto slot=llvm::find(captures,op.getOperand(0));
      if(slot==captures.end())return op.emitError("missing indexed vector");
      auto *record=(*args)[slot-captures.begin()].getAsObject();
      if(axes.size()!=2 || record->getString("dtype")!="i32" || record->getArray("dims")->size()!=1 ||
         record->getString("direction")!=(name=="index_read"?"In":"InOut") || !op.hasAttr("in_bounds"))
        return op.emitError("invalid indexed i32 access contract");
      if(name=="atomic_add" && op.getAttrOfType<StringAttr>("ordering")!="relaxed")return op.emitError("unsupported integer atomic ordering");
      if(name=="unique_put"){
        // Disjoint expert intervals and in-bounds keys are author preconditions.
        // The fetch-add supplies a fresh ticket in the keyed interval.
        auto *add=op.getOperand(1).getDefiningOp();
        if(!add || add->getName().stripDialect()!="add")return op.emitError("expected a keyed base plus a unique atomic ticket");
        auto *base=add->getOperand(0).getDefiningOp(), *ticket=add->getOperand(1).getDefiningOp();
        if(!base || !ticket || base->getName().stripDialect()!="index_read" || ticket->getName().stripDialect()!="atomic_add" ||
           base->getOperand(1)!=ticket->getOperand(1))return op.emitError("scatter requires an old ticket for the same key");
        auto *one=ticket->getOperand(2).getDefiningOp();
        if(!one || one->getAttrOfType<IntegerAttr>("value")!=IntegerAttr::get(IntegerType::get(fn.getContext(),32),1))return op.emitError("unique scatter requires unit tickets");
      }
      continue;
    }
    if (name == "ordinal" || name=="member_index") {
      if (op.getOperand(0) != captures[domainIndex] ||
          op.getOperand(1) != row.getArgument(0) ||
          (op.getResult(0).getType() !=
               RankedTensorType::get({ShapedType::kDynamic},
                                     IntegerType::get(fn.getContext(), 64,
                                                      IntegerType::Unsigned)) &&
           op.getResult(0).getType() !=
               RankedTensorType::get({ShapedType::kDynamic},
                                     IntegerType::get(fn.getContext(), 32))))
        return op.emitError(
            "ordinal must enumerate the declared row in the owning domain");
      continue;
    }
    if (name != "read_owner" && name != "write_owner" && name != "sqrt" && name != "and" && name != "xor" && name != "shl" &&
        name != "shr" && name != "gt" && name!="ne" && name != "select" &&
        name != "maximum" && name != "read" && name != "write" &&
        name != "end" && name != "sum" && name != "max_num" && name != "add" &&
        name != "sub" && name != "mul" && name != "div" && name != "cast" &&
        name != "exp" && name != "rsqrt")
      return op.emitError("operation outside logical row representation");
    if (name == "read" || name == "write" || name=="read_owner" || name=="write_owner") {
      auto view = cast<MemRefType>(op.getOperand(0).getType());
      auto arg = dyn_cast<BlockArgument>(op.getOperand(0));
      auto slot = llvm::find(captures, op.getOperand(0));
      if (!arg || slot == captures.end())
        return op.emitError("expected an external view");
      bool write = name == "write" || name=="write_owner";
      bool projected=name.ends_with("_owner");
      if (!indexed && !write && sawWrite &&
          llvm::any_of(*args, [](const llvm::json::Value &v) {
            return v.getAsObject()->getString("direction") == "InOut";
          }))
        return op.emitError(
            "dependent read after a write requires an ordered region");
      sawWrite |= write;
      size_t index = slot - captures.begin();
      if (indexed && !write && written.contains(index))
        return op.emitError("indexed read after a write to the same projection requires ordered control");
      auto *info = (*args)[index].getAsObject();
      if (ownerAxis) {
        const auto &dims=*info->getArray("dims");
        if (info->getString("direction")!=(write?"Out":"In") && info->getString("direction")!="InOut")
          return op.emitError("invalid direction in owner domain");
        if (op.getNumOperands()!=unsigned(2+write) || op.getOperand(1)!=row.getArgument(0) ||
            (projected ? (dims.size()!=1 || dims[0]!=axes[ownerAxis.getInt()]) : dims!=axes))
          return op.emitError("access does not share the declared owner domain");
        Type value=write?op.getOperand(2).getType():op.getResult(0).getType();
        if(value!=(projected?view.getElementType():Type(RankedTensorType::get({ShapedType::kDynamic},view.getElementType()))))
          return op.emitError("owner access value has wrong shape or type");
        if(write)written.insert(index);
        continue;
      }
      if(projected)return op.emitError("owner projection requires an owning axis");
      if ((info->getString("direction") != (write ? "Out" : "In") &&
           info->getString("direction") != "InOut") ||
          view.getRank() < 1 || view.getRank() > (indexed ? 4 : 2) ||
          op.getNumOperands() !=
              unsigned((indexed || domainType.getRank() == 1 ? 2 : view.getRank()) +
                       write) ||
          ((indexed || view.getRank() == 2 || domainType.getRank() == 1) &&
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
      if ((!allowed(valueType) &&
           !(name == "cast" && valueType.isInteger(1))) ||
          (name == "gt" || name=="ne" ? !element(result).isInteger(1)
                        : !allowed(element(result))) ||
          ((name == "xor" || name == "and" || name == "shr" || name == "shl") &&
           !valueType.isInteger(32)) ||
          (name != "gt" && name!="ne" && name != "cast" && element(result) != valueType) ||
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
  if (auto dm = fn->getAttr("denormal-fp-math")) {
    auto s = dyn_cast<StringAttr>(dm);
    if (!s || !kir::validDenormalMode(s.getValue()))
      return fn.emitError("invalid denormal-fp-math mode");
  }
  if (fn.walk([](Operation *op) {
          if (auto fm = op->getAttr("fastmath");
              fm && !isa<arith::FastMathFlagsAttr>(fm)) {
            op->emitError("fastmath attribute must be arith::FastMathFlagsAttr");
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        }).wasInterrupted())
    return failure();
  if(kir::matrixDomain(fn))return kir::verifyMatrix(fn,dialect,canonical);
  if (kir::orderedOwners(fn))
    return kir::verifyOwners(fn, dialect);
  if (fn.empty() || !llvm::hasSingleElement(fn.getBody()) || fn.getNumResults())
    return fn.emitError("expected a defined output-buffer kernel");
  auto &body = fn.front();
  if (body.getOperations().size() != 2 ||
      body.front().getName().getStringRef() != (dialect+".parallel").str())
    return fn.emitError("expected a parallel row domain and return");
  auto &loop = body.front();
  auto domain = dyn_cast<MemRefType>(loop.getOperand(0).getType());
  if (!domain || (!fn->hasAttr("kir.iteration") && !fn->hasAttr("kir.owner_axis") && domain.getRank() != 2 && domain.getRank() != 1) ||
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
  intentdsl::kir::FastMathPolicy policy;
  explicit Canonicalize(intentdsl::kir::FastMathPolicy p = {})
      : policy(std::move(p)) {}
  void runOnOperation() final {
    auto *ctx = &getContext();
    for(auto fn:getOperation().getOps<func::FuncOp>()) {
      auto owner=cast<BlockArgument>(fn.front().front().getOperand(0)).getArgNumber();
      fn->setAttr("kir.domain_arg", IntegerAttr::get(IntegerType::get(ctx,64),owner));
      llvm::StringRef fnDenorm = intentdsl::kir::getDenormalFpMath(fn);
      std::string targetDenorm =
          fnDenorm != "ieee" ? fnDenorm.str() : policy.denormalFpMath;
      if(intentdsl::kir::matrixDomain(fn)){
        auto &body=fn.front().front().getRegion(0).front();
        for(auto &op:llvm::make_early_inc_range(llvm::reverse(body))){
          // KIR's contraction expressions read immutable inputs. Only the
          // publication writes, and its complete operand chain is live.
          if (auto type = intentdsl::kir::storageFormat(op))
            op.setAttr("kir.storage", TypeAttr::get(type));
          if(op.getNumResults()==1 && op.getResult(0).use_empty()){op.erase();continue;}
          if (op.getName().getStringRef() == "kir.contract") {
            auto opDenorm = op.getAttrOfType<StringAttr>("denormal-fp-math");
            if (opDenorm && opDenorm.getValue() != "ieee")
              targetDenorm = opDenorm.getValue().str();
            else if (targetDenorm != "ieee" &&
                     mlir::isa<mlir::FloatType>(mlir::getElementTypeOrSelf(
                         op.getResult(0).getType())))
              op.setAttr("denormal-fp-math",
                         StringAttr::get(ctx, targetDenorm));
          }
        }
      }
      fn->setAttr("denormal-fp-math", StringAttr::get(ctx, targetDenorm));
      if (policy.defaultFlags != arith::FastMathFlags::none) {
        fn.walk([&](Operation *op) {
          if (!intentdsl::kir::isFloatingMathOp(op))
            return;
          auto addFlags = policy.defaultFlags;
          if (op->getName().getStringRef() == "kir.ordered_sum" &&
              !intentdsl::kir::hasFastMath(op, arith::FastMathFlags::reassoc))
            addFlags = bitEnumClear(addFlags, arith::FastMathFlags::reassoc);
          if (auto loop = dyn_cast_or_null<scf::ForOp>(op->getParentOp());
              loop && !loop->hasAttr("kir.reassociate")) {
            auto *term = loop.getBody()->getTerminator();
            if (llvm::is_contained(term->getOperands(), op->getResult(0)))
              addFlags = bitEnumClear(addFlags, arith::FastMathFlags::reassoc);
          }
          intentdsl::kir::addFastMathFlags(op, addFlags);
        });
      }
    }
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
    r.insert<intentdsl::kir::KirDialect, func::FuncDialect, arith::ArithDialect,
             scf::SCFDialect, memref::MemRefDialect, math::MathDialect>();
  };
  for (bool canonical : {false, true})
    unit.boundaries.push_back(
        {{canonical ? "canonical" : "imported",
          [canonical](OpPassManager &pm, const TargetProfile &profile) {
            if (canonical) {
              intentdsl::kir::FastMathPolicy p;
              if (auto *svc =
                      profile.services.get<intentdsl::kir::FastMathPolicy>())
                p = *svc;
              pm.addPass(std::make_unique<Canonicalize>(std::move(p)));
            }
            return llvm::Error::success();
          },
          [canonical](ModuleOp m, const TargetProfile &) {
            return intentdsl::verifyLogicalRows(m,"kir",canonical);
          }},
         COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
