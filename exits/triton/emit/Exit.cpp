#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "intentdsl/levels/execution/block/ir/Access.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/execution/block/decisions/Verify.h"
#include "intentdsl/exits/triton/ir/Dialect.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void legalize(OpPassManager &);
llvm::Expected<TargetProfile> capabilities(const llvm::json::Object &);
namespace {
struct BindCallable : PassWrapper<BindCallable, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BindCallable)
  void runOnOperation() final {
    auto fn = *getOperation().getOps<func::FuncOp>().begin();
    auto guard=fn->getAttrOfType<DenseI64ArrayAttr>("block.guard");
    Builder b(&getContext());
    fn->setAttr("triton_src.min_width", b.getI64IntegerAttr(guard[0]));
    fn->setAttr("triton_src.max_width", b.getI64IntegerAttr(guard[1]));
  }
};
class Writer {
  llvm::raw_ostream &out;
  llvm::DenseMap<Value, std::string> values;
  unsigned next = 0;
  void line(unsigned depth, const std::string &s) {
    out.indent(4 * depth) << s << "\n";
  }
  std::string name(Value v) { return values.lookup(v); }

public:
  explicit Writer(llvm::raw_ostream &out) : out(out) {}
  llvm::Error write(func::FuncOp fn) {
    out << "import triton\nimport triton.language as tl\n\n@triton.jit\ndef "
           "kernel(";
    for (auto arg : fn.getArguments()) {
      auto id = "a" + std::to_string(arg.getArgNumber());
      values[arg] = id;
      out << id << ", ";
      if (auto v=dyn_cast<MemRefType>(arg.getType()); v && v.getRank()>2)
        for(int i=0;i<v.getRank()-1;++i)out<<id<<"_d"<<i<<", ";
      if (auto v = dyn_cast<MemRefType>(arg.getType()))
        for (int i = 0; i < v.getRank(); ++i)
          out << id << "_s" << i << ", ";
    }
    out << "M, N, ROWS: tl.constexpr, WIDTH: tl.constexpr):\n";
    return body(fn.front(), 1);
  }
  llvm::Error body(Block &bb, unsigned depth) {
    for (auto &op : bb) {
      auto n = op.getName().stripDialect();
      if (op.getName().getStringRef() == "func.return")
        continue;
      if (op.getName().getDialectNamespace() != "triton_src")
        return llvm::createStringError("unexpected operation at emit");
      auto arg = [&](unsigned i) { return name(op.getOperand(i)); };
      auto alg = op.getAttrOfType<StringAttr>("algebra");
      if ((n=="reduce" || n=="identity" || n=="combine" || n=="neutralize") && (!alg || (alg!="sum" && alg!="max_num")))
        return llvm::createStringError("unrecognized Triton reduction algebra");
      bool sum = alg && alg == "sum";
      if (n == "tiles") {
        auto &loop = op.getRegion(0).front();
        auto k = "k" + std::to_string(next++);
        values[loop.getArgument(0)] = k;
        if (op.getNumResults()) {
          auto carry = "c" + std::to_string(next++);
          values[loop.getArgument(1)] = values[op.getResult(0)] = carry;
          line(depth, carry + " = " + arg(1));
        }
        line(depth, "for " + k + " in range(tl.cdiv(N, WIDTH)):");
        if (auto error = body(loop, depth + 1))
          return error;
        continue;
      }
      if (n == "yield") {
        if (op.getNumOperands())
          line(depth, name(bb.getArgument(1)) + " = " + arg(0));
        continue;
      }
      std::string e;
      if (n == "linear_index") {
        bool wide = cast<RankedTensorType>(op.getResult(0).getType())
                        .getElementType()
                        .isUnsignedInteger(64);
        e = "(" + arg(1) + (wide ? "[:,None].to(tl.uint64)" : "[:,None]") +
            "*N+" + arg(2) + "[None,:]).to(" +
            (wide ? "tl.uint64)" : "tl.int32)");
      } else if (n == "select")
        e = "tl.where(" + arg(0) + "," + arg(1) + "," + arg(2) + ")";
      else if (n == "xor" || n == "and" || n == "shr" || n == "shl" ||
               n == "gt") {
        auto token = n == "xor"   ? " ^ "
                     : n == "and" ? " & "
                     : n == "shl" ? " << "
                     : n == "shr" ? " >> "
                                  : " > ";
        e = arg(0) + token + arg(1);
      } else if (n == "constant")
        e = intentdsl::kir::literal(op.getAttrOfType<TypedAttr>("value"));
      else if (n == "maximum")
        e = "tl.maximum(" + arg(0) + "," + arg(1) +
            ",propagate_nan=tl.PropagateNan.ALL)";
      else if (n == "pid")
        e = "tl.program_id(0).to(tl.int64)";
      else if (n == "rows")
        e = arg(0) + " * ROWS + tl.arange(0, ROWS)";
      else if (n == "zero")
        e = "0";
      else if (n == "columns")
        e = "tl.arange(0, WIDTH).to(tl.int64) + " + arg(0) + " * WIDTH";
      else if (n == "active")
        e = "(" + arg(1) + "[:, None] < M) & (" + arg(2) + "[None, :] < N)";
      else if (n == "offsets") {
        int dims=cast<MemRefType>(op.getOperand(0).getType()).getRank();
        if(dims>2){
          auto a=arg(0),row=arg(1);
          e=arg(2)+"[None,:]*"+a+"_s"+std::to_string(dims-1);
          for(int i=dims-2;i>=0;--i){
            auto size=a+"_d"+std::to_string(i);
            e+="+(("+row+")%"+size+")[:,None]*"+a+"_s"+std::to_string(i);
            row="(("+row+")//"+size+")";
          }
          auto id="v"+std::to_string(next++);values[op.getResult(0)]=id;line(depth,id+" = "+e);continue;
        }
        auto rank = op.getAttrOfType<BoolAttr>("matrix").getValue() ? 2 : 1;
        bool swap = op.getAttrOfType<BoolAttr>("permuted") &&
                    op.getAttrOfType<BoolAttr>("permuted").getValue();
        auto base = rank == 2 ? arg(1) + "[:, None] * " + arg(0) +
                                    (swap ? "_s1" : "_s0")
                              : "tl.full((ROWS, 1), 0, tl.int64)";
        e = base + " + " + arg(2) + "[None, :] * " + arg(0) + "_s" +
            std::to_string(swap ? 0 : rank - 1);
      } else if (n == "load")
        e = "tl.load(" + arg(0) + " + " + arg(1) + ", mask=" + arg(2) +
            ", other=0)";
      else if (n == "store") {
        line(depth, "tl.store(" + arg(0) + " + " + arg(1) + ", " + arg(3) +
                        ", mask=" + arg(2) + ")");
        continue;
      } else if (n == "neutralize")
        e = "tl.where(" + arg(1) + ", " + arg(0) + ", " +
            (sum ? "0.0" : "-float('inf')") + ")";
      else if (n == "reduce")
        e = std::string(sum ? "tl.sum" : "tl.max") + "(" + arg(0) + ", 1)";
      else if (n == "identity")
        e = std::string("tl.full((ROWS,), ") + (sum ? "0.0" : "-float('inf')") +
            ", tl.float32)";
      else if (n == "combine")
        e = sum ? arg(0) + " + " + arg(1)
                : "tl.maximum(" + arg(0) + ", " + arg(1) + ")";
      else if (n == "broadcast_rows")
        e = arg(0) + "[:, None]";
      else if (n == "add" || n == "sub" || n == "mul")
        e = arg(0) + (n == "add" ? " + " : n == "sub" ? " - " : " * ") + arg(1);
      else if (n == "div")
        e = "tl.div_rn(" + arg(0) + ", " + arg(1) + ")";
      else if (n == "sqrt" || n == "exp" || n == "rsqrt")
        e = "tl." + n.str() + "(" + arg(0) + ")";
      else if (n == "cast") {
        auto t = op.getResult(0).getType();
        if (auto shaped = dyn_cast<ShapedType>(t))
          t = shaped.getElementType();
        e = arg(0) + ".to(tl." +
            std::string(t.isInteger(32) ? "int32"
                        : t.isBF16()    ? "bfloat16"
                        : t.isF16()     ? "float16"
                                        : "float32") +
            ")";
      } else
        return llvm::createStringError("unhandled legal operation " + n.str());
      auto id = "v" + std::to_string(next++);
      values[op.getResult(0)] = id;
      line(depth, id + " = " + e);
    }
    return llvm::Error::success();
  }
};
llvm::Error emit(ModuleOp m, llvm::raw_pwrite_stream &out) {
  auto fn = *m.getOps<func::FuncOp>().begin();
  auto args = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!args)
    return args.takeError();
  llvm::json::Array configs;
  for (Attribute a : fn->getAttrOfType<ArrayAttr>("block.configs")) {
    auto d = cast<DictionaryAttr>(a);
    llvm::json::Object row;
    for (StringRef k : {"id", "rows", "width"})
      row[k] = d.getAs<IntegerAttr>(k).getInt();
    auto native = llvm::json::parse(d.getAs<StringAttr>("payload").getValue());
    if (!native)
      return native.takeError();
    row["native"] = std::move(*native);
    configs.push_back(std::move(row));
  }
  std::string src;
  llvm::raw_string_ostream stream(src);
  if (auto err = Writer(stream).write(fn))
    return err;
  out << llvm::json::Value(llvm::json::Object{
      {"version", 2},{"grid", llvm::json::Array{llvm::json::Array{
      cast<StringAttr>(fn->getAttrOfType<ArrayAttr>("block.grid")[0]).str(),
      cast<StringAttr>(fn->getAttrOfType<ArrayAttr>("block.grid")[1]).str(),
      cast<StringAttr>(fn->getAttrOfType<ArrayAttr>("block.grid")[2]).str()},1,1}},
      {"entry", "kernel"},
      {"source", src},
      {"domain",intentdsl::publishDomain(fn)},
      {"abi", std::move(*args)},
      {"rows", std::move(configs)},
      {"topology", fn->getAttrOfType<StringAttr>("block.topology").str()},
      {"guard",
       llvm::json::Object{
           {"min_width",
            fn->getAttrOfType<IntegerAttr>("triton_src.min_width").getInt()},
           {"max_width",
            fn->getAttrOfType<IntegerAttr>("triton_src.max_width").getInt()>=0
                ? llvm::json::Value(
                      fn->getAttrOfType<IntegerAttr>("triton_src.max_width")
                          .getInt())
                : llvm::json::Value(nullptr)}}}});
  return llvm::Error::success();
}
} // namespace
void bind(Unit &unit) {
  unit.dialects = [](DialectRegistry &r) {
    r.insert<intentdsl::itsrc::ItsrcDialect>();
  };
  for (StringRef phase : {"legal", "callable"})
    unit.boundaries.push_back(
        {{phase.str(),
          [phase](OpPassManager &pm,
                  const TargetProfile &profile) -> llvm::Error {
            if (phase == "legal")
              legalize(pm);
            else {
              pm.addPass(std::make_unique<BindCallable>());
            }
            return llvm::Error::success();
          },
          [phase](ModuleOp m, const TargetProfile &profile) {
            auto funcs = m.getOps<func::FuncOp>();
            if (!llvm::hasSingleElement(funcs))
              return failure();
            auto fn = *funcs.begin();
            if (!fn->getAttrOfType<StringAttr>("kir.abi") ||
                !fn->getAttrOfType<ArrayAttr>("block.configs") ||
                !fn->getAttrOfType<StringAttr>("block.topology"))
              return failure();
            if (phase == "callable" &&
                (!fn->getAttrOfType<IntegerAttr>("triton_src.min_width") ||
                 !fn->getAttrOfType<IntegerAttr>("triton_src.max_width")))
              return failure();
            if(failed(intentdsl::verifyMapping(m,profile,"triton_src")))return failure();
            if(phase=="callable"){
              auto g=fn->getAttrOfType<DenseI64ArrayAttr>("block.guard");
              if(g[0]!=fn->getAttrOfType<IntegerAttr>("triton_src.min_width").getInt() ||
                 g[1]!=fn->getAttrOfType<IntegerAttr>("triton_src.max_width").getInt())
                return fn.emitError("call guard differs from verified block class"),failure();
              for(auto a:fn->getAttrOfType<ArrayAttr>("block.configs")){
                auto payload=llvm::json::parse(cast<DictionaryAttr>(a).getAs<StringAttr>("payload").getValue());
                if(!payload){llvm::consumeError(payload.takeError());return failure();}
                auto *opts=payload->getAsObject();
                if(!opts || opts->size()!=2 || opts->getInteger("num_warps").value_or(0)<=0 ||
                    opts->getInteger("num_stages").value_or(0)<=0)return fn.emitError("illegal native configuration"),failure();
              }
            }
            bool valid = true;
            m.walk([&](Operation *op) {
              auto ns = op->getName().getDialectNamespace();
              if (ns=="triton_src") {
                auto name=op->getName().stripDialect();
                auto matches=[&](unsigned i,StringRef opName){auto *v=op->getOperand(i).getDefiningOp();return v && v->getName().getStringRef()==opName;};
                if(name=="offsets"){auto axes=intentdsl::block::projection(op->getOperand(0).getType());valid &= succeeded(axes) && op->getAttrOfType<BoolAttr>("matrix").getValue()==(*axes==intentdsl::block::RowProjection::Matrix) && matches(1,"triton_src.rows") && matches(2,"triton_src.columns");}
                if(name=="neutralize"){auto a=op->getAttrOfType<StringAttr>("algebra"); valid &= (a=="sum" || a=="max_num") && matches(1,"triton_src.active") && op->getOperand(1).getParentBlock()==op->getBlock();}
                if(name=="reduce") valid &= matches(0,"triton_src.neutralize") && op->getOperand(0).getDefiningOp()->getAttr("algebra")==op->getAttr("algebra");
              }
              valid &= ns == "builtin" || ns == "func" || ns == "triton_src";
            });
            return success(valid);
          }},
         COMPILERINFRA_SOURCE});
  unit.exit =
      Located<ExitImplementation>{{emit, capabilities}, COMPILERINFRA_SOURCE};
}
} // namespace CI_UNIT_NAMESPACE
