#include "intentdsl/levels/execution/block/ir/Grid.h"
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
    out << "import triton\nimport triton.language as tl\n";
    if (fn.walk([](Operation *o){return o->getName().stripDialect()=="segment_fold" ? WalkResult::interrupt():WalkResult::advance();}).wasInterrupted())
      out << "from triton.language.extra.cuda import libdevice\n";
    bool moments=false;
    fn.walk([&](Operation *op){moments |= op->getName().stripDialect()=="summary_reduce";});
    if (moments) out <<
#include "Moments.inc"
    ;
    out << "\n@triton.jit\ndef "
           "kernel(";
    for (auto arg : fn.getArguments()) {
      auto id = "a" + std::to_string(arg.getArgNumber());
      values[arg] = id;
      out << id << ", ";
      if (auto v=dyn_cast<MemRefType>(arg.getType()); v && (v.getRank()>2 || (fn->hasAttr("kir.owner_axis") && v.getRank()>1)))
        for(int i=0;i<v.getRank()-(fn->hasAttr("kir.owner_axis")?0:1);++i)out<<id<<"_d"<<i<<": tl.constexpr, ";
      if (auto v = dyn_cast<MemRefType>(arg.getType()))
        for (int i = 0; i < v.getRank(); ++i)
          out << id << "_s" << i << (v.getRank()>2 ? ": tl.constexpr, " : ", ");
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
      auto ownerAxis=op.getParentOfType<func::FuncOp>()->getAttrOfType<IntegerAttr>("kir.owner_axis");
      if(n=="fill_ranges_at"){
        auto unit=std::to_string(op.getAttrOfType<IntegerAttr>("unit").getInt());
        line(depth,"valid="+arg(2)+"+1<N");
        line(depth,"begin=tl.load("+arg(1)+"+"+arg(2)+"*"+arg(1)+"_s0,mask=valid,other=0)//"+unit);
        line(depth,"end=tl.load("+arg(1)+"+("+arg(2)+"+1)*"+arg(1)+"_s0,mask=valid,other=0)//"+unit);
        line(depth,"for k in range(tl.max(end-begin)):");
        line(depth+1,"tl.store("+arg(0)+"+(begin+k)*"+arg(0)+"_s0,"+arg(2)+".to(tl.int32),mask=begin+k<end)");continue;
      }
      if(n=="publish_offsets"){
        auto index=arg(2)+"[None,:]";
        line(depth,"tl.store("+arg(0)+",0)");
        line(depth,"tl.store("+arg(0)+"+("+index+"+1)*"+arg(0)+"_s0,"+arg(4)+",mask="+arg(3)+")");
        line(depth,"tl.store("+arg(1)+"+0*"+index+","+arg(4)+",mask="+arg(3)+" & ("+index+"==N-1))");
        continue;
      }
      if(n=="index_read_at" || n=="atomic_add_at" || n=="unique_put_at"){
        auto ptr=arg(0)+"+"+arg(1)+"*"+arg(0)+"_s0";
        if(n=="unique_put_at"){line(depth,"tl.store("+ptr+","+arg(2)+",mask="+arg(3)+")");continue;}
        auto id="v"+std::to_string(next++);values[op.getResult(0)]=id;
        line(depth,id+" = "+(n=="index_read_at"?"tl.load("+ptr+",mask="+arg(2)+",other=0)":
            "tl.atomic_add("+ptr+","+arg(2)+",mask="+arg(3)+",sem='relaxed',scope='gpu')"));continue;
      }
      if(n=="scatter_fragment"){
        auto fill=std::to_string(op.getAttrOfType<IntegerAttr>("fill").getInt());
        line(depth,"total=tl.max(tl.where("+arg(4)+","+arg(6)+",0),1)");
        line(depth,"slot=tl.where("+arg(5)+","+arg(6)+"-1,total[:,None]+"+arg(3)+"[None,:]-"+arg(6)+")");
        line(depth,"tl.store("+arg(0)+"+"+arg(2)+"[:,None]*"+arg(0)+"_s0+slot*"+arg(0)+"_s1,tl.where("+arg(5)+","+arg(7)+","+fill+"),mask="+arg(4)+")");
        line(depth,"tl.store("+arg(1)+"+"+arg(2)+"*"+arg(1)+"_s0,total,mask="+arg(2)+"<M)");
        continue;
      }
      if(n=="ne" || n=="member_tile"){
        auto id="v"+std::to_string(next++);values[op.getResult(0)]=id;
        line(depth,id+" = "+(n=="ne"?arg(0)+"!="+arg(1):"tl.broadcast_to("+arg(0)+"[None,:],(ROWS,WIDTH)).to(tl.int32)"));continue;
      }
      if(n=="key_counts" || n=="sum_columns"){
        auto size=std::to_string(op.getAttrOfType<IntegerAttr>("extent").getInt()), base=arg(0);
        auto id="bins"+std::to_string(next++);line(depth,id+" = tl.full((WIDTH,),0,tl.int32)");
        if(n=="key_counts"){
          auto span=op.getAttrOfType<IntegerAttr>("span").getInt(), width=op.getAttrOfType<IntegerAttr>("native_panel").getInt();
          std::string start="(tl.program_id(0).to(tl.int64)*"+std::to_string(span)+")";
          line(depth,"for piece in range("+std::to_string(llvm::divideCeil(span,width))+"):");
          line(depth+1,"offset = piece*"+std::to_string(width)+"+tl.arange(0,"+std::to_string(width)+")");
          line(depth+1,"key = tl.load("+base+"+("+start+"+offset)*"+base+"_s0, mask=(offset<"+std::to_string(span)+") & ("+start+"+offset<"+size+"),other=-1)");
          bool wide=cast<MemRefType>(op.getOperand(0).getType()).getElementType().isInteger(64);
          line(depth+1,wide?"bin_id=key":"bin_id=key.to(tl.int32)");
          line(depth+1,"keep=(bin_id>=0)&(bin_id<N)");
          line(depth+1,id+" += tl.histogram(tl.where(keep,bin_id,0).to(tl.int32),WIDTH) - tl.where(tl.arange(0,WIDTH)==0,tl.sum((~keep).to(tl.int32),0),0)");
        }else{
          auto height=std::to_string(op.getAttrOfType<IntegerAttr>("native_panel").getInt());
          line(depth,"for panel in range(tl.cdiv("+size+","+height+")):");
          line(depth+1,"part=panel*"+height+"+tl.arange(0,"+height+").to(tl.int64)");
          line(depth+1,"v=tl.load("+base+"+part[:,None]*"+base+"_s0+"+arg(2)+"[None,:]*"+base+"_s1,mask=(part[:,None]<"+size+")&("+arg(2)+"[None,:]<N),other=0)");
          line(depth+1,id+" += tl.sum(v,0)");
        }
        values[op.getResult(0)]=id+"[None,:]";continue;
      }
      if(n=="segment_fold" || n=="segment_size"){
        auto fn=op.getParentOfType<func::FuncOp>();
        auto slot=fn->getAttrOfType<IntegerAttr>("kir.domain_arg").getInt();
        bool flat=cast<MemRefType>(fn.getArgument(slot).getType()).getRank()==1;
        std::string tag="seg"+std::to_string(next++), limit=tag+"_end", cursor=tag+"_i", acc=tag+"_sum", live=tag+"_live";
        auto owner=flat?arg(2)+"[None,:]":arg(1)+"[:,None]";
        auto offset=arg(0)+" + ("+owner+") * "+arg(0)+"_s0";
        line(depth,cursor+" = tl.load("+offset+", mask="+owner+" < "+(flat?"N":"M")+", other=0)");
        line(depth,limit+" = tl.load("+offset+" + "+arg(0)+"_s0, mask="+owner+" < "+(flat?"N":"M")+", other=0)");
        if(n=="segment_size"){
          values[op.getResult(0)]=tag;line(depth,tag+" = ("+limit+" - "+cursor+").to(tl.float32)");continue;
        }
        auto &terms=op.getRegion(0).front();
        auto batch=op.getAttrOfType<IntegerAttr>("member_batch");bool panel=bool(batch);
        auto width=panel?std::to_string(batch.getInt()):"1";
        values[terms.getArgument(0)]=panel?"("+cursor+"[:,:,None]+tl.arange(0,"+width+")[None,None,:])":cursor;
        values[terms.getArgument(1)]=arg(2)+(panel?"[None,:,None]":"[None,:]");
        line(depth,acc+" = tl.full((ROWS,WIDTH), 0., tl.float32)");
        line(depth,"for "+tag+"_step in range(tl.cdiv(tl.max("+limit+" - "+cursor+"),"+width+")):");
        if(panel)line(depth+1,live+" = ("+name(terms.getArgument(0))+" < "+limit+"[:,:,None]) & "+arg(3)+"[:,:,None]");
        else line(depth+1,live+" = ("+cursor+" < "+limit+") & "+arg(3));
        for(auto &v:terms){
          auto code=v.getName().stripDialect();
          auto a=[&](unsigned i){return name(v.getOperand(i));};
          if(code=="term"){
            if(panel){
              line(depth+1,"for lane in tl.static_range(0,"+width+"):");
              line(depth+2,"sample=tl.reshape(tl.gather("+a(0)+",tl.full((ROWS,WIDTH,1),lane,tl.int32),2),(ROWS,WIDTH))");
              line(depth+2,acc+" = tl.where(("+cursor+"+lane < "+limit+") & "+arg(3)+",libdevice.add_rn("+acc+",sample),"+acc+")");
            }else line(depth+1,acc+" = tl.where("+live+",libdevice.add_rn("+acc+","+a(0)+"),"+acc+")");
            break;
          }
          std::string rhs;
          if(code=="member_load"){
            auto pointer=a(0);
            for(unsigned i=1;i<v.getNumOperands();++i)pointer+=" + ("+a(i)+") * "+a(0)+"_s"+std::to_string(i-1);
            if(panel)pointer="tl.broadcast_to("+pointer+",(ROWS,WIDTH,"+width+"))";
            rhs="tl.load("+pointer+",mask="+live+",other=0)";
          } else if(code=="mul" || code=="add")rhs="libdevice."+code.str()+"_rn("+a(0)+","+a(1)+")";
          else return llvm::createStringError("unknown strict member expression "+code);
          auto dst="v"+std::to_string(next++);values[v.getResult(0)]=dst;line(depth+1,dst+" = "+rhs);
        }
        line(depth+1,cursor+" += "+width);values[op.getResult(0)]=acc;continue;
      }
      if(n=="owner_load") e="tl.load("+arg(0)+"+"+arg(1)+"*"+arg(0)+"_s0,mask="+arg(1)+"<M,other=0)";
      else if(n=="owner_store"){
        line(depth,"tl.store("+arg(0)+"+"+arg(1)+"*"+arg(0)+"_s0,"+arg(2)+",mask="+arg(1)+"<M)");
        continue;
      } else if (n=="summary_reduce")
        e="tl.reduce(("+arg(1)+".to(tl.int32),tl.where("+arg(1)+","+arg(0)+",0.),tl.full((ROWS,WIDTH),0.,tl.float32)),1,merge_moments)";
      else if (n=="summary_identity")
        e="(tl.full((ROWS,),0,tl.int32),tl.full((ROWS,),0.,tl.float32),tl.full((ROWS,),0.,tl.float32))";
      else if (n=="summary_merge")
        e="merge_moments("+arg(0)+"[0],"+arg(0)+"[1],"+arg(0)+"[2],"+arg(1)+"[0],"+arg(1)+"[1],"+arg(1)+"[2])";
      else if (n=="summary_field") e=arg(0)+"["+std::to_string(op.getAttrOfType<IntegerAttr>("position").getInt())+"]";
      else if(n=="indexed_tile_read"){
        auto key="key"+std::to_string(next++);
        line(depth,key+" = tl.load("+arg(1)+" + "+arg(2)+" * "+arg(1)+"_s0, mask="+arg(2)+" < M, other=0)");
        e="tl.load("+arg(0)+" + "+key+"[:,None]*"+arg(0)+"_s0 + "+arg(3)+"[None,:]*"+arg(0)+"_s1,mask="+arg(4)+",other=0)";
      }
      else if(n=="scan_sum")e="tl.cumsum("+arg(0)+",axis=1)";
      else if (n == "uniform") {
        auto prefix = "rng"+std::to_string(next++);
        auto lane = "("+arg(1)+" & 3)";
        line(depth,prefix+"a, "+prefix+"b, "+prefix+"c, "+prefix+"d = tl.randint4x("+arg(0)+", "+arg(1)+" >> 2)");
        e = "(tl.where("+lane+" == 0,"+prefix+"a,tl.where("+lane+" == 1,"+prefix+"b,tl.where("+lane+" == 2,"+prefix+"c,"+prefix+"d))).to(tl.uint32) >> 8).to(tl.float32) * 0.000000059604644775390625";
      } else if (n == "linear_index") {
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
        { e = "tl.program_id(0).to(tl.int64)";
          auto fn=op.getParentOfType<func::FuncOp>();
          Builder b(fn.getContext());
          if(fn->getAttr("block.grid")==intentdsl::block::grid(b,true))
            e += op.getAttrOfType<IntegerAttr>("axis").getInt()==0 ? " // tl.cdiv(N,WIDTH)" : " % tl.cdiv(N,WIDTH)";
        }
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
        if(ownerAxis){
          auto base=arg(0), col=arg(2)+"[None,:]";
          e=arg(1)+"[:,None]*"+base+"_s"+std::to_string(ownerAxis.getInt());
          for(int i=dims-1;i>=0;--i)if(i!=ownerAxis.getInt()){
            auto size="tl.maximum("+base+"_d"+std::to_string(i)+",1)";
            e+="+(("+col+")%"+size+")*"+base+"_s"+std::to_string(i);
            col="(("+col+")//"+size+")";
          }
        } else if(dims>2){
          auto a=arg(0),row=arg(1);
          e=arg(2)+"[None,:]*"+a+"_s"+std::to_string(dims-1);
          for(int i=dims-2;i>=0;--i){
            auto size=a+"_d"+std::to_string(i);
            e+="+(("+row+")%"+size+")[:,None]*"+a+"_s"+std::to_string(i);
            row="(("+row+")//"+size+")";
          }
          auto id="v"+std::to_string(next++);values[op.getResult(0)]=id;line(depth,id+" = "+e);continue;
        }
        if(!ownerAxis){
        auto rank = op.getAttrOfType<BoolAttr>("matrix").getValue() ? 2 : 1;
        bool swap = op.getAttrOfType<BoolAttr>("permuted") &&
                    op.getAttrOfType<BoolAttr>("permuted").getValue();
        auto base = rank == 2 ? arg(1) + "[:, None] * " + arg(0) +
                                    (swap ? "_s1" : "_s0")
                              : "tl.full((ROWS, 1), 0, tl.int64)";
        e = base + " + " + arg(2) + "[None, :] * " + arg(0) + "_s" +
            std::to_string(swap ? 0 : rank - 1);
        }
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
{"version", 2},{"grid", llvm::json::Array{intentdsl::block::gridExpression(fn->getAttr("block.grid")),1,1}},
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
            if(m.walk([](Operation *op){
              auto n=op->getName().getStringRef();
              if(n=="triton_src.key_counts" || n=="triton_src.sum_columns"){
                auto bits=op->getAttrOfType<IntegerAttr>("address_bits");
                if(!bits || bits.getInt()!=64){op->emitError("key partition addressing requires i64");return WalkResult::interrupt();}
              }
              return WalkResult::advance();
            }).wasInterrupted())return failure();
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
