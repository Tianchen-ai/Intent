#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "llvm/ADT/StringExtras.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
class Source {
  llvm::raw_ostream &out;
  llvm::DenseMap<Value, std::string> vars;
  int next = 0;
  unsigned depth = 0;
  void line(unsigned n, const std::string &s) {
    out.indent((n + depth) * 4) << s << "\n";
  }
  std::string var(Value v) { return vars.lookup(v); }
  std::string temp() { return "v" + std::to_string(next++); }
  std::string dtype(Type t) {
    if (auto v = dyn_cast<VectorType>(t))
      t = v.getElementType();
    if (t.isInteger(1))
      return "bool";
    if (t.isUnsignedInteger(64))
      return "uint64";
    if (t.isInteger(32))
      return "int32";
    return t.isBF16() ? "bfloat16" : t.isF16() ? "float16" : "float32";
  }

public:
  explicit Source(llvm::raw_ostream &stream) : out(stream) {}
  llvm::Error write(Operation *task) {
    SmallVector<std::string> params;
    SmallVector<std::string> constants;
    auto fn = task->getParentOfType<func::FuncOp>();
    auto &body = task->getRegion(0).front();
    vars[body.getArgument(0)] = "r";
    for (auto [i, v] : llvm::enumerate(body.getArguments().drop_front())) {
      std::string n = "a" + std::to_string(i);
      vars[v] = n;
      if (auto m = dyn_cast<MemRefType>(v.getType())) {
        params.push_back(n + ": Pointer[Scalar[DType." +
                         dtype(m.getElementType()) + "], MutUntrackedOrigin]");
        for(bool shape:{true,false}){
          auto bound=fn.getArgAttrOfType<DenseI64ArrayAttr>(i,shape?"mojo_row.shape":"mojo_row.strides");
          int count=shape?(m.getRank()>2?m.getRank()-1:0):m.getRank();
          for(int j=0;j<count;++j){
            std::string field=n+(shape?"d":"s")+std::to_string(j);
            params.push_back((bound?"abi_":"")+field+": Int64");
            if(bound)constants.push_back("comptime "+field+" = "+std::to_string(bound[j]));
          }
        }
      } else
        params.push_back(n + (v.getType().isUnsignedInteger(64) ? ": UInt64"
                              : v.getType().isInteger(32)       ? ": Int32"
                                                                : ": Float32"));
    }
    bool usesRandom = false;
    task->walk([&](Operation *op) {
      if (auto k = op->getAttrOfType<StringAttr>("kind"))
        usesRandom |= k.getValue() == "uniform";
    });
    if (usesRandom)
      out <<
#include "Philox.inc"
          ;
    out << "from std.math import exp, sqrt, rsqrt, max, "
           "inf\n@export(\"intent_row_interval\")\ndef intent_row_interval("
        << llvm::join(params, ",")
        << ", first: Int64, stop: Int64, cols: Int64, columns_per_block: "
           "Int64) abi(\"C\"):\n";
    for(auto &constant:constants)line(1,constant);
    line(1, "for r in range(Int(first),Int(stop)):");
    const bool projected = fn->hasAttr("kir.iteration");
    // For projected views the host supplies each interval's physical base.
    auto fullWidth =
        task->getAttrOfType<IntegerAttr>("mojo_row.lanes").getInt();
    for (auto &loop : body.without_terminator()) {
      if (loop.getName().getStringRef() != "mojo_row.sweep")
        return llvm::createStringError("Mojo requires realized sweeps");
      bool tail = loop.getAttrOfType<StringAttr>("range") == "tail";
      bool owners = loop.getAttrOfType<StringAttr>("range") == "owners";
      if(loop.getAttrOfType<StringAttr>("range")=="copy"){
        auto &read=loop.getRegion(0).front().front();auto &write=*read.getNextNode();
        auto address=[&](Operation &access){
          auto view=var(access.getOperand(0));bool swapped=access.hasAttr("axes");
          return view+"[unsafe_offset=r*Int("+view+(swapped?"s1)":"s0)")+
                 "+j*Int("+view+(swapped?"s0)]":"s1)]");
        };
        line(2,"for j in range(Int(cols)):");
        line(3,address(write)+" = "+address(read));
        continue;
      }
      auto kind = loop.getAttrOfType<StringAttr>("algebra");
      bool sum = kind == "sum", reduction = kind != "store";
      auto lanes =
          std::to_string(loop.getAttrOfType<IntegerAttr>("width").getInt());
      std::string result = temp(), acc = result + "_acc";
      bool moments=kind=="welford";
      if(moments){
        line(2,"var "+acc+" = "+(tail?var(loop.getOperand(0)):"(Int32(0), Float32(0), Float32(0))"));
      } else if (reduction)
        line(2, "var " + acc + " = SIMD[DType.float32, " + lanes + "](" +
                    (sum ? "0.0" : "-inf[DType.float32]()") + ")");
      std::string stop = "Int(cols)//" + std::to_string(fullWidth) + "*" +
                         std::to_string(fullWidth);
      if (!tail && !owners) {
        line(2, "for base in range(0, " + stop + ", Int(columns_per_block)):");
        depth = 1;
      }
      if(owners)line(2,"for j in range(1):");
      else line(2, "for j in range(" +
                  (tail ? stop + ", Int(cols), 1"
                        : "base, min(base + Int(columns_per_block), " + stop +
                              "), " + lanes) +
                  "):");
      auto &bb = loop.getRegion(0).front();
      vars[bb.getArgument(0)] = "j";
      for (auto &op : bb) {
        auto code = op.getName().stripDialect();
        auto arg = [&](unsigned i) { return var(op.getOperand(i)); };
        auto lhs = temp();
        if (code == "yield") {
          if(moments){
            line(3,"comptime for lane in range("+lanes+"):");
            line(4,"var total = "+acc+"[0]+1");
            line(4,"var delta = "+arg(0)+"[lane]-"+acc+"[1]");
            line(4,"var correction = delta/Float32(total)");
            line(4,"if "+acc+"[0] == 0:");
            line(5,acc+" = (total,"+arg(0)+"[lane], Float32(0))");
            line(4,"else:");
            line(5,acc+" = (total, "+acc+"[1]+correction, "+acc+"[2]+delta*Float32("+acc+"[0])*correction)");
          } else if (reduction)
            line(3, acc + " = " +
                        (sum ? acc + " + " + arg(0)
                             : "max(" + acc + "," + arg(0) + ")"));
        } else if(code=="owner_read" || code=="owner_write"){
          auto address=arg(0)+"[unsafe_offset=r*Int("+arg(0)+"s0)]";
          line(3,code=="owner_read"?"var "+lhs+" = "+address:address+" = "+arg(2));
        } else if (code == "read" || code == "write") {
          auto view = cast<MemRefType>(op.getOperand(0).getType());
          bool matrix = view.getRank() == 2;
          bool swap = bool(op.getAttr("axes"));
          auto a = arg(0),
               stride = "Int(" + a + "s" + (matrix && !swap ? "1" : "0") + ")";
          auto offset =
              (matrix ? "r*Int(" + a + (swap ? "s1)+" : "s0)+") : "") +
              std::string("j*") + stride;
          if(auto axis=fn->getAttrOfType<IntegerAttr>("kir.owner_axis")){
            int rank=view.getRank(), owner=axis.getInt();
            auto shape=fn.getArgAttrOfType<DenseI64ArrayAttr>(cast<BlockArgument>(op.getOperand(0)).getArgNumber()-1,"mojo_row.shape");
            std::string last="Int(cols)";
            for(int i=0;i+1<rank;++i)if(i!=owner)last+="//Int("+a+"d"+std::to_string(i)+")";
            auto size=[&](int i){return shape?std::to_string(std::max<int64_t>(1,shape[i])):rank==2?"Int(cols)":i==rank-1?"("+last+")":"Int("+a+"d"+std::to_string(i)+")";};
            auto coordinate=[&](std::string index){
              std::string relative="r*Int("+a+"s"+std::to_string(owner)+")";
              for(int i=rank-1;i>=0;--i)if(i!=owner){
                relative+="+("+index+"%"+size(i)+")*Int("+a+"s"+std::to_string(i)+")";
                index="("+index+"//"+size(i)+")";
              }
              return relative;
            };
            auto address=a+"[unsafe_offset="+coordinate("(j+lane)")+"]";
            if(code=="read")line(3,"var "+lhs+" = SIMD[DType."+dtype(view.getElementType())+","+lanes+"](0)");
            int fast=rank-1-(owner==rank-1);
            auto span=size(fast);
            auto step="Int("+a+"s"+std::to_string(fast)+")";
            auto base=coordinate("j");
            auto ptr=a+".unsafe_offset("+base+")";
            line(3,"if "+step+" == 1 and j % "+span+" + "+lanes+" <= "+span+":");
            line(4,code=="read"?lhs+" = "+ptr+".unsafe_load[width="+lanes+"]()":ptr+".unsafe_store("+arg(3)+")");
            if(code=="read"){
              line(3,"elif "+step+" == 0 and j % "+span+" + "+lanes+" <= "+span+":");
              line(4,lhs+" = SIMD[DType."+dtype(view.getElementType())+","+lanes+"]("+a+"[unsafe_offset="+base+"])");
            }
            line(3,"else:");
            line(4,"comptime for lane in range("+lanes+"):");
            line(5,code=="read"?lhs+"[lane] = "+address:address+" = "+arg(3)+"[lane]");
            if(code=="read")vars[op.getResult(0)]=lhs;
            continue;
          }
          if(view.getRank()>2){
            stride="Int("+a+"s"+std::to_string(view.getRank()-1)+")";
            offset="j*"+stride;
            std::string row="r";
            for(int i=view.getRank()-2;i>=0;--i){
              if(projected) {
                offset+="+(r-Int(first))*Int("+a+"s"+std::to_string(i)+")";
                break;
              }
              auto size="Int("+a+"d"+std::to_string(i)+")";offset+="+("+row+"%"+size+")*Int("+a+"s"+std::to_string(i)+")";
              row="("+row+"//"+size+")";
            }
          }
          if (code == "read") {
            line(3, "var " + lhs + " = SIMD[DType." +
                        dtype(view.getElementType()) + "," + lanes + "](0)");
            line(3, "if " + stride + " == 1:");
            line(4, lhs + " = " + a + ".unsafe_offset(" + offset +
                        ").unsafe_load[width=" + lanes + "]()");
            line(3, "elif "+stride+" == 0:");
            line(4, lhs+" = SIMD[DType."+dtype(view.getElementType())+","+lanes+"]("+a+"[unsafe_offset="+offset+"])");
            line(3, "else:");
            line(4, "comptime for lane in range(" + lanes + "):");
            line(5, lhs + "[lane] = " + a + "[unsafe_offset=" + offset +
                        "+lane*" + stride + "]");
          } else {
            line(3, "if " + stride + " == 1:");
            line(4, a + ".unsafe_offset(" + offset + ").unsafe_store(" +
                        arg(3) + ")");
            line(3, "else:");
            line(4, "comptime for lane in range(" + lanes + "):");
            line(5, a + "[unsafe_offset=" + offset + "+lane*" + stride +
                        "] = " + arg(3) + "[lane]");
          }
        } else if (code == "apply") {
          auto opKind = op.getAttrOfType<StringAttr>("kind").getValue();
          std::string expr;
          if (opKind == "ordinal") {
            line(3, "var " + lhs + " = SIMD[DType." +
                        dtype(op.getResult(0).getType()) + "," + lanes +
                        "](0)");
            line(3, "for lane in range(" + lanes + "):");
            line(4, lhs + "[lane] = Scalar[DType." +
                        dtype(op.getResult(0).getType()) +
                        "](r*Int(cols)+j+lane)");
            vars[op.getResult(0)] = lhs;
            continue;
          }
          if(opKind=="field")expr=arg(0)+"["+std::to_string(op.getAttrOfType<IntegerAttr>("position").getInt())+"]";
          else if (opKind == "uniform")
            expr = "uniform_at[" + lanes + "](" + arg(0) + "," + arg(1) + ")";
          else if (opKind == "select") {
            auto t = "SIMD[DType." + dtype(op.getResult(0).getType()) + "," +
                     lanes + "]";
            expr = arg(0) + ".select(" + arg(1) + "," +
                   (isa<VectorType>(op.getOperand(2).getType())
                        ? arg(2)
                        : t + "(" + arg(2) + ")") +
                   ")";
          } else if (opKind == "constant")
            expr =
                "Scalar[DType." + dtype(op.getResult(0).getType()) + "](" +
                intentdsl::kir::literal(op.getAttrOfType<TypedAttr>("value")) +
                ")";
          else if (opKind == "maximum") {
            auto expanded = [&](unsigned i) {
              return isa<VectorType>(op.getResult(0).getType()) &&
                             !isa<VectorType>(op.getOperand(i).getType())
                         ? "SIMD[DType." + dtype(op.getResult(0).getType()) +
                               "," + lanes + "](" + arg(i) + ")"
                         : arg(i);
            };
            auto a = expanded(0), b = expanded(1);
            expr = a + ".ne(" + a + ").select(" + a + "," + b + ".ne(" + b +
                   ").select(" + b + ",max(" + a + "," + b + ")))";
          } else if (opKind == "cast")
            expr = arg(0) + ".cast[DType." + dtype(op.getResult(0).getType()) +
                   "]()";
          else if (opKind == "sqrt" || opKind == "exp" || opKind == "rsqrt")
            expr = opKind.str() + "(" + arg(0) + ")";
          else {
            auto token = llvm::StringSwitch<StringRef>(opKind)
                             .Case("add", "+")
                             .Case("sub", "-")
                             .Case("mul", "*")
                             .Case("div", "/")
                             .Case("xor", "^")
                             .Case("and", "&")
                             .Case("shl", "<<")
                             .Case("shr", ">>")
                             .Case("gt", ">")
                             .Default("");
            if (token.empty())
              return llvm::createStringError("unsupported Mojo expression");
            auto expand = [&](unsigned i) {
              return isa<VectorType>(op.getResult(0).getType()) &&
                             !isa<VectorType>(op.getOperand(i).getType())
                         ? "SIMD[DType." + dtype(op.getOperand(i).getType()) +
                               "," + lanes + "](" + arg(i) + ")"
                         : arg(i);
            };
            expr = opKind == "gt"
                       ? expand(0) + ".gt(" + expand(1) + ")"
                       : expand(0) + " " + token.str() + " " + expand(1);
          }
          line(3, "var " + lhs + " = " + expr);
        } else
          return llvm::createStringError("unrecognized Mojo sweep instruction");
        if (op.getNumResults())
          vars[op.getResult(0)] = lhs;
      }
      depth = 0;
      if (reduction) {
        if(moments){vars[loop.getResult(0)]=acc;continue;}
        auto value = acc + (sum ? ".reduce_add()" : ".reduce_max()");
        if (loop.getNumOperands())
          value = sum ? var(loop.getOperand(0)) + " + " + value
                      : "max(" + var(loop.getOperand(0)) + "," + value + ")";
        line(2, "var " + result + " = " + value);
        vars[loop.getResult(0)] = result;
      }
    }

    return llvm::Error::success();
  }
};
} // namespace
llvm::Expected<std::string> mojoSource(Operation *task) {
  std::string s;
  llvm::raw_string_ostream out(s);
  if (auto e = Source(out).write(task))
    return e;
  return s;
}
} // namespace CI_UNIT_NAMESPACE
