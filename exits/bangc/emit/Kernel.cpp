#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
namespace {
using Text = std::string;
Text ctype(Type t) {
  if (auto s = dyn_cast<ShapedType>(t))
    t = s.getElementType();
  if (t.isBF16())
    return "unsigned short";
  if (t.isInteger(32))
    return "int";
  return t.isF16() ? "half" : "float";
}
bool tile(Value v) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  return t && t.getRank() == 2;
}
class Source {
  llvm::raw_ostream &o;
  func::FuncOp fn;
  bool resident;
  unsigned depth;
  llvm::DenseMap<Value, Text> names;
  unsigned seq = 0;
  Text fresh() { return "v" + std::to_string(seq++); }
  Text name(Value v) { return names.lookup(v); }
  void line(const Text &s) { o << s << "\n"; }
  Text args(bool decl) {
    Text s;
    for (auto a : fn.getArguments()) {
      auto t = dyn_cast<MemRefType>(a.getType());
      auto n = "a" + std::to_string(a.getArgNumber());
      names[a] = n;
      s += (decl ? ctype(a.getType()) + (t ? " *" : " ") : "") + n + ", ";
      if (t)
        for (int j = 0; j < t.getRank(); ++j)
          s += (decl ? "long long " : "") + n + "d" + std::to_string(j) + ", ";
      if (t)
        for (int j = 0; j < t.getRank(); ++j)
          s += (decl ? "long long " : "") + n + "s" + std::to_string(j) + ", ";
    }
    s += decl ? "long long M,long long N,long long grid" : "M,N,grid";
    return s;
  }
  // A submitted group owns complete, pitched slots, even for partial edge
  // groups.
  Text address(Operation &op, StringRef key = "bangc.offset") {
    auto parts = op.getAttrOfType<DenseI64ArrayAttr>(key);
    return "(storage+P*(" + std::to_string(parts[0] * depth) + "*R+" +
           std::to_string(parts[1]) + "))";
  }
  std::pair<Text, Text> strides(Operation &op, unsigned start,
                                const Text &view) {
    Text steps[2] = {"0", "0"};
    for (auto [i, c] : llvm::enumerate(op.getOperands().drop_front(start)))
      steps[cast<OpResult>(c).getResultNumber()] =
          view + "s" + std::to_string(i);
    return {steps[0], steps[1]};
  }
  void issue(Block &fetch, const Text &iteration, const Text &slot) {
    line("{ long long incoming=" + iteration + ";");
    line(resident ? "long long row0=incoming*R, col=0;"
                  : "long long row0=group*R, col=incoming*W;");
    line("int live=(int)((N-col)<W?(N-col):W);");
    for (auto &op : fetch.without_terminator()) {
      if (op.getName().getStringRef() == "dsa.window")
        continue;
      auto src = op.getOperand(0);
      auto n = name(src), dst = name(op.getResult(0)),
           type = ctype(src.getType());
      auto [rowStride, stride] = strides(op, 2, n);
      line("for(int j=0;j<R && row0+j<M;++j){");
      line("auto *src=" + n + "+" + "(row0+j)*" + rowStride + "+" + "col*" +
           stride + ";");
      line("auto *dst=" + dst + "+((" + slot + ")*R+j)*P;");
      line("if(" + stride + "==1){" +
           Text(op.getAttrOfType<StringAttr>("bangc.primitive") == "dma_async"
                    ? "__memcpy_async"
                    : "__memcpy") +
           "(dst,src,live*sizeof(" + type +
           "),GDRAM2NRAM);}else{for(int k=0;k<live;++k) dst[k]=src[k*" +
           stride + "];}}");
    }
    line("}");
  }
  llvm::Error phase(Operation &op) {
    auto &fetch = op.getRegion(0).front(), &body = op.getRegion(1).front();
    Text carry;
    if (op.getNumResults()) {
      carry = name(op.getOperand(0));
      names[op.getResult(0)] = carry + "[j]";
    }
    line("{");
    unsigned index = 0;
    for (auto &copy : fetch.without_terminator()) {
      if (copy.getName().getStringRef() == "dsa.window")
        continue;
      auto n = fresh();
      auto t = ctype(copy.getResult(0).getType());
      names[copy.getResult(0)] = n;
      names[body.getArgument(index++)] = n + "+(slot*R+j)*P";
      line("auto *" + n + "=(" + t + "*)" + address(copy) + ";");
    }
    if (!carry.empty())
      names[body.getArgument(index)] = carry + "[j]";
    auto status = emitSchedule(op.getRegion(2).front(), fetch, body, carry);
    line("}");
    return status;
  }
  llvm::Error emitSchedule(Block &steps, Block &fetch, Block &body,
                           const Text &carry) {
    for (auto &op : steps) {
      auto n = op.getName().stripDialect();
      auto arg = [&](unsigned i) { return name(op.getOperand(i)); };
      if (n == "range") {
        bool tasks =
            op.getAttrOfType<StringAttr>("source") == "participant_stride";
        for (auto [i, expr] : llvm::enumerate(SmallVector<Text>{
                 tasks ? "taskId" : "0", tasks ? "grid" : "((N+W-1)/W)",
                 tasks ? "taskDim" : "1"}))
          names[op.getResult(i)] = expr;
      } else if (n == "slot_zero")
        names[op.getResult(0)] = "0";
      else if (n == "iterate") {
        auto &bb = op.getRegion(0).front();
        names[bb.getArgument(0)] = "pos";
        names[bb.getArgument(1)] = "slot";
        line("int slot=" + arg(3) + ";for(long long pos=" + arg(0) + ";pos<" +
             arg(1) + ";pos+=" + arg(2) + "){");
        if (auto e = emitSchedule(bb, fetch, body, carry))
          return e;
        line("}");
      } else if (n == "issue_if_in_bounds") {
        line("if(" + arg(0) + "<" + arg(1) + ")");
        issue(fetch, arg(0), arg(2));
      } else if (n == "next")
        names[op.getResult(0)] = "(" + arg(0) + "+" + arg(1) + ")";
      else if (n == "other_slot")
        names[op.getResult(0)] = "(1-" + arg(0) + ")";
      else if (n == "wait_slot")
        line("__sync();");
      else if (n == "advance")
        line("slot=" + arg(0) + ";");
      else if (n == "compute_members") {
        if (auto e = compute(body, carry))
          return e;
      } else if (n != "end")
        return llvm::createStringError("unlowered supply schedule operation " +
                                       n);
    }
    return llvm::Error::success();
  }
  llvm::Error compute(Block &body, const Text &carry) {
    line(resident ? "long long row0=pos*R,col=0;"
                  : "long long row0=group*R,col=pos*W;");
    line("int live=(int)((N-col)<W?(N-col):W), lanes=(live+63)/64*64;");
    line("for(int j=0;j<R && row0+j<M;++j){");
    for (auto &inst : body) {
      auto n = inst.getName().stripDialect();
      auto prim = inst.getAttrOfType<StringAttr>("bangc.primitive").getValue();
      auto in = [&](unsigned i) { return name(inst.getOperand(i)); };
      if (n == "await") {
        names[inst.getResult(0)] = "(" + in(0) + ")";
        continue;
      }
      if (n == "window")
        continue;
      if (n == "complete") {
        if (inst.getNumOperands() > 1)
          line(carry + "[j]=" + in(1) + ";");
        continue;
      }
      if (n == "drain") {
        if (llvm::any_of(inst.getOperands(), [](Value v) {
              return v.getDefiningOp()
                  ->getAttrOfType<BoolAttr>("async")
                  .getValue();
            }))
          line("__sync();");
        continue;
      }
      if (n == "write_back") {
        auto ptr = in(0), src = in(1), t = ctype(inst.getOperand(0).getType());
        auto [rowStride, columnStride] = strides(inst, 3, ptr);
        line("{auto *out=" + ptr + "+(row0+j)*" + rowStride + "+col*" +
             columnStride + ";");
        Text move =
            prim == "async_store_extent" ? "__memcpy_async" : "__memcpy";
        line("if(" + columnStride + "==1)" + move + "(out," + src +
             ",live*sizeof(" + t +
             "),NRAM2GDRAM);else for(int k=0;k<live;++k)out[k*" + columnStride +
             "]=" + src + "[k];}");
        continue;
      }
      auto val = inst.getResult(0);
      Text id = fresh();
      names[val] = id;
      if (n == "reduce") {
        Text fn = prim == "row_sum" ? "row_sum" : "row_max_num";
        line("float " + id + "=" + fn + "((float*)" + address(inst) + "," +
             in(0) + ",live);");
        continue;
      }
      bool vec = tile(val);
      auto type = ctype(val.getType());
      if (vec)
        line("auto *" + id + "=(" + type + "*)" + address(inst) + ";");
      if (prim == "quotient_for_narrow")
        continue;
      if (prim == "certified_half_quotient") {
        auto *div = inst.getOperand(0).getDefiningOp();
        line("row_narrow_quotient(" + id + "," + in(0) + ",(float*)" +
             address(inst, "bangc.scratch") + "," + name(div->getOperand(0)) +
             "," + name(div->getOperand(1)) + ",live,lanes);");
        continue;
      }
      auto operand = [&](unsigned i) {
        return in(i) + (tile(inst.getOperand(i)) ? "[k]" : "");
      };
      auto a = operand(0);
      Text expr;
      if (n == "cast")
        expr = "(" + type + ")" + a;
      else if (n == "rsqrt")
        expr = "(1.0f/sqrtf(" + a + "))";
      else if (n == "exp")
        expr = "expf(" + a + ")";
      else if (n == "combine")
        expr = prim == "sum" ? a + "+" + operand(1)
                             : "fmaxf(" + a + "," + operand(1) + ")";
      else if (n == "mul" || n == "add" || n == "sub" || n == "div")
        expr = a +
               (n == "mul"   ? "*"
                : n == "add" ? "+"
                : n == "sub" ? "-"
                             : "/") +
               operand(1);
      else
        return llvm::createStringError("unsupported BANG row instruction " + n);
      if (vec && n == "cast" && ctype(inst.getOperand(0).getType()) != type)
        line(Text(type == "half" ? "__bang_float2half_rn"
                                 : "__bang_half2float") +
             "(" + id + "," + in(0) + ",lanes);");
      else if (vec && (n == "mul" || n == "add" || n == "sub") &&
               tile(inst.getOperand(0)) && tile(inst.getOperand(1)))
        line("__bang_" + n.str() + "(" + id + "," + in(0) + "," + in(1) +
             ",lanes);");
      else if (vec && (n == "mul" || n == "sub" || n == "add") &&
               tile(inst.getOperand(0)) && !tile(inst.getOperand(1)))
        line("__bang_" + Text(n == "sub" ? "add" : n.str()) + "_scalar(" + id +
             "," + in(0) + "," + (n == "sub" ? "-(" : "(") + in(1) +
             "),lanes);");
      else if (vec && n == "exp")
        line("row_exp(" + id + "," + in(0) + ",live,lanes);");
      else
        line((vec ? "for(int k=0;k<live;++k)" + id + "[k]" : type + " " + id) +
             "=" + expr + ";");
    }
    line("}");
    return llvm::Error::success();
  }

public:
  Source(llvm::raw_ostream &os, func::FuncOp function, unsigned d)
      : o(os), fn(function),
        resident(fn->getAttrOfType<StringAttr>("block.topology") == "resident"),
        depth(d) {}
  llvm::Error write() {
    line("#include <bang.h>\n#include <cnrt.h>\n#include <math.h>");
    o <<
#include "RowMath.inc"
        ;
    line("constexpr int R=INTENT_TILE_R, W=INTENT_TILE_W, P=INTENT_TILE_P, "
         "BYTES=INTENT_TILE_BYTES;");
    line("__mlu_global__ void tile(" + args(true) +
         "){ __nram__ unsigned char storage[BYTES] "
         "__attribute__((aligned(128)));");
    return emitKernel();
  }
  llvm::Error outer(Block &block) {
    for (auto &op : block) {
      auto code = op.getName().stripDialect();
      if (code == "task_groups") {
        line("for(long long group=taskId;group<grid;group+=taskDim){");
        if (auto e = outer(op.getRegion(0).front()))
          return e;
        line("}");
      } else if (code == "phase") {
        if (auto e = phase(op))
          return e;
      } else if (code == "identity") {
        auto n = fresh();
        names[op.getResult(0)] = n;
        line("float " + n + "[R];for(int j=0;j<R;++j)" + n + "[j]=" +
             (op.getAttrOfType<StringAttr>("algebra") == "sum" ? "0.0f"
                                                               : "-INFINITY") +
             ";");
      } else if (code != "return" && code != "end")
        return llvm::createStringError("unexpected native outer op " + code);
    }
    return llvm::Error::success();
  }
  llvm::Error emitKernel() {
    if (auto e = outer(fn.front()))
      return e;
    line("}");
    line("extern \"C\" int intent_enqueue(" + args(true) +
         ",int cores,void *queue){");
    line("if(grid==0 || N==0)return 0; cnrtDim3_t "
         "dim={(unsigned)(grid<cores?grid:cores),1,1};");
    line("tile<<<dim,cnrtFuncTypeBlock,(cnrtQueue_t)queue>>>(" + args(false) +
         ");return (int)cnrtGetLastError();}");
    return llvm::Error::success();
  }
};
} // namespace
llvm::Error emit(ModuleOp m, llvm::raw_pwrite_stream &out) {
  auto fn = *m.getOps<func::FuncOp>().begin();
  auto layout = fn->getAttrOfType<DictionaryAttr>("dsa.layout");
  std::string program;
  llvm::raw_string_ostream text(program);
  auto depth = layout.getAs<IntegerAttr>("depth").getInt();
  if (auto e = Source(text, fn, depth).write())
    return e;
  llvm::json::Array rows;
  for (auto a : layout.getAs<ArrayAttr>("rows")) {
    auto allocation = cast<DictionaryAttr>(a);
    auto id = allocation.getAs<IntegerAttr>("id");
    for (auto attr : fn->getAttrOfType<ArrayAttr>("block.configs")) {
      auto row = cast<DictionaryAttr>(attr);
      if (row.get("id") != id)
        continue;
      unsigned tasks;
      if (row.getAs<StringAttr>("payload").getValue().getAsInteger(10, tasks))
        return llvm::createStringError("invalid physical task count");
      rows.push_back(llvm::json::Object{
          {"id", id.getInt()},
          {"rows", row.getAs<IntegerAttr>("rows").getInt()},
          {"width", row.getAs<IntegerAttr>("width").getInt()},
          {"bytes", allocation.getAs<IntegerAttr>("bytes").getInt()},
          {"pitch", allocation.getAs<IntegerAttr>("pitch").getInt()},
          {"tasks", tasks}});
    }
  }
  auto abi =
      llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi").getValue());
  if (!abi)
    return abi.takeError();
  auto guard = fn->getAttrOfType<DenseI64ArrayAttr>("block.guard");
  llvm::json::Array axes;
  for (auto a : fn->getAttrOfType<ArrayAttr>("block.grid"))
    axes.push_back(cast<StringAttr>(a).str());
  out << llvm::json::Value(llvm::json::Object{
      {"version", 2},
      {"source", program},
      {"abi", std::move(*abi)},
      {"rows", std::move(rows)},
      {"depth", depth},
      {"grid", llvm::json::Array{std::move(axes), 1, 1}},
      {"guard",
       llvm::json::Object{{"min_width", guard[0]},
                          {"max_width", guard[1] < 0
                                            ? llvm::json::Value(nullptr)
                                            : llvm::json::Value(guard[1])}}}});
  return llvm::Error::success();
}
} // namespace CI_UNIT_NAMESPACE
