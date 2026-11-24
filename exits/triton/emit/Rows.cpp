#include "compilerinfra/Terminal.h"
#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "intentdsl/levels/execution/block/ir/Grid.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringExtras.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
namespace {
llvm::Error row(Operation &op, PrintPoint point, PrintContext &c) {
  auto n = op.getName().stripDialect();
  auto name = [&](Value v) { return c.name(v); };
  auto arg = [&](unsigned i) { return name(op.getOperand(i)); };
  if (n == "tiles") {
    if (point == PrintPoint::Enter) {
      auto &body = op.getRegion(0).front();
      std::string limit = "tl.cdiv(N, WIDTH)";
      for (unsigned i = 0; i < op.getNumResults(); ++i)
        c.line(name(op.getResult(i)) + " = " + arg(1 + i));
      c.line("for " + name(body.getArgument(0)) + " in range(" + limit + "):");
      ++c.indent;
      for (unsigned i = 0; i < op.getNumResults(); ++i)
        c.line(name(body.getArgument(1 + i)) + " = " + name(op.getResult(i)));
    } else if (point == PrintPoint::Leave)
      --c.indent;
    return llvm::Error::success();
  }
  if (n == "output_partition") {
    if (point == PrintPoint::Enter) {
      c.line("if tl.program_id(1) == " +
             std::to_string(op.getAttrOfType<IntegerAttr>("index").getInt()) +
             ":");
      ++c.indent;
    } else if (point == PrintPoint::Leave)
      --c.indent;
    return llvm::Error::success();
  }
  if (point != PrintPoint::Enter)
    return llvm::Error::success();
  if (n == "partition_end")
    return llvm::Error::success();
  if (n == "yield") {
    if (op.getNumOperands()) {
      SmallVector<std::string> lhs, rhs;
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        lhs.push_back(name(op.getParentOp()->getResult(i)));
        rhs.push_back(arg(i));
      }
      c.line(llvm::join(lhs, ", ") + " = " + llvm::join(rhs, ", "));
    }
    return llvm::Error::success();
  }
  auto ownerAxis =
      op.getParentOfType<func::FuncOp>()->getAttrOfType<IntegerAttr>(
          "kir.owner_axis");
  auto alg = op.getAttrOfType<StringAttr>("algebra");
  if ((n == "reduce" || n == "identity" || n == "combine" ||
       n == "neutralize") &&
      (!alg || (alg != "sum" && alg != "max_num")))
    return llvm::createStringError("unrecognized row algebra");
  bool sum = alg && alg == "sum";
  std::string e;
  if (n == "ne" || n == "member_tile") {
    auto id = name(op.getResult(0));
    c.line(id + " = " +
           (n == "ne" ? arg(0) + "!=" + arg(1)
                      : "tl.broadcast_to(" + arg(0) +
                            "[None,:],(ROWS,WIDTH)).to(tl.int32)"));
    return llvm::Error::success();
  }
  if (n == "cta_barrier") {
    c.line("tl.debug_barrier()");
    return llvm::Error::success();
  }
  if (n == "summary_reduce")
    e = "tl.reduce((" + arg(1) + ".to(tl.int32),tl.where(" + arg(1) + "," +
        arg(0) + ",0.),tl.full((ROWS,WIDTH),0.,tl.float32)),1,merge_moments)";
  else if (n == "summary_identity")
    e = "(tl.full((ROWS,),0,tl.int32),tl.full((ROWS,),0.,tl.float32),tl.full(("
        "ROWS,),0.,tl.float32))";
  else if (n == "summary_merge")
    e = "merge_moments(" + arg(0) + "[0]," + arg(0) + "[1]," + arg(0) + "[2]," +
        arg(1) + "[0]," + arg(1) + "[1]," + arg(1) + "[2])";
  else if (n == "summary_field")
    e = arg(0) + "[" +
        std::to_string(op.getAttrOfType<IntegerAttr>("position").getInt()) +
        "]";
  else if (n == "scan_sum")
    e = "tl.cumsum(" + arg(0) + ",axis=1)";
  else if (n == "uniform") {
    auto prefix = "rng" + name(op.getResult(0));
    auto lane = "(" + arg(1) + " & 3)";
    c.line(prefix + "a, " + prefix + "b, " + prefix + "c, " + prefix +
           "d = tl.randint4x(" + arg(0) + ", " + arg(1) + " >> 2)");
    e = "(tl.where(" + lane + " == 0," + prefix + "a,tl.where(" + lane +
        " == 1," + prefix + "b,tl.where(" + lane + " == 2," + prefix + "c," +
        prefix +
        "d))).to(tl.uint32) >> 8).to(tl.float32) * 0.000000059604644775390625";
  } else if (n == "linear_index") {
    bool wide = cast<RankedTensorType>(op.getResult(0).getType())
                    .getElementType()
                    .isUnsignedInteger(64);
    e = "(" + arg(1) + (wide ? "[:,None].to(tl.uint64)" : "[:,None]") + "*N+" +
        arg(2) + "[None,:]).to(" + (wide ? "tl.uint64)" : "tl.int32)");
  } else if (n == "select")
    e = "tl.where(" + arg(0) + "," + arg(1) + "," + arg(2) + ")";
  else if (n == "xor" || n == "and" || n == "shr" || n == "shl" || n == "gt") {
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
  else if (n == "pid") {
    e = "tl.program_id(0).to(tl.int64)";
    auto fn = op.getParentOfType<func::FuncOp>();
    Builder b(fn.getContext());
    if (fn->getAttr("block.grid") == intentdsl::block::grid(b, true))
      e += op.getAttrOfType<IntegerAttr>("axis").getInt() == 0
               ? " // tl.cdiv(N,WIDTH)"
               : " % tl.cdiv(N,WIDTH)";
  } else if (n == "rows")
    e = arg(0) + " * ROWS + tl.arange(0, ROWS)";
  else if (n == "zero")
    e = "0";
  else if (n == "columns")
    e = "tl.arange(0, WIDTH).to(tl.int64) + tl.cast(" + arg(0) +
        ",tl.int64) * WIDTH";
  else if (n == "active")
    e = "(" + arg(1) + "[:, None] < M) & (" + arg(2) + "[None, :] < N)";
  else if (n == "offsets") {
    int dims = cast<MemRefType>(op.getOperand(0).getType()).getRank();
    if (ownerAxis) {
      auto base = arg(0), col = arg(2) + "[None,:]";
      e = arg(1) + "[:,None]*" + base + "_s" +
          std::to_string(ownerAxis.getInt());
      for (int i = dims - 1; i >= 0; --i)
        if (i != ownerAxis.getInt()) {
          auto size = "tl.maximum(" + base + "_d" + std::to_string(i) + ",1)";
          e += "+((" + col + ")%" + size + ")*" + base + "_s" +
               std::to_string(i);
          col = "((" + col + ")//" + size + ")";
        }
    } else if (dims > 2) {
      auto a = arg(0), row = arg(1);
      e = arg(2) + "[None,:]*" + a + "_s" + std::to_string(dims - 1);
      for (int i = dims - 2; i >= 0; --i) {
        auto size = a + "_d" + std::to_string(i);
        e += "+((" + row + ")%" + size + ")[:,None]*" + a + "_s" +
             std::to_string(i);
        row = "((" + row + ")//" + size + ")";
      }
      auto id = name(op.getResult(0));
      c.line(id + " = " + e);
      return llvm::Error::success();
    }
    if (!ownerAxis) {
      auto rank = op.getAttrOfType<BoolAttr>("matrix").getValue() ? 2 : 1;
      bool swap = op.getAttrOfType<BoolAttr>("permuted") &&
                  op.getAttrOfType<BoolAttr>("permuted").getValue();
      auto base =
          rank == 2 ? arg(1) + "[:, None] * " + arg(0) + (swap ? "_s1" : "_s0")
                    : "tl.full((ROWS, 1), 0, tl.int64)";
      e = base + " + " + arg(2) + "[None, :] * " + arg(0) + "_s" +
          std::to_string(swap ? 0 : rank - 1);
    }
  } else if (n == "load")
    e = "tl.load(" + arg(0) + " + " + arg(1) + ", mask=" + arg(2) +
        ", other=0)";
  else if (n == "store") {
    c.line("tl.store(" + arg(0) + " + " + arg(1) + ", " + arg(3) +
           ", mask=" + arg(2) + ")");
    return llvm::Error::success();
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
  else if (n == "sqrt")
    e = "tl.sqrt_rn(" + arg(0) + ")";
  else if (n == "exp" || n == "rsqrt")
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

  c.line(name(op.getResult(0)) + " = " + e);
  return llvm::Error::success();
}

llvm::Error dynamicTile(Operation &op, PrintPoint point, PrintContext &c) {
  if (point != PrintPoint::Enter)
    return llvm::Error::success();
  auto arg = [&](unsigned i) { return c.name(op.getOperand(i)); };
  auto n = op.getName().stripDialect();
  auto num = [&](StringRef key) {
    return std::to_string(op.getAttrOfType<IntegerAttr>(key).getInt());
  };
  std::string s;
  if (n == "dimension")
    s = op.getAttrOfType<StringAttr>("name").str();
  else if (n == "expand")
    s = "tl.expand_dims(" + arg(0) + "," + num("axis") + ")";
  else if (n == "splat") {
    std::vector<std::string> ds;
    for (auto a : op.getAttrOfType<ArrayAttr>("dimensions"))
      ds.push_back(cast<StringAttr>(a).str());
    auto el = getElementTypeOrSelf(op.getResult(0).getType());
    s = "tl.full((" + llvm::join(ds, ",") + ",)," + arg(0) + "," +
        (el.isInteger(32) ? "tl.int32" : "tl.float32") + ")";
  } else if (n == "range")
    s = "tl.arange(0," + num("size") + ").to(tl.int64)";
  else if (n == "view_offset") {
    std::vector<std::string> parts;
    for (unsigned i = 1; i < op.getNumOperands(); ++i)
      parts.push_back("tl.cast(" + arg(i) + ",tl.int64)*" + arg(0) + "_s" +
                      std::to_string(i - 1));
    s = llvm::join(parts, " + ");
  } else if (n == "read_at")
    s = "tl.load(" + arg(0) + "+" + arg(1) + ",mask=" + arg(2) +
        ",other=" + num("fill") + ")";
  else if (n == "write_at")
    s = "tl.store(" + arg(0) + "+" + arg(1) + "," + arg(2) + ",mask=" + arg(3) +
        ")";
  else if (n == "fetch_add") {
    auto scope = op.getAttrOfType<StringAttr>("scope").getValue();
    if (scope != "workgroup" && scope != "device")
      return llvm::createStringError("invalid Triton atomic scope " +
                                     scope.str());
    std::string targetScope = scope == "workgroup" ? "cta" : "gpu";
    s = "tl.atomic_add(" + arg(0) + "+" + arg(1) + "," + arg(2) +
        ",mask=" + arg(3) + ",sem='relaxed',scope='" + targetScope + "')";
  } else if (n == "bin_count")
    s = "tl.histogram(" + arg(0) + ",WIDTH)";
  else if (n == "extract_lane") {
    auto ax = op.getAttrOfType<IntegerAttr>("axis").getInt();
    s = ax == 1
            ? "tl.reshape(tl.gather(" + arg(0) + ",tl.full((ROWS,1,WIDTH)," +
                  arg(1) + ",tl.int32),1),(ROWS,WIDTH))"
            : "tl.reshape(tl.gather(" + arg(0) + ",tl.full((ROWS,WIDTH,1)," +
                  arg(1) + ",tl.int32),2),(ROWS,WIDTH))";
  } else if (n == "axis_reduce") {
    auto axis = op.getAttrOfType<IntegerAttr>("axis").getInt();
    s = "tl." + op.getAttrOfType<StringAttr>("algebra").str() + "(" + arg(0) +
        "," + (axis == -1 ? "None" : std::to_string(axis)) + ")";
  } else if (n == "map") {
    auto kind = op.getAttrOfType<StringAttr>("kind").str();
    if (kind == "i64" || kind == "i32" || kind == "f32")
      s = "tl.cast(" + arg(0) + ",tl." +
          (kind == "i64"   ? "int64"
           : kind == "i32" ? "int32"
                           : "float32") +
          ")";
    else if (kind == "select")
      s = "tl.where(" + arg(0) + "," + arg(1) + "," + arg(2) + ")";
    else if (kind == "ceildiv")
      s = "tl.cdiv(" + arg(0) + "," + arg(1) + ")";
    else if (kind == "min")
      s = "tl.minimum(" + arg(0) + "," + arg(1) + ")";
    else if (kind == "add_rn" || kind == "mul_rn")
      s = "libdevice." + kind + "(" + arg(0) + "," + arg(1) + ")";
    else {
      static const std::map<std::string, std::string> binary = {
          {"and", "&"}, {"lt", "<"},  {"ge", ">="},       {"add", "+"},
          {"sub", "-"}, {"mul", "*"}, {"floordiv", "//"}, {"eq", "=="}};
      auto it = binary.find(kind);
      if (it == binary.end())
        return llvm::createStringError("unknown row map " + kind);
      s = arg(0) + it->second + arg(1);
    }
  }
  c.line(op.getNumResults() ? c.name(op.getResult(0)) + " = " + s : s);
  return llvm::Error::success();
}
} // namespace
void addRowRules(TerminalPrinter &printer) {
  for (auto n : {"dimension", "expand", "splat", "range", "view_offset",
                 "read_at", "write_at", "bin_count", "fetch_add",
                 "extract_lane", "axis_reduce", "map"})
    printer.operations["triton_src." + std::string(n)] = dynamicTile;
  for (auto n : {"tiles",
                 "output_partition",
                 "partition_end",
                 "cta_barrier",
                 "yield",
                 "ne",
                 "member_tile",
                 "summary_reduce",
                 "summary_identity",
                 "summary_merge",
                 "summary_field",
                 "scan_sum",
                 "uniform",
                 "linear_index",
                 "select",
                 "xor",
                 "and",
                 "shr",
                 "shl",
                 "gt",
                 "constant",
                 "maximum",
                 "pid",
                 "rows",
                 "zero",
                 "columns",
                 "active",
                 "offsets",
                 "load",
                 "store",
                 "neutralize",
                 "reduce",
                 "identity",
                 "combine",
                 "broadcast_rows",
                 "add",
                 "sub",
                 "mul",
                 "div",
                 "sqrt",
                 "exp",
                 "rsqrt",
                 "cast"})
    printer.operations["triton_src." + std::string(n)] = row;
}
} // namespace CI_UNIT_NAMESPACE
