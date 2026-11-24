#include "compilerinfra/Terminal.h"
#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"
#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "intentdsl/levels/execution/block/ir/Grid.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringExtras.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
namespace {
bool viewHasAnyFastMath(const LocalOpView &v, arith::FastMathFlags mask) {
  auto attr = v.attrOfType<arith::FastMathFlagsAttr>("fastmath");
  return attr && bitEnumContainsAny(attr.getValue(), mask);
}
bool viewHasFastMath(const LocalOpView &v, arith::FastMathFlags req) {
  auto attr = v.attrOfType<arith::FastMathFlagsAttr>("fastmath");
  return attr && bitEnumContainsAll(attr.getValue(), req);
}

llvm::Error customRowOp(Operation &op, PrintPoint point, PrintContext &c) {
  auto n = op.getName().stripDialect();
  if (point != PrintPoint::Enter)
    return llvm::Error::success();
  auto name = [&](Value v) { return c.name(v); };
  auto arg = [&](unsigned i) { return name(op.getOperand(i)); };
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
  if (n == "pid") {
    std::string e = "tl.program_id(0).to(tl.int64)";
    auto fn = op.getParentOfType<func::FuncOp>();
    Builder b(fn.getContext());
    if (fn->getAttr("block.grid") == intentdsl::block::grid(b, true))
      e += op.getAttrOfType<IntegerAttr>("axis").getInt() == 0
               ? " // tl.cdiv(N,WIDTH)"
               : " % tl.cdiv(N,WIDTH)";
    c.line(name(op.getResult(0)) + " = " + e);
    return llvm::Error::success();
  }
  if (n == "offsets") {
    auto ownerAxis = op.getParentOfType<func::FuncOp>()
                         ->getAttrOfType<IntegerAttr>("kir.owner_axis");
    int dims = cast<MemRefType>(op.getOperand(0).getType()).getRank();
    std::string e;
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
    } else {
      auto rank = op.getAttrOfType<BoolAttr>("matrix").getValue() ? 2 : 1;
      bool swap = op.getAttrOfType<BoolAttr>("permuted") &&
                  op.getAttrOfType<BoolAttr>("permuted").getValue();
      auto base =
          rank == 2 ? arg(1) + "[:, None] * " + arg(0) + (swap ? "_s1" : "_s0")
                    : "tl.full((ROWS, 1), 0, tl.int64)";
      e = base + " + " + arg(2) + "[None, :] * " + arg(0) + "_s" +
          std::to_string(swap ? 0 : rank - 1);
    }
    c.line(name(op.getResult(0)) + " = " + e);
    return llvm::Error::success();
  }
  return llvm::createStringError("unhandled custom row operation " + n.str());
}

llvm::Error localRowOp(const LocalOpView &v, PrintPoint point,
                       PrintContext &c) {
  auto n = v.operation().split('.').second;
  if (n == "tiles") {
    if (point == PrintPoint::Enter) {
      for (unsigned i = 0; i < v.numResults(); ++i)
        c.line(v.result(i).str() + " = " + v.operand(1 + i).str());
    } else if (point == PrintPoint::BeginBlock) {
      c.line("for " + v.blockArguments()[0] + " in range(tl.cdiv(N, WIDTH)):");
      ++c.indent;
      for (unsigned i = 0; i < v.numResults(); ++i)
        c.line(v.blockArguments()[1 + i] + " = " + v.result(i).str());
    } else if (point == PrintPoint::Leave) {
      --c.indent;
    }
    return llvm::Error::success();
  }
  if (point != PrintPoint::Enter)
    return llvm::Error::success();

  auto alg = v.attrOfType<StringAttr>("algebra");
  if ((n == "reduce" || n == "identity" || n == "combine" ||
       n == "neutralize") &&
      (!alg || (alg != "sum" && alg != "max_num")))
    return llvm::createStringError("unrecognized row algebra");
  bool sum = alg && alg == "sum";
  std::string e;
  if (n == "uniform") {
    auto prefix = ("rng" + v.result(0)).str();
    auto lane = ("(" + v.operand(1) + " & 3)").str();
    c.line(prefix + "a, " + prefix + "b, " + prefix + "c, " + prefix +
           "d = tl.randint4x(" + v.operand(0) + ", " + v.operand(1) + " >> 2)");
    e = "(tl.where(" + lane + " == 0," + prefix + "a,tl.where(" + lane +
        " == 1," + prefix + "b,tl.where(" + lane + " == 2," + prefix + "c," +
        prefix +
        "d))).to(tl.uint32) >> 8).to(tl.float32) * 0.000000059604644775390625";
  } else if (n == "linear_index") {
    bool wide = cast<RankedTensorType>(v.resultType(0))
                    .getElementType()
                    .isUnsignedInteger(64);
    e = ("(" + v.operand(1) + (wide ? "[:,None].to(tl.uint64)" : "[:,None]") +
         "*N+" + v.operand(2) + "[None,:]).to(" +
         (wide ? "tl.uint64)" : "tl.int32)"))
            .str();
  } else if (n == "constant") {
    e = intentdsl::kir::literal(v.attrOfType<TypedAttr>("value"));
  } else if (n == "neutralize") {
    e = ("tl.where(" + v.operand(1) + ", " + v.operand(0) + ", " +
         (sum ? "0.0" : "-float('inf')") + ")")
            .str();
  } else if (n == "reduce") {
    e = (std::string(sum ? "tl.sum" : "tl.max") + "(" + v.operand(0) + ", 1)")
            .str();
  } else if (n == "identity") {
    e = std::string("tl.full((ROWS,), ") + (sum ? "0.0" : "-float('inf')") +
        ", tl.float32)";
  } else if (n == "combine") {
    e = (sum ? v.operand(0) + " + " + v.operand(1)
             : "tl.maximum(" + v.operand(0) + ", " + v.operand(1) + ")")
            .str();
  } else if (n == "div") {
    e = (viewHasAnyFastMath(v, arith::FastMathFlags::arcp |
                                   arith::FastMathFlags::afn)
             ? v.operand(0) + " / " + v.operand(1)
             : "tl.div_rn(" + v.operand(0) + ", " + v.operand(1) + ")")
            .str();
  } else if (n == "sqrt") {
    e = (viewHasFastMath(v, arith::FastMathFlags::afn)
             ? "tl.sqrt(" + v.operand(0) + ")"
             : "tl.sqrt_rn(" + v.operand(0) + ")")
            .str();
  } else if (n == "cast") {
    auto t = getElementTypeOrSelf(v.resultType(0));
    e = (v.operand(0) + ".to(tl." +
         std::string(t.isInteger(32) ? "int32"
                     : t.isBF16()    ? "bfloat16"
                     : t.isF16()     ? "float16"
                                     : "float32") +
         ")")
            .str();
  } else {
    return llvm::createStringError("unhandled local row operation " + n.str());
  }
  c.line(v.result(0).str() + " = " + e);
  return llvm::Error::success();
}

llvm::Error localDynamicTile(const LocalOpView &v, PrintPoint point,
                             PrintContext &c) {
  if (point != PrintPoint::Enter)
    return llvm::Error::success();
  auto n = v.operation().split('.').second;
  auto num = [&](StringRef key) {
    return std::to_string(v.attrOfType<IntegerAttr>(key).getInt());
  };
  std::string s;
  if (n == "splat") {
    std::vector<std::string> ds;
    for (auto a : v.attrOfType<ArrayAttr>("dimensions"))
      ds.push_back(cast<StringAttr>(a).str());
    auto el = getElementTypeOrSelf(v.resultType(0));
    s = "tl.full((" + llvm::join(ds, ",") + ",)," + v.operand(0).str() + "," +
        (el.isInteger(32) ? "tl.int32" : "tl.float32") + ")";
  } else if (n == "view_offset") {
    std::vector<std::string> parts;
    for (unsigned i = 1; i < v.numOperands(); ++i)
      parts.push_back("tl.cast(" + v.operand(i).str() + ",tl.int64)*" +
                      v.operand(0).str() + "_s" + std::to_string(i - 1));
    s = llvm::join(parts, " + ");
  } else if (n == "fetch_add") {
    auto scope = v.attrOfType<StringAttr>("scope").getValue();
    if (scope != "workgroup" && scope != "device")
      return llvm::createStringError("invalid Triton atomic scope " +
                                     scope.str());
    std::string targetScope = scope == "workgroup" ? "cta" : "gpu";
    s = ("tl.atomic_add(" + v.operand(0) + "+" + v.operand(1) + "," +
         v.operand(2) + ",mask=" + v.operand(3) + ",sem='relaxed',scope='" +
         targetScope + "')")
            .str();
  } else if (n == "extract_lane") {
    auto ax = v.attrOfType<IntegerAttr>("axis").getInt();
    s = (ax == 1
             ? "tl.reshape(tl.gather(" + v.operand(0) +
                   ",tl.full((ROWS,1,WIDTH)," + v.operand(1) +
                   ",tl.int32),1),(ROWS,WIDTH))"
             : "tl.reshape(tl.gather(" + v.operand(0) +
                   ",tl.full((ROWS,WIDTH,1)," + v.operand(1) +
                   ",tl.int32),2),(ROWS,WIDTH))")
            .str();
  } else if (n == "axis_reduce") {
    auto axis = v.attrOfType<IntegerAttr>("axis").getInt();
    s = "tl." + v.attrOfType<StringAttr>("algebra").str() + "(" +
        v.operand(0).str() + "," +
        (axis == -1 ? "None" : std::to_string(axis)) + ")";
  } else if (n == "map") {
    auto kind = v.attrOfType<StringAttr>("kind").str();
    if (kind == "i64" || kind == "i32" || kind == "f32")
      s = "tl.cast(" + v.operand(0).str() + ",tl." +
          (kind == "i64"   ? "int64"
           : kind == "i32" ? "int32"
                           : "float32") +
          ")";
    else if (kind == "select")
      s = ("tl.where(" + v.operand(0) + "," + v.operand(1) + "," +
           v.operand(2) + ")")
              .str();
    else if (kind == "ceildiv")
      s = ("tl.cdiv(" + v.operand(0) + "," + v.operand(1) + ")").str();
    else if (kind == "min")
      s = ("tl.minimum(" + v.operand(0) + "," + v.operand(1) + ")").str();
    else if (kind == "add_rn" || kind == "mul_rn")
      s = ("libdevice." + kind + "(" + v.operand(0) + "," + v.operand(1) + ")")
              .str();
    else {
      static const std::map<std::string, std::string> binary = {
          {"and", "&"}, {"lt", "<"},  {"ge", ">="},       {"add", "+"},
          {"sub", "-"}, {"mul", "*"}, {"floordiv", "//"}, {"eq", "=="}};
      auto it = binary.find(kind);
      if (it == binary.end())
        return llvm::createStringError("unknown row map " + kind);
      s = v.operand(0).str() + it->second + v.operand(1).str();
    }
  } else {
    return llvm::createStringError("unhandled dynamic tile op " + n.str());
  }
  c.line(v.numResults() ? v.result(0).str() + " = " + s : s);
  return llvm::Error::success();
}
} // namespace

void addRowRules(TerminalPrinter &printer) {
  auto &ops = printer.operations;
  // Declarative templates
  ops["triton_src.output_partition"] =
      PrintRule::format("if tl.program_id(1) == {index}:", "", true);
  ops["triton_src.partition_end"] = PrintRule::format("");
  ops["triton_src.cta_barrier"] = PrintRule::format("tl.debug_barrier()");
  ops["triton_src.ne"] = PrintRule::format("$r0 = $0!=$1");
  ops["triton_src.member_tile"] = PrintRule::format(
      "$r0 = tl.broadcast_to($0[None,:],(ROWS,WIDTH)).to(tl.int32)");
  ops["triton_src.summary_reduce"] = PrintRule::format(
      "$r0 = tl.reduce(($1.to(tl.int32),tl.where($1,$0,0.),tl.full((ROWS,WIDTH),0.,tl.float32)),1,merge_moments)");
  ops["triton_src.summary_identity"] = PrintRule::format(
      "$r0 = (tl.full((ROWS,),0,tl.int32),tl.full((ROWS,),0.,tl.float32),tl.full((ROWS,),0.,tl.float32))");
  ops["triton_src.summary_merge"] = PrintRule::format(
      "$r0 = merge_moments($0[0],$0[1],$0[2],$1[0],$1[1],$1[2])");
  ops["triton_src.summary_field"] = PrintRule::format("$r0 = $0[{position}]");
  ops["triton_src.scan_sum"] = PrintRule::format("$r0 = tl.cumsum($0,axis=1)");
  ops["triton_src.select"] = PrintRule::format("$r0 = tl.where($0,$1,$2)");
  ops["triton_src.xor"] = PrintRule::format("$r0 = $0 ^ $1");
  ops["triton_src.and"] = PrintRule::format("$r0 = $0 & $1");
  ops["triton_src.shl"] = PrintRule::format("$r0 = $0 << $1");
  ops["triton_src.shr"] = PrintRule::format("$r0 = $0 >> $1");
  ops["triton_src.gt"] = PrintRule::format("$r0 = $0 > $1");
  ops["triton_src.maximum"] = PrintRule::format(
      "$r0 = tl.maximum($0,$1,propagate_nan=tl.PropagateNan.ALL)");
  ops["triton_src.rows"] =
      PrintRule::format("$r0 = $0 * ROWS + tl.arange(0, ROWS)");
  ops["triton_src.zero"] = PrintRule::format("$r0 = 0");
  ops["triton_src.columns"] = PrintRule::format(
      "$r0 = tl.arange(0, WIDTH).to(tl.int64) + tl.cast($0,tl.int64) * WIDTH");
  ops["triton_src.active"] =
      PrintRule::format("$r0 = ($1[:, None] < M) & ($2[None, :] < N)");
  ops["triton_src.load"] =
      PrintRule::format("$r0 = tl.load($0 + $1, mask=$2, other=0)");
  ops["triton_src.store"] =
      PrintRule::format("tl.store($0 + $1, $3, mask=$2)");
  ops["triton_src.broadcast_rows"] = PrintRule::format("$r0 = $0[:, None]");
  ops["triton_src.add"] = PrintRule::format("$r0 = $0 + $1");
  ops["triton_src.sub"] = PrintRule::format("$r0 = $0 - $1");
  ops["triton_src.mul"] = PrintRule::format("$r0 = $0 * $1");
  ops["triton_src.exp"] = PrintRule::format("$r0 = tl.exp($0)");
  ops["triton_src.rsqrt"] = PrintRule::format("$r0 = tl.rsqrt($0)");
  ops["triton_src.dimension"] = PrintRule::format("$r0 = {name}");
  ops["triton_src.expand"] = PrintRule::format("$r0 = tl.expand_dims($0,{axis})");
  ops["triton_src.range"] =
      PrintRule::format("$r0 = tl.arange(0,{size}).to(tl.int64)");
  ops["triton_src.read_at"] =
      PrintRule::format("$r0 = tl.load($0+$1,mask=$2,other={fill})");
  ops["triton_src.write_at"] =
      PrintRule::format("tl.store($0+$1,$2,mask=$3)");
  ops["triton_src.bin_count"] = PrintRule::format("$r0 = tl.histogram($0,WIDTH)");

  // Local rules (LocalOpView only)
  for (auto n : {"tiles", "uniform", "linear_index", "constant", "neutralize",
                 "reduce", "identity", "combine", "div", "sqrt", "cast"})
    ops["triton_src." + std::string(n)] = PrintRule::local(localRowOp);
  for (auto n :
       {"splat", "view_offset", "fetch_add", "extract_lane", "axis_reduce", "map"})
    ops["triton_src." + std::string(n)] = PrintRule::local(localDynamicTile);

  // Audited custom rules with explicit structural reasons
  ops["triton_src.yield"] = PrintRule::custom(
      customRowOp, "assigns yielded values to enclosing triton_src.tiles results");
  ops["triton_src.pid"] = PrintRule::custom(
      customRowOp, "reads verified function block.grid attribute for 2D tile decomposition");
  ops["triton_src.offsets"] = PrintRule::custom(
      customRowOp, "reads verified function kir.owner_axis for multi-axis stride decomposition");
}
} // namespace CI_UNIT_NAMESPACE
