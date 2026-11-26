#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/StringExtras.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
namespace {
bool rowHasAnyFastMath(const LocalOpView &v, arith::FastMathFlags mask) {
  auto attr = v.attrOfType<arith::FastMathFlagsAttr>("fastmath");
  return attr && bitEnumContainsAny(attr.getValue(), mask);
}

std::string memberCountExpr(Type viewTy, StringRef name, int64_t ownerAxis) {
  std::string product = "1";
  for (int i = 0; i < cast<MemRefType>(viewTy).getRank(); ++i)
    if (i != ownerAxis)
      product += "*" + name.str() + ".shape[" + std::to_string(i) + "]";
  return product;
}

llvm::Error localRowRule(const LocalOpView &v, PrintPoint at, PrintContext &c) {
  auto code = v.operation().split('.').second;
  if (code == "tiles") {
    unsigned nRes = v.numResults();
    auto idAttr = v.attrOfType<IntegerAttr>("ct_src.carry_id");
    if (nRes > 0 && !idAttr)
      return llvm::createStringError("ct_src.tiles requires ct_src.carry_id");
    std::string prefix =
        idAttr ? "_carry_" + std::to_string(idAttr.getInt()) + "_" : "";
    if (at == PrintPoint::Enter) {
      for (unsigned i = 0; i < nRes; ++i)
        c.line(prefix + std::to_string(i) + " = " + v.operand(1 + i).str());
    } else if (at == PrintPoint::BeginBlock) {
      auto owner = v.attrOfType<IntegerAttr>("ct_src.owner_axis");
      c.line("for " + v.blockArguments()[0] + " in range(ct.cdiv(" +
             (owner ? memberCountExpr(v.operandType(0), v.operand(0),
                                      owner.getInt())
                    : (v.operand(0) + ".shape[-1]").str()) +
             ",W)):");
      ++c.indent;
      for (unsigned i = 0; i < nRes; ++i)
        c.line(v.blockArguments()[1 + i] + " = " + prefix + std::to_string(i));
    } else if (at == PrintPoint::EndBlock) {
      --c.indent;
    } else if (at == PrintPoint::Leave) {
      for (unsigned i = 0; i < nRes; ++i)
        c.line(v.result(i).str() + " = " + prefix + std::to_string(i));
    }
    return llvm::Error::success();
  }
  if (code == "yield") {
    if (at == PrintPoint::Enter && v.numOperands()) {
      auto idAttr = v.attrOfType<IntegerAttr>("ct_src.carry_id");
      if (!idAttr)
        return llvm::createStringError("ct_src.yield requires ct_src.carry_id");
      std::string prefix = "_carry_" + std::to_string(idAttr.getInt()) + "_";
      SmallVector<std::string> lhs, rhs;
      for (unsigned i = 0; i < v.numOperands(); ++i) {
        lhs.push_back(prefix + std::to_string(i));
        rhs.push_back(v.operand(i).str());
      }
      c.line(llvm::join(lhs, ", ") + " = " + llvm::join(rhs, ", "));
    }
    return llvm::Error::success();
  }
  if (code == "bin_panels") {
    if (at == PrintPoint::BeginBlock) {
      c.line("for " + v.blockArguments()[0] + " in range(" +
             v.operand(0).str() + "):");
      ++c.indent;
    } else if (at == PrintPoint::Leave) {
      --c.indent;
    }
    return llvm::Error::success();
  }
  if (at != PrintPoint::Enter)
    return llvm::Error::success();

  auto line = [&](const std::string &s) { c.line(s); };
  auto arg = [&](unsigned n) { return v.operand(n).str(); };
  auto algebra = v.attrOfType<StringAttr>("algebra");
  bool sum = algebra && algebra == "sum";
  std::string expr;

  if (code == "bin_index") {
    auto type = getElementTypeOrSelf(v.resultType(0));
    line(v.result(0).str() + " = ct.astype(" + arg(0) +
         (type.isInteger(64) ? ",ct.int64)" : ",ct.int32)"));
    return llvm::Error::success();
  }
  if (code == "zero_bins") {
    bool swap = v.attrOfType<BoolAttr>("transposed").getValue();
    line("ct.store(" + arg(0) +
         (swap ? ",(0,ct.bid(0)),ct.full((W,1)"
               : ",(ct.bid(0),0),ct.full((1,W)") +
         ",0,ct.int32),allow_tma=False)");
    return llvm::Error::success();
  }
  if (code == "read_bin_keys") {
    auto width = std::to_string(
        cast<RankedTensorType>(v.resultType(0)).getDimSize(0));
    line(v.result(0).str() + " = ct.load(" + arg(0) + ",(" + arg(1) +
         ",),shape=(" + width +
         ",),padding_mode=ct.PaddingMode.ZERO,allow_tma=False)");
    return llvm::Error::success();
  }
  if (code == "atomic_bins") {
    bool swap = v.attrOfType<BoolAttr>("transposed").getValue();
    bool checked = v.attrOfType<BoolAttr>("checked_bounds").getValue();
    auto scope = v.attrOfType<StringAttr>("scope").getValue();
    if (scope != "workgroup" && scope != "device")
      return llvm::createStringError("invalid cuTile atomic scope " +
                                     scope.str());
    std::string memScope =
        scope == "workgroup" ? "ct.MemoryScope.BLOCK" : "ct.MemoryScope.DEVICE";
    line("ct.atomic_add(" + arg(0) +
         (swap ? ",(" + arg(1) + ",ct.bid(0))" : ",(ct.bid(0)," + arg(1) + ")") +
         "," + arg(2) + ",check_bounds=" + (checked ? "True" : "False") +
         ",memory_order=ct.MemoryOrder." + v.attrOfType<StringAttr>("order").getValue().upper() + ",memory_scope=" + memScope + ")");
    return llvm::Error::success();
  }

  if (code == "stripe_tile_load" || code == "stripe_tile_store") {
    auto s = std::to_string(v.attrOfType<IntegerAttr>("stride").getInt());
    auto idxTy = v.attrOfType<StringAttr>("index_type").str();
    std::string hint = v.attrOfType<BoolAttr>("allow_tma").getValue()
                           ? ""
                           : ", allow_tma=False";
    std::string index = "(ct.astype(" + arg(1) + ", " + idxTy + ") * (" + s +
                        " // R) + " + arg(2) + ", " + arg(3) + ")";
    if (code == "stripe_tile_store") {
      line("ct.store(" + arg(0) + ", " + index + ", " + arg(4) + hint + ")");
      return llvm::Error::success();
    }
    expr = "ct.load(" + arg(0) + ", " + index +
           ", shape=(R,W), padding_mode=ct.PaddingMode.ZERO" + hint + ")";
  } else if (code == "rect_slab_load" || code == "rect_slab_store") {
    auto base = arg(0), row = arg(1), tile = arg(2);
    int64_t owner = v.attrOfType<IntegerAttr>("owner_axis").getInt();
    int64_t extent = v.attrOfType<IntegerAttr>("inner_extent").getInt();
    auto inner = "min(W," + std::to_string(extent) + ")";
    auto segments = "(W//" + inner + ")";
    auto major =
        "(" + tile + "//(" + std::to_string(extent) + "//" + inner + "))";
    auto minor =
        "(" + tile + "%(" + std::to_string(extent) + "//" + inner + "))";
    auto shape = owner == 0 ? "(R," + segments + "," + inner + ")"
                            : "(" + segments + ",R," + inner + ")";
    auto coord = owner == 0 ? "(" + row + "," + major + "," + minor + ")"
                            : "(" + major + "," + row + "," + minor + ")";
    std::string slabHint =
        v.attrOfType<BoolAttr>("allow_tma").getValue() ? "" : ",allow_tma=False";
    if (code == "rect_slab_store") {
      auto val = arg(3);
      auto shaped =
          "ct.reshape(" + val + ",(R," + segments + "," + inner + "))";
      if (owner != 0)
        shaped = "ct.permute(" + shaped + ",(1,0,2))";
      line("ct.store(" + base + "," + coord + "," + shaped + slabHint + ")");
      return llvm::Error::success();
    }
    bool bcast = v.attrOfType<BoolAttr>("broadcast_inner").getValue();
    auto loadShape =
        owner == 0 ? "(R," + segments + "," + (bcast ? "1" : inner) + ")"
                   : "(" + segments + ",R," + (bcast ? "1" : inner) + ")";
    auto loadCoord =
        owner == 0 ? "(" + row + "," + major + "," + (bcast ? "0" : minor) + ")"
                   : "(" + major + "," + row + "," + (bcast ? "0" : minor) + ")";
    auto loaded = "ct.load(" + base + "," + loadCoord + ",shape=" + loadShape +
                  ",padding_mode=ct.PaddingMode.ZERO" + slabHint + ")";
    if (bcast)
      loaded = "ct.broadcast_to(" + loaded + "," + shape + ")";
    if (owner != 0)
      loaded = "ct.permute(" + loaded + ",(1,0,2))";
    expr = "ct.reshape(" + loaded + ",(R,W))";
  } else if (code == "gather_slab" || code == "scatter_slab") {
    auto base = arg(0), col = "ct.expand_dims(" + arg(2) + ",0)";
    int64_t owner = v.attrOfType<IntegerAttr>("owner_axis").getInt();
    int rank = cast<MemRefType>(v.operandType(0)).getRank();
    SmallVector<std::string> coords(rank);
    for (int i = rank - 1; i >= 0; --i) {
      auto size = "ct.maximum(" + base + ".shape[" + std::to_string(i) + "],1)";
      if (i == owner)
        coords[i] = "ct.expand_dims(" + arg(1) + ",1)";
      else {
        coords[i] = "(" + col + ")%" + size;
        col = "((" + col + ")//" + size + ")";
      }
    }
    auto address = base + ",(" + llvm::join(coords, ",") + ")";
    if (code == "scatter_slab") {
      line("ct.scatter(" + address + "," + arg(4) + ",mask=" + arg(3) + ")");
      return llvm::Error::success();
    }
    expr = "ct.gather(" + address + ",mask=" + arg(3) + ")";
  } else if (code == "active") {
    auto owner = v.attrOfType<IntegerAttr>("ct_src.owner_axis");
    int rank = cast<MemRefType>(v.operandType(0)).getRank();
    if (owner) {
      expr = "(ct.expand_dims(" + arg(1) + ",1)<" + arg(0) + ".shape[" +
             std::to_string(owner.getInt()) + "]) & (ct.expand_dims(" + arg(2) +
             ",0)<" +
             memberCountExpr(v.operandType(0), arg(0), owner.getInt()) + ")";
    } else if (rank == 1) {
      expr = "(ct.expand_dims(" + arg(1) + ",1)<1) & (ct.expand_dims(" +
             arg(2) + ",0)<" + arg(0) + ".shape[0])";
    } else if (rank > 2) {
      auto a = arg(0);
      std::string count = "1";
      for (int i = 0; i < rank - 1; ++i)
        count += "*" + a + ".shape[" + std::to_string(i) + "]";
      expr = "(ct.expand_dims(" + arg(1) + ",1)<" + count +
             ") & (ct.expand_dims(" + arg(2) + ",0)<" + a + ".shape[-1])";
    } else {
      expr = "(ct.expand_dims(" + arg(1) + ", 1) < " + arg(0) +
             ".shape[0]) & (ct.expand_dims(" + arg(2) + ", 0) < " + arg(0) +
             ".shape[1])";
    }
  } else if (code == "linear_index") {
    auto owner = v.attrOfType<IntegerAttr>("ct_src.owner_axis");
    bool wide = cast<RankedTensorType>(v.resultType(0))
                    .getElementType()
                    .isUnsignedInteger(64);
    std::string row = wide ? "ct.astype(" + arg(1) + ",ct.uint64)" : arg(1);
    std::string width =
        owner ? memberCountExpr(v.operandType(0), arg(0), owner.getInt())
              : arg(0) + ".shape[-1]";
    std::string column = arg(2);
    if (wide) {
      width = "ct.astype(" + width + ",ct.uint64)";
      column = "ct.astype(" + column + ",ct.uint64)";
    }
    expr = "ct.astype(ct.expand_dims(" + row + ",1)*" + width +
           "+ct.expand_dims(" + column + ",0)," +
           (wide ? "ct.uint64)" : "ct.int32)");
  } else if (code == "pid") {
    expr = "ct.astype(ct.bid(0), " +
           v.attrOfType<StringAttr>("index_type").str() + ")";
    if (v.attrOfType<BoolAttr>("column_tiles").getValue())
      expr += v.attrOfType<IntegerAttr>("axis").getInt() == 0
                  ? " // COLUMN_TILES"
                  : " % COLUMN_TILES";
  } else if (code == "div") {
    bool approx = rowHasAnyFastMath(
        v, arith::FastMathFlags::arcp);
    expr = "ct.truediv(" + arg(0) + ", " + arg(1) +
           (v.attr("ct_src.ftz") ? ", flush_to_zero=True" : "") +
           ", rounding_mode=ct.RoundingMode." + (approx ? "APPROX" : "RN") +
           ")";
  } else if (code == "splat") {
    SmallVector<std::string> dims;
    for (auto a : v.attrOfType<ArrayAttr>("dimensions"))
      dims.push_back(cast<StringAttr>(a) == "ROWS" ? "R" : "W");
    expr = "ct.broadcast_to(" + arg(0) + ",(" + llvm::join(dims, ",") + ",))";
  } else if (code == "view_offset") {
    SmallVector<std::string> coords;
    for (unsigned i = 1; i < v.numOperands(); ++i)
      coords.push_back(arg(i));
    expr = "(" + llvm::join(coords, ",") + ",)";
  } else if (code == "extract_lane") {
    auto ax = v.attrOfType<IntegerAttr>("axis").getInt();
    expr = ax == 1 ? "ct.reshape(ct.extract(" + arg(0) + ",(0,ct.astype(" +
                         arg(1) + ",ct.int32),0),shape=(R,1,W)),(R,W))"
                   : "ct.reshape(ct.extract(" + arg(0) + ",(0,0,ct.astype(" +
                         arg(1) + ",ct.int32)),shape=(R,W,1)),(R,W))";
  } else if (code == "axis_reduce") {
    auto ax = v.attrOfType<IntegerAttr>("axis").getInt();
    expr = "ct." + v.attrOfType<StringAttr>("algebra").str() + "(" + arg(0) +
           (ax < 0 ? "" : ",axis=" + std::to_string(ax)) + ")";
  } else if (code == "map") {
    auto kind = v.attrOfType<StringAttr>("kind").getValue();
    if (kind == "select")
      expr = "ct.where(" + arg(0) + "," + arg(1) + "," + arg(2) + ")";
    else if (kind == "i32" || kind == "i64" || kind == "f32")
      expr = "ct.astype(" + arg(0) + ",ct." +
             std::string(kind == "i32"   ? "int32"
                         : kind == "i64" ? "int64"
                                         : "float32") +
             ")";
    else if (kind == "ceildiv")
      expr = "ct.cdiv(" + arg(0) + "," + arg(1) + ")";
    else if (kind == "min")
      expr = "ct.minimum(" + arg(0) + "," + arg(1) + ")";
    else if (kind == "add_rn" || kind == "mul_rn")
      expr = "ct." + kind.drop_back(3).str() + "(" + arg(0) + "," + arg(1) +
             ",rounding_mode=ct.RoundingMode.RN)";
    else {
      static const std::map<std::string, std::string> tokens = {
          {"and", "&"}, {"lt", "<"},  {"ge", ">="}, {"eq", "=="},
          {"add", "+"}, {"sub", "-"}, {"mul", "*"}, {"floordiv", "//"}};
      auto it = tokens.find(kind.str());
      if (it == tokens.end())
        return llvm::createStringError("unknown row expression " + kind);
      expr = arg(0) + it->second + arg(1);
    }
  } else if (code == "uniform") {
    expr = v.attr("quad_aligned")
               ? "uniform_quad(" + arg(0) + "," + arg(1) + ",R,W)"
               : "uniform_at(" + arg(0) + "," + arg(1) + ")";
  } else if (code == "constant") {
    expr = intentdsl::kir::literal(v.attrOfType<TypedAttr>("value"));
  } else if (code == "tile_load" || code == "tile_store") {
    int rank = cast<MemRefType>(v.operandType(0)).getRank();
    bool bcast = bool(v.attr("ct_src.broadcast_inner"));
    std::string hint = v.attrOfType<BoolAttr>("allow_tma").getValue()
                           ? ""
                           : ", allow_tma=False";
    if (rank > 2) {
      auto array = arg(0), row = arg(1);
      SmallVector<std::string> indices(rank), shape(rank, "1");
      indices.back() = bcast ? "0" : arg(2);
      shape.back() = bcast ? "1" : "W";
      shape[rank - 2] = "R";
      for (int j = rank - 2; j >= 0; --j) {
        auto span = "D" + std::to_string(j);
        if (j == rank - 2)
          span = "(" + span + "//R)";
        indices[j] = "(" + row + " % " + span + ")";
        row = "(" + row + " // " + span + ")";
      }
      auto coord = "(" + llvm::join(indices, ",") + ")",
           tile = "(" + llvm::join(shape, ",") + ")";
      if (code == "tile_store")
        line("ct.store(" + array + "," + coord + ",ct.reshape(" + arg(3) + "," +
             tile + ")" + hint + ")");
      else {
        std::string loaded = "ct.load(" + array + "," + coord + ",shape=" +
                             tile + ",padding_mode=ct.PaddingMode.ZERO" + hint +
                             ")";
        line(v.result(0).str() + " = " +
             (bcast ? "ct.broadcast_to(ct.reshape(" + loaded + ",(R,1)),(R,W))"
                    : "ct.reshape(" + loaded + ",(R,W))"));
      }
      return llvm::Error::success();
    }
    bool matrix = v.attrOfType<BoolAttr>("matrix").getValue();
    std::string index =
        matrix ? "(" + arg(1) + ", " + (bcast ? "0" : arg(2)) + ")"
               : "(" + (bcast ? "0" : arg(2)) + ",)";
    if (code == "tile_store") {
      auto axes = v.attrOfType<DenseI64ArrayAttr>("block.member_axes");
      bool swap = axes && axes.size() == 2 && axes[0] == 1;
      if (swap)
        index = "(" + arg(2) + "," + arg(1) + ")";
      line("ct.store(" + arg(0) + ", " + index + ", " +
           (swap      ? "ct.transpose(" + arg(3) + ")"
            : !matrix ? "ct.reshape(" + arg(3) + ",(W,))"
                      : arg(3)) +
           hint + ")");
      return llvm::Error::success();
    }
    expr = "ct.load(" + arg(0) + ", " + index + ", shape=" +
           (matrix ? (bcast ? "(R,1)" : "(R,W)") : (bcast ? "(1,)" : "(W,)")) +
           ", padding_mode=ct.PaddingMode.ZERO" + hint + ")";
    if (!matrix)
      expr = "ct.broadcast_to(ct.expand_dims(" + expr + ",0),(R,W))";
    else if (bcast)
      expr = "ct.broadcast_to(" + expr + ",(R,W))";
  } else if (code == "reduce") {
    expr = std::string(sum ? "ct.sum" : "ct.max") + "(ct.where(" + arg(1) +
           ", " + arg(0) + ", " + (sum ? "0.0" : "-float('inf')") +
           "), axis=1)";
  } else if (code == "identity") {
    expr = std::string("ct.full((R,), ") + (sum ? "0.0" : "-float('inf')") +
           ", ct.float32)";
  } else if (code == "combine") {
    expr = sum ? arg(0) + " + " + arg(1)
               : "ct.maximum(" + arg(0) + ", " + arg(1) + ")";
  } else if (code == "cast") {
    auto t = getElementTypeOrSelf(v.resultType(0));
    if (!t.isF32() && !t.isF16() && !t.isBF16() && !t.isInteger(32))
      return llvm::createStringError("unsupported cuTile conversion");
    expr = "ct.astype(" + arg(0) + ", ct." +
           std::string(t.isInteger(32) ? "int32"
                       : t.isBF16()    ? "bfloat16"
                       : t.isF16()     ? "float16"
                                       : "float32") +
           ")";
  } else {
    return llvm::createStringError("unexpected local cuTile operation " + code);
  }

  c.line(v.result(0).str() + " = " + expr);
  return llvm::Error::success();
}
} // namespace

void rowRules(TerminalPrinter &printer) {
  auto &ops = printer.operations;
  // Declarative templates
  ops["ct_src.output_partition"] =
      PrintRule::format("if ct.bid(1)=={index}:", "", true);
  ops["ct_src.partition_end"] = PrintRule::format("");
  ops["ct_src.unique_put_at"] =
      PrintRule::format("ct.scatter($0,$1,$2,mask=$3)");
  ops["ct_src.index_read_at"] =
      PrintRule::format("$r0 = ct.gather($0,$1,mask=$2)");
  ops["ct_src.atomic_add_at"] = PrintRule::local(
      [](const LocalOpView &v, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter)
          c.line(v.result(0) + " = ct.atomic_add(" + v.operand(0) +
                 ",ct.where(" + v.operand(3) + "," + v.operand(1) + ",-1)," +
                 v.operand(2) + ",check_bounds=True,memory_order=ct.MemoryOrder." +
                 v.attrOfType<StringAttr>("ordering").getValue().upper() + ")");
        return llvm::Error::success();
      });
  ops["ct_src.ne"] = PrintRule::format("$r0 = $0!=$1");
  ops["ct_src.member_tile"] = PrintRule::format(
      "$r0 = ct.astype(ct.broadcast_to(ct.expand_dims($0,0),(R,W)),ct.int32)");
  ops["ct_src.array_dim"] = PrintRule::format("$r0 = $0.shape[{axis}]");
  ops["ct_src.private_bin_store"] = PrintRule::format("if $5:", "", true);
  ops["ct_src.range"] = PrintRule::format("$r0 = ct.arange({size},dtype=ct.int64)");
  ops["ct_src.expand"] = PrintRule::format("$r0 = ct.expand_dims($0,{axis})");
  ops["ct_src.read_at"] = PrintRule::format(
      "$r0 = ct.gather($0,$1,mask=$2,padding_value={fill})");
  ops["ct_src.write_at"] = PrintRule::format("ct.scatter($0,$1,$2,mask=$3)");
  ops["ct_src.owner_load"] = PrintRule::format("$r0 = ct.gather($0,$1)");
  ops["ct_src.owner_store"] = PrintRule::format("ct.scatter($0,$1,$2)");
  ops["ct_src.summary_reduce"] = PrintRule::format(
      "$r0 = ct.reduce((ct.astype($1,ct.int32),ct.where($1,$0,0.),ct.full((R,W),0.,ct.float32)),1,merge_moments,(0,0.,0.))");
  ops["ct_src.summary_identity"] = PrintRule::format(
      "$r0 = (ct.full((R,),0,ct.int32),ct.full((R,),0.,ct.float32),ct.full((R,),0.,ct.float32))");
  ops["ct_src.summary_merge"] = PrintRule::format(
      "$r0 = merge_moments($0[0],$0[1],$0[2],$1[0],$1[1],$1[2])");
  ops["ct_src.summary_field"] = PrintRule::format("$r0 = $0[{position}]");
  ops["ct_src.scan_sum"] = PrintRule::format("$r0 = ct.cumsum($0,axis=1)");
  ops["ct_src.select"] = PrintRule::format("$r0 = ct.where($0,$1,$2)");
  ops["ct_src.xor"] = PrintRule::format("$r0 = $0 ^ $1");
  ops["ct_src.and"] = PrintRule::format("$r0 = $0 & $1");
  ops["ct_src.shl"] = PrintRule::format("$r0 = $0 << $1");
  ops["ct_src.shr"] = PrintRule::format("$r0 = $0 >> $1");
  ops["ct_src.gt"] = PrintRule::format("$r0 = $0 > $1");
  ops["ct_src.maximum"] =
      PrintRule::format("$r0 = ct.maximum($0,$1,propagate_nan=True)");
  ops["ct_src.zero"] = PrintRule::format("$r0 = 0");
  ops["ct_src.broadcast_rows"] = PrintRule::format("$r0 = ct.expand_dims($0,1)");
  ops["ct_src.mul"] = PrintRule::format("$r0 = $0 * $1");
  ops["ct_src.sub"] = PrintRule::format("$r0 = $0 - $1");
  ops["ct_src.add"] = PrintRule::format("$r0 = $0 + $1");
  ops["ct_src.sqrt"] = PrintRule::format("$r0 = ct.sqrt($0)");
  ops["ct_src.exp"] = PrintRule::format("$r0 = ct.exp($0)");
  ops["ct_src.rsqrt"] = PrintRule::format("$r0 = ct.rsqrt($0)");
  ops["ct_src.rows"] =
      PrintRule::format("$r0 = $0 * R + ct.arange(R, dtype={index_type})");
  ops["ct_src.columns"] = PrintRule::format(
      "$r0 = ct.astype($0, {index_type}) * W + ct.arange(W, dtype={index_type})");
  ops["ct_src.stripe_owner_load"] = PrintRule::format(
      "$r0 = ct.broadcast_to(ct.expand_dims(ct.gather($0,ct.astype($1,{index_type})*{stride}+$2),1),(R,W))");

  // Local rules (LocalOpView only)
  for (auto op : {"tiles", "yield", "stripe_tile_load", "stripe_tile_store",
                  "rect_slab_load", "rect_slab_store", "gather_slab",
                  "scatter_slab", "active", "linear_index", "pid", "div",
                  "bin_index", "zero_bins", "bin_panels", "read_bin_keys",
                  "atomic_bins", "splat", "view_offset", "extract_lane",
                  "axis_reduce", "map", "uniform", "constant", "tile_load",
                  "tile_store", "reduce", "identity", "combine", "cast"})
    ops["ct_src." + std::string(op)] = PrintRule::local(localRowRule);
}
} // namespace CI_UNIT_NAMESPACE
