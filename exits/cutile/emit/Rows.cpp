#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Grid.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringExtras.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
namespace {
llvm::Error rowRule(Operation &op, PrintPoint at, PrintContext &c) {
  auto code = op.getName().stripDialect();
  bool scope = code == "tiles" || code == "output_partition" ||
               code == "private_bin_store" || code == "bin_panels";
  if (at != PrintPoint::Enter) {
    if (at == PrintPoint::Leave && scope)
      --c.indent;
    return llvm::Error::success();
  }
  auto line = [&](const std::string &s) { c.line(s); };
  auto arg = [&](unsigned n) { return c.name(op.getOperand(n)); };
  auto result = [&]() { return c.name(op.getResult(0)); };
  auto fn = op.getParentOfType<func::FuncOp>();
  std::string indexType =
      fn->getAttrOfType<IntegerAttr>("ct_src.index_bits").getInt() == 64
          ? "ct.int64"
          : "ct.int32";
  auto algebra = op.getAttrOfType<StringAttr>("algebra");
  auto owner = fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
  auto memberCount = [&](Value view, const std::string &name) {
    std::string product = "1";
    for (int i = 0; i < cast<MemRefType>(view.getType()).getRank(); ++i)
      if (i != owner.getInt())
        product += "*" + name + ".shape[" + std::to_string(i) + "]";
    return product;
  };
  bool sum = algebra && algebra == "sum";
  if (code == "tiles") {
    auto &bb = op.getRegion(0).front();
    if (op.getNumResults())
      line(result() + " = " + arg(1));
    line("for " + c.name(bb.getArgument(0)) + " in range(ct.cdiv(" +
         (owner ? memberCount(op.getOperand(0), arg(0))
                : arg(0) + ".shape[-1]") +
         ",W)):");
    ++c.indent;
    if (op.getNumResults())
      line(c.name(bb.getArgument(1)) + " = " + result());
    return llvm::Error::success();
  }
  if (code == "yield") {
    if (op.getNumOperands())
      line(c.name(op.getParentOp()->getResult(0)) + " = " + arg(0));
    return llvm::Error::success();
  }
  if (code == "output_partition") {
    line("if ct.bid(1)==" +
         std::to_string(op.getAttrOfType<IntegerAttr>("index").getInt()) + ":");
    ++c.indent;
    return llvm::Error::success();
  }
  if (code == "partition_end")
    return llvm::Error::success();
  std::string expr;
  if (code == "index_read_at" || code == "atomic_add_at" ||
      code == "unique_put_at") {
    if (code == "unique_put_at") {
      line("ct.scatter(" + arg(0) + "," + arg(1) + "," + arg(2) +
           ",mask=" + arg(3) + ")");
      return llvm::Error::success();
    }
    auto id = result();
    line(id + " = " +
         (code == "index_read_at"
              ? "ct.gather(" + arg(0) + "," + arg(1) + ",mask=" + arg(2) + ")"
              : "ct.atomic_add(" + arg(0) + ",ct.where(" + arg(3) + "," +
                    arg(1) + ",0),ct.where(" + arg(3) + "," + arg(2) +
                    ",0),memory_order=ct.MemoryOrder.RELAXED)"));
    return llvm::Error::success();
  }
  if (code == "ne" || code == "member_tile") {
    auto id = result();
    line(id + " = " +
         (code == "ne" ? arg(0) + "!=" + arg(1)
                       : "ct.astype(ct.broadcast_to(ct.expand_dims(" + arg(0) +
                             ",0),(R,W)),ct.int32)"));
    return llvm::Error::success();
  }
  if(code=="bin_index"){
    auto type=getElementTypeOrSelf(op.getResult(0).getType());
    line(result()+" = ct.astype("+arg(0)+(type.isInteger(64)?",ct.int64)":",ct.int32)"));
    return llvm::Error::success();
  }
  if (code == "private_bin_store" || code == "zero_bins" ||
      code == "bin_panels" || code == "read_bin_keys" ||
      code == "atomic_bins") {
    auto *guard = &op;
    while (guard->getName().stripDialect() != "private_bin_store")
      guard = guard->getParentOp();
    auto dest = c.name(guard->getOperand(1)),
         src = c.name(guard->getOperand(0));
    auto extent =
        std::to_string(guard->getAttrOfType<IntegerAttr>("extent").getInt());
    auto span =
        std::to_string(guard->getAttrOfType<IntegerAttr>("span").getInt());
    auto width = std::to_string(
        guard->getAttrOfType<IntegerAttr>("source_step").getInt());
    bool swap = guard->getAttrOfType<BoolAttr>("transposed").getValue();
    auto bins = dest + ".shape[" + std::to_string(swap ? 0 : 1) + "]";
    if (code == "private_bin_store") {
      line("if " + bins + " > 0:");
      ++c.indent;
    } else if (code == "zero_bins") {
      line("ct.store(" + dest +
           (swap ? ",(0,ct.bid(0)),ct.full((W,1)"
                 : ",(ct.bid(0),0),ct.full((1,W)") +
           ",0,ct.int32),allow_tma=False)");
    } else if (code == "bin_panels") {
      auto step = c.name(op.getRegion(0).front().getArgument(0)),
           limit = span + "//" + width;
      if (guard->getAttrOfType<IntegerAttr>("extent").getInt() %
          guard->getAttrOfType<IntegerAttr>("span").getInt())
        limit = "ct.astype(ct.minimum(" + limit + ",ct.cdiv(" + extent +
                "-ct.astype(ct.bid(0),ct.int64)*" + span + "," + width +
                ")),ct.int32)";

      line("for " + step + " in range(" + limit + "):");
      ++c.indent;
    } else if (code == "read_bin_keys") {
      auto keys = result();
      line(keys + "=ct.load(" + src + ",(ct.astype(ct.bid(0),ct.int64)*(" +
           span + "//" + width + ")+" + arg(0) + ",),shape=(" + width +
           ",),padding_mode=ct.PaddingMode.ZERO,allow_tma=False)");
    } else {
      auto pos = "ct.astype(ct.bid(0),ct.int64)*" + span + "+ct.astype(" +
                 arg(0) + ",ct.int64)*" + width + "+ct.arange(" + width +
                 ",dtype=ct.int32)";
      line("valid=(" + pos + "<" + extent + ")&(" + arg(1) + ">=0)&(" + arg(1) +
           "<" + bins + ")");
      auto key = "ct.where(valid,"+arg(1)+",0)";
      line("ct.atomic_add(" + dest +
           (swap ? ",(" + key + ",ct.bid(0))" : ",(ct.bid(0)," + key + ")") +
           ",ct.astype(valid,ct.int32),check_bounds=False,memory_order=ct."
           "MemoryOrder.RELAXED,memory_scope=ct.MemoryScope.BLOCK)");
    }
    return llvm::Error::success();
  }
  if (code == "dimension") {
    auto base = fn.getArgument(intentdsl::domainSlot(fn));
    auto n = c.name(base);
    int rank = cast<MemRefType>(base.getType()).getRank();
    bool m = op.getAttrOfType<StringAttr>("name") == "M";
    if (owner)
      expr = m ? n + ".shape[" + std::to_string(owner.getInt()) + "]"
               : memberCount(base, n);
    else if (!m)
      expr = n + ".shape[-1]";
    else {
      expr = "1";
      for (int i = 0; i < rank - 1; ++i)
        expr += "*" + n + ".shape[" + std::to_string(i) + "]";
    }
  } else if (code == "range")
    expr = "ct.arange(" +
           std::to_string(op.getAttrOfType<IntegerAttr>("size").getInt()) +
           ",dtype=ct.int64)";
  else if (code == "expand")
    expr = "ct.expand_dims(" + arg(0) + "," +
           std::to_string(op.getAttrOfType<IntegerAttr>("axis").getInt()) + ")";
  else if (code == "splat") {
    SmallVector<std::string> dims;
    for (auto a : op.getAttrOfType<ArrayAttr>("dimensions"))
      dims.push_back(cast<StringAttr>(a) == "ROWS" ? "R" : "W");
    expr = "ct.broadcast_to(" + arg(0) + ",(" + llvm::join(dims, ",") + ",))";
  } else if (code == "view_offset") {
    SmallVector<std::string> coords;
    for (unsigned i = 1; i < op.getNumOperands(); ++i)
      coords.push_back(arg(i));
    expr = "(" + llvm::join(coords, ",") + ",)";
  } else if (code == "read_at")
    expr = "ct.gather(" + arg(0) + "," + arg(1) + ",mask=" + arg(2) +
           ",padding_value=" +
           std::to_string(op.getAttrOfType<IntegerAttr>("fill").getInt()) + ")";
  else if (code == "write_at") {
    line("ct.scatter(" + arg(0) + "," + arg(1) + "," + arg(2) +
         ",mask=" + arg(3) + ")");
    return llvm::Error::success();
  } else if (code == "extract_lane")
    expr = "ct.reshape(ct.extract(" + arg(0) + ",(0,0,ct.astype(" + arg(1) +
           ",ct.int32)),shape=(R,W,1)),(R,W))";
  else if (code == "axis_reduce") {
    auto ax = op.getAttrOfType<IntegerAttr>("axis").getInt();
    expr = "ct." + op.getAttrOfType<StringAttr>("algebra").str() + "(" +
           arg(0) + (ax < 0 ? "" : ",axis=" + std::to_string(ax)) + ")";
  } else if (code == "map") {
    auto kind = op.getAttrOfType<StringAttr>("kind").getValue();
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
  } else if (code == "owner_load")
    expr = "ct.gather(" + arg(0) + "," + arg(1) + ")";
  else if (code == "owner_store") {
    line("ct.scatter(" + arg(0) + "," + arg(1) + "," + arg(2) + ")");
    return llvm::Error::success();
  } else if (code == "gather_slab" || code == "scatter_slab") {
    auto base = arg(0), col = "ct.expand_dims(" + arg(2) + ",0)";
    if (op.hasAttr("ct_src.rectangular")) {
      auto fn = op.getParentOfType<func::FuncOp>();
      auto slot = cast<BlockArgument>(op.getOperand(0)).getArgNumber();
      int64_t extent =
          fn.getArgAttrOfType<DenseI64ArrayAttr>(slot, "ct_src.shape")[2];
      auto row = c.name(op.getOperand(1).getDefiningOp()->getOperand(0));
      auto tile = c.name(op.getOperand(2).getDefiningOp()->getOperand(0));
      auto inner = "min(W," + std::to_string(extent) + ")";
      auto segments = "(W//" + inner + ")";
      auto major =
          "(" + tile + "//(" + std::to_string(extent) + "//" + inner + "))";
      auto minor =
          "(" + tile + "%(" + std::to_string(extent) + "//" + inner + "))";
      auto shape = owner.getInt() == 0 ? "(R," + segments + "," + inner + ")"
                                       : "(" + segments + ",R," + inner + ")";
      auto coord = owner.getInt() == 0
                       ? "(" + row + "," + major + "," + minor + ")"
                       : "(" + major + "," + row + "," + minor + ")";
      if (code == "scatter_slab") {
        auto v = arg(4);
        auto shaped =
            "ct.reshape(" + v + ",(R," + segments + "," + inner + "))";
        if (owner.getInt() != 0)
          shaped = "ct.permute(" + shaped + ",(1,0,2))";
        line("ct.store(" + base + "," + coord + "," + shaped +
             ",allow_tma=False)");
      } else {
        auto v = "ct.load(" + base + "," + coord + ",shape=" + shape +
                 ",padding_mode=ct.PaddingMode.ZERO,allow_tma=False)";
        if (owner.getInt() != 0)
          v = "ct.permute(" + v + ",(1,0,2))";
        auto id = result();
        line(id + " = ct.reshape(" + v + ",(R,W))");
      }
      return llvm::Error::success();
    }
    int rank = cast<MemRefType>(op.getOperand(0).getType()).getRank();
    SmallVector<std::string> coords(rank);
    for (int i = rank - 1; i >= 0; --i) {
      auto size = "ct.maximum(" + base + ".shape[" + std::to_string(i) + "],1)";
      if (i == owner.getInt())
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
  } else if (code == "active" && owner) {
    expr = "(ct.expand_dims(" + arg(1) + ",1)<" + arg(0) + ".shape[" +
           std::to_string(owner.getInt()) + "]) & (ct.expand_dims(" + arg(2) +
           ",0)<" + memberCount(op.getOperand(0), arg(0)) + ")";
  } else if (code == "summary_reduce")
    expr = "ct.reduce((ct.astype(" + arg(1) + ",ct.int32),ct.where(" + arg(1) +
           "," + arg(0) +
           ",0.),ct.full((R,W),0.,ct.float32)),1,merge_moments,(0,0.,0.))";
  else if (code == "summary_identity")
    expr = "(ct.full((R,),0,ct.int32),ct.full((R,),0.,ct.float32),ct.full((R,),"
           "0.,ct.float32))";
  else if (code == "summary_merge")
    expr = "merge_moments(" + arg(0) + "[0]," + arg(0) + "[1]," + arg(0) +
           "[2]," + arg(1) + "[0]," + arg(1) + "[1]," + arg(1) + "[2])";
  else if (code == "summary_field")
    expr = arg(0) + "[" +
           std::to_string(op.getAttrOfType<IntegerAttr>("position").getInt()) +
           "]";

  else if (code == "scan_sum")
    expr = "ct.cumsum(" + arg(0) + ",axis=1)";
  else if (code == "uniform")
    expr = "uniform_at(" + arg(0) + "," + arg(1) + ")";
  else if (code == "linear_index") {
    bool wide = cast<RankedTensorType>(op.getResult(0).getType())
                    .getElementType()
                    .isUnsignedInteger(64);
    std::string row = wide ? "ct.astype(" + arg(1) + ",ct.uint64)" : arg(1);
    auto width = owner ? memberCount(op.getOperand(0), arg(0))
                       : arg(0) + ".shape[-1]",
         column = arg(2);
    if (wide) {
      width = "ct.astype(" + width + ",ct.uint64)";
      column = "ct.astype(" + column + ",ct.uint64)";
    }
    expr = "ct.astype(ct.expand_dims(" + row + ",1)*" + width +
           "+ct.expand_dims(" + column + ",0)," +
           (wide ? "ct.uint64)" : "ct.int32)");
  } else if (code == "select")
    expr = "ct.where(" + arg(0) + "," + arg(1) + "," + arg(2) + ")";
  else if (code == "xor" || code == "and" || code == "shr" || code == "shl" ||
           code == "gt") {
    auto token = code == "xor"   ? " ^ "
                 : code == "and" ? " & "
                 : code == "shl" ? " << "
                 : code == "shr" ? " >> "
                                 : " > ";
    expr = arg(0) + token + arg(1);
  } else if (code == "constant")
    expr = intentdsl::kir::literal(op.getAttrOfType<TypedAttr>("value"));
  else if (code == "maximum")
    expr = "ct.maximum(" + arg(0) + "," + arg(1) + ",propagate_nan=True)";
  else if (code == "pid") {
    expr = "ct.astype(ct.bid(0), " + indexType + ")";
    auto fn = op.getParentOfType<func::FuncOp>();
    Builder b(fn.getContext());
    if (fn->getAttr("block.grid") == intentdsl::block::grid(b, true)) {
      expr += op.getAttrOfType<IntegerAttr>("axis").getInt() == 0
                  ? " // COLUMN_TILES"
                  : " % COLUMN_TILES";
    }
  } else if (code == "zero")
    expr = "0";
  else if (code == "rows")
    expr = arg(0) + " * R + ct.arange(R, dtype=" + indexType + ")";
  else if (code == "columns")
    expr = "ct.astype(" + arg(0) + ", " + indexType +
           ") * W + ct.arange(W, dtype=" + indexType + ")";
  else if (code == "active" &&
           cast<MemRefType>(op.getOperand(0).getType()).getRank() == 1)
    expr = "(ct.expand_dims(" + arg(1) + ",1)<1) & (ct.expand_dims(" + arg(2) +
           ",0)<" + arg(0) + ".shape[0])";
  else if (code == "active" &&
           cast<MemRefType>(op.getOperand(0).getType()).getRank() > 2) {
    auto a = arg(0);
    auto rank = cast<MemRefType>(op.getOperand(0).getType()).getRank();
    std::string count = "1";
    for (int i = 0; i < rank - 1; ++i)
      count += "*" + a + ".shape[" + std::to_string(i) + "]";
    expr = "(ct.expand_dims(" + arg(1) + ",1)<" + count +
           ") & (ct.expand_dims(" + arg(2) + ",0)<" + a + ".shape[-1])";
  } else if (code == "active")
    expr = "(ct.expand_dims(" + arg(1) + ", 1) < " + arg(0) +
           ".shape[0]) & (ct.expand_dims(" + arg(2) + ", 0) < " + arg(0) +
           ".shape[1])";
  else if (code == "tile_load" || code == "tile_store") {
    int rank = cast<MemRefType>(op.getOperand(0).getType()).getRank();
    std::string hint = op.getAttrOfType<BoolAttr>("allow_tma").getValue()
                           ? ""
                           : ", allow_tma=False";
    if (rank > 2) {
      auto array = arg(0), row = arg(1);
      SmallVector<std::string> indices(rank), shape(rank, "1");
      indices.back() = arg(2);
      shape.back() = "W";
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
        auto id = result();
        line(id + " = ct.reshape(ct.load(" + array + "," + coord + ",shape=" +
             tile + ",padding_mode=ct.PaddingMode.ZERO" + hint + "),(R,W))");
      }
      return llvm::Error::success();
    }
    bool matrix = op.getAttrOfType<BoolAttr>("matrix").getValue();
    std::string index =
        matrix ? "(" + arg(1) + ", " + arg(2) + ")" : "(" + arg(2) + ",)";
    if (code == "tile_store") {
      auto axes = op.getAttrOfType<DenseI64ArrayAttr>("block.member_axes");
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
    expr = "ct.load(" + arg(0) + ", " + index +
           ", shape=" + (matrix ? "(R,W)" : "(W,)") +
           ", padding_mode=ct.PaddingMode.ZERO" + hint + ")";
    if (!matrix)
      expr = "ct.broadcast_to(ct.expand_dims(" + expr + ",0),(R,W))";
  } else if (code == "reduce")
    expr = std::string(sum ? "ct.sum" : "ct.max") + "(ct.where(" + arg(1) +
           ", " + arg(0) + ", " + (sum ? "0.0" : "-float('inf')") +
           "), axis=1)";
  else if (code == "identity")
    expr = std::string("ct.full((R,), ") + (sum ? "0.0" : "-float('inf')") +
           ", ct.float32)";
  else if (code == "combine")
    expr = sum ? arg(0) + " + " + arg(1)
               : "ct.maximum(" + arg(0) + ", " + arg(1) + ")";
  else if (code == "broadcast_rows")
    expr = "ct.expand_dims(" + arg(0) + ",1)";
  else if (code == "mul" || code == "sub" || code == "add")
    expr = arg(0) +
           (code == "mul"   ? " * "
            : code == "sub" ? " - "
                            : " + ") +
           arg(1);
  else if (code == "div")
    expr = "ct.truediv(" + arg(0) + ", " + arg(1) +
           ", rounding_mode=ct.RoundingMode.RN)";
  else if (code == "sqrt" || code == "exp" || code == "rsqrt")
    expr = "ct." + code.str() + "(" + arg(0) + ")";
  else if (code == "cast") {
    auto t = op.getResult(0).getType();
    if (auto f = dyn_cast<ShapedType>(t))
      t = f.getElementType();
    if (!t.isF32() && !t.isF16() && !t.isBF16() && !t.isInteger(32))
      return llvm::createStringError("unsupported cuTile conversion");
    expr = "ct.astype(" + arg(0) + ", ct." +
           std::string(t.isInteger(32) ? "int32"
                       : t.isBF16()    ? "bfloat16"
                       : t.isF16()     ? "float16"
                                       : "float32") +
           ")";
  } else
    return llvm::createStringError("unexpected cuTile operation " + code);

  c.line(result() + " = " + expr);
  return llvm::Error::success();
}
} // namespace
void rowRules(TerminalPrinter &printer) {
  for (auto op : {"index_read_at",
                  "atomic_add_at",
                  "unique_put_at",
                  "ne",
                  "member_tile",
                  "private_bin_store",
                  "zero_bins",
                  "bin_panels",
                  "read_bin_keys",
                  "atomic_bins",
                  "bin_index",
                  "dimension",
                  "range",
                  "expand",
                  "splat",
                  "view_offset",
                  "read_at",
                  "write_at",
                  "extract_lane",
                  "axis_reduce",
                  "map",
                  "owner_load",
                  "owner_store",
                  "gather_slab",
                  "scatter_slab",
                  "active",
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
                  "zero",
                  "rows",
                  "columns",
                  "tile_load",
                  "tile_store",
                  "reduce",
                  "identity",
                  "combine",
                  "broadcast_rows",
                  "mul",
                  "sub",
                  "add",
                  "div",
                  "sqrt",
                  "exp",
                  "rsqrt",
                  "cast",
                  "tiles",
                  "yield",
                  "output_partition",
                  "partition_end"})
    printer.operations["ct_src." + std::string(op)] = rowRule;
}
} // namespace CI_UNIT_NAMESPACE
