#include "compilerinfra/Terminal.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "intentdsl/levels/algorithm/kir/ir/StorageLayout.h"
#include "intentdsl/levels/execution/block/ir/Grid.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringExtras.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
namespace {
std::string dtype(Type type) {
  type = getElementTypeOrSelf(type);
  if (type.isF32())
    return "ct.float32";
  if (type.isF16())
    return "ct.float16";
  if (type.isBF16())
    return "ct.bfloat16";
  if (isa<Float8E4M3FNType>(type))
    return "ct.float8_e4m3fn";
  if (type.isIndex() || type.isInteger(64))
    return "ct.int64";
  if (type.isInteger(1))
    return "ct.bool_";
  return "ct.int" + std::to_string(type.getIntOrFloatBitWidth());
}
std::string dims(Type type) {
  SmallVector<std::string> size;
  if (auto t = dyn_cast<RankedTensorType>(type))
    for (auto n : t.getShape())
      size.push_back(std::to_string(n));
  return "(" + llvm::join(size, ",") + (size.size() == 1 ? "," : "") + ")";
}
std::string axis(Type type) {
  return std::to_string(cast<RankedTensorType>(type).getDimSize(0));
}
std::string expand(std::string x, int axis) {
  return "ct.expand_dims(" + x + "," + std::to_string(axis) + ")";
}
} // namespace
void rowRules(TerminalPrinter &);
TerminalPrinter cutilePrinter() {
  TerminalPrinter p;
  auto rule = [&](StringRef name, auto format) {
    p.operations[name.str()] = [format](Operation &op, PrintPoint at,
                                        PrintContext &c) {
      if (at != PrintPoint::Enter)
        return llvm::Error::success();
      SmallVector<std::string> v;
      for (auto operand : op.getOperands())
        v.push_back(c.name(operand));
      std::string expr = format(op, v);
      if (!expr.empty())
        c.line(op.getNumResults() ? c.name(op.getResult(0)) + " = " + expr
                                  : expr);
      return llvm::Error::success();
    };
  };
  for (auto n :
       {"func.return", "ct_src.tile_program", "ct_src.end"})
    rule(n, [](auto &, auto &) { return ""; });
  p.operations["builtin.module"] = [](Operation &, PrintPoint at,
                                      PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.file = "kernel.py";
      c.line("import cuda.tile as ct");
    }
    return llvm::Error::success();
  };
  p.operations["ct_src.runtime_source"] =
      [](Operation &op, PrintPoint at, PrintContext &c) -> llvm::Error {
    if (at == PrintPoint::Enter) {
      auto impl = op.getAttrOfType<StringAttr>("implementation");
      if (!impl)
        return llvm::createStringError("missing runtime_source implementation");
      auto helper = c.source(impl.getValue());
      if (!helper)
        return helper.takeError();
      c.line(*helper);
    }
    return llvm::Error::success();
  };
  p.operations["func.func"] = [](Operation &op, PrintPoint at,
                                 PrintContext &c) -> llvm::Error {
    if (at == PrintPoint::Enter) {
      c.line("@ct.kernel");
      auto fn = cast<func::FuncOp>(op);
      SmallVector<std::string> args;
      for (auto a : fn.getArguments()) {
        auto n = c.name(a);
        if (isa<MemRefType>(a.getType()) &&
            op.getAttrOfType<IntegerAttr>("ct_src.index_bits").getInt() == 64)
          n += ": ct.IndexedWithInt64[ct.Array]";
        args.push_back(n);
      }
      auto convention =
          fn.front().front().getAttrOfType<StringAttr>("calling_convention");
      bool matrix = convention && convention.getValue() == "matrix";
      bool direct = convention || intentdsl::kir::orderedOwners(fn);
      if (!direct) {
        int rank = cast<MemRefType>(
                       fn.getArgument(intentdsl::domainSlot(fn)).getType())
                       .getRank();
        if (rank > 2)
          for (int i = 0; i < rank - 1; i++)
            args.push_back("D" + std::to_string(i) + ":ct.Constant[int]");
        Builder b(fn.getContext());
        if (fn->getAttr("block.grid") == intentdsl::block::grid(b, true))
          args.push_back("COLUMN_TILES:ct.Constant[int]");
      }
      args.push_back(matrix   ? "BM:ct.Constant[int]"
                     : direct ? "ROWS:ct.Constant[int]"
                              : "R:ct.Constant[int]");
      args.push_back(matrix   ? "BN:ct.Constant[int]"
                     : direct ? "WIDTH:ct.Constant[int]"
                              : "W:ct.Constant[int]");
      c.line("def kernel(" + llvm::join(args, ",") + "):");
      ++c.indent;
    } else if (at == PrintPoint::Leave)
      --c.indent;
    return llvm::Error::success();
  };
  p.operations["ct_src.owner_space"] = [](Operation &op, PrintPoint at,
                                          PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line(c.name(op.getRegion(0).front().getArgument(0)) +
             " = ct.astype(ct.bid(0),ct.int64)");
    return llvm::Error::success();
  };
  p.operations["scf.for"] = [](Operation &op, PrintPoint at, PrintContext &c) {
    auto loop = cast<scf::ForOp>(op);
    auto narrow = [](Value v) {
      auto constant = v.getDefiningOp<arith::ConstantOp>();
      if (!constant)
        return false;
      auto i = dyn_cast<IntegerAttr>(constant.getValue());
      return i && i.getValue().isSignedIntN(32);
    };
    bool native = loop.getInductionVar().getType().isInteger(32);
    bool counted = narrow(loop.getLowerBound()) &&
                   narrow(loop.getUpperBound()) && narrow(loop.getStep());
    auto index = c.name(loop.getInductionVar());
    if (at == PrintPoint::Enter) {
      for (auto [dst, src] : llvm::zip(loop.getResults(), loop.getInitArgs()))
        c.line(c.name(dst) + " = " + c.name(src));
      if (native) {
        auto i32 = [&](Value v) -> std::string {
          if (auto cst = v.getDefiningOp<arith::ConstantOp>())
            if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
              return std::to_string(ia.getInt());
          return "ct.astype(" + c.name(v) + ",ct.int32)";
        };
        c.line("for " + index + " in range(" + i32(loop.getLowerBound()) + "," +
               i32(loop.getUpperBound()) + "," + i32(loop.getStep()) + "):");
      } else if (counted) {
        auto bound = [&](Value v) {
          return std::to_string(
              cast<IntegerAttr>(v.getDefiningOp<arith::ConstantOp>().getValue())
                  .getInt());
        };
        c.line("for it_" + index + " in range(" + bound(loop.getLowerBound()) +
               "," + bound(loop.getUpperBound()) + "," + bound(loop.getStep()) +
               "):");
      } else {
        c.line(index + " = ct.astype(" + c.name(loop.getLowerBound()) +
               ",ct.int64)");
        c.line("while " + index + " < " + c.name(loop.getUpperBound()) + ":");
      }
      ++c.indent;
      if (!native && counted)
        c.line(index + " = ct.astype(it_" + index + ",ct.int64)");
      for (auto [arg, res] :
           llvm::zip(loop.getRegionIterArgs(), loop.getResults()))
        c.line(c.name(arg) + " = " + c.name(res));
    } else if (at == PrintPoint::Leave) {
      if (!native && !counted)
        c.line(index + " += " + c.name(loop.getStep()));
      --c.indent;
    }
    return llvm::Error::success();
  };
  p.operations["scf.if"] = [](Operation &op, PrintPoint at, PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("if " + c.name(op.getOperand(0)) + ":");
    if (at == PrintPoint::BeginRegion && !op.getRegion(c.region).empty()) {
      if (c.region)
        c.line("else:");
      ++c.indent;
    } else if (at == PrintPoint::EndRegion && !op.getRegion(c.region).empty())
      --c.indent;
    return llvm::Error::success();
  };
  p.operations["scf.yield"] = [](Operation &op, PrintPoint at,
                                 PrintContext &c) {
    if (at == PrintPoint::Enter && op.getNumOperands()) {
      SmallVector<std::string> lhs, rhs;
      for (auto [r, v] :
           llvm::zip(op.getParentOp()->getResults(), op.getOperands())) {
        lhs.push_back(c.name(r));
        rhs.push_back(c.name(v));
      }
      c.line(llvm::join(lhs, ",") + " = " + llvm::join(rhs, ","));
    }
    return llvm::Error::success();
  };
  rule("ct_src.from_le_half", [](Operation &, auto &v) {
    return "ct.astype(ct.bitcast(ct.astype(" + v[0] + "|(" + v[1] +
           "<<8),ct.uint16),ct.float16),ct.float32)";
  });
  rule("ct_src.e8m0", [](Operation &, auto &v) {
    using F = intentdsl::kir::MX8Block;
    return "ct.bitcast(ct.where(" + v[0] + "==" + std::to_string(F::nan) +
           ",2143289344,ct.where(" + v[0] + "==0,4194304," + v[0] +
           "<<23)),ct.float32)";
  });
  rule("ct_src.expand_scale", [](Operation &o, auto &v) {
    return "ct.broadcast_to(" +
           expand(v[0], o.getAttrOfType<IntegerAttr>("axis").getInt()) + "," +
           dims(o.getResult(0).getType()) + ")";
  });
  rule("ct_src.reshape", [](Operation &o, auto &v) {
    return "ct.reshape(" + v[0] + "," + dims(o.getResult(0).getType()) + ")";
  });
  rule("ct_src.transpose",
       [](Operation &, auto &v) { return "ct.transpose(" + v[0] + ")"; });
  rule("ct_src.rect_read", [](Operation &o, auto &v) {
    auto coords = llvm::join(llvm::drop_begin(v), ",");
    return "ct.load(" + v[0] + ",(" + coords +
           ",),shape=" + dims(o.getResult(0).getType()) +
           ",padding_mode=ct.PaddingMode.ZERO,allow_tma=" +
           (o.getAttrOfType<BoolAttr>("allow_tma").getValue() ? "True"
                                                              : "False") +
           ")";
  });
  rule("ct_src.rect_write", [](Operation &, auto &v) {
    return "ct.store(" + v[0] + ",(" + llvm::join(llvm::drop_begin(v, 2), ",") +
           ",)," + v[1] + ",allow_tma=False)";
  });
  rule("ct_src.scaled_mma", [](Operation &, auto &v) {
    return "ct.mma_scaled(" + v[0] + ",ct.bitcast(" + v[1] +
           ",ct.float8_e8m0fnu)," + v[2] + ",ct.bitcast(" + v[3] +
           ",ct.float8_e8m0fnu)," + v[4] + ")";
  });
  rule("math.roundeven", [](Operation &, auto &v) {
    auto lo = "ct.floor(" + v[0] + ")", frac = "(" + v[0] + "-" + lo + ")";
    return lo + "+ct.astype((" + frac + ">0.5)|((" + frac +
           "==0.5)&((ct.astype(" + lo + ",ct.int32)&1)!=0)),ct.float32)";
  });
  rule("ct_src.tile_pid", [](Operation &o, auto &) {
    return "ct.astype(ct.bid(" +
           std::to_string(o.getAttrOfType<IntegerAttr>("axis").getInt()) +
           "),ct.int64)";
  });
  rule("ct_src.tile_axis", [](Operation &o, auto &) {
    return "ct.arange(" + axis(o.getResult(0).getType()) + ",dtype=ct.int64)";
  });
  rule("ct_src.tile_address", [](Operation &o, auto &v) {
    auto axes = o.getAttrOfType<DenseI64ArrayAttr>("projection");
    auto type = dyn_cast<RankedTensorType>(o.getResult(0).getType());
    SmallVector<std::string> coords;
    for (auto [i, a] : llvm::enumerate(axes.asArrayRef())) {
      std::string n = "ct.astype(" + v[i + 1] + ",ct.int64)";
      if (type && type.getRank() == 2 && a >= 0)
        n = expand(n, 1 - a);
      coords.push_back(n);
    }
    return "(" + llvm::join(coords, ",") + ",)";
  });
  rule("ct_src.tile_read", [](Operation &, auto &v) {
    return "ct.gather(" + v[0] + "," + v[1] +
           (v.size() == 3 ? ",mask=" + v[2] + ",padding_value=0" : "") + ")";
  });
  rule("ct_src.tile_write", [](Operation &, auto &v) {
    return "ct.scatter(" + v[0] + "," + v[1] + "," + v[2] +
           (v.size() == 4 ? ",mask=" + v[3] : "") + ")";
  });
  rule("ct_src.tile_dot", [](Operation &o, auto &v) {
    auto type = o.getResult(0).getType();
    auto rhs = o.getAttrOfType<BoolAttr>("transpose_rhs").getValue()
                   ? "ct.transpose(" + v[1] + ")"
                   : v[1];
    auto acc = v.size() == 3
                   ? v[2]
                   : "ct.full(" + dims(type) + ",0," + dtype(type) + ")";
    return "ct.mma(" + v[0] + "," + rhs + "," + acc + ")";
  });
  rule("ct_src.tile_reduce", [](Operation &o, auto &v) {
    auto alg = o.getAttrOfType<StringAttr>("algebra").str(),
         ax = std::to_string(o.getAttrOfType<IntegerAttr>("axis").getInt());
    if (getElementTypeOrSelf(o.getOperand(0).getType()).isInteger(1))
      return "ct." + alg + "(ct.astype(" + v[0] + ",ct.int32),axis=" + ax +
             ")!=0";
    return "ct." + alg + "(" + v[0] + ",axis=" + ax + ")";
  });
  rule("ct_src.tile_broadcast", [](Operation &o, auto &v) {
    int a = o.getAttrOfType<IntegerAttr>("axis").getInt();
    auto t = o.getResult(0).getType();
    auto x = "ct.astype(" + v[0] + "," + dtype(t) + ")";
    return "ct.broadcast_to(" + (a >= 0 ? expand(x, 1 - a) : x) + "," +
           dims(t) + ")";
  });
  rule("ct_src.tile_prefix", [](Operation &o, auto &v) {
    return "ct.extract(" + v[0] +
           ",(0,),shape=" + dims(o.getResult(0).getType()) + ")";
  });
  rule("ct_src.drop_unit_owner", [](Operation &o, auto &v) {
    return "ct.reshape(" + v[0] + "," + dims(o.getResult(0).getType()) + ")";
  });
  rule("arith.constant", [](Operation &o, auto &) {
    auto val = o.getAttrOfType<TypedAttr>("value");
    auto t = val.getType();
    if (auto dense = dyn_cast<DenseElementsAttr>(val))
      return "ct.full(" + dims(t) + "," +
             intentdsl::kir::literal(dense.getSplatValue<TypedAttr>()) + "," +
             dtype(t) + ")";
    if (auto i = dyn_cast<IntegerAttr>(val)) {
      if (t.isInteger(1))
        return std::string(i.getValue().isZero() ? "False" : "True");
      return std::to_string(i.getInt());
    }
    auto number = cast<FloatAttr>(val).getValue();
    if (number.isInfinity())
      return std::string(number.isNegative() ? "-float('inf')"
                                             : "float('inf')");
    if (number.isNaN())
      return std::string("float('nan')");
    return intentdsl::kir::literal(val);
  });
  for (auto [n, token] : {std::pair{"muli", "*"},
                          {"addi", "+"},
                          {"subi", "-"},
                          {"divui", "//"},
                          {"floordivsi", "//"},
                          {"remui", "%"},
                          {"remsi", "%"},
                          {"andi", "&"},
                          {"ori", "|"},
                          {"xori", "^"},
                          {"shli", "<<"},
                          {"shrui", ">>"}})
    rule("arith." + std::string(n),
         [token](auto &, auto &v) { return v[0] + token + v[1]; });
  for (auto [n, method] : {std::pair{"mulf", "mul"},
                           {"subf", "sub"},
                           {"addf", "add"},
                           {"divf", "truediv"}})
    rule("arith." + std::string(n), [method](Operation &, auto &v) {
      return "ct." + std::string(method) + "(" + llvm::join(v, ",") +
             ",rounding_mode=ct.RoundingMode.RN)";
    });
  for (auto n : {"extf", "truncf", "extsi", "extui", "index_cast", "trunci",
                 "sitofp", "fptosi"})
    rule("arith." + std::string(n), [](Operation &o, auto &v) {
      return "ct.astype(" + v[0] + "," + dtype(o.getResult(0).getType()) + ")";
    });
  for (auto [n, method] : {std::pair{"arith.minsi", "minimum"},
                           {"arith.minimumf", "minimum"},
                           {"arith.minnumf", "minimum"},
                           {"arith.maxnumf", "maximum"},
                           {"math.exp", "exp"},
                           {"math.log", "log"},
                           {"math.exp2", "exp2"},
                           {"math.sqrt", "sqrt"},
                           {"math.absf", "abs"}})
    rule(n, [method](Operation &o, auto &v) {
      auto nan = o.getName().getStringRef() == "arith.minimumf"
                     ? ",propagate_nan=True"
                     : "";
      return "ct." + std::string(method) + "(" + llvm::join(v, ",") + nan + ")";
    });
  rule("arith.select",
       [](auto &, auto &v) { return "ct.where(" + llvm::join(v, ",") + ")"; });
  rule("arith.cmpi", [](Operation &o, auto &v) {
    auto pred = cast<arith::CmpIOp>(o).getPredicate();
    return v[0] +
           (pred == arith::CmpIPredicate::eq    ? "=="
            : pred == arith::CmpIPredicate::ne  ? "!="
            : pred == arith::CmpIPredicate::sge ? ">="
            : pred == arith::CmpIPredicate::sle ? "<="
            : pred == arith::CmpIPredicate::sgt ? ">"
                                                : "<") +
           v[1];
  });
  rule("arith.cmpf", [](Operation &o, auto &v) {
    return v[0] +
           (cast<arith::CmpFOp>(o).getPredicate() == arith::CmpFPredicate::OEQ
                ? "=="
                : "!=") +
           v[1];
  });
  rule("ct_src.read_owner_matrix", [](Operation &o, auto &v) {
    auto n = axis(o.getResult(0).getType());
    return "ct.reshape(ct.load(" + v[0] + ",(" + v[1] + ",0,0),shape=(1," + n +
           "," + n + ")),(" + n + "," + n + "))";
  });
  rule("ct_src.store_column", [](Operation &o, auto &v) {
    auto n = axis(o.getOperand(3).getType());
    return "ct.scatter(" + v[0] + ",(" + v[1] + ",ct.arange(" + n +
           ",dtype=ct.int64)," + v[2] + ")," + v[3] + ")";
  });
  rule("ct_src.column", [](Operation &o, auto &v) {
    auto n = axis(o.getResult(0).getType());
    return "ct.reshape(ct.extract(" + v[0] + ",(0,ct.astype(" + v[1] +
           ",ct.int32)),shape=(" + n + ",1)),(" + n + ",))";
  });
  rule("ct_src.element", [](Operation &, auto &v) {
    return "ct.reshape(ct.extract(" + v[0] + ",(ct.astype(" + v[1] +
           ",ct.int32),),shape=(1,)),())";
  });
  rule("ct_src.lower_column", [](Operation &o, auto &v) {
    auto i = "ct.arange(" + axis(o.getResult(0).getType()) + ",dtype=ct.int64)";
    return "ct.where(" + i + "<" + v[2] + ",0.,ct.where(" + i + "==" + v[2] +
           "," + v[1] + ",ct.truediv(" + v[0] + "," + v[1] +
           ",rounding_mode=ct.RoundingMode.RN)))";
  });
  rule("ct_src.outer_product", [](Operation &, auto &v) {
    return "ct.mul(" + expand(v[0], 1) + "," + expand(v[1], 0) +
           ",rounding_mode=ct.RoundingMode.RN)";
  });
  rowRules(p);
  p.package = [](DictionaryAttr attrs, const auto &files,
                 llvm::raw_pwrite_stream &os) -> llvm::Error {
    auto manifest = attrs.getAs<StringAttr>("ct_src.package");
    if (!manifest)
      return llvm::createStringError("missing cuTile package");
    auto data = llvm::json::parse(manifest.getValue());
    if (!data)
      return data.takeError();
    auto source = files.find("kernel.py");
    if (!data->getAsObject() || source == files.end())
      return llvm::createStringError("invalid cuTile artifact");
    (*data->getAsObject())["source"] = source->second;
    os << *data;
    return llvm::Error::success();
  };
  return p;
}
} // namespace CI_UNIT_NAMESPACE
