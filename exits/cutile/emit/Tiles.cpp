#include "compilerinfra/Terminal.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"
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
std::string expand(std::string x, int ax) {
  return "ct.expand_dims(" + x + "," + std::to_string(ax) + ")";
}
bool viewHasAnyFastMath(const LocalOpView &v, arith::FastMathFlags mask) {
  auto attr = v.attrOfType<arith::FastMathFlagsAttr>("fastmath");
  return attr && bitEnumContainsAny(attr.getValue(), mask);
}
bool viewHasFastMath(const LocalOpView &v, arith::FastMathFlags req) {
  auto attr = v.attrOfType<arith::FastMathFlagsAttr>("fastmath");
  return attr && bitEnumContainsAll(attr.getValue(), req);
}
} // namespace

void rowRules(TerminalPrinter &);
TerminalPrinter cutilePrinter() {
  TerminalPrinter p;
  auto localExpr = [&](StringRef name, auto format) {
    p.operations[name.str()] = PrintRule::local(
        [format](const LocalOpView &view, PrintPoint at, PrintContext &c) {
          if (at != PrintPoint::Enter)
            return llvm::Error::success();
          std::string expr = format(view);
          if (!expr.empty())
            c.line(view.numResults() ? view.result(0).str() + " = " + expr
                                     : expr);
          return llvm::Error::success();
        });
  };

  // Declarative templates
  for (auto n : {"func.return", "ct_src.tile_program", "ct_src.end"})
    p.operations[n] = PrintRule::format("");
  p.operations["ct_src.from_le_half"] = PrintRule::format(
      "$r0 = ct.astype(ct.bitcast(ct.astype($0|($1<<8),ct.uint16),ct.float16),ct.float32)");
  p.operations["ct_src.transpose"] =
      PrintRule::format("$r0 = ct.transpose($0)");
  p.operations["ct_src.scan_sum"] =
      PrintRule::format("$r0 = ct.cumsum($0,axis=1)");
  p.operations["ct_src.scaled_mma"] = PrintRule::format(
      "$r0 = ct.mma_scaled($0,ct.bitcast($1,ct.float8_e8m0fnu),$2,ct.bitcast($3,ct.float8_e8m0fnu),$4)");
  p.operations["math.roundeven"] = PrintRule::format(
      "$r0 = ct.floor($0)+ct.astype((($0-ct.floor($0))>0.5)|((($0-ct.floor($0))==0.5)&((ct.astype(ct.floor($0),ct.int32)&1)!=0)),ct.float32)");
  p.operations["ct_src.tile_pid"] =
      PrintRule::format("$r0 = ct.astype(ct.bid({axis}),ct.int64)");
  p.operations["ct_src.element"] = PrintRule::format(
      "$r0 = ct.reshape(ct.extract($0,(ct.astype($1,ct.int32),),shape=(1,)),())");
  p.operations["ct_src.outer_product"] = PrintRule::format(
      "$r0 = ct.mul(ct.expand_dims($0,1),ct.expand_dims($1,0),rounding_mode=ct.RoundingMode.RN)");
  p.operations["arith.select"] = PrintRule::format("$r0 = ct.where($operands)");
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
    p.operations["arith." + std::string(n)] =
        PrintRule::format(std::string("$r0 = $0") + token + "$1");
  p.operations["arith.minsi"] = PrintRule::format("$r0 = ct.minimum($operands)");
  p.operations["arith.minimumf"] =
      PrintRule::format("$r0 = ct.minimum($operands,propagate_nan=True)");
  p.operations["arith.minnumf"] =
      PrintRule::format("$r0 = ct.minimum($operands)");
  p.operations["arith.maxnumf"] =
      PrintRule::format("$r0 = ct.maximum($operands)");
  p.operations["math.log"] = PrintRule::format("$r0 = ct.log($operands)");
  p.operations["math.sqrt"] = PrintRule::format("$r0 = ct.sqrt($operands)");
  p.operations["math.absf"] = PrintRule::format("$r0 = ct.abs($operands)");

  // Local rules (LocalOpView only)
  localExpr("math.exp", [](const LocalOpView &v) {
    if (viewHasFastMath(v, arith::FastMathFlags::afn)) {
      std::string ftz = v.attr("ct_src.ftz") ? ",flush_to_zero=True" : "";
      return ("ct.exp2((" + v.operand(0) + ")*1.4426950408889634" + ftz + ")")
          .str();
    }
    return ("ct.exp(" + v.operand(0) + ")").str();
  });
  p.operations["builtin.module"] = PrintRule::local(
      [](const LocalOpView &, PrintPoint at, PrintContext &c) {
        if (at == PrintPoint::Enter) {
          c.file = "kernel.py";
          c.line("import cuda.tile as ct");
        }
        return llvm::Error::success();
      });
  p.operations["ct_src.runtime_source"] = PrintRule::local(
      [](const LocalOpView &v, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter) {
          auto impl = v.attrOfType<StringAttr>("implementation");
          if (!impl)
            return llvm::createStringError(
                "missing runtime_source implementation");
          auto helper = c.source(impl.getValue());
          if (!helper)
            return helper.takeError();
          c.line(*helper);
        }
        return llvm::Error::success();
      });
  p.operations["ct_src.owner_space"] = PrintRule::local(
      [](const LocalOpView &v, PrintPoint at, PrintContext &c) {
        if (at == PrintPoint::BeginBlock && !v.blockArguments().empty())
          c.line(v.blockArguments()[0] + " = ct.astype(ct.bid(0),ct.int64)");
        return llvm::Error::success();
      });
  p.operations["scf.if"] = PrintRule::local(
      [](const LocalOpView &v, PrintPoint at, PrintContext &c) {
        if (at == PrintPoint::Enter)
          c.line("if " + v.operand(0) + ":");
        if (at == PrintPoint::BeginRegion &&
            c.region < v.numRegions() && v.numBlocks(c.region) > 0) {
          if (c.region)
            c.line("else:");
          ++c.indent;
        } else if (at == PrintPoint::EndRegion &&
                   c.region < v.numRegions() && v.numBlocks(c.region) > 0) {
          --c.indent;
        }
        return llvm::Error::success();
      });
  localExpr("ct_src.e8m0", [](const LocalOpView &v) {
    using F = intentdsl::kir::MX8Block;
    return ("ct.bitcast(ct.where(" + v.operand(0) +
            "==" + std::to_string(F::nan) + ",2143289344,ct.where(" +
            v.operand(0) + "==0,4194304," + v.operand(0) +
            "<<23)),ct.float32)")
        .str();
  });
  localExpr("ct_src.expand_scale", [](const LocalOpView &v) {
    return "ct.broadcast_to(" +
           expand(v.operand(0).str(),
                  v.attrOfType<IntegerAttr>("axis").getInt()) +
           "," + dims(v.resultType(0)) + ")";
  });
  localExpr("ct_src.reshape", [](const LocalOpView &v) {
    return ("ct.reshape(" + v.operand(0) + "," + dims(v.resultType(0)) + ")")
        .str();
  });
  localExpr("ct_src.rect_read", [](const LocalOpView &v) {
    auto coords = llvm::join(llvm::drop_begin(v.operands()), ",");
    return ("ct.load(" + v.operand(0) + ",(" + coords +
            ",),shape=" + dims(v.resultType(0)) +
            ",padding_mode=ct.PaddingMode.ZERO,allow_tma=" +
            (v.attrOfType<BoolAttr>("allow_tma").getValue() ? "True"
                                                            : "False") +
            ")")
        .str();
  });
  // tileiras 1.6 deadlocks in mbarrier synchronization when a 4D TMA store
  // follows a pipelined TMA load loop; keep rect_write on regular tile stores.
  localExpr("ct_src.rect_write", [](const LocalOpView &v) {
    return ("ct.store(" + v.operand(0) + ",(" +
            llvm::join(llvm::drop_begin(v.operands(), 2), ",") + ",)," +
            v.operand(1) + ",allow_tma=False)")
        .str();
  });
  localExpr("ct_src.tile_axis", [](const LocalOpView &v) {
    return "ct.arange(" + axis(v.resultType(0)) + ",dtype=ct.int64)";
  });
  localExpr("ct_src.tile_address", [](const LocalOpView &v) {
    auto axes = v.attrOfType<DenseI64ArrayAttr>("projection");
    auto type = dyn_cast<RankedTensorType>(v.resultType(0));
    StringRef idxTy =
        v.attrOfType<IntegerAttr>("index_bits").getInt() == 32 ? "ct.int32"
                                                               : "ct.int64";
    SmallVector<std::string> coords;
    for (auto [i, a] : llvm::enumerate(axes.asArrayRef())) {
      std::string n = ("ct.astype(" + v.operand(i + 1) + "," + idxTy + ")").str();
      if (type && type.getRank() == 2 && a >= 0)
        n = expand(n, 1 - a);
      coords.push_back(n);
    }
    return "(" + llvm::join(coords, ",") + ",)";
  });
  localExpr("ct_src.tile_read", [](const LocalOpView &v) {
    return ("ct.gather(" + v.operand(0) + "," + v.operand(1) +
            (v.numOperands() == 3 ? ",mask=" + v.operand(2) + ",padding_value=0"
                                  : "") +
            ")")
        .str();
  });
  localExpr("ct_src.tile_write", [](const LocalOpView &v) {
    return ("ct.scatter(" + v.operand(0) + "," + v.operand(1) + "," +
            v.operand(2) + (v.numOperands() == 4 ? ",mask=" + v.operand(3) : "") +
            ")")
        .str();
  });
  localExpr("ct_src.tile_atomic_add", [](const LocalOpView &v) {
    if (v.numOperands() == 4) {
      int rank = cast<MemRefType>(v.operandType(0)).getRank();
      SmallVector<std::string, 4> masked;
      for (int i = 0; i < rank; ++i)
        masked.push_back("ct.where(" + v.operand(3).str() + "," +
                         v.operand(1).str() + "[" + std::to_string(i) +
                         "],-1)");
      return ("ct.atomic_add(" + v.operand(0) + ",(" + llvm::join(masked, ",") +
              ",)," + v.operand(2) +
              ",check_bounds=True,memory_order=ct.MemoryOrder.RELAXED)")
          .str();
    }
    return ("ct.atomic_add(" + v.operand(0) + "," + v.operand(1) + "," +
            v.operand(2) + ",memory_order=ct.MemoryOrder.RELAXED)")
        .str();
  });
  localExpr("ct_src.tile_dot", [](const LocalOpView &v) {
    auto type = v.resultType(0);
    auto rhs = v.attrOfType<BoolAttr>("transpose_rhs").getValue()
                   ? ("ct.transpose(" + v.operand(1) + ")").str()
                   : v.operand(1).str();
    auto acc = v.numOperands() == 3
                   ? v.operand(2).str()
                   : "ct.full(" + dims(type) + ",0," + dtype(type) + ")";
    return ("ct.mma(" + v.operand(0) + "," + rhs + "," + acc + ")").str();
  });
  localExpr("ct_src.tile_reduce", [](const LocalOpView &v) {
    auto alg = v.attrOfType<StringAttr>("algebra").str(),
         ax = std::to_string(v.attrOfType<IntegerAttr>("axis").getInt());
    if (getElementTypeOrSelf(v.operandType(0)).isInteger(1))
      return ("ct." + alg + "(ct.astype(" + v.operand(0) + ",ct.int32),axis=" +
              ax + ")!=0")
          .str();
    return ("ct." + alg + "(" + v.operand(0) + ",axis=" + ax + ")").str();
  });
  localExpr("ct_src.tile_broadcast", [](const LocalOpView &v) {
    int a = v.attrOfType<IntegerAttr>("axis").getInt();
    auto t = v.resultType(0);
    auto x = ("ct.astype(" + v.operand(0) + "," + dtype(t) + ")").str();
    return "ct.broadcast_to(" + (a >= 0 ? expand(x, 1 - a) : x) + "," +
           dims(t) + ")";
  });
  localExpr("ct_src.tile_prefix", [](const LocalOpView &v) {
    return ("ct.extract(" + v.operand(0) + ",(0,),shape=" +
            dims(v.resultType(0)) + ")")
        .str();
  });
  localExpr("ct_src.drop_unit_owner", [](const LocalOpView &v) {
    return ("ct.reshape(" + v.operand(0) + "," + dims(v.resultType(0)) + ")")
        .str();
  });
  localExpr("arith.constant", [](const LocalOpView &v) {
    auto val = v.attrOfType<TypedAttr>("value");
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
  for (auto [n, method, tok] : {std::tuple{"mulf", "mul", "*"},
                                {"subf", "sub", "-"},
                                {"addf", "add", "+"}})
    localExpr("arith." + std::string(n), [method, tok](const LocalOpView &v) {
      if (!v.attr("ct_src.round_nearest") &&
          viewHasAnyFastMath(v, arith::FastMathFlags::contract |
                                    arith::FastMathFlags::reassoc |
                                    arith::FastMathFlags::afn))
        return (v.operand(0) + tok + v.operand(1)).str();
      return "ct." + std::string(method) + "(" + llvm::join(v.operands(), ",") +
             ",rounding_mode=ct.RoundingMode.RN)";
    });
  for (auto n : {"extf", "truncf", "extsi", "extui", "index_cast", "trunci",
                 "sitofp", "fptosi"})
    localExpr("arith." + std::string(n), [](const LocalOpView &v) {
      return ("ct.astype(" + v.operand(0) + "," + dtype(v.resultType(0)) + ")")
          .str();
    });
  localExpr("math.exp2", [](const LocalOpView &v) {
    std::string ftz = v.attr("ct_src.ftz") ? ",flush_to_zero=True" : "";
    return "ct.exp2(" + llvm::join(v.operands(), ",") + ftz + ")";
  });
  localExpr("arith.divf", [](const LocalOpView &v) {
    bool approx = viewHasAnyFastMath(
        v, arith::FastMathFlags::arcp | arith::FastMathFlags::afn);
    return "ct.truediv(" + llvm::join(v.operands(), ",") +
           (v.attr("ct_src.ftz") ? ",flush_to_zero=True" : "") +
           ",rounding_mode=ct.RoundingMode." + (approx ? "APPROX" : "RN") + ")";
  });
  localExpr("arith.cmpi", [](const LocalOpView &v) {
    auto raw = v.attrOfType<IntegerAttr>("predicate").getValue().getZExtValue();
    auto pred = arith::symbolizeCmpIPredicate(raw).value_or(
        arith::CmpIPredicate::slt);
    return (v.operand(0) +
            (pred == arith::CmpIPredicate::eq    ? "=="
             : pred == arith::CmpIPredicate::ne  ? "!="
             : pred == arith::CmpIPredicate::sge ? ">="
             : pred == arith::CmpIPredicate::sle ? "<="
             : pred == arith::CmpIPredicate::sgt ? ">"
                                                 : "<") +
            v.operand(1))
        .str();
  });
  localExpr("arith.cmpf", [](const LocalOpView &v) {
    auto raw = v.attrOfType<IntegerAttr>("predicate").getValue().getZExtValue();
    auto pred = arith::symbolizeCmpFPredicate(raw).value_or(
        arith::CmpFPredicate::UNE);
    return (v.operand(0) +
            (pred == arith::CmpFPredicate::OEQ   ? "=="
             : pred == arith::CmpFPredicate::OGT ? ">"
             : pred == arith::CmpFPredicate::OGE ? ">="
             : pred == arith::CmpFPredicate::OLT ? "<"
             : pred == arith::CmpFPredicate::OLE ? "<="
                                                 : "!=") +
            v.operand(1))
        .str();
  });
  localExpr("ct_src.read_owner_matrix", [](const LocalOpView &v) {
    auto n = axis(v.resultType(0));
    return ("ct.reshape(ct.load(" + v.operand(0) + ",(" + v.operand(1) +
            ",0,0),shape=(1," + n + "," + n + ")),(" + n + "," + n + "))")
        .str();
  });
  localExpr("ct_src.store_column", [](const LocalOpView &v) {
    auto n = axis(v.operandType(3));
    return ("ct.scatter(" + v.operand(0) + ",(" + v.operand(1) + ",ct.arange(" +
            n + ",dtype=ct.int64)," + v.operand(2) + ")," + v.operand(3) + ")")
        .str();
  });
  localExpr("ct_src.column", [](const LocalOpView &v) {
    auto n = axis(v.resultType(0));
    return ("ct.reshape(ct.extract(" + v.operand(0) + ",(0,ct.astype(" +
            v.operand(1) + ",ct.int32)),shape=(" + n + ",1)),(" + n + ",))")
        .str();
  });
  localExpr("ct_src.lower_column", [](const LocalOpView &v) {
    auto i = "ct.arange(" + axis(v.resultType(0)) + ",dtype=ct.int64)";
    return ("ct.where(" + i + "<" + v.operand(2) + ",0.,ct.where(" + i +
            "==" + v.operand(2) + "," + v.operand(1) + ",ct.truediv(" +
            v.operand(0) + "," + v.operand(1) +
            ",rounding_mode=ct.RoundingMode.RN)))")
        .str();
  });

  // Audited custom rules with explicit structural reasons
  p.operations["func.func"] = PrintRule::custom(
      [](Operation &op, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter) {
          c.line("@ct.kernel");
          auto fn = cast<func::FuncOp>(op);
          SmallVector<std::string> args;
          for (auto a : fn.getArguments()) {
            auto n = c.name(a);
            if (isa<MemRefType>(a.getType()) &&
                op.getAttrOfType<IntegerAttr>("ct_src.index_bits").getInt() ==
                    64)
              n += ": ct.IndexedWithInt64[ct.Array]";
            args.push_back(n);
          }
          auto convention = fn.front().front().getAttrOfType<StringAttr>(
              "calling_convention");
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
      },
      "entry signature inspects inner program calling_convention and domain rank");
  p.operations["scf.for"] = PrintRule::custom(
      [](Operation &op, PrintPoint at, PrintContext &c) -> llvm::Error {
        auto loop = cast<scf::ForOp>(op);
        auto kindAttr = loop->getAttrOfType<StringAttr>("ct_src.loop_kind");
        if (!kindAttr)
          return llvm::createStringError(
              "missing ct_src.loop_kind on scf.for");
        StringRef kind = kindAttr.getValue();
        bool native = kind == "range_i32";
        bool counted = kind == "range_const";
        auto bounds = loop->getAttrOfType<ArrayAttr>("ct_src.range_bounds");
        auto index = c.name(loop.getInductionVar());
        if (at == PrintPoint::Enter) {
          for (auto [dst, src] :
               llvm::zip(loop.getResults(), loop.getInitArgs()))
            c.line(c.name(dst) + " = " + c.name(src));
          if (native) {
            auto i32Bound = [&](unsigned idx, Value v) -> std::string {
              if (bounds && idx < bounds.size())
                if (auto ia = dyn_cast<IntegerAttr>(bounds[idx]))
                  return std::to_string(ia.getInt());
              return "ct.astype(" + c.name(v) + ",ct.int32)";
            };
            c.line("for " + index + " in range(" +
                   i32Bound(0, loop.getLowerBound()) + "," +
                   i32Bound(1, loop.getUpperBound()) + "," +
                   i32Bound(2, loop.getStep()) + "):");
          } else if (counted) {
            auto cstBound = [&](unsigned idx) {
              return std::to_string(cast<IntegerAttr>(bounds[idx]).getInt());
            };
            c.line("for it_" + index + " in range(" + cstBound(0) + "," +
                   cstBound(1) + "," + cstBound(2) + "):");
          } else {
            c.line(index + " = ct.astype(" + c.name(loop.getLowerBound()) +
                   ",ct.int64)");
            c.line("while " + index + " < " + c.name(loop.getUpperBound()) +
                   ":");
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
      },
      "binds scf.for loop-carried SSA results and region block arguments across Enter/Leave");
  p.operations["scf.yield"] = PrintRule::custom(
      [](Operation &op, PrintPoint at, PrintContext &c) {
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
      },
      "assigns yielded values to enclosing scf.for/scf.if SSA result names");

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
