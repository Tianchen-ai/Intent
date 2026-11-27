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
  if (type.isF64())
    return "ct.float64";
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
  p.operations["ct_src.library_gemm"] = PrintRule::format("pass");
  p.operations["ct_src.from_le_half"] = PrintRule::format(
      "$r0 = ct.astype(ct.bitcast(ct.astype($0|($1<<8),ct.uint16),ct.float16),ct.float32)");
  p.operations["ct_src.transpose"] =
      PrintRule::format("$r0 = ct.transpose($0)");
  p.operations["ct_src.extract_half"] = PrintRule::format(
      "$r0 = ct.extract($0,(0,{side},0),shape=({groups},1,{span}))");
  p.operations["ct_src.cat_pair"] =
      PrintRule::format("$r0 = ct.cat(($0,$1),1)");
  p.operations["ct_src.run_length_scan"] = PrintRule::format(
      "$r0 = ct.scan(($0,$1),0,lambda "
      "a,n,b,m:(b,ct.where(a==b,n+m,m)),(-1,0),reverse=({reverse}!=0))[1]");
  p.operations["ct_src.scan_sum"] =
      PrintRule::format("$r0 = ct.cumsum($0,axis=1)");
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
                          {"shrsi", ">>"},
                          {"shrui", ">>"}})
    p.operations["arith." + std::string(n)] =
        PrintRule::format(std::string("$r0 = $0") + token + "$1");
  p.operations["arith.minsi"] = PrintRule::format("$r0 = ct.minimum($operands)");
  p.operations["arith.maxsi"] = PrintRule::format("$r0 = ct.maximum($operands)");
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
  localExpr("ct_src.scaled_mma", [](const LocalOpView &v) {
    auto fmtAttr = v.attrOfType<StringAttr>("format");
    if (fmtAttr && fmtAttr.getValue() == "e2m1") {
      auto aTy = cast<RankedTensorType>(v.operandType(0));
      auto bTy = cast<RankedTensorType>(v.operandType(2));
      int64_t m = aTy.getDimSize(0), kHalf = aTy.getDimSize(1), k = kHalf * 2,
              n = bTy.getDimSize(1);
      std::string lhs = "ct.reshape(unpack_from_bytes(ct.reshape(" +
                        v.operand(0).str() + ",(" + std::to_string(m * kHalf) +
                        ",)),ct.float4_e2m1fn),(" + std::to_string(m) + "," +
                        std::to_string(k) + "))";
      std::string rhs =
          "ct.transpose(ct.reshape(unpack_from_bytes(ct.reshape(ct.transpose(" +
          v.operand(2).str() + "),(" + std::to_string(n * kHalf) +
          ",)),ct.float4_e2m1fn),(" + std::to_string(n) + "," +
          std::to_string(k) + ")))";
      return ("ct.mma_scaled(" + lhs + ",ct.bitcast(" + v.operand(1).str() +
              ",ct.float8_e8m0fnu)," + rhs + ",ct.bitcast(" +
              v.operand(3).str() + ",ct.float8_e8m0fnu)," +
              v.operand(4).str() + ")");
    }
    return ("ct.mma_scaled(" + v.operand(0).str() + ",ct.bitcast(" +
            v.operand(1).str() + ",ct.float8_e8m0fnu)," + v.operand(2).str() +
            ",ct.bitcast(" + v.operand(3).str() + ",ct.float8_e8m0fnu)," +
            v.operand(4).str() + ")");
  });
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
          c.line("from cuda.tile._stub import unpack_from_bytes");
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
      [](const LocalOpView &v, PrintPoint at, PrintContext &c) -> llvm::Error {
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
        } else if (at == PrintPoint::Leave && v.numResults() > 0) {
          auto idAttr = v.attrOfType<IntegerAttr>("ct_src.carry_id");
          if (!idAttr)
            return llvm::createStringError(
                "scf.if with results requires ct_src.carry_id");
          std::string prefix = "_carry_" + std::to_string(idAttr.getInt()) + "_";
          for (unsigned i = 0; i < v.numResults(); ++i)
            c.line(v.result(i).str() + " = " + prefix + std::to_string(i));
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
  localExpr("ct_src.panel_extract", [](const LocalOpView &v) {
    auto inTy = cast<RankedTensorType>(v.operandType(0));
    auto outTy = cast<RankedTensorType>(v.resultType(0));
    int64_t ax = v.attrOfType<IntegerAttr>("axis").getInt();
    int64_t idx = v.attrOfType<IntegerAttr>("index").getInt();
    SmallVector<std::string> coords, extDims;
    int64_t extElems = 1;
    for (int64_t i = 0; i < inTy.getRank(); ++i) {
      int64_t d = i == ax ? 1 : inTy.getDimSize(i);
      coords.push_back(i == ax ? std::to_string(idx) : "0");
      extDims.push_back(std::to_string(d));
      extElems *= d;
    }
    int64_t outElems = 1;
    for (int64_t d : outTy.getShape())
      outElems *= d;
    std::string extracted = "ct.extract(" + v.operand(0).str() + ",(" +
                            llvm::join(coords, ",") + ",),shape=(" +
                            llvm::join(extDims, ",") + ",))";
    if (extElems == outElems)
      return "ct.reshape(" + extracted + "," + dims(outTy) + ")";
    std::string unitShape =
        outTy.getRank() == 2
            ? "(1," + std::to_string(outTy.getDimSize(1)) + ")"
            : "(1,)";
    return "ct.broadcast_to(ct.reshape(" + extracted + "," + unitShape + ")," +
           dims(outTy) + ")";
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
    bool narrow = v.attrOfType<IntegerAttr>("index_bits").getInt() == 32;
    StringRef idxTy = narrow ? "ct.int32" : "ct.int64";
    SmallVector<std::string> coords;
    for (auto [i, a] : llvm::enumerate(axes.asArrayRef())) {
      bool alreadyI32 =
          narrow && getElementTypeOrSelf(v.operandType(i + 1)).isInteger(32);
      std::string n =
          alreadyI32 ? v.operand(i + 1).str()
                     : ("ct.astype(" + v.operand(i + 1) + "," + idxTy + ")").str();
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
              ",check_bounds=True,memory_order=ct.MemoryOrder." +
              v.attrOfType<StringAttr>("order").getValue().upper() + ")")
          .str();
    }
    return ("ct.atomic_add(" + v.operand(0) + "," + v.operand(1) + "," +
            v.operand(2) + ",memory_order=ct.MemoryOrder." +
            v.attrOfType<StringAttr>("order").getValue().upper() + ")")
        .str();
  });
  auto atomicFmt = [](const LocalOpView &v, StringRef fnName) -> std::string {
    auto ord = v.attrOfType<StringAttr>("order").getValue();
    auto sc = v.attrOfType<StringAttr>("scope").getValue();
    std::string mo = ord == "acq_rel" ? "ct.MemoryOrder.ACQ_REL"
                                      : "ct.MemoryOrder.RELAXED";
    std::string ms = sc == "block" ? "ct.MemoryScope.BLOCK"
                                   : "ct.MemoryScope.DEVICE";
    if (v.numOperands() == 4) {
      int rank = cast<MemRefType>(v.operandType(0)).getRank();
      SmallVector<std::string, 4> masked;
      for (int i = 0; i < rank; ++i)
        masked.push_back("ct.where(" + v.operand(3).str() + "," +
                         v.operand(1).str() + "[" + std::to_string(i) +
                         "],-1)");
      return (fnName + "(" + v.operand(0) + ",(" + llvm::join(masked, ",") +
              ",)," + v.operand(2) + ",check_bounds=True,memory_order=" + mo +
              ",memory_scope=" + ms + ")")
          .str();
    }
    return (fnName + "(" + v.operand(0) + "," + v.operand(1) + "," +
            v.operand(2) + ",memory_order=" + mo + ",memory_scope=" + ms + ")")
        .str();
  };
  localExpr("ct_src.tile_atomic_fetch_add",
            [atomicFmt](const LocalOpView &v) {
              return atomicFmt(v, "ct.atomic_add");
            });
  localExpr("ct_src.tile_atomic_xchg",
            [atomicFmt](const LocalOpView &v) {
              return atomicFmt(v, "ct.atomic_xchg");
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
      if (!v.attr("ct_src.round_nearest"))
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
  localExpr("arith.bitcast", [](const LocalOpView &v) {
    return ("ct.bitcast(" + v.operand(0) + "," + dtype(v.resultType(0)) + ")")
        .str();
  });
  localExpr("math.exp2", [](const LocalOpView &v) {
    std::string ftz = v.attr("ct_src.ftz") ? ",flush_to_zero=True" : "";
    return "ct.exp2(" + llvm::join(v.operands(), ",") + ftz + ")";
  });
  localExpr("arith.divf", [](const LocalOpView &v) {
    bool approx = v.attrOfType<StringAttr>("ct_src.division") == "approx";
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

  p.operations["func.func"] = PrintRule::local(
      [](const LocalOpView &v, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::BeginBlock) {
          c.line("@ct.kernel");
          int64_t idxBits =
              v.attrOfType<IntegerAttr>("ct_src.index_bits").getInt();
          SmallVector<std::string> args;
          for (auto [name, ty] :
               llvm::zip(v.blockArguments(), v.blockArgumentTypes())) {
            std::string param = name;
            if (isa<MemRefType>(ty) && idxBits == 64)
              param += ": ct.IndexedWithInt64[ct.Array]";
            args.push_back(std::move(param));
          }
          auto extra = v.attrOfType<ArrayAttr>("ct_src.extra_params");
          if (!extra)
            return llvm::createStringError("missing ct_src.extra_params");
          for (Attribute a : extra)
            args.push_back(cast<StringAttr>(a).str());
          auto entryAttr = v.attrOfType<StringAttr>("ct_src.entry");
          std::string entryName =
              entryAttr ? entryAttr.getValue().str() : "kernel";
          c.line("def " + entryName + "(" + llvm::join(args, ",") + "):");
          ++c.indent;
        } else if (at == PrintPoint::Leave) {
          --c.indent;
        }
        return llvm::Error::success();
      });
  p.operations["scf.for"] = PrintRule::local(
      [](const LocalOpView &v, PrintPoint at, PrintContext &c) -> llvm::Error {
        auto kindAttr = v.attrOfType<StringAttr>("ct_src.loop_kind");
        if (!kindAttr)
          return llvm::createStringError(
              "missing ct_src.loop_kind on scf.for");
        StringRef kind = kindAttr.getValue();
        bool native = kind == "range_i32";
        bool counted = kind == "range_const";
        auto bounds = v.attrOfType<ArrayAttr>("ct_src.range_bounds");
        unsigned nRes = v.numResults();
        auto idAttr = v.attrOfType<IntegerAttr>("ct_src.carry_id");
        if (nRes > 0 && !idAttr)
          return llvm::createStringError(
              "scf.for with results requires ct_src.carry_id");
        std::string prefix =
            idAttr ? "_carry_" + std::to_string(idAttr.getInt()) + "_" : "";
        if (at == PrintPoint::Enter) {
          for (unsigned i = 0; i < nRes; ++i)
            c.line(prefix + std::to_string(i) + " = " + v.operand(3 + i).str());
        } else if (at == PrintPoint::BeginBlock) {
          auto index = v.blockArguments()[0];
          if (native) {
            auto i32Bound = [&](unsigned idx) -> std::string {
              if (bounds && idx < bounds.size())
                if (auto ia = dyn_cast<IntegerAttr>(bounds[idx]))
                  return std::to_string(ia.getInt());
              return "ct.astype(" + v.operand(idx).str() + ",ct.int32)";
            };
            c.line("for " + index + " in range(" + i32Bound(0) + "," +
                   i32Bound(1) + "," + i32Bound(2) + "):");
          } else if (counted) {
            auto cstBound = [&](unsigned idx) {
              return std::to_string(cast<IntegerAttr>(bounds[idx]).getInt());
            };
            c.line("for it_" + index + " in range(" + cstBound(0) + "," +
                   cstBound(1) + "," + cstBound(2) + "):");
          } else {
            std::string ivTy =
                v.operandType(0).isInteger(32) ? "ct.int32" : "ct.int64";
            c.line(index + " = ct.astype(" + v.operand(0).str() + "," + ivTy +
                   ")");
            c.line("while " + index + " < ct.astype(" + v.operand(1).str() +
                   "," + ivTy + "):");
          }
          ++c.indent;
          if (!native && counted)
            c.line(index + " = ct.astype(it_" + index + ",ct.int64)");
          for (unsigned i = 0; i < nRes; ++i)
            c.line(v.blockArguments()[1 + i] + " = " + prefix +
                   std::to_string(i));
        } else if (at == PrintPoint::EndBlock) {
          if (!native && !counted) {
            std::string ivTy =
                v.operandType(0).isInteger(32) ? "ct.int32" : "ct.int64";
            c.line(v.blockArguments()[0] + " += ct.astype(" +
                   v.operand(2).str() + "," + ivTy + ")");
          }
          --c.indent;
        } else if (at == PrintPoint::Leave) {
          for (unsigned i = 0; i < nRes; ++i)
            c.line(v.result(i).str() + " = " + prefix + std::to_string(i));
        }
        return llvm::Error::success();
      });
  p.operations["scf.yield"] = PrintRule::local(
      [](const LocalOpView &v, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter && v.numOperands()) {
          auto idAttr = v.attrOfType<IntegerAttr>("ct_src.carry_id");
          if (!idAttr)
            return llvm::createStringError(
                "value-carrying scf.yield requires ct_src.carry_id");
          std::string prefix = "_carry_" + std::to_string(idAttr.getInt()) + "_";
          SmallVector<std::string> lhs, rhs;
          for (unsigned i = 0; i < v.numOperands(); ++i) {
            lhs.push_back(prefix + std::to_string(i));
            rhs.push_back(v.operand(i).str());
          }
          c.line(llvm::join(lhs, ",") + " = " + llvm::join(rhs, ","));
        }
        return llvm::Error::success();
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
