#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringExtras.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
void registerColumnRules(TerminalPrinter &);
namespace {
using Shape = ArrayRef<int64_t>;
Shape shape(Type t) {
  auto s = dyn_cast<RankedTensorType>(t);
  return s ? s.getShape() : Shape{};
}
std::string tuple(Type t) {
  std::vector<std::string> dims;
  for (auto d : shape(t))
    dims.push_back(std::to_string(d));
  return "(" + llvm::join(dims, ",") + (dims.size() == 1 ? "," : "") + ")";
}
std::string dtype(Type t) {
  t = getElementTypeOrSelf(t);
  if (t.isBF16())
    return "tl.bfloat16";
  if (t.isF16())
    return "tl.float16";
  if (isa<Float8E4M3FNType>(t))
    return "tl.float8e4nv";
  if (t.isF32())
    return "tl.float32";
  if (t.isInteger(32))
    return "tl.int32";
  if (t.isInteger(64))
    return "tl.int64";
  if (t.isInteger(8))
    return "tl.int8";
  return "tl.int1";
}
TerminalPrinter tilePrinter() {
  TerminalPrinter p;
  registerColumnRules(p);
  auto leaf = [&](StringRef name, auto rule) {
    p.operations[name.str()] = [rule](Operation &op, PrintPoint at,
                                      PrintContext &c) {
      if (at != PrintPoint::Enter)
        return llvm::Error::success();
      std::vector<std::string> a;
      for (Value v : op.getOperands())
        a.push_back(c.name(v));
      auto expr = rule(op, a);
      if (!expr.empty())
        c.line(op.getNumResults() ? c.name(op.getResult(0)) + " = " + expr
                                  : expr);
      return llvm::Error::success();
    };
  };
  leaf("triton_src.tile_program", [](auto &, auto &) { return std::string(); });
  leaf("triton_src.tile_axis", [](Operation &op, auto &) {
    return "tl.arange(0," +
           std::to_string(shape(op.getResult(0).getType())[0]) +
           ").to(tl.int64)";
  });
  leaf("triton_src.tile_pid", [](Operation &op, auto &) {
    return "tl.program_id(" +
           std::to_string(op.getAttrOfType<IntegerAttr>("axis").getInt()) +
           ").to(tl.int64)";
  });
  leaf("triton_src.tile_prefix", [](Operation &op, auto &a) {
    return "tl.gather(" + a[0] + ",tl.arange(0," +
           std::to_string(shape(op.getResult(0).getType())[0]) + "),0)";
  });
  leaf("triton_src.tile_address", [](Operation &op, auto &a) {
    auto strides = op.getAttrOfType<DenseI64ArrayAttr>("strides");
    auto axes = op.getAttrOfType<DenseI64ArrayAttr>("projection");
    auto rank = shape(op.getResult(0).getType()).size();
    std::vector<std::string> terms;
    for (size_t i = 0; i < axes.size(); ++i) {
      std::string projection = rank == 2 && axes[i] >= 0
                                   ? (axes[i] == 0 ? "[:,None]" : "[None,:]")
                                   : "";
      terms.push_back("tl.cast(" + a[i + 1] + ",tl.int64)" + projection + "*" +
                      std::to_string(strides[i]));
    }
    return llvm::join(terms, " + ");
  });
  leaf("triton_src.tile_read", [](Operation &op, auto &a) {
    auto e = getElementTypeOrSelf(op.getResult(0).getType());
    bool fp8 = isa<Float8E4M3FNType>(e);
    return "tl.load(" + a[0] + (fp8 ? ".to(tl.pointer_type(tl.uint8))" : "") +
           "+" + a[1] + (a.size() == 3 ? ",mask=" + a[2] + ",other=0" : "") +
           ")" + (fp8 ? ".to(tl.float8e4nv,bitcast=True)" : "");
  });
  leaf("triton_src.from_le_half", [](auto &, auto &a) {
    return "(" + a[0] + "|(" + a[1] +
           "<<8)).to(tl.uint16).to(tl.float16,bitcast=True).to(tl.float32)";
  });
  leaf("triton_src.e8m0", [](Operation &op, auto &a) {
    using F = intentdsl::kir::MX8Block;
    return "tl.where(" + a[0] + "==" + std::to_string(F::nan) +
           ",float('nan'),libdevice.ldexp(tl.full(" +
           tuple(op.getResult(0).getType()) + ",1.,tl.float32)," + a[0] + "-" +
           std::to_string(F::bias) + "))";
  });
  leaf("triton_src.tile_write", [](auto &, auto &a) {
    return "tl.store(" + a[0] + "+" + a[1] + "," + a[2] +
           (a.size() == 4 ? ",mask=" + a[3] : "") + ")";
  });
  leaf("triton_src.tile_broadcast", [](Operation &op, auto &a) {
    auto axis = op.getAttrOfType<IntegerAttr>("axis").getInt();
    auto v = "tl.cast(" + a[0] + "," + dtype(op.getOperand(0).getType()) + ")" +
             (axis < 0    ? ""
              : axis == 0 ? "[:,None]"
                          : "[None,:]");
    return "tl.broadcast_to(" + v + "," + tuple(op.getResult(0).getType()) +
           ")";
  });
  leaf("triton_src.tile_dot", [](Operation &op, auto &a) {
    auto rhs = op.getAttrOfType<BoolAttr>("transpose_rhs").getValue()
                   ? "tl.trans(" + a[1] + ")"
                   : a[1];
    return "tl.dot(" + a[0] + "," + rhs + (a.size() == 3 ? "," + a[2] : "") +
           (op.hasAttr("triton_src.exact_acc") ? ",max_num_imprecise_acc=0"
                                               : "") +
           ",input_precision='ieee',out_dtype=" +
           dtype(op.getResult(0).getType()) + ")";
  });
  leaf("triton_src.tile_reduce", [](Operation &op, auto &a) {
    auto expression =
        "tl." + op.getAttrOfType<StringAttr>("algebra").str() + "(" + a[0] +
        "," + std::to_string(op.getAttrOfType<IntegerAttr>("axis").getInt()) +
        ")";
    return getElementTypeOrSelf(op.getResult(0).getType()).isInteger(1)
               ? expression + ".to(tl.int1)"
               : expression;
  });
  leaf("arith.constant", [](Operation &op, auto &) {
    auto v = op.getAttr("value");
    if (auto i = dyn_cast<IntegerAttr>(v))
      return std::to_string(i.getInt());
    if (auto f = dyn_cast<FloatAttr>(v)) {
      auto num = f.getValue();
      if (num.isInfinity())
        return std::string(num.isNegative() ? "-float('inf')" : "float('inf')");
      llvm::SmallString<32> txt;
      num.toString(txt);
      std::string number = txt.str().str();
      if (number.find_first_of(".eE") == std::string::npos)
        number += ".0";
      return number;
    }
    auto type = op.getResult(0).getType();
    return "tl.full(" + tuple(type) + ",0," + dtype(type) + ")";
  });
  for (auto [name, token] : {std::pair{"arith.muli", "*"},
                             {"arith.addi", "+"},
                             {"arith.divui", "//"},
                             {"arith.remui", "%"},
                             {"arith.mulf", "*"},
                             {"arith.divf", "/"},
                             {"arith.subf", "-"},
                             {"arith.addf", "+"},
                             {"arith.andi", "&"}})
    leaf(name, [token](Operation &op, auto &v) {
      if (op.hasAttr("triton_src.round_nearest"))
        return "libdevice." + op.getName().stripDialect().drop_back().str() +
               "_rn(" + llvm::join(v, ",") + ")";
      return v[0] + token + v[1];
    });
  leaf("arith.cmpi", [](Operation &op, auto &a) {
    auto pred = cast<arith::CmpIOp>(op).getPredicate();
    auto token = pred == arith::CmpIPredicate::sge  ? ">="
                 : pred == arith::CmpIPredicate::eq ? "=="
                 : pred == arith::CmpIPredicate::ne ? "!="
                                                    : "<";
    return a[0] + token + a[1];
  });
  leaf("arith.cmpf", [](Operation &op, auto &a) {
    return a[0] +
           (cast<arith::CmpFOp>(op).getPredicate() == arith::CmpFPredicate::OEQ
                ? "=="
                : "!=") +
           a[1];
  });
  leaf("arith.minimumf", [](auto &, auto &a) {
    return "tl.minimum(" + llvm::join(a, ",") +
           ",propagate_nan=tl.PropagateNan.ALL)";
  });
  leaf("math.absf", [](auto &, auto &a) { return "tl.abs(" + a[0] + ")"; });
  leaf("math.roundeven",
       [](auto &, auto &a) { return "libdevice.rint(" + a[0] + ")"; });
  leaf("arith.minnumf", [](auto &, auto &a) {
    return "tl.minimum(" + llvm::join(a, ",") + ")";
  });
  leaf("arith.shli", [](auto &, auto &a) { return a[0] + "<<" + a[1]; });
  leaf("arith.shrui", [](auto &, auto &a) {
    return "(" + a[0] + ".to(tl.uint32)>>" + a[1] + ").to(tl.int32)";
  });
  for (auto [name, op] :
       {std::pair{"arith.maxnumf", "maximum"}, {"arith.minsi", "minimum"}})
    leaf(name, [op](auto &, auto &a) {
      return "tl." + std::string(op) + "(" + llvm::join(a, ",") + ")";
    });
  for (auto [name, token] : {std::pair{"arith.subi", "-"},
                             {"arith.floordivsi", "//"},
                             {"arith.remsi", "%"},
                             {"arith.ori", "|"},
                             {"arith.xori", "^"}})
    leaf(name, [token](auto &, auto &a) { return a[0] + token + a[1]; });
  for (auto name : {"exp", "sqrt", "log"})
    leaf("math." + std::string(name), [name](auto &, auto &a) {
      return "tl." + std::string(name) + "(" + a[0] + ")";
    });
  leaf("arith.select",
       [](auto &, auto &a) { return "tl.where(" + llvm::join(a, ",") + ")"; });
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
  for (auto name :
       {"arith.truncf", "arith.extf", "arith.extsi", "arith.sitofp",
        "arith.fptosi", "arith.index_cast", "arith.trunci", "arith.extui"})
    leaf(name, [](Operation &op, auto &a) {
      return "tl.cast(" + a[0] + "," + dtype(op.getResult(0).getType()) + ")";
    });
  leaf("math.exp2", [](auto &, auto &a) { return "tl.exp2(" + a[0] + ")"; });
  p.package = [](DictionaryAttr, const auto &files,
                 llvm::raw_pwrite_stream &out) {
    out << files.at("kernel.py");
    return llvm::Error::success();
  };
  return p;
}
LogicalResult dotSignature(Operation *op) {
  auto l = shape(op->getOperand(0).getType()),
       r = shape(op->getOperand(1).getType()),
       m = shape(op->getResult(0).getType());
  bool tr = op->getAttrOfType<BoolAttr>("transpose_rhs").getValue();
  auto element = getElementTypeOrSelf(op->getOperand(0).getType());
  auto acc = getElementTypeOrSelf(op->getResult(0).getType());
  bool type = element.isInteger(8)
                  ? acc.isInteger(32)
                  : ((isa<Float8E4M3FNType>(element) || element.isBF16() ||
                      element.isF16() || element.isF32()) &&
                     acc.isF32());
  return success(
      type && element == getElementTypeOrSelf(op->getOperand(1).getType()) &&
      l.size() == 2 && r.size() == 2 && m.size() == 2 &&
      l[1] == r[tr ? 1 : 0] && m[0] == l[0] && m[1] == r[tr ? 0 : 1] &&
      l[0] >= 16 && l[1] >= 16 && m[1] >= 16 &&
      (op->getNumOperands() == 2 ||
       op->getOperand(2).getType() == op->getResult(0).getType()));
}
LogicalResult collectiveSignature(Operation *op) {
  auto input = op->getOperand(0).getType(), result = op->getResult(0).getType();
  SmallVector<int64_t> dims(shape(input));
  auto axis = op->getAttrOfType<IntegerAttr>("axis").getInt();
  auto alg = op->getAttrOfType<StringAttr>("algebra");
  if (axis < 0 || size_t(axis) >= dims.size() ||
      !llvm::isPowerOf2_64(dims[axis]))
    return failure();
  dims.erase(dims.begin() + axis);
  return success(ArrayRef<int64_t>(dims) == shape(result) &&
                 getElementTypeOrSelf(input) == getElementTypeOrSelf(result) &&
                 (alg == "sum" || alg == "min" || alg == "max"));
}
} // namespace
void tileContracts(Unit &unit) {
  for (bool isDot : {true, false}) {
    ImplementationContract c;
    c.name = isDot ? "native-dot" : "row-collective";
    c.operation = isDot ? "triton_src.tile_dot" : "triton_src.tile_reduce";
    c.symbol = isDot ? "tl.dot" : "tl.sum/tl.max";
    c.source = ImplementationSource::Builtin;
    c.minimum = Reuse::Native;
    c.operands =
        isDot ? "[M,K] x [K,N] or transposed RHS" : "f32[M,K] -> f32[M]";
    c.layout = isDot ? "register tiles, contracted axis agrees"
                     : "row fragment, power-of-two width";
    c.accumulator = isDot ? "f32 or i32" : "f32";
    c.effects = "pure register results";
    c.numerical = isDot ? "IEEE input precision, declared accumulator"
                        : "Triton sum or max along axis 1";
    c.estimator = "exits/triton::matrix-issues";
    c.costParameters = isDot ? std::vector<std::string>{"M", "N", "K"}
                             : std::vector<std::string>{"M", "K"};
    c.available = [](const TargetProfile &p) -> llvm::Error {
      if (!p.services.get<intentdsl::block::OwnerMachine>())
        return llvm::createStringError("missing CUDA execution capabilities");
      return llvm::Error::success();
    };
    c.verify = isDot ? dotSignature : collectiveSignature;
    unit.implementations.push_back({std::move(c), COMPILERINFRA_SOURCE});
  }
}
LogicalResult verifyTileProgram(func::FuncOp fn, const TargetProfile &profile) {
  auto &sp = fn.front().front();
  auto facts = profile.services.get<intentdsl::CallFacts>();
  if (!facts || !fn->getAttrOfType<StringAttr>("kir.abi") ||
      !fn->hasAttr("kir.domain_arg"))
    return failure();
  auto mode = sp.getAttrOfType<StringAttr>("calling_convention");
  auto rows = sp.getAttrOfType<IntegerAttr>("block.owner_tile"),
       owners = sp.getAttrOfType<IntegerAttr>("block.owners");
  if (mode == "owners") {
    if (!rows || rows.getInt() < 1 || !owners ||
        owners.getInt() != intentdsl::kir::ownerCount(&sp, *facts))
      return sp.emitError("invalid owner grid");
  } else if (mode == "matrix") {
    auto dims = sp.getAttrOfType<DenseI64ArrayAttr>("block.extents");
    auto tile = sp.getAttrOfType<DenseI64ArrayAttr>("block.output_tile");
    auto native = sp.getAttrOfType<StringAttr>("block.native_row");
    auto caps = profile.services.get<intentdsl::block::MatrixTarget>();
    if (!dims || dims.size() != 4 ||
        llvm::any_of(dims.asArrayRef(), [](auto i) { return i < 0; }) ||
        !tile || tile.size() != 3 || !native || !caps ||
        !llvm::any_of(caps->tiles, [&](auto &t) {
          return tile.asArrayRef() == ArrayRef<int64_t>{t.m, t.n, t.k} &&
                 native == t.native;
        }))
      return sp.emitError(
          "invalid installed contraction grid or native parameter row");
  } else
    return sp.emitError("unknown tile calling convention");
  auto check = sp.walk([&](Operation *op) {
    auto name = op->getName().getStringRef();
    if (name == "triton_src.tile_prefix") {
      auto input = op->getOperand(0).getType(),
           out = op->getResult(0).getType();
      auto a = shape(input), y = shape(out);
      if (a.size() != 1 || y.size() != 1 || y[0] < 1 || y[0] > a[0] ||
          !llvm::isPowerOf2_64(y[0]) || y[0] > INT32_MAX ||
          getElementTypeOrSelf(input) != getElementTypeOrSelf(out))
        return op->emitError("invalid logical fragment prefix"),
               WalkResult::interrupt();
    }
    if (name == "triton_src.tile_address") {
      auto view = dyn_cast<BlockArgument>(op->getOperand(0));
      auto axes = op->getAttrOfType<DenseI64ArrayAttr>("projection"),
           strides = op->getAttrOfType<DenseI64ArrayAttr>("strides");
      auto rank = cast<MemRefType>(op->getOperand(0).getType()).getRank();
      if (!view || view.getOwner() != &fn.front() || axes.size() != rank ||
          strides.size() != rank || op->getNumOperands() != rank + 1 ||
          !getElementTypeOrSelf(op->getResult(0).getType()).isInteger(64))
        return op->emitError("invalid address rank or index width"),
               WalkResult::interrupt();
      auto native =
          facts->arguments[view.getArgNumber()].getAsObject()->getArray(
              "strides");
      auto output = shape(op->getResult(0).getType());
      if (!native || native->size() != rank || output.size() > 2)
        return WalkResult::interrupt();
      for (int i = 0; i < rank; i++) {
        auto type = op->getOperand(i + 1).getType();
        auto coord = shape(type);
        auto axis = axes[i];
        if ((*native)[i].getAsInteger() != strides[i] ||
            !getElementTypeOrSelf(type).isInteger(64) ||
            (coord.empty()
                 ? axis != -1
                 : (coord.size() != 1 || axis < 0 ||
                    size_t(axis) >= output.size() || coord[0] != output[axis])))
          return op->emitError("coordinate projection or ABI stride mismatch"),
                 WalkResult::interrupt();
      }
    }
    if (name == "triton_src.tile_pid") {
      auto dim = op->getAttrOfType<IntegerAttr>("axis").getInt();
      if (dim < 0 || dim >= 3)
        return op->emitError("invalid launch dimension"),
               WalkResult::interrupt();
    }
    if (name == "triton_src.tile_axis") {
      auto result = op->getResult(0).getType();
      if (shape(result).size() != 1 || shape(result)[0] <= 0 ||
          !llvm::isPowerOf2_64(shape(result)[0]) ||
          !getElementTypeOrSelf(result).isInteger(64))
        return op->emitError("invalid i64 tile axis"), WalkResult::interrupt();
    }
    if (name == "triton_src.tile_broadcast") {
      auto src = op->getOperand(0).getType(), dst = op->getResult(0).getType();
      auto axis = op->getAttrOfType<IntegerAttr>("axis").getInt();
      if (getElementTypeOrSelf(src) != getElementTypeOrSelf(dst) ||
          (shape(src).empty()
               ? axis != -1
               : (shape(src).size() != 1 || shape(dst).size() != 2 ||
                  axis < 0 || axis > 1 || shape(src)[0] != shape(dst)[axis])))
        return op->emitError("invalid tile broadcast projection"),
               WalkResult::interrupt();
    }
    if (name == "triton_src.tile_read" || name == "triton_src.tile_write") {
      auto offset = op->getOperand(1);
      auto *addr = offset.getDefiningOp();
      bool load = name == "triton_src.tile_read";
      auto val = load ? op->getResult(0) : op->getOperand(2);
      if (op->getNumOperands() == (load ? 3 : 4)) {
        auto mask = op->getOperands().back().getType();
        if (shape(mask) != shape(offset.getType()) ||
            !getElementTypeOrSelf(mask).isInteger(1))
          return op->emitError("tile mask must cover its projected address"),
                 WalkResult::interrupt();
      }
      if (!addr ||
          addr->getName().getStringRef() != "triton_src.tile_address" ||
          addr->getOperand(0) != op->getOperand(0) ||
          shape(offset.getType()) != shape(val.getType()) ||
          getElementTypeOrSelf(val.getType()) !=
              cast<MemRefType>(op->getOperand(0).getType()).getElementType())
        return op->emitError("tile access does not match its verified view"),
               WalkResult::interrupt();
    }
    if (auto cmp = dyn_cast<arith::CmpIOp>(op);
        cmp && cmp.getPredicate() != arith::CmpIPredicate::eq &&
        cmp.getPredicate() != arith::CmpIPredicate::ne &&
        cmp.getPredicate() != arith::CmpIPredicate::sge &&
        cmp.getPredicate() != arith::CmpIPredicate::slt)
      return op->emitError("unsupported tile comparison"),
             WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (check.wasInterrupted())
    return failure();
  return tilePrinter().verify(fn, profile);
}
llvm::Error emitTileProgram(func::FuncOp fn, llvm::raw_pwrite_stream &out) {
  llvm::SmallVector<char> chars;
  llvm::raw_svector_ostream stream(chars);
  if (auto err = tilePrinter().emit(fn, stream))
    return err;
  StringRef src(chars.data(), chars.size());
  if (fn.front().front().getAttrOfType<StringAttr>("calling_convention") ==
      "matrix")
    return intentdsl::block::emitMatrixPackage(fn, src, out, 2);
  return intentdsl::block::emitOwnerArtifact(
      fn, src, llvm::json::Object{{"num_warps", 4}, {"num_stages", 1}}, 2, out);
}
} // namespace CI_UNIT_NAMESPACE
