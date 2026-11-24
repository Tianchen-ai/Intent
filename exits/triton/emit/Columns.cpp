#include "compilerinfra/Terminal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringExtras.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;

namespace {
std::string extent(Type t) {
  return std::to_string(cast<RankedTensorType>(t).getDimSize(0));
}

} // namespace
void registerColumnRules(TerminalPrinter &printer) {
  auto &rules = printer.operations;
  auto leaf = [&](std::string name, auto emit) {
    rules[name] = PrintRule::custom([emit](Operation &op, PrintPoint at, PrintContext &p) {
      if (at == PrintPoint::Enter)
        emit(op, p);
      return llvm::Error::success();
    }, "legacy_operation_rule");
  };
  leaf("func.return", [](auto &, auto &) {});
  leaf("triton_src.end", [](auto &, auto &) {});
  rules["func.func"] = PrintRule::custom([](Operation &op, PrintPoint at,
                          PrintContext &p) -> llvm::Error {
    if (at == PrintPoint::Enter) {
      bool row = op.hasAttr("triton_src.min_width");
      p.line("@triton.jit");
      std::vector<std::string> args;
      for (Value arg : cast<func::FuncOp>(op).getArguments()) {
        auto id = p.name(arg);
        args.push_back(id);
        if (row)
          if (auto view = dyn_cast<MemRefType>(arg.getType())) {
            auto dims = view.getRank();
            bool owner = op.hasAttr("kir.owner_axis");
            if (dims > 2 || (owner && dims > 1))
              for (int i = 0; i < dims - (owner ? 0 : 1); ++i)
                args.push_back(id + "_d" + std::to_string(i) + ":tl.constexpr");
            for (int i = 0; i < dims; ++i)
              args.push_back(id + "_s" + std::to_string(i) +
                             (dims > 2 ? ":tl.constexpr" : ""));
          }
      }
      if (row)
        args.insert(args.end(), {"M", "N"});
      args.insert(args.end(), {"ROWS:tl.constexpr", "WIDTH:tl.constexpr"});
      p.line("def kernel(" + llvm::join(args, ",") + "):");
      ++p.indent;
    } else if (at == PrintPoint::Leave)
      --p.indent;
    return llvm::Error::success();
  }, "legacy_operation_rule");
  leaf("triton_src.owner_space", [](Operation &op, PrintContext &p) {
    p.line(p.name(op.getRegion(0).front().getArgument(0)) +
           " = tl.program_id(0).to(tl.int64)");
  });
  leaf("arith.constant", [](Operation &op, PrintContext &p) {
    auto value = op.getAttr("value");
    auto lhs = p.name(op.getResult(0));
    if (auto i = dyn_cast<IntegerAttr>(value))
      p.line(lhs + ":tl.constexpr = " + std::to_string(i.getInt()));
    else {
      auto dense = cast<DenseFPElementsAttr>(value);
      auto width = extent(dense.getType());
      llvm::SmallString<32> literal;
      dense.getSplatValue<APFloat>().toString(literal);
      p.line(lhs + " = tl.full((" + width + "," + width + ")," + literal.str() +
             ",tl.float32)");
    }
  });
  rules["scf.for"] = PrintRule::custom([](Operation &op, PrintPoint at, PrintContext &p) {
    auto loop = cast<scf::ForOp>(op);
    if (at == PrintPoint::Enter) {
      for (auto [result, init] : llvm::zip(loop.getResults(), loop.getInitArgs()))
        p.line(p.name(result) + " = " + p.name(init));
      auto syntax = loop->hasAttr("triton_src.runtime_range")
                        ? " in range("
                        : " in tl.static_range(";
      auto bound = [&](Value v) {
        if (auto i = v.getDefiningOp<arith::ConstantIndexOp>())
          return std::to_string(i.value());
        return p.name(v);
      };
      p.line("for " + p.name(loop.getInductionVar()) + syntax +
             bound(loop.getLowerBound()) + "," + bound(loop.getUpperBound()) +
             "," + bound(loop.getStep()) + "):");
      ++p.indent;
      for (auto [carry, result] : llvm::zip(loop.getRegionIterArgs(), loop.getResults()))
        p.line(p.name(carry) + " = " + p.name(result));
    } else if (at == PrintPoint::Leave)
      --p.indent;
    return llvm::Error::success();
  }, "legacy_operation_rule");
  leaf("scf.yield", [](Operation &op, PrintContext &p) {
    std::vector<std::string> lhs, rhs;
    for (auto [result, next] : llvm::zip(op.getParentOp()->getResults(), op.getOperands())) {
      lhs.push_back(p.name(result));
      rhs.push_back(p.name(next));
    }
    if (!lhs.empty())
      p.line(llvm::join(lhs, ",") + " = " + llvm::join(rhs, ","));
  });
  // The layout, address width and view rank are checked during exit legalization.
  for (bool store : {false, true}) {
    leaf(store ? "triton_src.store_column" : "triton_src.read_owner_matrix",
         [store](Operation &op, PrintContext &p) {
      auto layout = op.getAttrOfType<DenseI64ArrayAttr>("triton_src.view_strides");
      auto width = std::to_string(op.getAttrOfType<IntegerAttr>("triton_src.side").getInt());
      auto row = "tl.arange(0," + width + ").to(tl.int64)";
      auto addr = p.name(op.getOperand(0)) + "+" + p.name(op.getOperand(1)) +
                  "*" + std::to_string(layout[0]) + "+";
      if (store)
        p.line("tl.store(" + addr + row + "*" + std::to_string(layout[1]) +
               "+tl.cast(" + p.name(op.getOperand(2)) + ",tl.int64)*" +
               std::to_string(layout[2]) + "," + p.name(op.getOperand(3)) + ")");
      else
        p.line(p.name(op.getResult(0)) + " = tl.load(" + addr + row +
               "[:,None]*" + std::to_string(layout[1]) + "+" + row +
               "[None,:]*" + std::to_string(layout[2]) + ")");
    });
  }
  auto expression = [&](std::string name, auto format) {
    leaf(name, [format](Operation &op, PrintContext &p) {
      std::vector<std::string> args;
      for (Value v : op.getOperands())
        args.push_back(p.name(v));
      p.line(p.name(op.getResult(0)) + " = " + format(op, args));
    });
  };
  expression("triton_src.column", [](Operation &op, auto &v) {
    auto n = extent(op.getResult(0).getType());
    return "tl.reshape(tl.gather(" + v[0] + ",tl.full((" + n + ",1)," + v[1] +
           ",tl.int32),1),(" + n + ",))";
  });
  expression("triton_src.element", [](Operation &, auto &v) {
    return "tl.reshape(tl.gather(" + v[0] + ",tl.full((1,)," + v[1] + ",tl.int32),0),())";
  });
  expression("triton_src.lower_column", [](Operation &op, auto &v) {
    auto r = "tl.arange(0," + extent(op.getResult(0).getType()) + ")";
    return "tl.where(" + r + "<" + v[2] + ",0.,tl.where(" + r + "==" + v[2] +
           "," + v[1] + ",tl.div_rn(" + v[0] + "," + v[1] + ")))";
  });
  expression("triton_src.outer_product", [](Operation &, auto &v) {
    return "libdevice.mul_rn(" + v[0] + "[:,None]," + v[1] + "[None,:])";
  });
}

} // namespace CI_UNIT_NAMESPACE
