#include "compilerinfra/Terminal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
  rules["func.return"] = PrintRule::format("");
  rules["triton_src.end"] = PrintRule::format("");
  rules["func.func"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &p) {
        if (at == PrintPoint::BeginBlock) {
          bool row = bool(view.attr("triton_src.min_width"));
          bool owner = bool(view.attr("kir.owner_axis"));
          p.line("@triton.jit");
          std::vector<std::string> args;
          for (auto [id, ty] :
               llvm::zip(view.blockArguments(), view.blockArgumentTypes())) {
            args.push_back(id);
            if (row)
              if (auto mem = dyn_cast<MemRefType>(ty)) {
                auto dims = mem.getRank();
                if (dims > 2 || (owner && dims > 1))
                  for (int i = 0; i < dims - (owner ? 0 : 1); ++i)
                    args.push_back(id + "_d" + std::to_string(i) +
                                   ":tl.constexpr");
                for (int i = 0; i < dims; ++i)
                  args.push_back(
                      id + "_s" + std::to_string(i) +
                      (dims > 2 ||
                               view.attr("triton_src.specialize_row_strides")
                           ? ":tl.constexpr"
                           : ""));
              }
          }
          if (row)
            args.insert(args.end(), {"M", "N"});
          args.insert(args.end(), {"ROWS:tl.constexpr", "WIDTH:tl.constexpr"});
          auto entryAttr = view.attrOfType<StringAttr>("block.entry");
          std::string entryName =
              entryAttr ? entryAttr.getValue().str() : "kernel";
          p.line("def " + entryName + "(" + llvm::join(args, ",") + "):");
          ++p.indent;
        } else if (at == PrintPoint::Leave) {
          --p.indent;
        }
        return llvm::Error::success();
      });
  rules["triton_src.owner_space"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &p) {
        if (at == PrintPoint::BeginBlock && !view.blockArguments().empty())
          p.line(view.blockArguments()[0] + " = tl.program_id(0).to(tl.int64)");
        return llvm::Error::success();
      });
  rules["scf.for"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &p) -> llvm::Error {
        unsigned nRes = view.numResults();
        auto idAttr = view.attrOfType<IntegerAttr>("triton_src.carry_id");
        if (nRes > 0 && !idAttr)
          return llvm::createStringError(
              "scf.for with results requires triton_src.carry_id");
        std::string prefix =
            idAttr ? "_carry_" + std::to_string(idAttr.getInt()) + "_" : "";
        if (at == PrintPoint::Enter) {
          for (unsigned i = 0; i < nRes; ++i)
            p.line(prefix + std::to_string(i) + " = " + view.operand(3 + i).str());
        } else if (at == PrintPoint::BeginBlock) {
          auto bounds = view.attrOfType<ArrayAttr>("triton_src.range_bounds");
          auto syntax = view.attr("triton_src.runtime_range")
                            ? " in range("
                            : " in tl.static_range(";
          auto bound = [&](unsigned idx) -> std::string {
            if (bounds && idx < bounds.size())
              if (auto ia = dyn_cast<IntegerAttr>(bounds[idx]))
                return std::to_string(ia.getInt());
            return view.operand(idx).str();
          };
          p.line("for " + view.blockArguments()[0] + syntax + bound(0) + "," +
                 bound(1) + "," + bound(2) + "):");
          ++p.indent;
          for (unsigned i = 0; i < nRes; ++i)
            p.line(view.blockArguments()[1 + i] + " = " + prefix +
                   std::to_string(i));
        } else if (at == PrintPoint::EndBlock) {
          --p.indent;
        } else if (at == PrintPoint::Leave) {
          for (unsigned i = 0; i < nRes; ++i)
            p.line(view.result(i).str() + " = " + prefix + std::to_string(i));
        }
        return llvm::Error::success();
      });
  rules["scf.yield"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &p) -> llvm::Error {
        if (at == PrintPoint::Enter && view.numOperands()) {
          auto idAttr = view.attrOfType<IntegerAttr>("triton_src.carry_id");
          if (!idAttr)
            return llvm::createStringError(
                "value-carrying scf.yield requires triton_src.carry_id");
          std::string prefix = "_carry_" + std::to_string(idAttr.getInt()) + "_";
          std::vector<std::string> lhs, rhs;
          for (unsigned i = 0; i < view.numOperands(); ++i) {
            lhs.push_back(prefix + std::to_string(i));
            rhs.push_back(view.operand(i).str());
          }
          p.line(llvm::join(lhs, ",") + " = " + llvm::join(rhs, ","));
        }
        return llvm::Error::success();
      });
  for (bool store : {false, true}) {
    rules[store ? "triton_src.store_column" : "triton_src.read_owner_matrix"] =
        PrintRule::local([store](const LocalOpView &view, PrintPoint at,
                                 PrintContext &p) {
          if (at != PrintPoint::Enter)
            return llvm::Error::success();
          auto layout =
              view.attrOfType<DenseI64ArrayAttr>("triton_src.view_strides");
          auto width = std::to_string(
              view.attrOfType<IntegerAttr>("triton_src.side").getInt());
          auto row = "tl.arange(0," + width + ").to(tl.int64)";
          auto addr = view.operand(0).str() + "+" + view.operand(1).str() +
                      "*" + std::to_string(layout[0]) + "+";
          if (store)
            p.line("tl.store(" + addr + row + "*" + std::to_string(layout[1]) +
                   "+tl.cast(" + view.operand(2).str() + ",tl.int64)*" +
                   std::to_string(layout[2]) + "," + view.operand(3).str() +
                   ")");
          else
            p.line(view.result(0).str() + " = tl.load(" + addr + row +
                   "[:,None]*" + std::to_string(layout[1]) + "+" + row +
                   "[None,:]*" + std::to_string(layout[2]) + ")");
          return llvm::Error::success();
        });
  }
  rules["triton_src.column"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &p) {
        if (at == PrintPoint::Enter) {
          auto n = extent(view.resultType(0));
          p.line(view.result(0).str() + " = tl.reshape(tl.gather(" +
                 view.operand(0).str() + ",tl.full((" + n + ",1)," +
                 view.operand(1).str() + ",tl.int32),1),(" + n + ",))");
        }
        return llvm::Error::success();
      });
  rules["triton_src.element"] = PrintRule::format(
      "$r0 = tl.reshape(tl.gather($0,tl.full((1,),$1,tl.int32),0),())");
  rules["triton_src.lower_column"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &p) {
        if (at == PrintPoint::Enter) {
          auto r = "tl.arange(0," + extent(view.resultType(0)) + ")";
          p.line(view.result(0).str() + " = tl.where(" + r + "<" +
                 view.operand(2).str() + ",0.,tl.where(" + r +
                 "==" + view.operand(2).str() + "," + view.operand(1).str() +
                 ",tl.div_rn(" + view.operand(0).str() + "," +
                 view.operand(1).str() + ")))");
        }
        return llvm::Error::success();
      });
  rules["triton_src.outer_product"] =
      PrintRule::format("$r0 = libdevice.mul_rn($0[:,None],$1[None,:])");
}

} // namespace CI_UNIT_NAMESPACE
