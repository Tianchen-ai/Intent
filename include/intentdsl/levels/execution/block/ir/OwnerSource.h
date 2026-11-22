#pragma once
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringExtras.h"

namespace intentdsl::block {
// Both block-language exits have structured Python control. An exit spells
// only native scalar/tile operations and addressing; the IR supplies the
// loop nest, carry dependencies and publication order.
class OwnerSource {
protected:
  mlir::func::FuncOp fn;
  mlir::Operation &space;
  llvm::DenseMap<mlir::Value, std::string> names;
  std::string result;
  llvm::raw_string_ostream out{result};
  unsigned counter = 0;
  std::string fresh() { return "v" + std::to_string(counter++); }
  void line(unsigned d, const llvm::Twine &s) {
    out.indent(4 * d) << s << "\n";
  }
  virtual std::string tileOwner() = 0;
  virtual std::string load(mlir::memref::LoadOp, llvm::StringRef mask) = 0;
  virtual std::string store(mlir::memref::StoreOp, llvm::StringRef mask) = 0;
  virtual std::string cast(llvm::StringRef, mlir::Type) = 0;
  virtual std::string broadcast(llvm::StringRef) = 0;
  virtual std::string memberAccess(mlir::Operation &,
                                   llvm::ArrayRef<std::string>,
                                   llvm::StringRef mask, bool write) = 0;
  virtual std::string memberRange(int64_t width) = 0;
  virtual llvm::StringRef language() = 0;
  virtual void signature() = 0;
  std::string name(mlir::Value v) { return names.at(v); }
  mlir::DenseI64ArrayAttr layout(mlir::Value v, llvm::StringRef key) {
    return fn.getArgAttrOfType<mlir::DenseI64ArrayAttr>(
        mlir::cast<mlir::BlockArgument>(v).getArgNumber(), key);
  }

public:
  explicit OwnerSource(mlir::func::FuncOp f) : fn(f), space(f.front().front()) {
    for (auto v : fn.getArguments())
      names[v] = "a" + std::to_string(v.getArgNumber());
  }
  llvm::Expected<std::string> write() {
    signature();
    auto &body = space.getRegion(0).front();
    auto axes = space.getAttrOfType<mlir::DenseI64ArrayAttr>("axes");
    line(1, "owner=" + tileOwner());
    line(1, "active=owner<" +
                std::to_string(
                    space.getAttrOfType<mlir::IntegerAttr>("block.owners")
                        .getInt()));
    int64_t pitch = 1;
    for (int i = axes.size() - 1; i >= 0; --i) {
      auto extent = layout(space.getOperand(i), "block.shape")[axes[i]];
      auto id = fresh();
      names[body.getArgument(i)] = id;
      line(1, id + "=(owner//" + std::to_string(pitch) + ")%" +
                  std::to_string(std::max<int64_t>(1, extent)));
      pitch *= std::max<int64_t>(1, extent);
    }
    if (auto e = emit(body, 1, "active"))
      return std::move(e);
    return result;
  }

private:
  llvm::Error emit(mlir::Block &body, unsigned depth, std::string mask) {
    using namespace mlir;
    for (auto &op : body) {
      auto n = op.getName().getStringRef();
      if (n == "scf.yield" || op.getName().stripDialect() == "end")
        continue;
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        SmallVector<std::string> bindings;
        for (auto [arg, v, ret] :
             llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs(),
                       loop.getResults())) {
          auto id = fresh();
          names[arg] = names[ret] = id;
          bindings.push_back(id);
          line(depth,
               id + "=" +
                   (isa<RankedTensorType>(v.getType()) ? name(v)
                                                       : broadcast(name(v))));
        }
        auto iv = fresh();
        names[loop.getInductionVar()] = iv;
        line(depth, "for " + iv + " in range(" + name(loop.getLowerBound()) +
                        "," + name(loop.getUpperBound()) + "," +
                        name(loop.getStep()) + "):");
        if (auto e = emit(*loop.getBody(), depth + 1, mask))
          return e;
        if (!bindings.empty()) {
          SmallVector<std::string> yielded;
          for (auto v : loop.getBody()->getTerminator()->getOperands())
            yielded.push_back(name(v));
          line(depth + 1,
               llvm::join(bindings, ",") + "=" + llvm::join(yielded, ","));
        }
        continue;
      }
      if (auto branch = dyn_cast<scf::IfOp>(op)) {
        auto cond = name(branch.getCondition());
        if (auto e = emit(*branch.thenBlock(), depth, mask + "&(" + cond + ")"))
          return e;
        if (auto e =
                emit(*branch.elseBlock(), depth, mask + "&~(" + cond + ")"))
          return e;
        for (auto [ret, t, f] :
             llvm::zip(op.getResults(), branch.thenYield().getOperands(),
                       branch.elseYield().getOperands())) {
          auto id = fresh();
          names[ret] = id;
          line(depth, id + "=" + language().str() + ".where(" + cond + "," +
                          name(t) + "," + name(f) + ")");
        }
        continue;
      }
      if (auto st = dyn_cast<memref::StoreOp>(op)) {
        line(depth, store(st, mask));
        continue;
      }
      auto arg = [&](unsigned i) { return name(op.getOperand(i)); };
      std::string expr;
      auto kind = op.getName().stripDialect();
      auto lib = language().str();
      auto expand = [&](StringRef v) {
        return lib + ".expand_dims(" + broadcast(v) + ",1)";
      };
      if (kind == "slice_write" || kind == "slice_read" ||
          kind == "lane_splat" || kind == "lane_sum") {
        bool write = kind == "slice_write";
        auto v =
            (write || kind == "lane_sum") ? op.getOperand(0) : op.getResult(0);
        auto k = kir::memberExtent(fn, v, "block.shape");
        auto width = std::max<uint64_t>(1, llvm::PowerOf2Ceil(k));
        auto lanes = lib + ".expand_dims(" + memberRange(width) + ",0)";
        if (kind == "lane_sum")
          expr = lib + ".sum(" + lib + ".where(" + lanes + "<" +
                 std::to_string(k) + "," + arg(0) + ",0.0),axis=1)";
        else if (kind == "lane_splat")
          expr = lib + ".broadcast_to(" + expand(arg(0)) + ",(ROWS," +
                 std::to_string(width) + "))";
        else {
          SmallVector<std::string> coords;
          for (auto c : op.getOperands().drop_front(write ? 2 : 1))
            coords.push_back(expand(name(c)));
          coords.insert(coords.begin() +
                            op.getAttrOfType<IntegerAttr>("axis").getInt(),
                        lanes);
          expr = memberAccess(op, coords,
                              expand(mask) + "&(" + lanes + "<" +
                                  std::to_string(k) + ")",
                              write);
          if (write) {
            line(depth, expr);
            continue;
          }
        }
      } else if (auto d = dyn_cast<memref::DimOp>(op)) {
        auto i = d.getConstantIndex();
        if (!i)
          return llvm::createStringError("expected static dimension axis");
        expr = std::to_string(layout(d.getSource(), "block.shape")[*i]);
      } else if (auto l = dyn_cast<memref::LoadOp>(op))
        expr = load(l, mask);
      else if (auto c = dyn_cast<arith::ConstantOp>(op)) {
        if (auto val = dyn_cast<FloatAttr>(c.getValue())) {
          llvm::SmallString<32> spelling;
          val.getValue().toString(spelling);
          expr = spelling.str().str();
          if (expr.find_first_of(".eE") == std::string::npos)
            expr += ".0";
        } else
          expr = std::to_string(mlir::cast<IntegerAttr>(c.getValue()).getInt());
      } else if (n == "arith.extf" || n == "arith.truncf") {
        auto t = op.getResult(0).getType();
        if (auto shaped = dyn_cast<ShapedType>(t))
          t = shaped.getElementType();
        expr = cast(arg(0), t);
      } else if (n == "math.exp" || n == "math.sqrt")
        expr = language().str() + "." + op.getName().stripDialect().str() +
               "(" + arg(0) + ")";
      else {
        auto token = llvm::StringSwitch<llvm::StringRef>(n)
                         .Cases("arith.addf", "arith.addi", "+")
                         .Cases("arith.subf", "arith.subi", "-")
                         .Cases("arith.mulf", "arith.muli", "*")
                         .Case("arith.divf", "/")
                         .Case("arith.floordivsi", "//")
                         .Case("arith.remsi", "%")
                         .Default("");
        if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
          switch (cmp.getPredicate()) {
          case arith::CmpIPredicate::sge:
            token = ">=";
            break;
          case arith::CmpIPredicate::slt:
            token = "<";
            break;
          case arith::CmpIPredicate::eq:
            token = "==";
            break;
          default:
            break;
          }
        }
        if (token.empty())
          return llvm::createStringError(
              "unsupported ordered tile instruction " + n);
        expr = arg(0) + token.str() + arg(1);
      }
      auto id = fresh();
      names[op.getResult(0)] = id;
      line(depth, id + "=" + expr);
    }
    return llvm::Error::success();
  }
};
} // namespace intentdsl::block
