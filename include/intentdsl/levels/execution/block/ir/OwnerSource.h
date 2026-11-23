#pragma once
#include "intentdsl/levels/execution/block/ir/FoldIssue.h"
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
  std::string fragmentRows() const {
    auto w = space.getAttrOfType<mlir::IntegerAttr>("block.matrix_rows");
    return w ? std::to_string(w.getInt()) : "ROWS";
  }
  void line(unsigned d, const llvm::Twine &s) {
    out.indent(4 * d) << s << "\n";
  }
  virtual std::string tileOwner() = 0;
  virtual bool countedPanels(mlir::scf::ForOp) { return false; }
  virtual std::string tileStart() = 0;
  virtual std::string load(mlir::memref::LoadOp, llvm::StringRef mask) = 0;
  virtual std::string gatherScalars(mlir::memref::LoadOp,
                                    llvm::ArrayRef<std::string>,
                                    llvm::StringRef) = 0;
  virtual std::string loadMemberPanel(mlir::Operation &,
                                      llvm::ArrayRef<std::string>, std::string,
                                      int64_t, int64_t, std::string) {
    return {};
  }
  virtual std::string ownerPanelAccess(mlir::Operation &, bool) { return {}; }
  virtual std::string loadPagePanel(mlir::Operation &,
                                    llvm::ArrayRef<std::string>, std::string,
                                    int64_t, int64_t, std::string) {
    return {};
  }
  virtual std::string store(mlir::memref::StoreOp, llvm::StringRef mask) = 0;
  virtual std::string cast(llvm::StringRef, mlir::Type) = 0;
  virtual std::string broadcast(llvm::StringRef) = 0;
  virtual std::string memberAccess(mlir::Operation &,
                                   llvm::ArrayRef<std::string>,
                                   llvm::StringRef mask, bool write) = 0;
  virtual std::string memberRange(int64_t width) = 0;
  virtual llvm::StringRef language() = 0;
  virtual std::string matrixProduct(std::string, std::string, bool,
                                    std::string) = 0;
  virtual std::string matrixAccumulate(std::string, std::string,
                                       std::string) = 0;
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
    auto nativeRows =
        space.getAttrOfType<mlir::IntegerAttr>("block.matrix_rows");
    auto logical =
        space.getAttrOfType<mlir::IntegerAttr>("block.owner_tile").getInt();
    if (nativeRows) {
      line(1, "owner=" + tileStart() + "+" + memberRange(nativeRows.getInt()) +
                  "%" + std::to_string(logical));
      line(1, "active=" + memberRange(nativeRows.getInt()) + "<" +
                  std::to_string(logical));
    } else {
      line(1, "owner=" + tileOwner());
      line(1, "active=owner<" +
                  std::to_string(
                      space.getAttrOfType<mlir::IntegerAttr>("block.owners")
                          .getInt()));
    }
    int64_t pitch = 1;
    llvm::SmallVector<unsigned> order;
    auto group = space.getAttrOfType<mlir::IntegerAttr>("block.group_axis");
    if (auto pair = space.getAttrOfType<mlir::IntegerAttr>("block.pair_axis"))
      group = pair;
    if (group)
      order.push_back(group.getInt());
    for (int i = axes.size() - 1; i >= 0; --i)
      if (!group || group.getInt() != i)
        order.push_back(i);
    for (auto i : order) {
      auto extent = layout(space.getOperand(i), "block.shape")[axes[i]];
      auto id = fresh();
      names[body.getArgument(i)] = id;
      auto width =
          space.getAttrOfType<mlir::IntegerAttr>("block.owner_tile").getInt();
      auto base = group && pitch % width == 0 ? tileStart() : "owner";
      line(1, id + "=(" + base + "//" + std::to_string(pitch) + ")%" +
                  std::to_string(std::max<int64_t>(1, extent)));
      pitch *= std::max<int64_t>(1, extent);
    }
    if (auto e = emit(body, 1, "active"))
      return std::move(e);
    return result;
  }

private:
#include "intentdsl/levels/execution/block/ir/FoldSource.inc"
#include "intentdsl/levels/execution/block/ir/PairSource.inc"
  bool uniformBound(mlir::Value v) const {
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
      if (auto loop =
              mlir::dyn_cast<mlir::scf::ForOp>(arg.getOwner()->getParentOp()))
        return arg == loop.getInductionVar() &&
               uniformBound(loop.getLowerBound());
      return arg.getOwner() == &fn->getRegion(0).front();
    }
    auto *op = v.getDefiningOp();
    if (mlir::isa<mlir::arith::ConstantOp, mlir::memref::DimOp>(op))
      return true;
    return op->getName().getDialectNamespace() == "arith" &&
           llvm::all_of(op->getOperands(),
                        [&](auto x) { return uniformBound(x); });
  }
  llvm::Error emit(mlir::Block &body, unsigned depth, std::string mask,
                   mlir::Operation *replay = nullptr) {
    using namespace mlir;
    for (auto &op : body) {
      if (replay && &op != replay)
        continue;
      auto n = op.getName().getStringRef();
      if (n == "scf.yield" || op.getName().stripDialect() == "end")
        continue;
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        if (auto panel =
                op.getAttrOfType<IntegerAttr>("block.contribution_width");
            panel && !replay) {
          auto err = space.hasAttr("block.pair_axis")
                         ? emitPairFold(loop, depth, panel.getInt())
                         : emitFold(loop, depth, panel.getInt());
          if (err)
            return err;
          continue;
        }
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
        auto lo = name(loop.getLowerBound()), hi = name(loop.getUpperBound());
        auto step = name(loop.getStep()), active = mask;
        bool ragged = !uniformBound(loop.getLowerBound()) ||
                      !uniformBound(loop.getUpperBound());
        std::string offset;
        if (ragged) {
          auto lib = language().str(), span = fresh();
          offset = fresh();
          line(depth, span + "=" + lib + ".max(" + lib + ".where(" + mask +
                          "," + hi + "-" + lo + ",0),axis=0)");
          auto indexType = mlir::IndexType::get(op.getContext());
          line(depth, offset + "=" + cast("0", indexType));
          line(depth, "while " + offset + "<" + span + ":");
          line(depth + 1, iv + "=" + lo + "+" + offset);
          active = "(" + mask + ")&(" + iv + "<" + hi + ")";
        } else
          line(depth,
               "for " + iv + " in range(" + lo + "," + hi + "," + step + "):");
        if (auto e = emit(*loop.getBody(), depth + 1, active))
          return e;
        if (!bindings.empty()) {
          SmallVector<std::string> yielded;
          for (auto [v, old] : llvm::zip(
                   loop.getBody()->getTerminator()->getOperands(), bindings)) {
            auto predicate =
                isa<RankedTensorType>(v.getType())
                    ? language().str() + ".expand_dims(" + active + ",1)"
                    : active;
            yielded.push_back(ragged
                                  ? language().str() + ".where(" + predicate +
                                        "," + name(v) + "," + old + ")"
                                  : name(v));
          }
          line(depth + 1,
               llvm::join(bindings, ",") + "=" + llvm::join(yielded, ","));
        }
        if (ragged)
          line(depth + 1, offset + "=" + offset + "+" + step);
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
          expr = lib + ".broadcast_to(" + expand(arg(0)) + ",(" +
                 fragmentRows() + "," + std::to_string(width) + "))";
        else {
          expr = ownerPanelAccess(op, write);
          if (!expr.empty()) {
            if (write)
              line(depth, expr);
            else {
              auto id = fresh();
              names[op.getResult(0)] = id;
              line(depth, id + "=" + expr);
            }
            continue;
          }
          SmallVector<std::string> coords;
          bool shared = op.hasAttr("block.broadcast_read");
          auto one = [&](StringRef x) {
            return lib + ".min(" + broadcast(x) + ",axis=0)";
          };
          for (auto c : op.getOperands().drop_front(write ? 2 : 1))
            coords.push_back(shared ? one(name(c)) : expand(name(c)));
          coords.insert(coords.begin() +
                            op.getAttrOfType<IntegerAttr>("axis").getInt(),
                        lanes);
          auto valid = "(" + lanes + "<" + std::to_string(k) + ")";
          expr = memberAccess(
              op, coords,
              (shared ? "(" + one(mask) + "!=0)" : expand(mask)) + "&" + valid,
              write);
          if (shared)
            expr = lib + ".broadcast_to(" + expr + ",(" + fragmentRows() + "," +
                   std::to_string(width) + "))";
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
          if (val.getValue().isInfinity())
            expr =
                val.getValue().isNegative() ? "-float('inf')" : "float('inf')";
          else if (expr.find_first_of(".eE") == std::string::npos)
            expr += ".0";
        } else
          expr = std::to_string(mlir::cast<IntegerAttr>(c.getValue()).getInt());
      } else if (n == "arith.extf" || n == "arith.truncf" ||
                 n == "arith.index_cast") {
        auto t = op.getResult(0).getType();
        if (auto shaped = dyn_cast<ShapedType>(t))
          t = shaped.getElementType();
        expr = cast(arg(0), t);
      } else if (n == "arith.maxnumf")
        expr = language().str() + ".maximum(" + arg(0) + "," + arg(1) + ")";
      else if (n == "math.exp" || n == "math.sqrt" || n == "math.exp2" ||
               n == "math.log")
        expr = language().str() + "." + op.getName().stripDialect().str() +
               "(" + arg(0) + ")";
      else if(n=="arith.minsi")
        expr=lib+".minimum("+arg(0)+","+arg(1)+")";
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
