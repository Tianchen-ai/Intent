#include "ScoreRows.h"
#include "intentdsl/levels/execution/cpu/ir/OwnerTraversal.h"
#include "intentdsl/levels/execution/cpu/ir/RepeatedScore.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/execution/cpu/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringExtras.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
struct OwnerLoop {
  func::FuncOp fn;
  Operation &space;
  llvm::DenseMap<Value, std::string> values;
  std::string source;
  llvm::raw_string_ostream out{source};
  unsigned serial = 0;
  explicit OwnerLoop(func::FuncOp f)
      : fn(f), space(f.front().front().getRegion(0).front().front()) {}
  std::string fresh() { return "v" + std::to_string(serial++); }
  void line(unsigned d, const Twine &s) { out.indent(d * 4) << s << "\n"; }
  std::string name(Value v) { return values.at(v); }
  std::string type(Type t) {
    if (t.isIndex())
      return "Int";
    return t.isF16()        ? "Float16"
           : t.isF32()      ? "Float32"
           : t.isBF16()     ? "BFloat16"
           : t.isInteger(1) ? "Bool"
                            : "Int32";
  }
  int64_t axis(Value v, StringRef key, unsigned a) {
    return fn.getArgAttrOfType<DenseI64ArrayAttr>(
        cast<BlockArgument>(v).getArgNumber(), key)[a];
  }
  std::string dtype(Type t) {
    return t.isBF16()  ? "DType.bfloat16"
           : t.isF16() ? "DType.float16"
                       : "DType.float32";
  }
  std::string at(Value v, ValueRange indices) {
    std::string address = "0";
    for (auto [i, j] : llvm::enumerate(indices))
      address +=
          "+" + name(j) + "*" + std::to_string(axis(v, "cpu.strides", i));
    return name(v) + "[unsafe_offset=" + address + "]";
  }
  llvm::Error body(Block &bb, unsigned depth, Operation *only = nullptr) {
    for (auto &op : bb) {
      if (only && &op != only)
        continue;
      auto id = fresh();
      auto n = op.getName().getStringRef();
      if (n == "scf.yield" || op.getName().stripDialect() == "end")
        continue;
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        for (auto [v, ret, init] :
             llvm::zip(loop.getRegionIterArgs(), loop.getResults(),
                       loop.getInitArgs())) {
          auto var = fresh();
          values[v] = values[ret] = var;
          line(depth, "var " + var + "=" + name(init));
        }
        values[loop.getInductionVar()] = id;
        line(depth, "for " + id + " in range(" + name(loop.getLowerBound()) +
                        "," + name(loop.getUpperBound()) + "," +
                        name(loop.getStep()) + "):");
        if (auto err = body(*loop.getBody(), depth + 1))
          return err;
        SmallVector<std::string> next;
        for (auto v : loop.getBody()->getTerminator()->getOperands()) {
          next.push_back(fresh());
          line(depth + 1, "var " + next.back() + "=" + name(v));
        }
        for (auto [v, r] : llvm::zip(next, loop.getResults()))
          line(depth + 1, name(r) + "=" + v);
        continue;
      }
      if (auto branch = dyn_cast<scf::IfOp>(op)) {
        for (auto r : op.getResults()) {
          auto var = fresh();
          values[r] = var;
          line(depth, "var " + var + ":" + type(r.getType()));
        }
        for (auto &region : op.getRegions()) {
          line(depth, region.getRegionNumber() == 0
                          ? "if " + name(branch.getCondition()) + ":"
                          : "else:");
          if (auto e = body(region.front(), depth + 1))
            return e;
          for (auto [r, v] :
               llvm::zip(op.getResults(),
                         region.front().getTerminator()->getOperands()))
            line(depth + 1, name(r) + "=" + name(v));
        }
        continue;
      }
      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        line(depth, at(store.getMemref(), store.getIndices()) + "=" +
                        name(store.getValue()));
        continue;
      }
      auto arg = [&](unsigned i) { return name(op.getOperand(i)); };
      if (op.hasAttr("cpu.replay_score")) {
        auto idx = name(op.getParentOfType<scf::ForOp>().getInductionVar());
        values[op.getResult(0)] = id;
        line(depth, "var " + id + "=saved[unsafe_offset=" + idx + "]");
        continue;
      }
      std::string e;
      auto kind = op.getName().stripDialect();
      if (kind == "slice_read" || kind == "slice_write" ||
          kind == "lane_splat") {
        bool write = kind == "slice_write";
        auto v = write ? op.getOperand(0) : op.getResult(0);
        auto k = intentdsl::kir::memberExtent(fn, v, "cpu.shape");
        auto width = std::max<uint64_t>(1, llvm::PowerOf2Ceil(k));
        auto ty = "SIMD[" +
                  dtype(cast<RankedTensorType>(v.getType()).getElementType()) +
                  "," + std::to_string(width) + "]";
        if (kind == "lane_splat")
          e = ty + "(" + arg(0) + ")";
        else {
          auto base = op.getOperand(write ? 1 : 0);
          auto ax = op.getAttrOfType<IntegerAttr>("axis").getInt();
          auto coords = op.getOperands().drop_front(write ? 2 : 1);
          std::string addr = "0";
          for (auto [j, c] : llvm::enumerate(coords))
            addr += "+" + name(c) + "*" +
                    std::to_string(
                        axis(base, "cpu.strides", j + (int64_t(j) >= ax)));
          auto step = axis(base, "cpu.strides", ax);
          if (op.hasAttr("mojo_row.contiguous_simd")) {
            auto ptr = name(base) + ".unsafe_offset(" + addr + ")";
            if (write)
              line(depth, ptr + ".unsafe_store(" + arg(0) + ")");
            else {
              values[v] = id;
              line(depth, "var " + id + "=" + ptr + ".unsafe_load[width=" +
                              std::to_string(k) + "]()");
            }
            continue;
          }
          if (!write)
            line(depth, "var " + id + "=" + ty + "(0)");
          auto lane = fresh();
          line(depth, "comptime for " + lane + " in range(" +
                          std::to_string(k) + "):");
          auto ref = name(base) + "[unsafe_offset=" + addr + "+" + lane + "*" +
                     std::to_string(step) + "]";
          line(depth + 1, write ? ref + "=" + arg(0) + "[" + lane + "]"
                                : id + "[" + lane + "]=" + ref);
          if (!write)
            values[v] = id;
          continue;
        }
      } else if (kind == "lane_sum") {
        auto k =
            intentdsl::kir::memberExtent(fn, op.getOperand(0), "cpu.shape");
        if (llvm::isPowerOf2_64(k))
          e = arg(0) + ".reduce_add()";
        else {
          line(depth, "var " + id + "=Float32(0)");
          auto lane = fresh();
          line(depth, "comptime for " + lane + " in range(" +
                          std::to_string(k) + "):");
          line(depth + 1, id + "+=" + arg(0) + "[" + lane + "]");
          values[op.getResult(0)] = id;
          continue;
        }
      } else if (auto load = dyn_cast<memref::LoadOp>(op))
        e = at(load.getMemref(), load.getIndices());
      else if (auto dim = dyn_cast<memref::DimOp>(op))
        e = "Int(" +
            std::to_string(
                axis(dim.getSource(), "cpu.shape", *dim.getConstantIndex())) +
            ")";
      else if (auto c = dyn_cast<arith::ConstantOp>(op)) {
        if (auto val = dyn_cast<FloatAttr>(c.getValue())) {
          llvm::SmallString<32> num;
          val.getValue().toString(num);
          e = "Float32(" + num.str().str() + ")";
          if (val.getValue().isInfinity())
            e = val.getValue().isNegative()
                    ? "bitcast[DType.float32](UInt32(0xff800000))"
                    : "bitcast[DType.float32](UInt32(0x7f800000))";
        } else
          e = type(op.getResult(0).getType()) + "(" +
              std::to_string(cast<IntegerAttr>(c.getValue()).getInt()) + ")";
      } else if (n == "arith.extf" || n == "arith.truncf" ||
                 n == "arith.index_cast") {
        auto t = op.getResult(0).getType();
        if (auto shaped = dyn_cast<RankedTensorType>(t))
          e = arg(0) + ".cast[" + dtype(shaped.getElementType()) + "]()";
        else
          e = type(t) + "(" + arg(0) + ")";
      } else if (n == "arith.maxnumf")
        e = arg(0) + " if isnan(" + arg(1) + ") else (" + arg(1) +
            " if isnan(" + arg(0) + ") else max(" + arg(0) + "," + arg(1) +
            "))";
      else if (n == "math.exp2")
        e = arg(0) + " if isnan(" + arg(0) + ") else exp2(" + arg(0) + ")";
      else if (n == "math.sqrt" || n == "math.exp" ||
               n == "math.log")
        e = op.getName().stripDialect().str() + "(" + arg(0) + ")";
      else {
        auto tok = llvm::StringSwitch<StringRef>(n)
                       .Cases("arith.addf", "arith.addi", "+")
                       .Cases("arith.subf", "arith.subi", "-")
                       .Cases("arith.mulf", "arith.muli", "*")
                       .Case("arith.divf", "/")
                       .Case("arith.floordivsi", "//")
                       .Case("arith.remsi", "%")
                       .Default("");
        if (auto cmp = dyn_cast<arith::CmpIOp>(op))
          tok = cmp.getPredicate() == arith::CmpIPredicate::sge   ? ">="
                : cmp.getPredicate() == arith::CmpIPredicate::slt ? "<"
                                                                  : "==";
        if (tok.empty())
          return llvm::createStringError("missing Mojo ordered instruction " +
                                         n);
        e = arg(0) + tok.str() + arg(1);
      }
      values[op.getResult(0)] = id;
      line(depth, "var " + id + "=" + e);
      if (op.hasAttr("cpu.record_score")) {
        auto idx = name(op.getParentOfType<scf::ForOp>().getInductionVar());
        line(depth, "saved[unsafe_offset=" + idx + "]=" + id);
      }
    }
    return llvm::Error::success();
  }
#include "OwnerBatch.inc"
};
} // namespace
llvm::Error emitOrdered(func::FuncOp fn, llvm::raw_pwrite_stream &output) {
  OwnerLoop w(fn);
  SmallVector<std::string> params, args, cParams;
  for (auto arg : fn.getArguments()) {
    std::string name = "a" + std::to_string(arg.getArgNumber());
    w.values[arg] = name;
    args.push_back(name);
    auto view = dyn_cast<MemRefType>(arg.getType());
    params.push_back(name + ":" +
                     (view ? "Pointer[" + w.type(view.getElementType()) +
                                 ",MutUntrackedOrigin]"
                           : w.type(arg.getType())));
    cParams.push_back(view                    ? "void*"
                      : arg.getType().isF32() ? "float"
                                              : "int32_t");
  }
  w.out
      << "from std.math import exp,sqrt,exp2,log,isnan\nfrom std.memory import "
         "bitcast,unsafe_stack_allocation\n@export(\"owner_task\")\ndef "
         "owner_task("
      << llvm::join(params, ",") << ",lo:Int64,hi:Int64) abi(\"C\"):\n";
  auto &space = w.space;
  if (space.hasAttr("mojo_row.score_microtile"))
    scoreRows(w.out, fn, space);
  else if (auto batch = space.getAttrOfType<IntegerAttr>("cpu.owner_batch")) {
    if (auto err = w.issueBatch(batch.getInt()))
      return err;
  } else {
    if (auto k = w.space.getAttrOfType<IntegerAttr>("cpu.score_cache"))
      w.line(1, "var saved=unsafe_stack_allocation[" +
                    std::to_string(k.getInt()) + ",Float32]()");
    w.line(1, "for owner in range(Int(lo),Int(hi)):");
    auto &body = space.getRegion(0).front();
    auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
    int64_t pitch = 1;
    for (unsigned i : intentdsl::cpu::ownerRadices(space)) {
      auto n = w.axis(space.getOperand(i), "cpu.shape", axes[i]);
      auto id = w.fresh();
      w.values[body.getArgument(i)] = id;
      w.line(2, "var " + id + "=(owner//" + std::to_string(pitch) + ")%" +
                    std::to_string(std::max<int64_t>(1, n)));
      pitch *= std::max<int64_t>(1, n);
    }
    if (auto err = w.body(body, 2))
      return err;
  }
  auto host = intentdsl::cpu::ownerDispatcher(
      fn,
      "extern void owner_task(" + llvm::join(cParams, ",") +
          ",int64_t,int64_t);",
      "owner_task(" + llvm::join(args, ",") + ",lo,hi)");
  auto abi = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!abi)
    return abi.takeError();
  output << llvm::json::Value(llvm::json::Object{
      {"version", 2},
      {"abi", std::move(*abi)},
      {"domain", intentdsl::publishDomain(fn)},
      {"rows",
       llvm::json::Array{llvm::json::Object{
           {"grain", space.getAttrOfType<IntegerAttr>("cpu.grain").getInt()},
           {"columns", 1}}}},
      {"returns_status", true},
      {"source", w.source},
      {"host", host}});
  return llvm::Error::success();
}
} // namespace CI_UNIT_NAMESPACE
