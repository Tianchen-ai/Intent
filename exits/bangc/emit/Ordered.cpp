#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/storage/dsa/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringExtras.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
struct OwnerCall {
  func::FuncOp fn;
  Operation &space;
  llvm::DenseMap<Value, std::string> values;
  std::string source;
  llvm::raw_string_ostream s{source};
  unsigned counter = 0;
  int64_t slab;
  llvm::DenseMap<Operation *, std::string> streams;
  void transfer(scf::ForOp loop, bool publish) {
    for (auto &access : *loop.getBody()) {
      auto d = access.getAttrOfType<DenseI64ArrayAttr>("dsa.stream");
      if (!d)
        continue;
      auto l = dyn_cast<memref::LoadOp>(access);
      auto w = dyn_cast<memref::StoreOp>(access);
      bool sliced = access.getName().stripDialect() == "slice_read";
      auto view = l ? l.getMemref() : w ? w.getMemref() : access.getOperand(0);
      SmallVector<Value> coords;
      if (sliced) {
        llvm::append_range(coords, access.getOperands().drop_front());
        coords.insert(coords.begin() +
                          access.getAttrOfType<IntegerAttr>("axis").getInt(),
                      Value{});
      } else
        llvm::append_range(coords, l ? l.getIndices() : w.getIndices());
      auto elem = cast<MemRefType>(view.getType()).getElementType();
      auto size = elem.getIntOrFloatBitWidth() / 8;
      auto local = streams.lookup(&access);
      std::string ptr = values.at(view);
      for (auto [axis, i] : llvm::enumerate(coords))
        if (i && int64_t(axis) != d[1])
          ptr += "+(int64_t)" + values.at(i) + "*" +
                 std::to_string(dim(view, "block.strides", axis));
      if (!publish) {
        local = fresh();
        streams[&access] = local;
        s << type(elem) << " *" << local << "=(" << type(elem)
          << "*)(streambuf+" << d[0] << ");\n";
      }
      if (bool(w) != publish)
        continue;
      auto bytes = size * d[3], step = size * dim(view, "block.strides", d[1]);
      if (w && access.getAttrOfType<StringAttr>("bangc.vector_impl") &&
          access.getAttrOfType<StringAttr>("bangc.vector_impl") ==
              "deferred-row-reduction") {
        auto k = count(w.getValue().getDefiningOp()->getOperand(0));
        s << "__bang_transpose(deferred+" << d[2] * k << ",deferred," << d[2]
          << "," << k << ");"
          << "__bang_sumpool(" << local << ",deferred+" << d[2] * k << ","
          << d[2] << "," << k << ",1," << k << ",1,1,1);\n";
      }
      s << "__memcpy(" << (publish ? ptr : local) << ","
        << (publish ? local : ptr) << ",";
      if (bytes == step)
        s << d[2] * bytes << "," << (publish ? "NRAM2GDRAM" : "GDRAM2NRAM")
          << ");\n";
      else
        s << bytes << "," << (publish ? "NRAM2GDRAM" : "GDRAM2NRAM") << ","
          << (publish ? step : bytes) << "," << (publish ? bytes : step) << ","
          << d[2] - 1 << ");\n";
    }
  }
  std::string scalarAddress(Operation &op, Value v, ValueRange at) {
    if (auto d = op.getAttrOfType<DenseI64ArrayAttr>("dsa.stream"))
      return streams.at(&op) + "[" + values.at(at[d[1]]) + "]";
    return address(v, at);
  }
  explicit OwnerCall(func::FuncOp f)
      : fn(f), space(f.front().front()),
        slab(space.getAttrOfType<IntegerAttr>("dsa.owner_bytes").getInt()) {}
  std::string fresh() { return "v" + std::to_string(counter++); }
  static StringRef type(Type t) {
    return t.isIndex()      ? "int64_t"
           : t.isF16()      ? "half"
           : t.isBF16()     ? "unsigned short"
           : t.isF32()      ? "float"
           : t.isInteger(1) ? "bool"
                            : "int32_t";
  }
  std::string slot(Value v, int offset) {
    auto t = cast<RankedTensorType>(v.getType());
    auto var = fresh();
    s << type(t.getElementType()) << " *" << var << "=("
      << type(t.getElementType()) << "*)(storage+" << offset << ");\n";
    return var;
  }
  int64_t count(Value v) {
    return intentdsl::kir::memberExtent(fn, v, "block.shape");
  }
  void copy(StringRef dest, StringRef src, Value v) {
    s << "__memcpy(" << dest << "," << src << ","
      << count(v) * cast<ShapedType>(v.getType()).getElementTypeBitWidth() / 8
      << ",NRAM2NRAM);\n";
  }
  std::string convert(Type from, Type to, std::string name) {
    if (from.isBF16() && to.isF32())
      return "widen_bf16(" + name + ")";
    if (to.isBF16() && from.isF32())
      return "round_bf16(" + name + ")";
    return "(" + type(to).str() + ")" + name;
  }
  int64_t dim(Value v, StringRef key, unsigned axis) {
    return fn.getArgAttrOfType<DenseI64ArrayAttr>(
        cast<BlockArgument>(v).getArgNumber(), key)[axis];
  }
  std::string address(Value v, ValueRange coords) {
    std::string expr = slab ? "local" : values.at(v);
    expr += "[0";
    for (auto [a, i] : llvm::enumerate(coords))
      if (!slab || a)
        expr += "+(int64_t)" + values.at(i) + "*" +
                std::to_string(dim(v, "block.strides", a));
    return expr + "]";
  }
  llvm::Error emit(Block &bb) {
    for (auto &op : bb) {
      auto n = op.getName().getStringRef(), kind = op.getName().stripDialect();
      auto id = fresh();
      if (kind == "end" || n == "scf.yield")
        continue;
      auto arg = [&](unsigned i) { return values.at(op.getOperand(i)); };
      auto implementation = op.getAttrOfType<StringAttr>("bangc.vector_impl");
      StringRef impl = implementation ? implementation.getValue() : StringRef();
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        transfer(loop, false);
        unsigned pos = 0;
        SmallVector<std::string> temps;
        for (auto [i, a, r] :
             llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs(),
                       loop.getResults())) {
          if (isa<RankedTensorType>(r.getType())) {
            auto offsets =
                op.getAttrOfType<DenseI64ArrayAttr>("dsa.member_offsets");
            auto dst = slot(r, offsets[pos++]);
            values[a] = values[r] = dst;
            copy(dst, values.at(i), r);
            temps.push_back(slot(r, offsets[pos++]));
          } else {
            auto v = fresh();
            values[a] = values[r] = v;
            s << type(r.getType()) << " " << v << "=" << values.at(i) << ";\n";
            temps.push_back(fresh());
          }
        }
        values[loop.getInductionVar()] = id;
        s << "for(int64_t " << id << "=" << arg(0) << ";" << id << "<" << arg(1)
          << ";" << id << "+=" << arg(2) << "){\n";
        if (auto err = emit(*loop.getBody()))
          return err;
        if (impl == "single-carry") {
          auto v = loop.getBody()->getTerminator()->getOperand(0);
          copy(values.at(loop.getResult(0)), values.at(v), v);
          s << "}\n";
          transfer(loop, true);
          continue;
        }
        for (auto [v, tmp] :
             llvm::zip(loop.getBody()->getTerminator()->getOperands(), temps)) {
          if (isa<RankedTensorType>(v.getType()))
            copy(tmp, values.at(v), v);
          else
            s << "auto " << tmp << "=" << values.at(v) << ";\n";
        }
        for (auto [r, tmp] : llvm::zip(loop.getResults(), temps)) {
          if (isa<RankedTensorType>(r.getType()))
            copy(values.at(r), tmp, r);
          else
            s << values.at(r) << "=" << tmp << ";\n";
        }
        s << "}\n";
        transfer(loop, true);
        continue;
      }
      if (auto iff = dyn_cast<scf::IfOp>(op)) {
        for (auto r : op.getResults()) {
          values[r] = fresh();
          s << type(r.getType()) << " " << values.at(r) << ";\n";
        }
        for (auto &reg : op.getRegions()) {
          if (!reg.getRegionNumber())
            s << "if(" << arg(0) << ")";
          else
            s << "else";
          s << "{\n";
          if (auto err = emit(reg.front()))
            return err;
          for (auto [r, v] : llvm::zip(
                   op.getResults(), reg.front().getTerminator()->getOperands()))
            s << values.at(r) << "=" << values.at(v) << ";\n";
          s << "}\n";
        }
        continue;
      }
      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        if (impl != "deferred-row-reduction")
          s << scalarAddress(op, store.getMemref(), store.getIndices()) << "="
            << arg(0) << ";\n";
        continue;
      }
      if (kind == "slice_write" ||
          (op.getNumResults() &&
           isa<RankedTensorType>(op.getResult(0).getType()))) {
        bool write = kind == "slice_write";
        auto v = write ? op.getOperand(0) : op.getResult(0);
        auto lanes = count(v);
        auto t = cast<ShapedType>(v.getType()).getElementType();
        if (impl == "broadcast-multiplier") {
          values[v] = arg(0);
          continue;
        }
        if (impl == "stream-view") {
          auto iv = cast<scf::ForOp>(op.getParentOp()).getInductionVar();
          s << "auto *" << id << "=" << streams.at(&op) << "+" << values.at(iv)
            << "*" << lanes << ";\n";
          values[v] = id;
          continue;
        }
        std::string local =
            write ? arg(0)
            : impl == "deferred-row-reduction"
                ? "(deferred+" +
                      values.at(cast<scf::ForOp>(op.getParentOp())
                                    .getInductionVar()) +
                      "*" + std::to_string(lanes) + ")"
                : slot(v, op.getAttrOfType<DenseI64ArrayAttr>(
                              "dsa.member_offsets")[0]);
        if (!write)
          values[v] = local;
        if (kind == "slice_read" || write) {
          auto base = op.getOperand(write ? 1 : 0);
          auto axis = op.getAttrOfType<IntegerAttr>("axis").getInt();
          std::string offset = "0";
          for (auto [i, c] :
               llvm::enumerate(op.getOperands().drop_front(write ? 2 : 1)))
            offset += "+(int64_t)" + values.at(c) + "*" +
                      std::to_string(
                          dim(base, "block.strides", i + (int64_t(i) >= axis)));
          auto ptr = values.at(base) + "+(" + offset + ")";
          auto step = dim(base, "block.strides", axis);
          if (auto stream = op.getAttrOfType<DenseI64ArrayAttr>("dsa.stream")) {
            auto iv = cast<scf::ForOp>(op.getParentOp()).getInductionVar();
            s << "__memcpy(" << local << "," << streams.at(&op) << "+"
              << values.at(iv) << "*" << lanes << ","
              << lanes * t.getIntOrFloatBitWidth() / 8 << ",NRAM2NRAM);\n";
          } else if (step == 1)
            s << "__memcpy(" << (write ? ptr : local) << ","
              << (write ? local : ptr) << ","
              << lanes * t.getIntOrFloatBitWidth() / 8 << ","
              << (write ? "NRAM2GDRAM" : "GDRAM2NRAM") << ");\n";
          else {
            auto ref = "(" + ptr + ")[j*" + std::to_string(step) + "]";
            s << "for(int64_t j=0;j<" << lanes << ";++j)"
              << (write ? ref : local + "[j]") << "="
              << (write ? local + "[j]" : ref) << ";\n";
          }
        } else if (kind == "lane_splat")
          s << "for(int j=0;j<" << lanes << ";++j)" << local << "[j]=" << arg(0)
            << ";\n";
        else if (impl == "widen-bits")
          s << "widen_bf16_tile(" << local << "," << arg(0) << "," << lanes
            << ");\n";
        else if (n == "arith.extf" || n == "arith.truncf")
          s << "for(int j=0;j<" << lanes << ";++j)" << local << "[j]="
            << convert(cast<ShapedType>(op.getOperand(0).getType())
                           .getElementType(),
                       t, arg(0) + "[j]")
            << ";\n";
        else {
          auto tok = llvm::StringSwitch<StringRef>(n)
                         .Case("arith.addf", "+")
                         .Case("arith.subf", "-")
                         .Case("arith.mulf", "*")
                         .Case("arith.divf", "/")
                         .Default("");
          if (tok.empty())
            return llvm::createStringError("unsupported lane instruction " + n);
          auto splat = [&](unsigned i) {
            auto *def = op.getOperand(i).getDefiningOp();
            auto tag = def ? def->getAttrOfType<StringAttr>("bangc.vector_impl")
                           : StringAttr();
            return tag && tag.getValue() == "broadcast-multiplier";
          };
          if (tok == "*" && splat(0) && splat(1))
            s << "__bang_write_value(" << local << "," << lanes << "," << arg(0)
              << "*" << arg(1) << ");\n";
          else if (tok == "*" && (splat(0) || splat(1)))
            s << "__bang_mul_scalar(" << local << "," << arg(splat(0) ? 1 : 0)
              << "," << arg(splat(0) ? 0 : 1) << "," << lanes << ");\n";
          else if (t.isF32() && lanes % 32 == 0 &&
                   (tok == "+" || tok == "-" || tok == "*"))
            s << "__bang_"
              << (tok == "+"   ? "add"
                  : tok == "-" ? "sub"
                               : "mul")
              << "(" << local << "," << arg(0) << "," << arg(1) << "," << lanes
              << ");\n";
          else
            s << "for(int j=0;j<" << lanes << ";++j)" << local
              << "[j]=" << arg(0) << "[j]" << tok << arg(1) << "[j];\n";
        }
        continue;
      }
      if (kind == "lane_sum") {
        if (impl == "deferred-row-reduction")
          continue;
        values[op.getResult(0)] = id;
        if (impl == "reduce32-reuse-product")
          s << "__bang_reduce_sum(" << arg(0) << "," << arg(0)
            << ",32);__sync_compute();float " << id << "=((volatile float*)"
            << arg(0) << ")[0];\n";
        else
          s << "float " << id << "=0;for(int j=0;j<" << count(op.getOperand(0))
            << ";++j)" << id << "+=" << arg(0) << "[j];\n";
        continue;
      }
      std::string e;
      if (auto l = dyn_cast<memref::LoadOp>(op))
        e = scalarAddress(op, l.getMemref(), l.getIndices());
      else if (auto d = dyn_cast<memref::DimOp>(op))
        e = std::to_string(
            dim(d.getSource(), "block.shape", *d.getConstantIndex()));
      else if (auto c = dyn_cast<arith::ConstantOp>(op)) {
        if (auto v = dyn_cast<FloatAttr>(c.getValue())) {
          llvm::SmallString<32> n;
          v.getValue().toString(n);
          e = n.str().str();
        } else
          e = std::to_string(cast<IntegerAttr>(c.getValue()).getInt());
      } else if (n == "arith.extf" || n == "arith.truncf")
        e = convert(op.getOperand(0).getType(), op.getResult(0).getType(),
                    arg(0));
      else if (n == "math.sqrt" || n == "math.exp")
        e = kind.str() + "f(" + arg(0) + ")";
      else {
        auto tok = llvm::StringSwitch<StringRef>(n)
                       .Cases("arith.addf", "arith.addi", "+")
                       .Cases("arith.subf", "arith.subi", "-")
                       .Cases("arith.mulf", "arith.muli", "*")
                       .Cases("arith.divf", "arith.floordivsi", "/")
                       .Case("arith.remsi", "%")
                       .Default("");
        if (auto cmp = dyn_cast<arith::CmpIOp>(op))
          tok = cmp.getPredicate() == arith::CmpIPredicate::sge   ? ">="
                : cmp.getPredicate() == arith::CmpIPredicate::slt ? "<"
                                                                  : "==";
        if (tok.empty())
          return llvm::createStringError("unhandled ordered BANG instruction " +
                                         n);
        e = arg(0) + tok.str() + arg(1);
      }
      values[op.getResult(0)] = id;
      s << type(op.getResult(0).getType()) << " " << id << "=" << e << ";\n";
    }
    return llvm::Error::success();
  }
};
} // namespace
llvm::Error emitOrdered(func::FuncOp fn, llvm::raw_pwrite_stream &out) {
  OwnerCall w(fn);
  auto &space = w.space;
  auto &bb = space.getRegion(0).front();
  SmallVector<std::string> decl, args;
  for (auto [i, v] : llvm::enumerate(fn.getArguments())) {
    auto name = "a" + std::to_string(i);
    auto view = dyn_cast<MemRefType>(v.getType());
    args.push_back(name);
    w.values[v] = name;
    decl.push_back(
        OwnerCall::type(view ? view.getElementType() : v.getType()).str() +
        (view ? " *" : " ") + name);
    if (view)
      for (auto kind : {"d", "s"})
        for (int j = 0; j < view.getRank(); ++j) {
          args.push_back(name + kind + std::to_string(j));
          decl.push_back("int64_t " + args.back());
        }
  }
  for (auto name : {"rows", "cols", "grid"}) {
    args.push_back(name);
    decl.push_back("int64_t " + std::string(name));
  }
  auto params = llvm::join(decl, ",");
  w.s << "#include <bang.h>\n#include <cnrt.h>\n#include <stdint.h>\n#include "
         "<math.h>\n";
  w.s <<
#include "RowMath.inc"
      ;
  w.s << "__mlu_global__ void ordered(" << params << "){\n";
  auto workspace =
      space.getAttrOfType<IntegerAttr>("dsa.member_bytes").getInt();
  if (workspace)
    w.s << "__nram__ unsigned char storage[" << workspace << "];\n";
  if (auto bytes = space.getAttrOfType<IntegerAttr>("dsa.stream_bytes"))
    w.s << "__nram__ unsigned char streambuf[" << bytes.getInt() << "];\n";
  if (auto bytes = space.getAttrOfType<IntegerAttr>("bangc.deferred_bytes");
      bytes && bytes.getInt())
    w.s << "__nram__ float deferred[" << bytes.getInt() / 4 << "];\n";
  if (w.slab)
    w.s << "__nram__ "
        << OwnerCall::type(
               cast<MemRefType>(fn.getArgument(0).getType()).getElementType())
        << " local["
        << w.slab / (cast<MemRefType>(fn.getArgument(0).getType())
                         .getElementTypeBitWidth() /
                     8)
        << "];\n";
  auto tile = space.getAttrOfType<IntegerAttr>("block.owner_tile").getInt();
  auto owners = space.getAttrOfType<IntegerAttr>("block.owners").getInt();
  w.s << "for(int64_t job=taskId;job<grid;job+=taskDim){for(int64_t owner=job*"
      << tile << ";owner<" << owners << " && owner-job*" << tile << "<" << tile
      << ";++owner){\n";
  auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
  int64_t pitch = 1;
  for (int i = axes.size() - 1; i >= 0; --i) {
    auto n = std::max<int64_t>(
        1, w.dim(space.getOperand(i), "block.shape", axes[i]));
    auto id = w.fresh();
    w.values[bb.getArgument(i)] = id;
    w.s << "int64_t " << id << "=(owner/" << pitch << ")%" << n << ";\n";
    pitch *= n;
  }
  if (w.slab)
    w.s << "__memcpy(local,a0+owner*a0s0," << w.slab << ",GDRAM2NRAM);\n";
  if (auto err = w.emit(bb))
    return err;
  if (w.slab)
    w.s << "__memcpy(a0+owner*a0s0,local," << w.slab << ",NRAM2GDRAM);\n";
  w.s << "}}}\nextern \"C\" int intent_enqueue(" << params
      << ",int cores,void *queue){if(!grid)return 0;cnrtDim3_t "
         "d={(unsigned)(grid<cores?grid:cores),1,1};"
         "ordered<<<d,cnrtFuncTypeBlock,(cnrtQueue_t)queue>>>("
      << llvm::join(args, ",") << ");return (int)cnrtGetLastError();}\n";
  auto abi = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!abi)
    return abi.takeError();
  out << llvm::json::Value(llvm::json::Object{
      {"version", 2},
      {"source", w.source},
      {"abi", std::move(*abi)},
      {"domain", intentdsl::publishDomain(fn)},
      {"grid",
       llvm::json::Array{int64_t(llvm::divideCeil(owners, tile)), 1, 1}},
      {"guard", llvm::json::Object{{"min_width", 0}, {"max_width", nullptr}}},
      {"rows",
       llvm::json::Array{llvm::json::Object{
           {"id", 0},
           {"rows", 1},
           {"width", 1},
           {"pitch", 1},
           {"bytes", w.slab},
           {"tasks",
            space.getAttrOfType<IntegerAttr>("bangc.workers").getInt()}}}}});
  return llvm::Error::success();
}
} // namespace CI_UNIT_NAMESPACE
