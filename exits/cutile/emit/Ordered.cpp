#include "intentdsl/levels/execution/block/ir/OwnerSource.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
struct Ordered : intentdsl::block::OwnerSource {
  using OwnerSource::OwnerSource;
  llvm::StringRef language() override { return "ct"; }
  std::string matrixProduct(std::string a, std::string b, bool trans,
                            std::string n) override {
    return "ct.mma(" + a + "," + (trans ? "ct.transpose(" + b + ")" : b) +
           ",ct.full((ROWS," + n + "),0.,ct.float32))";
  }
  std::string matrixAccumulate(std::string a, std::string b,
                               std::string c) override {
    return "ct.mma(" + a + "," + b + "," + c + ")";
  }
  void signature() override {
    out << "import cuda.tile as ct\n@ct.kernel\ndef kernel(";
    for (auto v : fn.getArguments()) {
      out << name(v);
      if (isa<MemRefType>(v.getType()) &&
          fn->getAttrOfType<IntegerAttr>("ct_src.index_bits").getInt() == 64)
        out << ":ct.IndexedWithInt64[ct.Array]";
      out << ",";
    }
    out << "ROWS:ct.Constant[int],WIDTH:ct.Constant[int]):\n";
  }
  std::string tileOwner() override {
    return "ct.astype(ct.bid(0),ct.int64)*ROWS+ct.arange(ROWS,dtype=ct.int64)";
  }
  std::string tileStart() override {
    return "(ct.astype(ct.bid(0),ct.int64)*ROWS)";
  }
  std::string broadcast(llvm::StringRef v) override {
    return "ct.broadcast_to(" + v.str() + ",(ROWS,))";
  }
  std::string memberRange(int64_t width) override {
    return "ct.arange(" + std::to_string(width) + ",dtype=ct.int64)";
  }
  std::string memberAccess(Operation &op, ArrayRef<std::string> coords,
                           StringRef mask, bool write) override {
    auto buf = name(op.getOperand(write ? 1 : 0));
    auto at = "(" + llvm::join(coords, ",") + ",)";
    return write ? "ct.scatter(" + buf + "," + at + "," +
                       name(op.getOperand(0)) + ",mask=" + mask.str() + ")"
                 : "ct.gather(" + buf + "," + at + ",mask=" + mask.str() +
                       ",padding_value=0)";
  }
  std::string coords(ValueRange indices) {
    SmallVector<std::string> parts;
    for (auto v : indices)
      parts.push_back(name(v));
    return "(" + llvm::join(parts, ",") + ",)";
  }
  std::string load(memref::LoadOp op, llvm::StringRef mask) override {
    return "ct.gather(" + name(op.getMemref()) + "," + coords(op.getIndices()) +
           ",mask=" + mask.str() + ",padding_value=0)";
  }
  std::string gatherScalars(memref::LoadOp op, ArrayRef<std::string> at,
                            StringRef mask) override {
    return "ct.gather(" + name(op.getMemref()) + ",(" + llvm::join(at, ",") +
           ",),mask=" + mask.str() + ",padding_value=0)";
  }
  std::string loadMemberPanel(Operation &op, ArrayRef<std::string> outer,
                              std::string base, int64_t span, int64_t lanes,
                              std::string live) override {
    SmallVector<std::string> at(outer), dims(outer.size(), "1");
    at.push_back(base + "//" + std::to_string(span));
    at.push_back("0");
    dims.push_back(std::to_string(span));
    dims.push_back(std::to_string(lanes));
    auto panel = "ct.reshape(ct.load(" + name(op.getOperand(0)) + ",(" +
                 llvm::join(at, ",") + "),shape=(" + llvm::join(dims, ",") +
                 "),padding_mode=ct.PaddingMode.ZERO,allow_tma=False),(" +
                 std::to_string(span) + "," + std::to_string(lanes) + "))";
    return "ct.where(" + live + "," + panel + ",0)";
  }
  std::string ownerPanelAccess(Operation &op, bool write) override {
    if (!op.hasAttr("ct_src.owner_rectangle"))
      return {};
    auto view = op.getOperand(write ? 1 : 0);
    auto at = op.getOperands().drop_front(write ? 2 : 1);
    auto n = std::to_string(layout(view, "block.shape").asArrayRef().back());
    SmallVector<std::string> indices, shape;
    for (auto c : at.drop_back()) {
      indices.push_back(name(c));
      shape.push_back("1");
    }
    auto rows = std::to_string(
        space.getAttrOfType<IntegerAttr>("block.owner_tile").getInt());
    indices.push_back("ct.min(" + broadcast(name(at.back())) +
                      ",axis=0)//ROWS");
    indices.push_back("0");
    shape.push_back(rows);
    shape.push_back(n);
    auto args = name(view) + ",(" + llvm::join(indices, ",") + ")",
         dims = "(" + llvm::join(shape, ",") + ")";
    if (write)
      return "ct.store(" + args + ",ct.reshape(" + name(op.getOperand(0)) +
             "," + dims + "),allow_tma=False)";
    return "ct.reshape(ct.load(" + args + ",shape=" + dims +
           ",allow_tma=False),(ROWS," + n + "))";
  }
  std::string store(memref::StoreOp op, llvm::StringRef mask) override {
    return "ct.scatter(" + name(op.getMemref()) + "," +
           coords(op.getIndices()) + "," + name(op.getValue()) +
           ",mask=" + mask.str() + ")";
  }
  std::string cast(llvm::StringRef v, Type t) override {
    if (t.isIndex())
      return "ct.astype(" + v.str() + ",ct.int64)";
    return "ct.astype(" + v.str() + ",ct." +
           (t.isBF16()  ? "bfloat16"
            : t.isF16() ? "float16"
                        : "float32") +
           ")";
  }
  std::string columns() {
    signature();
    auto n = layout(fn.getArgument(0), "block.shape")[1];
    out << "    batch=ct.bid(0)\n    i=ct.arange(" << n << ",dtype=ct.int32)\n"
        << "    matrix=ct.reshape(ct.load(a0,(batch,0,0),shape=(1," << n << ","
        << n << ")),(" << n << "," << n << "))\n"
        << "    prefix=ct.full((" << n << "," << n
        << "),0.,ct.float32)\n    for c in range(" << n << "):\n"
        << "        a=ct.reshape(ct.extract(matrix,(0,c),shape=(" << n
        << ",1)),(" << n << ",))\n"
        << "        old=ct.reshape(ct.extract(prefix,(0,c),shape=(" << n
        << ",1)),(" << n << ",))\n"
        << "        diff=ct.sub(a,old,rounding_mode=ct.RoundingMode.RN)\n"
        << "        "
           "diag=ct.sqrt(ct.reshape(ct.extract(diff,(c,),shape=(1,)),()))\n"
        << "        "
           "col=ct.where(i<c,0.,ct.where(i==c,diag,ct.truediv(diff,diag,"
           "rounding_mode=ct.RoundingMode.RN)))\n"
        << "        ct.scatter(a0,(batch,i,c),col)\n        "
           "prefix=ct.add(prefix,ct.mul(ct.expand_dims(col,1),ct.expand_dims("
           "col,0),rounding_mode=ct.RoundingMode.RN),rounding_mode=ct."
           "RoundingMode.RN)\n";
    return result;
  }
  std::string scorePanel() {
    signature();
    auto &dot = space.getRegion(0).front().front();
    auto *val = dot.getNextNode()->getNextNode(), *store = val->getNextNode();
    auto q = dot.getOperand(0), k = dot.getOperand(1), v = val->getOperand(1),
         y = store->getOperand(2);
    auto size = [&](Value buf, int axis) {
      return std::to_string(layout(buf, "block.shape")[axis]);
    };
    auto depth = size(k, 3), keys = size(k, 2), features = size(v, 3);
    line(1, "pid=ct.bid(0)");
    line(1, "tiles=" + size(q, 2) + "//ROWS");
    line(1, "tile=pid%tiles");
    line(1, "head=(pid//tiles)%" + size(q, 1));
    line(1, "batch=pid//(tiles*" + size(q, 1) + ")");
    line(1,
         "kv=head//" +
             std::to_string(dot.getAttrOfType<IntegerAttr>("group").getInt()));
    auto load = [&](Value buf, std::string idx, std::string shape) {
      return "ct.reshape(ct.load(" + name(buf) + ",(batch," + idx +
             ",0),shape=(1,1," + shape + ")),(" + shape + "))";
    };
    line(1, "q=" + load(q, "head,tile", "ROWS," + depth));
    line(1, "k=" + load(k, "kv,0", keys + "," + depth));
    line(1, "scores=ct.mma(q,ct.transpose(k),ct.full((ROWS," + keys +
                "),0.,ct.float32))*(" + name(dot.getOperand(2)) +
                "*1.4426950408889634)");
    line(1, "r=tile*ROWS+ct.arange(ROWS,dtype=ct.int32)");
    line(1, "t=ct.arange(" + keys + ",dtype=ct.int32)");
    line(1, "valid=ct.expand_dims(r,1)>=ct.expand_dims(t,0)");
    line(1, "peak=ct.max(ct.where(valid,scores,-float('inf')),axis=1)");
    line(1, "p=ct.where(valid,ct.exp2(scores-ct.expand_dims(peak,1)),0.)");
    line(1, "mass=ct.sum(p,axis=1)");
    line(1, "v=" + load(v, "kv,0", keys + "," + features));
    line(1, "z=ct.mma(ct.astype(p,ct.bfloat16),v,ct.full((ROWS," + features +
                "),0.,ct.float32))*ct.expand_dims(1./mass,1)");
    line(1, "ct.store(" + name(y) +
                ",(batch,head,tile,0),ct.reshape(ct.astype(z,ct.bfloat16),(1,1,"
                "ROWS," +
                features + ")))");
    return result;
  }
};
} // namespace
llvm::Error emitOrdered(func::FuncOp fn, llvm::raw_pwrite_stream &out) {
  Ordered writer(fn);
  auto src = fn.front().front().hasAttr("block.score_panel")
                 ? llvm::Expected<std::string>(writer.scorePanel())
             : fn.front().front().hasAttr("block.column_prefixes")
                 ? llvm::Expected<std::string>(writer.columns())
                 : writer.write();
  if (!src)
    return src.takeError();
  return intentdsl::block::emitOwnerArtifact(fn, *src, {}, 3, out);
}
} // namespace CI_UNIT_NAMESPACE
