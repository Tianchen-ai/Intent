#include "intentdsl/levels/execution/block/ir/OwnerSource.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
struct Ordered : intentdsl::block::OwnerSource {
  using OwnerSource::OwnerSource;
  llvm::StringRef language() override { return "ct"; }
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
  std::string store(memref::StoreOp op, llvm::StringRef mask) override {
    return "ct.scatter(" + name(op.getMemref()) + "," +
           coords(op.getIndices()) + "," + name(op.getValue()) +
           ",mask=" + mask.str() + ")";
  }
  std::string cast(llvm::StringRef v, Type t) override {
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
};
} // namespace
llvm::Error emitOrdered(func::FuncOp fn, llvm::raw_pwrite_stream &out) {
  Ordered writer(fn);
  auto src = fn.front().front().hasAttr("block.column_prefixes")
                 ? llvm::Expected<std::string>(writer.columns())
                 : writer.write();
  if (!src)
    return src.takeError();
  return intentdsl::block::emitOwnerArtifact(fn, *src, {}, 3, out);
}
} // namespace CI_UNIT_NAMESPACE
