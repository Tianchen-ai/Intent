#include "intentdsl/levels/execution/block/ir/OwnerSource.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
struct Ordered : intentdsl::block::OwnerSource {
  using OwnerSource::OwnerSource;
  llvm::StringRef language() override { return "tl"; }
  void signature() override {
    out << "import triton\nimport triton.language as tl\nfrom "
           "triton.language.extra.cuda import libdevice\n@triton.jit\ndef "
           "kernel(";
    for (auto v : fn.getArguments())
      out << name(v) << ",";
    out << "ROWS:tl.constexpr,WIDTH:tl.constexpr):\n";
  }
  std::string tileOwner() override {
    return "tl.program_id(0).to(tl.int64)*ROWS+tl.arange(0,ROWS)";
  }
  std::string broadcast(llvm::StringRef v) override {
    return "tl.broadcast_to(tl.to_tensor(" + v.str() + "),(ROWS,))";
  }
  std::string memberRange(int64_t width) override {
    return "tl.arange(0," + std::to_string(width) + ").to(tl.int64)";
  }
  std::string memberAccess(Operation &op, ArrayRef<std::string> coords,
                           StringRef mask, bool write) override {
    auto view = op.getOperand(write ? 1 : 0);
    auto addr = name(view);
    auto strides = layout(view, "block.strides");
    for (auto [a, c] : llvm::enumerate(coords))
      addr += "+tl.cast(" + c + ",tl.int64)*" + std::to_string(strides[a]);
    return write ? "tl.store(" + addr + "," + name(op.getOperand(0)) +
                       ",mask=" + mask.str() + ")"
                 : "tl.load(" + addr + ",mask=" + mask.str() + ",other=0)";
  }
  std::string address(Value buffer, ValueRange coords) {
    std::string a = name(buffer);
    auto strides = layout(buffer, "block.strides");
    for (auto [i, v] : llvm::enumerate(coords))
      a += "+tl.cast(" + name(v) + ",tl.int64)*" + std::to_string(strides[i]);
    return a;
  }
  std::string load(memref::LoadOp op, llvm::StringRef mask) override {
    return "tl.load(" + address(op.getMemref(), op.getIndices()) +
           "+owner*0,mask=" + mask.str() + ",other=0)";
  }
  std::string store(memref::StoreOp op, llvm::StringRef mask) override {
    return "tl.store(" + address(op.getMemref(), op.getIndices()) + "," +
           name(op.getValue()) + ",mask=" + mask.str() + ")";
  }
  std::string cast(llvm::StringRef v, Type t) override {
    return "tl.cast(" + v.str() + ",tl." +
           (t.isBF16()  ? "bfloat16"
            : t.isF16() ? "float16"
                        : "float32") +
           ")";
  }
  std::string columns() {
    signature();
    auto view = fn.getArgument(0);
    auto n = layout(view, "block.shape")[1];
    auto strides = layout(view, "block.strides");
    out << "    batch=tl.cast(tl.program_id(0),tl.int64)\n    i=tl.arange(0,"
        << n << ").to(tl.int64)\n"
        << "    base=a0+batch*" << strides[0]
        << "\n    matrix=tl.load(base+i[:,None]*" << strides[1] << "+i[None,:]*"
        << strides[2] << ")\n"
        << "    prefix=tl.full((" << n << "," << n
        << "),0.,tl.float32)\n    for c in tl.static_range(" << n << "):\n"
        << "        ix=tl.full((" << n
        << ",1),c,tl.int32)\n        raw=tl.reshape(tl.gather(matrix,ix,1),("
        << n << ",))\n"
        << "        prev=tl.reshape(tl.gather(prefix,ix,1),(" << n
        << ",))\n        diff=libdevice.sub_rn(raw,prev)\n"
        << "        "
           "diag=tl.sqrt_rn(tl.reshape(tl.gather(diff,tl.full((1,),c,tl.int32),"
           "0),()))\n"
        << "        "
           "col=tl.where(i<c,0.,tl.where(i==c,diag,tl.div_rn(diff,diag)))\n"
        << "        tl.store(base+i*" << strides[1] << "+c*" << strides[2]
        << ",col)\n"
        << "        "
           "prefix=libdevice.add_rn(prefix,libdevice.mul_rn(col[:,None],col["
           "None,:]))\n";
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
  return intentdsl::block::emitOwnerArtifact(
      fn, *src, llvm::json::Object{{"num_warps", 4}, {"num_stages", 1}}, 2,
      out);
}
} // namespace CI_UNIT_NAMESPACE
