#include "intentdsl/levels/execution/block/ir/OwnerSource.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
llvm::Expected<std::string> printColumns(func::FuncOp fn);
namespace {
struct Ordered : intentdsl::block::OwnerSource {
  using OwnerSource::OwnerSource;
  mlir::IntegerAttr nativeMatrixRows() const override {
    return space.getAttrOfType<mlir::IntegerAttr>("triton_src.matrix_rows");
  }
  llvm::StringRef language() override { return "tl"; }
  std::string logicalOwners(std::string fragment, std::string count) override {
    return "tl.gather("+fragment+",tl.arange(0,"+count+"),0)";
  }
  std::string matrixProduct(std::string a, std::string b, bool trans,
                            std::string) override {
    return "tl.dot(" + a + "," + (trans ? "tl.trans(" + b + ")" : b) + ")";
  }
  std::string matrixAccumulate(std::string a, std::string b,
                               std::string c) override {
    return "tl.dot(" + a + "," + b + "," + c + ")";
  }
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
  std::string tileStart() override {
    return "(tl.cast(tl.program_id(0),tl.int64)*ROWS)";
  }
  std::string broadcast(llvm::StringRef v) override {
    return "tl.broadcast_to(tl.to_tensor(" + v.str() + "),(" + fragmentRows() +
           ",))";
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
  std::string gatherScalars(memref::LoadOp op, ArrayRef<std::string> at,
                            StringRef mask) override {
    auto strides = layout(op.getMemref(), "block.strides");
    std::string ptr = name(op.getMemref());
    for (auto [j, a] : llvm::enumerate(at))
      ptr += "+tl.cast(" + a + ",tl.int64)*" + std::to_string(strides[j]);
    return "tl.load(" + ptr + ",mask=" + mask.str() + ",other=0)";
  }
  std::string store(memref::StoreOp op, llvm::StringRef mask) override {
    return "tl.store(" + address(op.getMemref(), op.getIndices()) + "," +
           name(op.getValue()) + ",mask=" + mask.str() + ")";
  }
  std::string cast(llvm::StringRef v, Type t) override {
    if (t.isIndex())
      return "tl.cast(" + v.str() + ",tl.int64)";
    return "tl.cast(" + v.str() + ",tl." +
           (t.isBF16()  ? "bfloat16"
            : t.isF16() ? "float16"
                        : "float32") +
           ")";
  }
  std::string scorePanel() {
    signature();
    auto &bb = space.getRegion(0).front();
    auto &dot = bb.front();
    auto *weights = dot.getNextNode(), *val = weights->getNextNode(),
         *store = val->getNextNode();
    auto q = dot.getOperand(0), k = dot.getOperand(1), v = val->getOperand(1),
         y = store->getOperand(2);
    auto size = [&](Value buf, int axis) {
      return std::to_string(layout(buf, "block.shape")[axis]);
    };
    auto address = [&](Value buf, std::array<std::string, 4> coords) {
      std::string ptr = name(buf);
      for (unsigned i = 0; i < 4; ++i)
        ptr += "+(" + coords[i] + ")*" +
               std::to_string(layout(buf, "block.strides")[i]);
      return ptr;
    };
    line(1, "pid=tl.cast(tl.program_id(0),tl.int64)");
    line(1, "tiles=" + size(q, 2) + "//ROWS");
    line(1, "head=(pid//tiles)%" + size(q, 1));
    line(1, "batch=pid//(tiles*" + size(q, 1) + ")");
    line(1,
         "kv=head//" +
             std::to_string(dot.getAttrOfType<IntegerAttr>("group").getInt()));
    line(1, "row=(pid%tiles)*ROWS+tl.arange(0,ROWS)");
    line(1, "t=tl.arange(0," + size(k, 2) + ").to(tl.int64)");
    line(1, "d=tl.arange(0," + size(k, 3) + ").to(tl.int64)");
    line(1, "f=tl.arange(0," + size(v, 3) + ").to(tl.int64)");
    line(1, "q=tl.load(" +
                address(q, {"batch", "head", "row[:,None]", "d[None,:]"}) +
                ")");
    line(1, "k=tl.load(" +
                address(k, {"batch", "kv", "t[:,None]", "d[None,:]"}) + ")");
    line(1, "scores=tl.dot(q,tl.trans(k))*(" + name(dot.getOperand(2)) +
                "*1.4426950408889634)");
    line(1, "valid=row[:,None]>=t[None,:]");
    line(1, "peak=tl.max(tl.where(valid,scores,-float('inf')),axis=1)");
    line(1, "p=tl.where(valid,tl.exp2(scores-peak[:,None]),0.)");
    line(1, "mass=tl.sum(p,axis=1)");
    line(1, "v=tl.load(" +
                address(v, {"batch", "kv", "t[:,None]", "f[None,:]"}) + ")");
    line(1, "z=tl.dot(p.to(tl.bfloat16),v)*(1./mass[:,None])");
    line(1, "tl.store(" +
                address(y, {"batch", "head", "row[:,None]", "f[None,:]"}) +
                ",z.to(tl.bfloat16))");
    return result;
  }
};
} // namespace
llvm::Error emitOrdered(func::FuncOp fn, llvm::raw_pwrite_stream &out) {
  Ordered writer(fn);
  auto src = fn.front().front().hasAttr("block.score_panel")
                 ? llvm::Expected<std::string>(writer.scorePanel())
             : fn.front().front().hasAttr("block.column_prefixes")
                 ? printColumns(fn)
                 : writer.write();
  if (!src)
    return src.takeError();
  return intentdsl::block::emitOwnerArtifact(
      fn, *src, llvm::json::Object{{"num_warps", 4}, {"num_stages", 1}}, 2,
      out);
}
} // namespace CI_UNIT_NAMESPACE
