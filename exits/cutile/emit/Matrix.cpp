#include "intentdsl/levels/execution/block/ir/MatrixAccess.h"
#include "llvm/ADT/StringExtras.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
struct Writer : intentdsl::block::MatrixAccess {
  using MatrixAccess::MatrixAccess;
  std::string text;
  llvm::raw_string_ostream s{text};
  std::string coords(Value v, int k, int batch, StringRef member,
                     StringRef inner) {
    SmallVector<std::string> indices;
    for (int i = 0; i < rank(v); ++i)
      indices.push_back(i == batch ? "gp"
                        : i == k   ? inner.str()
                                   : member.str());
    return "(" + llvm::join(indices, ",") + ")";
  }
  void records() {
    auto w = name(dot.getOperand(0)),
         x = name(dot.getOperand(1).getDefiningOp()->getOperand(0));
    s << "    acc=ct.full((BM,),0.,ct.float32)\n    for g in range("
      << shape[2] / 256 << "):\n"
      << "        xx=ct.reshape(ct.load(" << x
      << ",(g,0),shape=(1,256)),(256,))\n"
      << "        peak=ct.max(ct.abs(xx))\n        "
         "e=ct.where(ct.min(xx)==-peak,-peak,peak)\n"
      << "        "
         "inv=ct.truediv(-127.,ct.where(peak==0,1.,e),rounding_mode=ct."
         "RoundingMode.RN)\n"
      << "        "
         "ds=ct.where(peak==0,0.,ct.truediv(1.,inv,rounding_mode=ct."
         "RoundingMode.RN))\n";
    auto byte = [&](std::string i) {
      return "ct.astype(ct.gather(" + w + ",(r,g," + i + ")),ct.int32)";
    };
    s << "        d=ct.astype(ct.bitcast(ct.astype(" << byte("0") << "|("
      << byte("1") << "<<8),ct.uint16),ct.float16),ct.float32)\n"
      << "        dm=ct.astype(ct.bitcast(ct.astype(" << byte("2") << "|("
      << byte("3") << "<<8),ct.uint16),ct.float16),ct.float32)\n"
      << "        S=ct.full((BM,),0,ct.int32)\n        "
         "T=ct.full((BM,),0,ct.int32)\n        for j in range(8):\n"
      << "            L=" << byte("4+j%4")
      << "\n            H=" << byte("8+j%4")
      << "\n            P=" << byte("12+j%4") << "\n"
      << "            sc=ct.where(j<4,L&63,(P&15)|((L>>6)<<4))\n            "
         "mn=ct.where(j<4,H&63,(P>>4)|((H>>6)<<4))\n"
      << "            lane=ct.arange(32,dtype=ct.int32)\n            "
         "sample=ct.gather("
      << x << ",(g,j*32+lane))\n"
      << "            v=ct.mul(sample,inv,rounding_mode=ct.RoundingMode.RN)\n  "
         "          lo=ct.floor(v)\n            frac=v-lo\n"
      << "            "
         "q=ct.astype(lo,ct.int32)+ct.astype((frac>0.5)|((frac==0.5)&((ct."
         "astype(lo,ct.int32)&1)!=0)),ct.int32)\n"
      << "            q=ct.maximum(-128,ct.minimum(127,q))\n"
      << "            raw=ct.astype(ct.gather(" << w
      << ",(ct.expand_dims(r,1),g,16+32*(j//"
         "2)+ct.expand_dims(lane,0))),ct.int32)\n"
      << "            nib=(raw>>(4*(j%2)))&15\n            "
         "S+=sc*ct.sum(nib*ct.expand_dims(q,0),axis=1)\n            "
         "T+=mn*ct.sum(q)\n"
      << "        "
         "pos=ct.mul(d,ct.astype(S,ct.float32),rounding_mode=ct.RoundingMode."
         "RN)\n        "
         "neg=ct.mul(dm,ct.astype(T,ct.float32),rounding_mode=ct.RoundingMode."
         "RN)\n"
      << "        "
         "acc+=ct.mul(ds,ct.sub(pos,neg,rounding_mode=ct.RoundingMode.RN),"
         "rounding_mode=ct.RoundingMode.RN)\n";
  }
  void scaled() {
    auto a = name(dot.getOperand(0)), b = name(dot.getOperand(2)),
         sa = name(dot.getOperand(1)),
         sb = name(dot.getOperand(3).getDefiningOp()->getOperand(0));
    bool mx = dot.getAttrOfType<StringAttr>("ct_src.scaled_engine") == "mx";
    auto flat =
        dot.getAttrOfType<DenseI64ArrayAttr>("ct_src.flat_groups").asArrayRef();
    auto linear = [&](unsigned pos) {
      return llvm::is_contained(flat, slot(dot.getOperand(pos)));
    };
    s << "    acc=ct.full((BM,BN),0.,ct.float32)\n    for panel in "
         "range(ct.cdiv(K,BK)):\n"
      << "        l=ct.load(" << sa
      << ",(ct.bid(0),panel),shape=(BM,BK//"
         "32),padding_mode=ct.PaddingMode.ZERO,allow_tma=False)\n"
      << "        h=ct.load(" << sb
      << ",(panel,ct.bid(1)),shape=(BK//"
         "32,BN),padding_mode=ct.PaddingMode.ZERO,allow_tma=False)\n"
      << "        x=ct.reshape(ct.load(" << a
      << (linear(0) ? ",(ct.bid(0),panel),shape=(BM,BK)"
                    : ",(ct.bid(0),panel,0),shape=(BM,BK//32,32)")
      << ",padding_mode=ct.PaddingMode.ZERO,allow_tma="
      << (mx ? "True" : "False") << "),(BM,BK//32,32))\n"
      << "        y=ct.reshape(ct.load(" << b
      << (linear(2) ? ",(panel,ct.bid(1)),shape=(BK,BN)"
                    : ",(panel,0,ct.bid(1)),shape=(BK//32,32,BN)")
      << ",padding_mode=ct.PaddingMode.ZERO,allow_tma="
      << (mx ? "True" : "False") << "),(BK//32,32,BN))\n"
      << "        li=ct.astype(l,ct.int32)\n        hi=ct.astype(h,ct.int32)\n"
      << "        "
         "vl=ct.min(ct.reshape((li>=87)&(li<=163),(BM*(BK//32),)),axis=0)\n"
      << "        "
         "vr=ct.min(ct.reshape((hi>=87)&(hi<=163),(BN*(BK//32),)),axis=0)\n"
      << "        if vl & vr:\n";
    if (mx)
      s << "            "
           "acc=ct.mma_scaled(ct.reshape(x,(BM,BK)),ct.bitcast(l,ct.float8_"
           "e8m0fnu),ct.reshape(y,(BK,BN)),ct.bitcast(h,ct.float8_e8m0fnu),acc)"
           "\n";
    else
      s << "            ls=ct.bitcast(li<<23,ct.float32)\n            "
           "rs=ct.bitcast(hi<<23,ct.float32)\n"
        << "            "
           "av=ct.reshape(ct.astype(ct.astype(x,ct.float32)*ct.expand_dims(ls,"
           "2),ct.bfloat16),(BM,BK))\n"
        << "            "
           "bv=ct.reshape(ct.astype(ct.astype(y,ct.float32)*ct.expand_dims(rs,"
           "1),ct.bfloat16),(BK,BN))\n"
        << "            acc=ct.mma(av,bv,acc)\n";
    s << "        else:\n            "
         "ls=ct.bitcast(ct.where(li==255,2143289344,ct.where(li==0,4194304,li<<"
         "23)),ct.float32)\n"
      << "            "
         "rs=ct.bitcast(ct.where(hi==255,2143289344,ct.where(hi==0,4194304,hi<<"
         "23)),ct.float32)\n"
      << "            "
         "av=ct.reshape(ct.astype(x,ct.float32)*ct.expand_dims(ls,2),(BM,BK))\n"
      << "            "
         "bv=ct.reshape(ct.astype(y,ct.float32)*ct.expand_dims(rs,1),(BK,BN))\n"
      << "            acc=ct.mma(av,bv,acc)\n";
  }
  void ordinary() {
    auto a = dot.getOperand(0), b = dot.getOperand(1);
    auto ak = axis("lhs_k"), bk = axis("rhs_k"), ab = axis("lhs_batch"),
         bb = axis("rhs_batch");
    bool segmented = dot.getNumOperands() > 2;
    if (segmented) {
      auto o = name(dot.getOperand(2));
      s << "    start=ct.gather(" << o << ",gp)\n    end=ct.gather(" << o
        << ",gp+1)\n    r=r+start\n    src=r\n    target=r\n";
      s << "    if start+ct.bid(0)*BM>=end:\n        return\n";
      if (routed())
        s << "    target=ct.gather(" << name(dot.getOperand(3))
          << ",r,mask=r<end,padding_value=0)\n    src=target//"
          << fn.getArgAttrOfType<DenseI64ArrayAttr>(slot(domain.getOperand(0)),
                                                    "block.shape")[1]
          << "\n";
    }
    bool integer = cast<MemRefType>(a.getType()).getElementType().isInteger(8);
    s << "    acc=ct.full((BM,BN),0,ct." << (integer ? "int32" : "float32")
      << ")\n    for panel in range(ct.cdiv(K,BK)):\n";
    auto load = [&](Value v, bool left, int k, int batch) {
      std::string sym = left ? "av" : "bv";
      if ((!segmented || !left) && k == (left ? rank(v) - 1 : rank(v) - 2) &&
          (batch == -1 || batch == 0)) {
        std::string sizes = left ? "BM,BK" : "BK,BN",
                    index = left ? "ct.bid(0),panel" : "panel,ct.bid(1)";
        s << "        " << sym << "=ct.reshape(ct.load(" << name(v) << ",("
          << (batch != -1 ? "gp," : "") << index << "),shape=("
          << (batch != -1 ? "1," : "") << sizes
          << "),padding_mode=ct.PaddingMode.ZERO),(" << sizes << "))\n";
      } else if (!left && k == rank(v) - 1 && (batch == -1 || batch == 0)) {
        s << "        " << sym << "=ct.permute(ct.reshape(ct.load(" << name(v)
          << ",(" << (batch != -1 ? "gp," : "") << "ct.bid(1),panel),shape=("
          << (batch != -1 ? "1," : "")
          << "BN,BK),padding_mode=ct.PaddingMode.ZERO),(BN,BK)),(1,0))\n";
      } else {
        auto indices = coords(
            v, k, batch, left ? "ct.expand_dims(src,1)" : "ct.expand_dims(c,0)",
            left ? "ct.expand_dims(kk,0)" : "ct.expand_dims(kk,1)");
        s << "        " << sym << "=ct.gather(" << name(v) << "," << indices
          << ",mask="
          << (left ? "ct.expand_dims(r<end,1)&ct.expand_dims(kk<K,0)"
                   : "ct.expand_dims(kk<K,1)&ct.expand_dims(c<N,0)")
          << ",padding_value=0)\n";
      }
    };
    s << "        kk=panel*BK+ct.arange(BK,dtype=ct.int32)\n";
    load(a, true, ak, ab);
    load(b, false, bk, bb);
    s << "        acc=ct.mma(av,bv,acc)\n";
  }
  void write() {
    s << "import cuda.tile as ct\n@ct.kernel\ndef kernel(";
    for (auto v : fn.getArguments())
      s << name(v) << ",";
    s << "BM:ct.Constant[int],BN:ct.Constant[int]):\n    M=" << shape[0]
      << "\n    N=" << shape[1] << "\n    K=" << shape[2]
      << "\n    BK=" << tile[2]
      << "\n    gp=ct.bid(2)\n    "
         "r=ct.bid(0)*BM+ct.arange(BM,dtype=ct.int32)\n    "
         "c=ct.bid(1)*BN+ct.arange(BN,dtype=ct.int32)\n    src=r\n    "
         "target=r\n    end=M\n";
    auto n = dot.getName().stripDialect();
    if (n == "q4k_dot")
      records();
    else if (n == "scaled_contract")
      scaled();
    else
      ordinary();
    for (auto &op : domain.getRegion(0).front()) {
      auto kind = op.getName().stripDialect();
      if (kind == "add_columns") {
        if (rank(domain.getOperand(0)) == 1)
          s << "    acc+=ct.gather(" << name(op.getOperand(1))
            << ",r,mask=r<M,padding_value=0)\n";
        else
          s << "    acc+=ct.expand_dims(ct.gather(" << name(op.getOperand(1))
            << ",c,mask=c<N,padding_value=0),0)\n";
      }
      if (kind == "convert_matrix") {
        auto t = cast<ShapedType>(op.getResult(0).getType()).getElementType();
        s << "    acc=ct.astype(acc,ct."
          << (t.isBF16()        ? "bfloat16"
              : t.isF16()       ? "float16"
              : t.isInteger(32) ? "int32"
                                : "float32")
          << ")\n";
      }
      if (kind == "publish_matrix") {
        auto v = op.getOperand(0);
        auto ptr = name(v);
        if (rank(v) == 1)
          s << "    ct.scatter(" << ptr << ",r,acc,mask=r<M)\n";
        else if (dot.getName().stripDialect() == "contract" &&
                 dot.getNumOperands() > 2) {
          auto lead = "ct.expand_dims(target,1)";
          std::string idx = lead;
          if (routed()) {
            auto q = std::to_string(fn.getArgAttrOfType<DenseI64ArrayAttr>(
                slot(v), "block.shape")[1]);
            idx = lead + std::string("//") + q + "," + lead + "%" + q;
          }
          s << "    ct.scatter(" << ptr << ",(" << idx
            << ",ct.expand_dims(c,0)),acc,mask=ct.expand_dims(r<end,1)&ct."
               "expand_dims(c<N,0))\n";
        } else if (rank(v) == 2)
          s << "    ct.store(" << ptr << ",(ct.bid(0),ct.bid(1)),acc"
            << (n == "scaled_contract" ? ",allow_tma=False" : "") << ")\n";
        else
          s << "    ct.store(" << ptr
            << ",(gp,ct.bid(0),ct.bid(1)),ct.reshape(acc,(1,BM,BN)))\n";
      }
    }
  }
};
} // namespace
llvm::Error emitMatrix(func::FuncOp fn, llvm::raw_pwrite_stream &out) {
  Writer w(fn);
  w.write();
  llvm::json::Object views;
  if (auto flat = w.dot.getAttrOfType<DenseI64ArrayAttr>("ct_src.flat_groups"))
    for (auto i : flat.asArrayRef())
      views[std::to_string(i)] =
          llvm::json::Array(i == w.slot(w.dot.getOperand(0))
                                ? SmallVector<int64_t>{w.shape[0], w.shape[2]}
                                : SmallVector<int64_t>{w.shape[2], w.shape[1]});
  return intentdsl::block::emitMatrixPackage(fn, w.text, out, 3,
                                             std::move(views));
}
} // namespace CI_UNIT_NAMESPACE
