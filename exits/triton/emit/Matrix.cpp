#include "intentdsl/levels/execution/block/ir/MatrixAccess.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
struct Writer : intentdsl::block::MatrixAccess {
  using MatrixAccess::MatrixAccess;
  std::string text;
  llvm::raw_string_ostream s{text};
  std::string strideOf(Value v, int i) { return std::to_string(stride(v, i)); }
  std::string addr(Value v, StringRef row, int r, StringRef col, int c,
                   int batch = -1) {
    return name(v) + (batch < 0 ? "" : "+gp*" + strideOf(v, batch)) + "+" +
           row.str() + "*" + strideOf(v, r) + "+" + col.str() + "*" +
           strideOf(v, c);
  }
  void records() {
    auto w = dot.getOperand(0),
         x = dot.getOperand(1).getDefiningOp()->getOperand(0);
    s << "    acc=tl.full((ROWS,),0,tl.float32)\n"
      << "    for g in range(" << shape[2] / 256 << "):\n"
      << "        xs=" << name(x) << "+tl.cast(g,tl.int64)*" << stride(x, 0) << "\n"
      << "        v=tl.load(xs+tl.arange(0,256).to(tl.int64)*" << stride(x, 1) << ")\n"
      << "        peak=tl.max(tl.abs(v),0)\n"
      << "        extreme=tl.where(tl.sum((v == "
         "-peak).to(tl.int32),0)>0,-peak,peak)\n"
      << "        inv=tl.div_rn(-127.,tl.where(peak==0,1.,extreme))\n"
      << "        ds=tl.where(peak==0,0.,tl.div_rn(1.,inv))\n"
      << "        base=" << name(w) << "+r*" << stride(w, 0) << "+tl.cast(g,tl.int64)*"
      << stride(w, 1) << "\n";
    auto byte = [&](std::string i) {
      return "tl.load(base+(" + i + ")*" + strideOf(w, 2) +
             ",mask=r<M,other=0).to(tl.int32)";
    };
    s << "        d=(" << byte("0") << "|(" << byte("1")
      << "<<8)).to(tl.uint16).to(tl.float16,bitcast=True).to(tl.float32)\n"
      << "        dm=(" << byte("2") << "|(" << byte("3")
      << "<<8)).to(tl.uint16).to(tl.float16,bitcast=True).to(tl.float32)\n"
      << "        t=tl.arange(0,256).to(tl.int64)\n        j=t//32\n"
      << "        header=base[:,None]+(j%4)[None,:]*"<<stride(w,2)<<"\n"
      << "        L=tl.load(header+"<<4*stride(w,2)<<",r[:,None]<M,other=0).to(tl.int32)\n"
      << "        H=tl.load(header+"<<8*stride(w,2)<<",r[:,None]<M,other=0).to(tl.int32)\n"
      << "        P=tl.load(header+"<<12*stride(w,2)<<",r[:,None]<M,other=0).to(tl.int32)\n"
      << "        sc=tl.where(j[None,:]<4,L&63,(P&15)|((L>>6)<<4))\n"
      << "        mn=tl.where(j[None,:]<4,H&63,(P>>4)|((H>>6)<<4))\n"
      << "        q=tl.minimum(tl.maximum(libdevice.rint(libdevice.mul_rn(inv,v)),-128.),127.).to(tl.int32)\n"
      << "        raw=tl.load(base[:,None]+(16+32*(j//2)+t%32)[None,:]*"
      << stride(w,2)<<",r[:,None]<M,other=0).to(tl.int32)\n"
      << "        nib=(raw>>(4*(j%2))[None,:])&15\n"
      << "        S=tl.sum(sc*nib*q[None,:],1)\n        T=tl.sum(mn*q[None,:],1)\n"
      << "        "
         "acc+=libdevice.mul_rn(ds,libdevice.sub_rn(libdevice.mul_rn(d,S.to(tl."
         "float32)),libdevice.mul_rn(dm,T.to(tl.float32))))\n";
  }
  void window() {
    auto lhs = dot.getOperand(0), rhs = dot.getOperand(1),
         step = dot.getOperand(2), prefix = dot.getOperand(3);
    auto dims =
        fn.getArgAttrOfType<DenseI64ArrayAttr>(slot(step), "block.shape");
    s << "    head=gp%" << std::max<int64_t>(1, dims[1]) << "\n    chunk=(gp//"
      << std::max<int64_t>(1, dims[1]) << ")%" << std::max<int64_t>(1, dims[2])
      << "\n    batch=gp//" << std::max<int64_t>(1, dims[1] * dims[2])
      << "\n    group=head//" << axis("head_group") << "\n";
    auto weight = [&](Value v) {
      return name(v) + "+batch*" + strideOf(v, 0) + "+head*" + strideOf(v, 1) +
             "+chunk*" + strideOf(v, 2);
    };
    s << "    acc=tl.full((ROWS,WIDTH),0.,tl.float32)\n    for panel in "
         "range(tl.cdiv(K,BK)):\n"
      << "        t=tl.cast(panel,tl.int64)*BK+tl.arange(0,BK)\n        "
         "seq=chunk*K+t\n"
      << "        a=tl.load(" << name(lhs) << "+batch*" << stride(lhs, 0)
      << "+seq[None,:]*" << stride(lhs, 1) << "+head*" << stride(lhs, 2)
      << "+r[:,None]*" << stride(lhs, 3)
      << ",mask=(r[:,None]<M)&(t[None,:]<K),other=0)\n"
      << "        b=tl.load(" << name(rhs) << "+batch*" << stride(rhs, 0)
      << "+seq[:,None]*" << stride(rhs, 1) << "+group*" << stride(rhs, 2)
      << "+c[None,:]*" << stride(rhs, 3)
      << ",mask=(t[:,None]<K)&(c[None,:]<N),other=0)\n"
      << "        d=tl.load(" << weight(step) << "+t*" << stride(step, 3)
      << ",t<K,other=0)\n"
      << "        v=tl.load(" << weight(prefix) << "+t*" << stride(prefix, 3)
      << ",t<K,other=0)\n"
      << "        last=tl.load(" << weight(prefix) << "+(K-1)*"
      << stride(prefix, 3) << ")\n"
      << "        rhs="
      << intentdsl::kir::panelExpression(
             dot, {"b", "d[:,None]", "v[:,None]", "last"},
             {"tl.cast($,tl.float32)", "tl.cast($,tl.bfloat16)", "tl.minimum",
              "tl.exp", ",propagate_nan=tl.PropagateNan.ALL"})
      << "\n"
      << "        "
         "acc=tl.dot(a,tl.where(t[:,None]<K,rhs,0.).to(tl.bfloat16),acc)\n";
  }
  void scaled() {
    auto a = dot.getOperand(0), sa = dot.getOperand(1), b = dot.getOperand(2),
         sb = dot.getOperand(3).getDefiningOp()->getOperand(0);
    auto *store = domain.getRegion(0).front().back().getPrevNode();
    if (dot.getAttrOfType<BoolAttr>("triton_src.neutral_scale_dispatch")
            .getValue()) {
      auto v = store->getOperand(0);
      s << "    group=tl.arange(0," << shape[2] / 32 << ").to(tl.int64)\n"
        << "    left_scale=tl.load("
        << addr(sa, "r[:,None]", 0, "group[None,:]", 1)
        << ",r[:,None]<M,other=127)\n"
        << "    right_scale=tl.load("
        << addr(sb, "group[:,None]", 0, "c[None,:]", 1)
        << ",c[None,:]<N,other=127)\n"
        << "    if "
           "(tl.min((left_scale==127).to(tl.int32),None)==1)&(tl.min((right_"
           "scale==127).to(tl.int32),None)==1):\n"
        << "        result=tl.full((ROWS,WIDTH),0.,tl.float32)\n        for p "
           "in range(tl.cdiv(K,BK)):\n"
        << "            tk=tl.cast(p,tl.int64)*BK+tl.arange(0,BK)\n"
        << "            la=tl.load(" << name(a)
        << ".to(tl.pointer_type(tl.uint8))+r[:,None]*" << stride(a, 0)
        << "+(tk//32)[None,:]*" << stride(a, 1) << "+(tk%32)[None,:]*"
        << stride(a, 2)
        << ",mask=(r[:,None]<M)&(tk[None,:]<K),other=0).to(tl.float8e4nv,"
           "bitcast=True)\n"
        << "            rb=tl.load(" << name(b)
        << ".to(tl.pointer_type(tl.uint8))+(tk//32)[:,None]*" << stride(b, 0)
        << "+(tk%32)[:,None]*" << stride(b, 1) << "+c[None,:]*" << stride(b, 2)
        << ",mask=(tk[:,None]<K)&(c[None,:]<N),other=0).to(tl.float8e4nv,"
           "bitcast=True)\n"
        << "            result=tl.dot(la,rb,result,max_num_imprecise_acc=0)\n"
        << "        tl.store(" << addr(v, "r[:,None]", 0, "c[None,:]", 1)
        << ",result,mask=(r[:,None]<M)&(c[None,:]<N))\n        return\n";
    }
    s << "    acc=tl.full((ROWS,WIDTH),0.,tl.float32)\n    for g in range("
      << shape[2] / 32 << "):\n"
      << "        ax=tl.load(" << addr(sa, "r", 0, "g", 1)
      << ",r<M,other=127).to(tl.int32)\n"
      << "        bx=tl.load(" << addr(sb, "g", 0, "c", 1)
      << ",c<N,other=127).to(tl.int32)\n"
      << "        "
         "ls=tl.where(ax==255,float('nan'),libdevice.ldexp(tl.full((ROWS,),1.,"
         "tl.float32),ax-127))\n"
      << "        "
         "rs=tl.where(bx==255,float('nan'),libdevice.ldexp(tl.full((WIDTH,),1.,"
         "tl.float32),bx-127))\n"
      << "        t=tl.arange(0,32).to(tl.int64)\n"
      << "        av=tl.load(" << name(a)
      << ".to(tl.pointer_type(tl.uint8))+r[:,None]*" << stride(a, 0) << "+tl.cast(g,tl.int64)*"
      << stride(a, 1) << "+t[None,:]*" << stride(a, 2)
      << ",r[:,None]<M,other=0).to(tl.float8e4nv,bitcast=True).to(tl.float32)*"
         "ls[:,None]\n"
      << "        bv=tl.load(" << name(b) << ".to(tl.pointer_type(tl.uint8))+tl.cast(g,tl.int64)*"
      << stride(b, 0) << "+t[:,None]*" << stride(b, 1) << "+c[None,:]*"
      << stride(b, 2)
      << ",c[None,:]<N,other=0).to(tl.float8e4nv,bitcast=True).to(tl.float32)*"
         "rs[None,:]\n"
      << "        if "
         "(tl.min(ax,0)>=87)&(tl.max(ax,0)<=163)&(tl.min(bx,0)>=87)&(tl.max(bx,"
         "0)<=163):\n"
      << "            acc=tl.dot(av.to(tl.bfloat16),bv.to(tl.bfloat16),acc)\n"
      << "        else:\n"
      << "            acc=tl.dot(av,bv,acc,input_precision='ieee')\n";
  }
  llvm::Error write() {
    s << "import triton\nimport triton.language as tl\nfrom "
         "triton.language.extra.cuda import libdevice\n@triton.jit\ndef "
         "kernel(";
    for (Value v : fn.getArguments())
      s << name(v) << ",";
    s << "ROWS:tl.constexpr,WIDTH:tl.constexpr):\n    M:tl.constexpr="
      << shape[0] << "\n    N:tl.constexpr=" << shape[1]
      << "\n    K:tl.constexpr=" << shape[2]
      << "\n    BK:tl.constexpr=" << tile[2] << "\n"
      << "    gp=tl.program_id(2).to(tl.int64)\n    "
         "r=tl.program_id(0).to(tl.int64)*ROWS+tl.arange(0,ROWS)\n"
      << "    c=tl.program_id(1).to(tl.int64)*WIDTH+tl.arange(0,WIDTH)\n    "
         "end=M\n    src=r\n    target=r\n";
    auto n = dot.getName().stripDialect();
    bool segmented = n == "contract" && dot.getNumOperands() > 2;
    if (n == "window_contract")
      window();
    else if (n == "q4k_dot")
      records();
    else if (n == "scaled_contract")
      scaled();
    else {
      auto a = dot.getOperand(0), b = dot.getOperand(1);
      int ak = axis("lhs_k"), bk = axis("rhs_k"), ab = axis("lhs_batch"),
          bb = axis("rhs_batch");
      if (segmented) {
        auto o = dot.getOperand(2);
        s << "    start=tl.load(" << name(o) << "+gp*" << stride(o, 0)
          << ")\n    end=tl.load(" << name(o) << "+(gp+1)*" << stride(o, 0)
          << ")\n    r=r+start\n    target=r\n    src=r\n";
        s << "    if "
             "start.to(tl.int64)+tl.program_id(0).to(tl.int64)*ROWS>=end:\n    "
             "    return\n";
        if (routed()) {
          auto ids = dot.getOperand(3);
          s << "    target=tl.load(" << name(ids) << "+r*" << stride(ids, 0)
            << ",r<end,other=0).to(tl.int64)\n"
            << "    src=target//"
            << fn.getArgAttrOfType<DenseI64ArrayAttr>(
                   slot(domain.getOperand(0)), "block.shape")[1]
            << "\n";
        }
      }
      bool integer =
          cast<MemRefType>(a.getType()).getElementType().isInteger(8);
      s << "    acc=tl.full((ROWS,WIDTH),0,tl."
        << (integer ? "int32" : "float32")
        << ")\n    for panel in range(tl.cdiv(K,BK)):\n"
        << "        k=tl.cast(panel,tl.int64)*BK+tl.arange(0,BK)\n"
        << "        av=tl.load("
        << addr(a, "src[:,None]", freeAxis(a, ak, ab), "k[None,:]", ak, ab)
        << ",mask=(r[:,None]<end)&(k[None,:]<K),other=0)\n"
        << "        bv=tl.load("
        << addr(b, "k[:,None]", bk, "c[None,:]", freeAxis(b, bk, bb), bb)
        << ",mask=(k[:,None]<K)&(c[None,:]<N),other=0)\n"
        << "        acc=tl.dot(av,bv,acc,input_precision='ieee',out_dtype=tl."
        << (integer ? "int32" : "float32") << ")\n";
    }
    for (auto &op : domain.getRegion(0).front()) {
      auto kind = op.getName().stripDialect();
      if (kind == "add_columns") {
        auto v = op.getOperand(1);
        if (rank(domain.getOperand(0)) == 1)
          s << "    acc=acc+tl.load(" << name(v) << "+r*" << stride(v, 0)
            << ",r<M,other=0)\n";
        else
          s << "    acc=acc+tl.load(" << name(v) << "+c*" << stride(v, 0)
            << ",c<N,other=0)[None,:]\n";
      }
      if (kind == "convert_matrix") {
        auto t = cast<ShapedType>(op.getResult(0).getType()).getElementType();
        s << "    acc=acc.to(tl."
          << (t.isBF16()        ? "bfloat16"
              : t.isF16()       ? "float16"
              : t.isInteger(32) ? "int32"
                                : "float32")
          << ")\n";
      }
      if (kind == "publish_matrix") {
        auto v = op.getOperand(0);
        if (rank(v) == 1)
          s << "    tl.store(" << name(v) << "+r*" << stride(v, 0)
            << ",acc,r<M)\n";
        else {
          std::string ptr;
          if (routed()) {
            auto multiplicity = fn.getArgAttrOfType<DenseI64ArrayAttr>(
                slot(v), "block.shape")[1];
            ptr = addr(v,
                       "(target//" + std::to_string(multiplicity) + ")[:,None]",
                       0, "c[None,:]", 2) +
                  "+(target%" + std::to_string(multiplicity) + ")[:,None]*" +
                  strideOf(v, 1);
          } else if (n == "window_contract") {
            ptr = addr(v, "r[:,None]", 3, "c[None,:]", 4) + "+batch*" +
                  strideOf(v, 0) + "+chunk*" + strideOf(v, 1) + "+head*" +
                  strideOf(v, 2);
          } else
            ptr = addr(v, "target[:,None]", rank(v) - 2, "c[None,:]",
                       rank(v) - 1, rank(v) == 3 ? 0 : -1);
          s << "    tl.store(" << ptr
            << ",acc,mask=(r[:,None]<end)&(c[None,:]<N))\n";
        }
      }
    }
    return llvm::Error::success();
  }
};
} // namespace
llvm::Error emitMatrix(func::FuncOp fn, llvm::raw_pwrite_stream &out) {
  Writer w(fn);
  if (auto err = w.write())
    return err;
  return intentdsl::block::emitMatrixPackage(fn, w.text, out, 2);
}
} // namespace CI_UNIT_NAMESPACE
