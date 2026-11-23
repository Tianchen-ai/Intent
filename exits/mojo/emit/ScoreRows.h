#pragma once
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace CI_UNIT_NAMESPACE {
// One task owns several adjacent query rows. K and V stay shared across
// the rows; the denominator uses fp32 weights before probability narrowing.
inline void scoreRows(llvm::raw_ostream &s, mlir::func::FuncOp fn,
                      mlir::Operation &space) {
  auto dim = [&](int i, int j) {
    return fn.getArgAttrOfType<mlir::DenseI64ArrayAttr>(i, "cpu.shape")[j];
  };
  auto num = [](int64_t n) { return std::to_string(n); };
  auto address = [&](unsigned slot, llvm::StringRef head, llvm::StringRef row) {
    auto a = fn.getArgAttrOfType<mlir::DenseI64ArrayAttr>(slot, "cpu.strides");
    return "a" + num(slot) + ".unsafe_offset(b*" + num(a[0]) + "+" +
           head.str() + "*" + num(a[1]) + "+(" + row.str() + ")*" + num(a[2]) +
           ")";
  };
  auto width =
      space.getAttrOfType<mlir::IntegerAttr>("mojo_row.score_microtile")
          .getInt();
  auto cols = dim(0, 3), keys = dim(1, 2), features = dim(2, 3);
  for (int j = 0; j < width; ++j)
    s << "    var weights" << j << "=unsafe_stack_allocation[" << keys
      << ",Float32]()\n";
  s << "    for owner in range(Int(lo),Int(hi)," << width << "):\n"
    << "        var row=owner%" << dim(0, 2) << "\n        var h=(owner//"
    << dim(0, 2) << ")%" << dim(0, 1) << "\n        var b=owner//"
    << (dim(0, 2) * dim(0, 1)) << "\n        var kv=h//"
    << (dim(0, 1) / dim(1, 1))
    << "\n        var factor=a4*Float32(1.4426950408889634)\n"
    << "        var neginf=bitcast[DType.float32](UInt32(0xff800000))\n";
  for (int j = 0; j < width; ++j)
    s << "        var q" << j << "=" << address(0, "h", "row+" + num(j))
      << ".unsafe_load[width=" << cols << "]().cast[DType.float32]()\n";
  s << "        for t in range(min(row+" << width << "," << keys << ")):\n"
    << "            var key=" << address(1, "kv", "t")
    << ".unsafe_load[width=" << cols << "]().cast[DType.float32]()\n";
  for (int j = 0; j < width; ++j)
    s << "            weights" << j << "[unsafe_offset=t]=(q" << j
      << "*key).reduce_add()*factor\n";
  for (int j = 0; j < width; ++j) {
    auto w = "weights" + num(j);
    s << "        for t in range(min(row+" << j + 1 << "," << keys << "),"
      << keys << "):\n"
      << "            " << w << "[unsafe_offset=t]=neginf\n"
      << "        var peaks" << j << "=SIMD[DType.float32,16](neginf)\n"
      << "        for t in range(0," << keys << ",16):\n"
      << "            peaks" << j << "=max(peaks" << j << "," << w
      << ".unsafe_offset(t).unsafe_load[width=16]())\n"
      << "        var peak" << j << "=peaks" << j << ".reduce_max()\n"
      << "        var z" << j << "=SIMD[DType.float32,16](0)\n"
      << "        for t in range(0," << keys << ",16):\n"
      << "            var p=exp2(" << w
      << ".unsafe_offset(t).unsafe_load[width=16]()-peak" << j << ")\n"
      << "            comptime for i in range(16):\n"
      << "                if t+i>row+" << j << ": p[i]=0\n"
      << "            z" << j << "+=p\n"
      << "            " << w
      << ".unsafe_offset(t).unsafe_store(p.cast[DType.bfloat16]().cast[DType."
         "float32]())\n"
      << "        var inverse" << j << "=1/z" << j << ".reduce_add()\n"
      << "        var out" << j << "=SIMD[DType.float32," << features
      << "](0)\n";
  }
  s << "        for t in range(" << keys
    << "):\n            var value=" << address(2, "kv", "t")
    << ".unsafe_load[width=" << features << "]().cast[DType.float32]()\n";
  for (int j = 0; j < width; ++j)
    s << "            out" << j << "+=value*weights" << j
      << "[unsafe_offset=t]\n";
  for (int j = 0; j < width; ++j)
    s << "        " << address(3, "h", "row+" + num(j)) << ".unsafe_store((out"
      << j << "*inverse" << j << ").cast[DType.bfloat16]())\n";
}
} // namespace CI_UNIT_NAMESPACE
