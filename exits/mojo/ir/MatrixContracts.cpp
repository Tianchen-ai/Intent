#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"
#include "intentdsl/levels/algorithm/kir/ir/NativeDot.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;

void registerMatrixContracts(Unit &u) {
  auto add = [&](ImplementationContract c) {
    u.implementations.push_back({std::move(c), COMPILERINFRA_SOURCE});
  };

  {
    ImplementationContract c;
    c.name = "simd-fma";
    c.operation = "mojo_row.simd_fma";
    c.symbol = "std.math.fma";
    c.source = ImplementationSource::Builtin;
    c.minimum = Reuse::Native;
    c.operands = "f32|f16|bf16|i8 widened to f32|i32 SIMD lanes";
    c.layout = "row-major or arbitrary strided LHS/RHS, optional [K,nr] panels";
    c.accumulator = "f32 or i32 SIMD registers";
    c.effects = "pure register arithmetic";
    c.numerical = "IEEE-754 f32 FMA preserving subnormals, Inf, and NaN; exact i32 MAC";
    c.estimator = u.qualify("register-block");
    c.costParameters = {"vector_bits", "workers"};
    c.available = [](const TargetProfile &p) -> llvm::Error {
      auto *rm = p.services.get<intentdsl::cpu::RowMachine>();
      if (!rm || rm->vectorBits < 128)
        return llvm::createStringError("SIMD FMA requires >=128-bit CPU SIMD");
      return llvm::Error::success();
    };
    c.verify = [](Operation *op) -> LogicalResult {
      if (auto ci = dyn_cast<ContractOpInterface>(op))
        return success(succeeded(ci.verifyShapeAndTypes()) &&
                       succeeded(ci.verifyLayoutAndAddressSpace()));
      return failure();
    };
    TypedContract tc;
    tc.shape.minOperands = 3;
    tc.shape.maxOperands = 3;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    tc.shape.operandRanks = {1, 1, 1};
    tc.shape.resultRanks = {1};
    tc.shape.sameFirstOperandAndResultShape = true;
    tc.operandTypes = {ScalarTypeKind::F32, ScalarTypeKind::I32};
    tc.resultTypes = {ScalarTypeKind::F32, ScalarTypeKind::I32};
    tc.accumulatorTypes = {ScalarTypeKind::F32, ScalarTypeKind::I32};
    tc.effects = EffectMask::None;
    c.typed = std::move(tc);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }

  {
    ImplementationContract c;
    c.name = "vnni-dot4";
    c.operation = "mojo_row.vnni_dot4";
    c.symbol = "llvm.x86.avx512.vpdpbusd.512";
    c.source = ImplementationSource::Builtin;
    c.minimum = Reuse::Native;
    c.operands = "u8x64 (xor-shifted i8) x s8x64 -> s32x16";
    c.layout = "LHS unit K stride; RHS packed signed-dot4-panels [K/4, nr, 4] + colSums";
    c.accumulator = "s32x16 ZMM registers with -128*colSum bias correction";
    c.effects = "pure register arithmetic";
    c.numerical = "exact two's-complement s8*s8->s32 wraparound-free dot4";
    c.estimator = u.qualify("register-block");
    c.costParameters = {"vector_bits", "workers"};
    c.available = [](const TargetProfile &p) -> llvm::Error {
      if (!p.features.count("avx512_vnni"))
        return llvm::createStringError("requires AVX-512 VNNI on all active CPUs");
      return llvm::Error::success();
    };
    c.verify = [](Operation *op) -> LogicalResult {
      if (auto ci = dyn_cast<ContractOpInterface>(op))
        return success(succeeded(ci.verifyShapeAndTypes()) &&
                       succeeded(ci.verifyLayoutAndAddressSpace()));
      return failure();
    };
    TypedContract tc;
    tc.shape.minOperands = 3;
    tc.shape.maxOperands = 3;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    tc.shape.operandRanks = {1, 1, 1};
    tc.shape.resultRanks = {1};
    tc.shape.sameFirstOperandAndResultShape = true;
    tc.operandTypes = {ScalarTypeKind::I32, ScalarTypeKind::UI8,
                       ScalarTypeKind::I8};
    tc.resultTypes = {ScalarTypeKind::I32};
    tc.accumulatorTypes = {ScalarTypeKind::I32};
    tc.effects = EffectMask::None;
    c.typed = std::move(tc);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }

  {
    ImplementationContract c;
    c.name = "q4k-simd";
    c.operation = "mojo_row.q4k_superblock";
    c.symbol = "mojo_row.q4k_superblock";
    c.source = ImplementationSource::Builtin;
    c.minimum = Reuse::Native;
    c.operands = "Q4_K[144B] x Q8_K[292B] -> f32";
    c.layout = "Q4_K row records x shared Q8_K super-blocks";
    c.accumulator = "s32 subgroup dot products scaled into f32 per super-block";
    c.effects = "reads Q4_K and Q8_K buffers";
    c.numerical = "exact s32 8x32 subgroup sums with IEEE f16->f32 scales at super-block R32 boundary";
    c.estimator = u.qualify("q4k-simd-row");
    c.costParameters = {"vector_bits", "workers"};
    c.available = [](const TargetProfile &p) -> llvm::Error {
      auto *rm = p.services.get<intentdsl::cpu::RowMachine>();
      if (!rm || rm->vectorBits < 128)
        return llvm::createStringError("Q4_K SIMD requires >=128-bit CPU SIMD");
      return llvm::Error::success();
    };
    c.verify = [](Operation *op) -> LogicalResult {
      auto vnni = op->getAttrOfType<BoolAttr>("use_vnni");
      if (vnni && vnni.getValue())
        return failure();
      if (op->getAttr("kir.storage") ==
          TypeAttr::get(intentdsl::kir::Q4KType::get(op->getContext())))
        return success();
      auto fmt = op->getAttrOfType<StringAttr>("format");
      const auto *spec =
          fmt ? intentdsl::kir::lookupQuantFormat(fmt.getValue()) : nullptr;
      return success(spec &&
                     spec->roleLayout ==
                         intentdsl::kir::QuantRoleLayout::RecordGemv);
    };
    TypedContract tc;
    tc.shape.minOperands = 4;
    tc.shape.maxOperands = 4;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    tc.shape.resultRanks = {1};
    tc.operandTypes = {ScalarTypeKind::Index};
    tc.resultTypes = {ScalarTypeKind::F32};
    tc.accumulatorTypes = {ScalarTypeKind::I32};
    tc.effects = EffectMask::Read;
    c.typed = std::move(tc);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }

  {
    ImplementationContract c;
    c.name = "q4k-vnni";
    c.operation = "mojo_row.q4k_superblock";
    c.symbol = "llvm.x86.avx512.vpdpbusd.512";
    c.source = ImplementationSource::Builtin;
    c.minimum = Reuse::Native;
    c.operands = "Q4_K[144B] x Q8_K[292B] -> f32";
    c.layout = "Q4_K row records x shared Q8_K super-blocks";
    c.accumulator = "s32x16 AVX-512 VNNI vpdpbusd subgroup dot products scaled into f32 per super-block";
    c.effects = "reads Q4_K and Q8_K buffers";
    c.numerical = "exact u8*s8->s32 8x32 subgroup sums with IEEE f16->f32 scales at super-block R32 boundary";
    c.estimator = u.qualify("q4k-simd-row");
    c.costParameters = {"vector_bits", "workers"};
    c.available = [](const TargetProfile &p) -> llvm::Error {
      auto *rm = p.services.get<intentdsl::cpu::RowMachine>();
      if (!rm || rm->vectorBits < 512 || !p.features.count("avx512_vnni"))
        return llvm::createStringError(
            "Q4_K VNNI requires 512-bit AVX-512 VNNI on all active CPUs");
      return llvm::Error::success();
    };
    c.verify = [](Operation *op) -> LogicalResult {
      auto vnni = op->getAttrOfType<BoolAttr>("use_vnni");
      if (!vnni || !vnni.getValue())
        return failure();
      if (op->getAttr("kir.storage") ==
          TypeAttr::get(intentdsl::kir::Q4KType::get(op->getContext())))
        return success();
      auto fmt = op->getAttrOfType<StringAttr>("format");
      const auto *spec =
          fmt ? intentdsl::kir::lookupQuantFormat(fmt.getValue()) : nullptr;
      return success(spec &&
                     spec->roleLayout ==
                         intentdsl::kir::QuantRoleLayout::RecordGemv);
    };
    TypedContract tc;
    tc.shape.minOperands = 4;
    tc.shape.maxOperands = 4;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    tc.shape.resultRanks = {1};
    tc.operandTypes = {ScalarTypeKind::Index};
    tc.resultTypes = {ScalarTypeKind::F32};
    tc.accumulatorTypes = {ScalarTypeKind::I32};
    tc.effects = EffectMask::Read;
    c.typed = std::move(tc);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }

  auto makeBrgemm = [&](StringRef name, StringRef operands, StringRef layout,
                        StringRef acc, StringRef numerical, bool requiresFtz,
                        ScalarTypeKind accKind,
                        std::function<llvm::Error(const TargetProfile &)> avail) {
    ImplementationContract c;
    c.name = name.str();
    c.operation = "mojo_row.brgemm_call";
    c.symbol = "intent_brgemm_tile";
    c.source = ImplementationSource::External;
    c.minimum = Reuse::Vendor;
    c.operands = operands.str();
    c.layout = layout.str();
    c.accumulator = acc.str();
    c.effects = "writes tile accumulation buffer and per-worker scratchpad";
    c.numerical = numerical.str();
    c.estimator = u.qualify("brgemm-tile");
    c.costParameters = {"vector_bits", "workers"};
    c.dependencies = {"onednn"};
    c.available =
        [avail = std::move(avail)](const TargetProfile &p) -> llvm::Error {
      if (!p.features.count("onednn_runtime"))
        return llvm::createStringError(
            "Mojo host runtime lacks oneDNN ukernel library and headers");
      return avail(p);
    };
    c.verify = [requiresFtz](Operation *op) -> LogicalResult {
      if (op->getNumOperands() != 5 || op->getNumResults() != 1 ||
          !op->getResult(0).getType().isInteger(32))
        return failure();
      auto ftz = op->getAttrOfType<BoolAttr>("allow_subnormal_ftz");
      if (!ftz || ftz.getValue() != requiresFtz)
        return failure();
      auto fn = op->getParentOfType<func::FuncOp>();
      if (!fn || (requiresFtz && !intentdsl::kir::permitsBf16Dot(op)))
        return failure();
      return success();
    };
    TypedContract tc;
    tc.shape.minOperands = 5;
    tc.shape.maxOperands = 5;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    tc.operandTypes = {ScalarTypeKind::Any};
    tc.resultTypes = {ScalarTypeKind::I32};
    tc.accumulatorTypes = {accKind};
    tc.effects = EffectMask::ReadWrite;
    // The BF16 verifier checks the resolved dot contract.
    c.typed = std::move(tc);
    populateTypedContractSummaries(c);
    add(std::move(c));
  };

  makeBrgemm(
      "onednn-brgemm-f32", "f32 x f32 -> f32",
      "LHS row-major (unit K stride), RHS row-major or [K, nr] panels",
      "f32 ZMM registers written to tile stack buffer",
      "IEEE-754 f32 FMA preserving subnormals, signed zero, Inf, and qNaN",
      false, ScalarTypeKind::F32,
      [](const TargetProfile &p) -> llvm::Error {
        if (!p.features.count("avx2") && !p.features.count("avx512f"))
          return llvm::createStringError("oneDNN f32 BRGeMM requires AVX2 or AVX-512");
        return llvm::Error::success();
      });

  makeBrgemm(
      "onednn-brgemm-f16", "f16 x f16 -> f32",
      "LHS row-major (unit K stride), RHS row-major or [K, nr] panels",
      "f32 ZMM registers written to tile stack buffer",
      "IEEE-754 f16 widened to f32 FMA preserving f16 subnormals, Inf, and qNaN",
      false, ScalarTypeKind::F32,
      [](const TargetProfile &p) -> llvm::Error {
        if (!p.features.count("avx512_fp16") && !p.features.count("avx512f"))
          return llvm::createStringError("oneDNN f16 BRGeMM requires AVX-512");
        return llvm::Error::success();
      });

  makeBrgemm(
      "onednn-brgemm-bf16", "bf16 x bf16 -> f32",
      "LHS row-major (unit K stride, K%2==0), RHS VNNI pack32 [K/2, nr, 2]",
      "f32 AMX/AVX-512 BF16 dot-pair accumulator",
      "AMX/AVX512-BF16 dot2 to f32; may flush input, intermediate, and output subnormals to zero (requires allow_subnormal_ftz)",
      true, ScalarTypeKind::F32,
      [](const TargetProfile &p) -> llvm::Error {
        if (!p.features.count("amx_bf16") && !p.features.count("avx512_bf16"))
          return llvm::createStringError(
              "oneDNN bf16 BRGeMM requires AMX-BF16 or AVX512-BF16");
        return llvm::Error::success();
      });

  {
    ImplementationContract c;
    c.name = "pack-bf16-rows";
    c.operation = "mojo_row.pack_value_table";
    c.symbol = "intent_pack_bf16_rows";
    c.source = ImplementationSource::External;
    c.minimum = Reuse::Vendor;
    c.dependencies = {"onednn"};
    c.layout = "dense BF16 planes to VNNI row pairs";
    c.numerical = "bit-preserving BF16 copy";
    c.estimator = u.qualify("brgemm-tile");
    c.costParameters = {"vector_bits", "workers"};
    c.available = [](const TargetProfile &p) {
      return p.checkImplementation("exits/mojo::onednn-brgemm-bf16");
    };
    c.verify = [](Operation *op) {
      auto slot = op->getAttrOfType<IntegerAttr>("slot").getInt();
      auto fn = op->getParentOfType<func::FuncOp>();
      if (!fn || slot < 0 || slot >= fn.getNumArguments())
        return failure();
      auto view = dyn_cast<MemRefType>(fn.getArgument(slot).getType());
      auto shape = fn.getArgAttrOfType<DenseI64ArrayAttr>(slot, "cpu.shape");
      auto strides =
          fn.getArgAttrOfType<DenseI64ArrayAttr>(slot, "cpu.strides");
      if (!view || !view.getElementType().isBF16() || !shape || !strides ||
          shape.size() < 2 || shape.size() != strides.size())
        return failure();
      int64_t count = 1;
      for (int i = shape.size() - 1; i >= 0; --i)
        if (shape[i] <= 0 || strides[i] != count ||
            __builtin_mul_overflow(count, shape[i], &count))
          return failure();
      auto cols = shape.asArrayRef().back(), rows = shape[shape.size() - 2];
      return success(
          rows % 2 == 0 && count <= INT64_MAX / 2 &&
          op->getAttrOfType<IntegerAttr>("rows").getInt() == count / cols &&
          op->getAttrOfType<IntegerAttr>("plane_rows").getInt() == rows &&
          op->getAttrOfType<IntegerAttr>("columns").getInt() == cols);
    };
    TypedContract t;
    t.shape.minOperands = t.shape.maxOperands = 0;
    t.shape.minResults = t.shape.maxResults = 0;
    t.effects = EffectMask::Allocate | EffectMask::ReadWrite;
    c.typed = std::move(t);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }

  {
    ImplementationContract c;
    c.name = "transpose-bf16-weights";
    c.operation = "mojo_row.transpose_weights";
    c.symbol = "intent_transpose_bf16";
    c.source = ImplementationSource::External;
    c.minimum = Reuse::Vendor;
    c.dependencies = {"onednn"};
    c.operands = "BF16 source, BF16 destination, row count";
    c.layout = "out-of-place rectangular transpose";
    c.accumulator = "none";
    c.numerical = "bit-preserving BF16 copy";
    c.effects = "read source, write destination";
    c.estimator = u.qualify("brgemm-tile");
    c.costParameters = {"vector_bits", "workers"};
    c.available = [](const TargetProfile &p) {
      return p.checkImplementation("exits/mojo::onednn-brgemm-bf16");
    };
    c.verify = [](Operation *op) {
      return success(op->getAttrOfType<IntegerAttr>("columns").getInt() > 0 &&
                     op->getAttrOfType<IntegerAttr>("pitch").getInt() > 0);
    };
    TypedContract t;
    t.shape.minOperands = t.shape.maxOperands = 3;
    t.shape.minResults = t.shape.maxResults = 0;
    t.operandTypes = {ScalarTypeKind::Index};
    t.effects = EffectMask::ReadWrite;
    c.typed = std::move(t);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }
  {
    ImplementationContract c;
    c.name = "bf16-value-prefix";
    c.operation = "mojo_row.value_prefix";
    c.symbol = "intent_brgemm_bf16_prefix";
    c.source = ImplementationSource::External;
    c.minimum = Reuse::Vendor;
    c.dependencies = {"onednn"};
    c.operands = "BF16 weights[M,lda], VNNI packed BF16 [K/2,N,2], F32 "
                 "output[M,ldc], active K, QK context and scratch";
    c.layout = "dense weight rows and interleaved value pairs";
    c.accumulator = "F32";
    c.effects = "reads active prefix, writes output and scratch, restores the "
                "QK hardware context";
    c.numerical = "BF16 pair dot with f32 accumulator and flushed subnormals";
    c.estimator = u.qualify("brgemm-tile");
    c.costParameters = {"vector_bits", "workers"};
    c.available = [](const TargetProfile &p) {
      return p.checkImplementation("exits/mojo::onednn-brgemm-bf16");
    };
    c.verify = [](Operation *op) {
      auto dim = [&](StringRef name) {
        return op->getAttrOfType<IntegerAttr>(name).getInt();
      };
      return success(dim("rows") > 0 && dim("columns") > 0 && dim("lda") > 0 &&
                     dim("ldc") >= dim("columns") &&
                     intentdsl::kir::permitsBf16Dot(op));
    };
    TypedContract t;
    t.shape.minOperands = t.shape.maxOperands = 6;
    t.shape.minResults = t.shape.maxResults = 1;
    t.operandTypes = {ScalarTypeKind::Index};
    t.resultTypes = {ScalarTypeKind::I32};
    t.accumulatorTypes = {ScalarTypeKind::F32};
    t.effects = EffectMask::ReadWrite;
    // The verifier checks the resolved BF16 dot contract.
    c.typed = std::move(t);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }
  makeBrgemm(
      "onednn-brgemm-s8", "s8 x s8 -> s32",
      "LHS row-major (unit K stride, K%4==0), RHS VNNI pack32 [K/4, nr, 4]",
      "s32 AMX/VNNI tile accumulator",
      "exact two's-complement s8*s8->s32 accumulation",
      false, ScalarTypeKind::I32,
      [](const TargetProfile &p) -> llvm::Error {
        if (!p.features.count("amx_int8") && !p.features.count("avx512_vnni"))
          return llvm::createStringError(
              "oneDNN s8 BRGeMM requires AMX-INT8 or AVX512-VNNI");
        return llvm::Error::success();
      });

  makeBrgemm(
      "onednn-brgemm-u8s8", "u8 x s8 -> s32 (K=32 Q4_K subgroup)",
      "LHS unpacked u8[R, 32] nibbles, RHS Q8_K s8[32] subgroup",
      "s32 subgroup accumulator",
      "exact u8*s8->s32 32-element subgroup dot preserving Q4_K super-block R32 contract",
      false, ScalarTypeKind::I32,
      [](const TargetProfile &p) -> llvm::Error {
        if (!p.features.count("amx_int8") && !p.features.count("avx512_vnni"))
          return llvm::createStringError(
              "oneDNN u8s8 BRGeMM requires AMX-INT8 or AVX512-VNNI");
        return llvm::Error::success();
      });

  auto requireSimd = [](const TargetProfile &p) -> llvm::Error {
    auto *rm = p.services.get<intentdsl::cpu::RowMachine>();
    if (!rm || rm->vectorBits < 128)
      return llvm::createStringError("requires >=128-bit CPU SIMD");
    return llvm::Error::success();
  };

  {
    ImplementationContract c;
    c.name = "philox-uniform";
    c.operation = "mojo_row.uniform";
    c.symbol = "uniform_at";
    c.source = ImplementationSource::Runtime;
    c.resource = "runtime/mojo/kernels/philox.mojo";
    c.minimum = Reuse::Native;
    c.operands = "ui64 seed x ui64[W] counter -> f32[W]";
    c.layout = "SIMD register lanes";
    c.accumulator = "u32x4[W] 10-round Philox-4x32 state";
    c.effects = "pure register arithmetic";
    c.numerical = "10-round Philox-4x32 upper-24-bit mantissa * 2^-24 in [0, 1)";
    c.estimator = u.qualify("simd-row-primitive");
    c.costParameters = {"vector_bits", "workers"};
    c.available = requireSimd;
    c.verify = [](Operation *op) -> LogicalResult {
      if (op->getNumOperands() != 2 || op->getNumResults() != 1 ||
          !op->getOperand(0).getType().isUnsignedInteger(64))
        return failure();
      auto wAttr = op->getAttrOfType<IntegerAttr>("width");
      if (!wAttr || wAttr.getInt() < 1)
        return failure();
      if (wAttr.getInt() == 1 && op->getOperand(1).getType().isUnsignedInteger(64) &&
          op->getResult(0).getType().isF32())
        return success();
      auto out = dyn_cast<ShapedType>(op->getResult(0).getType());
      if (op->getOperand(1).getType().isUnsignedInteger(64))
        return success(out && out.getElementType().isF32() &&
                       out.getRank() == 1 &&
                       out.getDimSize(0) == wAttr.getInt());
      auto pos = dyn_cast<ShapedType>(op->getOperand(1).getType());
      return success(pos && out && pos.getElementType().isUnsignedInteger(64) &&
                     out.getElementType().isF32() &&
                     pos.getShape() == out.getShape());
    };
    TypedContract tc;
    tc.shape.minOperands = 2;
    tc.shape.maxOperands = 2;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    tc.operandTypes = {ScalarTypeKind::I64};
    tc.resultTypes = {ScalarTypeKind::F32};
    tc.accumulatorTypes = {ScalarTypeKind::I32};
    tc.effects = EffectMask::None;
    c.typed = std::move(tc);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }

  {
    ImplementationContract c;
    c.name = "simd-reduce-16x16";
    c.operation = "mojo_row.reduce_sum_16x16";
    c.symbol = "reduce_sum_16x16";
    c.source = ImplementationSource::Runtime;
    c.resource = "runtime/mojo/kernels/ordered_reduce.mojo";
    c.minimum = Reuse::Native;
    c.operands = "16 x f32[16] -> f32[16]";
    c.layout = "16 SIMD width-16 vectors reduced pairwise into 1 width-16 vector";
    c.accumulator = "f32 pairwise butterfly tree sum";
    c.effects = "pure register arithmetic";
    c.numerical = "exact IEEE-754 height-4 balanced pairwise binary tree sum";
    c.estimator = u.qualify("simd-row-primitive");
    c.costParameters = {"vector_bits", "workers"};
    c.available = requireSimd;
    c.verify = [](Operation *op) -> LogicalResult {
      if (op->getNumOperands() != 16 || op->getNumResults() != 1)
        return failure();
      auto out = dyn_cast<ShapedType>(op->getResult(0).getType());
      return success(out && out.getRank() == 1 && out.getDimSize(0) == 16 &&
                     out.getElementType().isF32());
    };
    TypedContract tc;
    tc.shape.minOperands = 16;
    tc.shape.maxOperands = 16;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    tc.operandTypes = {ScalarTypeKind::F32};
    tc.resultTypes = {ScalarTypeKind::F32};
    tc.accumulatorTypes = {ScalarTypeKind::F32};
    tc.effects = EffectMask::None;
    c.typed = std::move(tc);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }

  {
    ImplementationContract c;
    c.name = "welford-step";
    c.operation = "mojo_row.welford_vec_step";
    c.symbol = "welford_vec_step";
    c.source = ImplementationSource::Runtime;
    c.resource = "runtime/mojo/kernels/moments.mojo";
    c.minimum = Reuse::Native;
    c.operands = "f32[W] samples, f32[W] mean, f32[W] m2, i32 step_n -> (f32[W], f32[W])";
    c.layout = "SIMD register lanes";
    c.accumulator = "f32[W] running mean and M2 vectors";
    c.effects = "pure register arithmetic";
    c.numerical = "IEEE-754 f32 per-lane Welford step with shared reciprocal 1/step_n";
    c.estimator = u.qualify("simd-row-primitive");
    c.costParameters = {"vector_bits", "workers"};
    c.available = requireSimd;
    c.verify = [](Operation *op) -> LogicalResult {
      return success(op->getNumOperands() == 4 && op->getNumResults() == 2 &&
                     op->getOperand(3).getType().isInteger(32));
    };
    TypedContract tc;
    tc.shape.minOperands = 4;
    tc.shape.maxOperands = 4;
    tc.shape.minResults = 2;
    tc.shape.maxResults = 2;
    tc.operandTypes = {ScalarTypeKind::F32, ScalarTypeKind::I32};
    tc.resultTypes = {ScalarTypeKind::F32};
    tc.accumulatorTypes = {ScalarTypeKind::F32};
    tc.effects = EffectMask::None;
    c.typed = std::move(tc);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }

  {
    ImplementationContract c;
    c.name = "welford-reduce";
    c.operation = "mojo_row.welford_vec_reduce";
    c.symbol = "welford_vec_reduce";
    c.source = ImplementationSource::Runtime;
    c.resource = "runtime/mojo/kernels/moments.mojo";
    c.minimum = Reuse::Native;
    c.operands = "i32 step_n, f32[W] mean, f32[W] m2 -> tuple<i32, f32, f32>";
    c.layout = "SIMD register lanes";
    c.accumulator = "f32 pairwise Chan Welford tree merge";
    c.effects = "pure register arithmetic";
    c.numerical = "exact power-of-two pairwise Chan Welford merge across W lanes";
    c.estimator = u.qualify("simd-row-primitive");
    c.costParameters = {"vector_bits", "workers"};
    c.available = requireSimd;
    c.verify = [](Operation *op) -> LogicalResult {
      return success(op->getNumOperands() == 3 && op->getNumResults() == 1 &&
                     isa<TupleType>(op->getResult(0).getType()));
    };
    TypedContract tc;
    tc.shape.minOperands = 3;
    tc.shape.maxOperands = 3;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    tc.operandTypes = {ScalarTypeKind::I32, ScalarTypeKind::F32};
    tc.resultTypes = {ScalarTypeKind::Any};
    tc.accumulatorTypes = {ScalarTypeKind::F32};
    tc.effects = EffectMask::None;
    c.typed = std::move(tc);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }

  {
    ImplementationContract c;
    c.name = "welford-combine";
    c.operation = "mojo_row.welford_scalar_step";
    c.symbol = "welford_scalar_step";
    c.source = ImplementationSource::Runtime;
    c.resource = "runtime/mojo/kernels/moments.mojo";
    c.minimum = Reuse::Native;
    c.operands = "tuple<i32, f32, f32> state, f32 sample -> tuple<i32, f32, f32>";
    c.layout = "scalar registers";
    c.accumulator = "(i32 count, f32 mean, f32 m2) scalar summary";
    c.effects = "pure register arithmetic";
    c.numerical = "IEEE-754 f32 scalar Welford step";
    c.estimator = u.qualify("simd-row-primitive");
    c.costParameters = {"vector_bits", "workers"};
    c.available = requireSimd;
    c.verify = [](Operation *op) -> LogicalResult {
      return success(op->getNumOperands() == 2 && op->getNumResults() == 1 &&
                     isa<TupleType>(op->getOperand(0).getType()) &&
                     op->getOperand(1).getType().isF32() &&
                     isa<TupleType>(op->getResult(0).getType()));
    };
    TypedContract tc;
    tc.shape.minOperands = 2;
    tc.shape.maxOperands = 2;
    tc.shape.minResults = 1;
    tc.shape.maxResults = 1;
    tc.operandTypes = {ScalarTypeKind::Any};
    tc.resultTypes = {ScalarTypeKind::Any};
    tc.accumulatorTypes = {ScalarTypeKind::F32};
    tc.effects = EffectMask::None;
    c.typed = std::move(tc);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }

  {
    ImplementationContract c;
    c.name = "simd-transpose-8x8";
    c.operation = "mojo_row.transpose_8x8";
    c.symbol = "transpose_tile_8x8";
    c.source = ImplementationSource::Runtime;
    c.resource = "runtime/mojo/kernels/transpose.mojo";
    c.minimum = Reuse::Native;
    c.operands = "src ptr, src_off, src_s0, dst ptr, dst_off, dst_s0";
    c.layout = "8x8 contiguous-row sub-block within 32x32 matrix tile";
    c.accumulator = "8x SIMD[dt, 8] register rows";
    c.effects = "reads 8x8 source block and writes transposed 8x8 destination block";
    c.numerical = "bitwise exact element permutation";
    c.estimator = u.qualify("simd-tile-transpose");
    c.costParameters = {"vector_bits", "workers"};
    c.available = requireSimd;
    c.verify = [](Operation *op) -> LogicalResult {
      return success(op->getNumOperands() == 6 && op->getNumResults() == 0);
    };
    TypedContract tc;
    tc.shape.minOperands = 6;
    tc.shape.maxOperands = 6;
    tc.shape.minResults = 0;
    tc.shape.maxResults = 0;
    tc.operandTypes = {ScalarTypeKind::Any};
    tc.effects = EffectMask::ReadWrite;
    c.typed = std::move(tc);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }

  {
    ImplementationContract c;
    c.name = "simd-scan-sum";
    c.operation = "mojo_row.simd_scan_sum";
    c.symbol = "simd_scan_sum";
    c.source = ImplementationSource::Runtime;
    c.resource = "runtime/mojo/kernels/scan.mojo";
    c.minimum = Reuse::Native;
    c.operands = "T[W] input, T carry -> (T[W] prefix, T next_carry)";
    c.layout = "SIMD register lanes";
    c.accumulator = "T[W] inclusive Hillis-Steele prefix sum";
    c.effects = "pure register arithmetic";
    c.numerical = "log2(W) SIMD shift-add inclusive prefix with interval carry";
    c.estimator = u.qualify("simd-row-primitive");
    c.costParameters = {"vector_bits", "workers"};
    c.available = requireSimd;
    c.verify = [](Operation *op) -> LogicalResult {
      return success(op->getNumOperands() == 2 && op->getNumResults() == 2);
    };
    TypedContract tc;
    tc.shape.minOperands = 2;
    tc.shape.maxOperands = 2;
    tc.shape.minResults = 2;
    tc.shape.maxResults = 2;
    tc.operandTypes = {ScalarTypeKind::F32, ScalarTypeKind::I32,
                       ScalarTypeKind::I64};
    tc.resultTypes = {ScalarTypeKind::F32, ScalarTypeKind::I32,
                      ScalarTypeKind::I64};
    tc.accumulatorTypes = {ScalarTypeKind::F32, ScalarTypeKind::I32,
                           ScalarTypeKind::I64};
    tc.effects = EffectMask::None;
    c.typed = std::move(tc);
    populateTypedContractSummaries(c);
    add(std::move(c));
  }
}

void populateMojoMatrixContracts(intentdsl::cpu::MatrixMachine &m, int64_t v) {
  auto add = [&](int64_t r, int64_t c, int64_t d, int64_t lanes,
                 llvm::StringRef id, llvm::StringRef est, llvm::StringRef dtype,
                 int64_t kAlign = 1, bool contigK = false, bool ftz = false) {
    intentdsl::cpu::MatrixImplementation impl;
    impl.rows = r;
    impl.cols = c;
    impl.depth = d;
    impl.lanes = lanes;
    impl.id = ("exits/mojo::" + id).str();
    impl.estimator = ("exits/mojo::" + est).str();
    impl.dtype = dtype.str();
    impl.kAlign = kAlign;
    impl.requiresContiguousLhsK = contigK;
    impl.flushesSubnormals = ftz;
    m.contracts.push_back(std::move(impl));
  };

  for (auto [r, c, d] : std::initializer_list<std::tuple<int64_t, int64_t, int64_t>>{
           {2, v, 32}, {4, 2 * v, 64}, {8, 2 * v, 64}, {8, 4 * v, 64},
           {8, 3 * v, 128}, {12, 2 * v, 128}}) {
    add(r, c, d, v, "simd-fma", "register-block", "any");
    if (!m.integerPacking.empty())
      add(r, c, d, v, "vnni-dot4", "register-block", "vnni-dot4", 4, true,
          false);
  }
  for (int64_t r : {16, 8, 64}) {
    if (!m.integerPacking.empty())
      add(r, 1, 256, v, "q4k-vnni", "q4k-simd-row", "q4k");
    add(r, 1, 256, v, "q4k-simd", "q4k-simd-row", "q4k");
  }

  // Vendor oneDNN BRGeMM candidates (AVX-512 32-ZMM bounded vs AMX 8-TMM)
  for (auto [r, c, d] : std::initializer_list<std::tuple<int64_t, int64_t, int64_t>>{
           {4, 2 * v, 64}, {8, 2 * v, 64}, {12, 2 * v, 64}, {14, 2 * v, 64},
           {6, 4 * v, 64}, {12, 4 * v, 64}, {8, 3 * v, 64}, {12, 3 * v, 64}}) {
    add(r, c, d, v, "onednn-brgemm-f32", "brgemm-tile", "f32", 1, true, false);
    add(r, c, d, v, "onednn-brgemm-f16", "brgemm-tile", "f16", 1, true, false);
  }
  for (auto [r, c, d] :
       std::initializer_list<std::tuple<int64_t, int64_t, int64_t>>{
           {16, 2 * v, 64}, {32, 2 * v, 64}, {16, 4 * v, 64}, {32, 4 * v, 64}}) {
    add(r, c, d, v, "onednn-brgemm-bf16", "brgemm-tile", "bf16", 2, true, true);
    add(r, c, d, v, "onednn-brgemm-s8", "brgemm-tile", "s8", 4, true, false);
  }
  add(16, 1, 256, v, "onednn-brgemm-u8s8", "brgemm-tile", "q4k", 32, false,
      false);
}
} // namespace CI_UNIT_NAMESPACE
