#include "../PassSupport.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Bufferization.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"

using namespace mlir;

namespace intent::cpu {

#define GEN_PASS_DEF_CPUBUFFERIZEVALUES
#define GEN_PASS_DEF_CPUREUSEPRIVATESTORAGE
#define GEN_PASS_DEF_CPUREUSESCRATCHSTORAGE
#define GEN_PASS_DEF_CPUOPTIMIZEMEMORYACCESSES
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

namespace {

class BufferizeValuesPass
    : public impl::CPUBufferizeValuesBase<BufferizeValuesPass> {
public:
  using CPUBufferizeValuesBase::CPUBufferizeValuesBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(detail::verifyPassInput(module, getArgument(),
                                       CPUProgramStage::Values)) ||
        failed(detail::finishTransform(module, getArgument(),
                                        bufferizeValues(module))))
      signalPassFailure();
  }
};

class ReusePrivateStoragePass
    : public impl::CPUReusePrivateStorageBase<ReusePrivateStoragePass> {
public:
  using CPUReusePrivateStorageBase::CPUReusePrivateStorageBase;
  void runOnOperation() final {
    if (failed(detail::transformFunctions(getOperation(), getArgument(),
                                          reusePrivateStorage)))
      signalPassFailure();
  }
};

class ReuseScratchStoragePass
    : public impl::CPUReuseScratchStorageBase<ReuseScratchStoragePass> {
public:
  using CPUReuseScratchStorageBase::CPUReuseScratchStorageBase;
  void runOnOperation() final {
    if (failed(detail::transformFunctions(getOperation(), getArgument(),
            [&](func::FuncOp function) {
              return reuseScratchStorage(function, linearCapacity
                  ? ScratchRepresentation::LinearCapacity
                  : ScratchRepresentation::PreserveDescriptors);
            })))
      signalPassFailure();
  }
};

class OptimizeMemoryAccessesPass
    : public impl::CPUOptimizeMemoryAccessesBase<OptimizeMemoryAccessesPass> {
public:
  using CPUOptimizeMemoryAccessesBase::CPUOptimizeMemoryAccessesBase;
  void runOnOperation() final {
    if (failed(detail::transformFunctions(getOperation(), getArgument(),
                                          optimizeMemoryAccesses)))
      signalPassFailure();
  }
};

} // namespace
} // namespace intent::cpu
