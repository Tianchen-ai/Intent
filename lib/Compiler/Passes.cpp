#include "Intent/Compiler/Passes.h"
#include "Intent/Compiler/Backend.h"
#include "Intent/Conversion/KIRToCPU/KIRToCPU.h"
#include "Intent/Conversion/KIRToBlock/KIRToBlock.h"
#include "Intent/Dialect/CPU/IR/CPUDialect.h"
#include "Intent/Dialect/Block/IR/BlockDialect.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Block/Transforms/Configuration/TuningProfiles.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/AsmParser/AsmParser.h"

using namespace mlir;

namespace intent::compiler {
#define GEN_PASS_DEF_CONSTRUCTBLOCK
#define GEN_PASS_DEF_CONSTRUCTCPU
#define GEN_PASS_DEF_RESOLVEBLOCKPROFILES
#include "Intent/Compiler/Passes.h.inc"

namespace {

class ConstructBlockPass : public impl::ConstructBlockBase<ConstructBlockPass> {
public:
  using Base::Base;
  void runOnOperation() final {
    auto selectedTarget = parseAttribute(target, &getContext());
    if (!selectedTarget) return signalPassFailure();
    BlockCapabilities capabilities{selectedTarget, matrixUnits, dynamicVectorWidth,
        nativeTupleReductions, nativeTupleReductionRequiresConstantIdentity,
        nativeFragmentGather};
    if (failed(lowerCanonicalKIRToBlock(getOperation(), capabilities)))
      signalPassFailure();
  }
};

class ConstructCPUPass : public impl::ConstructCPUBase<ConstructCPUPass> {
public:
  using Base::Base;
  void runOnOperation() final {
    if (failed(lowerCanonicalKIRToCPU(getOperation(), stridedInputs
            ? CPUEntryLayout::StridedInputs : CPUEntryLayout::Contiguous)))
      signalPassFailure();
  }
};

class ResolveBlockProfilesPass : public impl::ResolveBlockProfilesBase<ResolveBlockProfilesPass> {
public:
  using Base::Base;
  void runOnOperation() final {
    auto module = getOperation();
    if (directory.empty()) {
      module.emitError("Block profile resolution requires a resource directory");
      return signalPassFailure();
    }
    auto sources = blockProfileSources(directory.getValue(), provider.getValue());
    if (sources.size() != 1) {
      module.emitError("Block profiles require an available selected provider");
      return signalPassFailure();
    }
    auto profiles = block::TuningProfiles::read(module.getLoc(), sources, overrides);
    if (failed(profiles)) return signalPassFailure();
    profiles->attach(module);
    auto kernel = block::getPhysicalKernel(module);
    if (failed(kernel) || failed(profiles->declareProviderParameters(*kernel, sources.front().schema)))
      signalPassFailure();
  }
};

} // namespace

#define GEN_PASS_REGISTRATION
#include "Intent/Compiler/Passes.h.inc"
void registerCompilationPasses() { registerIntentCompilerPasses(); }

} // namespace intent::compiler
