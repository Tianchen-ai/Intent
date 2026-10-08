#include "Intent/Compiler/Passes.h"
#include "Intent/Compiler/Backend.h"
#include "Intent/Conversion/KIRToCPU/KIRToCPU.h"
#include "Intent/Conversion/KIRToDSA/KIRToDSA.h"
#include "Intent/Conversion/KIRToGPU/KIRToGPU.h"
#include "Intent/Dialect/CPU/IR/CPUDialect.h"
#include "Intent/Dialect/GPU/IR/GPUDialect.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/TuningProfiles.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace intent::compiler {
#define GEN_PASS_DEF_CONSTRUCTGPU
#define GEN_PASS_DEF_CONSTRUCTCPU
#define GEN_PASS_DEF_CONSTRUCTDSA
#define GEN_PASS_DEF_RESOLVEGPUPROFILES
#include "Intent/Compiler/Passes.h.inc"

namespace {

class ConstructGPUPass : public impl::ConstructGPUBase<ConstructGPUPass> {
public:
  using Base::Base;
  void runOnOperation() final {
    GPUCapabilities capabilities{computeUnits, sharedMemoryPerUnit,
        maxDynamicSharedMemoryPerBlock, registersPerUnit, maxThreadsPerBlock,
        computeCapabilityMajor, computeCapabilityMinor,
        singleToDoublePrecisionPerfRatio, matrixUnits, dynamicVectorWidth,
        nativeTupleReductions, nativeTupleReductionRequiresConstantIdentity,
        nativeFragmentGather};
    if (failed(lowerCanonicalKIRToGPU(getOperation(), capabilities)))
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

class ConstructDSAPass : public impl::ConstructDSABase<ConstructDSAPass> {
public:
  using Base::Base;
  void runOnOperation() final {
    auto module = getOperation();
    auto configurations = readDSAConfigurations(module, directory.getValue(), overrides.getValue());
    auto shapeBindings = parseDSAParameterBindings(module, shapes, true);
    auto strideBindings = parseDSAParameterBindings(module, strides, false);
    if (failed(configurations) || failed(shapeBindings) || failed(strideBindings))
      return signalPassFailure();
    SmallVector<OwningOpRef<ModuleOp>> candidates;
    StringAttr entryName;
    for (auto [index, attribute] : llvm::enumerate(*configurations)) {
      OwningOpRef<ModuleOp> candidate(cast<ModuleOp>(module->clone()));
      if (failed(lowerCanonicalKIRToDSA(*candidate, cast<dsa::ConfigurationAttr>(attribute),
                                       *shapeBindings, *strideBindings)))
        return signalPassFailure();
      func::FuncOp entry;
      for (auto function : candidate->getOps<func::FuncOp>())
        if (!function.isExternal()) { entry = function; break; }
      if (!entry) {
        module.emitError("DSA configuration has no executable entry");
        return signalPassFailure();
      }
      if (!entryName) entryName = StringAttr::get(module.getContext(), entry.getName());
      SymbolTable symbols(*candidate);
      if (failed(symbols.rename(entry, entryName.getValue().str() + "_config_" + std::to_string(index))))
        return signalPassFailure();
      (*candidate)->setAttr(SymbolTable::getSymbolAttrName(),
                           StringAttr::get(module.getContext(), "config_" + std::to_string(index)));
      candidates.push_back(std::move(candidate));
    }
    module.getBody()->clear();
    module->setAttr("intent.entry_name", entryName);
    for (auto &candidate : candidates) module.getBody()->push_back(candidate.release());
  }
};

class ResolveGPUProfilesPass : public impl::ResolveGPUProfilesBase<ResolveGPUProfilesPass> {
public:
  using Base::Base;
  void runOnOperation() final {
    auto module = getOperation();
    if (directory.empty()) {
      module.emitError("GPU profile resolution requires a resource directory");
      return signalPassFailure();
    }
    auto sources = gpuProfileSources(directory.getValue(), provider.getValue());
    if (sources.size() != 1) {
      module.emitError("GPU profiles require an available selected provider");
      return signalPassFailure();
    }
    auto profiles = gpu::TuningProfiles::read(module.getLoc(), sources, overrides);
    if (failed(profiles)) return signalPassFailure();
    profiles->attach(module);
    auto kernel = gpu::getPhysicalKernel(module);
    if (failed(kernel) || failed(profiles->declareProviderParameters(*kernel, sources.front().schema)))
      signalPassFailure();
  }
};

} // namespace

#define GEN_PASS_REGISTRATION
#include "Intent/Compiler/Passes.h.inc"
void registerCompilationPasses() { registerIntentCompilerPasses(); }

} // namespace intent::compiler
