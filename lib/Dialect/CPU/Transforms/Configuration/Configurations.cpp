#include "Intent/Dialect/CPU/Transforms/Configuration/Configuration.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "TuningProfiles.h"

using namespace mlir;

namespace intent::cpu {

LogicalResult materializeCPUConfigurations(
    ModuleOp module, const ImplementationRegistry &implementations,
    StringRef defaults, StringRef overrides) {
  auto profiles = TuningProfiles::read(module, defaults, overrides, implementations);
  if (failed(profiles))
    return failure();
  auto capabilities = module->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities");
  if (!capabilities)
    return module.emitError("CPU candidate formation requires declared target capabilities");
  auto functions = module.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(functions))
    return module.emitError("CPU candidate formation requires one executable source function");
  func::FuncOp original = *functions.begin();
  if (original.isExternal())
    return original.emitError("CPU candidate formation requires an executable source function");
  StringRef family = implementations.profile(original);
  auto rows = profiles->get(family);
  if (rows.empty())
    return original.emitError("CPU candidate family is empty or missing: ") << family;

  bool hasContraction = false, hasRegion = false;
  original.walk([&](Operation *operation) {
    if (auto generic = dyn_cast<linalg::GenericOp>(operation))
      hasContraction |= isMatrixContraction(generic);
    hasRegion |= isa<RegionFoldOp, RegionScanOp>(operation);
  });
  struct Candidate {
    Configuration configuration;
    SmallVector<ImplementationAttr> bindings;
  };
  struct Rejection {
    Location location;
    size_t row;
    std::string operation, implementation, reason;
  };
  SmallVector<Candidate> candidates;
  SmallVector<Rejection> rejections;
  for (auto [row, configuration] : llvm::enumerate(rows)) {
    auto rejected = [&](Operation *operation, StringRef implementation, StringRef reason) {
      rejections.push_back({operation->getLoc(), row, operation->getName().getStringRef().str(),
                            implementation.str(), reason.str()});
    };
    if (!hasContraction && (configuration.tileM != 1 || configuration.tileN != 1 ||
                            configuration.tileK != 1)) {
      return original.emitError("M/N/K block parameters require a matrix contraction consumer; otherwise they must be 1")
          << "; profile '" << family << "' row " << row + 1;
    }
    if (!hasRegion && configuration.regionSize != 1) {
      return original.emitError("region size requires a region consumer; otherwise it must be 1")
          << "; profile '" << family << "' row " << row + 1;
    }
    auto bindings = implementations.select(original, capabilities, configuration, rejected);
    if (failed(bindings))
      continue;
    if (llvm::any_of(candidates, [&](const Candidate &previous) {
          const auto &other = previous.configuration;
          return other.taskGrain == configuration.taskGrain &&
                 other.tileM == configuration.tileM && other.tileN == configuration.tileN &&
                 other.tileK == configuration.tileK && other.regionSize == configuration.regionSize &&
                 previous.bindings == *bindings;
        }))
      continue;
    candidates.push_back({configuration, std::move(*bindings)});
  }
  if (candidates.empty()) {
    auto diagnostic = original.emitError("no legal CPU candidates remain for family '");
    diagnostic << family << "'";
    for (const auto &rejection : rejections) {
      const auto &configuration = rows[rejection.row];
      auto &note = diagnostic.attachNote(rejection.location);
      note << "profile '" << family << "' row " << rejection.row + 1 << " shared=["
           << configuration.taskGrain << ", " << configuration.tileM << ", "
           << configuration.tileN << ", " << configuration.tileK << ", "
           << configuration.regionSize << "] local=" << configuration.local
           << " implementations=" << configuration.implementations
           << ", computation " << rejection.operation;
      if (!rejection.implementation.empty()) note << ", implementation '" << rejection.implementation << "'";
      note << ": " << rejection.reason;
    }
    return failure();
  }

  // Each accepted profile row forms one complete executable candidate.
  for (auto [number, candidate] : llvm::enumerate(candidates)) {
    auto function = cast<func::FuncOp>(original->clone());
    function.setName(original.getName().str() + "_config_" + std::to_string(number));
    module.push_back(function);
    if (failed(implementations.bind(function, capabilities, candidate.configuration,
                                    candidate.bindings)))
      return failure();
  }
  original.erase();
  return verifyCPUProgram(module, CPUProgramStage::Buffers);
}

} // namespace intent::cpu
