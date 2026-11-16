#include "Intent/Compiler/Registration.h"
#include "Intent/Compiler/Backend.h"
#include "Intent/Compiler/Passes.h"
#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Analysis/BufferStorage.h"
#include "Intent/Dialect/CPU/IR/ImplementationProvider.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Bufferization.h"
#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "Intent/Dialect/Block/IR/BlockDialect.h"
#include "Intent/Dialect/Block/IR/IntegerRanges.h"
#include "Intent/Dialect/Block/Transforms/Passes.h"
#include "Intent/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;

namespace intent::compiler {
namespace {
class CPUProviders final : public cpu::ImplementationProviderInterface {
public:
  explicit CPUProviders(Dialect *dialect)
      : ImplementationProviderInterface(dialect) {
    for (const Backend &adapter : backends())
      if (adapter.available)
        if (const auto *implementations = adapter.model->cpuImplementations())
          registries.try_emplace(adapter.name, *implementations);
  }

  FailureOr<const cpu::ImplementationRegistry *> lookup(StringRef name) const final {
    auto found = registries.find(name);
    if (found == registries.end()) return failure();
    return &found->second;
  }
private:
  llvm::StringMap<cpu::ImplementationRegistry> registries;
};
} // namespace

void registerDialects(DialectRegistry &registry) {
  registerAllDialects(registry);
  func::registerInlinerExtension(registry);
  cpu::registerExtentRelations(registry);
  registerBufferStorageInterfaces(registry);
  block::registerIntegerRangeInterfaces(registry);
  cpu::registerValueBufferizationInterfaces(registry);
  registry.insert<IntentDialect, block::IntentBlockDialect, cpu::IntentCPUDialect,
      dsa::IntentDSADialect>();
  registry.addExtension(+[](MLIRContext *, cpu::IntentCPUDialect *dialect) {
    dialect->addInterfaces<CPUProviders>();
  });
  for (const Backend &adapter : backends())
    if (adapter.available && adapter.registerDialects)
      adapter.registerDialects(registry);
}

void registerPasses() {
  static const bool registered = [] {
    registerTransformsPasses();
    registerIntentPasses();
    registerCompilationPasses();
    block::registerBlockPasses();
    cpu::registerCPUPasses();
    dsa::registerDSAPasses();
    dsa::registerDSAPipelines();
    for (const Backend &adapter : backends())
      if (adapter.available) adapter.registerPasses();
    return true;
  }();
  (void)registered;
}
} // namespace intent::compiler
