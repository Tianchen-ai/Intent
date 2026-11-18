#include "Intent/Target/CuTile/Transforms/Passes.h"
#include "Intent/Target/CuTile/Analysis/Program.h"
#include "Intent/Target/CuTile/IR/CuTileDialect.h"
#include "Intent/Target/CuTile/IR/Configuration.h"
#include "Intent/Dialect/Block/IR/BlockDialect.h"
#include "Intent/Dialect/Block/IR/BlockOps.h"
#include "Intent/Dialect/Block/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Block/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
namespace intent::cutile {
#define GEN_PASS_DEF_SELECTINDEXEDMATRIXCONFIGURATIONSPASS
#include "Intent/Target/CuTile/Transforms/Passes.h.inc"

namespace {
class SelectIndexedMatrixConfigurationsPass final
    : public impl::SelectIndexedMatrixConfigurationsPassBase<SelectIndexedMatrixConfigurationsPass> {
public:
  void runOnOperation() final {
    auto kernel = block::getPhysicalKernel(getOperation());
    if (failed(kernel)) return signalPassFailure();
    if (!hasDataIndexedMatrixSupply(*kernel)) return;
    auto space = block::ParameterSpace::read(*kernel);
    if (failed(space)) return signalPassFailure();
    auto rows = space->configurations(block::ConfigurationStage::Shared);
    auto options = readCompileOptions(getOperation());
    if (failed(rows) || failed(options)) return signalPassFailure();
    SmallVector<DictionaryAttr> selected;
    for (DictionaryAttr row : *rows) {
      auto access = row.getAs<IntegerAttr>(accessFormParameter);
      if (access && (access.getInt() == nativeNoTMAForm || access.getInt() == gatherAccessForm))
        selected.push_back(row);
    }
    if (selected.size() == rows->size()) return;
    // TMA supplies interleaved with data-indexed MMA operands make the native
    // compiler's scheduling work excessive in otherwise small kernels. Keep
    // the declared register/tile-load forms; no parameter columns are patched.
    if (selected.empty()) {
      kernel->emitError("data-indexed cuTile matrix supplies require a declared non-TMA configuration when indexed supply selection is enabled");
      return signalPassFailure();
    }
    if (options->getOptimizationRemarks())
      kernel->emitRemark("selected complete non-TMA configurations for data-indexed matrix supply");
    if (failed(block::writeConfigurations(*kernel, selected, block::ConfigurationStage::Shared)))
      signalPassFailure();
  }
};
} // namespace
} // namespace intent::cutile
