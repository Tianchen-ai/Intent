#include "Intent/Target/CuTile/Transforms/Passes.h"
#include "Intent/Target/CuTile/IR/CuTileDialect.h"
#include "Intent/Target/CuTile/IR/Configuration.h"
#include "Intent/Dialect/Block/IR/BlockDialect.h"
#include "Intent/Dialect/Block/IR/BlockOps.h"
#include "Intent/Dialect/Block/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/Block/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "Intent/Analysis/ControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
namespace intent::cutile {
#define GEN_PASS_DEF_SELECTCLUSTERCONFIGURATIONSPASS
#include "Intent/Target/CuTile/Transforms/Passes.h.inc"

namespace {
bool hasComputedOperand(ValueRange operands) {
  SmallVector<Value> pending(operands);
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second) continue;
    auto fragment = dyn_cast<block::FragmentType>(value.getType());
    if (!fragment || fragment.getElementType().isInteger(1)) continue;
    Operation *producer = value.getDefiningOp();
    if (producer && isa<block::LoadOp>(producer)) continue;
    if (auto select = dyn_cast_or_null<block::SelectOp>(producer)) {
      pending.push_back(select.getTrueValue());
      pending.push_back(select.getFalseValue());
      continue;
    }
    // Follow numeric operands, not address/mask construction. Shape changes,
    // casts and control forwarding retain a direct load's supply. Arithmetic
    // makes the MMA input a computed tile that must be distributed as well.
    if (producer && isa<block::UnaryOp, block::BinaryOp, block::ContractOp,
                        block::ReduceOp, block::ScanOp>(producer))
      return true;
    auto incoming = queryControlFlowIncoming(value);
    if (incoming.complete && !incoming.edges.empty()) {
      for (const auto &edge : incoming.edges)
        if (edge.operand) pending.push_back(edge.operand->get());
    } else if (producer && !producer->getNumRegions()) {
      llvm::append_range(pending, producer->getOperands());
    }
  }
  return false;
}

class SelectClusterConfigurationsPass final
    : public impl::SelectClusterConfigurationsPassBase<SelectClusterConfigurationsPass> {
public:
  void runOnOperation() final {
    auto kernel = block::getPhysicalKernel(getOperation());
    if (failed(kernel)) return signalPassFailure();
    bool computed = false;
    kernel->walk([&](block::ContractOp matrix) {
      computed |= hasComputedOperand({matrix.getLhs(), matrix.getRhs()});
    });
    if (!computed) return;
    auto space = block::ParameterSpace::read(*kernel);
    if (failed(space)) return signalPassFailure();
    auto rows = space->configurations(block::ConfigurationStage::Shared);
    auto options = readCompileOptions(getOperation());
    if (failed(rows) || failed(options)) return signalPassFailure();
    SmallVector<DictionaryAttr> selected;
    for (DictionaryAttr row : *rows) {
      auto ctas = row.getAs<IntegerAttr>(ctasParameter);
      if (!ctas || ctas.getInt() == 1) selected.push_back(row);
    }
    if (selected.size() == rows->size()) return;
    // Keep complete author/configuration rows. Do not synthesize a replacement
    // CTA count or remove cluster candidates from ordinary load-fed GEMM.
    // Multi-CTA lowering of computed operands currently causes excessive
    // native compile work on both Hopper and Blackwell.
    if (selected.empty()) {
      kernel->emitError("computed cuTile matrix operands require a single-CTA configuration when cluster selection is enabled");
      return signalPassFailure();
    }
    if (options->getOptimizationRemarks())
      kernel->emitRemark("selected single-CTA complete configurations for computed matrix operands");
    if (failed(block::writeConfigurations(*kernel, selected,
                                         block::ConfigurationStage::Shared)))
      signalPassFailure();
  }
};
} // namespace
} // namespace intent::cutile
