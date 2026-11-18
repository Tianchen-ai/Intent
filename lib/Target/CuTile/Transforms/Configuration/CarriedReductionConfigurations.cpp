#include "Intent/Target/CuTile/Transforms/Passes.h"
#include "Intent/Target/CuTile/IR/CuTileDialect.h"
#include "Intent/Dialect/Block/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Block/Analysis/Resources.h"
#include "Intent/Dialect/Block/IR/BlockDialect.h"
#include "Intent/Dialect/Block/IR/BlockOps.h"
#include "Intent/Dialect/Block/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
namespace intent::cutile {
#define GEN_PASS_DEF_SELECTCARRIEDREDUCTIONCONFIGURATIONSPASS
#include "Intent/Target/CuTile/Transforms/Passes.h.inc"

namespace {

block::ParameterAttr extentParameter(func::FuncOp kernel,
                                     block::FragmentType fragment, unsigned axis) {
  auto extent = dyn_cast<block::PhysicalExprAttr>(fragment.getShape()[axis]);
  if (!extent || extent.getKind() != block::PhysicalExprKind::Parameter)
    return {};
  return block::lookupParameter(kernel, extent.getParameterReference());
}

bool readsCarry(Value input, BlockArgument carry) {
  SmallVector<Value> pending{input};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (value == carry)
      return true;
    if (!visited.insert(value).second)
      continue;
    auto *producer = value.getDefiningOp();
    if (!producer || producer->getBlock() != carry.getOwner() ||
        producer->getNumRegions() || !isMemoryEffectFree(producer))
      continue;
    llvm::append_range(pending, producer->getOperands());
  }
  return false;
}

class SelectCarriedReductionConfigurationsPass final
    : public impl::SelectCarriedReductionConfigurationsPassBase<SelectCarriedReductionConfigurationsPass> {
public:
  void runOnOperation() final {
    auto kernel = block::getPhysicalKernel(getOperation());
    if (failed(kernel)) return signalPassFailure();
    llvm::SmallDenseSet<StringAttr> widths;
    kernel->walk([&](block::ReduceOp reduce) {
      auto loop = dyn_cast<scf::ForOp>(reduce->getParentOp());
      if (!loop || reduce.getAxes().size() != 1) return;
      for (Value input : reduce.getSources()) {
        auto fragment = dyn_cast<block::FragmentType>(input.getType());
        if (!fragment || fragment.getShape().size() != 2) continue;
        auto axis = reduce.getAxes()[0];
        if (axis < 0 || axis > 1) continue;
        auto full = extentParameter(*kernel, fragment, axis);
        auto width = extentParameter(*kernel, fragment, 1 - axis);
        if (!full || full.getRole() != block::ParameterRole::FullCoverage ||
            !width || !block::isPointwiseTraversalParameter(width)) continue;
        for (BlockArgument carry : loop.getRegionIterArgs()) {
          auto type = dyn_cast<block::FragmentType>(carry.getType());
          if (type && type.getShape() == fragment.getShape() && readsCarry(input, carry))
            widths.insert(width.getName());
        }
      }
    });
    if (widths.empty()) return;
    auto space = block::ParameterSpace::read(*kernel);
    if (failed(space)) return signalPassFailure();
    auto rows = space->configurations(block::ConfigurationStage::Shared);
    auto options = readCompileOptions(getOperation());
    if (failed(rows) || failed(options)) return signalPassFailure();
    SmallVector<DictionaryAttr> selected;
    for (DictionaryAttr row : *rows) {
      // The other axis is fully resident across every iteration. Wide output
      // fragments multiply that state and the native compiler's schedule.
      bool narrow = llvm::all_of(widths, [&](StringAttr width) {
        auto binding = row.getAs<IntegerAttr>(width);
        return binding && binding.getInt() <= 32;
      });
      if (narrow)
        selected.push_back(row);
    }
    if (selected.empty()) {
      if (options->getOptimizationRemarks())
        kernel->emitRemark("retained declared cuTile configurations because none bound a narrow loop-carried fragment");
      return;
    }
    if (selected.size() == rows->size()) return;
    if (options->getOptimizationRemarks())
      kernel->emitRemark("preferred narrow output fragments for a loop-carried full-axis reduction");
    if (failed(block::writeConfigurations(*kernel, selected,
                                         block::ConfigurationStage::Shared)))
      signalPassFailure();
  }
};
} // namespace
} // namespace intent::cutile
