#include "Intent/Target/DSA/BangC/BlockLowering.h"
#include "Intent/Conversion/KIRToBlock/KIRToBlock.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Block/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/Block/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
namespace intent::bangc {
LogicalResult constructBlockStorage(ModuleOp module, physical::ConfigurationAttr config,
    physical::CapabilitiesAttr hardware, DictionaryAttr shapes, DictionaryAttr strides) {
  if (failed(lowerCanonicalKIRToBlock(module, {hardware, true, false, true, false, true}))) return failure();
  Builder builder(module.getContext());
  SmallVector<Attribute> columns;
  for (StringRef name : {"ownership_m", "ownership_n", "reduction", "reduction_outer", "scan", "traversal_workers", "traversal_group"})
    columns.push_back(builder.getStringAttr(name));
  NamedAttrList families;
  // Project one complete BANG C candidate onto shared configuration families.
  // Sequential state uses the declared region working-set extent; streaming
  // pointwise and reduction use the declared tile extent.
  for (StringRef name : {"persistent_contraction_narrow", "persistent_contraction",
      "contraction_narrow", "contraction", "region_contraction_multi_accumulator",
      "region_contraction_f32", "region_contraction_small_rows", "region_contraction",
      "region_reduction", "scan", "multi_axis_reduction", "histogram", "reduction",
      "stateful_reduction", "execution", "online_moment", "pointwise_reduction_two_axis",
      "pointwise_reduction", "pointwise_ordered", "pointwise_only_fixed_local", "pointwise_fixed_local",
      "pointwise_two_axis_narrow", "pointwise_two_axis", "pointwise_narrow", "pointwise"}) {
    bool matrix = name.contains("contraction");
    bool stateful = name == "pointwise_ordered" || name == "pointwise_only_fixed_local" || name == "pointwise_fixed_local";
    auto row = builder.getDenseI64ArrayAttr({matrix ? config.getTileM() : 1,
        matrix ? config.getTileN() : stateful ? config.getRegionTile() : config.getTile(),
        matrix ? config.getTileK() : config.getTile(), 1, config.getRegionTile(), 1, 1});
    families.append(name, builder.getArrayAttr({row}));
  }
  auto table = block::TuningProfileTableAttr::get(module.getContext(), builder.getArrayAttr(columns), families.getDictionary(module.getContext()));
  module->setAttr(block::tuningProfilesAttr, block::TuningProfilesAttr::get(module.getContext(),
      builder.getDictionaryAttr({builder.getNamedAttr("bangc", table)})));
  PassManager manager(module.getContext());
  block::buildSharedBlockPipeline(manager);
  if (failed(manager.run(module))) return failure();
  auto kernel = block::getPhysicalKernel(module);
  if (failed(kernel)) return failure();
  auto space = block::ParameterSpace::read(*kernel);
  if (failed(space)) return failure();
  auto rows = space->configurations(block::ConfigurationStage::Shared);
  if (failed(rows) || rows->size() != 1)
    return module.emitError("one BANG C candidate must retain one complete Block configuration");
  return lowerBlockToBangC(module, config, rows->front(), shapes, strides);
}
}
