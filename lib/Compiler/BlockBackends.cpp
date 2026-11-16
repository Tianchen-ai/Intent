#include "Intent/Compiler/Backend.h"
#include "Intent/Compiler/Passes.h"
#include "Intent/Dialect/Block/Transforms/Passes.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/Transforms/Configuration/PhysicalParameters.h"

using namespace mlir;

namespace intent::compiler {

void BlockBackend::buildConstruction(OpPassManager &manager, const Request &request) const {
  ConstructBlockOptions options;
  std::string target;
  llvm::raw_string_ostream stream(target);
  request.block.target.print(stream);
  options.target = target;
  options.matrixUnits = request.block.matrixUnits;
  options.dynamicVectorWidth = request.block.dynamicVectorWidth;
  options.nativeTupleReductions = nativeTupleReductions;
  options.nativeTupleReductionRequiresConstantIdentity =
      nativeTupleReductionRequiresConstantIdentity;
  options.nativeFragmentGather = nativeFragmentGather;
  manager.addPass(createConstructBlock(options));
}

void BlockBackend::buildShared(OpPassManager &manager, const Request &request, StringRef provider) const {
  ResolveBlockProfilesOptions options;
  options.directory = request.profileDirectory;
  options.overrides = request.tuningConfig;
  options.provider = provider.str();
  manager.addPass(createResolveBlockProfiles(options));
  block::buildSharedBlockPipeline(manager, preferredReductionResidentPrograms);
}

LogicalResult BlockBackend::verifySharedInput(ModuleOp module, const Request &request, StringRef provider) const {
  if (failed(block::verifyBlockProgram(module))) return failure();
  auto kernel = block::getPhysicalKernel(module);
  if (failed(kernel) || failed(block::verifySharedConfigTuples(*kernel))) return failure();
  const auto &device = request.block;
  auto expected = block::CapabilitiesAttr::getChecked([&] { return module.emitError(); },
      module.getContext(), device.target, device.matrixUnits,
      device.dynamicVectorWidth, nativeTupleReductions,
      nativeTupleReductionRequiresConstantIdentity,
      nativeFragmentGather);
  if (!expected) return failure();
  if ((*kernel)->getAttrOfType<block::CapabilitiesAttr>(block::capabilitiesAttr) != expected)
    return module.emitError("shared Block capabilities disagree with the requested device or provider primitive contract");
  auto resolved = block::TuningProfiles::from(module);
  if (failed(resolved)) return failure();
  auto tables = module->getAttrOfType<block::TuningProfilesAttr>(block::tuningProfilesAttr);
  if (tables.getSpaces().size() != 1 || !tables.getSpaces().get(provider))
    return module.emitError("shared Block configurations belong to a different provider");
  for (const auto &schema : {profiles}) {
    auto table = tables.getSpaces().getAs<block::TuningProfileTableAttr>(schema.space);
    if (!table)
      return module.emitError("shared Block program is missing the resolved profile namespace '")
          << schema.space << "'";
    for (NamedAttribute family : table.getFamilies())
      if (failed(resolved->get(schema, family.getName().getValue(), module.getLoc())))
        return failure();
  }
  return success();
}

} // namespace intent::compiler
