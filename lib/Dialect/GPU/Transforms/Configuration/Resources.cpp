#include "Intent/Dialect/GPU/Transforms/Configuration/Resources.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {

FailureOr<SmallVector<DictionaryAttr>> filterConfigurationRequirements(
    func::FuncOp kernel, ArrayRef<DictionaryAttr> rows,
    ArrayRef<ConfigurationRequirementAttr> requirements) {
  auto options = readCompileOptions(kernel);
  if (failed(options)) return failure();
  struct Rejection {
    DictionaryAttr row;
    ConfigurationRequirementAttr requirement;
    RequirementEvaluation evaluation;
  };
  SmallVector<Rejection> rejected;
  SmallVector<DictionaryAttr> accepted;
  for (DictionaryAttr row : rows) {
    bool valid = true;
    for (ConfigurationRequirementAttr requirement : requirements) {
      auto evaluation = evaluateConfigurationRequirement(requirement, row, kernel);
      if (evaluation.status == RequirementStatus::Unknown &&
          options->getOptimizationRemarks()) {
        kernel.emitRemark("configuration requirement deferred to invocation binding: ")
            << stringifyConfigurationRequirementKind(requirement.getKind())
            << ", " << stringifyConfigurationRequirementMetric(requirement.getMetric())
            << ", " << stringifyConfigurationRequirementPredicate(requirement.getPredicate())
            << "; " << requirement.getMessage().getValue() << "; bindings=" << row;
      }
      if (evaluation.status != RequirementStatus::Violated &&
          evaluation.status != RequirementStatus::Invalid)
        continue;
      valid = false;
      rejected.push_back({row, requirement, evaluation});
      if (options->getOptimizationRemarks()) {
        auto remark = kernel.emitRemark("configuration rejected: ");
        remark << stringifyConfigurationRequirementKind(requirement.getKind())
               << ", " << stringifyConfigurationRequirementMetric(requirement.getMetric())
               << ", " << stringifyConfigurationRequirementPredicate(requirement.getPredicate())
               << "; " << requirement.getMessage().getValue() << "; bindings=" << row;
        if (evaluation.usage) remark << "; usage=" << *evaluation.usage;
        if (evaluation.limit) remark << "; limit=" << *evaluation.limit;
      }
    }
    if (valid) accepted.push_back(row);
  }
  if (accepted.empty()) {
    auto diagnostic = kernel.emitError("no candidate satisfies the current configuration requirements");
    for (const Rejection &rejection : rejected) {
      auto &note = diagnostic.attachNote(kernel.getLoc());
      note << stringifyConfigurationRequirementKind(rejection.requirement.getKind())
           << ", " << stringifyConfigurationRequirementMetric(rejection.requirement.getMetric())
           << ", " << stringifyConfigurationRequirementPredicate(rejection.requirement.getPredicate())
           << "; " << rejection.requirement.getMessage().getValue()
           << "; bindings=" << rejection.row;
      if (rejection.evaluation.usage) note << "; usage=" << *rejection.evaluation.usage;
      if (rejection.evaluation.limit) note << "; limit=" << *rejection.evaluation.limit;
    }
    return failure();
  }
  return accepted;
}

} // namespace intent::gpu
