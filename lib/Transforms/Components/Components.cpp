#include "Intent/Transforms/Components.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace intent {
namespace {
struct RegisteredComponent {
  std::string owner, stage, description;
  bool required;
};

llvm::StringMap<RegisteredComponent> &catalog() {
  static llvm::StringMap<RegisteredComponent> components;
  return components;
}

class OptionalTransformation final
    : public PassWrapper<OptionalTransformation, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptionalTransformation)
  OptionalTransformation() : pipeline(ModuleOp::getOperationName()) {}
  explicit OptionalTransformation(std::unique_ptr<Pass> pass)
      : name(pass->getArgument().str()), pipeline(ModuleOp::getOperationName()) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    pass->printAsTextualPipeline(stream);
    transformation = text;
    pipeline.addPass(std::move(pass));
  }
  OptionalTransformation(const OptionalTransformation &other)
      : PassWrapper(other), name(other.name), pipeline(other.pipeline) {
    transformation = other.transformation.getValue();
  }
  StringRef getArgument() const final { return "intent-optional-transformation"; }
  StringRef getName() const final { return name; }
  StringRef getDescription() const final { return "Run an enabled complete transformation"; }
  LogicalResult initializeOptions(StringRef options,
      function_ref<LogicalResult(const Twine &)> errorHandler) final {
    if (failed(Pass::initializeOptions(options, errorHandler))) return failure();
    std::string diagnostic;
    llvm::raw_string_ostream stream(diagnostic);
    if (failed(parsePassPipeline(transformation, pipeline, stream)))
      return errorHandler(diagnostic);
    if (llvm::range_size(pipeline.getPasses()) != 1)
      return errorHandler("optional transformation requires exactly one registered component");
    name = pipeline.getPasses().begin()->getArgument().str();
    auto registered = catalog().find(name);
    if (registered == catalog().end() || registered->second.required)
      return errorHandler("optional transformation requires a registered optional component");
    return success();
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    pipeline.getDependentDialects(registry);
  }
  void runOnOperation() final {
    auto options = readCompileOptions(getOperation());
    if (failed(options)) return signalPassFailure();
    auto selected = options->getOptimizations().getAs<BoolAttr>(name);
    bool enabled = !selected || selected.getValue();
    if (options->getOptimizationRemarks())
      getOperation().emitRemark() << name << (enabled ? ": enabled" : ": disabled by compile options");
    if (!enabled) { markAllAnalysesPreserved(); return; }
    if (failed(runPipeline(pipeline, getOperation()))) signalPassFailure();
  }
private:
  Option<std::string> transformation{*this, "transformation",
      llvm::cl::desc("The complete optional MLIR pass with its typed options")};
  std::string name;
  OpPassManager pipeline;
};
} // namespace

void registerTransformation(TransformationComponent component,
                            std::unique_ptr<Pass> pass) {
  static PassRegistration<OptionalTransformation> optionalRegistration;
  StringRef name = pass->getArgument();
  if (name.empty() || component.owner.empty() || component.stage.empty())
    llvm::report_fatal_error("transformation components require a named pass, owner and stage");
  auto [entry, inserted] = catalog().try_emplace(name, RegisteredComponent{
      component.owner.str(), component.stage.str(), pass->getDescription().str(), component.required});
  if (!inserted && (entry->second.owner != component.owner ||
      entry->second.stage != component.stage || entry->second.required != component.required))
    llvm::report_fatal_error("conflicting transformation component registration: " + name);
}

void addTransformation(OpPassManager &manager, TransformationComponent component,
                       std::unique_ptr<Pass> pass) {
  if (component.required) manager.addPass(std::move(pass));
  else manager.addPass(std::make_unique<OptionalTransformation>(std::move(pass)));
}

llvm::json::Array transformationComponents() {
  SmallVector<StringRef> names;
  for (const auto &entry : catalog()) names.push_back(entry.first());
  llvm::sort(names);
  llvm::json::Array result;
  for (StringRef name : names) {
    const auto &entry = catalog().find(name)->second;
    result.push_back(llvm::json::Object{{"name", name}, {"owner", entry.owner},
        {"stage", entry.stage}, {"required", entry.required},
        {"description", entry.description}});
  }
  return result;
}

LogicalResult verifyTransformationControls(Operation *operation, CompileOptionsAttr options) {
  for (NamedAttribute selected : options.getOptimizations()) {
    auto found = catalog().find(selected.getName());
    if (found == catalog().end())
      return operation->emitError("unknown optimization component: ") << selected.getName();
    auto enabled = dyn_cast<BoolAttr>(selected.getValue());
    if (!enabled) return operation->emitError("optimization controls must be boolean");
    if (found->second.required && !enabled.getValue())
      return operation->emitError("cannot disable required lowering component: ") << selected.getName();
  }
  return success();
}
} // namespace intent
