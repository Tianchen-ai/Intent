#include "Intent/Compiler/Backend.h"
#include "Intent/Compiler/Passes.h"
#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "Intent/Target/BangC/Passes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace mlir;

namespace intent::compiler {

FailureOr<ArrayAttr> readDSAConfigurations(ModuleOp module, StringRef directory,
                                         StringRef overrides) {
  auto read = [&](StringRef filename) -> FailureOr<llvm::json::Value> {
    auto buffer = llvm::MemoryBuffer::getFile(filename);
    if (!buffer) {
      module.emitError("cannot read DSA configurations: ") << filename << ": " << buffer.getError().message();
      return failure();
    }
    auto value = llvm::json::parse((*buffer)->getBuffer());
    if (!value) {
      module.emitError("invalid DSA configuration JSON: ") << filename << ": " << llvm::toString(value.takeError());
      return failure();
    }
    return std::move(*value);
  };
  llvm::SmallString<256> filename(directory);
  llvm::sys::path::append(filename, "bangc.json");
  auto defaults = read(filename);
  if (failed(defaults)) return failure();
  static const StringRef columns[] = {
      "tile", "tile_m", "tile_n", "tile_k", "region_tile", "tasks", "local_bytes"};
  auto root = defaults->getAsObject();
  auto names = root ? root->getArray("columns") : nullptr;
  auto families = root ? root->getObject("families") : nullptr;
  auto rows = families ? families->getArray("block") : nullptr;
  if (!root || root->size() != 2 || !names || names->size() != 7 ||
      !families || families->size() != 1 || !rows ||
      !llvm::equal(*names, columns, [](const llvm::json::Value &value, StringRef expected) {
        auto name = value.getAsString();
        return name && *name == expected;
      }))
    return module.emitError("DSA defaults require seven configuration columns and a block family"), failure();
  std::optional<llvm::json::Value> replacement;
  if (!overrides.empty()) {
    auto value = read(overrides);
    if (failed(value)) return failure();
    replacement = std::move(*value);
    auto object = replacement->getAsObject();
    auto space = object ? object->getObject("bangc") : nullptr;
    if (!object || object->size() != 1 || !space || space->size() != 1 || !space->getArray("block"))
      return module.emitError("DSA overrides require bangc.block complete configuration rows"), failure();
    rows = space->getArray("block");
  }
  if (rows->empty()) return module.emitError("DSA configuration list must not be empty"), failure();
  SmallVector<Attribute> configurations;
  for (const llvm::json::Value &value : *rows) {
    auto row = value.getAsArray();
    if (!row || row->size() != 7)
      return module.emitError("DSA configuration requires seven integer bindings"), failure();
    SmallVector<int64_t> bindings;
    for (const llvm::json::Value &value : *row) {
      auto integer = value.getAsInteger();
      if (!integer) return module.emitError("DSA configuration bindings must be integers"), failure();
      bindings.push_back(*integer);
    }
    auto configuration = dsa::ConfigurationAttr::getChecked(
        [&] { return module.emitError(); }, module.getContext(), bindings[0],
        bindings[1], bindings[2], bindings[3], bindings[4], bindings[5], bindings[6]);
    if (!configuration) return failure();
    if (llvm::is_contained(configurations, Attribute(configuration)))
      return module.emitError("DSA configuration list contains a duplicate row"), failure();
    configurations.push_back(configuration);
  }
  return ArrayAttr::get(module.getContext(), configurations);
}

FailureOr<DictionaryAttr> parseDSAParameterBindings(ModuleOp module,
                                                    StringRef text, bool shapes) {
  auto parsed = llvm::json::parse(text);
  if (!parsed) {
    module.emitError() << "invalid DSA " << (shapes ? "shapes" : "strides")
                       << ": " << llvm::toString(parsed.takeError());
    return failure();
  }
  auto object = parsed->getAsObject();
  if (!object)
    return module.emitError("DSA bindings must be a JSON object"), failure();
  Builder builder(module.getContext());
  NamedAttrList bindings;
  for (auto &[name, value] : *object) {
    auto entries = value.getAsArray();
    if (!entries)
      return module.emitError("DSA parameter binding must be an integer array"), failure();
    SmallVector<int64_t> integers;
    for (auto &entry : *entries) {
      auto integer = entry.getAsInteger();
      if (!integer || (shapes && *integer < -1))
        return module.emitError("invalid DSA bound dimension/stride"), failure();
      integers.push_back(*integer);
    }
    bindings.append(name.str(), builder.getDenseI64ArrayAttr(integers));
  }
  return bindings.getDictionary(module.getContext());
}

namespace {

LogicalResult verifyExistingBindings(func::FuncOp function,
                                      DictionaryAttr bindings, bool shapes) {
  auto interface = getPublicInterface(function);
  for (NamedAttribute binding : bindings) {
    intent::ViewType view;
    for (Attribute attribute : interface.getArguments()) {
      auto parameter = cast<PublicParameterAttr>(attribute);
      if (parameter.getName() == binding.getName()) {
        view = dyn_cast<intent::ViewType>(parameter.getType());
        break;
      }
    }
    if (!view)
      return function.emitError("shared DSA binding does not name an existing public view: ")
             << binding.getName();
    auto values = cast<DenseI64ArrayAttr>(binding.getValue());
    auto tensor = publicViewTensor(view);
    if (values.size() != tensor.getRank())
      return function.emitError("shared DSA binding rank disagrees with the public view: ")
             << binding.getName();
    for (auto [axis, value] : llvm::enumerate(values.asArrayRef())) {
      if (shapes) {
        if (value >= 0 && tensor.getDimSize(axis) != value)
          return function.emitError("shared DSA shape binding would change the existing interface: ")
                 << binding.getName() << " axis " << axis;
      } else {
        auto constraints = view.getConstraints();
        auto fixed = constraints.getHasStrides()
                         ? dyn_cast<IntegerAttr>(constraints.getStrides()[axis])
                         : IntegerAttr();
        if (!fixed || fixed.getInt() != value)
          return function.emitError("shared DSA stride binding would change the existing interface: ")
                 << binding.getName() << " axis " << axis;
      }
    }
  }
  return success();
}

} // namespace

ArrayRef<Backend> dsaBackends() {
  static const Backend adapters[] = {
      {Provider::BangC, "bangc", true, DSABackend{}, nullptr,
       [] { bangc::registerBangCPasses(); bangc::registerBangCPipelines(); },
       [](OpPassManager &manager, const Request &request) {
         bangc::buildBangCPipeline(manager.nest<ModuleOp>(), request.dsa.architecture);
       },
       bangc::serializeProgram}};
  return adapters;
}

void DSABackend::buildConstruction(OpPassManager &manager, const Request &request) const {
  ConstructDSAOptions options;
  options.directory = request.profileDirectory;
  options.overrides = request.tuningConfig;
  options.shapes = request.dsa.shapes;
  options.strides = request.dsa.strides;
  manager.addPass(createConstructDSA(options));
}

void DSABackend::buildShared(OpPassManager &manager, const Request &, StringRef) const {
  dsa::buildDSAPipeline(manager.nest<ModuleOp>());
}

LogicalResult DSABackend::verifySharedInput(ModuleOp module, const Request &request, StringRef) const {
  if (!module->getAttrOfType<StringAttr>("intent.entry_name") ||
      module.getOps<ModuleOp>().empty())
    return module.emitError("shared DSA input requires a complete configuration portfolio");
  for (Operation &operation : *module.getBody())
    if (!isa<ModuleOp>(operation))
      return operation.emitError("DSA portfolio contains an operation outside its candidate modules");
  auto options = module->getAttrOfType<CompileOptionsAttr>(compileOptionsAttr);
  if (!options) return module.emitError("DSA portfolio requires its numerical compile options");
  if (module->hasAttr("bangc.architecture"))
    return module.emitError("provider-specific BANG C input cannot be resumed as shared DSA IR");
  if (request.dsa.architecture != "mtp_372")
    return module.emitError("shared DSA input requires the supported mtp_372 implementation profile");
  auto shapes = parseDSAParameterBindings(module, request.dsa.shapes, true);
  auto strides = parseDSAParameterBindings(module, request.dsa.strides, false);
  if (failed(shapes) || failed(strides))
    return failure();
  for (ModuleOp candidate : module.getOps<ModuleOp>()) {
    if (candidate->getAttrOfType<CompileOptionsAttr>(compileOptionsAttr) != options)
      return candidate.emitError("DSA candidate compile options disagree with the portfolio");
    if (failed(dsa::verifyRealizedProgram(candidate))) return failure();
    if (candidate->hasAttr("bangc.architecture"))
      return candidate.emitError("provider-specific BANG C input cannot be resumed as shared DSA IR");
    for (auto function : candidate.getOps<func::FuncOp>()) {
      if (function.isExternal())
        continue;
      if (!function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration"))
        return function.emitError("shared DSA candidate requires its complete configuration");
      if (failed(verifyExistingBindings(function, *shapes, true)) ||
          failed(verifyExistingBindings(function, *strides, false)))
        return failure();
    }
  }
  return success();
}

} // namespace intent::compiler
