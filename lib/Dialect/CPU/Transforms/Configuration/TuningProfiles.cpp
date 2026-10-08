#include "Intent/Dialect/CPU/Transforms/Configuration/Configuration.h"
#include "TuningProfiles.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace mlir;

namespace intent::cpu {

FailureOr<TuningProfiles> TuningProfiles::read(
    ModuleOp module, StringRef defaults, StringRef overrides,
    const ImplementationRegistry &implementations) {
  TuningProfiles result;
  if (failed(result.readFile(module, defaults, implementations)))
    return failure();
  if (!overrides.empty() &&
      failed(result.readFile(module, overrides, implementations)))
    return failure();
  return result;
}

LogicalResult TuningProfiles::readFile(
    ModuleOp module, StringRef path,
    const ImplementationRegistry &implementations) {
  auto content = llvm::MemoryBuffer::getFile(path);
  if (!content)
    return module.emitError("cannot read CPU tuning profiles: ") << path;
  auto parsed = llvm::json::parse((*content)->getBuffer());
  if (!parsed)
    return module.emitError("invalid CPU tuning JSON: ")
           << path << ": " << llvm::toString(parsed.takeError());
  auto root = parsed->getAsObject();
  if (!root || root->size() != 1 || !root->getObject("cpu"))
    return module.emitError("CPU tuning profiles require only the cpu namespace: ")
           << path;
  Builder builder(module.getContext());
  for (auto &family : *root->getObject("cpu")) {
    StringRef familyName = family.first;
    auto schema = implementations.profileParameters(familyName);
    if (!schema)
      return module.emitError("unregistered CPU tuning family '")
             << familyName << "' in " << path;
    auto rows = family.second.getAsArray();
    if (!rows || rows->empty())
      return module.emitError("CPU tuning family '")
             << familyName << "' must contain a non-empty row array in " << path;
    SmallVector<Configuration> configurations;
    for (auto [ordinal, entry] : llvm::enumerate(*rows)) {
      auto error = [&]() -> InFlightDiagnostic {
        auto diagnostic = module.emitError("CPU tuning family '");
        diagnostic << familyName << "' row " << ordinal + 1 << " in " << path << ": ";
        return diagnostic;
      };
      auto candidate = entry.getAsObject();
      if (!candidate || candidate->size() != 3 || !candidate->getObject("local") ||
          !candidate->getArray("implementations"))
        return error() << "requires shared/local bindings and an implementation selection";
      auto shared = candidate->getArray("shared");
      if (!shared || shared->size() != 5)
        return error() << "shared binding requires task grain, M/N/K outer blocks and region size";
      SmallVector<int64_t, 5> values;
      for (const llvm::json::Value &column : *shared) {
        auto value = column.getAsInteger();
        if (!value || *value <= 0)
          return error() << "shared parameters must be positive integers";
        values.push_back(*value);
      }
      SmallVector<Attribute> selected;
      for (const llvm::json::Value &entry : *candidate->getArray("implementations")) {
        auto name = entry.getAsString();
        if (!name || name->empty() || !implementations.hasImplementation(*name))
          return error() << "implementation selection requires registered implementation names";
        auto attribute = builder.getStringAttr(*name);
        if (llvm::is_contained(selected, Attribute(attribute)))
          return error() << "implementation selection contains a duplicate: " << *name;
        selected.push_back(attribute);
      }
      if (selected.empty())
        return error() << "implementation selection must not be empty";
      auto selection = builder.getArrayAttr(selected);
      auto declared = implementations.localParameters(selection);
      if (failed(declared)) return error() << "invalid implementation selection";
      const auto &local = *candidate->getObject("local");
      for (auto &parameter : local) {
        StringRef parameterName = parameter.first;
        if (!llvm::is_contained(*schema, parameterName))
          return error() << "unconsumed local parameter '" << parameterName << "'";
        if (!llvm::is_contained(*declared, parameterName))
          return error() << "local parameter '" << parameterName
                         << "' is not declared by a selected implementation";
        auto value = parameter.second.getAsInteger();
        if (!value || *value <= 0)
          return error() << "local parameter '" << parameterName << "' must be a positive integer";
      }
      SmallVector<NamedAttribute> bindings;
      for (StringRef name : *schema) {
        auto value = local.getInteger(name);
        if (!value)
          return error() << "missing local parameter '" << name << "'";
        bindings.push_back(builder.getNamedAttr(name, builder.getI64IntegerAttr(*value)));
      }
      configurations.push_back({values[0], values[1], values[2], values[3], values[4],
                                builder.getDictionaryAttr(bindings), selection});
    }
    // An explicitly supplied family replaces its complete ordered row list.
    families[familyName] = std::move(configurations);
  }
  return success();
}

ArrayRef<Configuration> TuningProfiles::get(StringRef family) const {
  auto found = families.find(family);
  if (found == families.end())
    return {};
  return found->second;
}

} // namespace intent::cpu
