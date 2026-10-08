#include "TaskABI.h"
#include "Intent/Target/Weft/Serialization/HostScalar.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
namespace wk = ::weft::kernel;

namespace intent::weft_provider {

FailureOr<llvm::json::Object> serializeTaskABI(wk::KernelOp kernel) {
  llvm::json::Array arguments, symbols;
  for (Attribute symbol : kernel.getShapeSymbols())
    symbols.push_back(cast<StringAttr>(symbol).getValue().str());
  for (auto [index, argument] : llvm::enumerate(kernel.getBody().front().getArguments())) {
    auto view = cast<wk::ViewType>(argument.getType());
    auto encoding = cast<wk::EncodingType>(view.getEncoding());
    int64_t bytes, elements, alignment;
    std::string scalar;
    if (encoding.getKind() == "dense") {
      auto storage = denseStorageType(encoding.getFamily());
      if (!storage)
        return kernel.emitError("CPU task ABI has no native dense element representation"), failure();
      scalar = storage->cType;
      bytes = storage->bytes;
      elements = 1; alignment = bytes;
    } else {
      wk::EncodingDeclOp declaration;
      for (auto candidate : kernel->getParentOfType<ModuleOp>().getOps<wk::EncodingDeclOp>())
        if (candidate.getSymName() == encoding.getFamily()) declaration = candidate;
      if (encoding.getKind() != "base" || !declaration)
        return kernel.emitError("CPU task ABI requires a declared base encoding"), failure();
      scalar = "uint8_t"; bytes = declaration.getStorageBits() / 8;
      elements = declaration.getElements(); alignment = declaration.getAlignment();
    }
    StringRef access = cast<StringAttr>(kernel.getArgAccess()[index]).getValue();
    bool writable = access == "write" || access == "readwrite";
    int64_t alias = kernel.getArgAliasSets()[index];
    bool restricted = alias >= 0 && llvm::count(kernel.getArgAliasSets(), alias) == 1;
    llvm::json::Array dimensions;
    for (int64_t extent : view.getShape().asArrayRef())
      dimensions.push_back(extent >= 0 ? std::to_string(extent)
          : cast<StringAttr>(kernel.getShapeSymbols()[-extent - 1]).getValue().str());
    arguments.push_back(llvm::json::Object{
        {"name", cast<StringAttr>(kernel.getArgNames()[index]).getValue().str()},
        {"c_type", (writable ? "" : "const ") + scalar + (restricted ? " *restrict" : " *")},
        {"encoding", encoding.getLayoutIdentity().str()}, {"shape", std::move(dimensions)},
        {"storage_bytes", bytes}, {"record_elements", elements}, {"alignment", alignment},
        {"interleave_rows", 0}, {"alias_set", alias}, {"writable", writable}});
  }
  return llvm::json::Object{{"symbol", kernel.getSymName().str()},
      {"arguments", std::move(arguments)}, {"shape_parameters", std::move(symbols)}};
}

} // namespace intent::weft_provider
