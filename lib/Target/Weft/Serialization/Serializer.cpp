#include "Intent/Target/Weft/Serialization/Serializer.h"
#include "Intent/Target/Weft/IR/Program.h"
#include "Intent/Target/Weft/IR/WeftAttrs.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "Intent/Serialization/NativeABI.h"
#include "TaskABI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace wk = ::weft::kernel;

namespace intent::weft_provider {
namespace {

llvm::json::Object serializeCandidate(func::FuncOp function) {
  auto config = function->getAttrOfType<cpu::ConfigurationAttr>("intent_cpu.configuration");
  llvm::json::Array implementations;
  for (Attribute entry : function->getAttrOfType<ArrayAttr>("intent_cpu.implementations")) {
    auto binding = cast<cpu::ImplementationAttr>(entry);
    llvm::json::Object values;
    for (NamedAttribute parameter : binding.getParameters())
      values[parameter.getName().getValue()] = cast<IntegerAttr>(parameter.getValue()).getInt();
    implementations.push_back(llvm::json::Object{{"name", binding.getName().getValue().str()},
                                               {"parameters", std::move(values)}});
  }
  return llvm::json::Object{{"entry", function.getName().str()},
      {"values", llvm::json::Array{config.getTaskGrain(), config.getTileM(), config.getTileN(),
                                   config.getTileK(), config.getRegionSize()}},
      {"requires_matrix_i8_i32", function->getAttrOfType<BoolAttr>("intent_cpu.requires_matrix_i8_i32").getValue()},
      {"implementations", std::move(implementations)}};
}

FailureOr<llvm::json::Array> serializeTasks(ModuleOp program) {
  llvm::json::Array tasks;
  for (Attribute entry : program->getAttrOfType<ArrayAttr>(taskBindingsAttr)) {
    auto binding = cast<TaskBindingAttr>(entry);
    auto hostEntry = cast<func::FuncOp>(SymbolTable::lookupSymbolIn(program, binding.getHostEntry()));
    auto kernel = cast<wk::KernelOp>(SymbolTable::lookupSymbolIn(program, binding.getDeviceKernel()));
    auto abi = serializeTaskABI(kernel);
    if (failed(abi)) return failure();
    tasks.push_back(llvm::json::Object{{"cpu_entry", hostEntry.getName().str()},
        {"task_ordinal", binding.getOrdinal()}, {"weft_entry", kernel.getSymName().str()},
        {"coordinate_argument", binding.getCoordinateArgument()}, {"abi", std::move(*abi)},
        {"argument_count", static_cast<int64_t>(kernel.getBody().front().getNumArguments())}});
  }
  return std::move(tasks);
}

} // namespace

LogicalResult serializeProgram(ModuleOp program, std::string &source, std::string &metadata) {
  auto options = readCompileOptions(program);
  if (failed(options)) return failure();
  if (failed(verifyProgram(program))) return failure();
  auto modules = getProgramModules(program);
  if (failed(modules)) return failure();
  SmallVector<func::FuncOp> candidates;
  for (auto function : modules->host.getOps<func::FuncOp>())
    if (!function.isExternal()) candidates.push_back(function);
  auto interface = getPublicInterface(candidates.front());
  auto parameters = serializePublicInterface(candidates.front(), interface);
  auto nativeABI = queryHostABI(candidates.front());
  auto tasks = serializeTasks(program);
  if (failed(parameters) || failed(tasks) || failed(nativeABI)) return failure();
  llvm::json::Array configurations;
  for (auto function : candidates) configurations.push_back(serializeCandidate(function));
  std::string hostSource;
  if (failed(serializeHostProgram(modules->host, hostSource))) return failure();

  llvm::raw_string_ostream output(source);
  modules->device.print(output, OpPrintingFlags().enableDebugInfo());
  output << '\n';
  auto entry = candidates.front()->getAttrOfType<cpu::EntryRequirementsAttr>(cpu::entryRequirementsAttr);
  auto requirements = queryNativeEntryRequirements(candidates.front(), interface,
      entry.getDisjointOutputs(), [&](unsigned index, intent::ViewType view) -> FailureOr<NativeViewRequirements> {
        auto layout = entry.getContiguousViews() || view.getAccess() != 0
            ? NativeViewLayout::Contiguous : NativeViewLayout::Strided;
        auto alignment = candidates.front().getArgAttrOfType<ArgumentAlignmentAttr>(
            index, argumentAlignmentAttr);
        return NativeViewRequirements{layout, alignment.getBytes()};
      });
  if (failed(requirements)) return failure();
  auto capabilities = modules->host->getAttrOfType<cpu::CapabilitiesAttr>("intent_cpu.capabilities");
  llvm::raw_string_ostream metadataStream(metadata);
  metadataStream << llvm::json::Value(llvm::json::Object{{"kind", "weft-generation"},
      {"provider", "weft"}, {"entry_name", candidates.front().getName()},
      {"compile_options", serializeCompileOptions(*options)},
      {"target", llvm::json::Object{
          {"family", "cpu"}, {"vector_bits", capabilities.getVectorBits()},
          {"workers", capabilities.getWorkers()},
          {"private_bytes", capabilities.getPrivateBytes()},
          {"matrix_i8_i32", capabilities.getMatrixI8I32()}}},
      {"host_source", hostSource}, {"tasks", std::move(*tasks)},
      {"interface", std::move(*parameters)}, {"candidates", std::move(configurations)},
      {"native", llvm::json::Object{{"requirements", requirements->serialize()},
          {"slots", nativeABI->serialize()}}}});
  return success();
}

} // namespace intent::weft_provider
