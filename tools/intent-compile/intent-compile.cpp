#include "Intent/Compiler/Compiler.h"
#include "Intent/Compiler/Backend.h"
#include "Intent/Compiler/Registration.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace {
using intent::compiler::Failure;
int exitCode(Failure failure) { return static_cast<int>(failure); }

bool writeFile(llvm::StringRef path,
               llvm::function_ref<void(llvm::raw_ostream &)> write) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error) { llvm::errs() << "cannot open " << path << ": " << error.message() << "\n"; return false; }
  write(output);
  output.close();
  if (output.has_error()) {
    llvm::errs() << "cannot write " << path << ": " << output.error().message() << "\n";
    output.clear_error();
    return false;
  }
  return true;
}
} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initialization(argc, argv);
  intent::compiler::registerPasses();
  mlir::registerMLIRContextCLOptions();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input Intent KIR>"), llvm::cl::init("-"));
  llvm::cl::opt<std::string> target("target", llvm::cl::desc("source provider"));
  llvm::cl::opt<std::string> inputStage("input-stage", llvm::cl::init("kir"),
      llvm::cl::desc("Input IR stage: kir or shared; shared input requires the original target resources"));
  llvm::cl::opt<int64_t> computeUnits("compute-units", llvm::cl::init(0));
  llvm::cl::opt<int64_t> sharedMemoryPerUnit("shared-memory-per-unit", llvm::cl::init(0));
  llvm::cl::opt<int64_t> maxDynamicSharedMemoryPerBlock("max-dynamic-shared-memory-per-block", llvm::cl::init(0));
  llvm::cl::opt<int64_t> registersPerUnit("registers-per-unit", llvm::cl::init(0));
  llvm::cl::opt<int64_t> maxThreadsPerBlock("max-threads-per-block", llvm::cl::init(0));
  llvm::cl::opt<int64_t> computeCapabilityMajor("compute-capability-major", llvm::cl::init(0));
  llvm::cl::opt<int64_t> computeCapabilityMinor("compute-capability-minor", llvm::cl::init(-1));
  llvm::cl::opt<int64_t> singleToDoublePrecisionPerfRatio("single-to-double-precision-perf-ratio", llvm::cl::init(0));
  llvm::cl::opt<bool> matrixUnits("matrix-units", llvm::cl::init(false));
  llvm::cl::opt<bool> dynamicVectorWidth("dynamic-vector-width", llvm::cl::init(false));
  llvm::cl::opt<std::string> irOutputFilename("ir-output", llvm::cl::init(""));
  llvm::cl::opt<std::string> sourceOutputFilename("source-output", llvm::cl::init(""));
  llvm::cl::opt<std::string> metadataOutputFilename("metadata-output", llvm::cl::init(""));
  llvm::cl::opt<std::string> tuningConfigFilename("tuning-config", llvm::cl::init(""),
      llvm::cl::desc("JSON overrides of declared shared/provider profile families"));
  llvm::cl::opt<std::string> numerics("numerics", llvm::cl::init("source"),
      llvm::cl::desc("Numerical permissions: source or relaxed_normalization (finite active score/value precondition)"));
  llvm::cl::opt<bool> onlineReduction("online-reduction", llvm::cl::init(true),
      llvm::cl::desc("Enable eligible online normalized reductions"));
  llvm::cl::opt<bool> optimizationRemarks("optimization-remarks", llvm::cl::init(false),
      llvm::cl::desc("Explain online reduction selection and rejection"));
  llvm::cl::opt<int64_t> cpuVectorBits("cpu-vector-bits", llvm::cl::init(0));
  llvm::cl::opt<int64_t> cpuWorkers("cpu-workers", llvm::cl::init(0));
  llvm::cl::opt<bool> cpuMatrixI8I32("cpu-matrix-i8-i32", llvm::cl::init(false));
  llvm::cl::opt<int64_t> cpuPrivateBytes("cpu-private-bytes", llvm::cl::init(256 * 1024));
  llvm::cl::opt<std::string> dsaArchitecture("dsa-architecture", llvm::cl::init("mtp_372"));
  llvm::cl::opt<std::string> dsaShapes("dsa-shapes", llvm::cl::init("{}"),
      llvm::cl::desc("JSON parameter shapes; -1 keeps an axis dynamic"));
  llvm::cl::opt<std::string> dsaStrides("dsa-strides", llvm::cl::init("{}"),
      llvm::cl::desc("JSON parameter element strides"));
  llvm::cl::opt<bool> stopAfterShared("stop-after-shared", llvm::cl::init(false),
      llvm::cl::desc("Stop after the selected family's complete shared pipeline"));
  llvm::cl::opt<bool> stopAfterKIR("stop-after-kir", llvm::cl::init(false),
      llvm::cl::desc("Normalize and verify KIR without selecting a target"));
  llvm::cl::opt<bool> compilerInfo("compiler-info", llvm::cl::init(false),
      llvm::cl::desc("Print compiled providers, stages, outputs and failure categories"));
  llvm::cl::ParseCommandLineOptions(argc, argv, "Intent compiler\n");

  llvm::SmallString<256> profileDirectory(
      llvm::sys::fs::getMainExecutable(argv[0], reinterpret_cast<void *>(&main)));
  llvm::sys::path::remove_filename(profileDirectory);
  llvm::sys::path::append(profileDirectory, "profiles");
  const llvm::SmallVector<llvm::cl::Option *> gpuOptions{
      &computeUnits, &sharedMemoryPerUnit, &maxDynamicSharedMemoryPerBlock,
      &registersPerUnit, &maxThreadsPerBlock, &computeCapabilityMajor,
      &computeCapabilityMinor, &singleToDoublePrecisionPerfRatio,
      &matrixUnits, &dynamicVectorWidth};
  const llvm::SmallVector<llvm::cl::Option *> cpuOptions{
      &cpuVectorBits, &cpuWorkers, &cpuMatrixI8I32, &cpuPrivateBytes};
  const llvm::SmallVector<llvm::cl::Option *> dsaOptions{
      &dsaArchitecture, &dsaShapes, &dsaStrides};
  auto rejectOptions = [&](llvm::ArrayRef<llvm::cl::Option *> options,
                           llvm::StringRef reason) {
    for (llvm::cl::Option *option : options)
      if (option->getNumOccurrences()) {
        llvm::errs() << reason << ": "
                     << (option->ArgStr.empty() ? "<input Intent KIR>" : "--" + option->ArgStr.str())
                     << "\n";
        return true;
      }
    return false;
  };
  if (compilerInfo) {
    if (rejectOptions(gpuOptions, "--compiler-info does not compile a program") ||
        rejectOptions(cpuOptions, "--compiler-info does not compile a program") ||
        rejectOptions(dsaOptions, "--compiler-info does not compile a program") ||
        rejectOptions({&inputFilename, &target, &inputStage, &irOutputFilename,
                       &sourceOutputFilename, &metadataOutputFilename,
                       &tuningConfigFilename, &numerics, &onlineReduction,
                       &optimizationRemarks, &stopAfterKIR, &stopAfterShared},
                      "--compiler-info does not compile a program"))
      return exitCode(Failure::Invocation);
    llvm::outs() << llvm::json::Value(intent::compiler::information(profileDirectory)) << "\n";
    return 0;
  }
  if (stopAfterKIR && stopAfterShared) {
    llvm::errs() << "--stop-after-kir and --stop-after-shared are mutually exclusive\n";
    return exitCode(Failure::Invocation);
  }
  intent::compiler::Request request;
  if (numerics.getNumOccurrences() || onlineReduction.getNumOccurrences() ||
      optimizationRemarks.getNumOccurrences()) {
    auto mode = intent::symbolizeNumericsMode(numerics);
    if (!mode) {
      llvm::errs() << "--numerics must be source or relaxed_normalization\n";
      return exitCode(Failure::Invocation);
    }
    request.options = intent::compiler::CompilationOptions{*mode, onlineReduction, optimizationRemarks};
  }
  if (inputStage == "shared") request.inputStage = intent::compiler::InputStage::Shared;
  else if (inputStage != "kir") {
    llvm::errs() << "--input-stage must be kir or shared\n";
    return exitCode(Failure::Invocation);
  }
  if (request.inputStage == intent::compiler::InputStage::Shared &&
      rejectOptions({&tuningConfigFilename}, "shared input already contains resolved tuning profiles"))
    return exitCode(Failure::Invocation);
  if (!target.empty()) {
    request.provider = intent::compiler::parseProvider(target);
    if (!request.provider) {
      llvm::errs() << "unknown source provider: " << target << "\n";
      return exitCode(Failure::Invocation);
    }
  }
  request.stopAfter = stopAfterKIR ? intent::compiler::Stage::Kernel
      : stopAfterShared ? intent::compiler::Stage::Shared : intent::compiler::Stage::Provider;
  if (!stopAfterKIR && !request.provider) {
    llvm::errs() << "--target is required unless --stop-after-kir is selected\n";
    return exitCode(Failure::Invocation);
  }
  if (request.stopAfter == intent::compiler::Stage::Kernel) {
    if (rejectOptions({&target, &tuningConfigFilename},
                      "KIR output does not select a physical target or tuning profile") ||
        rejectOptions(gpuOptions, "KIR output does not consume GPU resources") ||
        rejectOptions(cpuOptions, "KIR output does not consume CPU resources") ||
        rejectOptions(dsaOptions, "KIR output does not consume DSA bindings"))
      return exitCode(Failure::Invocation);
  } else {
    const auto family = intent::compiler::backend(*request.provider).family();
    if ((family != intent::compiler::Family::GPU &&
         rejectOptions(gpuOptions, "selected provider does not consume GPU resources")) ||
        (family != intent::compiler::Family::CPU &&
         rejectOptions(cpuOptions, "selected provider does not consume CPU resources")) ||
        (family != intent::compiler::Family::DSA &&
         rejectOptions(dsaOptions, "selected provider does not consume DSA bindings")))
      return exitCode(Failure::Invocation);
  }
  if (request.stopAfter != intent::compiler::Stage::Provider &&
      rejectOptions({&sourceOutputFilename, &metadataOutputFilename},
                    "selected stage only emits IR"))
    return exitCode(Failure::Invocation);
  if (irOutputFilename.empty() ||
      (request.stopAfter == intent::compiler::Stage::Provider && sourceOutputFilename.empty())) {
    llvm::errs() << "--ir-output is required; provider compilation also requires --source-output\n";
    return exitCode(Failure::Output);
  }
  request.gpu = {computeUnits, sharedMemoryPerUnit, maxDynamicSharedMemoryPerBlock,
      registersPerUnit, maxThreadsPerBlock, computeCapabilityMajor, computeCapabilityMinor,
      singleToDoublePrecisionPerfRatio, matrixUnits, dynamicVectorWidth, false, false};
  request.cpu = {cpuVectorBits, cpuWorkers, cpuMatrixI8I32, cpuPrivateBytes};
  request.dsa = {dsaArchitecture, dsaShapes, dsaStrides};
  request.tuningConfig = tuningConfigFilename;
  request.profileDirectory = std::string(profileDirectory);

  mlir::DialectRegistry registry;
  intent::compiler::registerDialects(registry);
  mlir::MLIRContext context(registry);
  llvm::SourceMgr sourceManager;
  mlir::SourceMgrDiagnosticHandler diagnostics(sourceManager, &context);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, sourceManager, &context);
  auto result = intent::compiler::compile(std::move(module), request);
  if (result.failure != Failure::None) return exitCode(result.failure);
  if (!writeFile(irOutputFilename, [&](llvm::raw_ostream &output) {
        result.module->print(output, mlir::OpPrintingFlags().enableDebugInfo());
        output << "\n";
      })) return exitCode(Failure::Output);
  if (request.stopAfter != intent::compiler::Stage::Provider) return 0;
  if (!writeFile(sourceOutputFilename, [&](llvm::raw_ostream &output) { output << result.source; }))
    return exitCode(Failure::Output);
  if (!metadataOutputFilename.empty() &&
      !writeFile(metadataOutputFilename, [&](llvm::raw_ostream &output) { output << result.metadata << "\n"; }))
    return exitCode(Failure::Output);
  return 0;
}
