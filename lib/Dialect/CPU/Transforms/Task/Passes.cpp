#include "../PassSupport.h"
#include "Intent/Dialect/CPU/Transforms/Contraction/Contraction.h"
#include "Intent/Dialect/CPU/Transforms/Task/Tasks.h"

using namespace mlir;

namespace intent::cpu {

#define GEN_PASS_DEF_CPUGROUPWORKSETS
#define GEN_PASS_DEF_CPUBLOCKCOMPUTATIONS
#define GEN_PASS_DEF_CPUPARTITIONTASKS
#define GEN_PASS_DEF_CPUISOLATETASKS
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

namespace {

class GroupWorksetsPass
    : public impl::CPUGroupWorksetsBase<GroupWorksetsPass> {
public:
  using CPUGroupWorksetsBase::CPUGroupWorksetsBase;
  void runOnOperation() final {
    if (failed(detail::transformWithImplementations(
            getOperation(), getArgument(), provider.getValue(),
            groupWorksetComputations)))
      signalPassFailure();
  }
};

class BlockComputationsPass
    : public impl::CPUBlockComputationsBase<BlockComputationsPass> {
public:
  using CPUBlockComputationsBase::CPUBlockComputationsBase;
  void runOnOperation() final {
    auto transform = [](func::FuncOp function,
                        const ImplementationRegistry &implementations) -> LogicalResult {
      auto configuration = detail::currentConfiguration(function);
      if (failed(configuration) ||
          failed(blockContractions(function, *configuration, implementations)))
        return failure();
      return blockStructuredComputations(function, implementations);
    };
    if (failed(detail::transformWithImplementations(
            getOperation(), getArgument(), provider.getValue(), transform)))
      signalPassFailure();
  }
};

class PartitionTasksPass
    : public impl::CPUPartitionTasksBase<PartitionTasksPass> {
public:
  using CPUPartitionTasksBase::CPUPartitionTasksBase;
  void runOnOperation() final {
    auto transform = [](func::FuncOp function,
                        const ImplementationRegistry &implementations) {
      auto configuration = detail::currentConfiguration(function);
      return failed(configuration)
                 ? failure()
                 : partitionTasks(function, configuration->taskGrain,
                                  implementations);
    };
    if (failed(detail::transformWithImplementations(
            getOperation(), getArgument(), provider.getValue(), transform)))
      signalPassFailure();
  }
};

class IsolateTasksPass : public impl::CPUIsolateTasksBase<IsolateTasksPass> {
public:
  using CPUIsolateTasksBase::CPUIsolateTasksBase;
  void runOnOperation() final {
    if (failed(detail::transformFunctions(getOperation(), getArgument(),
                                          isolateTasks)))
      signalPassFailure();
  }
};

} // namespace
} // namespace intent::cpu
