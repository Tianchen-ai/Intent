#ifndef INTENT_TRANSFORMS_PASSMANAGER_H
#define INTENT_TRANSFORMS_PASSMANAGER_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"

namespace intent {

// Apply instrumentation once per manager; reuse that manager for repeated
// runs so native printer counters and timing data remain in one session.
// Registration is idempotent and also supports library callers without a CLI.
inline mlir::LogicalResult configurePassManager(mlir::PassManager &manager) {
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  if (mlir::failed(mlir::applyPassManagerCLOptions(manager))) return mlir::failure();
  mlir::applyDefaultTimingPassManagerCLOptions(manager);
  return mlir::success();
}

} // namespace intent
#endif
