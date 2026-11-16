#ifndef INTENT_TARGET_NVIDIA_CONFIGURATION_H
#define INTENT_TARGET_NVIDIA_CONFIGURATION_H
#include "Intent/Compiler/Compiler.h"
namespace intent::compiler {
mlir::LogicalResult configureNVIDIATarget(mlir::ModuleOp, Request &,
    bool tupleReductions, bool constantIdentity, bool fragmentGather);
}
#endif
