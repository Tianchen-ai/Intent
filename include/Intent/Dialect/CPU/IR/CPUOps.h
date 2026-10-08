#ifndef INTENT_DIALECT_CPU_IR_CPUOPS_H
#define INTENT_DIALECT_CPU_IR_CPUOPS_H
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/CPU/IR/CPUOpInterfaces.h"
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferViewFlowOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#define GET_OP_CLASSES
#include "Intent/Dialect/CPU/IR/CPUOps.h.inc"
namespace intent::cpu {
inline constexpr llvm::StringLiteral entryRequirementsAttr = "intent_cpu.entry_requirements";
}
#endif
