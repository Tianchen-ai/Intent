#include "Intent/Dialect/Block/IR/ProgramInterface.h"
#include "Intent/Dialect/Block/IR/BlockOps.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "Intent/Dialect/Block/IR/BlockOps.cpp.inc"
