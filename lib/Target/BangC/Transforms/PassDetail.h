#ifndef INTENT_TARGET_BANGC_TRANSFORMS_PASSDETAIL_H
#define INTENT_TARGET_BANGC_TRANSFORMS_PASSDETAIL_H
#include "Intent/Target/BangC/Passes.h"
#include "Intent/Dialect/DSA/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/DSA/Analysis/Storage.h"
#include "Intent/Dialect/DSA/Analysis/UniformValues.h"
#include "Intent/Dialect/DSA/IR/Views.h"
#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/MathExtras.h"
#include <functional>
#include <limits>

namespace intent::bangc {
using namespace mlir;
using dsa::integerInterval;
struct StorageUsage { int64_t nram = 0, wram = 0, sram = 0; };
StorageUsage measureStorage(func::FuncOp function);
void measureStorage(func::FuncOp function, int64_t &nram, int64_t &wram);
StorageUsage bindStorage(func::FuncOp function);
bool storageFitsBudget(func::FuncOp function, dsa::ConfigurationAttr config,
                       StorageUsage usage);
LogicalResult realizeMatMul(dsa::MatMulOp matrix, dsa::ConfigurationAttr config);
bool supportedScalarBinary(BinaryOperator kind);
bool initializeMatrixStorage(func::FuncOp function);
bool retainNarrowExtremaInputs(func::FuncOp function, dsa::ConfigurationAttr config);
bool bindBroadcastOperands(func::FuncOp function);
bool propagateLocalTransposeSupply(func::FuncOp function, dsa::ConfigurationAttr config);
bool realizeAffineRanges(func::FuncOp function, dsa::ConfigurationAttr config);
bool realizeFullWidthMasks(func::FuncOp function, dsa::ConfigurationAttr config);
bool vectorizeIndexLoops(func::FuncOp function, dsa::ConfigurationAttr config);
bool batchIndependentRowPrograms(func::FuncOp function, dsa::ConfigurationAttr config);
Value allocate(OpBuilder &b, Location loc, Type element, ArrayRef<int64_t> shape, int64_t space);
bool reuseConsumedBinaryInputs(func::FuncOp function, dsa::ConfigurationAttr config);
bool reuseConsumedExp2Inputs(func::FuncOp function, dsa::ConfigurationAttr config);
bool placeInvariantSupply(func::FuncOp function, dsa::ConfigurationAttr config);
bool reuseCompletedSupply(func::FuncOp function, dsa::ConfigurationAttr config);
bool coarsenLocalPrograms(func::FuncOp function, dsa::ConfigurationAttr config);
void pipelineLocalSupply(func::FuncOp function, dsa::ConfigurationAttr config);
LogicalResult realizeNativeComputations(ModuleOp module);
LogicalResult realizeNativeWorkspace(ModuleOp module,
    llvm::function_ref<LogicalResult()> cleanup);
LogicalResult selectNativeImplementations(ModuleOp module);
LogicalResult composeLocalStorage(func::FuncOp function,
    llvm::function_ref<LogicalResult()> cleanup);
LogicalResult composeLocalProgram(ModuleOp module,
    llvm::function_ref<LogicalResult()> cleanup);
LogicalResult reuseStorageScopes(ModuleOp module,
    llvm::function_ref<LogicalResult()> cleanup);
LogicalResult scheduleProgramSupply(ModuleOp module);
LogicalResult bindProgramStorage(ModuleOp module);
LogicalResult verifySurfaceOperations(func::FuncOp function);
LogicalResult verifyNativeProgram(ModuleOp module);
LogicalResult verifyUnboundNativeProgram(ModuleOp module);
} // namespace intent::bangc
#endif
