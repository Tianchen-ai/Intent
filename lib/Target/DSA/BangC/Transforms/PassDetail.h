#ifndef INTENT_TARGET_BANGC_TRANSFORMS_PASSDETAIL_H
#define INTENT_TARGET_BANGC_TRANSFORMS_PASSDETAIL_H
#include "Intent/Target/DSA/BangC/Passes.h"
#include "Intent/Target/DSA/BangC/Resources.h"
#include "Intent/Target/DSA/BangC/Physical/Analysis/PhysicalProgram.h"
#include "Intent/Target/DSA/BangC/Physical/Analysis/Storage.h"
#include "Intent/Target/DSA/BangC/Physical/Analysis/UniformValues.h"
#include "Intent/Target/DSA/BangC/Physical/IR/Views.h"
#include "Intent/Target/DSA/BangC/Physical/Transforms/Passes.h"
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
using bangc::physical::integerInterval;
struct StorageUsage { int64_t nram = 0, wram = 0, sram = 0; };
StorageUsage measureStorage(func::FuncOp function);
void measureStorage(func::FuncOp function, int64_t &nram, int64_t &wram);
StorageUsage bindStorage(func::FuncOp function);
bool storageFitsBudget(func::FuncOp function, bangc::physical::ConfigurationAttr config,
                       StorageUsage usage);
LogicalResult realizeMatMul(bangc::physical::MatMulOp matrix, bangc::physical::ConfigurationAttr config);
bool supportedScalarBinary(BinaryOperator kind);
bool initializeMatrixStorage(func::FuncOp function);
bool retainNarrowExtremaInputs(func::FuncOp function, bangc::physical::ConfigurationAttr config);
bool bindBroadcastOperands(func::FuncOp function);
bool propagateLocalTransposeSupply(func::FuncOp function, bangc::physical::ConfigurationAttr config);
bool realizeAffineRanges(func::FuncOp function, bangc::physical::ConfigurationAttr config);
bool realizeFullWidthMasks(func::FuncOp function, bangc::physical::ConfigurationAttr config);
bool vectorizeIndexLoops(func::FuncOp function, bangc::physical::ConfigurationAttr config);
bool lowerPowerOfTwoDivisions(func::FuncOp function);
bool batchIndependentRowPrograms(func::FuncOp function, bangc::physical::ConfigurationAttr config);
Value allocate(OpBuilder &b, Location loc, Type element, ArrayRef<int64_t> shape, int64_t space);
bool reuseConsumedBinaryInputs(func::FuncOp function, bangc::physical::ConfigurationAttr config);
bool reuseConsumedExp2Inputs(func::FuncOp function, bangc::physical::ConfigurationAttr config);
bool placeInvariantSupply(func::FuncOp function, bangc::physical::ConfigurationAttr config);
bool reuseCompletedSupply(func::FuncOp function, bangc::physical::ConfigurationAttr config);
bool coarsenLocalPrograms(func::FuncOp function, bangc::physical::ConfigurationAttr config);
void pipelineLocalSupply(func::FuncOp function, bangc::physical::ConfigurationAttr config);
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
