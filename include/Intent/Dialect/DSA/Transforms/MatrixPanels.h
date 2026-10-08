#ifndef INTENT_DIALECT_DSA_TRANSFORMS_MATRIXPANELS_H
#define INTENT_DIALECT_DSA_TRANSFORMS_MATRIXPANELS_H

#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace intent::dsa {

struct MatrixPanelStorage {
  int64_t nram, wram;
};

// Same-dtype matrix input/packing storage, including an alternate LHS panel.
// Failure includes unsupported native geometry and unrepresentable sizes.
std::optional<MatrixPanelStorage>
matrixPanelStorage(mlir::Type element, int64_t rows, int64_t columns,
                   int64_t depth);

// Formation reserves every still-live source allocation and its complete local
// producer/consumer storage before adding native matrix preparation.
bool matrixPanelsFitStorage(mlir::func::FuncOp function, ConfigurationAttr config,
                            llvm::ArrayRef<MatrixPanelStorage> panels,
                            int64_t localReserve);

struct MatrixPanelShape { int64_t rows, columns; };

// Larger free-axis tiles for this already materialized matrix. Candidates
// reduce the number of rectangles and pass a standalone storage bound; callers
// must still measure the actual program before accepting one.
llvm::SmallVector<MatrixPanelShape> largerMatrixPanels(
    ConfigurationAttr config, mlir::Type element, int64_t rows, int64_t columns,
    int64_t depth, MatrixPanelShape baseline);

// Call only while forming one ordinary contraction. availableDepth is a
// complete source panel; selection never authorizes merging ordered updates.
// Existing allocations are conservatively retained alongside the new panels.
int64_t selectMatrixPanelDepth(mlir::func::FuncOp function,
                              ConfigurationAttr config, mlir::Type element,
                              int64_t rows, int64_t columns,
                              int64_t availableDepth, int64_t baselineDepth,
                              unsigned products = 1);

struct StreamedMatrixPanel {
  int64_t depth, sliceDepth, sliceColumns;
  bool pipeline;
};

// A complete WRAM RHS prepared through bounded NRAM slices. Its consumers
// retain one or two full-K LHS slots; both SRAM supply slots are reserved.
std::optional<StreamedMatrixPanel>
selectStreamedMatrixPanel(mlir::func::FuncOp function, ConfigurationAttr config,
                         mlir::Type element, int64_t rows, int64_t columns,
                         int64_t depth, int64_t localReserve);

} // namespace intent::dsa
#endif
