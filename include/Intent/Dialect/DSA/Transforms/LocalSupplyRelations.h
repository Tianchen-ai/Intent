#ifndef INTENT_DSA_TRANSFORMS_LOCALSUPPLYRELATIONS_H
#define INTENT_DSA_TRANSFORMS_LOCALSUPPLYRELATIONS_H

#include "Intent/Analysis/IntegerRanges.h"
#include "Intent/Dialect/DSA/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/DSA/Analysis/Storage.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/AffineExpr.h"

namespace intent::dsa {

// Current typed scalar coordinates of a local supply. Discard after mutation.
class LocalSupplyRelations {
public:
  explicit LocalSupplyRelations(mlir::func::FuncOp function);
  dsa::SignedInterval interval(mlir::Value value);
  bool equal(mlir::Value value, int64_t expected);
  bool equal(mlir::Value first, mlir::Value second);
  mlir::AffineExpr difference(mlir::Value first, mlir::Value second);

private:
  std::optional<int64_t> constant(mlir::Value value);
  mlir::AffineExpr expression(mlir::Value value);
  mlir::func::FuncOp function;
  intent::IntegerRangeAnalysis ranges;
  llvm::DenseMap<mlir::Value, mlir::AffineExpr> expressions;
  unsigned symbols = 0;
};

struct LocalIndexSequence {
  mlir::scf::ForOp initialization;
  mlir::Value value;
  llvm::SmallVector<mlir::Operation *> definitions;
  mlir::AffineExpr begin;
  SignedInterval bounds;
  mlir::Value materializeBegin(mlir::OpBuilder &builder, mlir::Location location) const;
};

// A completed, full unit-step row/column snapshot at this actual read point.
// Copies are followed at their original read points, never at a later reader.
std::optional<LocalIndexSequence> queryLocalIndexSequence(
    mlir::Value memory, mlir::Operation *reader, StorageAnalysis &storage,
    mlir::DominanceInfo &dominance, LocalSupplyRelations &relations);

struct LocalCoordinateGrid { LoadTileOp rows, columns; };
std::optional<LocalCoordinateGrid> queryLocalCoordinateGrid(
    CompareOp comparison, StorageAnalysis &storage, LocalSupplyRelations &relations);

} // namespace intent::dsa
#endif
