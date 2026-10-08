#ifndef INTENT_DSA_TRANSFORMS_COLLECTIVE_LOWERING_H
#define INTENT_DSA_TRANSFORMS_COLLECTIVE_LOWERING_H

#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>

namespace intent::dsa::collective {

mlir::Value index(mlir::OpBuilder &builder, mlir::Location location,
                  int64_t value);
mlir::Value allocate(mlir::OpBuilder &builder, mlir::Location location,
                     mlir::Type element, llvm::ArrayRef<int64_t> shape);
mlir::Value view(mlir::OpBuilder &builder, mlir::Location location,
                 mlir::Value storage, llvm::ArrayRef<int64_t> shape);
mlir::Value physicalView(mlir::OpBuilder &builder, mlir::Location location,
                         mlir::Value storage);
void fill(mlir::OpBuilder &builder, mlir::Location location,
          mlir::Value storage, mlir::Value scalar);
void copy(mlir::OpBuilder &builder, mlir::Location location,
          mlir::Value source, mlir::Value destination);
mlir::LogicalResult forEach(
    mlir::OpBuilder &builder, mlir::Location location, mlir::ValueRange counts,
    llvm::function_ref<mlir::LogicalResult(mlir::ValueRange)> body);
mlir::Value load(mlir::OpBuilder &builder, mlir::Location location,
                 mlir::Value storage, mlir::ValueRange coordinates);
void store(mlir::OpBuilder &builder, mlir::Location location, mlir::Value value,
           mlir::Value storage, mlir::ValueRange coordinates);
mlir::Value linearOffset(mlir::OpBuilder &builder, mlir::Location location,
                         mlir::MemRefType type, mlir::ValueRange coordinates);

// This query concerns a concrete scalar computation in the current IR. It
// neither recognizes source functions nor substitutes an algorithm template.
std::optional<BinaryOperator> binaryKind(mlir::Operation *operation);
bool canLiftScalarCombine(mlir::Block &combine);
mlir::FailureOr<llvm::SmallVector<mlir::Value>> liftScalarCombine(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Block &combine,
    mlir::ValueRange arguments, llvm::ArrayRef<int64_t> shape);

struct Field {
  mlir::Value source;
  mlir::Value initial;
  mlir::Value output;
  mlir::Value final;
  mlir::Type formalType;
  llvm::SmallVector<mlir::Value> counts;
  llvm::SmallVector<int64_t> freeShape;
  llvm::SmallVector<mlir::Value> freeCounts;
};

class Lowering {
public:
  explicit Lowering(SliceReduceOp operation);
  explicit Lowering(ScanOp operation);
  mlir::LogicalResult run();

private:
  void initialize(mlir::ValueRange sources, mlir::ValueRange initials,
                  mlir::ValueRange outputs, mlir::ValueRange counts,
                  mlir::ValueRange finals);
  bool nativeReduction();
  bool treeReduction();
  mlir::LogicalResult realize();
  mlir::LogicalResult realizeAt(mlir::ValueRange independent);
  bool transferFreeAxis(const Field &field, mlir::Value slot,
                        mlir::ValueRange reduced, bool read);
  llvm::SmallVector<mlir::Value> sourceCoordinates(
      const Field &field, mlir::ValueRange free,
      mlir::ValueRange reduced) const;
  mlir::FailureOr<llvm::SmallVector<mlir::Value>> combine(
      mlir::ValueRange left, mlir::ValueRange right, bool lifted);

  mlir::Operation *operation;
  mlir::OpBuilder builder;
  mlir::Location location;
  mlir::Block &body;
  llvm::SmallVector<Field> fields;
  llvm::SmallVector<mlir::Value> captures;
  llvm::SmallVector<int64_t> axes;
  bool scan = false;
  bool inclusive = true;
  bool reverse = false;
  bool scalar = false;
  bool lifted = false;
};

} // namespace intent::dsa::collective

#endif
