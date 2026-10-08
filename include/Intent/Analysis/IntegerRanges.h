#ifndef INTENT_ANALYSIS_INTEGERRANGES_H
#define INTENT_ANALYSIS_INTEGERRANGES_H

#include "Intent/Dialect/Intent/IR/IntentAttrs.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>
#include <optional>

namespace intent {

class IntegerRangeAnalysis;

struct IntegerRangePolicy {
  // A family's shaped integer value may use a dialect-owned element type.
  std::function<mlir::Type(mlir::Value)> elementType;
  // Current-IR facts such as parameter domains or task coordinates. Returning
  // nullopt delegates to standard operations/control/descriptor inference.
  std::function<std::optional<mlir::ConstantIntRanges>(
      mlir::Value, IntegerRangeAnalysis &)> infer;
};

// A demand-driven analysis of one IR snapshot. It never traverses incomplete
// sibling regions or starts a whole-program solver. Recreate after mutation.
// Unknown integers have their full bit-width range; noninteger values are null.
// Logical index is signed 64-bit, independently of a target's address width.
class IntegerRangeAnalysis {
public:
  explicit IntegerRangeAnalysis(IntegerRangePolicy policy = {});
  std::optional<mlir::ConstantIntRanges> range(mlir::Value value);
  std::optional<mlir::ConstantIntRanges> dimension(mlir::Value shaped,
                                                unsigned axis);
  bool isNonNegative(mlir::Value value);
  bool isPositive(mlir::Value value);

private:
  mlir::Type elementType(mlir::Value value) const;
  std::optional<mlir::ConstantIntRanges> infer(mlir::Value value);
  IntegerRangePolicy policy;
  llvm::DenseMap<mlir::Value, mlir::ConstantIntRanges> known;
  llvm::DenseSet<mlir::Value> active;
  llvm::DenseSet<std::pair<mlir::Value, unsigned>> activeDimensions;
};

std::optional<mlir::ConstantIntRanges>
inferIntegerBinary(BinaryOperator kind, mlir::Type elementType,
                   const mlir::ConstantIntRanges &lhs,
                   const mlir::ConstantIntRanges &rhs);
std::optional<mlir::ConstantIntRanges>
inferIntegerUnary(UnaryOperator kind, mlir::Type elementType,
                  const mlir::ConstantIntRanges &input);
std::optional<mlir::ConstantIntRanges>
inferIntegerCompare(ComparePredicate predicate, mlir::Type inputType,
                    const mlir::ConstantIntRanges &lhs,
                    const mlir::ConstantIntRanges &rhs);
std::optional<mlir::ConstantIntRanges>
inferIntegerCast(mlir::Type sourceType, mlir::Type targetType,
                 const mlir::ConstantIntRanges &input);

// These certificates concern mathematical operand bounds, not the sign of a
// wrapped result. No nsw/nuw assumption is added to the program.
bool provesSignedNoWrap(BinaryOperator kind,
                        const mlir::ConstantIntRanges &lhs,
                        const mlir::ConstantIntRanges &rhs);
bool isValuePreservingIntegerCast(mlir::Type sourceType, mlir::Type targetType,
                                  const mlir::ConstantIntRanges &input);

} // namespace intent
#endif
