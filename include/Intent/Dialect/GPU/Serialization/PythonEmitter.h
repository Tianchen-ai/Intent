#ifndef INTENT_DIALECT_GPU_SERIALIZATION_PYTHONEMITTER_H
#define INTENT_DIALECT_GPU_SERIALIZATION_PYTHONEMITTER_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/Serialization/Python.h"
#include "Intent/Serialization/Source.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"

namespace intent::gpu {

// Python source providers consume the same executable GPU values and regions.
// These hooks spell the language differences; they do not select physical forms.
class PythonEmitter : public SourceEmitter {
public:
  using Emitters = OperationEmitters<PythonEmitter>;

  PythonEmitter(mlir::func::FuncOp kernel, llvm::raw_ostream &output,
                PythonScalarSyntax scalarSyntax,
                PythonExpressionSyntax expressionSyntax,
                llvm::StringRef constexprAnnotation);
  virtual ~PythonEmitter() = default;

  static void addCommonOperations(Emitters &emitters);
  void emitOperation(mlir::Operation &operation);
  void emitBlock(mlir::Block &block, llvm::ArrayRef<std::string> results = {});
  void emitHelper(mlir::Operation *owner, mlir::Region &region,
                  llvm::StringRef name, llvm::StringRef decorator,
                  llvm::StringRef argumentPrefix);

  std::string expressionString(PhysicalExprAttr expression);
  std::string fragmentShape(FragmentType type);
  std::string pythonType(mlir::Type type) const;
  std::string literal(mlir::Attribute value);
  static mlir::Type elementType(mlir::Type type);
  static std::string stringTuple(llvm::ArrayRef<std::string> values);
  static std::string stringList(llvm::ArrayRef<std::string> values);
  static std::string axisTuple(llvm::ArrayRef<int64_t> axes);
  std::string tuple(mlir::ValueRange values, bool control = false);
  std::string helperName(mlir::Operation *owner) const;
  void assign(mlir::Value value, llvm::StringRef expression,
              bool compileTime = false);
  void assignResults(mlir::ResultRange results, llvm::StringRef expression);
  bool isConstexprValue(mlir::Value value) const { return constexprValues.contains(value); }
  void bindConstexpr(mlir::Value value, llvm::StringRef expression) {
    bind(value, expression);
    constexprValues.insert(value);
  }

  virtual const Emitters &operationEmitters() const = 0;
  virtual void emitConstant(mlir::arith::ConstantOp operation) = 0;
  virtual std::string programId(ProgramIdOp operation) = 0;
  virtual std::string makeRange(MakeRangeOp operation) = 0;
  virtual std::string splat(SplatOp operation) = 0;
  virtual std::string castValue(mlir::Value value, mlir::Type result,
                                bool bitcast) = 0;
  virtual std::string reshape(ReshapeOp operation) = 0;
  virtual std::string join(JoinOp operation) = 0;
  virtual std::string broadcastValue(mlir::Value value,
                                     FragmentType result) = 0;
  virtual std::string controlValueString(mlir::Value value);
  virtual std::string loopInitialValue(mlir::Value value);
  virtual std::string forRange(mlir::scf::ForOp operation) = 0;
  virtual std::string select(SelectOp operation) = 0;
  virtual std::string permute(mlir::Value value,
                              llvm::ArrayRef<int64_t> permutation) = 0;
  virtual bool inlineRecords() const { return false; }

protected:
  mlir::func::FuncOp kernel;
  llvm::DenseSet<mlir::Value> constexprValues;
  bool emittingHelper = false;

private:
  void emitFor(mlir::scf::ForOp operation);
  void emitIf(mlir::scf::IfOp operation);
  void emitWhile(mlir::scf::WhileOp operation);
  void assignYield(mlir::ValueRange values, llvm::ArrayRef<std::string> results);
  PythonScalarSyntax scalarSyntax;
  PythonExpressionSyntax expressionSyntax;
  std::string constexprAnnotation;
  llvm::DenseMap<mlir::Operation *, std::string> helperNames;
};

} // namespace intent::gpu
#endif
