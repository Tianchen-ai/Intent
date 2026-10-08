#ifndef INTENT_SERIALIZATION_NATIVESOURCE_H
#define INTENT_SERIALIZATION_NATIVESOURCE_H

#include "Intent/Serialization/MemoryDescriptor.h"
#include "Intent/Serialization/Source.h"
#include "mlir/IR/Builders.h"
#include <optional>

namespace intent {

enum class NativeSourceSyntax { C, Mojo };

// Translates the metadata and control transfers already expressed by ranked
// memrefs and SCF. Allocation, storage spaces and native accesses remain target
// operations; this class neither bufferizes values nor selects an execution model.
class NativeSourceEmitter : public SourceEmitter {
public:
  NativeSourceEmitter(llvm::raw_ostream &output, NativeSourceSyntax syntax);

  virtual std::string nativeType(mlir::Type type) = 0;
  virtual std::string offsetPointer(llvm::StringRef base,
                                    llvm::StringRef offset) = 0;
  virtual std::string pointerAsIndex(llvm::StringRef base) = 0;
  virtual mlir::LogicalResult emitNativeOperation(mlir::Operation *operation) = 0;

  // Entry pointers already point at the public view's origin. Dynamic extents
  // and strides are filled from the existing native ABI slots by the provider.
  mlir::LogicalResult bindEntryMemory(mlir::Value value, llvm::StringRef base);
  MemoryDescriptor allocationDescriptor(mlir::MemRefType type,
                                        llvm::StringRef base,
                                        mlir::ValueRange dynamicSizes);
  std::string memoryOffset(mlir::Value memory, mlir::ValueRange indices = {});
  std::string memoryPointer(mlir::Value memory, mlir::ValueRange indices = {});
  std::string foldIndex(mlir::OpFoldResult value);

  llvm::SmallVector<std::string> components(mlir::Value value);
  llvm::SmallVector<mlir::Type> componentTypes(mlir::Type type);
  void declare(mlir::Value value, mlir::Value initial = {});
  void alias(mlir::Value result, mlir::Value source);
  void transfer(mlir::ValueRange from, mlir::ValueRange to);
  void declareComponent(llvm::StringRef name, mlir::Type type,
                         std::optional<llvm::StringRef> initial = std::nullopt);
  void assignComponent(llvm::StringRef name, llvm::StringRef expression);
  void bindExpression(mlir::Value value, llvm::StringRef expression);
  mlir::LogicalResult emitNativeBlock(mlir::Block &block);

  static const OperationEmitters<NativeSourceEmitter> &metadataEmitters();
  static const OperationEmitters<NativeSourceEmitter> &controlEmitters();

  // Per-translation spellings, never an additional program or layout authority.
  llvm::DenseMap<mlir::Value, MemoryDescriptor> memories;

protected:
  NativeSourceSyntax syntax;
};

// Common checks for the standard memory operations used by target handlers.
// Target checks additionally establish supported types, memory spaces and APIs.
mlir::LogicalResult verifyNativeMemoryType(mlir::Operation *owner,
                                          mlir::MemRefType type);
mlir::LogicalResult verifyNativeAllocation(mlir::Operation *operation,
                                          bool requireStaticShape);

} // namespace intent
#endif
