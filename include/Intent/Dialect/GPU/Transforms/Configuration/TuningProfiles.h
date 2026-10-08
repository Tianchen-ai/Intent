#ifndef INTENT_DIALECT_GPU_TRANSFORMS_TUNINGPROFILES_H
#define INTENT_DIALECT_GPU_TRANSFORMS_TUNINGPROFILES_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace intent::gpu {

struct ProviderOption {
  llvm::StringRef name;
  ParameterRole role;
  bool isBoolean = false;
};

struct TuningProfileSchema {
  llvm::StringRef space;
  llvm::ArrayRef<llvm::StringRef> columns;
  llvm::ArrayRef<ProviderOption> options;
};

struct TuningProfileSource {
  TuningProfileSchema schema;
  std::string filename;
};

class TuningProfiles {
public:
  using Row = llvm::SmallVector<int64_t, 7>;
  using Table = llvm::SmallVector<Row, 5>;

  static mlir::FailureOr<TuningProfiles>
  read(mlir::Location location, llvm::ArrayRef<TuningProfileSource> defaults,
       llvm::StringRef overrideFilename);

  static mlir::FailureOr<TuningProfiles> from(mlir::ModuleOp module);
  void attach(mlir::ModuleOp module) const;
  mlir::FailureOr<TuningProfileTableAttr> table() const;
  mlir::LogicalResult declareProviderParameters(
      mlir::func::FuncOp kernel, const TuningProfileSchema &schema) const;

  mlir::FailureOr<Table>
  get(const TuningProfileSchema &schema, llvm::StringRef family,
      mlir::Location location) const;

private:
  explicit TuningProfiles(TuningProfilesAttr profiles) : profiles(profiles) {}
  TuningProfilesAttr profiles;
};

} // namespace intent::gpu

#endif
