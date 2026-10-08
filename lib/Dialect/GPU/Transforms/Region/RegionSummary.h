#ifndef INTENT_GPU_TRANSFORMS_REGIONSUMMARY_H
#define INTENT_GPU_TRANSFORMS_REGIONSUMMARY_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include <optional>

namespace intent::gpu::region {

struct SummaryEmptinessPlan {
  unsigned optionalField;
  RecordType fullType;
  RecordType payloadType;
  mlir::Value summarizeValidity;
};

std::optional<bool> booleanConstant(mlir::Value value);
bool isRecordField(mlir::Value value, mlir::BlockArgument record, unsigned field);
std::optional<SummaryEmptinessPlan>
summaryEmptinessPlan(RegionFoldOp fold, mlir::ValueRange identities,
                     mlir::Value membershipPredicate);

mlir::Value allTrueValue(mlir::OpBuilder &builder, mlir::Location location,
                        mlir::Type type);
mlir::FailureOr<mlir::Value>
stripOptionalRecord(mlir::OpBuilder &builder, mlir::Location location,
                    mlir::Value value, const SummaryEmptinessPlan &plan);
mlir::Value restoreOptionalRecord(mlir::OpBuilder &builder,
                                 mlir::Location location, mlir::Value payload,
                                 const SummaryEmptinessPlan &plan,
                                 mlir::Value optionalValidity = {});
mlir::FailureOr<mlir::Value> restoreOnlineRecord(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value payload,
    const SummaryEmptinessPlan &plan, unsigned massField,
    std::optional<unsigned> encodedMaximum = {});

void simplifyKnownRecordValues(mlir::func::FuncOp kernel);
void foldKnownRecordProjections(mlir::Operation *structured);
void forwardUnusedRecordFields(RegionFoldOp fold);

} // namespace intent::gpu::region
#endif
