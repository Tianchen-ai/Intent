#ifndef INTENT_DIALECT_GPU_TRANSFORMS_ONLINESUMMARY_H
#define INTENT_DIALECT_GPU_TRANSFORMS_ONLINESUMMARY_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"

#include "mlir/IR/Builders.h"

namespace intent::gpu {

struct NormalizedSummaryStructure {
  ReduceOp validity;
  ReduceOp maximum;
  ReduceOp mass;
  ContractOp moment;
  SelectOp maximumOrEmpty;
  SelectOp maskedScore;
  SelectOp probability;
  CastOp probabilityCast;
  UnaryOp exponential;
  mlir::Value memberValidity;
  mlir::Value score;
  mlir::Value values;
  unsigned reductionAxis;
  unsigned valueReductionAxis;
  PhysicalSourceAxis traversal;
  SelectOp momentOrEmpty;
};

struct OnlineSummaryStructure : NormalizedSummaryStructure {
  MakeRecordOp record;
  unsigned validityField;
  unsigned maximumField;
  unsigned massField;
  unsigned momentField;
};

struct OnlineSummaryMerge {
  MakeRecordOp record;
  mlir::Value combinedMaximum;
  mlir::Value leftScale;
  mlir::Value rightScale;
  mlir::Value leftMassTerm;
  mlir::Value leftMomentTerm;
};

mlir::FailureOr<OnlineSummaryStructure>
matchOnlineSummaryStructure(MakeRecordOp record);

mlir::FailureOr<NormalizedSummaryStructure>
matchNormalizedSummaryStructure(ContractOp moment);

mlir::FailureOr<OnlineSummaryMerge>
matchOnlineSummaryMerge(mlir::Region &region,
                        const OnlineSummaryStructure &summary);

ReduceOp cloneReductionWithSource(mlir::OpBuilder &builder,
                                  mlir::Location location, ReduceOp source,
                                  mlir::Value value);

} // namespace intent::gpu

#endif
