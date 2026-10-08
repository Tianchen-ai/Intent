#include "Intent/Dialect/GPU/IR/IntegerRanges.h"
#include "Intent/Analysis/IntegerRanges.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

namespace intent::gpu {
namespace {

Type elementType(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return fragment.getElementType();
  return type;
}

template <typename Model, typename Op>
struct IntegerRangeModel
    : InferIntRangeInterface::ExternalModel<Model, Op> {
  void inferResultRangesFromOptional(Operation *operation,
                                    ArrayRef<IntegerValueRange> operands,
                                    SetIntLatticeFn setResult) const {
    intrange::detail::defaultInferResultRanges(
        cast<InferIntRangeInterface>(operation), operands, setResult);
  }
};

struct BinaryRanges : IntegerRangeModel<BinaryRanges, BinaryOp> {
  void inferResultRanges(Operation *operation,
                         ArrayRef<ConstantIntRanges> operands,
                         SetIntRangeFn setResult) const {
    auto binary = cast<BinaryOp>(operation);
    if (auto result = inferIntegerBinary(
            binary.getOperatorKind(), elementType(binary.getResult().getType()),
            operands[0], operands[1]))
      setResult(binary.getResult(), *result);
  }
};

struct UnaryRanges : IntegerRangeModel<UnaryRanges, UnaryOp> {
  void inferResultRanges(Operation *operation,
                         ArrayRef<ConstantIntRanges> operands,
                         SetIntRangeFn setResult) const {
    auto unary = cast<UnaryOp>(operation);
    if (auto result = inferIntegerUnary(
            unary.getOperatorKind(), elementType(unary.getResult().getType()),
            operands[0]))
      setResult(unary.getResult(), *result);
  }
};

struct CompareRanges : IntegerRangeModel<CompareRanges, CompareOp> {
  void inferResultRanges(Operation *operation,
                         ArrayRef<ConstantIntRanges> operands,
                         SetIntRangeFn setResult) const {
    auto compare = cast<CompareOp>(operation);
    if (auto result = inferIntegerCompare(
            compare.getPredicate(), elementType(compare.getLhs().getType()),
            operands[0], operands[1]))
      setResult(compare.getResult(), *result);
  }
};

struct CastRanges : IntegerRangeModel<CastRanges, CastOp> {
  void inferResultRanges(Operation *operation,
                         ArrayRef<ConstantIntRanges> operands,
                         SetIntRangeFn setResult) const {
    auto conversion = cast<CastOp>(operation);
    if (auto result = inferIntegerCast(
            elementType(conversion.getValue().getType()),
            elementType(conversion.getResult().getType()), operands[0]))
      setResult(conversion.getResult(), *result);
  }
};

struct SelectRanges : IntegerRangeModel<SelectRanges, SelectOp> {
  void inferResultRanges(Operation *operation,
                         ArrayRef<ConstantIntRanges> operands,
                         SetIntRangeFn setResult) const {
    auto select = cast<SelectOp>(operation);
    auto condition = operands[0].getConstantValue();
    setResult(select.getResult(), condition
                                      ? operands[condition->isZero() ? 2 : 1]
                                      : operands[1].rangeUnion(operands[2]));
  }
};

template <typename Op>
struct ForwardRanges
    : IntegerRangeModel<ForwardRanges<Op>, Op> {
  void inferResultRanges(Operation *operation,
                         ArrayRef<ConstantIntRanges> operands,
                         SetIntRangeFn setResult) const {
    Type source = elementType(operation->getOperand(0).getType());
    Type result = elementType(operation->getResult(0).getType());
    if (isa<IntegerType, IndexType>(source) && source == result)
      setResult(operation->getResult(0), operands[0]);
  }
};

struct JoinRanges : IntegerRangeModel<JoinRanges, JoinOp> {
  void inferResultRanges(Operation *operation,
                         ArrayRef<ConstantIntRanges> operands,
                         SetIntRangeFn setResult) const {
    setResult(operation->getResult(0), operands[0].rangeUnion(operands[1]));
  }
};

} // namespace

void registerIntegerRangeInterfaces(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, IntentGPUDialect *) {
    BinaryOp::attachInterface<BinaryRanges>(*context);
    UnaryOp::attachInterface<UnaryRanges>(*context);
    CompareOp::attachInterface<CompareRanges>(*context);
    CastOp::attachInterface<CastRanges>(*context);
    SelectOp::attachInterface<SelectRanges>(*context);
    SplatOp::attachInterface<ForwardRanges<SplatOp>>(*context);
    BroadcastOp::attachInterface<ForwardRanges<BroadcastOp>>(*context);
    ReshapeOp::attachInterface<ForwardRanges<ReshapeOp>>(*context);
    TransposeOp::attachInterface<ForwardRanges<TransposeOp>>(*context);
    JoinOp::attachInterface<JoinRanges>(*context);
  });
}

} // namespace intent::gpu
