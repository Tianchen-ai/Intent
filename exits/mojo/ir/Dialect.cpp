#include "compilerinfra/Contract.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <algorithm>
#include "Dialect.h.inc"
#include "Dialect.cpp.inc"
#define GET_OP_CLASSES
#include "Ops.h.inc"
#define GET_OP_CLASSES
#include "Ops.cpp.inc"

using namespace mlir;
using namespace intentdsl::rowmojo;

namespace {
using EffectList =
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>;

struct MojoSimdFmaContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          MojoSimdFmaContract, MojoSimdFma> {
  Type getAccumulatorType(Operation *op) const {
    if (auto isInt = op->getAttrOfType<BoolAttr>("is_int");
        isInt && isInt.getValue())
      return IntegerType::get(op->getContext(), 32);
    Type elem = getElementTypeOrSelf(op->getResult(0).getType());
    if (elem.isInteger(32))
      return elem;
    return Float32Type::get(op->getContext());
  }
};

template <typename ConcreteOp>
struct MojoF32AccContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          MojoF32AccContract<ConcreteOp>, ConcreteOp> {
  Type getAccumulatorType(Operation *op) const {
    return Float32Type::get(op->getContext());
  }
};

template <typename ConcreteOp>
struct MojoI32AccContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          MojoI32AccContract<ConcreteOp>, ConcreteOp> {
  Type getAccumulatorType(Operation *op) const {
    return IntegerType::get(op->getContext(), 32);
  }
};

struct MojoBrgemmContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          MojoBrgemmContract, MojoBrgemmCall> {
  Type getAccumulatorType(Operation *op) const {
    if (auto impl = op->getAttrOfType<StringAttr>("ci.implementation")) {
      StringRef s = impl.getValue();
      if (s.ends_with("-s8") || s.ends_with("-u8s8"))
        return IntegerType::get(op->getContext(), 32);
    }
    return Float32Type::get(op->getContext());
  }
};

struct MojoScanContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          MojoScanContract, MojoSimdScanSum> {
  Type getAccumulatorType(Operation *op) const {
    auto dt = op->getAttrOfType<StringAttr>("dtype");
    if (dt && dt.getValue() == "int32")
      return IntegerType::get(op->getContext(), 32);
    if (dt && dt.getValue() == "int64")
      return IntegerType::get(op->getContext(), 64);
    Type elem = getElementTypeOrSelf(op->getResult(0).getType());
    if (elem.isInteger(32) || elem.isInteger(64) || elem.isF32())
      return elem;
    return Float32Type::get(op->getContext());
  }
};

struct MojoTransposeContract
    : public compilerinfra::ContractOpInterface::ExternalModel<
          MojoTransposeContract, MojoTranspose8x8> {};

struct MojoOwnerCoordRangeModel
    : public InferIntRangeInterface::ExternalModel<MojoOwnerCoordRangeModel,
                                                   MojoOwnerCoord> {
  void inferResultRanges(Operation *op, ArrayRef<ConstantIntRanges> argRanges,
                         SetIntRangeFn setResultRanges) const {
    intrange::detail::defaultInferResultRangesFromOptional(
        cast<InferIntRangeInterface>(op), argRanges, setResultRanges);
  }
  void inferResultRangesFromOptional(Operation *op,
                                     ArrayRef<IntegerValueRange>,
                                     SetIntLatticeFn setResultRanges) const {
    auto ext = op->getAttrOfType<IntegerAttr>("extent");
    if (!ext || ext.getInt() <= 0) {
      setResultRanges(op->getResult(0), ConstantIntRanges::maxRange(64));
      return;
    }
    APInt lo(64, 0, /*isSigned=*/true);
    APInt hi(64, uint64_t(ext.getInt() - 1), /*isSigned=*/true);
    setResultRanges(op->getResult(0), ConstantIntRanges(lo, hi, lo, hi));
  }
};
} // namespace

void MojoQ4KSuperblock::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}

void MojoBrgemmCall::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

void MojoTranspose8x8::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

void intentdsl::rowmojo::RowMojoDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
  MLIRContext *ctx = getContext();
  MojoSimdFma::attachInterface<MojoSimdFmaContract>(*ctx);
  MojoVnniDot4::attachInterface<MojoI32AccContract<MojoVnniDot4>>(*ctx);
  MojoQ4KSuperblock::attachInterface<MojoI32AccContract<MojoQ4KSuperblock>>(
      *ctx);
  MojoBrgemmCall::attachInterface<MojoBrgemmContract>(*ctx);
  MojoUniform::attachInterface<MojoI32AccContract<MojoUniform>>(*ctx);
  MojoReduceSum16x16::attachInterface<MojoF32AccContract<MojoReduceSum16x16>>(
      *ctx);
  MojoWelfordVecStep::attachInterface<MojoF32AccContract<MojoWelfordVecStep>>(
      *ctx);
  MojoWelfordVecReduce::attachInterface<
      MojoF32AccContract<MojoWelfordVecReduce>>(*ctx);
  MojoWelfordScalarStep::attachInterface<
      MojoF32AccContract<MojoWelfordScalarStep>>(*ctx);
  MojoTranspose8x8::attachInterface<MojoTransposeContract>(*ctx);
  MojoSimdScanSum::attachInterface<MojoScanContract>(*ctx);
  MojoOwnerCoord::attachInterface<MojoOwnerCoordRangeModel>(*ctx);
}
