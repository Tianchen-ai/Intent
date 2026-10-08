#include "Intent/Target/Triton/IR/TritonOps.h"

#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/Analysis/ProgramInterface.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <limits>

using namespace mlir;

namespace intent::triton {
namespace {

template <typename CollectiveOp>
LogicalResult inferCollectiveTypes(MLIRContext *context,
    std::optional<Location> location, ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions, bool scan,
    SmallVectorImpl<Type> &results) {
  typename CollectiveOp::Adaptor operation(operands, attributes, properties, regions);
  if (failed(operation.verify(location.value_or(UnknownLoc::get(context)))))
    return failure();
  return gpu::inferScalarCollectiveResultTypes(
      location, operation.getSources(), operation.getAxisAttr().getInt(), scan, results);
}

LogicalResult verifyDescriptorAccess(Operation *owner, Value descriptorValue,
                                     gpu::FragmentType fragment,
                                     ValueRange offsets) {
  auto descriptor = descriptorValue.getDefiningOp<TensorDescriptorOp>();
  if (!descriptor)
    return owner->emitOpError(
        "requires an explicit tensor-descriptor value");
  auto base = cast<gpu::ViewType>(descriptor.getBase().getType());
  if (base.getElementType() != fragment.getElementType())
    return owner->emitOpError(
        "tensor descriptor and fragment element types must match");
  if (offsets.size() != descriptor.getShape().size() ||
      fragment.getShape().size() != descriptor.getBlockShape().size())
    return owner->emitOpError(
        "descriptor offsets and fragment rank must match its declared shape");
  for (auto [extent, blockShape] :
       llvm::zip(fragment.getShape(), descriptor.getBlockShape())) {
    auto expression = gpu::queryLaunchExpression(blockShape);
    if (!expression || expression != extent)
      return owner->emitOpError(
          "fragment extents must equal the declared descriptor block shape");
  }
  return success();
}

bool hasStrideBinding(func::FuncOp kernel, gpu::ViewType view, unsigned axis) {
  auto stride = cast<gpu::PhysicalExprAttr>(view.getLayout().getStrides()[axis]);
  if (stride.getKind() == gpu::PhysicalExprKind::Constant)
    return stride.getValue() > 0;
  return bool(gpu::resolveArgument(kernel, stride.getArgumentReference()));
}

bool matchesStrideBinding(func::FuncOp kernel, gpu::ViewType view, unsigned axis,
                          Value value) {
  return gpu::queryLaunchExpression(value) ==
      cast<gpu::PhysicalExprAttr>(view.getLayout().getStrides()[axis]);
}

bool hasDescriptorLayout(func::FuncOp kernel, gpu::ViewType view) {
  auto layout = view.getLayout();
  for (unsigned axis = 0; axis < view.getRank(); ++axis)
    if (!hasStrideBinding(kernel, view, axis))
      return false;

  auto lastStride = cast<gpu::PhysicalExprAttr>(
      layout.getStrides()[layout.getStrides().size() - 1]);
  if (lastStride.getKind() == gpu::PhysicalExprKind::Constant &&
      lastStride.getValue() != 1)
    return false;
  unsigned elementBytes = view.getElementType().getIntOrFloatBitWidth() / 8;
  for (unsigned axis = 0; axis + 1 < view.getRank(); ++axis) {
    auto stride = cast<gpu::PhysicalExprAttr>(layout.getStrides()[axis]);
    if (stride.getKind() == gpu::PhysicalExprKind::Constant &&
        (stride.getValue() * elementBytes) % 16 != 0)
      return false;
  }
  return true;
}

} // namespace

LogicalResult ReduceOp::inferReturnTypes(MLIRContext *context,
    std::optional<Location> location, ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions, SmallVectorImpl<Type> &results) {
  return inferCollectiveTypes<ReduceOp>(context, location, operands, attributes,
                                      properties, regions, false, results);
}

LogicalResult ScanOp::inferReturnTypes(MLIRContext *context,
    std::optional<Location> location, ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions, SmallVectorImpl<Type> &results) {
  return inferCollectiveTypes<ScanOp>(context, location, operands, attributes,
                                    properties, regions, true, results);
}

LogicalResult ReduceOp::verify() {
  if (getReverse())
    return emitOpError("native reduction does not reverse logical order");
  return gpu::verifyScalarCollective(getOperation(), getSources(), getIdentities(),
                                     getResults(), getCombine(), getAxisAttr().getInt(), false);
}

LogicalResult ScanOp::verify() {
  return gpu::verifyScalarCollective(getOperation(), getSources(), getIdentities(),
                                     getResults(), getCombine(), getAxisAttr().getInt(), true);
}

LogicalResult TensorDescriptorChoiceOp::verify() {
  auto kernel = (*this)->getParentOfType<func::FuncOp>();
  if (!kernel || !kernel->hasAttr(gpu::kernelAttr))
    return emitOpError("must belong to a physical GPU kernel");
  unsigned choices = 0;
  kernel.walk([&](TensorDescriptorChoiceOp) { ++choices; });
  if (choices != 1)
    return emitOpError(
        "requires exactly one tensor descriptor choice per physical kernel");
  if (getConstruction() != "host")
    return emitOpError("supports only explicit host descriptor binding");
  if (getSelection() != "all_eligible")
    return emitOpError(
        "supports only an all-descriptor runtime eligibility contract");
  StringRef configName = getConfigParameter().getName().getValue();
  if (configName.empty() || getEligibilityArgument().empty() ||
      configName == getEligibilityArgument())
    return emitOpError(
        "requires distinct non-empty config and runtime eligibility names");
  if (getDescriptors().empty())
    return emitOpError("requires a non-empty explicit descriptor set");
  llvm::SmallPtrSet<Operation *, 8> declared;
  for (Value value : getDescriptors()) {
    auto descriptor = value.getDefiningOp<TensorDescriptorOp>();
    if (!descriptor || descriptor->getParentOfType<func::FuncOp>() != kernel ||
        !declared.insert(descriptor.getOperation()).second)
      return emitOpError(
          "descriptor operands must uniquely enumerate this physical kernel's declarations");
  }
  unsigned descriptors = 0;
  bool completeDescriptorSet = true;
  kernel.walk([&](TensorDescriptorOp descriptor) {
    ++descriptors;
    if (!declared.contains(descriptor.getOperation()))
      completeDescriptorSet = false;
  });
  if (!completeDescriptorSet || descriptors != declared.size())
    return emitOpError(
        "descriptor operands must enumerate every descriptor declaration exactly once");
  auto declaration = gpu::lookupParameterDeclaration(getOperation(), getConfigParameter());
  const int64_t domain[] = {0, 1};
  if (!declaration || !declaration.getValueType().isSignlessInteger(1) ||
      declaration.getRole() != gpu::ParameterRole::ProviderAccessForm ||
      declaration.getPhase() != gpu::ConfigurationBindingPhase::Provider ||
      declaration.getCandidates().asArrayRef() != ArrayRef<int64_t>(domain))
    return emitOpError("requires a declared boolean provider access-form parameter");
  if (gpu::lookupParameterDeclaration(getOperation(),
          gpu::ParameterRefAttr::get(getContext(), getEligibilityArgumentAttr())))
    return emitOpError(
        "descriptor eligibility name must not collide with a compile-time declaration");
  return success();
}

LogicalResult TensorDescriptorAllocatorOp::verify() {
  auto kernel = (*this)->getParentOfType<func::FuncOp>();
  if (!kernel || !kernel->hasAttr(gpu::kernelAttr))
    return emitOpError("must belong to a physical GPU kernel");
  unsigned allocators = 0;
  unsigned descriptors = 0;
  kernel.walk([&](TensorDescriptorAllocatorOp) { ++allocators; });
  kernel.walk([&](TensorDescriptorOp) { ++descriptors; });
  if (allocators != 1 || descriptors == 0)
    return emitOpError(
        "requires one allocator declaration for a non-empty descriptor set");
  if (getSizeArgument() != 0 || getAlignmentArgument() != 1 ||
      getStreamArgument() != 2)
    return emitOpError(
        "allocator ABI must bind runtime size, alignment, and stream in order");
  if (getLifetime() != "launch")
    return emitOpError("supports only launch-lifetime descriptor allocation");
  if (getImplementation() != "torch_cuda_current_device")
    return emitOpError(
        "supports only the bound current-device Torch allocator implementation");
  return success();
}

LogicalResult TensorDescriptorOp::verify() {
  auto kernel = (*this)->getParentOfType<func::FuncOp>();
  if (!kernel || !kernel->hasAttr(gpu::kernelAttr))
    return emitOpError("must belong to a physical GPU kernel");
  auto base = cast<gpu::ViewType>(getBase().getType());
  auto baseArgument = dyn_cast<BlockArgument>(getBase());
  if (!baseArgument || baseArgument.getOwner() != &kernel.getBody().front() ||
      getResult().getType() != base)
    return emitOpError(
        "must preserve one external-view kernel argument as its base");
  unsigned rank = base.getRank();
  if (rank < 1 || rank > 5 || getShape().size() != rank ||
      getStrides().size() != rank || getBlockShape().size() != rank)
    return emitOpError(
        "host tensor descriptor must preserve a rank-1-to-5 view");
  auto blockAxes = getSourceBlockAxes();
  if (blockAxes.empty() ||
      !llvm::is_contained(blockAxes, static_cast<int64_t>(rank) - 1))
    return emitOpError(
        "tensor descriptor block axes must include the contiguous source axis");
  llvm::SmallBitVector blocked(rank);
  for (int64_t axis : blockAxes) {
    if (axis < 0 || axis >= rank || blocked.test(axis))
      return emitOpError("tensor descriptor block axes must be distinct source axes");
    blocked.set(axis);
  }
  int64_t elementBytes =
      static_cast<int64_t>(base.getElementType().getIntOrFloatBitWidth() / 8);
  if (getInitialBlockShape().size() != rank ||
      llvm::any_of(getInitialBlockShape().drop_back(),
                   [](int64_t extent) { return extent != 1; }) ||
      getInitialBlockShape().back() <= 0 ||
      !llvm::isPowerOf2_64(getInitialBlockShape().back()) ||
      getInitialBlockShape().back() * elementBytes <
          static_cast<int64_t>(getMinimumContiguousBytes()))
    return emitOpError(
        "host tensor descriptor must declare a legal provider dummy block shape");
  if (getBaseLayout() != "strided" || getPadding() != "zero" ||
      getAlignment() != 16 || getMinimumContiguousBytes() != 16 ||
      (getPipelineBlockAlignment() != 1 && getPipelineBlockAlignment() != 128) ||
      !getRequirePositiveShape() || !getRequirePositiveStrides() ||
      !getRequirePowerOfTwoBlockShape() ||
      getMaximumShapeExtent() != std::numeric_limits<int32_t>::max() ||
      getMaximumBlockElements() <= 0 ||
      getMaximumBlockElements() > (1 << 20))
    return emitOpError(
        "host tensor descriptor requires the complete native strided runtime contract");
  SmallVector<int64_t> alignedAxes;
  for (unsigned axis = 0; axis + 1 < rank; ++axis)
    alignedAxes.push_back(axis);
  if (getAlignedStrideAxes() != ArrayRef<int64_t>(alignedAxes))
    return emitOpError(
        "runtime contract must align every leading source stride");
  if (getUnitStrideAxes() !=
      ArrayRef<int64_t>{static_cast<int64_t>(base.getRank()) - 1})
    return emitOpError(
        "runtime contract must require the represented column stride to be one");
  if (!hasDescriptorLayout(kernel, base))
    return emitOpError(
        "host tensor descriptor base must carry a complete aligned stride ABI");
  unsigned bitWidth = base.getElementType().getIntOrFloatBitWidth();
  if (bitWidth < 8 || bitWidth % 8 != 0)
    return emitOpError(
        "tensor descriptor element type must occupy whole bytes");
  for (unsigned axis = 0; axis < rank; ++axis) {
    auto dimension = gpu::queryLaunchExpression(getShape()[axis]);
    if (!dimension || dimension != base.getLayout().getExtents()[axis] ||
        (axis + 1 < rank &&
         !matchesStrideBinding(kernel, base, axis, getStrides()[axis])))
      return emitOpError(
          "descriptor shape and strides must preserve each source axis");
    auto extent = gpu::queryLaunchExpression(getBlockShape()[axis]);
    if (!extent || !gpu::isCompileTimePhysicalExpr(extent))
      return emitOpError("descriptor block shape must use compile-time physical expressions");
    if (!blocked.test(axis) &&
        gpu::constantPhysicalExpression(extent) != 1)
      return emitOpError("scalar source axes must have descriptor block extent one");
  }
  auto unitStride = gpu::queryLaunchExpression(getStrides().back());
  if (!unitStride || gpu::constantPhysicalExpression(unitStride) != 1)
    return emitOpError("tensor descriptor final stride must be one");
  unsigned choices = 0;
  unsigned allocators = 0;
  bool declaredByChoice = false;
  kernel.walk([&](TensorDescriptorChoiceOp choice) {
    ++choices;
    declaredByChoice |= llvm::is_contained(choice.getDescriptors(), getResult());
  });
  kernel.walk([&](TensorDescriptorAllocatorOp) { ++allocators; });
  if (choices != 1 || allocators != 1 || !declaredByChoice)
    return emitOpError(
        "requires one explicit descriptor choice and one allocator declaration");
  return success();
}

LogicalResult DescriptorLoadOp::verify() {
  return verifyDescriptorAccess(*this, getDescriptor(), getResult().getType(),
                                getOffsets());
}

void DescriptorLoadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getDescriptorMutable());
}

LogicalResult DescriptorStoreOp::verify() {
  if (cast<gpu::ViewType>(getDescriptor().getType()).getAccess() == 0)
    return emitOpError("In-only external view cannot be written");
  return verifyDescriptorAccess(*this, getDescriptor(), getValue().getType(),
                                getOffsets());
}

void DescriptorStoreOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDescriptorMutable());
}

Value TensorDescriptorOp::getViewSource() { return getBase(); }

LogicalResult SplitOp::verify() {
  auto source = cast<gpu::FragmentType>(getSource().getType());
  auto low = cast<gpu::FragmentType>(getLow().getType());
  auto high = cast<gpu::FragmentType>(getHigh().getType());
  if (low != high || source.getShape().size() != low.getShape().size() + 1 ||
      source.getElementType() != low.getElementType() ||
      source.getValidity() != low.getValidity() ||
      source.getOwner() != low.getOwner())
    return emitOpError(
        "requires two equal prefix fragments from one trailing pair axis");
  auto trailing = dyn_cast<gpu::PhysicalExprAttr>(
      source.getShape()[source.getShape().size() - 1]);
  if (!trailing ||
      trailing.getKind() !=
          gpu::PhysicalExprKind::Constant ||
      trailing.getValue() != 2)
    return emitOpError("requires a constant trailing extent of two");
  if (!std::equal(low.getShape().begin(), low.getShape().end(),
                  source.getShape().begin()) ||
      !std::equal(low.getAxisMaps().begin(), low.getAxisMaps().end(),
                  source.getAxisMaps().begin()))
    return emitOpError(
        "results must preserve the source prefix shape and axis relations");
  return success();
}

LogicalResult MapElementwiseOp::verify() {
  auto result = getResult().getType();
  Block &body = getBody().front();
  auto yield = dyn_cast<gpu::YieldOp>(body.getTerminator());
  if (getInputs().empty() || body.getNumArguments() != getInputs().size() ||
      !yield || yield.getValues().size() != 1 ||
      yield.getValues().front().getType() != result.getElementType())
    return emitOpError("requires explicit scalar arguments and one element-typed yield");
  bool hasFragment = false;
  for (auto [input, argument] : llvm::zip(getInputs(), body.getArguments())) {
    Type element = input.getType();
    if (auto fragment = dyn_cast<gpu::FragmentType>(element)) {
      hasFragment = true;
      if (fragment.getShape() != result.getShape() ||
          fragment.getAxisMaps() != result.getAxisMaps() ||
          fragment.getValidity() != result.getValidity() ||
          fragment.getOwner() != result.getOwner())
        return emitOpError("inputs must preserve the result elementwise relation");
      element = fragment.getElementType();
    }
    if (!isa<IntegerType, IndexType, FloatType>(element) ||
        argument.getType() != element)
      return emitOpError("body arguments must be numeric scalar input elements");
  }
  if (!hasFragment)
    return emitOpError("requires at least one fragment input");
  WalkResult valid = getBody().walk([&](Operation *operation) {
    if (!isMemoryEffectFree(operation) ||
        !llvm::all_of(operation->getResultTypes(), [](Type type) {
          return isa<IntegerType, IndexType, FloatType>(type);
        }))
      return WalkResult::interrupt();
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        if (!llvm::all_of(block.getArgumentTypes(), [](Type type) {
              return isa<IntegerType, IndexType, FloatType>(type);
            }))
          return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (valid.wasInterrupted())
    return emitOpError("body must contain only pure scalar computations");
  return success();
}

} // namespace intent::triton

#define GET_OP_CLASSES
#include "Intent/Target/Triton/IR/TritonOps.cpp.inc"
