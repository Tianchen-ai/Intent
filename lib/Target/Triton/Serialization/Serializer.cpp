#include "Intent/Target/Triton/Serialization/Serializer.h"
#include "Intent/Target/Triton/Serialization/Numerical.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/ProgramInterface.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Serialization/Interface.h"
#include "Intent/Dialect/GPU/Serialization/PythonEmitter.h"
#include "Intent/Target/Triton/IR/Configuration.h"
#include "Intent/Target/Triton/IR/Program.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "Intent/Target/Triton/Analysis/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"

#include <set>
#include <optional>

using namespace mlir;

namespace intent::triton {
namespace {


class Serializer : public gpu::PythonEmitter {
public:
  Serializer(func::FuncOp kernel, raw_ostream &output)
      : gpu::PythonEmitter(kernel, output,
            {"tl.", "int1", "float64", "float8e4nv", "float8e5"},
            {integerDivision, "min", "max", "", "triton.next_power_of_2", true},
            ": tl.constexpr") {}

  LogicalResult emit(std::string &metadata) {
    bindArguments();
    if (failed) return failure();
    if (mlir::failed(emitPreamble())) return failure();
    emitHelpers();
    emitKernel();
    return failed ? failure() : emitMetadata(metadata);
  }

  static const Emitters &sourceOperations() {
    static const Emitters emitters = [] {
      Emitters result;
      gpu::PythonEmitter::addCommonOperations(result);
      addNumericalOperations(result);
      addNative<CtaBarrierOp>(result);
      addBinding<TensorDescriptorChoiceOp>(result);
      addBinding<TensorDescriptorAllocatorOp>(result);
      addBinding<TensorDescriptorOp>(result);
      addNative<MapElementwiseOp>(result);
      addNative<SplitOp>(result);
      addNative<DescriptorLoadOp>(result);
      addNative<gpu::LoadOp>(result);
      addNative<gpu::GatherOp>(result);
      addNative<gpu::ContractOp>(result);
      addNative<gpu::ScaledContractOp>(result);
      addNative<ReduceOp>(result);
      addNative<ScanOp>(result);
      addNative<gpu::HistogramOp>(result);
      addNative<gpu::RandomBitsOp>(result);
      addNative<gpu::AtomicStoreOp>(result);
      addNative<gpu::AtomicRMWOp>(result);
      addNative<gpu::AtomicCompareExchangeOp>(result);
      addNative<DescriptorStoreOp>(result);
      addNative<gpu::StoreOp>(result);
      return result;
    }();
    return emitters;
  }

  const Emitters &operationEmitters() const override { return sourceOperations(); }

private:
  template <typename Op> static void addNative(Emitters &result) {
    result.add<Op>([](Op) { return success(); }, [](Op op, gpu::PythonEmitter &emitter) {
      static_cast<Serializer &>(emitter).emitTyped(op);
      return failure(emitter.hasFailed());
    });
  }
  template <typename Op> static void addBinding(Emitters &result) {
    result.add<Op>([](Op) { return success(); },
                   [](Op, gpu::PythonEmitter &) { return success(); });
  }

  void emitTyped(ReduceOp op) { emitCollective(*op.getOperation()); }
  void emitTyped(ScanOp op) { emitCollective(*op.getOperation()); }

  void emitConstant(arith::ConstantOp constant) override {
    if (mlir::failed(emitNumericalConstant(constant, *this)))
      failed = true;
  }

  std::string programId(gpu::ProgramIdOp op) override {
    return "tl.program_id(" + std::to_string(op.getAxis()) + ").to(" +
        pythonType(op.getResult().getType()) + ")";
  }
  std::string makeRange(gpu::MakeRangeOp op) override {
    auto type = cast<gpu::FragmentType>(op.getResult().getType());
    return "(" + valueString(op.getStart()) + " + tl.arange(0, " +
        expressionString(cast<gpu::PhysicalExprAttr>(type.getShape()[0])) +
        ").to(" + pythonType(type.getElementType()) + ") * " +
        valueString(op.getStep()) + ")";
  }
  std::string splat(gpu::SplatOp op) override {
    if (emittingHelper) return valueString(op.getValue());
    return "tl.full(" + fragmentShape(op.getResult().getType()) + ", " + valueString(op.getValue()) +
        ", " + pythonType(op.getResult().getType().getElementType()) + ")";
  }
  std::string castValue(Value value, Type type, bool bitcast) override {
    auto expression = numericalCast(kernel, value.getType(), type,
                                    valueString(value), bitcast);
    if (mlir::failed(expression)) {
      failed = true;
      return {};
    }
    return std::move(*expression);
  }
  std::string reshape(gpu::ReshapeOp op) override {
    auto source = cast<gpu::FragmentType>(op.getValue().getType());
    auto target = cast<gpu::FragmentType>(op.getResult().getType());
    return (source.getShape().empty() ? "tl.broadcast_to(" : "tl.reshape(") +
        valueString(op.getValue()) + ", " + fragmentShape(target) +
        (source.getShape().empty() ? ")" : ", can_reorder=False)");
  }
  std::string join(gpu::JoinOp op) override {
    return "tl.join(" + valueString(op.getLhs()) + ", " + valueString(op.getRhs()) + ")";
  }
  std::string select(gpu::SelectOp op) override {
    return "tl.where(" + valueString(op.getCondition()) + ", " + controlValueString(op.getTrueValue()) +
        ", " + controlValueString(op.getFalseValue()) + ")";
  }
  std::string permute(Value value, ArrayRef<int64_t> permutation) override {
    return "tl.permute(" + valueString(value) + ", " + axisTuple(permutation) + ")";
  }
  std::string forRange(scf::ForOp loop) override {
    auto unroll = loop->getAttrOfType<IntegerAttr>(loopUnrollFactorAttr);
    Attribute stages = loop->getAttr(loopStagesAttr);
    std::string result = unroll || stages ? "tl.range(" : "range(";
    result += valueString(loop.getLowerBound()) + ", " + valueString(loop.getUpperBound()) + ", " + valueString(loop.getStep());
    if (unroll) result += ", loop_unroll_factor=" + std::to_string(unroll.getInt());
    if (stages) {
      result += ", num_stages=";
      if (auto constant = dyn_cast<IntegerAttr>(stages))
        result += std::to_string(constant.getInt());
      else
        result += cast<gpu::ParameterRefAttr>(stages).getName().getValue().str();
    }
    return result + ")";
  }

  using ViewABI = gpu::PythonArgument;
  using ScalarABI = gpu::PythonArgument;
  using MetadataABI = gpu::PythonArgument;
  struct DescriptorABI {
    TensorDescriptorOp operation;
    std::string name;
  };

  void bindArguments() {
    auto configurationSchema = ConfigurationSchema::read(kernel);
    if (mlir::failed(configurationSchema)) {
      failed = true;
      return;
    }
    configuration = std::move(*configurationSchema);
    auto interface = gpu::PythonSignature::read(kernel);
    if (mlir::failed(interface)) {
      failed = true;
      return;
    }
    views = std::move(interface->views);
    scalars = std::move(interface->scalars);
    metadataArguments = std::move(interface->metadata);
    for (const ScalarABI &scalar : scalars)
      values[scalar.value] = scalar.name;
    for (const MetadataABI &metadata : metadataArguments)
      values[metadata.value] = metadata.name;
    for (const ViewABI &view : views)
      values[view.value] = view.name;
    llvm::StringSet<> argumentNames;
    for (const auto &entry : values) argumentNames.insert(entry.second);
    for (const MetadataABI &metadata : metadataArguments)
      constexprValues.insert(metadata.value);
    for (Attribute attribute : gpu::getParameterDeclarations(kernel)) {
      auto schema = cast<gpu::ParameterAttr>(attribute);
      std::string name = schema.getName().getValue().str();
      argumentNames.insert(name);
      if (schema.isDeferred())
        coverageNames.insert(name);
    }
    kernel.walk([&](TensorDescriptorChoiceOp choice) {
      descriptorChoice = choice;
      values[choice.getResult()] = choice.getConfigParameter().getName().getValue().str();
      argumentNames.insert(choice.getConfigParameter().getName().getValue());
      argumentNames.insert(choice.getEligibilityArgument());
    });
    kernel.walk([&](TensorDescriptorAllocatorOp allocator) {
      descriptorAllocator = allocator;
    });
    while (!argumentNames.insert(metadataArgument).second)
      metadataArgument += "_";
    kernel.walk([&](TensorDescriptorOp descriptor) {
      std::string name = "_intent_descriptor_" + std::to_string(descriptors.size());
      while (!argumentNames.insert(name).second)
        name += "_";
      values[descriptor.getResult()] = name;
      descriptors.push_back({descriptor, name});
    });
    while (!argumentNames.insert(overlapArgument).second)
      overlapArgument += "_";
    kernel.walk([&](gpu::ViewOverlapOp overlap) {
      values[overlap.getResult()] = overlapArgument + "[" +
                                   std::to_string(overlapFacts.size()) + "]";
      overlapFacts.push_back(overlap);
      constexprValues.insert(overlap.getResult());
    });
    for (const auto &name : argumentNames)
      reserveName(name.getKey());
    if (descriptorChoice && !descriptorAllocator) {
      kernel.emitError(
          "tensor descriptor form has no declared allocator requirement");
      failed = true;
    }
  }

  LogicalResult emitPreamble() {
    auto dependencies = sourceOperations().collectDependencies(kernel);
    if (mlir::failed(dependencies)) return failure();
    output << "import triton\nimport triton.language as tl\n";
    for (const std::string &dependency : *dependencies)
      output << dependency << "\n";
    output << "from intent.runtime.triton.math import contract_fma\n\n";
    return success();
  }

  void emitHelpers() {
    auto emit = [&](Operation *owner, Region &region, StringRef role) {
      std::string name = ("_intent_" + role + "_" + Twine(helperCounter++)).str();
      emitHelper(owner, region, name, "@triton.jit", "a");
    };
    kernel.walk([&](Operation *operation) {
      if (auto reduce = dyn_cast<ReduceOp>(operation)) emit(operation, reduce.getCombine(), "reduce");
      else if (auto scan = dyn_cast<ScanOp>(operation)) emit(operation, scan.getCombine(), "scan");
      else if (auto map = dyn_cast<MapElementwiseOp>(operation)) emit(operation, map.getBody(), "map");
    });
  }

  void emitKernel() {
    output << "@triton.jit\ndef _intent_kernel(";
    bool first = true;
    for (const ViewABI &view : views) {
      if (!first)
        output << ", ";
      first = false;
      output << view.name;
    }
    for (const ScalarABI &scalar : scalars) {
      if (!first)
        output << ", ";
      first = false;
      output << scalar.name;
      output << ": " << pythonType(scalar.value.getType());
    }
    if (!metadataArguments.empty()) {
      if (!first)
        output << ", ";
      first = false;
      output << metadataArgument << ": tl.constexpr";
    }
    if (!overlapFacts.empty()) {
      if (!first)
        output << ", ";
      first = false;
      output << overlapArgument << ": tl.constexpr";
    }
    if (descriptorChoice) {
      if (!first)
        output << ", ";
      first = false;
      output << descriptorChoice.getEligibilityArgument() << ": tl.constexpr";
    }
    for (StringAttr parameter : configuration.kernelParameters)
      output << ", " << parameter.getValue() << ": tl.constexpr";
    for (const DescriptorABI &descriptor : descriptors)
      output << ", " << descriptor.name;
    output << "):\n";
    indent = 1;
    for (auto [index, metadata] : llvm::enumerate(metadataArguments))
      line(metadata.name + ": tl.constexpr = " + metadataArgument + "[" +
           std::to_string(index) + "]");
    line(configuration.warps.getValue().str() +
         ": tl.constexpr = tl.extra.cuda.num_warps()");
    emitBlock(kernel.getBody().front());
    output << "\n";
  }

  void emitTyped(CtaBarrierOp) {
      line("tl.debug_barrier()");
      return;
      }

  void emitTyped(MapElementwiseOp map) {
      bool scalar = map.getResult().getType().getShape().empty();
      std::string call = scalar ? helperName(map.getOperation()) + "("
                                : "tl.map_elementwise(" + helperName(map.getOperation());
      for (auto [index, input] : llvm::enumerate(map.getInputs())) {
        if (!scalar || index)
          call += ", ";
        call += controlValueString(input);
      }
      assign(map.getResult(), call + ")");
      return;
      }

  void emitTyped(SplitOp split) {
      assignResults(split.getResults(),
                    "tl.split(" + valueString(split.getSource()) + ")");
      return;
      }

  void emitTyped(DescriptorLoadOp load) {
      assign(load.getResult(),
             valueString(load.getDescriptor()) + ".load(" +
                 tuple(load.getOffsets()) + ")");
      return;
      }

  void emitTyped(gpu::LoadOp load) {
      auto access = cast<gpu::AccessOpInterface>(load.getOperation());
      std::string call = "tl.load(" + pointer(access);
      if (access.getAccessValidity())
        call += ", mask=" + valueString(access.getAccessValidity()) +
                ", other=" + valueString(access.getAccessFill());
      assign(access.getAccessResult(), call + ")");
      return;
      }

  void emitTyped(gpu::GatherOp gather) {
      auto access = cast<gpu::AccessOpInterface>(gather.getOperation());
      Value coordinate = access.getAccessCoordinates().front();
      std::string indices = valueString(coordinate);
      bool scalar = !isa<gpu::FragmentType>(access.getAccessValueType());
      if (scalar)
        indices = "tl.full((1,), " + indices + ", " +
                  pythonType(coordinate.getType()) + ")";
      std::string call = "tl.gather(" + valueString(access.getAccessResource()) + ", " +
                         indices +
                         ", axis=" +
                         std::to_string(access.getAccessSourceAxes().front()) + ")";
      if (scalar)
        call = "tl.reshape(" + call + ", ())";
      assign(access.getAccessResult(), call);
      return;
      }

  void emitTyped(gpu::ContractOp contract) {
      auto form = contract->getAttrOfType<StringAttr>(contractFormAttr);
      if (form && form.getValue() == "fma") {
        assign(contract.getResult(),
               "contract_fma(" + valueString(contract.getLhs()) + ", " +
                   valueString(contract.getRhs()) + ", " +
                   valueString(contract.getAccumulator()) + ")");
        return;
      }
      if (form && form.getValue() == "multiply_sum") {
        unsigned rank = contract.getLhs().getType().getShape().size();
        std::string element = pythonType(elementType(contract.getResult().getType()));
        std::string lhs = "tl.expand_dims(" + valueString(contract.getLhs()) +
                          ", axis=" + std::to_string(rank) + ").to(" +
                          element + ")";
        std::string rhs = "tl.expand_dims(" + valueString(contract.getRhs()) +
                          ", axis=" + std::to_string(rank - 2) + ").to(" +
                          element + ")";
        std::string reduced =
            "tl.sum((" + lhs + " * " + rhs + "), axis=" +
            std::to_string(rank - 1) + ")";
        assign(contract.getResult(), "(" + reduced + " + " +
                                         valueString(contract.getAccumulator()) +
                                         ")");
        return;
      }
      assign(contract.getResult(), "tl.dot(" + valueString(contract.getLhs()) +
                                      ", " + valueString(contract.getRhs()) +
                                      ", " + valueString(contract.getAccumulator()) +
                                      ", input_precision=\"ieee\", max_num_imprecise_acc=0, out_dtype=" +
                                      pythonType(contract.getResult()
                                                     .getType()
                                                     .getElementType()) + ")");
      return;
      }

  void emitTyped(gpu::ScaledContractOp contract) {
      auto format = [](ScaledFormat value) -> StringRef {
        switch (value) {
        case ScaledFormat::E2M1: return "e2m1";
        case ScaledFormat::E4M3: return "e4m3";
        case ScaledFormat::E8M0: return "e8m0";
        }
        llvm_unreachable("unhandled Intent scaled format");
      };
      auto extent = [&](gpu::FragmentType type, unsigned axis) {
        return expressionString(
            cast<gpu::PhysicalExprAttr>(type.getShape()[axis]));
      };
      auto lhs = contract.getLhs().getType();
      auto rhs = contract.getRhs().getType();
      // The verified scaled-contract schema already fixes contiguous groups
      // and packed carriers. dot_scaled spells that grouped K as one axis.
      std::string lhsValue =
          "tl.reshape(" + valueString(contract.getLhs()) + ", (" +
          extent(lhs, 0) + ", " + extent(lhs, 1) + " * " + extent(lhs, 2) + "))";
      std::string rhsValue =
          "tl.reshape(" + valueString(contract.getRhs()) + ", (" +
          extent(rhs, 0) + " * " + extent(rhs, 1) + ", " + extent(rhs, 2) + "))";
      assign(contract.getResult(),
             "tl.dot_scaled(" + lhsValue + ", " +
                 valueString(contract.getLhsScale()) + ", \"" +
                 format(contract.getLhsFormat()).str() + "\", " +
                 rhsValue + ", " +
                 valueString(contract.getRhsScale()) + ", \"" +
                 format(contract.getRhsFormat()).str() + "\", " +
                 valueString(contract.getAccumulator()) + ")");
      return;
      }

  void emitCollective(Operation &operation) {
      auto reduce = dyn_cast<ReduceOp>(operation);
      auto scan = dyn_cast<ScanOp>(operation);
      ValueRange sourceValues = reduce ? reduce.getSources() : scan.getSources();
      unsigned count = sourceValues.size();
      int64_t axis = reduce ? reduce.getAxisAttr().getInt() : scan.getAxisAttr().getInt();
      std::string sources = count == 1 ? valueString(sourceValues.front()) : "(";
      if (count != 1) {
        for (unsigned i = 0; i < count; ++i) {
          if (i) sources += ", ";
          sources += valueString(sourceValues[i]);
        }
        sources += ")";
      }
      std::string call = (reduce ? "tl.reduce(" : "tl.associative_scan(") + sources +
          ", axis=" + (reduce && axis == -1 ? "None" : std::to_string(axis)) +
          ", combine_fn=" + helperName(&operation);
      if (scan) call += std::string(", reverse=") + (scan.getReverse() ? "True" : "False");
      assignResults(operation.getResults(), call + ")");
      return;
      }

  void emitTyped(gpu::HistogramOp histogram) {
      std::string counts = "tl.histogram(" +
                           valueString(histogram.getValues()) + ", " +
                           valueString(histogram.getBins()) + ", mask=" +
                           valueString(histogram.getValid()) + ")";
      Type resultElement = histogram.getResult().getType().getElementType();
      assign(histogram.getResult(),
             "tl.cast(" + counts + ", " + pythonType(resultElement) + ")");
      return;
      }

  void emitTyped(gpu::RandomBitsOp random) {
      assign(random.getResult(), "tl.randint(" + valueString(random.getSeed()) +
                                      ", " + valueString(random.getCounter()) +
                                      ").to(tl.uint32, bitcast=True)");
      return;
      }

  void emitTyped(gpu::AtomicStoreOp atomic) {
      auto access = cast<gpu::AccessOpInterface>(atomic.getOperation());
      std::string call = "tl.atomic_xchg(" +
                         pointer(access) + ", " + valueString(access.getAccessPayloads().front());
      if (access.getAccessValidity())
        call += ", mask=" + valueString(access.getAccessValidity());
      call += ", sem=\"" + atomicSemantics(atomic.getOrdering()) +
              "\", scope=\"" + atomicScope(atomic.getSharing()) + "\")";
      line(call);
      return;
      }

  void emitTyped(gpu::AtomicRMWOp atomic) {
      auto access = cast<gpu::AccessOpInterface>(atomic.getOperation());
      auto operationName = [](AtomicRMWKind kind) -> StringRef {
        switch (kind) {
        case AtomicRMWKind::Exchange: return "xchg";
        case AtomicRMWKind::Add: return "add";
        case AtomicRMWKind::Maximum: return "max";
        case AtomicRMWKind::Minimum: return "min";
        case AtomicRMWKind::BitwiseAnd: return "and";
        case AtomicRMWKind::BitwiseOr: return "or";
        case AtomicRMWKind::BitwiseXor: return "xor";
        }
        llvm_unreachable("unhandled Intent atomic RMW kind");
      };
      std::string call = "tl.atomic_" + operationName(atomic.getKind()).str() +
                         "(" +
                         pointer(access) + ", " + valueString(access.getAccessPayloads().front());
      if (access.getAccessValidity())
        call += ", mask=" + valueString(access.getAccessValidity());
      call += ", sem=\"" + atomicSemantics(atomic.getOrdering()) +
              "\", scope=\"" + atomicScope(atomic.getSharing()) + "\")";
      assign(atomic.getResult(), call);
      return;
      }

  void emitTyped(gpu::AtomicCompareExchangeOp atomic) {
      auto access = cast<gpu::AccessOpInterface>(atomic.getOperation());
      auto payloads = access.getAccessPayloads();
      std::string old = newName();
      line(old + " = tl.atomic_cas(" +
           pointer(access) + ", " + valueString(payloads[0]) + ", " +
           valueString(payloads[1]) + ", sem=\"" +
           atomicSemantics(atomic.getOrdering()) + "\", scope=\"" +
           atomicScope(atomic.getSharing()) + "\")");
      assign(atomic.getResult(), "(" + old + ", (" + old + " == " +
                                     valueString(payloads[0]) + "))");
      return;
      }

  void emitTyped(DescriptorStoreOp store) {
      auto fragment = cast<gpu::FragmentType>(store.getValue().getType());
      line(valueString(store.getDescriptor()) + ".store(" +
           tuple(store.getOffsets()) +
           ", tl.cast(" + valueString(store.getValue()) + ", " +
           pythonType(fragment.getElementType()) + "))");
      return;
      }

  void emitTyped(gpu::StoreOp store) {
      auto access = cast<gpu::AccessOpInterface>(store.getOperation());
      std::string call = "tl.store(" +
                         pointer(access) + ", " + valueString(access.getAccessPayloads().front());
      if (access.getAccessValidity())
        call += ", mask=" + valueString(access.getAccessValidity());
      line(call + ")");
      return;
      }

  std::string atomicSemantics(AtomicOrdering ordering) const {
    switch (ordering) {
    case AtomicOrdering::Relaxed: return "relaxed";
    case AtomicOrdering::Acquire: return "acquire";
    case AtomicOrdering::Release: return "release";
    case AtomicOrdering::AcquireRelease: return "acq_rel";
    }
    llvm_unreachable("unhandled Intent atomic ordering");
  }

  std::string atomicScope(gpu::AtomicSharingDomain sharing) const {
    switch (sharing) {
    case gpu::AtomicSharingDomain::ProgramInstance: return "cta";
    case gpu::AtomicSharingDomain::KernelInvocation: return "gpu";
    }
    llvm_unreachable("unhandled atomic sharing domain");
  }

  std::string projectedValue(Value value, gpu::FragmentType target,
                             const gpu::BroadcastProjection &projection) {
    auto source = dyn_cast<gpu::FragmentType>(value.getType());
    if (!source)
      return "tl.full(" + fragmentShape(target) + ", " + valueString(value) +
             ", " + pythonType(elementType(value.getType())) + ")";
    if (!projection.isExact()) {
      kernel.emitError("Triton broadcast lost its shared axis projection")
          << "; source=" << source << "; target=" << target
          << "; value=" << value;
      failed = true;
      return {};
    }
    SmallVector<int64_t> order;
    for (std::optional<unsigned> sourceAxis : projection.targetToSource)
      if (sourceAxis)
        order.push_back(*sourceAxis);
    bool identity = llvm::all_of(llvm::enumerate(order), [](auto item) {
      return item.index() == static_cast<unsigned>(item.value());
    });
    std::string input = identity ? valueString(value) : permute(value, order);
    if (source == target && identity)
      return input;
    if (source.getShape().size() == target.getShape().size())
      return "tl.broadcast_to(" + input + ", " +
             fragmentShape(target) + ")";
    SmallVector<std::string> selectors;
    for (std::optional<unsigned> sourceIndex : projection.targetToSource)
      selectors.push_back(sourceIndex ? ":" : "None");
    std::string result = input + "[";
    for (auto [index, selector] : llvm::enumerate(selectors)) {
      if (index)
        result += ", ";
      result += selector;
    }
    return "tl.broadcast_to(" + result + "], " + fragmentShape(target) + ")";
  }

  std::string broadcastValue(Value value, gpu::FragmentType target) override {
    auto source = dyn_cast<gpu::FragmentType>(value.getType());
    return projectedValue(value, target, source
        ? gpu::queryBroadcastProjection(source, target) : gpu::BroadcastProjection());
  }

  std::string pointer(gpu::AccessOpInterface access) {
    Value resource = access.getAccessResource();
    auto coordinates = access.getAccessCoordinates();
    auto sourceAxes = access.getAccessSourceAxes();
    Type valueType = access.getAccessValueType();
    auto argument = dyn_cast<BlockArgument>(resource);
    if (!argument) {
      failed = true;
      return {};
    }
    auto view = cast<gpu::ViewType>(resource.getType());
    auto strides = view.getLayout().getStrides();
    auto fragment = dyn_cast<gpu::FragmentType>(valueType);
    std::string result = values.lookup(resource);
    for (auto [axis, coordinate] : llvm::enumerate(coordinates)) {
      std::string stride = expressionString(
          cast<gpu::PhysicalExprAttr>(strides[sourceAxes[axis]]));
      std::string coordinateExpression = fragment
          ? projectedValue(coordinate, fragment,
                           gpu::queryAccessCoordinateProjection(access, axis))
          : valueString(coordinate);
      result += " + (" + coordinateExpression + ") * " + stride;
    }
    return "(" + result + ")";
  }

  std::string controlValueString(Value value) {
    std::string result = valueString(value);
    if (isa<IntegerType, IndexType, FloatType>(value.getType())) {
      if (value.getDefiningOp<arith::ConstantOp>())
        return "tl.full((), " + result + ", " + pythonType(value.getType()) + ")";
      return "tl.cast(" + result + ", " + pythonType(value.getType()) + ")";
    }
    return result;
  }

  FailureOr<llvm::json::Value> descriptorHostValue(Value value) {
    if (auto expression = gpu::queryLaunchExpression(value))
      return gpu::serializeExpression(expression);
    return kernel.emitError("Triton descriptor has no host-evaluable value binding");
  }

  LogicalResult emitMetadata(std::string &metadata) {
    auto artifact = gpu::serializeInterface(
        kernel, "triton", [&](Value value) { return valueString(value); });
    if (mlir::failed(artifact)) return failure();
    llvm::json::Object details{{"kernel", "_intent_kernel"}};
    llvm::json::Array kernelParameters;
    for (StringAttr parameter : configuration.kernelParameters)
      kernelParameters.push_back(parameter.getValue());
    details["kernel_parameters"] = std::move(kernelParameters);
    details["native_options"] = llvm::json::Object{
        {"num_warps", configuration.warps.getValue()},
        {"num_stages", configuration.stages.getValue()},
        {"num_ctas", configuration.ctas.getValue()}};
    llvm::json::Array arguments, key;
    for (const ViewABI &view : views) arguments.push_back(view.name);
    for (const ScalarABI &scalar : scalars) arguments.push_back(scalar.name);
    details["metadata_argument"] = nullptr;
    if (!metadataArguments.empty()) {
      details["metadata_argument"] = metadataArgument;
      arguments.push_back(metadataArgument);
      key.push_back(metadataArgument);
    }
    details["overlap_argument"] = nullptr;
    if (!overlapFacts.empty()) {
      details["overlap_argument"] = overlapArgument;
      arguments.push_back(overlapArgument);
      key.push_back(overlapArgument);
    }
    details["descriptor_choice"] = nullptr;
    if (descriptorChoice) {
      details["descriptor_choice"] = llvm::json::Object{
          {"config", descriptorChoice.getConfigParameter().getName().getValue()},
          {"eligibility", descriptorChoice.getEligibilityArgument()}};
      arguments.push_back(descriptorChoice.getEligibilityArgument());
      key.push_back(descriptorChoice.getEligibilityArgument());
    }
    for (const std::string &name : coverageNames) key.push_back(name);
    llvm::json::Array encodedDescriptors;
    for (const DescriptorABI &descriptor : descriptors) {
      TensorDescriptorOp operation = descriptor.operation;
      llvm::json::Object encoded{
          {"name", descriptor.name},
          {"base", gpu::getArgumentReference(operation.getBase()).getId()},
          {"rank", cast<gpu::ViewType>(operation.getBase().getType()).getRank()},
          {"require_positive_shape", operation.getRequirePositiveShape()},
          {"require_positive_strides", operation.getRequirePositiveStrides()},
          {"alignment", operation.getAlignment()},
          {"maximum_shape_extent", operation.getMaximumShapeExtent()},
          {"padding", operation.getPadding()}};
      auto encodeValues = [&](StringRef field, ValueRange values) -> LogicalResult {
        llvm::json::Array expressions;
        for (Value value : values) {
          auto expression = descriptorHostValue(value);
          if (mlir::failed(expression)) return failure();
          expressions.push_back(std::move(*expression));
        }
        encoded[field] = std::move(expressions);
        return success();
      };
      if (mlir::failed(encodeValues("shape", operation.getShape())) ||
          mlir::failed(encodeValues("strides", operation.getStrides())) ||
          mlir::failed(encodeValues("block_shape", operation.getBlockShape())))
        return failure();
      llvm::json::Array aligned, unit;
      for (int64_t axis : operation.getAlignedStrideAxes()) aligned.push_back(axis);
      for (int64_t axis : operation.getUnitStrideAxes()) unit.push_back(axis);
      encoded["aligned_stride_axes"] = std::move(aligned);
      encoded["unit_stride_axes"] = std::move(unit);
      encodedDescriptors.push_back(std::move(encoded));
    }
    details["descriptors"] = std::move(encodedDescriptors);
    details["allocator"] = nullptr;
    if (descriptorAllocator)
      details["allocator"] = llvm::json::Object{
          {"size_argument", descriptorAllocator.getSizeArgument()},
          {"alignment_argument", descriptorAllocator.getAlignmentArgument()},
          {"stream_argument", descriptorAllocator.getStreamArgument()},
          {"lifetime", descriptorAllocator.getLifetime()},
          {"implementation", descriptorAllocator.getImplementation()}};
    details["kernel_arguments"] = std::move(arguments);
    details["autotune_key"] = std::move(key);
    (*artifact)["triton"] = std::move(details);
    llvm::raw_string_ostream(metadata) << llvm::json::Value(std::move(*artifact));
    return failed ? failure() : success();
  }

  SmallVector<ViewABI> views;
  SmallVector<ScalarABI> scalars;
  SmallVector<MetadataABI> metadataArguments;
  std::string metadataArgument = "_intent_metadata";
  SmallVector<gpu::ViewOverlapOp> overlapFacts;
  std::string overlapArgument = "_intent_overlaps";
  std::set<std::string> coverageNames;
  ConfigurationSchema configuration;
  TensorDescriptorChoiceOp descriptorChoice;
  TensorDescriptorAllocatorOp descriptorAllocator;
  SmallVector<DescriptorABI> descriptors;
  unsigned helperCounter = 0;
};

} // namespace

LogicalResult verifySourceOperation(Operation *operation) {
  return Serializer::sourceOperations().verify(operation);
}

LogicalResult serializeProgram(ModuleOp module, std::string &source,
                               std::string &metadata) {
  if (failed(verifyTritonProgram(module, verifySourceOperation)))
    return failure();
  auto kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  llvm::raw_string_ostream stream(source);
  Serializer serializer(*kernel, stream);
  LogicalResult result = serializer.emit(metadata);
  stream.flush();
  return result;
}

} // namespace intent::triton
