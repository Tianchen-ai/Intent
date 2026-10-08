#include "Intent/Target/CuTile/Serialization/Serializer.h"
#include "Intent/Target/CuTile/Serialization/Numerical.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/ProgramInterface.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Serialization/Interface.h"
#include "Intent/Dialect/GPU/Serialization/PythonEmitter.h"
#include "Intent/Target/CuTile/Analysis/Tuning.h"
#include "Intent/Target/CuTile/Analysis/Program.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <numeric>

using namespace mlir;

namespace intent::cutile {
namespace {


StringRef providerHint(gpu::ParameterAttr parameter) {
  auto role = parameter.getRole();
  if (role == gpu::ParameterRole::ProviderOccupancy)
    return "occupancy";
  if (role == gpu::ParameterRole::ProviderCTAs)
    return "num_ctas";
  if (role == gpu::ParameterRole::ProviderWarps)
    return "num_worker_warps";
  return {};
}

class Serializer : public gpu::PythonEmitter {
public:
  Serializer(func::FuncOp kernel, raw_ostream &output)
      : gpu::PythonEmitter(kernel, output,
            {"ct.", "bool_", "float64", "float8_e4m3fn", "float8_e5m2"},
            {integerDivision, "min", "max", "", "_intent_next_power_of_2", false},
            "") {}

  LogicalResult emit(std::string &metadata) {
    bindArguments();
    if (failed) return failure();
    if (mlir::failed(emitPreamble())) return failure();
    emitCollectiveHelpers();
    emitKernel();
    return failed ? failure() : emitMetadata(metadata);
  }

  static const Emitters &sourceOperations() {
    static const Emitters emitters = [] {
      Emitters result;
      gpu::PythonEmitter::addCommonOperations(result);
      addNumericalOperations(result);
      addBinding<ArrayViewOp>(result);
      addNative<TileLoadOp>(result);
      addNative<ScalarLoadOp>(result);
      addNative<GatherLoadOp>(result);
      addNative<MMAOp>(result, [](MMAOp operation) -> FailureOr<Emitters::Dependencies> {
        if (operation.getReductionChunk())
          return Emitters::Dependencies{libraryMathImport.str()};
        return Emitters::Dependencies{};
      });
      addNative<ScaledMMAOp>(result);
      addNative<ReduceOp>(result);
      addNative<ScanOp>(result);
      addNative<AtomicRMWOp>(result);
      addNative<ExtractOp>(result);
      addNative<TileAtomicAddOp>(result);
      addNative<TileStoreOp>(result);
      addNative<ScalarStoreOp>(result);
      addNative<ScatterStoreOp>(result);
      return result;
    }();
    return emitters;
  }

  const Emitters &operationEmitters() const override { return sourceOperations(); }

private:
  template <typename Op> static void addNative(
      Emitters &result,
      std::function<FailureOr<Emitters::Dependencies>(Op)> dependencies = {}) {
    result.add<Op>([](Op) { return success(); }, [](Op op, gpu::PythonEmitter &emitter) {
      static_cast<Serializer &>(emitter).emitTyped(op);
      return failure(emitter.hasFailed());
    }, std::move(dependencies));
  }
  template <typename Op> static void addBinding(Emitters &result) {
    result.add<Op>([](Op) { return success(); },
                   [](Op, gpu::PythonEmitter &) { return success(); });
  }

  void emitConstant(arith::ConstantOp constant) override {
      std::string value = literal(constant.getValue());
      if (constant.getType().isInteger(64))
        value = "ct.astype(" + value + ", ct.int64)";
      values[constant.getResult()] = value;
      }

  std::string programId(gpu::ProgramIdOp op) override {
    return "ct.astype(ct.bid(" + std::to_string(op.getAxis()) + "), " + pythonType(op.getResult().getType()) + ")";
  }
  std::string makeRange(gpu::MakeRangeOp op) override {
    return "(" + valueString(op.getStart()) + " + ct.arange(" + valueString(op.getExtent()) +
        ", dtype=" + pythonType(op.getResult().getType().getElementType()) + ") * " + valueString(op.getStep()) + ")";
  }
  std::string splat(gpu::SplatOp op) override {
    return "ct.full(" + fragmentShape(op.getResult().getType()) + ", " + valueString(op.getValue()) +
        ", dtype=" + pythonType(op.getResult().getType().getElementType()) + ")";
  }
  std::string castValue(Value value, Type type, bool bitcast) override {
    return numericalCast(*this, value, type, bitcast);
  }
  std::string reshape(gpu::ReshapeOp op) override {
    return "ct.reshape(" + valueString(op.getValue()) + ", " + fragmentShape(cast<gpu::FragmentType>(op.getResult().getType())) + ")";
  }
  std::string join(gpu::JoinOp op) override {
    SmallVector<std::string> extents;
    for (Attribute extent : op.getLhs().getType().getShape())
      extents.push_back(expressionString(cast<gpu::PhysicalExprAttr>(extent)));
    extents.push_back("1");
    std::string expanded = stringTuple(extents);
    return "ct.cat((ct.reshape(" + valueString(op.getLhs()) + ", " + expanded +
        "), ct.reshape(" + valueString(op.getRhs()) + ", " + expanded + ")), axis=" + std::to_string(op.getAxis()) + ")";
  }
  std::string select(gpu::SelectOp op) override {
    return numericalSelect(*this, op);
  }
  std::string permute(Value value, ArrayRef<int64_t> permutation) override {
    return "ct.permute(" + valueString(value) + ", " + axisTuple(permutation) + ")";
  }
  std::string loopInitialValue(Value value) override {
    if (value.getType().isIntOrIndex())
      return "ct.astype(" + valueString(value) + ", " + pythonType(value.getType()) + ")";
    return valueString(value);
  }
  std::string forRange(scf::ForOp loop) override {
    auto type = pythonType(loop.getInductionVar().getType());
    return "range(ct.astype(" + valueString(loop.getLowerBound()) + ", " + type + "), ct.astype(" +
        valueString(loop.getUpperBound()) + ", " + type + "), ct.astype(" + valueString(loop.getStep()) + ", " + type + "))";
  }
  bool inlineRecords() const override { return true; }

  using ViewABI = gpu::PythonArgument;
  using ScalarABI = gpu::PythonArgument;
  using MetadataABI = gpu::PythonArgument;
  struct ArrayViewABI {
    ArrayViewOp operation;
    unsigned sourceView;
    std::string name;
    std::string eligible;
  };

  void bindArguments() {
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
    AttrTypeWalker parameterUses;
    parameterUses.addWalk([&](gpu::ParameterRefAttr reference) { nativeParameters.insert(reference); });
    kernel.walk([&](Operation *operation) {
      if (operation == kernel.getOperation()) return;
      if (auto parameter = dyn_cast<gpu::ParameterOp>(operation);
          parameter && parameter.getResult().use_empty()) return;
      parameterUses.walk(operation->getAttrDictionary());
      for (Type type : operation->getResultTypes()) parameterUses.walk(type);
      for (Region &region : operation->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments()) parameterUses.walk(argument.getType());
    });
    for (Attribute attribute : gpu::getParameterDeclarations(kernel)) {
      auto parameter = cast<gpu::ParameterAttr>(attribute);
      if (StringRef hint = providerHint(parameter); !hint.empty()) {
        if (!providerHintParameters.emplace(
                hint.str(), parameter.getName().getValue().str()).second) {
          kernel.emitError("duplicates a cuTile compiler hint");
          failed = true;
          return;
        }
      }
    }
    llvm::StringSet<> occupied;
    for (const auto &entry : values)
      occupied.insert(entry.second);
    for (Attribute attribute : gpu::getParameterDeclarations(kernel))
      occupied.insert(cast<gpu::ParameterAttr>(attribute).getName().getValue());
    auto fresh = [&](StringRef stem) {
      unsigned suffix = 0;
      std::string name;
      do {
        name = (Twine(stem) + Twine(suffix++)).str();
      } while (!occupied.insert(name).second);
      return name;
    };
    kernel.walk([&](ArrayViewOp array) {
      unsigned argument = cast<BlockArgument>(array.getBase()).getArgNumber();
      for (auto [index, view] : llvm::enumerate(views)) {
        if (view.value.getArgNumber() != argument)
          continue;
        ArrayViewABI binding{
            array, static_cast<unsigned>(index), fresh("_intent_array_view_"),
            fresh("_intent_array_valid_")};
        values[array.getResult()] = binding.name;
        values[array.getEligible()] = binding.eligible;
        arrayViews.push_back(std::move(binding));
      }
    });
    kernel.walk([&](gpu::ViewOverlapOp overlap) {
      values[overlap.getResult()] = fresh("_intent_overlap_");
      overlapFacts.push_back(overlap);
    });
    for (const auto &name : occupied)
      reserveName(name.getKey());
  }

  LogicalResult emitPreamble() {
    auto dependencies = sourceOperations().collectDependencies(kernel);
    if (mlir::failed(dependencies)) return failure();
    for (const std::string &dependency : *dependencies)
      output << dependency << "\n";
    output << "from typing import Annotated\n"
              "import cuda.tile as ct\n"
              "from intent.runtime.cutile import array_index_kernels\n\n"
              "ConstInt = ct.Constant[int]\n\n"
              "@ct.function(host=True)\n"
              "def _intent_next_power_of_2(value):\n"
              "    value = max(value, 1) - 1\n"
              "    value = value | (value >> 1)\n"
              "    value = value | (value >> 2)\n"
              "    value = value | (value >> 4)\n"
              "    value = value | (value >> 8)\n"
              "    value = value | (value >> 16)\n"
              "    value = value | (value >> 32)\n"
              "    return value + 1\n\n";
    return success();
  }

  void emitCollectiveHelpers() {
      kernel.walk([&](Operation *operation) {
      if (isa<ReduceOp, ScanOp>(operation) && !operation->getRegion(0).empty())
        emitHelper(operation, operation->getRegion(0),
                   "_intent_combine_" + std::to_string(counter++), "@ct.function", "arg");
    });
  }

  void emitKernel() {
    if (!kernel->hasAttr(arrayIndexTileBoundsAttr))
      output << "@ct.kernel\n";
    output << "def _intent_kernel(";
    bool first = true;
    auto argument = [&](StringRef text) {
      if (!first)
        output << ", ";
      first = false;
      output << text;
    };
    auto arrayArgument = [&](StringRef name, unsigned rank) {
      // Shapes and strides specialize the metadata ABI and launch cache. Preserve
      // the same facts in the provider array type used for access lowering.
      std::string dimensions = "(";
      for (unsigned axis = 0; axis < rank; ++axis)
        dimensions += std::to_string(axis) + ", ";
      dimensions += ")";
      argument(name.str() +
               ": Annotated[ct.Array, ct.ArrayAnnotation(index_dtype=ct.int64, static_shape_dims=" +
               dimensions + ", static_stride_dims=" + dimensions + ")]");
    };
    for (const ViewABI &view : views)
      arrayArgument(view.name, view.viewType().getLayout().getExtents().size());
    for (ArrayViewABI view : arrayViews) {
      arrayArgument(view.name, view.operation.getResult().getType().getRank());
      argument(view.eligible + ": ct.Constant[bool]");
    }
    for (const ScalarABI &scalar : scalars) {
      auto integer = dyn_cast<IntegerType>(scalar.value.getType());
      bool wide = scalar.value.getType().isIndex() ||
                  (integer && integer.getWidth() == 64);
      argument(scalar.name + (wide ? ": ct.ScalarInt64" : ""));
    }
    for (const MetadataABI &metadata : metadataArguments)
      argument(metadata.name + ": ConstInt");
    for (gpu::ViewOverlapOp overlap : overlapFacts)
      argument(valueString(overlap.getResult()) + ": ct.Constant[bool]");
    for (Attribute attribute : gpu::getParameterDeclarations(kernel)) {
      auto parameter = cast<gpu::ParameterAttr>(attribute);
      if (!providerHint(parameter).empty() || !nativeParameters.contains(parameter.getReference()))
        continue;
      std::string name = parameter.getName().getValue().str();
      argument(name + ": ConstInt");
    }
    output << "):\n";
    indent = 1;
    emitBlock(kernel.getBody().front());
    output << "\n";
    if (kernel->hasAttr(arrayIndexTileBoundsAttr)) {
      output << "_intent_i32_kernel, _intent_kernel = array_index_kernels(_intent_kernel, (";
      for (const ViewABI &view : views) {
        llvm::json::OStream(output).value(view.name);
        output << ", ";
      }
      for (const ArrayViewABI &view : arrayViews) {
        llvm::json::OStream(output).value(view.name);
        output << ", ";
      }
      output << "))\n\n";
    }
  }

  void emitTyped(TileLoadOp load) {
      std::string padding = "ct.PaddingMode.ZERO";
      if (Value fullTiles = load.getFullTiles())
        padding = "(ct.PaddingMode.UNDETERMINED if " + valueString(fullTiles) +
                  " else ct.PaddingMode.ZERO)";
      std::string call = "ct.load(" + valueString(load.getResource()) +
                         ", index=" + tuple(load.getTileIndices()) +
                         ", shape=" + fragmentShape(load.getResult().getType()) +
                         ", padding_mode=" + padding + ", allow_tma=" +
                         valueString(load.getAllowTma());
      if (auto latency = load.getLatencyPolicy())
        call += ", latency=(None if " + valueString(latency) + " == " +
                std::to_string(inferredLoadPolicy) + " else " +
                valueString(latency) + ")";
      call += ")";
      assign(load.getResult(), call);
      }

  void emitTyped(ScalarLoadOp load) {
      std::string call = "ct.gather(" + valueString(load.getResource()) + ", " +
                         tuple(load.getIndices());
      if (load.getValid())
        call += ", mask=" + valueString(load.getValid()) +
                ", padding_value=" + valueString(load.getFill());
      call += load.getInBounds() ? ", check_bounds=False)"
                                 : ", check_bounds=True)";
      assign(load.getResult(), call);
      }

  void emitTyped(GatherLoadOp gather) {
      std::string call = "ct.gather(" + valueString(gather.getResource()) +
                         ", " + tuple(gather.getCoordinates());
      if (gather.getValid())
        call += ", mask=" + valueString(gather.getValid());
      if (gather.getFill())
        call += ", padding_value=" + valueString(gather.getFill());
      if (auto latency = gather.getLatencyPolicy())
        call += ", latency=(None if " + valueString(latency) + " == " +
                std::to_string(inferredLoadPolicy) + " else " +
                valueString(latency) + ")";
      call += gather.getInBounds() ? ", check_bounds=False)"
                                   : ", check_bounds=True)";
      assign(gather.getResult(), call);
      }

  void emitTyped(MMAOp mma) {
      std::string call = mma.getReductionChunk() ? "cutile_math.mma_chunks(" : "ct.mma(";
      call += valueString(mma.getLhs()) + ", " + valueString(mma.getRhs()) +
              ", " + valueString(mma.getAccumulator());
      if (auto chunk = mma.getReductionChunk())
        call += ", " + std::to_string(*chunk);
      assign(mma.getResult(), call + ")");
      }

  void emitTyped(ScaledMMAOp mma) {
      auto lhs = mma.getLhs().getType();
      auto rhs = mma.getRhs().getType();
      std::string lhsK =
          "(" + expressionString(
                     mlir::cast<gpu::PhysicalExprAttr>(lhs.getShape()[1])) +
          " * " + expressionString(
                        mlir::cast<gpu::PhysicalExprAttr>(lhs.getShape()[2])) +
          ")";
      std::string rhsK =
          "(" + expressionString(
                     mlir::cast<gpu::PhysicalExprAttr>(rhs.getShape()[0])) +
          " * " + expressionString(
                        mlir::cast<gpu::PhysicalExprAttr>(rhs.getShape()[1])) +
          ")";
      std::string lhsShape =
          "(" + expressionString(
                     mlir::cast<gpu::PhysicalExprAttr>(lhs.getShape()[0])) +
          ", " + lhsK + ")";
      std::string rhsShape =
          "(" + rhsK + ", " +
          expressionString(
              mlir::cast<gpu::PhysicalExprAttr>(rhs.getShape()[2])) +
          ")";
      assign(mma.getResult(),
             "ct.mma_scaled(ct.reshape(" + valueString(mma.getLhs()) + ", " +
                 lhsShape + "), ct.bitcast(" + valueString(mma.getLhsScale()) +
                 ", ct.float8_e8m0fnu)" +
                 ", ct.reshape(" + valueString(mma.getRhs()) + ", " + rhsShape +
                 "), ct.bitcast(" + valueString(mma.getRhsScale()) +
                 ", ct.float8_e8m0fnu), " +
                 valueString(mma.getAccumulator()) + ")");
      }

  void emitTyped(ReduceOp reduce) {
      if (!reduce.getCombine().empty()) { emitCollective(*reduce.getOperation()); return; }
      auto nativeReduction = [](BinaryOperator kind) -> StringRef {
        switch (kind) {
        case BinaryOperator::Add: return "ct.sum";
        case BinaryOperator::MaximumNum:
        case BinaryOperator::LogicalOr: return "ct.max";
        case BinaryOperator::MinimumNum:
        case BinaryOperator::LogicalAnd: return "ct.min";
        default: llvm_unreachable("unverified cuTile native reduction kind");
        }
      };
      std::string expression =
          nativeReduction(*reduce.getKind()).str() + "(" +
          valueString(reduce.getSources().front()) + ", axis=" +
          std::to_string(reduce.getAxis()) + ")";
      if (elementType(reduce.getResult(0).getType()).isInteger(1))
        expression = "ct.astype(" + expression + ", ct.bool_)";
      assign(reduce.getResult(0), expression);
      }

  void emitTyped(ScanOp scan) {
      if (!scan.getCombine().empty()) { emitCollective(*scan.getOperation()); return; }
      assign(scan.getResult(0), "ct.cumsum(" + valueString(scan.getSources().front()) +
                                   ", axis=" + std::to_string(scan.getAxis()) +
                                   ", reverse=" +
                                   (scan.getReverse() ? "True" : "False") + ")");
      }

  void emitTyped(AtomicRMWOp atomic) {
      auto atomicOperation = [](AtomicRMWKind kind) -> StringRef {
        switch (kind) {
        case AtomicRMWKind::Exchange: return "xchg";
        case AtomicRMWKind::Add: return "add";
        case AtomicRMWKind::Maximum: return "max";
        case AtomicRMWKind::Minimum: return "min";
        case AtomicRMWKind::BitwiseAnd: return "and";
        case AtomicRMWKind::BitwiseOr: return "or";
        case AtomicRMWKind::BitwiseXor: return "xor";
        }
        llvm_unreachable("unhandled atomic RMW kind");
      };
      auto atomicOrder = [](AtomicOrdering ordering) -> StringRef {
        switch (ordering) {
        case AtomicOrdering::Relaxed: return "RELAXED";
        case AtomicOrdering::Acquire: return "ACQUIRE";
        case AtomicOrdering::Release: return "RELEASE";
        case AtomicOrdering::AcquireRelease: return "ACQ_REL";
        }
        llvm_unreachable("unhandled atomic ordering");
      };
      auto atomicScope = [](gpu::AtomicSharingDomain sharing) -> StringRef {
        switch (sharing) {
        case gpu::AtomicSharingDomain::ProgramInstance: return "BLOCK";
        case gpu::AtomicSharingDomain::KernelInvocation: return "DEVICE";
        }
        llvm_unreachable("unhandled atomic sharing domain");
      };
      assign(atomic.getResult(),
             "ct.atomic_" + atomicOperation(atomic.getKind()).str() + "(" +
                 valueString(atomic.getResource()) + ", " +
                 tuple(atomic.getCoordinates()) + ", " +
                 valueString(atomic.getValue()) +
                 ", check_bounds=True, memory_order=ct.MemoryOrder." +
                 atomicOrder(atomic.getOrdering()).str() +
                 ", memory_scope=ct.MemoryScope." +
                 atomicScope(atomic.getSharing()).str() + ")");
      }

  void emitTyped(ExtractOp extract) {
      std::string shape = "(";
      for (Attribute extent : extract.getExtractionShape())
        shape += expressionString(mlir::cast<gpu::PhysicalExprAttr>(extent)) + ", ";
      shape += ")";
      std::string result = "ct.extract(" + valueString(extract.getSource()) +
                           ", index=" + tuple(extract.getCoordinates()) +
                           ", shape=" + shape + ")";
      if (auto fragment = dyn_cast<gpu::FragmentType>(extract.getResult().getType()))
        result += ".reshape(" + fragmentShape(fragment) + ")";
      else
        result += ".item()";
      assign(extract.getResult(), result);
      }

  void emitTyped(TileAtomicAddOp atomic) {
      line(valueString(atomic.getResource()) + ".tiled_view(" +
           fragmentShape(atomic.getValue().getType()) +
           ").atomic_store_add(" + tuple(atomic.getTileIndices()) + ", " +
           valueString(atomic.getValue()) + ")");
      }

  void emitTyped(TileStoreOp store) {
      line("ct.store(" + valueString(store.getResource()) + ", index=" +
           tuple(store.getTileIndices()) + ", tile=" +
           valueString(store.getValue()) + ", allow_tma=" +
           valueString(store.getAllowTma()) + ")");
      }

  void emitTyped(ScalarStoreOp store) {
      std::string call = "ct.scatter(" + valueString(store.getResource()) +
                         ", " + tuple(store.getIndices()) + ", " +
                         valueString(store.getValue());
      if (store.getValid())
        call += ", mask=" + valueString(store.getValid());
      line(call + (store.getInBounds() ? ", check_bounds=False)"
                                       : ", check_bounds=True)"));
      }

  void emitTyped(ScatterStoreOp scatter) {
      std::string call = "ct.scatter(" + valueString(scatter.getResource()) +
                         ", " + tuple(scatter.getCoordinates()) +
                         ", " +
                         valueString(scatter.getValue());
      if (scatter.getValid())
        call += ", mask=" + valueString(scatter.getValid());
      line(call + (scatter.getInBounds() ? ", check_bounds=False)"
                                         : ", check_bounds=True)"));
      }

  void emitCollective(Operation &operation) {
      auto reduce = dyn_cast<ReduceOp>(operation);
      auto scan = dyn_cast<ScanOp>(operation);
      ValueRange sources = reduce ? reduce.getSources() : scan.getSources();
      ValueRange identities = reduce ? reduce.getIdentities() : scan.getIdentities();
      unsigned count = sources.size();
      std::string source = count == 1 ? valueString(sources.front())
                                      : tuple(sources);
      std::string identity;
      if (count == 1) {
        identity = literal(getCompileTimeScalar(identities.front()));
      } else {
        identity = "(";
        for (auto [index, value] : llvm::enumerate(identities)) {
          if (index)
            identity += ", ";
          identity += literal(getCompileTimeScalar(value));
        }
        identity += ")";
      }
      std::string call = std::string(reduce ? "ct.reduce(" : "ct.scan(") +
                         source + ", axis=" +
                         std::to_string(reduce ? reduce.getAxis()
                                               : scan.getAxis()) +
                         ", func=" +
                         helperName(&operation) +
                         ", identity=" + identity;
      if (scan)
        call += std::string(", reverse=") + (scan.getReverse() ? "True" : "False");
      call += ")";
      if (count == 1) {
        assign(operation.getResult(0), call);
      } else {
        std::string resultNames;
        for (auto [index, result] : llvm::enumerate(operation.getResults())) {
          if (index)
            resultNames += ", ";
          std::string name = newName();
          values[result] = name;
          resultNames += name;
        }
        line(resultNames + " = " + call);
      }
      }


  std::string broadcastValue(Value value, gpu::FragmentType target) override {
    auto source = dyn_cast<gpu::FragmentType>(value.getType());
    if (!source)
      return "ct.full(" + fragmentShape(target) + ", " + valueString(value) +
             ", dtype=" + pythonType(target.getElementType()) + ")";
    if (source == target)
      return valueString(value);
    gpu::BroadcastProjection projection =
        gpu::queryBroadcastProjection(source, target);
    if (!projection.isExact()) {
      kernel.emitError("cuTile broadcast lost its shared axis projection");
      failed = true;
      return "<invalid-cutile-broadcast>";
    }
    if (source.getShape().size() == target.getShape().size())
      return "ct.broadcast_to(" + valueString(value) + ", " +
             fragmentShape(target) + ")";
    SmallVector<int64_t> targetForSource(source.getShape().size(), -1);
    for (auto [targetIndex, sourceIndex] :
         llvm::enumerate(projection.targetToSource))
      if (sourceIndex)
        targetForSource[*sourceIndex] = targetIndex;

    SmallVector<unsigned> sourceOrder(source.getShape().size());
    std::iota(sourceOrder.begin(), sourceOrder.end(), 0);
    llvm::sort(sourceOrder, [&](unsigned lhs, unsigned rhs) {
      return targetForSource[lhs] < targetForSource[rhs];
    });
    std::string input = valueString(value);
    bool permuted = llvm::any_of(
        llvm::enumerate(sourceOrder),
        [](auto item) { return item.index() != item.value(); });
    if (permuted) {
      std::string permutation = "(";
      for (unsigned axis : sourceOrder)
        permutation += std::to_string(axis) + ", ";
      permutation += ")";
      input = "ct.permute(" + input + ", " + permutation + ")";
    }

    std::string reshape = "(";
    unsigned orderedSource = 0;
    for (unsigned targetAxis = 0; targetAxis < target.getShape().size();
         ++targetAxis) {
      if (orderedSource < sourceOrder.size() &&
          targetForSource[sourceOrder[orderedSource]] ==
              static_cast<int64_t>(targetAxis)) {
        reshape += expressionString(
                       cast<gpu::PhysicalExprAttr>(
                           source.getShape()[sourceOrder[orderedSource]])) +
                   ", ";
        ++orderedSource;
      } else {
        reshape += "1, ";
      }
    }
    reshape += ")";
    return "ct.broadcast_to(ct.reshape(" + input + ", " + reshape + "), " +
           fragmentShape(target) + ")";
  }

  LogicalResult emitMetadata(std::string &metadata) {
    auto artifact = gpu::serializeInterface(kernel, "cutile", [&](Value value) {
      return valueString(value);
    });
    if (mlir::failed(artifact)) return failure();
    llvm::json::Object details{{"kernel", "_intent_kernel"}};
    llvm::json::Array arguments, arrays;
    for (const ViewABI &view : views) arguments.push_back(view.name);
    for (const ArrayViewABI &view : arrayViews) {
      llvm::json::Array groups;
      ArrayViewOp operation = view.operation;
      for (int64_t end : operation.getGroupEnds()) groups.push_back(end);
      arrays.push_back(llvm::json::Object{
          {"name", view.name}, {"eligible", view.eligible},
          {"base", gpu::getArgumentReference(views[view.sourceView].value).getId()},
          {"group_ends", std::move(groups)}});
      arguments.push_back(view.name);
      arguments.push_back(view.eligible);
    }
    for (const ScalarABI &scalar : scalars) arguments.push_back(scalar.name);
    for (const MetadataABI &argument : metadataArguments) arguments.push_back(argument.name);
    for (gpu::ViewOverlapOp overlap : overlapFacts)
      arguments.push_back(valueString(overlap.getResult()));
    for (Attribute attribute : gpu::getParameterDeclarations(kernel)) {
      auto parameter = cast<gpu::ParameterAttr>(attribute);
      if (providerHint(parameter).empty() && nativeParameters.contains(parameter.getReference()))
        arguments.push_back(parameter.getName().getValue());
    }
    details["kernel_arguments"] = std::move(arguments);
    details["array_views"] = std::move(arrays);
    details["index_tile_bounds"] = nullptr;
    details["narrow_kernel"] = nullptr;
    if (kernel->hasAttr(arrayIndexTileBoundsAttr)) {
      llvm::json::Array encoded;
      auto encodeArray = [&](Value resource, StringRef name, StringRef eligible) {
        llvm::json::Array axes;
        for (Attribute bound : getNativeArrayIndexBounds(resource))
          axes.push_back(gpu::serializeExpression(cast<gpu::PhysicalExprAttr>(bound)));
        encoded.push_back(llvm::json::Object{
            {"array", name.str()}, {"bounds", std::move(axes)},
            {"eligible", eligible.empty() ? llvm::json::Value(nullptr)
                                         : llvm::json::Value(eligible.str())}});
      };
      for (const ViewABI &view : views)
        encodeArray(view.value, view.name, {});
      for (ArrayViewABI view : arrayViews)
        encodeArray(view.operation.getResult(), view.name, view.eligible);
      details["index_tile_bounds"] = std::move(encoded);
      details["narrow_kernel"] = "_intent_i32_kernel";
    }
    llvm::json::Object hints;
    for (const auto &[hint, parameter] : providerHintParameters)
      hints[hint] = parameter;
    details["compiler_hints"] = std::move(hints);
    details["inferred_worker_warps"] = inferredWorkerWarps;
    llvm::json::Array scalarKeys;
    llvm::SmallBitVector keyArguments = getTuningKeyScalarArguments(kernel);
    for (const ScalarABI &scalar : scalars)
      if (keyArguments.test(scalar.value.getArgNumber())) scalarKeys.push_back(scalar.name);
    details["tuning_key_scalars"] = std::move(scalarKeys);
    (*artifact)["cutile"] = std::move(details);
    llvm::raw_string_ostream(metadata) << llvm::json::Value(std::move(*artifact));
    return failed ? failure() : success();
  }

  SmallVector<ViewABI> views;
  SmallVector<ArrayViewABI> arrayViews;
  SmallVector<ScalarABI> scalars;
  SmallVector<MetadataABI> metadataArguments;
  SmallVector<gpu::ViewOverlapOp> overlapFacts;
  std::map<std::string, std::string> providerHintParameters;
  llvm::DenseSet<gpu::ParameterRefAttr> nativeParameters;
};

} // namespace

LogicalResult verifySourceOperation(Operation *operation) {
  return Serializer::sourceOperations().verify(operation);
}

LogicalResult serializeProgram(ModuleOp module, std::string &source,
                               std::string &metadata) {
  if (failed(verifyCuTileProgram(module, verifySourceOperation)))
    return failure();
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  llvm::raw_string_ostream stream(source);
  Serializer serializer(*kernel, stream);
  LogicalResult result = serializer.emit(metadata);
  stream.flush();
  return result;
}

} // namespace intent::cutile
