#include "Construction.h"
#include "Intent/Dialect/DSA/Transforms/MatrixPanels.h"
#include "Intent/Target/BangC/NativeWorkspace.h"
#include <tuple>

namespace intent::kir_to_dsa {
namespace {

std::optional<int64_t> positiveConstant(Value value) {
  APInt number;
  if (!matchPattern(value, m_ConstantInt(&number)) ||
      !number.isSignedIntN(64) || number.getSExtValue() <= 0)
    return std::nullopt;
  return number.getSExtValue();
}

bool reserve(int64_t &total, int64_t elements, int64_t bytes) {
  constexpr int64_t limit = std::numeric_limits<int64_t>::max();
  if (elements <= 0 || bytes <= 0 || elements > (limit - 127) / bytes)
    return false;
  int64_t amount = ((elements * bytes + 127) / 128) * 128;
  if (amount > limit - total) return false;
  total += amount;
  return true;
}

SmallVector<int64_t> cohortExtents(int64_t tile, int64_t extent) {
  SmallVector<int64_t> result{tile};
  // These are cohorts of complete bound tiles, not replacement bindings.
  while (tile <= extent / 2) {
    tile *= 2;
    if (extent % tile == 0) result.push_back(tile);
  }
  return result;
}

} // namespace

void Construction::coarsenMatrixWorkset(Block &block,
    ArrayRef<WorksetTiling> slices, SmallVectorImpl<TileDomain> &tileDomains,
    bool distribute) {
  if (slices.size() != 2 || slices[0].axis != 0 || slices[1].axis != 1 ||
      tileDomains.size() != 2)
    return;
  auto M = positiveConstant(tileDomains[0].extent);
  auto N = positiveConstant(tileDomains[1].extent);
  if (!M || !N) return;

  // A previously materialized complete operand may independently select a
  // larger K panel. Leave its established representation and reuse untouched.
  auto materialized = [&](Value value) {
    while (auto transpose = value.getDefiningOp<TransposeOp>())
      value = transpose.getInput();
    return bool(values.lookupOrNull(value));
  };
  struct Product {
    ContractOp operation;
    ContractionAxes axes;
    Type element;
    int64_t extent, depth;
    WorksetTiling reduction;
    std::optional<dsa::StreamedMatrixPanel> resident;
  };
  SmallVector<Product, 0> products;
  DenseSet<Value> tensors;
  for (const WorksetTiling &slice : slices)
    for (const auto &entry : slice.requirements) tensors.insert(entry.first);
  for (Value value : tensors) {
    auto matrix = value.getDefiningOp<ContractOp>();
    if (!matrix) continue;
    if (matrix->getBlock() != &block) return;
    auto axes = contractionAxes(matrix);
    auto lhs = dyn_cast<RankedTensorType>(matrix.getLhs().getType());
    auto rhs = dyn_cast<RankedTensorType>(matrix.getRhs().getType());
    auto output = dyn_cast<RankedTensorType>(value.getType());
    if (!axes || axes->reduction.size() != 1 || !axes->batch.empty() ||
        axes->lhsFree.size() != 1 || axes->rhsFree.size() != 1 ||
        !lhs || !rhs || !output || lhs.getRank() != 2 || rhs.getRank() != 2 ||
        output.getRank() != 2 || !output.getElementType().isF32() ||
        lhs.getElementType() != rhs.getElementType()) return;
    for (unsigned axis = 0; axis != 2; ++axis) {
      auto found = slices[axis].requirements.find(value);
      if (found == slices[axis].requirements.end() || found->second.size() != 2 ||
          !found->second[axis] || found->second[1 - axis]) return;
    }
    Type element = lhs.getElementType();
    if (!element.isF16() && !element.isBF16() && !element.isF32()) return;
    auto paired = axes->reduction.front();
    auto K = knownExtent(matrix.getLhs(), paired.lhs).constant;
    if (!K || *K <= 0 || knownExtent(matrix.getRhs(), paired.rhs).constant != K ||
        materialized(matrix.getLhs()) || materialized(matrix.getRhs())) return;
    for (auto pair : {std::pair{*M, *N}, std::pair{*M, *K}, std::pair{*K, *N}})
      if (static_cast<__int128>(pair.first) * pair.second >
          std::numeric_limits<int64_t>::max() / 4) return;
    SmallVector<std::pair<Value, unsigned>> roots{
        {matrix.getLhs(), paired.lhs}, {matrix.getRhs(), paired.rhs}};
    auto plan = planExecutionSlices(block, 0, roots);
    if (!plan) return;
    // Use the same complete-domain/current-slice restrictions as contraction
    // formation. A profile value alone is not the physical K capacity.
    auto domain = sliceDomain(*plan, config.getTileK(), !(distribute || distributedTiles));
    if (!domain) return;
    products.push_back({matrix, *axes, element, *K, domain->capacity, std::move(*plan), {}});
  }
  if (products.empty()) return;
  for (const Product &product : products)
    for (const auto &entry : product.reduction.requirements) tensors.insert(entry.first);
  auto workspace = [&](Operation *operation, ArrayRef<int64_t> shape,
                       int64_t &total) {
    auto resources = bangc::queryNativeWorkspace(operation, shape);
    if (!resources) return false;
    for (MemRefType buffer : resources->buffers) {
      int64_t width = (buffer.getElementTypeBitWidth() + 7) / 8;
      if (!reserve(total, buffer.getNumElements(), width)) return false;
    }
    return !resources->internalBytes || reserve(total, 1, resources->internalBytes);
  };
  // Other already-formed scopes keep their allocations. Account for pending
  // native workspace as well, rather than treating an unexpanded operation as
  // zero-cost storage. Unknown native computations keep the original workset.
  int64_t existingWorkspace = 0;
  bool knownWorkspace = true;
  function.walk([&](Operation *operation) {
    if (!knownWorkspace || operation->getNumRegions() || isMemoryEffectFree(operation) ||
        isa<memref::AllocaOp, memref::CopyOp, memref::LoadOp, memref::StoreOp,
            dsa::FillOp, dsa::LoadTileOp, dsa::StoreTileOp, dsa::LoadScalarOp,
            dsa::CastOp, dsa::TransposeOp>(operation)) return;
    Value output;
    if (auto op = dyn_cast<dsa::UnaryOp>(operation)) output = op.getOutput();
    if (auto op = dyn_cast<dsa::BinaryOp>(operation)) output = op.getOutput();
    if (auto op = dyn_cast<dsa::SelectOp>(operation)) output = op.getOutput();
    if (auto op = dyn_cast<dsa::GatherRowsOp>(operation)) output = op.getOutput();
    knownWorkspace = output && workspace(operation,
        cast<MemRefType>(output.getType()).getShape(), existingWorkspace);
  });
  if (!knownWorkspace) return;
  auto requested = [](const WorksetTiling &slice, Value value, unsigned axis) {
    auto found = slice.requirements.find(value);
    return found != slice.requirements.end() && found->second[axis];
  };
  auto localReserve = [&](dsa::MatrixPanelShape shape) -> std::optional<int64_t> {
    int64_t total = existingWorkspace;
    for (Value value : tensors) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || type.getRank() > 2) return std::nullopt;
      Type scalar = type.getElementType();
      if (!scalar.isIntOrIndexOrFloat()) return std::nullopt;
      int64_t elements = 1;
      SmallVector<int64_t, 2> physicalShape;
      for (unsigned axis = 0; axis < type.getRank(); ++axis) {
        std::optional<int64_t> capacity;
        if (requested(slices[0], value, axis)) capacity = shape.rows;
        if (requested(slices[1], value, axis)) {
          if (capacity) return std::nullopt;
          capacity = shape.columns;
        }
        bool freeAxis = capacity.has_value();
        for (const Product &product : products)
          if (requested(product.reduction, value, axis)) {
            if (freeAxis) return std::nullopt;
            capacity = std::max(capacity.value_or(0), product.depth);
          }
        if (capacity && (requested(slices[0], value, axis) ||
                         requested(slices[1], value, axis)))
          if (auto selected = selectedAxis(value, axis);
              selected && *capacity > selected->capacity)
            return std::nullopt;
        if (!capacity) {
          if (auto selected = selectedAxis(value, axis)) capacity = selected->capacity;
          else capacity = knownExtent(value, axis).constant;
        }
        if (!capacity || *capacity <= 0 ||
            elements > std::numeric_limits<int64_t>::max() / *capacity)
          return std::nullopt;
        elements *= *capacity;
        physicalShape.push_back(*capacity);
      }
      // Count every tensor, including broadcasts and aliases that a later pass
      // may eliminate. Existing allocations are separately retained by the
      // shared storage bound; these are not asserted to be a liveness peak.
      int64_t width = scalar.isIndex() ? 8 : (scalar.getIntOrFloatBitWidth() + 7) / 8;
      if (!reserve(total, elements, width)) return std::nullopt;
      if (values.lookupOrNull(value)) continue;
      Operation *definition = value.getDefiningOp();
      if (!definition || definition->getNumRegions()) return std::nullopt;
      if (!workspace(definition, physicalShape, total)) return std::nullopt;
    }
    return total;
  };

  dsa::MatrixPanelShape baseline{tileDomains[0].capacity, tileDomains[1].capacity};
  auto baselineReserve = localReserve(baseline);
  if (!baselineReserve) return;
  // The cooperative form is already owned by MatrixSupply. Recognize its
  // source-side complete matrix domain only to compare resident supply costs;
  // its current-IR effect/coordinate proof still decides actual formation.
  for (Product &product : products) {
    auto matrix = product.operation;
    auto lhsLoad = matrix.getLhs().getDefiningOp<ViewLoadOp>();
    auto rhsLoad = matrix.getRhs().getDefiningOp<ViewLoadOp>();
    if (!distribute || !isa<func::FuncOp>(block.getParentOp()) ||
        !product.axes.hasCanonicalMatrixAxes() || !lhsLoad || !rhsLoad ||
        cast<RankedTensorType>(cast<ViewType>(lhsLoad.getSource().getType()).getTensor()).getRank() != 2 ||
        cast<RankedTensorType>(cast<ViewType>(rhsLoad.getSource().getType()).getTensor()).getRank() != 2 ||
        config.getTasks() < 4 || config.getTasks() % 4 ||
        *N / baseline.columns < config.getTasks()) continue;
    // A cooperative launch must own every observable output. Independent
    // products still coarsen together through the ordinary local supply path.
    DenseMap<Value, std::optional<bool>> derived;
    std::function<std::optional<bool>(Value)> fromMatrix = [&](Value value) -> std::optional<bool> {
      if (value == matrix.getResult()) return true;
      if (auto found = derived.find(value); found != derived.end()) return found->second;
      Operation *definition = value.getDefiningOp();
      if (!definition || definition->getBlock() != &block) return false;
      if (isa<ContractOp, ViewLoadOp, GatherOp>(definition) || definition->getNumRegions()) return std::nullopt;
      bool contains = false;
      for (Value input : definition->getOperands()) {
        auto parent = fromMatrix(input);
        if (!parent) return derived[value] = std::nullopt;
        contains |= *parent;
      }
      return derived[value] = contains;
    };
    bool complete = llvm::all_of(slices[0].writes, [&](Operation *write) {
      auto value = cast<IndexedAccessOpInterface>(write).getStoredValue();
      auto origin = fromMatrix(value);
      return origin && *origin;
    });
    if (complete)
      product.resident = dsa::selectStreamedMatrixPanel(function, config, product.element,
          baseline.rows, baseline.columns, product.extent, *baselineReserve);
  }
  using Score = std::tuple<__int128, __int128, __int128, int64_t>;
  auto score = [&](dsa::MatrixPanelShape shape, int64_t storage) -> Score {
    __int128 mt = (*M - 1) / shape.rows + 1;
    __int128 nt = (*N - 1) / shape.columns + 1;
    __int128 external = 0, local = 0, calls = 0;
    for (const Product &product : products) {
      int64_t elementBytes = product.element.getIntOrFloatBitWidth() / 8;
      __int128 lhs = static_cast<__int128>(*M) * product.extent * elementBytes;
      __int128 rhs = static_cast<__int128>(product.extent) * *N * elementBytes;
      if (product.resident) {
        external += lhs * ((nt + 3) / 4);
        local += rhs;
        calls += mt * nt;
      } else {
        // Every actual product retains its own K panel and native preparation.
        // N>64 additionally permutes a filter before copying it into WRAM.
        local += lhs * nt + (3 + (shape.columns == 64 ? 0 : 2)) * rhs * mt;
        calls += mt * nt * ((product.extent - 1) / product.depth + 1);
      }
    }
    return {external, local, calls, storage};
  };
  auto best = score(baseline, *baselineReserve);
  auto selected = baseline;
  for (int64_t rows : cohortExtents(baseline.rows, *M))
    for (int64_t columns : cohortExtents(baseline.columns, *N)) {
      if (rows == baseline.rows && columns == baseline.columns) continue;
      dsa::MatrixPanelShape shape{rows, columns};
      auto storage = localReserve(shape);
      if (!storage) continue;
      SmallVector<dsa::MatrixPanelStorage> panels;
      bool complete = true;
      for (const Product &product : products) {
        auto panel = dsa::matrixPanelStorage(product.element, rows, columns, product.depth);
        if (!panel) { complete = false; break; }
        panels.push_back(*panel);
        if (product.resident) {
          auto resident = dsa::selectStreamedMatrixPanel(function, config, product.element,
              rows, columns, product.extent, *storage);
          if (*N / columns < config.getTasks() || !resident ||
              (product.resident->pipeline && !resident->pipeline)) {
            complete = false; break;
          }
        }
      }
      if (!complete || !dsa::matrixPanelsFitStorage(function, config, panels, *storage)) continue;
      auto cost = score(shape, *storage);
      if (cost < best) { best = cost; selected = shape; }
    }
  tileDomains[0].capacity = selected.rows;
  tileDomains[1].capacity = selected.columns;
}

} // namespace intent::kir_to_dsa
