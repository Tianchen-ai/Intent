#include "Intent/Dialect/DSA/Transforms/MatrixPanels.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <limits>

using namespace mlir;
namespace intent::dsa {
namespace {
bool add(int64_t &total, int64_t amount) {
  if (amount < 0 || total > std::numeric_limits<int64_t>::max() - amount)
    return false;
  total += amount;
  return true;
}

std::optional<int64_t> bytes(int64_t rows, int64_t columns,
                             int64_t elementBytes, int64_t alignment) {
  if (rows <= 0 || columns <= 0 || elementBytes <= 0 ||
      rows > std::numeric_limits<int64_t>::max() / columns)
    return std::nullopt;
  int64_t elements = rows * columns;
  if (elements > (std::numeric_limits<int64_t>::max() - alignment + 1) /
                     elementBytes)
    return std::nullopt;
  int64_t size = elements * elementBytes;
  return ((size + alignment - 1) / alignment) * alignment;
}

struct AvailableStorage { int64_t nram, wram, sram; };

std::optional<AvailableStorage> availableStorage(func::FuncOp function,
                                                ConfigurationAttr config) {
  int64_t nram = 0, wram = 0, sram = 0, internalNram = 0;
  bool complete = true;
  // This is an upper bound retaining even disjoint frames, not an arena peak.
  // Final target formation measures actual allocation lifetimes independently.
  function.walk([&](Operation *operation) {
    if (auto reserve = operation->getAttrOfType<IntegerAttr>("bangc.internal_nram_bytes"))
      internalNram = std::max(internalNram, reserve.getInt());
    auto allocation = dyn_cast<memref::AllocaOp>(operation);
    if (!allocation) return;
    auto type = allocation.getType();
    Type element = type.getElementType();
    if (!type.hasStaticShape() || !type.getLayout().isIdentity() ||
        (!element.isIntOrFloat() && !element.isIndex())) {
      complete = false;
      return;
    }
    int64_t count = 1;
    for (int64_t size : type.getShape()) {
      if (size <= 0 || count > std::numeric_limits<int64_t>::max() / size) {
        complete = false;
        return;
      }
      count *= size;
    }
    int64_t space = type.getMemorySpaceAsInt();
    int64_t width = element.isIndex() ? 8 : (element.getIntOrFloatBitWidth() + 7) / 8;
    auto size = bytes(count, 1, width, space == matrixSpace ? 16 * 64 : 128);
    if (!size) { complete = false; return; }
    int64_t *total = space == nramSpace ? &nram : space == matrixSpace ? &wram
                              : space == sharedSpace ? &sram : nullptr;
    if (!total || !add(*total, *size)) complete = false;
  });
  int64_t nramLimit = std::min<int64_t>(config.getLocalBytes(), 768 * 1024);
  if (!complete || internalNram < 0 || internalNram > nramLimit ||
      nram > nramLimit - internalNram || wram > 1024 * 1024 || sram > 3968 * 1024)
    return std::nullopt;
  return AvailableStorage{nramLimit - internalNram - nram,
                          1024 * 1024 - wram, 3968 * 1024 - sram};
}
} // namespace

std::optional<MatrixPanelStorage>
matrixPanelStorage(Type element, int64_t rows, int64_t columns,
                   int64_t depth) {
  if ((!element.isF16() && !element.isBF16() && !element.isF32()) ||
      rows <= 0 || columns <= 0 || depth <= 0 || columns % 64)
    return std::nullopt;
  int64_t elementBytes = element.getIntOrFloatBitWidth() / 8;
  // MTP3xx same-dtype conv_partial requires a 64-byte input-channel quantum;
  // filter output channels use groups of 64. WRAM offsets span sixteen banks.
  if (depth % (64 / elementBytes)) return std::nullopt;
  auto lhs = bytes(rows, depth, elementBytes, 128);
  auto rhs = bytes(depth, columns, elementBytes, 128);
  auto accumulator = bytes(rows, columns, 4, 128);
  auto packed = bytes(depth, columns, elementBytes, 16 * 64);
  if (!lhs || !rhs || !accumulator || !packed) return std::nullopt;
  int64_t nram = *accumulator;
  // Two input slots plus raw, transposed and reshaped RHS storage bound both
  // the direct and streamed native preparation paths without assuming reuse.
  for (int64_t size : {*lhs, *lhs, *rhs, *rhs, *rhs})
    if (!add(nram, size)) return std::nullopt;
  return MatrixPanelStorage{nram, *packed};
}

bool matrixPanelsFitStorage(func::FuncOp function, ConfigurationAttr config,
                            ArrayRef<MatrixPanelStorage> panels, int64_t localReserve) {
  auto available = availableStorage(function, config);
  if (!available || localReserve < 0) return false;
  int64_t nram = localReserve, wram = 0;
  for (const MatrixPanelStorage &panel : panels)
    if (!add(nram, panel.nram) || !add(wram, panel.wram)) return false;
  return nram <= available->nram && wram <= available->wram;
}

llvm::SmallVector<MatrixPanelShape> largerMatrixPanels(
    ConfigurationAttr config, Type element, int64_t rows, int64_t columns,
    int64_t depth, MatrixPanelShape baseline) {
  llvm::SmallVector<MatrixPanelShape> result;
  if (rows <= 0 || columns <= 0 || baseline.rows <= 0 || baseline.columns <= 0 ||
      baseline.columns % 64 ||
      columns > std::numeric_limits<int64_t>::max() - 63)
    return result;
  baseline.rows = std::min(baseline.rows, rows);
  int64_t paddedColumns = ((columns + 63) / 64) * 64;
  if (baseline.columns > paddedColumns) return result;
  auto steps = [](int64_t first, int64_t last) {
    llvm::SmallVector<int64_t> values{first};
    while (first < last) {
      first = first > last / 2 ? last : first * 2;
      values.push_back(first);
    }
    return values;
  };
  auto rectangles = [&](MatrixPanelShape shape) -> __int128 {
    return static_cast<__int128>((rows - 1) / shape.rows + 1) *
        ((columns - 1) / shape.columns + 1);
  };
  struct Candidate { MatrixPanelShape shape; int64_t bytes; };
  llvm::SmallVector<Candidate> candidates;
  for (int64_t m : steps(baseline.rows, rows))
    for (int64_t n : steps(baseline.columns, paddedColumns)) {
      MatrixPanelShape shape{m, n};
      if (rectangles(shape) >= rectangles(baseline)) continue;
      auto storage = matrixPanelStorage(element, m, n, depth);
      if (!storage || storage->nram > std::min<int64_t>(config.getLocalBytes(), 768 * 1024) ||
          storage->wram > 1024 * 1024) continue;
      candidates.push_back({shape, storage->nram});
    }
  llvm::stable_sort(candidates, [&](const Candidate &a, const Candidate &b) {
    auto left = rectangles(a.shape), right = rectangles(b.shape);
    return left != right ? left < right : a.bytes < b.bytes;
  });
  for (const Candidate &candidate : candidates) result.push_back(candidate.shape);
  return result;
}

int64_t selectMatrixPanelDepth(func::FuncOp function, ConfigurationAttr config,
                              Type element, int64_t rows, int64_t columns,
                              int64_t availableDepth, int64_t baselineDepth,
                              unsigned products) {
  if (baselineDepth <= 0 || availableDepth <= baselineDepth || !products ||
      availableDepth % baselineDepth)
    return baselineDepth;
  auto available = availableStorage(function, config);
  if (!available) return baselineDepth;
  int64_t selected = baselineDepth;
  // Grow within the available panel without introducing a partial K update.
  for (int64_t depth = baselineDepth; depth <= availableDepth / 2;) {
    depth *= 2;
    if (availableDepth % depth) continue;
    auto storage = matrixPanelStorage(element, rows, columns, depth);
    if (!storage) continue;
    if (storage->nram > available->nram / products ||
        storage->wram > available->wram / products)
      break;
    selected = depth;
  }
  return selected;
}

std::optional<StreamedMatrixPanel> selectStreamedMatrixPanel(
    func::FuncOp function, ConfigurationAttr config, Type element,
    int64_t rows, int64_t columns, int64_t depth, int64_t localReserve) {
  auto geometry = matrixPanelStorage(element, rows, columns, depth);
  auto available = availableStorage(function, config);
  if (!geometry || !available || geometry->wram > available->wram ||
      localReserve < 0 || localReserve > available->nram)
    return std::nullopt;
  available->nram -= localReserve;
  int64_t elementBytes = element.getIntOrFloatBitWidth() / 8;
  auto lhs = bytes(rows, depth, elementBytes, 128);
  auto accumulator = bytes(rows, columns, 4, 128);
  if (!lhs || !accumulator || *lhs > available->sram / 2)
    return std::nullopt;
  int64_t sliceColumns = 256;
  while (columns % sliceColumns) sliceColumns /= 2;
  int64_t minimumDepth = 64 / elementBytes;
  auto minimumSlice = bytes(minimumDepth, sliceColumns, elementBytes, 128);
  if (!minimumSlice) return std::nullopt;
  int64_t fixed = *accumulator;
  if (!add(fixed, *lhs) || fixed > available->nram ||
      *minimumSlice > (available->nram - fixed) / 2)
    return std::nullopt;
  bool pipeline = *lhs <= available->nram - fixed - 2 * *minimumSlice;
  if (pipeline && !add(fixed, *lhs)) return std::nullopt;
  int64_t sliceDepth = 2048;
  for (; sliceDepth >= minimumDepth; sliceDepth /= 2) {
    if (sliceDepth > depth || depth % sliceDepth) continue;
    auto slice = bytes(sliceDepth, sliceColumns, elementBytes, 128);
    if (slice && *slice <= (available->nram - fixed) / 2)
      return StreamedMatrixPanel{depth, sliceDepth, sliceColumns, pipeline};
  }
  return std::nullopt;
}
} // namespace intent::dsa
