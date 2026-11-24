#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace compilerinfra;

namespace intentdsl::cpu {
namespace {
int64_t clampCount(int64_t a, int64_t b, int64_t c = 1, int64_t d = 1) {
  __int128 prod = __int128(std::max<int64_t>(0, a)) *
                  std::max<int64_t>(0, b) * std::max<int64_t>(0, c) *
                  std::max<int64_t>(0, d);
  if (prod > INT64_MAX)
    return INT64_MAX;
  return int64_t(prod);
}
} // namespace

llvm::Error materializeMatrixBuffers(func::FuncOp fn, RewriterBase &rw,
                                     StringRef realization) {
  auto *task = findMatrixTask(fn);
  if (!task)
    return llvm::createStringError("missing cpu.matrix_task");
  auto *taskJoin = task->getNextNode();
  if (!taskJoin || taskJoin->getName().getStringRef() != "cpu.join")
    return llvm::createStringError("missing join after cpu.matrix_task");

  auto &domain = task->getRegion(0).front().front();
  auto &dot = calculation(domain);
  auto kind = dot.getName().stripDialect();
  auto extents = domain.getAttrOfType<DenseI64ArrayAttr>("cpu.extents");
  auto tile = domain.getAttrOfType<DenseI64ArrayAttr>("cpu.output_tile");
  auto packingAttr = domain.getAttrOfType<StringAttr>("cpu.packing");
  StringRef packing = packingAttr ? packingAttr.getValue() : StringRef();
  bool routed = domain.hasAttr("cpu.route_storage");

  int64_t M = extents[0], N = extents[1], K = extents[2], G = extents[3];
  int64_t nr = tile[1];
  int64_t tiles = nr > 0 ? (N + nr - 1) / nr : 0;

  Location loc = task->getLoc();
  OpBuilder::InsertionGuard guard(rw);
  rw.setInsertionPoint(task);

  SmallVector<Value> buffers;
  SmallVector<std::pair<Value, StringRef>> heapReleases;

  auto allocBuf = [&](StringRef placement, int64_t elements, int64_t elemBytes,
                      StringRef role, Type elemType) -> Value {
    OperationState s(loc, "cpu.alloc_buffer");
    s.addAttribute("placement", rw.getStringAttr(placement));
    s.addAttribute("elements",
                   rw.getI64IntegerAttr(std::max<int64_t>(1, elements)));
    s.addAttribute("element_bytes", rw.getI64IntegerAttr(elemBytes));
    s.addAttribute("alignment", rw.getI64IntegerAttr(64));
    s.addAttribute("role", rw.getStringAttr(role));
    s.addTypes(MemRefType::get({ShapedType::kDynamic}, elemType));
    Value buf = rw.create(s)->getResult(0);
    buffers.push_back(buf);
    if (placement == "heap")
      heapReleases.push_back({buf, placement});
    return buf;
  };
  auto emitPrep = [&](ValueRange operands, StringRef prepKind,
                      ArrayRef<NamedAttribute> extraAttrs = {},
                      Region *cloneRegion = nullptr) {
    OperationState s(loc, "cpu.prepare_panels");
    s.addOperands(operands);
    s.addAttribute("kind", rw.getStringAttr(prepKind));
    for (auto attr : extraAttrs)
      s.addAttribute(attr.getName(), attr.getValue());
    s.addTypes(rw.getIndexType());
    if (cloneRegion) {
      IRMapping mapping;
      cloneRegion->cloneInto(s.addRegion(), mapping);
    }
    auto *prep = rw.create(s);
    OperationState j(loc, "cpu.join");
    j.addOperands(prep->getResult(0));
    rw.create(j);
  };

  auto f32Ty = rw.getF32Type();
  auto i8Ty = rw.getI8Type();

  if (packing == "mapped-right-per-call" || packing == "mapped-f32-panels") {
    bool wide = packing == "mapped-f32-panels";
    if (wide) {
      Value left =
          allocBuf("heap", clampCount(G, M, K), 4, "lhs-panels", f32Ty);
      emitPrep({dot.getOperand(0), left}, "widen-window-lhs");
    }
    Value right = allocBuf("heap", clampCount(G, K, N), wide ? 4 : 2,
                           "rhs-panels", wide ? Type(f32Ty) : rw.getBF16Type());
    emitPrep({dot.getOperand(1), dot.getOperand(2), dot.getOperand(3), right},
             wide ? "map-window-rhs-f32" : "map-window-rhs-bf16",
             {rw.getNamedAttr("head_group", dot.getAttr("head_group"))},
             &dot.getRegion(0));
  } else if (packing == "rhs-panels-per-call") {
    auto b = dot.getOperand(1);
    auto bTy = cast<MemRefType>(b.getType()).getElementType();
    int64_t bytes = bTy.getIntOrFloatBitWidth() / 8;
    int64_t batch = dot.getAttrOfType<IntegerAttr>("rhs_batch").getInt();
    int64_t banks = batch < 0 ? 1 : G;
    Value rhs =
        allocBuf("heap", clampCount(banks, tiles, K, nr), bytes, "rhs-panels", bTy);
    emitPrep({b, rhs}, "pack-rhs-panels",
             {rw.getNamedAttr("rhs_k", dot.getAttr("rhs_k")),
              rw.getNamedAttr("rhs_batch", dot.getAttr("rhs_batch")),
              rw.getNamedAttr("panel_cols", rw.getI64IntegerAttr(nr))});
  } else if (packing == "signed-dot4-panels") {
    auto b = dot.getOperand(1);
    int64_t batch = dot.getAttrOfType<IntegerAttr>("rhs_batch").getInt();
    int64_t banks = batch < 0 ? 1 : G;
    Value rhs = allocBuf("heap", clampCount(banks, tiles, nr, K + 4), 1,
                         "rhs-panels", i8Ty);
    emitPrep({b, rhs}, "pack-signed-dot4",
             {rw.getNamedAttr("rhs_k", dot.getAttr("rhs_k")),
              rw.getNamedAttr("rhs_batch", dot.getAttr("rhs_batch")),
              rw.getNamedAttr("panel_cols", rw.getI64IntegerAttr(nr))});
  } else if (packing == "vnni-pack32-panels") {
    auto b = dot.getOperand(1);
    auto bTy = cast<MemRefType>(b.getType()).getElementType();
    int64_t bytes = bTy.getIntOrFloatBitWidth() / 8;
    int64_t kBlk = bTy.isInteger(8) ? 4 : 2;
    int64_t kPad = ((K + kBlk - 1) / kBlk) * kBlk;
    int64_t batch = dot.getAttrOfType<IntegerAttr>("rhs_batch").getInt();
    int64_t banks = batch < 0 ? 1 : G;
    Value rhs = allocBuf("heap", clampCount(banks, tiles, kPad, nr), bytes,
                         "rhs-panels", bTy);
    emitPrep({b, rhs}, "pack-vnni",
             {rw.getNamedAttr("rhs_k", dot.getAttr("rhs_k")),
              rw.getNamedAttr("rhs_batch", dot.getAttr("rhs_batch")),
              rw.getNamedAttr("panel_cols", rw.getI64IntegerAttr(nr)),
              rw.getNamedAttr("k_block", rw.getI64IntegerAttr(kBlk))});
  } else if (packing == "f32-panels-per-call") {
    if (!routed) {
      auto a = dot.getOperand(0);
      int64_t ab = dot.getAttrOfType<IntegerAttr>("lhs_batch").getInt();
      int64_t lhsBanks = ab < 0 ? 1 : G;
      Value left =
          allocBuf("heap", clampCount(lhsBanks, M, K), 4, "lhs-panels", f32Ty);
      emitPrep({a, left}, "widen-lhs-f32",
               {rw.getNamedAttr("lhs_k", dot.getAttr("lhs_k")),
                rw.getNamedAttr("lhs_batch", dot.getAttr("lhs_batch"))});
    }
    auto b = dot.getOperand(1);
    int64_t bb = dot.getAttrOfType<IntegerAttr>("rhs_batch").getInt();
    int64_t rhsBanks = bb < 0 ? 1 : G;
    Value rhs = allocBuf("heap", clampCount(rhsBanks, tiles, K, nr), 4,
                         "rhs-panels", f32Ty);
    emitPrep({b, rhs}, "widen-pack-rhs-f32",
             {rw.getNamedAttr("rhs_k", dot.getAttr("rhs_k")),
              rw.getNamedAttr("rhs_batch", dot.getAttr("rhs_batch")),
              rw.getNamedAttr("panel_cols", rw.getI64IntegerAttr(nr))});
  } else if (packing == "decoded-f32-per-call") {
    auto mxAttr =
        rw.getNamedAttr("kir.storage", TypeAttr::get(kir::MX8Type::get(fn.getContext())));
    Value left = allocBuf("heap", clampCount(M, K), 4, "lhs-panels", f32Ty);
    emitPrep({dot.getOperand(0), dot.getOperand(1), left}, "decode-mx8-lhs",
             {mxAttr});
    kir::RightScale rs(dot);
    Value right = allocBuf("heap", clampCount(K, N), 4, "rhs-panels", f32Ty);
    emitPrep({dot.getOperand(2), rs.view, right}, "decode-mx8-rhs",
             {mxAttr,
              rw.getNamedAttr("scale_column_axis",
                              rw.getI64IntegerAttr(rs.column)),
              rw.getNamedAttr("scale_group_axis",
                              rw.getI64IntegerAttr(rs.group))});
  } else if (packing == "shared-q8k-per-call") {
    auto placement =
        domain.getAttrOfType<StringAttr>("cpu.record_storage").getValue();
    auto q8Attr =
        rw.getNamedAttr("kir.storage", TypeAttr::get(kir::Q8KType::get(fn.getContext())));
    Value x = dot.getOperand(1).getDefiningOp()->getOperand(0);
    Value packed = allocBuf(placement, std::max<int64_t>(1, K / 256),
                            kir::Q8KRecord::bytes, "q8k-records", i8Ty);
    emitPrep({x, packed}, "encode-q8k", {q8Attr});
  }

  if (routed) {
    auto a = dot.getOperand(0);
    auto aTy = cast<MemRefType>(a.getType()).getElementType();
    bool wide = packing == "f32-panels-per-call";
    int64_t bytes = wide ? 4 : (aTy.getIntOrFloatBitWidth() / 8);
    auto outSlot = cast<BlockArgument>(domain.getOperand(0)).getArgNumber();
    int64_t mult = fn.getArgAttrOfType<DenseI64ArrayAttr>(outSlot, "cpu.shape")[1];
    Value members = allocBuf("heap", clampCount(M, K), bytes, "routed-lhs",
                             wide ? Type(f32Ty) : aTy);
    emitPrep({a, dot.getOperand(2), dot.getOperand(3), members},
             wide ? "gather-widen-routed-lhs-f32" : "gather-routed-lhs",
             {rw.getNamedAttr("lhs_k", dot.getAttr("lhs_k")),
              rw.getNamedAttr("multiplicity", rw.getI64IntegerAttr(mult))});
  }

  SmallVector<Value> captures(fn.getArguments().begin(),
                              fn.getArguments().end());
  captures.append(buffers.begin(), buffers.end());

  OperationState newTaskState(loc, "cpu.matrix_task");
  newTaskState.addOperands(captures);
  newTaskState.addTypes(rw.getIndexType());
  for (auto attr : task->getAttrs())
    if (attr.getName() != "cpu.storage")
      newTaskState.addAttribute(attr.getName(), attr.getValue());
  newTaskState.addAttribute("cpu.realization",
                            rw.getStringAttr(realization));
  newTaskState.addRegion()->takeBody(task->getRegion(0));
  auto *newTask = rw.create(newTaskState);

  taskJoin->setOperand(0, newTask->getResult(0));
  rw.eraseOp(task);

  rw.setInsertionPointAfter(taskJoin);
  for (auto &[buf, place] : llvm::reverse(heapReleases)) {
    OperationState rel(loc, "cpu.release_buffer");
    rel.addOperands(buf);
    rel.addAttribute("placement", rw.getStringAttr(place));
    rw.create(rel);
  }
  return llvm::Error::success();
}

LogicalResult verifyMatrixBuffers(func::FuncOp fn, const TargetProfile &p,
                                  StringRef realization) {
  auto *task = findMatrixTask(fn);
  if (!task)
    return fn.emitError("missing cpu.matrix_task at cpu@buffers");
  if (!llvm::hasSingleElement(task->getRegion(0)) ||
      task->getRegion(0).front().getOperations().size() != 2)
    return task->emitError("expected a matrix region and task completion");
  auto &region = task->getRegion(0).front().front();
  if (region.getName().getStringRef() != "cpu.matrix_domain" ||
      failed(kir::verifyMatrixBody(fn, region, "cpu", true)) ||
      failed(verifyMatrixPartition(fn, region, p)))
    return failure();
  if (task->getAttrOfType<StringAttr>("cpu.realization") != realization ||
      task->hasAttr("cpu.storage"))
    return task->emitError("invalid matrix buffer realization state");

  SmallVector<Value> allocs;
  llvm::DenseMap<Value, StringRef> placements;
  bool seenTask = false, seenTaskJoin = false;
  llvm::DenseSet<Value> released;

  for (auto &op : fn.front()) {
    auto name = op.getName().getStringRef();
    if (name == "func.return")
      break;
    if (!seenTask) {
      if (name == "cpu.alloc_buffer") {
        auto place = op.getAttrOfType<StringAttr>("placement");
        if (!place || (place != "heap" && place != "stack") ||
            op.getAttrOfType<IntegerAttr>("elements").getInt() < 1 ||
            op.getAttrOfType<IntegerAttr>("element_bytes").getInt() < 1)
          return op.emitError("invalid matrix buffer allocation");
        allocs.push_back(op.getResult(0));
        placements[op.getResult(0)] = place.getValue();
        continue;
      }
      if (name == "cpu.prepare_panels") {
        auto *next = op.getNextNode();
        if (!next || next->getName().getStringRef() != "cpu.join" ||
            next->getOperand(0) != op.getResult(0))
          return op.emitError(
              "panel preparation must complete before matrix_task");
        auto k = op.getAttrOfType<StringAttr>("kind").getValue();
        if ((k == "decode-mx8-lhs" || k == "decode-mx8-rhs") &&
            op.getAttr("kir.storage") !=
                TypeAttr::get(kir::MX8Type::get(fn.getContext())))
          return op.emitError("MX8 panel decode missing kir.storage");
        if (k == "encode-q8k" &&
            op.getAttr("kir.storage") !=
                TypeAttr::get(kir::Q8KType::get(fn.getContext())))
          return op.emitError("Q8_K panel encode missing kir.storage");
        continue;
      }
      if (name == "cpu.join") {
        auto *def = op.getOperand(0).getDefiningOp();
        if (!def || def->getName().getStringRef() != "cpu.prepare_panels")
          return op.emitError("unexpected pre-task join");
        continue;
      }
      if (&op == task) {
        seenTask = true;
        continue;
      }
      return op.emitError("unexpected operation before cpu.matrix_task");
    }
    if (!seenTaskJoin) {
      if (name != "cpu.join" || op.getOperand(0) != task->getResult(0))
        return op.emitError("cpu.matrix_task must be immediately joined");
      seenTaskJoin = true;
      continue;
    }
    if (name == "cpu.release_buffer") {
      Value buf = op.getOperand(0);
      if (!placements.count(buf) || placements[buf] != "heap" ||
          !released.insert(buf).second)
        return op.emitError("invalid or duplicate buffer release");
      continue;
    }
    return op.emitError("unexpected operation after matrix task completion");
  }

  if (!seenTask || !seenTaskJoin)
    return fn.emitError("incomplete matrix task synchronization");
  for (auto &[buf, place] : placements)
    if (place == "heap" && !released.count(buf))
      return fn.emitError("leaked heap matrix workspace buffer");

  SmallVector<Value> expected(fn.getArguments().begin(),
                              fn.getArguments().end());
  expected.append(allocs.begin(), allocs.end());
  if (task->getOperands() != ValueRange(expected))
    return task->emitError(
        "cpu.matrix_task must capture host arguments and prepared buffers");
  return success();
}
} // namespace intentdsl::cpu
