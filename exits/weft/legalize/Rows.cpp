#include "Rows.h"
#include "compilerinfra/Proofs.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"
#include "intentdsl/levels/algorithm/kir/ir/ProjectedLayout.h"
#include "intentdsl/levels/execution/cpu/ir/IndexedEffects.h"
#include "intentdsl/levels/execution/cpu/ir/Pack.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;

bool canNarrowWeftAddress(Operation &op);
TerminalPrinter weftTerminalPrinter();

namespace {
struct TaskTerm {
  Value v;
  std::string type;
};

Operation *makeOp(OpBuilder &b, Location loc, StringRef name,
                  ValueRange operands = {}, TypeRange types = {},
                  ArrayRef<NamedAttribute> attrs = {},
                  unsigned numRegions = 0) {
  OperationState s(loc, ("weft_host." + name).str());
  s.addOperands(operands);
  s.addTypes(types);
  s.addAttributes(attrs);
  for (unsigned i = 0; i < numRegions; ++i)
    s.addRegion();
  return b.create(s);
}

std::string printMlirType(Type t) {
  std::string s;
  llvm::raw_string_ostream out(s);
  out << t;
  return s;
}

std::string weftSpelling(Type t) {
  if (t.isInteger(1))
    return "i1";
  if (t.isUnsignedInteger(64))
    return "ui64";
  if (t.isSignlessInteger(64) || t.isInteger(64))
    return "si64";
  if (t.isInteger(32))
    return "si32";
  return printMlirType(t);
}

std::string regionType(StringRef kind, StringRef element, int rank,
                       int columnDim = -2, int axisId = 2) {
  std::string shape = rank == 2 ? "[-1,-2],[1,2]"
                                : "[" + std::to_string(columnDim) + "],[" +
                                      std::to_string(axisId) + "]";
  if (kind == "value")
    return "!weft_kernel.value<" + element.str() + "," + shape + ">";
  std::string family = element == "si64"   ? "i64"
                       : element == "si32" ? "i32"
                                           : element.str();
  return "!weft_kernel." + kind.str() + "<!weft_kernel.encoding<\"" + family +
         "\",\"dense\",\"dense." + family + "\",[]>," + shape + ">";
}

std::string stripTypeSuffix(Attribute valAttr) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << valAttr;
  os.flush();
  auto colon = s.rfind(':');
  if (colon != std::string::npos) {
    while (colon > 0 && s[colon - 1] == ' ')
      --colon;
    s.resize(colon);
  }
  return s;
}

struct SlotFacts {
  int64_t rank = 0;
  std::optional<int64_t> tableRows;
  bool taskIndices = false;
  SmallVector<int64_t, 4> indirectExtents;
  bool contiguous = false;
  bool swapAxes = false;
  bool zeroInnerStride = false;
};

struct SweepDesc {
  std::string combine; // "store", "publish", "sum", "max_num", "welford"
  Operation *anchor = nullptr;
  SmallVector<Operation *, 4> anchors;
  SmallVector<Operation *> writes;
  bool privateBins = false;
  bool verticalSum = false;
  bool segmentLoop = false;
  bool prefixCarry = false;
};

struct RowTaskBuilder {
  func::FuncOp fn;
  Operation &task;
  OpBuilder &b;
  Location loc;
  ArrayRef<SlotFacts> slotFacts;
  llvm::DenseMap<Value, TaskTerm> terms;
  llvm::DenseMap<Value, SmallVector<TaskTerm, 3>> records;
  TaskTerm extent, columns, panelWidth, origin, keyLo, keyHi;
  TaskTerm zeroIdx, oneIdx, zeroCount, oneCount, zeroF32, minusInfF32;
  std::optional<int64_t> ownerAxis;
  SmallVector<TaskTerm, 4> extents;
  bool ownedKeys = false;

  RowTaskBuilder(func::FuncOp fn, Operation &task, OpBuilder &b, Location loc,
                 ArrayRef<SlotFacts> slotFacts)
      : fn(fn), task(task), b(b), loc(loc), slotFacts(slotFacts) {}

  TaskTerm idx(int64_t val) {
    Value v =
        makeOp(b, loc, "task_const", {}, {b.getIndexType()},
               {b.getNamedAttr("lit", b.getStringAttr(std::to_string(val))),
                b.getNamedAttr("val_type", b.getStringAttr("index"))})
            ->getResult(0);
    return {v, "index"};
  }

  TaskTerm lit(StringRef s, StringRef ty) {
    Value v = makeOp(b, loc, "task_const", {}, {b.getIndexType()},
                     {b.getNamedAttr("lit", b.getStringAttr(s)),
                      b.getNamedAttr("val_type", b.getStringAttr(ty))})
                  ->getResult(0);
    return {v, ty.str()};
  }

  TaskTerm bin(StringRef kind, TaskTerm lhs, TaskTerm rhs, StringRef resTy) {
    Value v = makeOp(b, loc, "task_binary", {lhs.v, rhs.v}, {b.getIndexType()},
                     {b.getNamedAttr("kind", b.getStringAttr(kind)),
                      b.getNamedAttr("lhs_type", b.getStringAttr(lhs.type)),
                      b.getNamedAttr("rhs_type", b.getStringAttr(rhs.type)),
                      b.getNamedAttr("val_type", b.getStringAttr(resTy))})
                  ->getResult(0);
    return {v, resTy.str()};
  }

  TaskTerm cmp(StringRef pred, TaskTerm lhs, TaskTerm rhs, StringRef resTy) {
    Value v = makeOp(b, loc, "task_compare", {lhs.v, rhs.v}, {b.getIndexType()},
                     {b.getNamedAttr("predicate", b.getStringAttr(pred)),
                      b.getNamedAttr("lhs_type", b.getStringAttr(lhs.type)),
                      b.getNamedAttr("rhs_type", b.getStringAttr(rhs.type)),
                      b.getNamedAttr("val_type", b.getStringAttr(resTy))})
                  ->getResult(0);
    return {v, resTy.str()};
  }

  TaskTerm sel(TaskTerm cond, TaskTerm tVal, TaskTerm fVal, StringRef resTy) {
    Value v =
        makeOp(b, loc, "task_select", {cond.v, tVal.v, fVal.v},
               {b.getIndexType()},
               {b.getNamedAttr("cond_type", b.getStringAttr(cond.type)),
                b.getNamedAttr("true_type", b.getStringAttr(tVal.type)),
                b.getNamedAttr("false_type", b.getStringAttr(fVal.type)),
                b.getNamedAttr("val_type", b.getStringAttr(resTy))})
            ->getResult(0);
    return {v, resTy.str()};
  }

  TaskTerm castOp(TaskTerm in, StringRef resTy) {
    Value v = makeOp(b, loc, "task_cast", {in.v}, {b.getIndexType()},
                     {b.getNamedAttr("op_kind", b.getStringAttr("cast")),
                      b.getNamedAttr("in_type", b.getStringAttr(in.type)),
                      b.getNamedAttr("val_type", b.getStringAttr(resTy))})
                  ->getResult(0);
    return {v, resTy.str()};
  }

  TaskTerm unaryOp(StringRef kind, TaskTerm in) {
    Value v = makeOp(b, loc, "task_unary", {in.v}, {b.getIndexType()},
                     {b.getNamedAttr("kind", b.getStringAttr(kind)),
                      b.getNamedAttr("val_type", b.getStringAttr(in.type))})
                  ->getResult(0);
    return {v, in.type};
  }

  TaskTerm splatOp(TaskTerm sc, StringRef vTy) {
    Value sp = makeOp(b, loc, "task_new", {sc.v}, {b.getIndexType()},
                      {b.getNamedAttr("in_type", b.getStringAttr(sc.type)),
                       b.getNamedAttr("val_type", b.getStringAttr(vTy))})
                   ->getResult(0);
    return {sp, vTy.str()};
  }

  void commitOp(TaskTerm val, TaskTerm slice) {
    makeOp(b, loc, "task_commit", {val.v, slice.v}, {},
           {b.getNamedAttr("val_type", b.getStringAttr(val.type)),
            b.getNamedAttr("slice_type", b.getStringAttr(slice.type))});
  }

  TaskTerm scalarSlice(Value view, ArrayRef<TaskTerm> coords, bool admit,
                       StringRef admitTy = "") {
    auto vt = cast<MemRefType>(view.getType());
    std::string elem = printMlirType(vt.getElementType());
    TaskTerm viewTerm = terms.lookup(view);
    SmallVector<Value> ops{viewTerm.v};
    SmallVector<Attribute> sels;
    for (const TaskTerm &c : coords) {
      ops.push_back(c.v);
      sels.push_back(b.getStringAttr("index"));
    }
    std::string slTy = "!weft_kernel.slice<!weft_kernel.encoding<\"" + elem +
                       "\",\"dense\",\"dense." + elem + "\",[]>,[],[]>";
    std::string outTy = admit ? admitTy.str() : slTy;
    Value res =
        makeOp(b, loc, "task_slice", ops, {b.getIndexType()},
               {b.getNamedAttr("view_type", b.getStringAttr(viewTerm.type)),
                b.getNamedAttr("selectors", b.getArrayAttr(sels)),
                b.getNamedAttr("slice_type", b.getStringAttr(slTy)),
                b.getNamedAttr("val_type", b.getStringAttr(outTy)),
                b.getNamedAttr("admit", b.getBoolAttr(admit))})
            ->getResult(0);
    return {res, outTy};
  }

  TaskTerm memberSlice(Value view, TaskTerm row, TaskTerm member, bool owned,
                       bool admit, StringRef admitTy = "") {
    int rank = cast<MemRefType>(view.getType()).getRank();
    SmallVector<TaskTerm, 4> coords(rank);
    TaskTerm cur = member;
    for (int i = rank - 1; i >= 0; --i) {
      if (owned) {
        coords[i] = row;
        continue;
      }
      if (!ownerAxis) {
        coords[i] = (i == rank - 1) ? cur : row;
        continue;
      }
      if (i == *ownerAxis) {
        coords[i] = row;
        continue;
      }
      TaskTerm sz = extents[i];
      coords[i] = bin("mod", cur, sz, "index");
      cur = bin("div", cur, sz, "index");
    }
    return scalarSlice(view, coords, admit, admitTy);
  }

  std::array<TaskTerm, 4> emitPhilox4x32Words(TaskTerm seed, TaskTerm base,
                                              int width, int axisId) {
    std::string vecU64 = regionType("value", "ui64", 1, width, axisId);
    (void)vecU64;
    auto c64 = [&](uint64_t val) {
      return lit(std::to_string(val), "ui64");
    };
    auto b64 = [&](StringRef kind, TaskTerm lhs, TaskTerm rhs) {
      return bin(kind, lhs, rhs, lhs.type);
    };
    TaskTerm mask = c64(UINT32_MAX), bits = c64(32), zero = c64(0);
    TaskTerm a = b64("and", base, mask), rb = b64("shr", base, bits);
    TaskTerm c = b64("and", base, zero), d = c;
    TaskTerm lo = b64("and", seed, mask), hi = b64("shr", seed, bits);
    TaskTerm mulA = c64(0xd2511f53), mulB = c64(0xcd9e8d57);
    TaskTerm incA = c64(0x9e3779b9), incB = c64(0xbb67ae85);
    for (unsigned round = 0; round < 10; ++round) {
      TaskTerm left = b64("mul", a, mulA), right = b64("mul", c, mulB);
      a = b64("xor", b64("xor", b64("shr", right, bits), rb), lo);
      c = b64("xor", b64("xor", b64("shr", left, bits), d), hi);
      rb = b64("and", right, mask);
      d = b64("and", left, mask);
      lo = b64("and", b64("add", lo, incA), mask);
      hi = b64("and", b64("add", hi, incB), mask);
    }
    return {a, rb, c, d};
  }

  TaskTerm emitUniform(TaskTerm seed, TaskTerm position, int dim,
                       int axisId = 2) {
    std::string vecU64 = regionType("value", "ui64", 1, dim, axisId);
    auto c64 = [&](uint64_t val) {
      return lit(std::to_string(val), "ui64");
    };
    auto b64 = [&](StringRef kind, TaskTerm lhs, TaskTerm rhs) {
      return bin(kind, lhs, rhs, lhs.type);
    };
    TaskTerm base = b64("shr", position, c64(2));
    auto [a, rb, c, d] = emitPhilox4x32Words(seed, base, dim, axisId);
    TaskTerm word = b64("and", position, c64(3));
    std::string vecI1 = regionType("value", "i1", 1, dim, axisId);
    for (auto [index, val] :
         {std::pair{2, c}, std::pair{1, rb}, std::pair{0, a}}) {
      TaskTerm active = cmp("eq", word, c64(index), vecI1);
      d = sel(active, val, d, vecU64);
    }
    std::string vecF32 = regionType("value", "f32", 1, dim, axisId);
    TaskTerm f = castOp(b64("shr", d, c64(8)), vecF32);
    return bin("mul", f, lit("0x33800000", "f32"), vecF32);
  }

  TaskTerm emitContiguousUniform(TaskTerm seed, TaskTerm startIdx, int dim,
                                 int axisId = 2) {
    int qDim = dim / 4;
    int qAxisId = axisId + 40;
    std::string vecF32 = regionType("value", "f32", 1, dim, axisId);
    TaskTerm startU64 = castOp(startIdx, "ui64");
    TaskTerm rem = bin("and", startU64, lit("3", "ui64"), "ui64");
    TaskTerm aligned = cmp("eq", rem, lit("0", "ui64"), "i1");
    Operation *ifOp =
        makeOp(b, loc, "task_if", {aligned.v}, {b.getIndexType()},
               {b.getNamedAttr("res_types",
                               b.getArrayAttr({b.getStringAttr(vecF32)}))},
               2);
    Block &thenBb = ifOp->getRegion(0).emplaceBlock();
    Block &elseBb = ifOp->getRegion(1).emplaceBlock();
    {
      OpBuilder::InsertionGuard tg(b);
      b.setInsertionPointToEnd(&thenBb);
      std::string qU64 = regionType("value", "ui64", 1, qDim, qAxisId);
      std::string qF32 = regionType("value", "f32", 1, qDim, qAxisId);
      TaskTerm qStart = bin("shr", startU64, lit("2", "ui64"), "ui64");
      Value qIota =
          makeOp(b, loc, "task_iota", {}, {b.getIndexType()},
                 {b.getNamedAttr("start", b.getI64IntegerAttr(0)),
                  b.getNamedAttr("end", b.getI64IntegerAttr(qDim)),
                  b.getNamedAttr("val_type", b.getStringAttr(qU64))})
              ->getResult(0);
      TaskTerm qBase = bin("add", {qIota, qU64}, qStart, qU64);
      auto words = emitPhilox4x32Words(seed, qBase, qDim, qAxisId);
      TaskTerm scale = lit("0x33800000", "f32");
      TaskTerm eight = lit("8", "ui64");
      std::string idxTy = regionType("value", "ui32", 1, dim, axisId);
      std::string vecI1 = regionType("value", "i1", 1, dim, axisId);
      Value fullIota =
          makeOp(b, loc, "task_iota", {}, {b.getIndexType()},
                 {b.getNamedAttr("start", b.getI64IntegerAttr(0)),
                  b.getNamedAttr("end", b.getI64IntegerAttr(dim)),
                  b.getNamedAttr("val_type", b.getStringAttr(idxTy))})
              ->getResult(0);
      TaskTerm blkIdx = bin("shr", {fullIota, idxTy}, lit("2", "ui32"), idxTy);
      TaskTerm wordIdx = bin("and", {fullIota, idxTy}, lit("3", "ui32"), idxTy);
      std::array<TaskTerm, 4> gathered;
      for (unsigned w = 0; w < 4; ++w) {
        TaskTerm fQuarter = bin(
            "mul", castOp(bin("shr", words[w], eight, qU64), qF32), scale, qF32);
        Value gv =
            makeOp(b, loc, "task_extract", {fQuarter.v, blkIdx.v},
                   {b.getIndexType()},
                   {b.getNamedAttr("selector", b.getStringAttr("gather")),
                    b.getNamedAttr("in_type", b.getStringAttr(qF32)),
                    b.getNamedAttr("point_type", b.getStringAttr(idxTy)),
                    b.getNamedAttr("val_type", b.getStringAttr(vecF32))})
                ->getResult(0);
        gathered[w] = {gv, vecF32};
      }
      TaskTerm out = gathered[3];
      for (int w : {2, 1, 0}) {
        TaskTerm active =
            cmp("eq", wordIdx, lit(std::to_string(w), "ui32"), vecI1);
        out = sel(active, gathered[w], out, vecF32);
      }
      makeOp(b, loc, "task_yield", {out.v}, {},
             {b.getNamedAttr("val_types",
                             b.getArrayAttr({b.getStringAttr(vecF32)}))});
    }
    {
      OpBuilder::InsertionGuard eg(b);
      b.setInsertionPointToEnd(&elseBb);
      std::string vecU64 = regionType("value", "ui64", 1, dim, axisId);
      Value uIota =
          makeOp(b, loc, "task_iota", {}, {b.getIndexType()},
                 {b.getNamedAttr("start", b.getI64IntegerAttr(0)),
                  b.getNamedAttr("end", b.getI64IntegerAttr(dim)),
                  b.getNamedAttr("val_type", b.getStringAttr(vecU64))})
              ->getResult(0);
      TaskTerm pos = bin("add", {uIota, vecU64}, startU64, vecU64);
      TaskTerm out = emitUniform(seed, pos, dim, axisId);
      makeOp(b, loc, "task_yield", {out.v}, {},
             {b.getNamedAttr("val_types",
                             b.getArrayAttr({b.getStringAttr(vecF32)}))});
    }
    return {ifOp->getResult(0), vecF32};
  }

  void collectSubgraph(Operation *root,
                       const llvm::DenseSet<Value> &summarySet,
                       llvm::DenseSet<Operation *> &visited,
                       SmallVectorImpl<Operation *> &order) {
    if (!root ||
        (root->getNumResults() == 1 &&
         summarySet.contains(root->getResult(0))) ||
        !visited.insert(root).second)
      return;
    for (Value opnd : root->getOperands()) {
      if (summarySet.contains(opnd))
        continue;
      if (Operation *def = opnd.getDefiningOp())
        if (def->getBlock() == root->getBlock())
          collectSubgraph(def, summarySet, visited, order);
    }
    order.push_back(root);
  }

  LogicalResult emitVectorPanelOps(
      ArrayRef<Operation *> ops, ArrayRef<Operation *> writes, TaskTerm row,
      ArrayRef<TaskTerm> prefixCoords, std::optional<TaskTerm> startOff,
      TaskTerm activeExt, int dim, int baseColDim, int vecAxisId,
      bool isFullPanel, llvm::DenseMap<Value, TaskTerm> &localMap,
      const llvm::DenseSet<Operation *> *skipReduceOps = nullptr) {
    auto lookup = [&](Value v) -> TaskTerm {
      auto it = localMap.find(v);
      if (it != localMap.end())
        return it->second;
      return terms.lookup(v);
    };

    llvm::DenseMap<Value, TaskTerm> ordinalStarts;
    for (Operation *opPtr : ops) {
      Operation &op = *opPtr;
      StringRef code = op.getName().stripDialect();
      if (code == "write" || code == "write_owner")
        continue;
      if (skipReduceOps && skipReduceOps->contains(opPtr))
        continue;
      if (code == "read_owner") {
        localMap[op.getResult(0)] = memberSlice(
            op.getOperand(0), row, {}, true, true,
            weftSpelling(op.getResult(0).getType()));
        continue;
      }
      if (code == "gather_row") {
        TaskTerm keys = lookup(op.getOperand(1));
        TaskTerm table = lookup(op.getOperand(0));
        std::string kElem = printMlirType(
            cast<MemRefType>(op.getOperand(1).getType()).getElementType());
        std::string kSp = kElem == "i64" ? "si64" : "si32";
        TaskTerm keyVal = scalarSlice(op.getOperand(1), {row}, true, kSp);
        TaskTerm srcIdx = castOp(keyVal, "index");
        std::string dSp = weftSpelling(
            cast<MemRefType>(op.getOperand(0).getType()).getElementType());
        std::string baseSlTy =
            regionType("slice", dSp, 1, baseColDim, vecAxisId);
        if (!startOff) {
          std::string vTy = regionType("value", dSp, 1, baseColDim, vecAxisId);
          Value sl =
              makeOp(b, loc, "task_slice", {table.v, srcIdx.v},
                     {b.getIndexType()},
                     {b.getNamedAttr("view_type", b.getStringAttr(table.type)),
                      b.getNamedAttr("selectors",
                                     b.getArrayAttr({b.getStringAttr("index"),
                                                     b.getStringAttr("all")})),
                      b.getNamedAttr("slice_type", b.getStringAttr(baseSlTy)),
                      b.getNamedAttr("val_type", b.getStringAttr(vTy)),
                      b.getNamedAttr("admit", b.getBoolAttr(true))})
                  ->getResult(0);
          localMap[op.getResult(0)] = {sl, vTy};
        } else {
          Value sl =
              makeOp(b, loc, "task_slice", {table.v, srcIdx.v},
                     {b.getIndexType()},
                     {b.getNamedAttr("view_type", b.getStringAttr(table.type)),
                      b.getNamedAttr("selectors",
                                     b.getArrayAttr({b.getStringAttr("index"),
                                                     b.getStringAttr("all")})),
                      b.getNamedAttr("slice_type", b.getStringAttr(baseSlTy)),
                      b.getNamedAttr("val_type", b.getStringAttr(baseSlTy)),
                      b.getNamedAttr("admit", b.getBoolAttr(false))})
                  ->getResult(0);
          std::string vTy = regionType("value", dSp, 1, dim, vecAxisId);
          SmallVector<Value> subOps{sl, startOff->v};
          if (dim <= 0)
            subOps.push_back(activeExt.v);
          Value val =
              makeOp(b, loc, "task_subview", subOps, {b.getIndexType()},
                     {b.getNamedAttr("static_extent", b.getI64IntegerAttr(dim)),
                      b.getNamedAttr("slice_type", b.getStringAttr(baseSlTy)),
                      b.getNamedAttr("val_type", b.getStringAttr(vTy)),
                      b.getNamedAttr("admit", b.getBoolAttr(true))})
                  ->getResult(0);
          localMap[op.getResult(0)] = {val, vTy};
        }
        (void)keys;
        continue;
      }

      if (code == "read") {
        Value view = op.getOperand(0);
        auto vt = cast<MemRefType>(view.getType());
        unsigned slot = cast<BlockArgument>(view).getArgNumber() - 1;
        std::string eSp = weftSpelling(vt.getElementType());
        std::string vTy = regionType("value", eSp, 1, dim, vecAxisId);
        if (isFullPanel && dim > 0 && slotFacts[slot].zeroInnerStride) {
          SmallVector<TaskTerm, 4> scCoords;
          if (vt.getRank() >= 2)
            scCoords.append(prefixCoords.begin(), prefixCoords.end());
          scCoords.push_back(zeroIdx);
          TaskTerm sc = scalarSlice(view, scCoords, true, eSp);
          localMap[op.getResult(0)] = splatOp(sc, vTy);
          continue;
        }
        TaskTerm viewTerm = lookup(view);
        SmallVector<Value> slOps{viewTerm.v};
        SmallVector<Attribute> sels;
        if (vt.getRank() >= 2) {
          for (const TaskTerm &c : prefixCoords) {
            slOps.push_back(c.v);
            sels.push_back(b.getStringAttr("index"));
          }
        }
        sels.push_back(b.getStringAttr("all"));
        std::string baseSlTy =
            regionType("slice", eSp, 1, baseColDim, vecAxisId);
        if (!startOff) {
          Value val =
              makeOp(b, loc, "task_slice", slOps, {b.getIndexType()},
                     {b.getNamedAttr("view_type",
                                     b.getStringAttr(viewTerm.type)),
                      b.getNamedAttr("selectors", b.getArrayAttr(sels)),
                      b.getNamedAttr("slice_type", b.getStringAttr(baseSlTy)),
                      b.getNamedAttr("val_type", b.getStringAttr(vTy)),
                      b.getNamedAttr("admit", b.getBoolAttr(true))})
                  ->getResult(0);
          localMap[op.getResult(0)] = {val, vTy};
        } else {
          Value sl =
              makeOp(b, loc, "task_slice", slOps, {b.getIndexType()},
                     {b.getNamedAttr("view_type",
                                     b.getStringAttr(viewTerm.type)),
                      b.getNamedAttr("selectors", b.getArrayAttr(sels)),
                      b.getNamedAttr("slice_type", b.getStringAttr(baseSlTy)),
                      b.getNamedAttr("val_type", b.getStringAttr(baseSlTy)),
                      b.getNamedAttr("admit", b.getBoolAttr(false))})
                  ->getResult(0);
          SmallVector<Value> subOps{sl, startOff->v};
          if (dim <= 0)
            subOps.push_back(activeExt.v);
          Value val =
              makeOp(b, loc, "task_subview", subOps, {b.getIndexType()},
                     {b.getNamedAttr("static_extent", b.getI64IntegerAttr(dim)),
                      b.getNamedAttr("slice_type", b.getStringAttr(baseSlTy)),
                      b.getNamedAttr("val_type", b.getStringAttr(vTy)),
                      b.getNamedAttr("admit", b.getBoolAttr(true))})
                  ->getResult(0);
          localMap[op.getResult(0)] = {val, vTy};
        }
        continue;
      }

      if (code == "ordinal") {
        bool wide = cast<RankedTensorType>(op.getResult(0).getType())
                        .getElementType()
                        .isUnsignedInteger(64);
        if (dim <= 0 || !startOff)
          return op.emitError(
              "absolute lane coordinates require static panels");
        TaskTerm off =
            task.hasAttr("cpu.member_tasks")
                ? bin("add", origin, *startOff, "index")
                : bin("add", bin("mul", bin("add", row, origin, "index"),
                                 columns, "index"),
                      *startOff, "index");
        if (wide && dim >= 4 && (dim % 4) == 0) {
          ordinalStarts[op.getResult(0)] = off;
          if (llvm::all_of(op.getResult(0).getUsers(), [](Operation *u) {
                return u->getName().stripDialect() == "uniform";
              }))
            continue;
        }
        std::string elem = wide ? "ui64" : "ui32";
        std::string ixTy = regionType("value", elem, 1, dim, vecAxisId);
        TaskTerm offElem = castOp(off, elem);
        Value lanesV =
            makeOp(b, loc, "task_iota", {}, {b.getIndexType()},
                   {b.getNamedAttr("start", b.getI64IntegerAttr(0)),
                    b.getNamedAttr("end", b.getI64IntegerAttr(dim)),
                    b.getNamedAttr("val_type", b.getStringAttr(ixTy))})
                ->getResult(0);
        TaskTerm idxVec = bin("add", {lanesV, ixTy}, offElem, ixTy);
        localMap[op.getResult(0)] =
            wide ? idxVec
                 : castOp(idxVec,
                          regionType("value", "si32", 1, dim, vecAxisId));
        continue;
      }

      if (code == "scan_sum") {
        TaskTerm in = lookup(op.getOperand(0));
        std::string eTy = weftSpelling(
            cast<ShapedType>(op.getResult(0).getType()).getElementType());
        Operation *scanOp =
            makeOp(b, loc, "task_scan", {in.v}, {b.getIndexType()},
                   {b.getNamedAttr("axis", b.getI64IntegerAttr(0)),
                    b.getNamedAttr("reverse", b.getBoolAttr(false)),
                    b.getNamedAttr("elem_type", b.getStringAttr(eTy)),
                    b.getNamedAttr("val_type", b.getStringAttr(in.type))},
                   1);
        Block &sBb = scanOp->getRegion(0).emplaceBlock();
        Value lhs = sBb.addArgument(b.getIndexType(), loc);
        Value rhs = sBb.addArgument(b.getIndexType(), loc);
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToEnd(&sBb);
          TaskTerm sum = bin("add", {lhs, eTy}, {rhs, eTy}, eTy);
          makeOp(b, loc, "task_scan_yield", {sum.v}, {},
                 {b.getNamedAttr("val_type", b.getStringAttr(eTy))});
        }
        localMap[op.getResult(0)] = {scanOp->getResult(0), in.type};
        continue;
      }

      if (code == "segment_length" || code == "ordered_sum") {
        if (!startOff)
          return op.emitError("vector segment sweep requires panel offsets");
        TaskTerm owner = bin("add", origin, row, "index");
        auto readScalar = [&](Value src, ArrayRef<TaskTerm> coords) {
          auto vt = cast<MemRefType>(src.getType());
          return scalarSlice(src, coords, true,
                             weftSpelling(vt.getElementType()));
        };
        TaskTerm lo = castOp(readScalar(op.getOperand(0), {owner}), "index");
        TaskTerm next = bin("add", owner, oneIdx, "index");
        TaskTerm hi = castOp(readScalar(op.getOperand(0), {next}), "index");
        std::string vTy = regionType("value", "f32", 1, dim, vecAxisId);
        if (code == "segment_length") {
          TaskTerm lenSc = castOp(bin("sub", hi, lo, "index"), "f32");
          localMap[op.getResult(0)] = splatOp(lenSc, vTy);
          continue;
        }
        TaskTerm zVec = splatOp(lit("0.0", "f32"), vTy);
        Operation *segFor = makeOp(
            b, loc, "task_for", {lo.v, hi.v, oneIdx.v, zVec.v},
            {b.getIndexType()},
            {b.getNamedAttr("iter_types",
                            b.getArrayAttr({b.getStringAttr(vTy)}))},
            1);
        Block &sBb = segFor->getRegion(0).emplaceBlock();
        TaskTerm segMem{sBb.addArgument(b.getIndexType(), loc), "index"};
        TaskTerm segCarry{sBb.addArgument(b.getIndexType(), loc), vTy};
        auto &segBb = op.getRegion(0).front();
        localMap[segBb.getArgument(0)] = segMem;
        {
          OpBuilder::InsertionGuard sg(b);
          b.setInsertionPointToEnd(&sBb);
          for (Operation &scOp : segBb) {
            StringRef scCode = scOp.getName().stripDialect();
            if (scCode == "member_load") {
              Value src = scOp.getOperand(0);
              auto vt = cast<MemRefType>(src.getType());
              std::string e = weftSpelling(vt.getElementType());
              if (scOp.getOperands().back() == segBb.getArgument(1)) {
                unsigned slot = cast<BlockArgument>(src).getArgNumber() - 1;
                int fullCols = int(slotFacts[slot].indirectExtents.back());
                TaskTerm viewTerm = lookup(src);
                TaskTerm rCoord = lookup(scOp.getOperand(1));
                if (rCoord.type != "index")
                  rCoord = castOp(rCoord, "index");
                std::string baseSlTy =
                    regionType("slice", e, 1, fullCols, vecAxisId);
                Value sl =
                    makeOp(
                        b, loc, "task_slice", {viewTerm.v, rCoord.v},
                        {b.getIndexType()},
                        {b.getNamedAttr("view_type",
                                        b.getStringAttr(viewTerm.type)),
                         b.getNamedAttr("selectors",
                                        b.getArrayAttr({b.getStringAttr("index"),
                                                        b.getStringAttr("all")})),
                         b.getNamedAttr("slice_type",
                                        b.getStringAttr(baseSlTy)),
                         b.getNamedAttr("val_type", b.getStringAttr(baseSlTy)),
                         b.getNamedAttr("admit", b.getBoolAttr(false))})
                        ->getResult(0);
                std::string ldTy = regionType("value", e, 1, dim, vecAxisId);
                SmallVector<Value, 3> subOps{sl, startOff->v};
                if (dim <= 0)
                  subOps.push_back(activeExt.v);
                Value val =
                    makeOp(b, loc, "task_subview", subOps, {b.getIndexType()},
                           {b.getNamedAttr("static_extent",
                                           b.getI64IntegerAttr(dim)),
                            b.getNamedAttr("slice_type",
                                           b.getStringAttr(baseSlTy)),
                            b.getNamedAttr("val_type", b.getStringAttr(ldTy)),
                            b.getNamedAttr("admit", b.getBoolAttr(true))})
                        ->getResult(0);
                TaskTerm loaded{val, ldTy};
                localMap[scOp.getResult(0)] =
                    (e == "f32") ? loaded : castOp(loaded, vTy);
              } else {
                SmallVector<TaskTerm, 4> coords;
                for (Value ci : scOp.getOperands().drop_front()) {
                  TaskTerm ct = lookup(ci);
                  coords.push_back(ct.type == "index" ? ct
                                                      : castOp(ct, "index"));
                }
                TaskTerm sc = readScalar(src, coords);
                localMap[scOp.getResult(0)] =
                    (sc.type == "f32") ? sc : castOp(sc, "f32");
              }
            } else if (scCode == "term") {
              TaskTerm termVal = lookup(scOp.getOperand(0));
              TaskTerm total = bin("add", segCarry, termVal, vTy);
              makeOp(b, loc, "task_yield", {total.v}, {},
                     {b.getNamedAttr("val_types",
                                     b.getArrayAttr({b.getStringAttr(vTy)}))});
            } else if (scCode == "mul" || scCode == "add") {
              TaskTerm lhs = lookup(scOp.getOperand(0));
              TaskTerm rhs = lookup(scOp.getOperand(1));
              std::string rTy =
                  (lhs.type == vTy || rhs.type == vTy) ? vTy : "f32";
              localMap[scOp.getResult(0)] = bin(scCode, lhs, rhs, rTy);
            } else {
              return scOp.emitError("illegal ordered member operation ")
                     << scCode;
            }
          }
        }
        localMap[op.getResult(0)] = {segFor->getResult(0), vTy};
        continue;
      }

      if (code == "uniform") {
        auto startIt = ordinalStarts.find(op.getOperand(1));
        if (startIt != ordinalStarts.end() && dim >= 4 && (dim % 4) == 0) {
          localMap[op.getResult(0)] = emitContiguousUniform(
              lookup(op.getOperand(0)), startIt->second, dim, vecAxisId);
        } else {
          localMap[op.getResult(0)] = emitUniform(
              lookup(op.getOperand(0)), lookup(op.getOperand(1)), dim,
              vecAxisId);
        }
        continue;
      }

      if (code == "constant") {
        if (auto integer = op.getAttrOfType<IntegerAttr>("value")) {
          std::string t = weftSpelling(integer.getType());
          localMap[op.getResult(0)] =
              lit(t == "i1" ? (integer.getInt() ? "true" : "false")
                            : std::to_string(integer.getInt()),
                  t);
        } else {
          auto flt = op.getAttrOfType<FloatAttr>("value");
          localMap[op.getResult(0)] =
              lit(stripTypeSuffix(flt), printMlirType(flt.getType()));
        }
        continue;
      }

      if (code == "field") {
        auto saved = records.find(op.getOperand(0));
        if (saved == records.end())
          return op.emitError("unbound Weft tuple field");
        localMap[op.getResult(0)] =
            saved->second[op.getAttrOfType<IntegerAttr>("position").getInt()];
        continue;
      }

      Type resType = op.getResult(0).getType();
      std::string outTy =
          isa<RankedTensorType>(resType)
              ? regionType("value",
                           weftSpelling(
                               cast<RankedTensorType>(resType).getElementType()),
                           1, dim, vecAxisId)
              : weftSpelling(resType);

      if (code == "max_num" || code == "sum") {
        TaskTerm in = lookup(op.getOperand(0));
        Value v =
            makeOp(b, loc, "task_reduce", {in.v}, {b.getIndexType()},
                   {b.getNamedAttr("kind", b.getStringAttr(code == "max_num"
                                                               ? "max"
                                                               : "add")),
                    b.getNamedAttr("axis", b.getI64IntegerAttr(0)),
                    b.getNamedAttr("in_type", b.getStringAttr(in.type)),
                    b.getNamedAttr("val_type", b.getStringAttr(outTy))})
                ->getResult(0);
        localMap[op.getResult(0)] = {v, outTy};
      } else if (code == "cast") {
        TaskTerm in = lookup(op.getOperand(0));
        if (!StringRef(in.type).starts_with("!weft_kernel.value<") &&
            StringRef(outTy).starts_with("!weft_kernel.value<")) {
          std::string scOut = weftSpelling(
              cast<RankedTensorType>(resType).getElementType());
          TaskTerm scCast = (in.type == scOut) ? in : castOp(in, scOut);
          localMap[op.getResult(0)] = splatOp(scCast, outTy);
        } else {
          localMap[op.getResult(0)] = castOp(in, outTy);
        }
      } else if (code == "sqrt" || code == "rsqrt" || code == "exp") {
        TaskTerm in = lookup(op.getOperand(0));
        if (!StringRef(in.type).starts_with("!weft_kernel.value<") &&
            StringRef(outTy).starts_with("!weft_kernel.value<"))
          in = splatOp(in, outTy);
        localMap[op.getResult(0)] = unaryOp(code, in);
      } else if (code == "select") {
        localMap[op.getResult(0)] =
            sel(lookup(op.getOperand(0)), lookup(op.getOperand(1)),
                lookup(op.getOperand(2)), outTy);
      } else if (code == "gt" || code == "ne") {
        localMap[op.getResult(0)] =
            cmp(code, lookup(op.getOperand(0)), lookup(op.getOperand(1)), outTy);
      } else if (code == "xor" || code == "and" || code == "shl" ||
                 code == "shr" || code == "maximum" || code == "mul" ||
                 code == "add" || code == "div" || code == "sub") {
        StringRef bKind = code == "maximum" ? StringRef("max") : code;
        localMap[op.getResult(0)] = bin(bKind, lookup(op.getOperand(0)),
                                        lookup(op.getOperand(1)), outTy);
      } else {
        return op.emitError("unsupported Weft task operation ") << code;
      }
    }

    for (Operation *wPtr : writes) {
      Value view = wPtr->getOperand(0);
      auto vt = cast<MemRefType>(view.getType());
      std::string eSp = weftSpelling(vt.getElementType());
      TaskTerm viewTerm = lookup(view);
      SmallVector<Value> slOps{viewTerm.v};
      SmallVector<Attribute> sels;
      if (vt.getRank() >= 2) {
        for (const TaskTerm &c : prefixCoords) {
          slOps.push_back(c.v);
          sels.push_back(b.getStringAttr("index"));
        }
      }
      sels.push_back(b.getStringAttr("all"));
      std::string baseSlTy =
          regionType("slice", eSp, 1, baseColDim, vecAxisId);
      TaskTerm valTerm = lookup(wPtr->getOperand(wPtr->getNumOperands() - 1));
      if (!StringRef(valTerm.type).starts_with("!weft_kernel.value<")) {
        int vDim = startOff ? dim : baseColDim;
        valTerm = splatOp(valTerm,
                          regionType("value", valTerm.type, 1, vDim, vecAxisId));
      }
      if (!startOff) {
        Value sl =
            makeOp(b, loc, "task_slice", slOps, {b.getIndexType()},
                   {b.getNamedAttr("view_type", b.getStringAttr(viewTerm.type)),
                    b.getNamedAttr("selectors", b.getArrayAttr(sels)),
                    b.getNamedAttr("slice_type", b.getStringAttr(baseSlTy)),
                    b.getNamedAttr("val_type", b.getStringAttr(baseSlTy)),
                    b.getNamedAttr("admit", b.getBoolAttr(false))})
                ->getResult(0);
        commitOp(valTerm, {sl, baseSlTy});
      } else {
        Value sl =
            makeOp(b, loc, "task_slice", slOps, {b.getIndexType()},
                   {b.getNamedAttr("view_type", b.getStringAttr(viewTerm.type)),
                    b.getNamedAttr("selectors", b.getArrayAttr(sels)),
                    b.getNamedAttr("slice_type", b.getStringAttr(baseSlTy)),
                    b.getNamedAttr("val_type", b.getStringAttr(baseSlTy)),
                    b.getNamedAttr("admit", b.getBoolAttr(false))})
                ->getResult(0);
        std::string subSlTy = regionType("slice", eSp, 1, dim, vecAxisId);
        SmallVector<Value> subOps{sl, startOff->v};
        if (dim <= 0)
          subOps.push_back(activeExt.v);
        Value win =
            makeOp(b, loc, "task_subview", subOps, {b.getIndexType()},
                   {b.getNamedAttr("static_extent", b.getI64IntegerAttr(dim)),
                    b.getNamedAttr("slice_type", b.getStringAttr(baseSlTy)),
                    b.getNamedAttr("val_type", b.getStringAttr(subSlTy)),
                    b.getNamedAttr("admit", b.getBoolAttr(false))})
                ->getResult(0);
        commitOp(valTerm, {win, subSlTy});
      }
    }
    return success();
  }

  LogicalResult emitScalarSweep(const SweepDesc &sw,
                                ArrayRef<Operation *> ops, TaskTerm row) {
    bool publish = sw.combine == "publish";
    bool moments = sw.combine == "welford";
    bool scalarRed = sw.combine == "sum" || sw.combine == "max_num";
    if (!publish && !moments && !scalarRed && sw.combine != "store")
      return task.emitError("unsupported member sweep algebra");

    SmallVector<TaskTerm, 4> scanInits;
    for (Operation *opPtr : ops) {
      if (opPtr->getName().stripDialect() == "scan_sum") {
        std::string t = weftSpelling(
            cast<ShapedType>(opPtr->getResult(0).getType()).getElementType());
        scanInits.push_back(lit(t == "f32" ? "0.0" : "0", t));
      }
    }
    if (!scanInits.empty() && sw.combine != "store")
      return task.emitError("prefix sweep requires a store consumer");

    SmallVector<Operation *, 4> redOps;
    if (scalarRed) {
      if (!sw.anchors.empty())
        redOps.assign(sw.anchors.begin(), sw.anchors.end());
      else if (sw.anchor)
        redOps.push_back(sw.anchor);
    }

    constexpr int kSegVec = 8;
    auto canVecGatherOp = [&](Operation &segOp, int64_t width) -> bool {
      if (segOp.getName().stripDialect() != "ordered_sum" ||
          segOp.getNumRegions() != 1 || segOp.getRegion(0).empty())
        return false;
      auto &segBb = segOp.getRegion(0).front();
      llvm::DenseSet<Value> vecVals;
      vecVals.insert(segBb.getArgument(0));
      for (Operation &scOp : segBb) {
        StringRef scCode = scOp.getName().stripDialect();
        if (scCode == "term")
          return vecVals.contains(scOp.getOperand(0));
        if (scCode == "member_load") {
          if (scOp.getNumOperands() != 2)
            return false;
          auto vt = dyn_cast<MemRefType>(scOp.getOperand(0).getType());
          auto ba = dyn_cast<BlockArgument>(scOp.getOperand(0));
          if (!vt || vt.getRank() != 1 || !ba)
            return false;
          unsigned slot = ba.getArgNumber() - 1;
          if (slotFacts[slot].indirectExtents.size() != 1 ||
              slotFacts[slot].indirectExtents[0] <= width)
            return false;
          Value cVal = scOp.getOperand(1);
          if (cVal != segBb.getArgument(0) && !vecVals.contains(cVal))
            return false;
          if (cVal != segBb.getArgument(0) &&
              !cVal.getType().isInteger(32) && !cVal.getType().isInteger(64))
            return false;
          vecVals.insert(scOp.getResult(0));
        } else if (scCode == "cast") {
          if (!vecVals.contains(scOp.getOperand(0)))
            return false;
          vecVals.insert(scOp.getResult(0));
        } else if (scCode == "mul" || scCode == "add") {
          if (!scOp.getResult(0).getType().isF32())
            return false;
          vecVals.insert(scOp.getResult(0));
        } else {
          return false;
        }
      }
      return false;
    };

    if (!publish && !moments && !scalarRed && scanInits.empty() &&
        sw.writes.size() == 1) {
      int segCnt = 0;
      Operation *onlySeg = nullptr;
      for (Operation *opPtr : ops) {
        StringRef n = opPtr->getName().stripDialect();
        if (n == "write" || n == "write_owner")
          continue;
        ++segCnt;
        onlySeg = opPtr;
      }
      int64_t segVec = kSegVec;
      if (segCnt == 1 && onlySeg) {
        const bool reassocSeg =
            intentdsl::kir::hasFastMath(onlySeg, arith::FastMathFlags::reassoc);
        if (reassocSeg) {
          if (auto swAttr = task.getAttrOfType<IntegerAttr>("cpu.segment_width"))
            segVec = swAttr.getInt();
        }
      }
      if (segCnt == 1 && onlySeg && canVecGatherOp(*onlySeg, segVec)) {
        Operation &op = *onlySeg;
        const bool reassocSeg =
            intentdsl::kir::hasFastMath(&op, arith::FastMathFlags::reassoc);
        auto &segBb = op.getRegion(0).front();
        const std::string f32SegVecTy = regionType("value", "f32", 1, segVec, 2);
        TaskTerm zeroSegVec = reassocSeg ? splatOp(zeroF32, f32SegVecTy) : TaskTerm{};
        bool flat =
            cast<MemRefType>(
                task.getOperand(
                    task.getAttrOfType<IntegerAttr>("domain_arg").getInt())
                    .getType())
                .getRank() == 1;
        llvm::DenseMap<Value, TaskTerm> hoistedDirectSlices;
        for (Operation &scOp : segBb) {
          if (scOp.getName().stripDialect() == "member_load" &&
              scOp.getOperand(1) == segBb.getArgument(0)) {
            Value view = scOp.getOperand(0);
            if (hoistedDirectSlices.count(view))
              continue;
            auto vt = cast<MemRefType>(view.getType());
            unsigned slot = cast<BlockArgument>(view).getArgNumber() - 1;
            int64_t ext = slotFacts[slot].indirectExtents[0];
            std::string eSp = weftSpelling(vt.getElementType());
            TaskTerm viewTerm = terms.lookup(view);
            std::string slTy = regionType("slice", eSp, 1, ext, 2);
            Value sl =
                makeOp(b, loc, "task_slice", {viewTerm.v}, {b.getIndexType()},
                       {b.getNamedAttr("view_type",
                                       b.getStringAttr(viewTerm.type)),
                        b.getNamedAttr("selectors",
                                       b.getArrayAttr({b.getStringAttr("all")})),
                        b.getNamedAttr("slice_type", b.getStringAttr(slTy)),
                        b.getNamedAttr("val_type", b.getStringAttr(slTy)),
                        b.getNamedAttr("admit", b.getBoolAttr(false))})
                    ->getResult(0);
            hoistedDirectSlices[view] = {sl, slTy};
          }
        }
        auto readScalar = [&](Value src, ArrayRef<TaskTerm> coords) {
          auto vt = cast<MemRefType>(src.getType());
          return scalarSlice(src, coords, true,
                             weftSpelling(vt.getElementType()));
        };
        auto reduceSegVec = [&](TaskTerm vecAcc) -> TaskTerm {
          Value v =
              makeOp(b, loc, "task_reduce", {vecAcc.v}, {b.getIndexType()},
                     {b.getNamedAttr("kind", b.getStringAttr("add")),
                      b.getNamedAttr("axis", b.getI64IntegerAttr(0)),
                      b.getNamedAttr("in_type", b.getStringAttr(vecAcc.type)),
                      b.getNamedAttr("val_type", b.getStringAttr("f32"))})
                  ->getResult(0);
          return {v, "f32"};
        };
        auto evalVecStepInBlock = [&](TaskTerm m0,
                                      TaskTerm colTerm) -> TaskTerm {
          llvm::DenseMap<Value, TaskTerm> vMap;
          for (Operation &scOp : segBb) {
            StringRef scCode = scOp.getName().stripDialect();
            if (scCode == "term")
              return vMap.at(scOp.getOperand(0));
            if (scCode == "member_load") {
              Value view = scOp.getOperand(0);
              auto vt = cast<MemRefType>(view.getType());
              std::string eSp = weftSpelling(vt.getElementType());
              std::string valVecTy = regionType("value", eSp, 1, segVec, 2);
              Value cVal = scOp.getOperand(1);
              if (cVal == segBb.getArgument(0)) {
                TaskTerm baseSl = hoistedDirectSlices.at(view);
                Value ld =
                    makeOp(b, loc, "task_subview", {baseSl.v, m0.v},
                           {b.getIndexType()},
                           {b.getNamedAttr("static_extent",
                                           b.getI64IntegerAttr(segVec)),
                            b.getNamedAttr("slice_type",
                                           b.getStringAttr(baseSl.type)),
                            b.getNamedAttr("val_type",
                                           b.getStringAttr(valVecTy)),
                            b.getNamedAttr("admit", b.getBoolAttr(true))})
                        ->getResult(0);
                vMap[scOp.getResult(0)] = {ld, valVecTy};
              } else {
                TaskTerm idxSigned = vMap.at(cVal);
                std::string uElem =
                    cVal.getType().isInteger(64) ? "ui64" : "ui32";
                std::string uVecTy = regionType("value", uElem, 1, segVec, 2);
                Value idxU =
                    makeOp(b, loc, "task_cast", {idxSigned.v},
                           {b.getIndexType()},
                           {b.getNamedAttr("op_kind", b.getStringAttr("bitcast")),
                            b.getNamedAttr("in_type",
                                           b.getStringAttr(idxSigned.type)),
                            b.getNamedAttr("val_type",
                                           b.getStringAttr(uVecTy))})
                        ->getResult(0);
                // Keep admit in the same block as lookup so Weft lowers it via
                // indexedSnapshotLookup (direct gather) instead of copying the table.
                unsigned slot = cast<BlockArgument>(view).getArgNumber() - 1;
                int64_t ext = slotFacts[slot].indirectExtents[0];
                TaskTerm viewTerm = terms.lookup(view);
                std::string slTy = regionType("slice", eSp, 1, ext, 2);
                std::string tblValTy = regionType("value", eSp, 1, ext, 2);
                Value tblVal =
                    makeOp(b, loc, "task_slice", {viewTerm.v},
                           {b.getIndexType()},
                           {b.getNamedAttr("view_type",
                                           b.getStringAttr(viewTerm.type)),
                            b.getNamedAttr("selectors",
                                           b.getArrayAttr({b.getStringAttr("all")})),
                            b.getNamedAttr("slice_type", b.getStringAttr(slTy)),
                            b.getNamedAttr("val_type", b.getStringAttr(tblValTy)),
                            b.getNamedAttr("admit", b.getBoolAttr(true))})
                        ->getResult(0);
                Value gathered =
                    makeOp(b, loc, "task_lookup", {tblVal, idxU},
                           {b.getIndexType()},
                           {b.getNamedAttr("bounds",
                                           b.getStringAttr("in_bounds")),
                            b.getNamedAttr("table_type",
                                           b.getStringAttr(tblValTy)),
                            b.getNamedAttr("idx_type", b.getStringAttr(uVecTy)),
                            b.getNamedAttr("val_type",
                                           b.getStringAttr(valVecTy))})
                        ->getResult(0);
                vMap[scOp.getResult(0)] = {gathered, valVecTy};
              }
            } else if (scCode == "cast") {
              std::string dt = weftSpelling(scOp.getResult(0).getType());
              std::string dstVecTy = regionType("value", dt, 1, segVec, 2);
              TaskTerm inV = vMap.at(scOp.getOperand(0));
              vMap[scOp.getResult(0)] =
                  inV.type == dstVecTy ? inV : castOp(inV, dstVecTy);
            } else if (scCode == "mul" || scCode == "add") {
              auto getVOp = [&](Value v) -> TaskTerm {
                if (vMap.count(v))
                  return vMap.at(v);
                return v == segBb.getArgument(1) ? colTerm : terms.lookup(v);
              };
              vMap[scOp.getResult(0)] = bin(scCode, getVOp(scOp.getOperand(0)),
                                            getVOp(scOp.getOperand(1)), f32SegVecTy);
            }
          }
          return {};
        };
        auto evalScalarStep = [&](TaskTerm segPt,
                                  TaskTerm colTerm) -> FailureOr<TaskTerm> {
          llvm::DenseMap<Value, TaskTerm> stepMap;
          stepMap[segBb.getArgument(0)] = segPt;
          stepMap[segBb.getArgument(1)] = colTerm;
          auto stepLookup = [&](Value v) -> TaskTerm {
            auto it = stepMap.find(v);
            return it != stepMap.end() ? it->second : terms.lookup(v);
          };
          for (Operation &scOp : segBb) {
            StringRef scCode = scOp.getName().stripDialect();
            if (scCode == "member_load") {
              SmallVector<TaskTerm, 4> coords;
              for (Value ci : scOp.getOperands().drop_front()) {
                TaskTerm ct = stepLookup(ci);
                coords.push_back(ct.type == "index" ? ct : castOp(ct, "index"));
              }
              stepMap[scOp.getResult(0)] = readScalar(scOp.getOperand(0), coords);
            } else if (scCode == "cast") {
              std::string dt = weftSpelling(scOp.getResult(0).getType());
              stepMap[scOp.getResult(0)] =
                  castOp(stepLookup(scOp.getOperand(0)), dt);
            } else if (scCode == "mul" || scCode == "add") {
              stepMap[scOp.getResult(0)] =
                  bin(scCode, stepLookup(scOp.getOperand(0)),
                      stepLookup(scOp.getOperand(1)), "f32");
            } else if (scCode == "term") {
              return stepLookup(scOp.getOperand(0));
            } else {
              return failure();
            }
          }
          return failure();
        };
        auto emitRemAndStore = [&](TaskTerm remLo, TaskTerm hiBnd,
                                   TaskTerm initAcc,
                                   TaskTerm colTerm) -> LogicalResult {
          TaskTerm hasRem = cmp("lt", remLo, hiBnd, "i1");
          ArrayAttr rTy = b.getArrayAttr({b.getStringAttr("f32")});
          Operation *remIf =
              makeOp(b, loc, "task_if", {hasRem.v}, {b.getIndexType()},
                     {b.getNamedAttr("res_types", rTy)}, 2);
          {
            Block &tBb = remIf->getRegion(0).emplaceBlock();
            OpBuilder::InsertionGuard tg(b);
            b.setInsertionPointToEnd(&tBb);
            TaskTerm rawRem = bin("sub", hiBnd, remLo, "index");
            TaskTerm alV = castOp(
                bin("and", castOp(rawRem, "si64"),
                    lit(std::to_string(-segVec), "si64"), "si64"),
                "index");
            TaskTerm stV = bin("add", remLo, alV, "index");
            TaskTerm stepV = idx(segVec);
            TaskTerm afterVecAcc;
            if (reassocSeg) {
              ArrayAttr rvTy = b.getArrayAttr({b.getStringAttr(f32SegVecTy)});
              Operation *rvFor = makeOp(
                  b, loc, "task_for", {remLo.v, stV.v, stepV.v, zeroSegVec.v},
                  {b.getIndexType()},
                  {b.getNamedAttr("iter_types", rvTy)}, 1);
              {
                Block &rvBb = rvFor->getRegion(0).emplaceBlock();
                TaskTerm m0{rvBb.addArgument(b.getIndexType(), loc), "index"};
                TaskTerm accVec{rvBb.addArgument(b.getIndexType(), loc), f32SegVecTy};
                OpBuilder::InsertionGuard rvg(b);
                b.setInsertionPointToEnd(&rvBb);
                TaskTerm pv = evalVecStepInBlock(m0, colTerm);
                TaskTerm nxtVec = bin("add", accVec, pv, f32SegVecTy);
                makeOp(b, loc, "task_yield", {nxtVec.v}, {},
                       {b.getNamedAttr("val_types", rvTy)});
              }
              TaskTerm redSum = reduceSegVec({rvFor->getResult(0), f32SegVecTy});
              afterVecAcc = bin("add", initAcc, redSum, "f32");
            } else {
              Operation *rvFor = makeOp(
                  b, loc, "task_for", {remLo.v, stV.v, stepV.v, initAcc.v},
                  {b.getIndexType()},
                  {b.getNamedAttr("iter_types", rTy)}, 1);
              {
                Block &rvBb = rvFor->getRegion(0).emplaceBlock();
                TaskTerm m0{rvBb.addArgument(b.getIndexType(), loc), "index"};
                TaskTerm accV{rvBb.addArgument(b.getIndexType(), loc), "f32"};
                OpBuilder::InsertionGuard rvg(b);
                b.setInsertionPointToEnd(&rvBb);
                TaskTerm pv = evalVecStepInBlock(m0, colTerm);
                TaskTerm cur = accV;
                for (int ln = 0; ln < segVec; ++ln) {
                  TaskTerm lnIdx = idx(ln);
                  Value lv =
                      makeOp(b, loc, "task_extract", {pv.v, lnIdx.v},
                             {b.getIndexType()},
                             {b.getNamedAttr("selector", b.getStringAttr("index")),
                              b.getNamedAttr("in_type", b.getStringAttr(pv.type)),
                              b.getNamedAttr("point_type", b.getStringAttr("index")),
                              b.getNamedAttr("val_type", b.getStringAttr("f32"))})
                          ->getResult(0);
                  cur = bin("add", cur, {lv, "f32"}, "f32");
                }
                makeOp(b, loc, "task_yield", {cur.v}, {},
                       {b.getNamedAttr("val_types", rTy)});
              }
              afterVecAcc = {rvFor->getResult(0), "f32"};
            }
            Operation *rtFor = makeOp(
                b, loc, "task_for",
                {stV.v, hiBnd.v, oneIdx.v, afterVecAcc.v},
                {b.getIndexType()},
                {b.getNamedAttr("iter_types", rTy)}, 1);
            {
              Block &rtBb = rtFor->getRegion(0).emplaceBlock();
              TaskTerm segMem{rtBb.addArgument(b.getIndexType(), loc), "index"};
              TaskTerm segCarry{rtBb.addArgument(b.getIndexType(), loc), "f32"};
              OpBuilder::InsertionGuard rtg(b);
              b.setInsertionPointToEnd(&rtBb);
              auto sc = evalScalarStep(segMem, colTerm);
              if (failed(sc))
                return failure();
              TaskTerm tot = bin("add", segCarry, *sc, "f32");
              makeOp(b, loc, "task_yield", {tot.v}, {},
                     {b.getNamedAttr("val_types", rTy)});
            }
            makeOp(b, loc, "task_yield", {rtFor->getResult(0)}, {},
                   {b.getNamedAttr("val_types", rTy)});
          }
          {
            Block &eBb = remIf->getRegion(1).emplaceBlock();
            OpBuilder::InsertionGuard eg(b);
            b.setInsertionPointToEnd(&eBb);
            makeOp(b, loc, "task_yield", {initAcc.v}, {},
                   {b.getNamedAttr("val_types", rTy)});
          }
          Operation *wPtr = sw.writes[0];
          StringRef wn = wPtr->getName().stripDialect();
          TaskTerm addr = memberSlice(wPtr->getOperand(0), row, colTerm,
                                      wn == "write_owner", false);
          commitOp({remIf->getResult(0), "f32"}, addr);
          return success();
        };

        if (reassocSeg) {
          TaskTerm stepV = idx(segVec);
          ArrayAttr v1Ty = b.getArrayAttr({b.getStringAttr(f32SegVecTy)});
          ArrayAttr s1Ty = b.getArrayAttr({b.getStringAttr("f32")});
          Operation *m1For =
              makeOp(b, loc, "task_for", {zeroIdx.v, columns.v, oneIdx.v}, {},
                     {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
          {
            Block &m1Bb = m1For->getRegion(0).emplaceBlock();
            TaskTerm mCol{m1Bb.addArgument(b.getIndexType(), loc), "index"};
            OpBuilder::InsertionGuard m1g(b);
            b.setInsertionPointToEnd(&m1Bb);
            TaskTerm own = bin("add", origin, flat ? mCol : row, "index");
            TaskTerm lo = castOp(readScalar(op.getOperand(0), {own}), "index");
            TaskTerm hi = castOp(
                readScalar(op.getOperand(0), {bin("add", own, oneIdx, "index")}),
                "index");
            TaskTerm rawSp = bin("sub", hi, lo, "index");
            TaskTerm span =
                sel(cmp("gt", hi, lo, "i1"), rawSp, zeroIdx, "index");
            TaskTerm alV = castOp(
                bin("and", castOp(span, "si64"),
                    lit(std::to_string(-segVec), "si64"), "si64"),
                "index");
            TaskTerm stV = bin("add", lo, alV, "index");
            Operation *vFor = makeOp(
                b, loc, "task_for", {lo.v, stV.v, stepV.v, zeroSegVec.v},
                {b.getIndexType()}, {b.getNamedAttr("iter_types", v1Ty)}, 1);
            {
              Block &vBb = vFor->getRegion(0).emplaceBlock();
              TaskTerm m0{vBb.addArgument(b.getIndexType(), loc), "index"};
              TaskTerm accVec{vBb.addArgument(b.getIndexType(), loc),
                              f32SegVecTy};
              OpBuilder::InsertionGuard vg(b);
              b.setInsertionPointToEnd(&vBb);
              TaskTerm prod = evalVecStepInBlock(m0, mCol);
              TaskTerm nxtVec = bin("add", accVec, prod, f32SegVecTy);
              makeOp(b, loc, "task_yield", {nxtVec.v}, {},
                     {b.getNamedAttr("val_types", v1Ty)});
            }
            TaskTerm redSum = reduceSegVec({vFor->getResult(0), f32SegVecTy});
            Operation *tFor = makeOp(
                b, loc, "task_for", {stV.v, hi.v, oneIdx.v, redSum.v},
                {b.getIndexType()}, {b.getNamedAttr("iter_types", s1Ty)}, 1);
            {
              Block &tBb = tFor->getRegion(0).emplaceBlock();
              TaskTerm segMem{tBb.addArgument(b.getIndexType(), loc), "index"};
              TaskTerm segCarry{tBb.addArgument(b.getIndexType(), loc), "f32"};
              OpBuilder::InsertionGuard tg(b);
              b.setInsertionPointToEnd(&tBb);
              auto sc = evalScalarStep(segMem, mCol);
              if (failed(sc))
                return failure();
              TaskTerm tot = bin("add", segCarry, *sc, "f32");
              makeOp(b, loc, "task_yield", {tot.v}, {},
                     {b.getNamedAttr("val_types", s1Ty)});
            }
            Operation *wPtr = sw.writes[0];
            StringRef wn = wPtr->getName().stripDialect();
            TaskTerm addr = memberSlice(wPtr->getOperand(0), row, mCol,
                                        wn == "write_owner", false);
            commitOp({tFor->getResult(0), "f32"}, addr);
            makeOp(b, loc, "task_yield", {}, {},
                   {b.getNamedAttr("val_types", b.getArrayAttr({}))});
          }
          return success();
        }

        TaskTerm colAligned2 = castOp(
            bin("and", castOp(columns, "si64"), lit("-2", "si64"), "si64"),
            "index");
        TaskTerm step2 = idx(2);
        TaskTerm stepV = idx(segVec);
        Operation *m2For =
            makeOp(b, loc, "task_for", {zeroIdx.v, colAligned2.v, step2.v}, {},
                   {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
        {
          Block &m2Bb = m2For->getRegion(0).emplaceBlock();
          TaskTerm mBase{m2Bb.addArgument(b.getIndexType(), loc), "index"};
          OpBuilder::InsertionGuard m2g(b);
          b.setInsertionPointToEnd(&m2Bb);
          TaskTerm cols2[2], los[2], his[2], spans[2];
          for (int u = 0; u < 2; ++u) {
            cols2[u] = u == 0 ? mBase : bin("add", mBase, oneIdx, "index");
            TaskTerm own = bin("add", origin, flat ? cols2[u] : row, "index");
            los[u] = (u > 0 && flat)
                         ? his[u - 1]
                         : castOp(readScalar(op.getOperand(0), {own}), "index");
            TaskTerm nxt = bin("add", own, oneIdx, "index");
            his[u] = castOp(readScalar(op.getOperand(0), {nxt}), "index");
            TaskTerm rSp = bin("sub", his[u], los[u], "index");
            spans[u] = sel(cmp("gt", his[u], los[u], "i1"), rSp, zeroIdx, "index");
          }
          TaskTerm minAll =
              sel(cmp("lt", spans[0], spans[1], "i1"), spans[0], spans[1], "index");
          TaskTerm commonV = castOp(
              bin("and", castOp(minAll, "si64"),
                  lit(std::to_string(-segVec), "si64"), "si64"),
              "index");
          TaskTerm rowInitAccs[2];
          {
            ArrayAttr f2Tys =
                b.getArrayAttr({b.getStringAttr("f32"), b.getStringAttr("f32")});
            Operation *v2For = makeOp(
                b, loc, "task_for",
                {zeroIdx.v, commonV.v, stepV.v, zeroF32.v, zeroF32.v},
                {b.getIndexType(), b.getIndexType()},
                {b.getNamedAttr("iter_types", f2Tys)}, 1);
            {
              Block &v2Bb = v2For->getRegion(0).emplaceBlock();
              TaskTerm tOff{v2Bb.addArgument(b.getIndexType(), loc), "index"};
              TaskTerm curAccs[2] = {
                  {v2Bb.addArgument(b.getIndexType(), loc), "f32"},
                  {v2Bb.addArgument(b.getIndexType(), loc), "f32"}};
              OpBuilder::InsertionGuard v2g(b);
              b.setInsertionPointToEnd(&v2Bb);
              TaskTerm prods[2];
              for (int u = 0; u < 2; ++u)
                prods[u] = evalVecStepInBlock(bin("add", los[u], tOff, "index"),
                                              cols2[u]);
              for (int ln = 0; ln < segVec; ++ln) {
                TaskTerm lnIdx = idx(ln);
                for (int u = 0; u < 2; ++u) {
                  Value lv =
                      makeOp(b, loc, "task_extract", {prods[u].v, lnIdx.v},
                             {b.getIndexType()},
                             {b.getNamedAttr("selector", b.getStringAttr("index")),
                              b.getNamedAttr("in_type",
                                             b.getStringAttr(prods[u].type)),
                              b.getNamedAttr("point_type",
                                             b.getStringAttr("index")),
                              b.getNamedAttr("val_type", b.getStringAttr("f32"))})
                          ->getResult(0);
                  curAccs[u] = bin("add", curAccs[u], {lv, "f32"}, "f32");
                }
              }
              makeOp(b, loc, "task_yield", {curAccs[0].v, curAccs[1].v}, {},
                     {b.getNamedAttr("val_types", f2Tys)});
            }
            for (int u = 0; u < 2; ++u)
              rowInitAccs[u] = {v2For->getResult(u), "f32"};
          }
          for (int u = 0; u < 2; ++u) {
            TaskTerm remLo = bin("add", los[u], commonV, "index");
            if (failed(emitRemAndStore(remLo, his[u], rowInitAccs[u], cols2[u])))
              return failure();
          }
          makeOp(b, loc, "task_yield", {}, {},
                 {b.getNamedAttr("val_types", b.getArrayAttr({}))});
        }

        Operation *mTailFor =
            makeOp(b, loc, "task_for", {colAligned2.v, columns.v, oneIdx.v}, {},
                   {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
        {
          Block &mtBb = mTailFor->getRegion(0).emplaceBlock();
          TaskTerm mCol{mtBb.addArgument(b.getIndexType(), loc), "index"};
          OpBuilder::InsertionGuard mtg(b);
          b.setInsertionPointToEnd(&mtBb);
          TaskTerm own = bin("add", origin, flat ? mCol : row, "index");
          TaskTerm lo1 = castOp(readScalar(op.getOperand(0), {own}), "index");
          TaskTerm hi1 = castOp(
              readScalar(op.getOperand(0), {bin("add", own, oneIdx, "index")}),
              "index");
          if (failed(emitRemAndStore(lo1, hi1, zeroF32, mCol)))
            return failure();
          makeOp(b, loc, "task_yield", {}, {},
                 {b.getNamedAttr("val_types", b.getArrayAttr({}))});
        }
        return success();
      }
    }

    Operation *forOp = nullptr;
    TaskTerm memberIdx;
    SmallVector<TaskTerm, 4> iterArgs;
    SmallVector<Attribute, 4> iterTys;
    if (!publish) {
      SmallVector<Value, 6> forOperands{zeroIdx.v, columns.v, oneIdx.v};
      SmallVector<Type, 4> resTypes;
      if (moments) {
        forOperands.append({zeroCount.v, zeroF32.v, zeroF32.v});
        iterTys.append({b.getStringAttr("si32"), b.getStringAttr("f32"),
                        b.getStringAttr("f32")});
        resTypes.append(3, b.getIndexType());
      } else if (scalarRed) {
        Value initV = (sw.combine == "sum") ? zeroF32.v : minusInfF32.v;
        for (size_t k = 0; k < redOps.size(); ++k) {
          forOperands.push_back(initV);
          iterTys.push_back(b.getStringAttr("f32"));
          resTypes.push_back(b.getIndexType());
        }
      } else if (!scanInits.empty()) {
        for (const TaskTerm &c : scanInits) {
          forOperands.push_back(c.v);
          iterTys.push_back(b.getStringAttr(c.type));
          resTypes.push_back(b.getIndexType());
        }
      }
      forOp = makeOp(b, loc, "task_for", forOperands, resTypes,
                     {b.getNamedAttr("iter_types", b.getArrayAttr(iterTys))}, 1);
      Block &fBb = forOp->getRegion(0).emplaceBlock();
      memberIdx = {fBb.addArgument(b.getIndexType(), loc), "index"};
      for (Attribute a : iterTys)
        iterArgs.push_back(
            {fBb.addArgument(b.getIndexType(), loc), cast<StringAttr>(a).str()});
    }

    OpBuilder::InsertionGuard guard(b);
    if (forOp)
      b.setInsertionPointToEnd(&forOp->getRegion(0).front());

    llvm::DenseMap<Value, TaskTerm> localMap;
    auto lookup = [&](Value v) -> TaskTerm {
      auto it = localMap.find(v);
      return it != localMap.end() ? it->second : terms.lookup(v);
    };

    unsigned scanCursor = 0;
    SmallVector<TaskTerm, 4> nextCarries;
    for (Operation *opPtr : ops) {
      Operation &op = *opPtr;
      StringRef n = op.getName().stripDialect();
      if (n == "write" || n == "write_owner")
        continue;
      if (moments && opPtr == sw.anchor)
        continue;

      Value resVal = op.getResult(0);
      Type rawT = resVal.getType();
      if (auto sh = dyn_cast<ShapedType>(rawT))
        rawT = sh.getElementType();
      std::string ty = weftSpelling(rawT);

      if (n == "read" || n == "read_owner") {
        localMap[resVal] = memberSlice(op.getOperand(0), row, memberIdx,
                                       n == "read_owner", true, ty);
      } else if (n == "ordered_sum" || n == "segment_length") {
        bool flat =
            cast<MemRefType>(
                task.getOperand(
                    task.getAttrOfType<IntegerAttr>("domain_arg").getInt())
                    .getType())
                .getRank() == 1;
        TaskTerm owner =
            bin("add", origin, flat ? memberIdx : row, "index");
        auto readScalar = [&](Value src, ArrayRef<TaskTerm> coords) {
          auto vt = cast<MemRefType>(src.getType());
          return scalarSlice(src, coords, true,
                             weftSpelling(vt.getElementType()));
        };
        TaskTerm lo = castOp(readScalar(op.getOperand(0), {owner}), "index");
        TaskTerm next = bin("add", owner, oneIdx, "index");
        TaskTerm hi = castOp(readScalar(op.getOperand(0), {next}), "index");
        if (n == "segment_length") {
          localMap[resVal] = castOp(bin("sub", hi, lo, "index"), "f32");
        } else {
          auto &segBb = op.getRegion(0).front();
          auto emitSegStep = [&](TaskTerm segPt) -> FailureOr<TaskTerm> {
            llvm::DenseMap<Value, TaskTerm> stepMap;
            stepMap[segBb.getArgument(0)] = segPt;
            stepMap[segBb.getArgument(1)] = memberIdx;
            auto stepLookup = [&](Value v) -> TaskTerm {
              auto it = stepMap.find(v);
              return it != stepMap.end() ? it->second : lookup(v);
            };
            for (Operation &scOp : segBb) {
              StringRef scCode = scOp.getName().stripDialect();
              if (scCode == "member_load") {
                SmallVector<TaskTerm, 4> coords;
                for (Value ci : scOp.getOperands().drop_front()) {
                  TaskTerm ct = stepLookup(ci);
                  coords.push_back(ct.type == "index" ? ct
                                                      : castOp(ct, "index"));
                }
                stepMap[scOp.getResult(0)] =
                    readScalar(scOp.getOperand(0), coords);
              } else if (scCode == "cast") {
                std::string dt = weftSpelling(scOp.getResult(0).getType());
                stepMap[scOp.getResult(0)] =
                    castOp(stepLookup(scOp.getOperand(0)), dt);
              } else if (scCode == "mul" || scCode == "add") {
                stepMap[scOp.getResult(0)] =
                    bin(scCode, stepLookup(scOp.getOperand(0)),
                        stepLookup(scOp.getOperand(1)), "f32");
              } else if (scCode == "term") {
                return stepLookup(scOp.getOperand(0));
              } else {
                scOp.emitError("illegal ordered member operation ") << scCode;
                return failure();
              }
            }
            op.emitError("missing ordered_sum terminator");
            return failure();
          };

          TaskTerm rawSpan = bin("sub", hi, lo, "index");
          TaskTerm posSpan = sel(cmp("gt", hi, lo, "i1"), rawSpan, zeroIdx, "index");

          constexpr int kSegVec = 16;
          auto canVecGather = [&]() -> bool {
            llvm::DenseSet<Value> vecVals;
            vecVals.insert(segBb.getArgument(0));
            for (Operation &scOp : segBb) {
              StringRef scCode = scOp.getName().stripDialect();
              if (scCode == "term")
                return vecVals.contains(scOp.getOperand(0));
              if (scCode == "member_load") {
                if (scOp.getNumOperands() != 2)
                  return false;
                auto vt = dyn_cast<MemRefType>(scOp.getOperand(0).getType());
                auto ba = dyn_cast<BlockArgument>(scOp.getOperand(0));
                if (!vt || vt.getRank() != 1 || !ba)
                  return false;
                unsigned slot = ba.getArgNumber() - 1;
                if (slotFacts[slot].indirectExtents.size() != 1 ||
                    slotFacts[slot].indirectExtents[0] <= kSegVec)
                  return false;
                Value cVal = scOp.getOperand(1);
                if (cVal != segBb.getArgument(0))
                  return false;
                vecVals.insert(scOp.getResult(0));
              } else if (scCode == "cast") {
                if (!vecVals.contains(scOp.getOperand(0)))
                  return false;
                vecVals.insert(scOp.getResult(0));
              } else if (scCode == "mul" || scCode == "add") {
                if (!scOp.getResult(0).getType().isF32())
                  return false;
                vecVals.insert(scOp.getResult(0));
              } else {
                return false;
              }
            }
            return false;
          };

          TaskTerm remStart = lo;
          TaskTerm remInit = zeroF32;
          if (canVecGather()) {
            TaskTerm alignedV = castOp(
                bin("and", castOp(posSpan, "si64"),
                    lit(std::to_string(-kSegVec), "si64"), "si64"),
                "index");
            TaskTerm stopV = bin("add", lo, alignedV, "index");
            TaskTerm stepV = idx(kSegVec);
            llvm::DenseMap<Value, TaskTerm> hoistedDirectSlices;
            for (Operation &scOp : segBb) {
              if (scOp.getName().stripDialect() == "member_load" &&
                  scOp.getOperand(1) == segBb.getArgument(0)) {
                Value view = scOp.getOperand(0);
                if (hoistedDirectSlices.count(view))
                  continue;
                auto vt = cast<MemRefType>(view.getType());
                unsigned slot = cast<BlockArgument>(view).getArgNumber() - 1;
                int64_t ext = slotFacts[slot].indirectExtents[0];
                std::string eSp = weftSpelling(vt.getElementType());
                TaskTerm viewTerm = lookup(view);
                std::string slTy = regionType("slice", eSp, 1, ext, 2);
                Value sl = makeOp(
                               b, loc, "task_slice", {viewTerm.v},
                               {b.getIndexType()},
                               {b.getNamedAttr("view_type",
                                               b.getStringAttr(viewTerm.type)),
                                b.getNamedAttr(
                                    "selectors",
                                    b.getArrayAttr({b.getStringAttr("all")})),
                                b.getNamedAttr("slice_type",
                                               b.getStringAttr(slTy)),
                                b.getNamedAttr("val_type",
                                               b.getStringAttr(slTy)),
                                b.getNamedAttr("admit", b.getBoolAttr(false))})
                               ->getResult(0);
                hoistedDirectSlices[view] = {sl, slTy};
              }
            }

            Operation *segVFor = makeOp(
                b, loc, "task_for", {lo.v, stopV.v, stepV.v, zeroF32.v},
                {b.getIndexType()},
                {b.getNamedAttr("iter_types",
                                b.getArrayAttr({b.getStringAttr("f32")}))},
                1);
            Block &svBb = segVFor->getRegion(0).emplaceBlock();
            TaskTerm m0{svBb.addArgument(b.getIndexType(), loc), "index"};
            TaskTerm accV{svBb.addArgument(b.getIndexType(), loc), "f32"};
            {
              OpBuilder::InsertionGuard svg(b);
              b.setInsertionPointToEnd(&svBb);
              llvm::DenseMap<Value, TaskTerm> vMap;
              std::string f32VecTy = regionType("value", "f32", 1, kSegVec, 2);
              TaskTerm prodVec;
              for (Operation &scOp : segBb) {
                StringRef scCode = scOp.getName().stripDialect();
                if (scCode == "term") {
                  prodVec = vMap.at(scOp.getOperand(0));
                  break;
                }
                if (scCode == "member_load") {
                  Value view = scOp.getOperand(0);
                  auto vt = cast<MemRefType>(view.getType());
                  unsigned slot = cast<BlockArgument>(view).getArgNumber() - 1;
                  int64_t ext = slotFacts[slot].indirectExtents[0];
                  std::string eSp = weftSpelling(vt.getElementType());
                  std::string valVecTy =
                      regionType("value", eSp, 1, kSegVec, 2);
                  Value cVal = scOp.getOperand(1);
                  if (cVal == segBb.getArgument(0)) {
                    TaskTerm baseSl = hoistedDirectSlices.at(view);
                    Value ld =
                        makeOp(b, loc, "task_subview", {baseSl.v, m0.v},
                               {b.getIndexType()},
                               {b.getNamedAttr("static_extent",
                                               b.getI64IntegerAttr(kSegVec)),
                                b.getNamedAttr("slice_type",
                                               b.getStringAttr(baseSl.type)),
                                b.getNamedAttr("val_type",
                                               b.getStringAttr(valVecTy)),
                                b.getNamedAttr("admit", b.getBoolAttr(true))})
                            ->getResult(0);
                    vMap[scOp.getResult(0)] = {ld, valVecTy};
                  } else {
                    TaskTerm idxSigned = vMap.at(cVal);
                    std::string uElem =
                        cVal.getType().isInteger(64) ? "ui64" : "ui32";
                    std::string uVecTy =
                        regionType("value", uElem, 1, kSegVec, 2);
                    Value idxU =
                        makeOp(b, loc, "task_cast", {idxSigned.v},
                               {b.getIndexType()},
                               {b.getNamedAttr("op_kind",
                                               b.getStringAttr("bitcast")),
                                b.getNamedAttr(
                                    "in_type",
                                    b.getStringAttr(idxSigned.type)),
                                b.getNamedAttr("val_type",
                                               b.getStringAttr(uVecTy))})
                            ->getResult(0);
                    TaskTerm viewTerm = lookup(view);
                    std::string tblSlTy = regionType("slice", eSp, 1, ext, 2);
                    std::string tblValTy = regionType("value", eSp, 1, ext, 2);
                    Value tblVal =
                        makeOp(b, loc, "task_slice", {viewTerm.v},
                               {b.getIndexType()},
                               {b.getNamedAttr("view_type",
                                               b.getStringAttr(viewTerm.type)),
                                b.getNamedAttr(
                                    "selectors",
                                    b.getArrayAttr({b.getStringAttr("all")})),
                                b.getNamedAttr("slice_type",
                                               b.getStringAttr(tblSlTy)),
                                b.getNamedAttr("val_type",
                                               b.getStringAttr(tblValTy)),
                                b.getNamedAttr("admit", b.getBoolAttr(true))})
                            ->getResult(0);
                    Value gathered =
                        makeOp(b, loc, "task_lookup", {tblVal, idxU},
                               {b.getIndexType()},
                               {b.getNamedAttr("bounds",
                                               b.getStringAttr("in_bounds")),
                                b.getNamedAttr("table_type",
                                               b.getStringAttr(tblValTy)),
                                b.getNamedAttr("idx_type",
                                               b.getStringAttr(uVecTy)),
                                b.getNamedAttr("val_type",
                                               b.getStringAttr(valVecTy))})
                            ->getResult(0);
                    vMap[scOp.getResult(0)] = {gathered, valVecTy};
                  }
                } else if (scCode == "cast") {
                  std::string dt = weftSpelling(scOp.getResult(0).getType());
                  std::string dstVecTy = regionType("value", dt, 1, kSegVec, 2);
                  TaskTerm inV = vMap.at(scOp.getOperand(0));
                  vMap[scOp.getResult(0)] =
                      inV.type == dstVecTy ? inV : castOp(inV, dstVecTy);
                } else if (scCode == "mul" || scCode == "add") {
                  auto getVOp = [&](Value v) -> TaskTerm {
                    if (vMap.count(v))
                      return vMap.at(v);
                    return v == segBb.getArgument(1) ? memberIdx : lookup(v);
                  };
                  vMap[scOp.getResult(0)] =
                      bin(scCode, getVOp(scOp.getOperand(0)),
                          getVOp(scOp.getOperand(1)), f32VecTy);
                }
              }
              TaskTerm curAcc = accV;
              for (int u = 0; u < kSegVec; ++u) {
                TaskTerm uIdx = idx(u);
                Value laneV =
                    makeOp(b, loc, "task_extract", {prodVec.v, uIdx.v},
                           {b.getIndexType()},
                           {b.getNamedAttr("selector",
                                           b.getStringAttr("index")),
                            b.getNamedAttr("in_type",
                                           b.getStringAttr(prodVec.type)),
                            b.getNamedAttr("point_type",
                                           b.getStringAttr("index")),
                            b.getNamedAttr("val_type",
                                           b.getStringAttr("f32"))})
                        ->getResult(0);
                curAcc = bin("add", curAcc, {laneV, "f32"}, "f32");
              }
              makeOp(b, loc, "task_yield", {curAcc.v}, {},
                     {b.getNamedAttr("val_types",
                                     b.getArrayAttr({b.getStringAttr("f32")}))});
            }
            remStart = stopV;
            remInit = {segVFor->getResult(0), "f32"};
          } else {
            TaskTerm aligned8 = castOp(
                bin("and", castOp(posSpan, "si64"), lit("-8", "si64"), "si64"),
                "index");
            TaskTerm stop8 = bin("add", lo, aligned8, "index");
            TaskTerm c8 = idx(8);

            Operation *seg8For = makeOp(
                b, loc, "task_for", {lo.v, stop8.v, c8.v, zeroF32.v},
                {b.getIndexType()},
                {b.getNamedAttr("iter_types",
                                b.getArrayAttr({b.getStringAttr("f32")}))},
                1);
            Block &s8Bb = seg8For->getRegion(0).emplaceBlock();
            TaskTerm m0{s8Bb.addArgument(b.getIndexType(), loc), "index"};
            TaskTerm acc8{s8Bb.addArgument(b.getIndexType(), loc), "f32"};
            {
              OpBuilder::InsertionGuard s8g(b);
              b.setInsertionPointToEnd(&s8Bb);
              TaskTerm tVals[8];
              for (int u = 0; u < 8; ++u) {
                TaskTerm mu = u == 0 ? m0 : bin("add", m0, idx(u), "index");
                auto tuRes = emitSegStep(mu);
                if (failed(tuRes))
                  return failure();
                tVals[u] = *tuRes;
              }
              TaskTerm curAcc = acc8;
              for (int u = 0; u < 8; ++u)
                curAcc = bin("add", curAcc, tVals[u], "f32");
              makeOp(b, loc, "task_yield", {curAcc.v}, {},
                     {b.getNamedAttr("val_types",
                                     b.getArrayAttr({b.getStringAttr("f32")}))});
            }
            remStart = stop8;
            remInit = {seg8For->getResult(0), "f32"};
          }

          Operation *segFor = makeOp(
              b, loc, "task_for", {remStart.v, hi.v, oneIdx.v, remInit.v},
              {b.getIndexType()},
              {b.getNamedAttr("iter_types",
                              b.getArrayAttr({b.getStringAttr("f32")}))},
              1);
          Block &sBb = segFor->getRegion(0).emplaceBlock();
          TaskTerm segMem{sBb.addArgument(b.getIndexType(), loc), "index"};
          TaskTerm segCarry{sBb.addArgument(b.getIndexType(), loc), "f32"};
          {
            OpBuilder::InsertionGuard sg(b);
            b.setInsertionPointToEnd(&sBb);
            auto termSc = emitSegStep(segMem);
            if (failed(termSc))
              return failure();
            TaskTerm total = bin("add", segCarry, *termSc, "f32");
            makeOp(b, loc, "task_yield", {total.v}, {},
                   {b.getNamedAttr("val_types",
                                   b.getArrayAttr({b.getStringAttr("f32")}))});
          }
          localMap[resVal] = {segFor->getResult(0), "f32"};
        }
      } else if (n == "scan_sum") {
        TaskTerm nextC =
            bin("add", iterArgs[scanCursor++], lookup(op.getOperand(0)), ty);
        nextCarries.push_back(nextC);
        localMap[resVal] = nextC;
      } else if (scalarRed && llvm::is_contained(redOps, opPtr)) {
        localMap[resVal] = lookup(op.getOperand(0));
      } else if (n == "field") {
        auto saved = records.find(op.getOperand(0));
        if (saved == records.end())
          return op.emitError("unbound Weft tuple field");
        localMap[resVal] =
            saved->second[op.getAttrOfType<IntegerAttr>("position").getInt()];
      } else if (n == "constant") {
        if (auto integer = op.getAttrOfType<IntegerAttr>("value")) {
          localMap[resVal] =
              lit(ty == "i1" ? (integer.getInt() ? "true" : "false")
                             : std::to_string(integer.getInt()),
                  ty);
        } else {
          localMap[resVal] = lit(stripTypeSuffix(op.getAttr("value")), ty);
        }
      } else if (n == "cast") {
        localMap[resVal] = castOp(lookup(op.getOperand(0)), ty);
      } else if (n == "rsqrt" || n == "exp" || n == "sqrt") {
        localMap[resVal] = unaryOp(n, lookup(op.getOperand(0)));
      } else if (n == "select") {
        localMap[resVal] =
            sel(lookup(op.getOperand(0)), lookup(op.getOperand(1)),
                lookup(op.getOperand(2)), ty);
      } else if (n == "gt" || n == "ne") {
        localMap[resVal] =
            cmp(n, lookup(op.getOperand(0)), lookup(op.getOperand(1)), ty);
      } else if (n == "mul" || n == "add" || n == "div" || n == "sub" ||
                 n == "maximum" || n == "xor" || n == "and" || n == "shl" ||
                 n == "shr") {
        StringRef bKind = n == "maximum" ? StringRef("max") : n;
        localMap[resVal] =
            bin(bKind, lookup(op.getOperand(0)), lookup(op.getOperand(1)), ty);
      } else {
        return op.emitError("unsupported scalar member calculation ") << n;
      }
    }

    for (Operation *wPtr : sw.writes) {
      StringRef wn = wPtr->getName().stripDialect();
      TaskTerm addr = memberSlice(wPtr->getOperand(0), row, memberIdx,
                                  wn == "write_owner", false);
      commitOp(lookup(wPtr->getOperands().back()), addr);
    }

    if (moments) {
      TaskTerm nCnt = iterArgs[0], mean = iterArgs[1], m2 = iterArgs[2];
      TaskTerm v = lookup(sw.anchor->getOperand(0));
      TaskTerm count = bin("add", nCnt, oneCount, "si32");
      TaskTerm delta = bin("sub", v, mean, "f32");
      TaskTerm change = bin("div", delta, castOp(count, "f32"), "f32");
      TaskTerm m = bin("add", mean, change, "f32");
      TaskTerm variance = bin(
          "add", m2,
          bin("mul", bin("mul", delta, castOp(nCnt, "f32"), "f32"), change,
              "f32"),
          "f32");
      TaskTerm first = cmp("eq", nCnt, zeroCount, "i1");
      m = sel(first, v, m, "f32");
      variance = sel(first, zeroF32, variance, "f32");
      makeOp(b, loc, "task_yield", {count.v, m.v, variance.v}, {},
             {b.getNamedAttr(
                 "val_types",
                 b.getArrayAttr({b.getStringAttr("si32"),
                                 b.getStringAttr("f32"),
                                 b.getStringAttr("f32")}))});
      records[sw.anchor->getResult(0)] = {{forOp->getResult(0), "si32"},
                                          {forOp->getResult(1), "f32"},
                                          {forOp->getResult(2), "f32"}};
    } else if (scalarRed) {
      SmallVector<Value, 4> yVals;
      for (size_t k = 0; k < redOps.size(); ++k) {
        TaskTerm acc = iterArgs[k];
        TaskTerm val = lookup(redOps[k]->getResult(0));
        TaskTerm next =
            bin(sw.combine == "sum" ? "add" : "max", acc, val, "f32");
        yVals.push_back(next.v);
      }
      makeOp(b, loc, "task_yield", yVals, {},
             {b.getNamedAttr("val_types", b.getArrayAttr(iterTys))});
      for (size_t k = 0; k < redOps.size(); ++k)
        terms[redOps[k]->getResult(0)] = {forOp->getResult(k), "f32"};
    } else if (!nextCarries.empty()) {
      SmallVector<Value, 4> yVals;
      for (const TaskTerm &c : nextCarries)
        yVals.push_back(c.v);
      makeOp(b, loc, "task_yield", yVals, {},
             {b.getNamedAttr("val_types", b.getArrayAttr(iterTys))});
    } else if (forOp) {
      makeOp(b, loc, "task_yield", {}, {},
             {b.getNamedAttr("val_types", b.getArrayAttr({}))});
    }
    return success();
  }
};
bool dependsOn(Value v, Value target) {
  if (v == target)
    return true;
  if (Operation *def = v.getDefiningOp()) {
    for (Value opnd : def->getOperands())
      if (dependsOn(opnd, target))
        return true;
  }
  return false;
}

bool canVectorizeSegmentSweep(Operation &task, ArrayRef<Operation *> ops) {
  int64_t domArg = task.getAttrOfType<IntegerAttr>("domain_arg").getInt();
  if (cast<MemRefType>(task.getOperand(domArg).getType()).getRank() < 2)
    return false;
  for (Operation *opPtr : ops) {
    if (opPtr->getName().stripDialect() != "ordered_sum")
      continue;
    if (opPtr->getNumRegions() != 1 || opPtr->getRegion(0).empty())
      return false;
    Block &segBb = opPtr->getRegion(0).front();
    if (segBb.getNumArguments() != 2)
      return false;
    Value colArg = segBb.getArgument(1);
    for (Operation *user : colArg.getUsers())
      if (user->getBlock() != &segBb ||
          user->getName().stripDialect() != "member_load")
        return false;
    for (Operation &scOp : segBb) {
      if (scOp.getName().stripDialect() != "member_load")
        continue;
      auto vt = cast<MemRefType>(scOp.getOperand(0).getType());
      auto coords = scOp.getOperands().drop_front();
      if (coords.empty())
        return false;
      if (coords.back() == colArg) {
        if (vt.getRank() != 2)
          return false;
        for (Value c : coords.drop_back())
          if (dependsOn(c, colArg))
            return false;
      } else {
        for (Value c : coords)
          if (dependsOn(c, colArg))
            return false;
      }
    }
  }
  return true;
}
} // namespace

LogicalResult lowerWeftRow(func::FuncOp fn, const intentdsl::CallFacts &facts,
                           bool grouped, int64_t tailVecWidth,
                           int64_t privateStackBytes) {
  auto &task = fn.front().front();
  Location loc = fn.getLoc();
  OpBuilder b(&task);

  auto abiAttr = fn->getAttrOfType<StringAttr>("kir.abi");
  auto parsedAbi = llvm::json::parse(abiAttr.getValue());
  if (!parsedAbi)
    return fn.emitError() << llvm::toString(parsedAbi.takeError());
  const auto &abi = *parsedAbi->getAsArray();

  auto &bb = task.getRegion(0).front();
  SmallVector<SlotFacts> slotFacts(fn.getNumArguments());
  for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
    if (auto view = dyn_cast<MemRefType>(fn.getArgument(i).getType()))
      slotFacts[i].rank = view.getRank();
  }

  auto factSlot = [&](unsigned slot) -> unsigned {
    if (auto src = fn.getArgAttrOfType<IntegerAttr>(slot, "kir.source"))
      return unsigned(src.getInt());
    return slot;
  };

  bool absolute = false;
  bool badFacts = false;
  task.walk([&](Operation *o) {
    if (badFacts)
      return;
    absolute |= o->getName().getStringRef() == "cpu.ordinal";
    StringRef n = o->getName().stripDialect();
    if (n == "write_offsets" || n == "fill_ranges") {
      for (Value v : o->getOperands().take_front(n == "write_offsets" ? 2 : 1)) {
        unsigned i = cast<BlockArgument>(v).getArgNumber() - 1;
        auto *shape =
            facts.arguments[factSlot(i)].getAsObject()->getArray("shape");
        slotFacts[i].indirectExtents = {*shape->front().getAsInteger()};
      }
    }
    if (n == "reserve_tickets" || n == "distribute_tickets") {
      SmallVector<Value, 2> shared;
      if (n == "reserve_tickets")
        shared.push_back(o->getOperand(0));
      else {
        shared.push_back(o->getOperand(1));
        shared.push_back(o->getOperand(3));
        absolute = true;
      }
      for (Value v : shared) {
        unsigned i = cast<BlockArgument>(v).getArgNumber() - 1;
        auto *shape =
            facts.arguments[factSlot(i)].getAsObject()->getArray("shape");
        slotFacts[i].indirectExtents = {*shape->front().getAsInteger()};
      }
    }
    if (n == "pack_members" || n == "prefix_scatter") {
      unsigned slot =
          cast<BlockArgument>(o->getOperand(n == "pack_members" ? 2 : 1))
              .getArgNumber() -
          1;
      slotFacts[slot].taskIndices = true;
    }
    if (n == "ordered_sum" || n == "segment_length" || n == "member_load" ||
        n == "count_keys" || n == "column_sum" || n == "atomic_add" ||
        n == "unique_put" || n == "index_read") {
      auto argument = dyn_cast<BlockArgument>(o->getOperand(0));
      if (!argument || argument.getOwner() != &bb || !argument.getArgNumber()) {
        o->emitError("segment read requires a captured input");
        badFacts = true;
        return;
      }
      unsigned slot = argument.getArgNumber() - 1;
      auto *record = facts.arguments[factSlot(slot)].getAsObject();
      auto *shape = record ? record->getArray("shape") : nullptr;
      SmallVector<int64_t, 4> dims;
      if (shape)
        for (auto &d : *shape)
          if (auto ext = d.getAsInteger())
            dims.push_back(*ext);
      if (dims.size() != cast<MemRefType>(argument.getType()).getRank()) {
        o->emitError("missing indexed input extents");
        badFacts = true;
        return;
      }
      slotFacts[slot].indirectExtents = std::move(dims);
      absolute |= n == "ordered_sum" || n == "segment_length" ||
                  n == "member_load" || n == "count_keys";
    }
    if (o->getName().getStringRef() == "cpu.gather_row") {
      unsigned table = cast<BlockArgument>(o->getOperand(0)).getArgNumber() - 1;
      unsigned ids = cast<BlockArgument>(o->getOperand(1)).getArgNumber() - 1;
      auto *slot = facts.arguments[factSlot(table)].getAsObject();
      auto *shape = slot ? slot->getArray("shape") : nullptr;
      if (!shape || shape->size() != 2 || !shape->front().getAsInteger()) {
        o->emitError("missing indexed table height");
        badFacts = true;
        return;
      }
      slotFacts[table].tableRows = *shape->front().getAsInteger();
      slotFacts[ids].taskIndices = true;
    }
    if (o->hasAttr("axes") && o->getNumOperands() > 0) {
      if (auto ba = dyn_cast<BlockArgument>(o->getOperand(0)))
        if (ba.getOwner() == &bb && ba.getArgNumber() > 0)
          slotFacts[ba.getArgNumber() - 1].swapAxes = true;
    }
  });
  if (badFacts)
    return failure();

  std::optional<int64_t> ownerAxis;
  if (auto oa = fn->getAttrOfType<IntegerAttr>("kir.owner_axis"))
    ownerAxis = oa.getInt();
  for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
    if (!isa<MemRefType>(fn.getArgument(i).getType()) ||
        slotFacts[i].swapAxes || fn.getArgAttr(i, "kir.owner_projection"))
      continue;
    if (auto mapAttr = fn.getArgAttrOfType<AffineMapAttr>(i, "kir.indexing")) {
      AffineMap m = mapAttr.getValue();
      if (m.getNumDims() >= 2 &&
          llvm::none_of(m.getResults(), [&](AffineExpr e) {
            return e.isFunctionOfDim(m.getNumDims() - 1);
          }))
        slotFacts[i].zeroInnerStride = true;
    }
  }
  StringRef walkEarly = task.getAttrOfType<StringAttr>("cpu.walk").getValue();
  if (!ownerAxis && walkEarly != "tile" && walkEarly != "destination") {
    bool hasPublicAbi = fn->hasAttr("kir.public_abi");
    for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
      if (!isa<MemRefType>(fn.getArgument(i).getType()) ||
          slotFacts[i].swapAxes || slotFacts[i].rank > 2)
        continue;
      SmallVector<int64_t, 4> dims, st;
      bool haveLayout = false;
      if (fn.getArgAttr(i, "kir.indexing")) {
        auto pShape = intentdsl::projectedLayout(fn, i, facts, true);
        auto pStrides = intentdsl::projectedLayout(fn, i, facts, false);
        if (pShape && pStrides && pShape.size() == pStrides.size()) {
          dims.assign(pShape.asArrayRef().begin(), pShape.asArrayRef().end());
          st.assign(pStrides.asArrayRef().begin(), pStrides.asArrayRef().end());
          haveLayout = true;
        }
      } else if (!hasPublicAbi && i < facts.arguments.size()) {
        auto *a = facts.arguments[i].getAsObject();
        auto *shape = a ? a->getArray("shape") : nullptr;
        auto *strides = a ? a->getArray("strides") : nullptr;
        if (shape && strides && shape->size() == strides->size()) {
          haveLayout = true;
          for (size_t k = 0; k < shape->size(); ++k) {
            auto n = (*shape)[k].getAsInteger(),
                 s = (*strides)[k].getAsInteger();
            if (!n || !s) {
              haveLayout = false;
              break;
            }
            dims.push_back(*n);
            st.push_back(*s);
          }
        }
      }
      bool dense = haveLayout;
      int64_t step = 1;
      if (dense) {
        for (int k = int(dims.size()) - 1; k >= 0; --k) {
          if (dims[k] <= 0 || st[k] != step || step > INT64_MAX / dims[k]) {
            dense = false;
            break;
          }
          step *= dims[k];
        }
      }
      slotFacts[i].contiguous = dense;
    }
  }
  int64_t domainArg = task.getAttrOfType<IntegerAttr>("domain_arg").getInt();
  int64_t rankDomain = slotFacts[domainArg].rank;
  bool ownedKeys = task.hasAttr("cpu.key_extent");

  SmallVector<Attribute> shapeSyms{b.getStringAttr("T"), b.getStringAttr("N"),
                                   b.getStringAttr("C")};
  if (absolute)
    shapeSyms.push_back(b.getStringAttr("O"));
  if (ownedKeys) {
    shapeSyms.push_back(b.getStringAttr("KL"));
    shapeSyms.push_back(b.getStringAttr("KH"));
  }
  if (ownerAxis) {
    for (int64_t j = 0; j < rankDomain; ++j)
      shapeSyms.push_back(b.getStringAttr("D" + std::to_string(j)));
  }

  // Detect private_bins output buffer so arg_access uses "readwrite".
  llvm::DenseSet<Value> privateBinOutputs;
  for (Operation &op : bb.without_terminator()) {
    if (op.getName().stripDialect() == "count_keys") {
      for (Operation *user : op.getUsers())
        if (user->getName().stripDialect() == "write")
          privateBinOutputs.insert(user->getOperand(0));
    }
  }

  SmallVector<Attribute> viewTypes, argAccess;
  SmallVector<int64_t> argAliases, stridedSlots, contiguousSlots, taskRanks;
  SmallVector<std::string> viewTypeStrs;
  for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
    Value captureArg = bb.getArgument(i + 1);
    const auto *a = abi[i].getAsObject();
    std::string elem = a->getString("dtype")->str();
    if (elem == "i32")
      elem = "si32";
    else if (elem == "i64")
      elem = "si64";
    else if (elem == "u64")
      elem = "ui64";
    int rank = std::min<size_t>(2, a->getArray("dims")->size());
    std::string t = rank ? regionType("view", elem, rank) : elem;
    if (slotFacts[i].tableRows) {
      t.replace(t.find("[-1,-2]"), 7,
                "[" + std::to_string(*slotFacts[i].tableRows) + ",-2]");
    }
    if (slotFacts[i].taskIndices)
      t.replace(t.find("[-2],[2]"), 8, "[-1],[1]");
    if (!slotFacts[i].indirectExtents.empty()) {
      SmallVector<std::string, 4> bound;
      for (int64_t d : slotFacts[i].indirectExtents)
        bound.push_back(std::to_string(d));
      auto at = t.rfind(">,[") + 2;
      t.replace(at, t.find(']', at) - at + 1, "[" + llvm::join(bound, ",") + "]");
    }
    if (ownerAxis && rank) {
      int fullRank = a->getArray("dims")->size();
      int d0SymIdx = 3 + (absolute ? 1 : 0) + (ownedKeys ? 2 : 0);
      SmallVector<std::string, 4> shape, axes;
      for (int j = 0; j < fullRank; ++j) {
        shape.push_back(j == *ownerAxis || fullRank == 1
                            ? "-1"
                            : std::to_string(-(1 + d0SymIdx + j)));
        axes.push_back(std::to_string(j + 1));
      }
      t = "!weft_kernel.view<!weft_kernel.encoding<\"" + elem +
          "\",\"dense\",\"dense." + elem + "\",[]>,[" + llvm::join(shape, ",") +
          "],[" + llvm::join(axes, ",") + "]>";
    }
    viewTypes.push_back(b.getStringAttr(t));
    viewTypeStrs.push_back(t);

    bool initBins = privateBinOutputs.contains(captureArg);
    auto dir = a->getString("direction");
    argAccess.push_back(
        b.getStringAttr(!rank          ? "none"
                        : dir == "Out" ? (initBins ? "readwrite" : "write")
                        : dir == "InOut" ? "readwrite"
                                         : "read"));
    argAliases.push_back(rank ? 0 : -1);
    if (rank) {
      if (slotFacts[i].contiguous)
        contiguousSlots.push_back(int64_t(i));
      else
        stridedSlots.push_back(int64_t(i));
    }
    if (ownerAxis)
      taskRanks.push_back(slotFacts[i].rank);
  }

  auto rowsAttr = task.getAttrOfType<ArrayAttr>("rows");
  OperationState progSt(loc, "weft_host.row_program");
  progSt.addAttribute("rows", rowsAttr);
  progSt.addAttribute("shape_parameters", b.getArrayAttr(shapeSyms));
  progSt.addAttribute("contiguous_slots",
                      b.getDenseI64ArrayAttr(contiguousSlots));
  progSt.addAttribute("task_ranks", b.getDenseI64ArrayAttr(taskRanks));
  progSt.addRegion();
  progSt.addRegion();
  Operation *prog = b.create(progSt);

  // ---------------- Region 0: task.weftir ----------------
  Block &tOuter = prog->getRegion(0).emplaceBlock();
  b.setInsertionPointToEnd(&tOuter);
  Operation *tk = makeOp(
      b, loc, "task_kernel", {}, {},
      {b.getNamedAttr("sym_name", b.getStringAttr("row_task")),
       b.getNamedAttr("source", b.getStringAttr("CPU row interval")),
       b.getNamedAttr("view_types", b.getArrayAttr(viewTypes)),
       b.getNamedAttr("arg_access", b.getArrayAttr(argAccess)),
       b.getNamedAttr("arg_aliases", b.getDenseI64ArrayAttr(argAliases)),
       b.getNamedAttr("shape_symbols", b.getArrayAttr(shapeSyms)),
       b.getNamedAttr("strided_slots", b.getDenseI64ArrayAttr(stridedSlots))},
      1);
  makeOp(b, loc, "end");

  Block &tkBb = tk->getRegion(0).emplaceBlock();
  RowTaskBuilder rb{fn, task, b, loc, slotFacts};
  rb.ownerAxis = ownerAxis;
  rb.ownedKeys = ownedKeys;
  for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
    Value bArg = tkBb.addArgument(b.getIndexType(), loc);
    rb.terms[bb.getArgument(i + 1)] = {bArg, viewTypeStrs[i]};
  }
  rb.extent = {tkBb.addArgument(b.getIndexType(), loc), "index"};
  rb.columns = {tkBb.addArgument(b.getIndexType(), loc), "index"};
  rb.panelWidth = {tkBb.addArgument(b.getIndexType(), loc), "index"};
  if (absolute)
    rb.origin = {tkBb.addArgument(b.getIndexType(), loc), "index"};
  if (ownedKeys) {
    rb.keyLo = {tkBb.addArgument(b.getIndexType(), loc), "index"};
    rb.keyHi = {tkBb.addArgument(b.getIndexType(), loc), "index"};
  }
  if (ownerAxis) {
    for (int64_t j = 0; j < rankDomain; ++j)
      rb.extents.push_back({tkBb.addArgument(b.getIndexType(), loc), "index"});
  }
  tkBb.addArgument(b.getIndexType(), loc); // root_domain

  b.setInsertionPointToEnd(&tkBb);
  rb.zeroIdx = rb.idx(0);
  rb.oneIdx = rb.idx(1);
  rb.zeroCount = rb.lit("0", "si32");
  rb.oneCount = rb.lit("1", "si32");
  rb.zeroF32 = rb.lit("0.0", "f32");
  rb.minusInfF32 = rb.lit("0xFF800000", "f32");

  int64_t firstCols =
      cast<DictionaryAttr>(rowsAttr[0]).getAs<IntegerAttr>("columns").getInt();
  bool whole = (firstCols == 0);
  int64_t fixed = firstCols;
  if (!llvm::all_of(rowsAttr, [&](Attribute row) {
        return cast<DictionaryAttr>(row)
                   .getAs<IntegerAttr>("columns")
                   .getInt() == fixed;
      }))
    fixed = 0;
  if (fixed > 0)
    rb.panelWidth = rb.idx(fixed);

  bool vecOwnerInner = false;
  if (ownerAxis && *ownerAxis >= 0 && *ownerAxis < rankDomain - 1 && !whole &&
      fixed > 0) {
    vecOwnerInner = true;
    for (Operation &op : bb.without_terminator()) {
      StringRef n = op.getName().stripDialect();
      if (n == "read" || n == "write") {
        auto vt = cast<MemRefType>(op.getOperand(0).getType());
        if (vt.getRank() != rankDomain) {
          vecOwnerInner = false;
          break;
        }
      } else if (n == "gather_row" || n == "ordered_sum" ||
                 n == "segment_length" || n == "scan_sum" ||
                 n == "count_keys" || n == "column_sum" || n == "ordinal" ||
                 n == "uniform") {
        vecOwnerInner = false;
        break;
      }
    }
  }
  bool scalarMembers = ownerAxis.has_value() && !vecOwnerInner;

  TaskTerm panelCount, tailStart;
  if (!whole) {
    panelCount = rb.bin("div", rb.columns, rb.panelWidth, "index");
    tailStart = rb.bin("mul", panelCount, rb.panelWidth, "index");
  }

  StringRef walk = task.getAttrOfType<StringAttr>("cpu.walk").getValue();
  bool isTransposeTile =
      (walk == "tile" && !bb.empty() &&
       bb.front().getName().getStringRef() == "cpu.transpose_tile");

  if (intentdsl::cpu::reservations(bb)) {
    int64_t bins = task.getAttrOfType<IntegerAttr>("cpu.batch_extent").getInt();
    TaskTerm size = rb.idx(bins);
    TaskTerm oneI32 = rb.lit("1", "si32"), zeroI32 = rb.lit("0", "si32");
    TaskTerm trueMask = rb.cmp("eq", oneI32, oneI32, "i1");
    Value ptrVal =
        makeOp(b, loc, "task_local_alloc", {}, {b.getIndexType()},
               {b.getNamedAttr("elements", b.getI64IntegerAttr(bins)),
                b.getNamedAttr("alignment", b.getI64IntegerAttr(64)),
                b.getNamedAttr("ptr_type",
                               b.getStringAttr("!weft_kernel.pointer<si32>"))})
            ->getResult(0);
    auto readLocal = [&](TaskTerm k) -> TaskTerm {
      TaskTerm off = rb.castOp(k, "si64");
      Value ld = makeOp(b, loc, "task_ptr_load",
                        {ptrVal, off.v, trueMask.v, zeroI32.v},
                        {b.getIndexType()},
                        {b.getNamedAttr("ptr_type",
                                        b.getStringAttr(
                                            "!weft_kernel.pointer<si32>")),
                         b.getNamedAttr("val_type", b.getStringAttr("si32"))})
                     ->getResult(0);
      return {ld, "si32"};
    };
    auto storeLocal = [&](TaskTerm k, TaskTerm val) {
      TaskTerm off = rb.castOp(k, "si64");
      makeOp(b, loc, "task_ptr_store", {ptrVal, off.v, trueMask.v, val.v}, {},
             {b.getNamedAttr("ptr_type",
                             b.getStringAttr("!weft_kernel.pointer<si32>")),
              b.getNamedAttr("val_type", b.getStringAttr("si32"))});
    };

    for (Operation &stage : bb.without_terminator()) {
      StringRef n = stage.getName().stripDialect();
      if (n == "tally_keys") {
        Operation *zFor = makeOp(
            b, loc, "task_for", {rb.zeroIdx.v, size.v, rb.oneIdx.v}, {},
            {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
        Block &zBb = zFor->getRegion(0).emplaceBlock();
        Value k0 = zBb.addArgument(b.getIndexType(), loc);
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToEnd(&zBb);
          storeLocal(rb.castOp({k0, "index"}, "si32"), zeroI32);
          makeOp(b, loc, "task_yield", {}, {},
                 {b.getNamedAttr("val_types", b.getArrayAttr({}))});
        }
      }
      if (n == "reserve_tickets") {
        Operation *rFor = makeOp(
            b, loc, "task_for", {rb.zeroIdx.v, size.v, rb.oneIdx.v}, {},
            {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
        Block &rBb = rFor->getRegion(0).emplaceBlock();
        Value k1 = rBb.addArgument(b.getIndexType(), loc);
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToEnd(&rBb);
          TaskTerm kTerm{k1, "index"};
          TaskTerm offI32 = rb.castOp(kTerm, "si32");
          TaskTerm count = readLocal(offI32);
          TaskTerm used = rb.cmp("ne", count, zeroI32, "i1");
          Operation *ifOp =
              makeOp(b, loc, "task_if", {used.v}, {},
                     {b.getNamedAttr("res_types", b.getArrayAttr({}))}, 2);
          Block &tBb = ifOp->getRegion(0).emplaceBlock();
          {
            OpBuilder::InsertionGuard ig(b);
            b.setInsertionPointToEnd(&tBb);
            TaskTerm sl = rb.scalarSlice(stage.getOperand(0), {kTerm}, false);
            Value baseV =
                makeOp(b, loc, "task_atomic_rmw", {count.v, sl.v},
                       {b.getIndexType()},
                       {b.getNamedAttr("kind", b.getStringAttr("add")),
                        b.getNamedAttr("ordering", b.getStringAttr("relaxed")),
                        b.getNamedAttr("val_type", b.getStringAttr("si32")),
                        b.getNamedAttr("slice_type", b.getStringAttr(sl.type))})
                    ->getResult(0);
            storeLocal(offI32, {baseV, "si32"});
            makeOp(b, loc, "task_yield", {}, {},
                   {b.getNamedAttr("val_types", b.getArrayAttr({}))});
          }
          makeOp(b, loc, "task_yield", {}, {},
                 {b.getNamedAttr("val_types", b.getArrayAttr({}))});
        }
        continue;
      }

      Operation *rFor = makeOp(
          b, loc, "task_for", {rb.zeroIdx.v, rb.extent.v, rb.oneIdx.v}, {},
          {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
      Block &rBb = rFor->getRegion(0).emplaceBlock();
      TaskTerm rIv{rBb.addArgument(b.getIndexType(), loc), "index"};
      {
        OpBuilder::InsertionGuard rg(b);
        b.setInsertionPointToEnd(&rBb);
        Operation *jFor = makeOp(
            b, loc, "task_for", {rb.zeroIdx.v, rb.columns.v, rb.oneIdx.v}, {},
            {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
        Block &jBb = jFor->getRegion(0).emplaceBlock();
        TaskTerm jIv{jBb.addArgument(b.getIndexType(), loc), "index"};
        {
          OpBuilder::InsertionGuard jg(b);
          b.setInsertionPointToEnd(&jBb);
          TaskTerm key =
              rb.scalarSlice(stage.getOperand(0), {rIv, jIv}, true, "si32");
          TaskTerm old = readLocal(key);
          if (n == "distribute_tickets") {
            TaskTerm off = rb.scalarSlice(
                stage.getOperand(1), {rb.castOp(key, "index")}, true, "si32");
            TaskTerm pos = rb.castOp(rb.bin("add", old, off, "si32"), "index");
            TaskTerm global = rb.bin(
                "mul", rb.bin("add", rIv, rb.origin, "index"), rb.columns,
                "index");
            TaskTerm outVal =
                rb.castOp(rb.bin("add", global, jIv, "index"), "si32");
            rb.commitOp(outVal, rb.scalarSlice(stage.getOperand(3), {pos}, false));
          }
          storeLocal(key, rb.bin("add", old, oneI32, "si32"));
          makeOp(b, loc, "task_yield", {}, {},
                 {b.getNamedAttr("val_types", b.getArrayAttr({}))});
        }
        makeOp(b, loc, "task_yield", {}, {},
               {b.getNamedAttr("val_types", b.getArrayAttr({}))});
      }
    }
  } else {
    SmallVector<Operation *> allOps, writes, ownedWrites;
    bool hasSpecialSweep = false;
    for (Operation &op : bb.without_terminator()) {
      allOps.push_back(&op);
      StringRef n = op.getName().stripDialect();
      if (n == "write")
        writes.push_back(&op);
      else if (n == "write_owner")
        ownedWrites.push_back(&op);
      else if (n == "ordered_sum" || n == "segment_length" ||
               n == "scan_sum" || n == "count_keys" || n == "column_sum" ||
               n == "gather_row")
        hasSpecialSweep = true;
    }

    auto isReductionOp = [](Operation *op) {
      StringRef n = op->getName().stripDialect();
      return n == "sum" || n == "max_num" || n == "welford";
    };

    SmallVector<SmallVector<Operation *, 4>, 8> redStages;
    llvm::DenseSet<Operation *> scheduledReds;
    if (!isTransposeTile && !intentdsl::cpu::atomicMembers(bb) &&
        !intentdsl::cpu::packed(bb)) {
      for (Operation *op : allOps) {
        if (!isReductionOp(op) || scheduledReds.contains(op))
          continue;
        StringRef kind = op->getName().stripDialect();
        if (kind == "welford") {
          scheduledReds.insert(op);
          redStages.push_back({op});
          continue;
        }
        auto dependsOnUnscheduledRed = [&](Operation *root) {
          llvm::DenseSet<Operation *> seen;
          SmallVector<Operation *, 16> work{root};
          while (!work.empty()) {
            Operation *cur = work.pop_back_val();
            if (!seen.insert(cur).second)
              continue;
            if (cur != root && isReductionOp(cur) &&
                !scheduledReds.contains(cur))
              return true;
            if (cur != root && isReductionOp(cur))
              continue;
            for (Value in : cur->getOperands())
              if (auto *def = in.getDefiningOp(); def && def->getBlock() == &bb)
                work.push_back(def);
          }
          return false;
        };
        SmallVector<Operation *, 4> stage{op};
        const size_t maxFusedReds =
            (grouped && allOps.size() > 32) ? 2 : 4;
        for (Operation *cand : allOps) {
          if (stage.size() >= maxFusedReds)
            break;
          if (cand == op || cand->getName().stripDialect() != kind ||
              scheduledReds.contains(cand))
            continue;
          if (!dependsOnUnscheduledRed(cand))
            stage.push_back(cand);
        }
        for (Operation *sOp : stage)
          scheduledReds.insert(sOp);
        redStages.push_back(std::move(stage));
      }
    }

    int64_t staticInnerCols = -1;
    if (!isTransposeTile && walk != "destination" &&
        !task.hasAttr("cpu.member_tasks")) {
      int64_t domSlot = task.getAttrOfType<IntegerAttr>("domain_arg").getInt();
      if (domSlot >= 0 && unsigned(domSlot) < fn.getNumArguments()) {
        if (auto *sh = facts.shapeOf(fn.getArgument(domSlot));
            sh && !sh->empty()) {
          if (auto cExt = sh->back().getAsInteger(); cExt && *cExt > 0)
            staticInnerCols = *cExt;
        }
      }
    }
    bool staticExactInnerPanels = false;
    if (staticInnerCols > 0 && fixed > 0 && !ownerAxis) {
      OpBuilder sb(b.getContext());
      Location sloc = b.getUnknownLoc();
      OperationState scopeSt(sloc, "weft_host.task_kernel");
      scopeSt.addAttribute("sym_name", sb.getStringAttr("panel_tail_probe"));
      scopeSt.addAttribute("source", sb.getStringAttr("panel_tail_probe"));
      scopeSt.addAttribute("view_types", sb.getArrayAttr({}));
      scopeSt.addAttribute("strided_slots", sb.getDenseI64ArrayAttr({}));
      Block &sBb = scopeSt.addRegion()->emplaceBlock();
      Value dummySl = sBb.addArgument(sb.getIndexType(), sloc);
      OwningOpRef<Operation *> scopeOp(Operation::create(scopeSt));
      sb.setInsertionPointToEnd(&sBb);
      Value z0 = sb.create<arith::ConstantIndexOp>(sloc, 0);
      Value cCols = sb.create<arith::ConstantIndexOp>(sloc, staticInnerCols);
      Value cStep = sb.create<arith::ConstantIndexOp>(sloc, fixed);
      Operation *pFor =
          makeOp(sb, sloc, "task_for", {z0, cCols, cStep}, {},
                 {sb.getNamedAttr("iter_types", sb.getArrayAttr({}))}, 1);
      makeOp(sb, sloc, "end");
      Block &pBb = pFor->getRegion(0).emplaceBlock();
      Value pIv = pBb.addArgument(sb.getIndexType(), sloc);
      sb.setInsertionPointToEnd(&pBb);
      std::string baseSlTy = regionType("slice", "f32", 1, -2);
      std::string vTy = regionType("value", "f32", 1, int(fixed));
      Operation *subOp = makeOp(
          sb, sloc, "task_subview", {dummySl, pIv}, {sb.getIndexType()},
          {sb.getNamedAttr("static_extent", sb.getI64IntegerAttr(fixed)),
           sb.getNamedAttr("container_extent",
                           sb.getI64IntegerAttr(staticInnerCols)),
           sb.getNamedAttr("slice_type", sb.getStringAttr(baseSlTy)),
           sb.getNamedAttr("val_type", sb.getStringAttr(vTy)),
           sb.getNamedAttr("admit", sb.getBoolAttr(true))});
      makeOp(sb, sloc, "task_yield", {}, {},
             {sb.getNamedAttr("val_types", sb.getArrayAttr({}))});
      auto tp = compilerinfra::proveTailAccess(subOp);
      staticExactInnerPanels =
          tp.proven && tp.kind == compilerinfra::TailKind::ExactInBounds;
    }

    int64_t maxRowCacheCols = 8192;
    SmallVector<Value, 4> cachedCut;
    SmallVector<Value, 4> cachePtrs;
    SmallVector<std::string, 4> cacheElemSps;
    if (!isTransposeTile && !intentdsl::cpu::atomicMembers(bb) &&
        !intentdsl::cpu::packed(bb) && !ownerAxis && !whole && fixed > 0 &&
        !hasSpecialSweep &&
        (redStages.size() == 1 || redStages.size() == 2) &&
        llvm::none_of(redStages,
                      [](const auto &st) {
                        return st.front()->getName().stripDialect() ==
                               "welford";
                      }) &&
        !writes.empty() && ownedWrites.empty()) {
      llvm::DenseSet<Operation *> priorReds;
      for (size_t sIdx = 0; sIdx + 1 < redStages.size(); ++sIdx)
        for (Operation *ro : redStages[sIdx])
          priorReds.insert(ro);
      const auto &lastStage = redStages.back();
      llvm::DenseSet<Operation *> s1Ops, s2Ops, storeDirect;
      {
        SmallVector<Operation *, 16> work;
        for (Operation *ro : lastStage)
          if (auto *def = ro->getOperand(0).getDefiningOp();
              def && def->getBlock() == &bb && !priorReds.contains(def))
            work.push_back(def);
        while (!work.empty()) {
          Operation *cur = work.pop_back_val();
          if (!s1Ops.insert(cur).second)
            continue;
          if (cur->getName().stripDialect() == "read_owner" ||
              priorReds.contains(cur))
            continue;
          for (Value in : cur->getOperands())
            if (auto *def = in.getDefiningOp();
                def && def->getBlock() == &bb && !priorReds.contains(def))
              work.push_back(def);
        }
      }
      {
        SmallVector<Operation *, 16> work;
        for (Operation *wo : writes) {
          if (auto *def = wo->getOperand(2).getDefiningOp();
              def && def->getBlock() == &bb && !scheduledReds.contains(def)) {
            storeDirect.insert(def);
            work.push_back(def);
          }
        }
        while (!work.empty()) {
          Operation *cur = work.pop_back_val();
          if (!s2Ops.insert(cur).second)
            continue;
          if (cur->getName().stripDialect() == "read_owner")
            continue;
          for (Value in : cur->getOperands())
            if (auto *def = in.getDefiningOp();
                def && def->getBlock() == &bb && !scheduledReds.contains(def))
              work.push_back(def);
        }
      }
      bool unwrapped = true;
      llvm::DenseSet<Operation *> forcedRecompute;
      while (unwrapped) {
        unwrapped = false;
        for (Operation *op : allOps) {
          if (!s1Ops.contains(op) || forcedRecompute.contains(op) ||
              op->getName().stripDialect() != "cast")
            continue;
          auto inTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
          auto outTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
          auto *inDef = op->getOperand(0).getDefiningOp();
          if (!inTy || !outTy || !inDef || !s1Ops.contains(inDef) ||
              inTy.getElementTypeBitWidth() >= outTy.getElementTypeBitWidth() ||
              !(inTy.getElementType().isBF16() ||
                inTy.getElementType().isF16()))
            continue;
          bool usedInS2 =
              storeDirect.contains(op) ||
              llvm::any_of(op->getResult(0).getUsers(), [&](Operation *u) {
                return (s2Ops.contains(u) && !s1Ops.contains(u)) ||
                       forcedRecompute.contains(u);
              });
          if (usedInS2) {
            forcedRecompute.insert(op);
            unwrapped = true;
          }
        }
      }
      bool validCut = true;
      int64_t bytesPerCol = 0;
      for (Operation *op : allOps) {
        StringRef opKind = op->getName().stripDialect();
        if (!s1Ops.contains(op) || forcedRecompute.contains(op) ||
            opKind == "uniform" || opKind == "read" || opKind == "constant" ||
            op->getNumResults() != 1 ||
            !isa<ShapedType>(op->getResult(0).getType()))
          continue;
        Value v = op->getResult(0);
        bool usedOutside =
            storeDirect.contains(op) ||
            llvm::any_of(v.getUsers(), [&](Operation *u) {
              return (s2Ops.contains(u) && !s1Ops.contains(u)) ||
                     forcedRecompute.contains(u);
            });
        if (!usedOutside)
          continue;
        auto rt = dyn_cast<RankedTensorType>(v.getType());
        if (!rt || !(rt.getElementType().isBF16() ||
                     rt.getElementType().isF16() ||
                     rt.getElementType().isF32())) {
          validCut = false;
          break;
        }
        bytesPerCol += std::max<int64_t>(1, rt.getElementTypeBitWidth() / 8);
        cachedCut.push_back(v);
      }
      int64_t maxCacheBytes =
          std::max<int64_t>(16384, (privateStackBytes * 3) / 4);
      maxRowCacheCols =
          bytesPerCol > 0
              ? std::min<int64_t>(8192,
                                  (maxCacheBytes / bytesPerCol) & ~int64_t(63))
              : 0;
      if (staticInnerCols > 0)
        maxRowCacheCols = std::min<int64_t>(
            maxRowCacheCols,
            std::max<int64_t>(64, (staticInnerCols + 63) & ~int64_t(63)));
      if (!validCut || cachedCut.empty() || cachedCut.size() > 4 ||
          maxRowCacheCols < std::max<int64_t>(64, fixed)) {
        cachedCut.clear();
      } else {
        for (Value cv : cachedCut) {
          std::string eSp = weftSpelling(
              cast<RankedTensorType>(cv.getType()).getElementType());
          std::string ptrTy = "!weft_kernel.pointer<" + eSp + ">";
          Value ptrV = makeOp(b, loc, "task_local_alloc", {},
                              {b.getIndexType()},
                              {b.getNamedAttr("elements",
                                              b.getI64IntegerAttr(
                                                  maxRowCacheCols)),
                               b.getNamedAttr("alignment",
                                              b.getI64IntegerAttr(64)),
                               b.getNamedAttr("ptr_type",
                                              b.getStringAttr(ptrTy))})
                           ->getResult(0);
          cachePtrs.push_back(ptrV);
          cacheElemSps.push_back(eSp);
        }
      }
    }

    // Outer loop over rows %r = 0 to %T step 1
    Operation *rowFor = makeOp(
        b, loc, "task_for", {rb.zeroIdx.v, rb.extent.v, rb.oneIdx.v}, {},
        {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
    Block &rowBb = rowFor->getRegion(0).emplaceBlock();
    TaskTerm rowIv{rowBb.addArgument(b.getIndexType(), loc), "index"};
    OpBuilder::InsertionGuard rowGuard(b);
    b.setInsertionPointToEnd(&rowBb);

    if (isTransposeTile) {
      Operation &tOp = bb.front();
      TaskTerm inView = rb.terms.lookup(tOp.getOperand(0));
      TaskTerm outView = rb.terms.lookup(tOp.getOperand(1));
      Type rawElemTy = cast<MemRefType>(tOp.getOperand(0).getType()).getElementType();
      std::string elem = printMlirType(rawElemTy);
      std::string sp = weftSpelling(rawElemTy);
      std::string baseSlTy = regionType("slice", elem, 1, -2);
      int64_t tileW = task.getAttrOfType<IntegerAttr>("cpu.tile").getInt();
      int64_t tailVec =
          std::min<int64_t>(tileW, std::max<int64_t>(1, tailVecWidth));
      int64_t staticCols = -1;
      if (fn.getNumArguments() >= 1) {
        if (auto *sh = facts.shapeOf(fn.getArgument(0));
            sh && sh->size() == 2) {
          if (auto cExt = (*sh)[1].getAsInteger(); cExt && *cExt > 0)
            staticCols = *cExt;
        }
      }
      bool exactFullTile = false;
      bool exactTailVec = false;
      if (staticCols > 0 && tileW > 0) {
        auto proveStaticStep = [&](int64_t stepW) -> bool {
          if (stepW <= 0)
            return false;
          OpBuilder sb(b.getContext());
          Location sloc = b.getUnknownLoc();
          OperationState scopeSt(sloc, "weft_host.task_kernel");
          scopeSt.addAttribute("sym_name", sb.getStringAttr("tail_probe"));
          scopeSt.addAttribute("source", sb.getStringAttr("tail_probe"));
          scopeSt.addAttribute("view_types", sb.getArrayAttr({}));
          scopeSt.addAttribute("strided_slots", sb.getDenseI64ArrayAttr({}));
          Block &sBb = scopeSt.addRegion()->emplaceBlock();
          Value dummySl = sBb.addArgument(sb.getIndexType(), sloc);
          OwningOpRef<Operation *> scopeOp(Operation::create(scopeSt));
          sb.setInsertionPointToEnd(&sBb);
          Value z0 = sb.create<arith::ConstantIndexOp>(sloc, 0);
          Value cCols = sb.create<arith::ConstantIndexOp>(sloc, staticCols);
          Value cStep = sb.create<arith::ConstantIndexOp>(sloc, stepW);
          Operation *pFor =
              makeOp(sb, sloc, "task_for", {z0, cCols, cStep}, {},
                     {sb.getNamedAttr("iter_types", sb.getArrayAttr({}))}, 1);
          makeOp(sb, sloc, "end");
          Block &pBb = pFor->getRegion(0).emplaceBlock();
          Value pIv = pBb.addArgument(sb.getIndexType(), sloc);
          sb.setInsertionPointToEnd(&pBb);
          std::string vTy = regionType("value", elem, 1, int(stepW));
          Operation *subOp = makeOp(
              sb, sloc, "task_subview", {dummySl, pIv}, {sb.getIndexType()},
              {sb.getNamedAttr("static_extent", sb.getI64IntegerAttr(stepW)),
               sb.getNamedAttr("container_extent",
                               sb.getI64IntegerAttr(staticCols)),
               sb.getNamedAttr("slice_type", sb.getStringAttr(baseSlTy)),
               sb.getNamedAttr("val_type", sb.getStringAttr(vTy)),
               sb.getNamedAttr("admit", sb.getBoolAttr(true))});
          makeOp(sb, sloc, "task_yield", {}, {},
                 {sb.getNamedAttr("val_types", sb.getArrayAttr({}))});
          auto tp = compilerinfra::proveTailAccess(subOp);
          return tp.proven && tp.kind == compilerinfra::TailKind::ExactInBounds;
        };
        exactFullTile = proveStaticStep(tileW);
        if (!exactFullTile && tailVec > 1 && (tileW % tailVec == 0))
          exactTailVec = proveStaticStep(tailVec);
      }
      TaskTerm tileTerm = rb.idx(tileW);
      TaskTerm tileCnt = rb.bin("div", rb.columns, tileTerm, "index");
      TaskTerm tileEnd = rb.bin("mul", tileCnt, tileTerm, "index");

      auto emitVecCopy = [&](TaskTerm lb, TaskTerm ub, TaskTerm step, int dim,
                             bool mulTile) {
        Operation *pFor = makeOp(
            b, loc, "task_for", {lb.v, ub.v, step.v}, {},
            {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
        Block &pBb = pFor->getRegion(0).emplaceBlock();
        TaskTerm pIv{pBb.addArgument(b.getIndexType(), loc), "index"};
        OpBuilder::InsertionGuard pg(b);
        b.setInsertionPointToEnd(&pBb);
        TaskTerm off =
            mulTile ? rb.bin("mul", pIv, tileTerm, "index") : pIv;
        Value inSl =
            makeOp(b, loc, "task_slice", {inView.v, rowIv.v},
                   {b.getIndexType()},
                   {b.getNamedAttr("view_type", b.getStringAttr(inView.type)),
                    b.getNamedAttr("selectors",
                                   b.getArrayAttr({b.getStringAttr("index"),
                                                   b.getStringAttr("all")})),
                    b.getNamedAttr("slice_type", b.getStringAttr(baseSlTy)),
                    b.getNamedAttr("val_type", b.getStringAttr(baseSlTy)),
                    b.getNamedAttr("admit", b.getBoolAttr(false))})
                ->getResult(0);
        std::string vTy = regionType("value", elem, 1, dim);
        Value val =
            makeOp(b, loc, "task_subview", {inSl, off.v}, {b.getIndexType()},
                   {b.getNamedAttr("static_extent", b.getI64IntegerAttr(dim)),
                    b.getNamedAttr("slice_type", b.getStringAttr(baseSlTy)),
                    b.getNamedAttr("val_type", b.getStringAttr(vTy)),
                    b.getNamedAttr("admit", b.getBoolAttr(true))})
                ->getResult(0);
        Value outSl =
            makeOp(b, loc, "task_slice", {outView.v, rowIv.v},
                   {b.getIndexType()},
                   {b.getNamedAttr("view_type", b.getStringAttr(outView.type)),
                    b.getNamedAttr("selectors",
                                   b.getArrayAttr({b.getStringAttr("index"),
                                                   b.getStringAttr("all")})),
                    b.getNamedAttr("slice_type", b.getStringAttr(baseSlTy)),
                    b.getNamedAttr("val_type", b.getStringAttr(baseSlTy)),
                    b.getNamedAttr("admit", b.getBoolAttr(false))})
                ->getResult(0);
        std::string outSubSlTy = regionType("slice", elem, 1, dim);
        Value win =
            makeOp(b, loc, "task_subview", {outSl, off.v}, {b.getIndexType()},
                   {b.getNamedAttr("static_extent", b.getI64IntegerAttr(dim)),
                    b.getNamedAttr("slice_type", b.getStringAttr(baseSlTy)),
                    b.getNamedAttr("val_type", b.getStringAttr(outSubSlTy)),
                    b.getNamedAttr("admit", b.getBoolAttr(false))})
                ->getResult(0);
        rb.commitOp({val, vTy}, {win, outSubSlTy});
        makeOp(b, loc, "task_yield", {}, {},
               {b.getNamedAttr("val_types", b.getArrayAttr({}))});
      };

      emitVecCopy(rb.zeroIdx, tileCnt, rb.oneIdx, int(tileW), true);
      if (!exactFullTile) {
        TaskTerm vecEnd = tileEnd;
        if (tailVec > 1 && tailVec < tileW) {
          TaskTerm rem = rb.bin("sub", rb.columns, tileEnd, "index");
          TaskTerm vecStep = rb.idx(tailVec);
          TaskTerm vecCnt = rb.bin("div", rem, vecStep, "index");
          vecEnd = rb.bin("add", tileEnd,
                          rb.bin("mul", vecCnt, vecStep, "index"), "index");
          emitVecCopy(tileEnd, vecEnd, vecStep, int(tailVec), false);
        }
        if (!exactTailVec) {
          Operation *sFor = makeOp(
              b, loc, "task_for", {vecEnd.v, rb.columns.v, rb.oneIdx.v}, {},
              {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
          Block &sBb = sFor->getRegion(0).emplaceBlock();
          TaskTerm sIv{sBb.addArgument(b.getIndexType(), loc), "index"};
          OpBuilder::InsertionGuard sg(b);
          b.setInsertionPointToEnd(&sBb);
          TaskTerm sc = rb.scalarSlice(tOp.getOperand(0), {rowIv, sIv}, true, sp);
          rb.commitOp(sc, rb.scalarSlice(tOp.getOperand(1), {rowIv, sIv}, false));
          makeOp(b, loc, "task_yield", {}, {},
                 {b.getNamedAttr("val_types", b.getArrayAttr({}))});
        }
      }
    } else if (intentdsl::cpu::atomicMembers(bb)) {
      auto &effects = bb.front().getRegion(0).front();
      TaskTerm zeroI32 = rb.lit("0", "si32");
      SmallVector<Value, 4> scans;
      SmallVector<Value, 6> jOps{rb.zeroIdx.v, rb.columns.v, rb.oneIdx.v};
      SmallVector<Attribute, 4> scanTys;
      SmallVector<Type, 4> scanResTypes;
      for (Operation &v : effects) {
        if (v.getName().stripDialect() == "scan_sum") {
          scans.push_back(v.getResult(0));
          std::string ty = weftSpelling(
              cast<ShapedType>(v.getResult(0).getType()).getElementType());
          TaskTerm init = ty == "si32" ? zeroI32 : rb.lit("0.0", ty);
          jOps.push_back(init.v);
          scanTys.push_back(b.getStringAttr(ty));
          scanResTypes.push_back(b.getIndexType());
        }
      }
      auto scanSlot = [&](Value v) -> unsigned {
        return llvm::find(scans, v) - scans.begin();
      };

      Operation *jFor =
          makeOp(b, loc, "task_for", jOps, scanResTypes,
                 {b.getNamedAttr("iter_types", b.getArrayAttr(scanTys))}, 1);
      Block &jBb = jFor->getRegion(0).emplaceBlock();
      TaskTerm jIv{jBb.addArgument(b.getIndexType(), loc), "index"};
      SmallVector<TaskTerm, 4> scanCarries;
      for (Attribute a : scanTys)
        scanCarries.push_back(
            {jBb.addArgument(b.getIndexType(), loc), cast<StringAttr>(a).str()});

      {
        OpBuilder::InsertionGuard jg(b);
        b.setInsertionPointToEnd(&jBb);
        Block *guardThenBb = nullptr;
        llvm::DenseMap<Value, TaskTerm> localMap;
        auto a = [&](Operation &v, unsigned i) {
          auto it = localMap.find(v.getOperand(i));
          return it != localMap.end() ? it->second
                                      : rb.terms.lookup(v.getOperand(i));
        };

        for (Operation &v : effects.without_terminator()) {
          StringRef n = v.getName().stripDialect();
          TaskTerm rhs;
          if (n == "read") {
            SmallVector<TaskTerm, 2> coords;
            if (cast<MemRefType>(v.getOperand(0).getType()).getRank() == 2)
              coords.push_back(rowIv);
            coords.push_back(jIv);
            rhs = rb.scalarSlice(
                v.getOperand(0), coords, true,
                weftSpelling(
                    cast<MemRefType>(v.getOperand(0).getType()).getElementType()));
          } else if (n == "fill_ranges") {
            TaskTerm hi = rb.bin("add", jIv, rb.oneIdx, "index");
            TaskTerm in = rb.cmp("lt", hi, rb.columns, "i1");
            Operation *ifOp =
                makeOp(b, loc, "task_if", {in.v}, {},
                       {b.getNamedAttr("res_types", b.getArrayAttr({}))}, 2);
            Block &tBb = ifOp->getRegion(0).emplaceBlock();
            {
              OpBuilder::InsertionGuard ig(b);
              b.setInsertionPointToEnd(&tBb);
              TaskTerm unit =
                  rb.idx(v.getAttrOfType<IntegerAttr>("unit").getInt());
              auto blockAt = [&](TaskTerm pos) {
                TaskTerm nVal =
                    rb.scalarSlice(v.getOperand(1), {pos}, true, "si32");
                return rb.bin("div", rb.castOp(nVal, "index"), unit, "index");
              };
              TaskTerm first = blockAt(jIv), end = blockAt(hi);
              Operation *bFor = makeOp(
                  b, loc, "task_for", {first.v, end.v, rb.oneIdx.v}, {},
                  {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
              Block &bBb = bFor->getRegion(0).emplaceBlock();
              TaskTerm bIv{bBb.addArgument(b.getIndexType(), loc), "index"};
              {
                OpBuilder::InsertionGuard bg(b);
                b.setInsertionPointToEnd(&bBb);
                rb.commitOp(rb.castOp(jIv, "si32"),
                            rb.scalarSlice(v.getOperand(0), {bIv}, false));
                makeOp(b, loc, "task_yield", {}, {},
                       {b.getNamedAttr("val_types", b.getArrayAttr({}))});
              }
              makeOp(b, loc, "task_yield", {}, {},
                     {b.getNamedAttr("val_types", b.getArrayAttr({}))});
            }
            continue;
          } else if (n == "constant") {
            if (auto x = v.getAttrOfType<IntegerAttr>("value"))
              rhs = rb.lit(std::to_string(x.getInt()), "si32");
            else
              rhs = rb.lit(stripTypeSuffix(v.getAttr("value")), "f32");
          } else if (n == "member_index") {
            rhs = rb.castOp(jIv, "si32");
          } else if (n == "ne") {
            rhs = rb.cmp("ne", a(v, 0), a(v, 1), "i1");
          } else if (n == "cast") {
            rhs = rb.castOp(
                a(v, 0),
                weftSpelling(
                    cast<ShapedType>(v.getResult(0).getType()).getElementType()));
          } else if (n == "prefix_scatter") {
            Operation *ifOp =
                makeOp(b, loc, "task_if", {a(v, 3).v}, {},
                       {b.getNamedAttr("res_types", b.getArrayAttr({}))}, 2);
            Block &tBb = ifOp->getRegion(0).emplaceBlock();
            {
              OpBuilder::InsertionGuard ig(b);
              b.setInsertionPointToEnd(&tBb);
              TaskTerm ix =
                  rb.bin("sub", rb.castOp(a(v, 4), "index"), rb.oneIdx, "index");
              rb.commitOp(a(v, 5),
                          rb.scalarSlice(v.getOperand(0), {rowIv, ix}, false));
              makeOp(b, loc, "task_yield", {}, {},
                     {b.getNamedAttr("val_types", b.getArrayAttr({}))});
            }
            continue;
          } else if (n == "ordinal") {
            TaskTerm rGlob = rb.bin("add", rb.origin, rowIv, "index");
            TaskTerm rOff = rb.bin("mul", rGlob, rb.columns, "index");
            rhs = rb.castOp(rb.bin("add", rOff, jIv, "index"), "si32");
          } else if (n == "add" || n == "shr" || n == "shl") {
            rhs = rb.bin(n, a(v, 0), a(v, 1), "si32");
          } else if (n == "scan_sum") {
            unsigned sl = scanSlot(v.getResult(0));
            TaskTerm in = a(v, 0);
            rhs = rb.bin("add", scanCarries[sl], in, in.type);
          } else if (n == "write_offsets") {
            TaskTerm dst = rb.bin("add", jIv, rb.oneIdx, "index");
            rb.commitOp(a(v, 3),
                        rb.scalarSlice(v.getOperand(0), {dst}, false));
            continue;
          } else if (n == "atomic_add" || n == "index_read" ||
                     n == "unique_put") {
            TaskTerm index = rb.castOp(a(v, 1), "index");
            if (ownedKeys && n == "atomic_add") {
              TaskTerm lower = rb.cmp("ge", index, rb.keyLo, "i1");
              TaskTerm upper = rb.cmp("lt", index, rb.keyHi, "i1");
              TaskTerm live = rb.bin("and", lower, upper, "i1");
              Operation *ifOp =
                  makeOp(b, loc, "task_if", {live.v}, {},
                         {b.getNamedAttr("res_types", b.getArrayAttr({}))}, 2);
              guardThenBb = &ifOp->getRegion(0).emplaceBlock();
              b.setInsertionPointToEnd(guardThenBb);
            }
            if (n == "atomic_add") {
              TaskTerm delta = a(v, 2);
              if (ownedKeys || rb.task.hasAttr("cpu.single_dispatcher")) {
                TaskTerm prev =
                    rb.scalarSlice(v.getOperand(0), {index}, true, "si32");
                TaskTerm next = rb.bin("add", prev, delta, "si32");
                rb.commitOp(next, rb.scalarSlice(v.getOperand(0), {index}, false));
                rhs = prev;
              } else {
                TaskTerm dst = rb.scalarSlice(v.getOperand(0), {index}, false);
                Value rmw =
                    makeOp(b, loc, "task_atomic_rmw", {delta.v, dst.v},
                           {b.getIndexType()},
                           {b.getNamedAttr("kind", b.getStringAttr("add")),
                            b.getNamedAttr("ordering",
                                           b.getStringAttr("relaxed")),
                            b.getNamedAttr("val_type", b.getStringAttr("si32")),
                            b.getNamedAttr("slice_type",
                                           b.getStringAttr(dst.type))})
                        ->getResult(0);
                rhs = {rmw, "si32"};
              }
            } else if (n == "unique_put") {
              rb.commitOp(a(v, 2),
                          rb.scalarSlice(v.getOperand(0), {index}, false));
              continue;
            } else {
              rhs = rb.scalarSlice(v.getOperand(0), {index}, true, "si32");
            }
          } else {
            return v.emitError("unsupported atomic task expression ") << n;
          }
          localMap[v.getResult(0)] = rhs;
        }
        if (guardThenBb) {
          makeOp(b, loc, "task_yield", {}, {},
                 {b.getNamedAttr("val_types", b.getArrayAttr({}))});
          b.setInsertionPointToEnd(&jBb);
        }
        SmallVector<Value, 4> yVals;
        for (Value sv : scans)
          yVals.push_back(localMap.at(sv).v);
        makeOp(b, loc, "task_yield", yVals, {},
               {b.getNamedAttr("val_types", b.getArrayAttr(scanTys))});
      }

      for (Operation &ends : effects) {
        if (ends.getName().stripDialect() == "write_offsets") {
          rb.commitOp(zeroI32,
                      rb.scalarSlice(ends.getOperand(0), {rb.zeroIdx}, false));
          unsigned sl = scanSlot(ends.getOperand(3));
          rb.commitOp({jFor->getResult(sl), "si32"},
                      rb.scalarSlice(ends.getOperand(1), {rb.zeroIdx}, false));
        }
      }
      for (Operation &v : effects) {
        if (v.getName().stripDialect() == "prefix_scatter") {
          unsigned sl = scanSlot(v.getOperand(4));
          TaskTerm total{jFor->getResult(sl), "si32"};
          rb.commitOp(total, rb.scalarSlice(v.getOperand(1), {rowIv}, false));
          TaskTerm first = rb.castOp(total, "index");
          TaskTerm pad = rb.lit(
              std::to_string(v.getAttrOfType<IntegerAttr>("fill").getInt()),
              "si32");
          Operation *tFor = makeOp(
              b, loc, "task_for", {first.v, rb.columns.v, rb.oneIdx.v}, {},
              {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
          Block &tBb = tFor->getRegion(0).emplaceBlock();
          TaskTerm tailIv{tBb.addArgument(b.getIndexType(), loc), "index"};
          {
            OpBuilder::InsertionGuard tg(b);
            b.setInsertionPointToEnd(&tBb);
            rb.commitOp(pad,
                        rb.scalarSlice(v.getOperand(0), {rowIv, tailIv}, false));
            makeOp(b, loc, "task_yield", {}, {},
                   {b.getNamedAttr("val_types", b.getArrayAttr({}))});
          }
        }
      }
    } else if (intentdsl::cpu::packed(bb)) {
      Operation &pack = bb.front();
      TaskTerm rhs = rb.lit(stripTypeSuffix(pack.getAttr("threshold")), "f32");
      TaskTerm pad = rb.lit(
          std::to_string(pack.getAttrOfType<IntegerAttr>("fill").getInt()),
          "si32");
      Operation *cFor = makeOp(
          b, loc, "task_for",
          {rb.zeroIdx.v, rb.columns.v, rb.oneIdx.v, rb.zeroIdx.v},
          {b.getIndexType()},
          {b.getNamedAttr("iter_types",
                          b.getArrayAttr({b.getStringAttr("index")}))},
          1);
      Block &cBb = cFor->getRegion(0).emplaceBlock();
      TaskTerm colIv{cBb.addArgument(b.getIndexType(), loc), "index"};
      TaskTerm posIv{cBb.addArgument(b.getIndexType(), loc), "index"};
      {
        OpBuilder::InsertionGuard cg(b);
        b.setInsertionPointToEnd(&cBb);
        TaskTerm x =
            rb.scalarSlice(pack.getOperand(0), {rowIv, colIv}, true, "f32");
        TaskTerm keep = rb.cmp("ne", x, rhs, "i1");
        Operation *ifOp =
            makeOp(b, loc, "task_if", {keep.v}, {},
                   {b.getNamedAttr("res_types", b.getArrayAttr({}))}, 2);
        Block &tBb = ifOp->getRegion(0).emplaceBlock();
        {
          OpBuilder::InsertionGuard ig(b);
          b.setInsertionPointToEnd(&tBb);
          rb.commitOp(rb.castOp(colIv, "si32"),
                      rb.scalarSlice(pack.getOperand(1), {rowIv, posIv}, false));
          makeOp(b, loc, "task_yield", {}, {},
                 {b.getNamedAttr("val_types", b.getArrayAttr({}))});
        }
        TaskTerm next = rb.bin(
            "add", posIv, rb.sel(keep, rb.oneIdx, rb.zeroIdx, "index"), "index");
        makeOp(b, loc, "task_yield", {next.v}, {},
               {b.getNamedAttr("val_types",
                               b.getArrayAttr({b.getStringAttr("index")}))});
      }
      TaskTerm count{cFor->getResult(0), "index"};
      Operation *tFor = makeOp(
          b, loc, "task_for", {count.v, rb.columns.v, rb.oneIdx.v}, {},
          {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
      Block &tBb = tFor->getRegion(0).emplaceBlock();
      TaskTerm tailIv{tBb.addArgument(b.getIndexType(), loc), "index"};
      {
        OpBuilder::InsertionGuard tg(b);
        b.setInsertionPointToEnd(&tBb);
        rb.commitOp(pad,
                    rb.scalarSlice(pack.getOperand(1), {rowIv, tailIv}, false));
        makeOp(b, loc, "task_yield", {}, {},
               {b.getNamedAttr("val_types", b.getArrayAttr({}))});
      }
      rb.commitOp(rb.castOp(count, "si32"),
                  rb.scalarSlice(pack.getOperand(2), {rowIv}, false));
    } else if (whole && !ownerAxis && redStages.empty() && !hasSpecialSweep) {
      SmallVector<Operation *> ops;
      for (Operation *op : allOps)
        if (op->getName().stripDialect() != "write")
          ops.push_back(op);
      llvm::DenseMap<Value, TaskTerm> localMap;
      if (failed(rb.emitVectorPanelOps(ops, writes, rowIv, {rowIv},
                                       std::nullopt, rb.panelWidth, -2, -2, 2,
                                       false, localMap)))
        return failure();
    } else {
      // Decompose logical row body into staged sweeps:
      // 1) read_owner + hoisted row-invariant scalar math
      // 2) fused reduction / SIMD Welford stages (+ scalar math between stages)
      // 3) panel store / owner publish sweeps (with optional L1d row cache)
      llvm::DenseSet<Value> summaryValues;
      for (Operation *op : allOps) {
        if (op->getName().stripDialect() == "read_owner") {
          rb.terms[op->getResult(0)] = rb.memberSlice(
              op->getOperand(0), rowIv, {}, true, true,
              weftSpelling(op->getResult(0).getType()));
          summaryValues.insert(op->getResult(0));
        }
      }

      auto hasBoundScalar = [&](Value v) {
        return !isa<MemRefType>(v.getType()) && !isa<ShapedType>(v.getType()) &&
               rb.terms.count(v) > 0;
      };

      auto emitAvailableScalars = [&]() -> LogicalResult {
        for (Operation *op : allOps) {
          if (op->getNumResults() != 1)
            continue;
          Value res = op->getResult(0);
          if (isa<ShapedType>(res.getType()) || rb.terms.count(res) ||
              rb.records.count(res))
            continue;
          StringRef n = op->getName().stripDialect();
          if (n == "read" || n == "read_owner" || isReductionOp(op) ||
              n == "ordered_sum" || n == "segment_length" ||
              n == "gather_row" || n == "count_keys" || n == "column_sum" ||
              n == "scan_sum" || n == "ordinal" || n == "uniform" ||
              n == "member_index")
            continue;
          if (n == "field") {
            auto saved = rb.records.find(op->getOperand(0));
            if (saved == rb.records.end())
              continue;
            rb.terms[res] = saved->second[op->getAttrOfType<IntegerAttr>(
                                                "position")
                                              .getInt()];
            summaryValues.insert(res);
            continue;
          }
          std::string ty = weftSpelling(res.getType());
          if (n == "constant") {
            if (auto integer = op->getAttrOfType<IntegerAttr>("value")) {
              rb.terms[res] = rb.lit(
                  ty == "i1" ? (integer.getInt() ? "true" : "false")
                             : std::to_string(integer.getInt()),
                  ty);
            } else {
              rb.terms[res] = rb.lit(stripTypeSuffix(op->getAttr("value")), ty);
            }
            summaryValues.insert(res);
            continue;
          }
          if (!llvm::all_of(op->getOperands(), hasBoundScalar))
            continue;
          if (n == "cast") {
            rb.terms[res] = rb.castOp(rb.terms.lookup(op->getOperand(0)), ty);
          } else if (n == "sqrt" || n == "rsqrt" || n == "exp") {
            rb.terms[res] = rb.unaryOp(n, rb.terms.lookup(op->getOperand(0)));
          } else if (n == "select") {
            rb.terms[res] =
                rb.sel(rb.terms.lookup(op->getOperand(0)),
                       rb.terms.lookup(op->getOperand(1)),
                       rb.terms.lookup(op->getOperand(2)), ty);
          } else if (n == "gt" || n == "ne") {
            rb.terms[res] =
                rb.cmp(n, rb.terms.lookup(op->getOperand(0)),
                       rb.terms.lookup(op->getOperand(1)), ty);
          } else if (n == "xor" || n == "and" || n == "shl" || n == "shr" ||
                     n == "maximum" || n == "mul" || n == "add" || n == "div" ||
                     n == "sub") {
            StringRef bKind = n == "maximum" ? StringRef("max") : n;
            rb.terms[res] =
                rb.bin(bKind, rb.terms.lookup(op->getOperand(0)),
                       rb.terms.lookup(op->getOperand(1)), ty);
          } else {
            continue;
          }
          summaryValues.insert(res);
        }
        return success();
      };

      if (failed(emitAvailableScalars()))
        return failure();

      int d0SymIdx = 3 + (absolute ? 1 : 0) + (ownedKeys ? 2 : 0);
      int ownerBaseColDim =
          vecOwnerInner ? -(1 + d0SymIdx + int(rankDomain - 1)) : -2;
      int ownerVecAxisId = vecOwnerInner ? int(rankDomain) : 2;
      SmallVector<int64_t, 4> outerAxes;
      TaskTerm outerTotal = rb.oneIdx;
      TaskTerm innerCols, innerPanelCount, innerTailStart;
      if (vecOwnerInner) {
        for (int64_t j = 0; j < rankDomain - 1; ++j)
          if (j != *ownerAxis)
            outerAxes.push_back(j);
        if (!outerAxes.empty()) {
          outerTotal = rb.extents[outerAxes[0]];
          for (size_t k = 1; k < outerAxes.size(); ++k)
            outerTotal =
                rb.bin("mul", outerTotal, rb.extents[outerAxes[k]], "index");
        }
        innerCols = rb.extents[rankDomain - 1];
        innerPanelCount = rb.bin("div", innerCols, rb.panelWidth, "index");
        innerTailStart = rb.bin("mul", innerPanelCount, rb.panelWidth, "index");
      }

      auto buildOwnerPrefix = [&](TaskTerm outerIv) {
        SmallVector<TaskTerm, 4> prefix(rankDomain - 1);
        prefix[*ownerAxis] = rowIv;
        if (outerAxes.size() == 1) {
          prefix[outerAxes[0]] = outerIv;
        } else if (outerAxes.size() > 1) {
          TaskTerm cur = outerIv;
          for (int idx = int(outerAxes.size()) - 1; idx >= 0; --idx) {
            int64_t ax = outerAxes[idx];
            if (idx == 0) {
              prefix[ax] = cur;
            } else {
              TaskTerm sz = rb.extents[ax];
              prefix[ax] = rb.bin("mod", cur, sz, "index");
              cur = rb.bin("div", cur, sz, "index");
            }
          }
        }
        return prefix;
      };

      int64_t tailVec =
          fixed > 0 ? std::min<int64_t>(fixed, std::max<int64_t>(1, tailVecWidth))
                    : std::max<int64_t>(1, tailVecWidth);

      auto canScalarEvalOps = [](ArrayRef<Operation *> ops,
                                 const llvm::DenseSet<Operation *> *skipSet =
                                     nullptr) {
        return llvm::all_of(ops, [&](Operation *opPtr) {
          if (skipSet && skipSet->contains(opPtr))
            return true;
          StringRef k = opPtr->getName().stripDialect();
          return k == "write" || k == "write_owner" || k == "read" ||
                 k == "read_owner" || k == "field" || k == "constant" ||
                 k == "cast" || k == "rsqrt" || k == "exp" || k == "sqrt" ||
                 k == "select" || k == "gt" || k == "ne" || k == "mul" ||
                 k == "add" || k == "div" || k == "sub" || k == "maximum" ||
                 k == "xor" || k == "and" || k == "shl" || k == "shr";
        });
      };

      auto evalScalarAt = [&](ArrayRef<Operation *> ops,
                              ArrayRef<TaskTerm> prefixCoords, TaskTerm colIv,
                              Operation *stopOp,
                              llvm::DenseMap<Value, TaskTerm> &localMap,
                              const llvm::DenseSet<Operation *> *skipSet =
                                  nullptr)
          -> LogicalResult {
        auto lookup = [&](Value v) -> TaskTerm {
          auto it = localMap.find(v);
          return it != localMap.end() ? it->second : rb.terms.lookup(v);
        };
        for (Operation *opPtr : ops) {
          if (opPtr == stopOp || (skipSet && skipSet->contains(opPtr)))
            continue;
          Operation &op = *opPtr;
          StringRef n = op.getName().stripDialect();
          if (n == "write" || n == "write_owner")
            continue;
          Value resVal = op.getResult(0);
          Type rawT = resVal.getType();
          if (auto sh = dyn_cast<ShapedType>(rawT))
            rawT = sh.getElementType();
          std::string ty = weftSpelling(rawT);
          if (n == "read") {
            auto vt = cast<MemRefType>(op.getOperand(0).getType());
            SmallVector<TaskTerm, 4> coords;
            if (vt.getRank() >= 2)
              coords.append(prefixCoords.begin(), prefixCoords.end());
            coords.push_back(colIv);
            localMap[resVal] = rb.scalarSlice(op.getOperand(0), coords, true, ty);
          } else if (n == "read_owner") {
            localMap[resVal] = rb.memberSlice(op.getOperand(0), rowIv, {}, true,
                                              true, ty);
          } else if (n == "uniform") {
            localMap[resVal] = lookup(op.getOperand(0));
          } else if (n == "field") {
            auto saved = rb.records.find(op.getOperand(0));
            if (saved == rb.records.end())
              return op.emitError("unbound Weft tuple field");
            localMap[resVal] =
                saved->second[op.getAttrOfType<IntegerAttr>("position")
                                  .getInt()];
          } else if (n == "constant") {
            if (auto integer = op.getAttrOfType<IntegerAttr>("value"))
              localMap[resVal] = rb.lit(
                  ty == "i1" ? (integer.getInt() ? "true" : "false")
                             : std::to_string(integer.getInt()),
                  ty);
            else
              localMap[resVal] =
                  rb.lit(stripTypeSuffix(op.getAttr("value")), ty);
          } else if (n == "cast") {
            localMap[resVal] = rb.castOp(lookup(op.getOperand(0)), ty);
          } else if (n == "rsqrt" || n == "exp" || n == "sqrt") {
            localMap[resVal] = rb.unaryOp(n, lookup(op.getOperand(0)));
          } else if (n == "select") {
            localMap[resVal] =
                rb.sel(lookup(op.getOperand(0)), lookup(op.getOperand(1)),
                       lookup(op.getOperand(2)), ty);
          } else if (n == "gt" || n == "ne") {
            localMap[resVal] =
                rb.cmp(n, lookup(op.getOperand(0)), lookup(op.getOperand(1)),
                       ty);
          } else if (n == "mul" || n == "add" || n == "div" || n == "sub" ||
                     n == "maximum" || n == "xor" || n == "and" || n == "shl" ||
                     n == "shr") {
            StringRef bKind = n == "maximum" ? StringRef("max") : n;
            localMap[resVal] = rb.bin(bKind, lookup(op.getOperand(0)),
                                      lookup(op.getOperand(1)), ty);
          } else {
            return op.emitError("unsupported scalar tail op ") << n;
          }
        }
        return success();
      };

      auto emitReductionStage =
          [&](ArrayRef<Operation *> stage, ArrayRef<Value> cacheWriteVals,
              bool onlyFullPanels) -> LogicalResult {
        Operation *first = stage.front();
        StringRef combine = first->getName().stripDialect();
        llvm::DenseSet<Operation *> visited;
        SmallVector<Operation *> subOps;
        if (combine == "welford") {
          rb.collectSubgraph(first->getOperand(0).getDefiningOp(),
                             summaryValues, visited, subOps);
          subOps.push_back(first);
        } else {
          for (Operation *ro : stage)
            rb.collectSubgraph(ro, summaryValues, visited, subOps);
        }

        SweepDesc sw;
        sw.combine = combine.str();
        sw.anchor = first;
        sw.anchors.assign(stage.begin(), stage.end());
        for (Operation *so : subOps) {
          StringRef sn = so->getName().stripDialect();
          if (sn == "ordered_sum")
            sw.segmentLoop = true;
          if (sn == "scan_sum")
            sw.prefixCarry = true;
        }

        if (scalarMembers || sw.prefixCarry || sw.segmentLoop || whole ||
            fixed <= 0) {
          if (failed(rb.emitScalarSweep(sw, subOps, rowIv)))
            return failure();
          for (Operation *ro : stage)
            summaryValues.insert(ro->getResult(0));
          return success();
        }

        int axId = vecOwnerInner ? ownerVecAxisId : 2;
        int baseDim = vecOwnerInner ? ownerBaseColDim : -2;
        int w = int(fixed);
        std::string vecF32Ty = regionType("value", "f32", 1, w, axId);

        if (combine == "welford") {
          std::string vecI1Ty = regionType("value", "i1", 1, w, axId);
          TaskTerm initZeroVec = rb.splatOp(rb.zeroF32, vecF32Ty);
          llvm::DenseSet<Operation *> skipW{first};

          auto emitWelfordInner = [&](ArrayRef<TaskTerm> prefix,
                                      TaskTerm colCnt, TaskTerm pCnt,
                                      TaskTerm tStart, TaskTerm inNPanels,
                                      TaskTerm inMeanVec, TaskTerm inM2Vec,
                                      TaskTerm inTCnt, TaskTerm inTMean,
                                      TaskTerm inTM2,
                                      std::array<TaskTerm, 6> &outState)
              -> LogicalResult {
            Operation *pFor = makeOp(
                b, loc, "task_for",
                {rb.zeroIdx.v, pCnt.v, rb.oneIdx.v, inNPanels.v, inMeanVec.v,
                 inM2Vec.v},
                {b.getIndexType(), b.getIndexType(), b.getIndexType()},
                {b.getNamedAttr("iter_types",
                                b.getArrayAttr({b.getStringAttr("si32"),
                                                b.getStringAttr(vecF32Ty),
                                                b.getStringAttr(vecF32Ty)}))},
                1);
            Block &pBb = pFor->getRegion(0).emplaceBlock();
            TaskTerm pIv{pBb.addArgument(b.getIndexType(), loc), "index"};
            TaskTerm nCnt{pBb.addArgument(b.getIndexType(), loc), "si32"};
            TaskTerm meanV{pBb.addArgument(b.getIndexType(), loc), vecF32Ty};
            TaskTerm m2V{pBb.addArgument(b.getIndexType(), loc), vecF32Ty};
            {
              OpBuilder::InsertionGuard pg(b);
              b.setInsertionPointToEnd(&pBb);
              TaskTerm off = rb.bin("mul", pIv, rb.panelWidth, "index");
              llvm::DenseMap<Value, TaskTerm> localMap;
              if (failed(rb.emitVectorPanelOps(subOps, {}, rowIv, prefix, off,
                                               rb.panelWidth, w, baseDim, axId,
                                               true, localMap, &skipW)))
                return failure();
              TaskTerm vVec = localMap.at(first->getOperand(0));
              TaskTerm count = rb.bin("add", nCnt, rb.oneCount, "si32");
              TaskTerm delta = rb.bin("sub", vVec, meanV, vecF32Ty);
              TaskTerm change =
                  rb.bin("div", delta, rb.castOp(count, "f32"), vecF32Ty);
              TaskTerm mNext = rb.bin("add", meanV, change, vecF32Ty);
              TaskTerm m2Next = rb.bin(
                  "add", m2V,
                  rb.bin("mul",
                         rb.bin("mul", delta, rb.castOp(nCnt, "f32"), vecF32Ty),
                         change, vecF32Ty),
                  vecF32Ty);
              TaskTerm firstMask = rb.splatOp(
                  rb.cmp("eq", nCnt, rb.zeroCount, "i1"), vecI1Ty);
              mNext = rb.sel(firstMask, vVec, mNext, vecF32Ty);
              m2Next = rb.sel(firstMask, initZeroVec, m2Next, vecF32Ty);
              makeOp(b, loc, "task_yield", {count.v, mNext.v, m2Next.v}, {},
                     {b.getNamedAttr(
                         "val_types",
                         b.getArrayAttr({b.getStringAttr("si32"),
                                         b.getStringAttr(vecF32Ty),
                                         b.getStringAttr(vecF32Ty)}))});
            }

            Operation *tFor = makeOp(
                b, loc, "task_for",
                {tStart.v, colCnt.v, rb.oneIdx.v, inTCnt.v, inTMean.v,
                 inTM2.v},
                {b.getIndexType(), b.getIndexType(), b.getIndexType()},
                {b.getNamedAttr("iter_types",
                                b.getArrayAttr({b.getStringAttr("si32"),
                                                b.getStringAttr("f32"),
                                                b.getStringAttr("f32")}))},
                1);
            Block &tBb = tFor->getRegion(0).emplaceBlock();
            TaskTerm colIv{tBb.addArgument(b.getIndexType(), loc), "index"};
            TaskTerm sCnt{tBb.addArgument(b.getIndexType(), loc), "si32"};
            TaskTerm sMean{tBb.addArgument(b.getIndexType(), loc), "f32"};
            TaskTerm sM2{tBb.addArgument(b.getIndexType(), loc), "f32"};
            {
              OpBuilder::InsertionGuard tg(b);
              b.setInsertionPointToEnd(&tBb);
              llvm::DenseMap<Value, TaskTerm> localMap;
              if (failed(evalScalarAt(subOps, prefix, colIv, first, localMap)))
                return failure();
              TaskTerm v = localMap.at(first->getOperand(0));
              TaskTerm count = rb.bin("add", sCnt, rb.oneCount, "si32");
              TaskTerm delta = rb.bin("sub", v, sMean, "f32");
              TaskTerm change =
                  rb.bin("div", delta, rb.castOp(count, "f32"), "f32");
              TaskTerm m = rb.bin("add", sMean, change, "f32");
              TaskTerm variance = rb.bin(
                  "add", sM2,
                  rb.bin("mul", rb.bin("mul", delta, rb.castOp(sCnt, "f32"), "f32"),
                         change, "f32"),
                  "f32");
              TaskTerm isFirst = rb.cmp("eq", sCnt, rb.zeroCount, "i1");
              m = rb.sel(isFirst, v, m, "f32");
              variance = rb.sel(isFirst, rb.zeroF32, variance, "f32");
              makeOp(b, loc, "task_yield", {count.v, m.v, variance.v}, {},
                     {b.getNamedAttr(
                         "val_types",
                         b.getArrayAttr({b.getStringAttr("si32"),
                                         b.getStringAttr("f32"),
                                         b.getStringAttr("f32")}))});
            }

            outState = {TaskTerm{pFor->getResult(0), "si32"},
                        TaskTerm{pFor->getResult(1), vecF32Ty},
                        TaskTerm{pFor->getResult(2), vecF32Ty},
                        TaskTerm{tFor->getResult(0), "si32"},
                        TaskTerm{tFor->getResult(1), "f32"},
                        TaskTerm{tFor->getResult(2), "f32"}};
            return success();
          };

          std::array<TaskTerm, 6> finState;
          if (!vecOwnerInner) {
            if (failed(emitWelfordInner({rowIv}, rb.columns, panelCount,
                                        tailStart, rb.zeroCount, initZeroVec,
                                        initZeroVec, rb.zeroCount, rb.zeroF32,
                                        rb.zeroF32, finState)))
              return failure();
          } else {
            SmallVector<Attribute, 6> oTys{
                b.getStringAttr("si32"),     b.getStringAttr(vecF32Ty),
                b.getStringAttr(vecF32Ty),   b.getStringAttr("si32"),
                b.getStringAttr("f32"),      b.getStringAttr("f32")};
            SmallVector<Type, 6> oResTys(6, b.getIndexType());
            Operation *oFor = makeOp(
                b, loc, "task_for",
                {rb.zeroIdx.v, outerTotal.v, rb.oneIdx.v, rb.zeroCount.v,
                 initZeroVec.v, initZeroVec.v, rb.zeroCount.v, rb.zeroF32.v,
                 rb.zeroF32.v},
                oResTys, {b.getNamedAttr("iter_types", b.getArrayAttr(oTys))},
                1);
            Block &oBb = oFor->getRegion(0).emplaceBlock();
            TaskTerm outerIv{oBb.addArgument(b.getIndexType(), loc), "index"};
            TaskTerm cNP{oBb.addArgument(b.getIndexType(), loc), "si32"};
            TaskTerm cMV{oBb.addArgument(b.getIndexType(), loc), vecF32Ty};
            TaskTerm cS2V{oBb.addArgument(b.getIndexType(), loc), vecF32Ty};
            TaskTerm cTC{oBb.addArgument(b.getIndexType(), loc), "si32"};
            TaskTerm cTM{oBb.addArgument(b.getIndexType(), loc), "f32"};
            TaskTerm cTS2{oBb.addArgument(b.getIndexType(), loc), "f32"};
            {
              OpBuilder::InsertionGuard og(b);
              b.setInsertionPointToEnd(&oBb);
              auto prefix = buildOwnerPrefix(outerIv);
              std::array<TaskTerm, 6> stepOut;
              if (failed(emitWelfordInner(prefix, innerCols, innerPanelCount,
                                          innerTailStart, cNP, cMV, cS2V, cTC,
                                          cTM, cTS2, stepOut)))
                return failure();
              makeOp(b, loc, "task_yield",
                     {stepOut[0].v, stepOut[1].v, stepOut[2].v, stepOut[3].v,
                      stepOut[4].v, stepOut[5].v},
                     {}, {b.getNamedAttr("val_types", b.getArrayAttr(oTys))});
            }
            finState = {TaskTerm{oFor->getResult(0), "si32"},
                        TaskTerm{oFor->getResult(1), vecF32Ty},
                        TaskTerm{oFor->getResult(2), vecF32Ty},
                        TaskTerm{oFor->getResult(3), "si32"},
                        TaskTerm{oFor->getResult(4), "f32"},
                        TaskTerm{oFor->getResult(5), "f32"}};
          }

          // Reduce equal-count SIMD lanes via Chan's parallel axis formula.
          TaskTerm nPanels = finState[0], meanVec = finState[1],
                   m2Vec = finState[2];
          TaskTerm vecCnt =
              rb.bin("mul", nPanels, rb.lit(std::to_string(w), "si32"), "si32");
          Value sumMeanV =
              makeOp(b, loc, "task_reduce", {meanVec.v}, {b.getIndexType()},
                     {b.getNamedAttr("kind", b.getStringAttr("add")),
                      b.getNamedAttr("axis", b.getI64IntegerAttr(0)),
                      b.getNamedAttr("in_type", b.getStringAttr(vecF32Ty)),
                      b.getNamedAttr("val_type", b.getStringAttr("f32"))})
                  ->getResult(0);
          TaskTerm vecMeanCand =
              rb.bin("div", {sumMeanV, "f32"},
                     rb.lit(std::to_string(w) + ".0", "f32"), "f32");
          TaskTerm dVec = rb.bin("sub", meanVec, vecMeanCand, vecF32Ty);
          TaskTerm m2Terms = rb.bin(
              "add", m2Vec,
              rb.bin("mul", rb.castOp(nPanels, "f32"),
                     rb.bin("mul", dVec, dVec, vecF32Ty), vecF32Ty),
              vecF32Ty);
          Value sumM2V =
              makeOp(b, loc, "task_reduce", {m2Terms.v}, {b.getIndexType()},
                     {b.getNamedAttr("kind", b.getStringAttr("add")),
                      b.getNamedAttr("axis", b.getI64IntegerAttr(0)),
                      b.getNamedAttr("in_type", b.getStringAttr(vecF32Ty)),
                      b.getNamedAttr("val_type", b.getStringAttr("f32"))})
                  ->getResult(0);
          TaskTerm hasVec = rb.cmp("gt", vecCnt, rb.zeroCount, "i1");
          TaskTerm vecMean = rb.sel(hasVec, vecMeanCand, rb.zeroF32, "f32");
          TaskTerm vecM2 = rb.sel(hasVec, {sumM2V, "f32"}, rb.zeroF32, "f32");

          // Combine vector summary with scalar tail summary.
          TaskTerm tCnt = finState[3], tMean = finState[4], tM2 = finState[5];
          TaskTerm totCnt = rb.bin("add", vecCnt, tCnt, "si32");
          TaskTerm safeTotF = rb.castOp(
              rb.sel(rb.cmp("gt", totCnt, rb.zeroCount, "i1"), totCnt,
                     rb.oneCount, "si32"),
              "f32");
          TaskTerm delta = rb.bin("sub", tMean, vecMean, "f32");
          TaskTerm combMean = rb.bin(
              "add", vecMean,
              rb.bin("mul", delta,
                     rb.bin("div", rb.castOp(tCnt, "f32"), safeTotF, "f32"),
                     "f32"),
              "f32");
          TaskTerm combM2 = rb.bin(
              "add", rb.bin("add", vecM2, tM2, "f32"),
              rb.bin("mul", rb.bin("mul", delta, delta, "f32"),
                     rb.bin("div",
                            rb.bin("mul", rb.castOp(vecCnt, "f32"),
                                   rb.castOp(tCnt, "f32"), "f32"),
                            safeTotF, "f32"),
                     "f32"),
              "f32");
          TaskTerm tailEmpty = rb.cmp("eq", tCnt, rb.zeroCount, "i1");
          TaskTerm vecEmpty = rb.cmp("eq", vecCnt, rb.zeroCount, "i1");
          TaskTerm finMean = rb.sel(tailEmpty, vecMean,
                                    rb.sel(vecEmpty, tMean, combMean, "f32"),
                                    "f32");
          TaskTerm finM2 =
              rb.sel(tailEmpty, vecM2,
                     rb.sel(vecEmpty, tM2, combM2, "f32"), "f32");
          rb.records[first->getResult(0)] = {totCnt, finMean, finM2};
          summaryValues.insert(first->getResult(0));
          return success();
        }

        // Vector-accumulated fused sum / max_num stage.
        size_t m = stage.size();
        StringRef bKind = (combine == "sum") ? StringRef("add") : StringRef("max");
        StringRef rKind = (combine == "sum") ? StringRef("add") : StringRef("max");
        TaskTerm initSc = (combine == "sum") ? rb.zeroF32 : rb.minusInfF32;
        TaskTerm initVec = rb.splatOp(initSc, vecF32Ty);
        llvm::DenseSet<Operation *> skipReds(stage.begin(), stage.end());

        auto emitRedInner = [&](ArrayRef<TaskTerm> prefix, TaskTerm colCnt,
                                TaskTerm pCnt, TaskTerm tStart,
                                ArrayRef<TaskTerm> inVecs,
                                ArrayRef<TaskTerm> inTailScs,
                                SmallVectorImpl<TaskTerm> &outVecs,
                                SmallVectorImpl<TaskTerm> &outTailScs)
            -> LogicalResult {
          SmallVector<Value, 8> pOps{rb.zeroIdx.v, pCnt.v, rb.oneIdx.v};
          SmallVector<Attribute, 4> pIterTys;
          SmallVector<Type, 4> pResTys(m, b.getIndexType());
          for (size_t k = 0; k < m; ++k) {
            pOps.push_back(inVecs[k].v);
            pIterTys.push_back(b.getStringAttr(vecF32Ty));
          }
          Operation *pFor = makeOp(
              b, loc, "task_for", pOps, pResTys,
              {b.getNamedAttr("iter_types", b.getArrayAttr(pIterTys))}, 1);
          Block &pBb = pFor->getRegion(0).emplaceBlock();
          TaskTerm pIv{pBb.addArgument(b.getIndexType(), loc), "index"};
          SmallVector<TaskTerm, 4> accVecs;
          for (size_t k = 0; k < m; ++k)
            accVecs.push_back({pBb.addArgument(b.getIndexType(), loc), vecF32Ty});
          {
            OpBuilder::InsertionGuard pg(b);
            b.setInsertionPointToEnd(&pBb);
            TaskTerm off = rb.bin("mul", pIv, rb.panelWidth, "index");
            llvm::DenseMap<Value, TaskTerm> localMap;
            if (failed(rb.emitVectorPanelOps(subOps, {}, rowIv, prefix, off,
                                             rb.panelWidth, w, baseDim, axId,
                                             true, localMap, &skipReds)))
              return failure();
            if (!cacheWriteVals.empty()) {
              std::string u64Ty = regionType("value", "ui64", 1, w, axId);
              std::string ixTy = regionType("value", "si64", 1, w, axId);
              std::string mskTy = regionType("value", "i1", 1, w, axId);
              Value lanesV =
                  makeOp(b, loc, "task_iota", {}, {b.getIndexType()},
                         {b.getNamedAttr("start", b.getI64IntegerAttr(0)),
                          b.getNamedAttr("end", b.getI64IntegerAttr(w)),
                          b.getNamedAttr("val_type", b.getStringAttr(u64Ty))})
                      ->getResult(0);
              TaskTerm addr64 =
                  rb.bin("add", rb.castOp({lanesV, u64Ty}, ixTy),
                         rb.castOp(off, "si64"), ixTy);
              TaskTerm trueVec = rb.splatOp(rb.lit("true", "i1"), mskTy);
              for (auto [cIdx, cv] : llvm::enumerate(cacheWriteVals)) {
                TaskTerm cVal = localMap.at(cv);
                std::string pTy =
                    "!weft_kernel.pointer<" + cacheElemSps[cIdx] + ">";
                makeOp(b, loc, "task_ptr_store",
                       {cachePtrs[cIdx], addr64.v, trueVec.v, cVal.v}, {},
                       {b.getNamedAttr("ptr_type", b.getStringAttr(pTy)),
                        b.getNamedAttr("val_type", b.getStringAttr(cVal.type))});
              }
            }
            SmallVector<Value, 4> yVals;
            for (size_t k = 0; k < m; ++k) {
              TaskTerm inV = localMap.at(stage[k]->getOperand(0));
              TaskTerm nextV = rb.bin(bKind, accVecs[k], inV, vecF32Ty);
              yVals.push_back(nextV.v);
            }
            makeOp(b, loc, "task_yield", yVals, {},
                   {b.getNamedAttr("val_types", b.getArrayAttr(pIterTys))});
          }
          outVecs.clear();
          for (size_t k = 0; k < m; ++k)
            outVecs.push_back({pFor->getResult(k), vecF32Ty});

          outTailScs.clear();
          if (onlyFullPanels) {
            outTailScs.assign(inTailScs.begin(), inTailScs.end());
            return success();
          }

          bool scalarTail = (grouped && subOps.size() >= 64);
          bool splitScalarRem =
              grouped && !scalarTail && tailVec > 1 &&
              canScalarEvalOps(subOps, &skipReds);
          TaskTerm vecTailEnd = colCnt;
          if (splitScalarRem) {
            TaskTerm rem = rb.bin("sub", colCnt, tStart, "index");
            TaskTerm vStep = rb.idx(tailVec);
            TaskTerm vCnt = rb.bin("div", rem, vStep, "index");
            vecTailEnd = rb.bin("add", tStart,
                                rb.bin("mul", vCnt, vStep, "index"), "index");
          }
          TaskTerm tStep = scalarTail ? rb.oneIdx : rb.idx(tailVec);
          SmallVector<Value, 8> tOps{tStart.v, vecTailEnd.v, tStep.v};
          SmallVector<Attribute, 4> tIterTys;
          SmallVector<Type, 4> tResTys(m, b.getIndexType());
          for (size_t k = 0; k < m; ++k) {
            tOps.push_back(inTailScs[k].v);
            tIterTys.push_back(b.getStringAttr("f32"));
          }
          Operation *tFor = makeOp(
              b, loc, "task_for", tOps, tResTys,
              {b.getNamedAttr("iter_types", b.getArrayAttr(tIterTys))}, 1);
          Block &tBb = tFor->getRegion(0).emplaceBlock();
          TaskTerm tIv{tBb.addArgument(b.getIndexType(), loc), "index"};
          SmallVector<TaskTerm, 4> accScs;
          for (size_t k = 0; k < m; ++k)
            accScs.push_back({tBb.addArgument(b.getIndexType(), loc), "f32"});
          {
            OpBuilder::InsertionGuard tg(b);
            b.setInsertionPointToEnd(&tBb);
            llvm::DenseMap<Value, TaskTerm> localMap;
            SmallVector<Value, 4> yVals;
            if (scalarTail) {
              if (failed(evalScalarAt(subOps, prefix, tIv, nullptr, localMap,
                                      &skipReds)))
                return failure();
              for (size_t k = 0; k < m; ++k) {
                TaskTerm part = localMap.at(stage[k]->getOperand(0));
                TaskTerm nextSc = rb.bin(bKind, accScs[k], part, "f32");
                yVals.push_back(nextSc.v);
              }
            } else {
              if (failed(rb.emitVectorPanelOps(subOps, {}, rowIv, prefix, tIv,
                                               rb.panelWidth, int(tailVec),
                                               baseDim, axId, false, localMap,
                                               nullptr)))
                return failure();
              for (size_t k = 0; k < m; ++k) {
                TaskTerm part = localMap.at(stage[k]->getResult(0));
                TaskTerm nextSc = rb.bin(bKind, accScs[k], part, "f32");
                yVals.push_back(nextSc.v);
              }
            }
            makeOp(b, loc, "task_yield", yVals, {},
                   {b.getNamedAttr("val_types", b.getArrayAttr(tIterTys))});
          }
          if (splitScalarRem) {
            SmallVector<Value, 8> rOps{vecTailEnd.v, colCnt.v, rb.oneIdx.v};
            for (size_t k = 0; k < m; ++k)
              rOps.push_back(tFor->getResult(k));
            Operation *rFor = makeOp(
                b, loc, "task_for", rOps, tResTys,
                {b.getNamedAttr("iter_types", b.getArrayAttr(tIterTys))}, 1);
            Block &rBb = rFor->getRegion(0).emplaceBlock();
            TaskTerm rIv{rBb.addArgument(b.getIndexType(), loc), "index"};
            SmallVector<TaskTerm, 4> rAccs;
            for (size_t k = 0; k < m; ++k)
              rAccs.push_back({rBb.addArgument(b.getIndexType(), loc), "f32"});
            {
              OpBuilder::InsertionGuard rg(b);
              b.setInsertionPointToEnd(&rBb);
              llvm::DenseMap<Value, TaskTerm> localMap;
              if (failed(evalScalarAt(subOps, prefix, rIv, nullptr, localMap,
                                      &skipReds)))
                return failure();
              SmallVector<Value, 4> yVals;
              for (size_t k = 0; k < m; ++k) {
                TaskTerm part = localMap.at(stage[k]->getOperand(0));
                TaskTerm nextSc = rb.bin(bKind, rAccs[k], part, "f32");
                yVals.push_back(nextSc.v);
              }
              makeOp(b, loc, "task_yield", yVals, {},
                     {b.getNamedAttr("val_types", b.getArrayAttr(tIterTys))});
            }
            for (size_t k = 0; k < m; ++k)
              outTailScs.push_back({rFor->getResult(k), "f32"});
          } else {
            for (size_t k = 0; k < m; ++k)
              outTailScs.push_back({tFor->getResult(k), "f32"});
          }
          return success();
        };

        SmallVector<TaskTerm, 4> initVecs(m, initVec), initScs(m, initSc),
            finVecs, finTailScs;
        if (!vecOwnerInner) {
          if (failed(emitRedInner({rowIv}, rb.columns, panelCount, tailStart,
                                  initVecs, initScs, finVecs, finTailScs)))
            return failure();
          for (size_t k = 0; k < m; ++k) {
            Value redV =
                makeOp(b, loc, "task_reduce", {finVecs[k].v},
                       {b.getIndexType()},
                       {b.getNamedAttr("kind", b.getStringAttr(rKind)),
                        b.getNamedAttr("axis", b.getI64IntegerAttr(0)),
                        b.getNamedAttr("in_type", b.getStringAttr(vecF32Ty)),
                        b.getNamedAttr("val_type", b.getStringAttr("f32"))})
                    ->getResult(0);
            TaskTerm total =
                onlyFullPanels
                    ? TaskTerm{redV, "f32"}
                    : rb.bin(bKind, {redV, "f32"}, finTailScs[k], "f32");
            rb.terms[stage[k]->getResult(0)] = total;
            summaryValues.insert(stage[k]->getResult(0));
          }
        } else {
          // Carry only scalar f32 accumulators across outerIv so vector
          // accumulators never cross inner scf.for control boundaries.
          SmallVector<Value, 8> oOps{rb.zeroIdx.v, outerTotal.v, rb.oneIdx.v};
          SmallVector<Attribute, 4> oTys;
          SmallVector<Type, 4> oResTys(m, b.getIndexType());
          for (size_t k = 0; k < m; ++k) {
            oOps.push_back(initSc.v);
            oTys.push_back(b.getStringAttr("f32"));
          }
          Operation *oFor = makeOp(
              b, loc, "task_for", oOps, oResTys,
              {b.getNamedAttr("iter_types", b.getArrayAttr(oTys))}, 1);
          Block &oBb = oFor->getRegion(0).emplaceBlock();
          TaskTerm outerIv{oBb.addArgument(b.getIndexType(), loc), "index"};
          SmallVector<TaskTerm, 4> cScs;
          for (size_t k = 0; k < m; ++k)
            cScs.push_back({oBb.addArgument(b.getIndexType(), loc), "f32"});
          {
            OpBuilder::InsertionGuard og(b);
            b.setInsertionPointToEnd(&oBb);
            auto prefix = buildOwnerPrefix(outerIv);
            SmallVector<TaskTerm, 4> zVecs(m, initVec), zScs(m, initSc),
                stepVecs, stepScs;
            if (failed(emitRedInner(prefix, innerCols, innerPanelCount,
                                    innerTailStart, zVecs, zScs, stepVecs,
                                    stepScs)))
              return failure();
            SmallVector<Value, 4> yVals;
            for (size_t k = 0; k < m; ++k) {
              Value redV =
                  makeOp(b, loc, "task_reduce", {stepVecs[k].v},
                         {b.getIndexType()},
                         {b.getNamedAttr("kind", b.getStringAttr(rKind)),
                          b.getNamedAttr("axis", b.getI64IntegerAttr(0)),
                          b.getNamedAttr("in_type", b.getStringAttr(vecF32Ty)),
                          b.getNamedAttr("val_type", b.getStringAttr("f32"))})
                      ->getResult(0);
              TaskTerm sliceTotal =
                  onlyFullPanels
                      ? TaskTerm{redV, "f32"}
                      : rb.bin(bKind, {redV, "f32"}, stepScs[k], "f32");
              TaskTerm nextSc = rb.bin(bKind, cScs[k], sliceTotal, "f32");
              yVals.push_back(nextSc.v);
            }
            makeOp(b, loc, "task_yield", yVals, {},
                   {b.getNamedAttr("val_types", b.getArrayAttr(oTys))});
          }
          for (size_t k = 0; k < m; ++k) {
            rb.terms[stage[k]->getResult(0)] = {oFor->getResult(k), "f32"};
            summaryValues.insert(stage[k]->getResult(0));
          }
        }
        return success();
      };

      auto emitStoreSubset =
          [&](ArrayRef<Operation *> activeWrites, bool includeOwned,
              ArrayRef<Value> cacheReadVals,
              bool onlyFullPanels) -> LogicalResult {
        if (includeOwned && !ownedWrites.empty()) {
          SweepDesc pubSw;
          pubSw.combine = "publish";
          pubSw.anchor = ownedWrites.front();
          pubSw.writes = ownedWrites;
          llvm::DenseSet<Operation *> visited;
          SmallVector<Operation *> subOps;
          for (Operation *w : ownedWrites)
            rb.collectSubgraph(w->getOperand(2).getDefiningOp(), summaryValues,
                               visited, subOps);
          if (failed(rb.emitScalarSweep(pubSw, subOps, rowIv)))
            return failure();
        }

        if (activeWrites.empty())
          return success();

        llvm::DenseSet<Value> stopSet = summaryValues;
        for (Value cv : cacheReadVals)
          stopSet.insert(cv);

        SweepDesc sw;
        sw.combine = "store";
        sw.anchor = activeWrites.front();
        sw.writes.assign(activeWrites.begin(), activeWrites.end());
        llvm::DenseSet<Operation *> visited;
        SmallVector<Operation *> subOps;
        for (Operation *w : activeWrites)
          rb.collectSubgraph(w->getOperand(2).getDefiningOp(), stopSet, visited,
                             subOps);

        for (Operation *so : subOps) {
          StringRef sn = so->getName().stripDialect();
          if (sn == "count_keys")
            sw.privateBins = true;
          if (sn == "column_sum")
            sw.verticalSum = true;
          if (sn == "ordered_sum")
            sw.segmentLoop = true;
          if (sn == "scan_sum")
            sw.prefixCarry = true;
        }

        if (sw.privateBins || sw.verticalSum) {
          Operation *keyOp = subOps.front();
          Value src = keyOp->getOperand(0);
          Value dst = sw.writes.front()->getOperand(0);
          unsigned srcSlot = cast<BlockArgument>(src).getArgNumber() - 1;
          TaskTerm size = rb.idx(slotFacts[srcSlot].indirectExtents[0]);
          TaskTerm zeroI32 = rb.lit("0", "si32");
          if (sw.privateBins) {
            Operation *bFor = makeOp(
                b, loc, "task_for", {rb.zeroIdx.v, rb.columns.v, rb.oneIdx.v},
                {}, {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
            Block &bBb = bFor->getRegion(0).emplaceBlock();
            TaskTerm binIv{bBb.addArgument(b.getIndexType(), loc), "index"};
            {
              OpBuilder::InsertionGuard bg(b);
              b.setInsertionPointToEnd(&bBb);
              rb.commitOp(zeroI32, rb.scalarSlice(dst, {rowIv, binIv}, false));
              makeOp(b, loc, "task_yield", {}, {},
                     {b.getNamedAttr("val_types", b.getArrayAttr({}))});
            }
            TaskTerm span =
                rb.idx(keyOp->getAttrOfType<IntegerAttr>("span").getInt());
            TaskTerm rGlob = rb.bin("add", rb.origin, rowIv, "index");
            TaskTerm start = rb.bin("mul", rGlob, span, "index");
            TaskTerm limit = rb.bin("add", start, span, "index");
            TaskTerm shorter = rb.cmp("lt", limit, size, "i1");
            TaskTerm end = rb.sel(shorter, limit, size, "index");
            Operation *sFor = makeOp(
                b, loc, "task_for", {start.v, end.v, rb.oneIdx.v}, {},
                {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
            Block &sBb = sFor->getRegion(0).emplaceBlock();
            TaskTerm sampleIv{sBb.addArgument(b.getIndexType(), loc), "index"};
            {
              OpBuilder::InsertionGuard sg(b);
              b.setInsertionPointToEnd(&sBb);
              TaskTerm val = rb.scalarSlice(
                  src, {sampleIv}, true,
                  weftSpelling(
                      cast<MemRefType>(src.getType()).getElementType()));
              TaskTerm binI32 =
                  val.type == "f32" ? rb.castOp(val, "si32") : val;
              TaskTerm k = rb.castOp(binI32, "index");
              TaskTerm lo = rb.cmp("ge", k, rb.zeroIdx, "i1");
              TaskTerm hi = rb.cmp("lt", k, rb.columns, "i1");
              TaskTerm keep = rb.bin("and", lo, hi, "i1");
              Operation *ifOp =
                  makeOp(b, loc, "task_if", {keep.v}, {},
                         {b.getNamedAttr("res_types", b.getArrayAttr({}))}, 2);
              Block &tBb = ifOp->getRegion(0).emplaceBlock();
              {
                OpBuilder::InsertionGuard ig(b);
                b.setInsertionPointToEnd(&tBb);
                TaskTerm ptrSl = rb.scalarSlice(dst, {rowIv, k}, false);
                TaskTerm old = rb.scalarSlice(dst, {rowIv, k}, true, "si32");
                TaskTerm oneI32 = rb.lit("1", "si32");
                rb.commitOp(rb.bin("add", old, oneI32, "si32"), ptrSl);
                makeOp(b, loc, "task_yield", {}, {},
                       {b.getNamedAttr("val_types", b.getArrayAttr({}))});
              }
              makeOp(b, loc, "task_yield", {}, {},
                     {b.getNamedAttr("val_types", b.getArrayAttr({}))});
            }
          } else {
            Operation *bFor = makeOp(
                b, loc, "task_for", {rb.zeroIdx.v, rb.columns.v, rb.oneIdx.v},
                {}, {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
            Block &bBb = bFor->getRegion(0).emplaceBlock();
            TaskTerm binIv{bBb.addArgument(b.getIndexType(), loc), "index"};
            {
              OpBuilder::InsertionGuard bg(b);
              b.setInsertionPointToEnd(&bBb);
              Operation *pFor = makeOp(
                  b, loc, "task_for",
                  {rb.zeroIdx.v, size.v, rb.oneIdx.v, zeroI32.v},
                  {b.getIndexType()},
                  {b.getNamedAttr("iter_types",
                                  b.getArrayAttr({b.getStringAttr("si32")}))},
                  1);
              Block &pBb = pFor->getRegion(0).emplaceBlock();
              TaskTerm partIv{pBb.addArgument(b.getIndexType(), loc), "index"};
              TaskTerm cntCarry{pBb.addArgument(b.getIndexType(), loc), "si32"};
              {
                OpBuilder::InsertionGuard pg(b);
                b.setInsertionPointToEnd(&pBb);
                TaskTerm x =
                    rb.scalarSlice(src, {partIv, binIv}, true, "si32");
                TaskTerm sum = rb.bin("add", cntCarry, x, "si32");
                makeOp(b, loc, "task_yield", {sum.v}, {},
                       {b.getNamedAttr(
                           "val_types",
                           b.getArrayAttr({b.getStringAttr("si32")}))});
              }
              TaskTerm total{pFor->getResult(0), "si32"};
              std::string dstElem = weftSpelling(
                  cast<MemRefType>(dst.getType()).getElementType());
              if (dstElem != total.type)
                total = rb.castOp(total, dstElem);
              rb.commitOp(total, rb.scalarSlice(dst, {binIv}, false));
              makeOp(b, loc, "task_yield", {}, {},
                     {b.getNamedAttr("val_types", b.getArrayAttr({}))});
            }
          }
          return success();
        }

        if (scalarMembers || sw.prefixCarry || whole ||
            (sw.segmentLoop && !canVectorizeSegmentSweep(task, subOps)))
          return rb.emitScalarSweep(sw, subOps, rowIv);

        int axId = vecOwnerInner ? ownerVecAxisId : 2;
        int baseDim = vecOwnerInner ? ownerBaseColDim : -2;
        auto emitStorePanels = [&](ArrayRef<TaskTerm> prefix, TaskTerm colCnt,
                                   TaskTerm pCnt, TaskTerm tStart)
            -> LogicalResult {
          for (bool tail : {false, true}) {
            if (tail && onlyFullPanels)
              continue;
            bool scalarTail =
                tail && ((grouped && subOps.size() >= 64) || sw.segmentLoop);
            bool splitScalarRem =
                tail && grouped && !scalarTail && tailVec > 1 &&
                canScalarEvalOps(subOps);
            TaskTerm vecTailEnd = colCnt;
            if (splitScalarRem) {
              TaskTerm rem = rb.bin("sub", colCnt, tStart, "index");
              TaskTerm vStep = rb.idx(tailVec);
              TaskTerm vCnt = rb.bin("div", rem, vStep, "index");
              vecTailEnd = rb.bin("add", tStart,
                                  rb.bin("mul", vCnt, vStep, "index"), "index");
            }
            TaskTerm lb = tail ? tStart : rb.zeroIdx;
            TaskTerm ub = tail ? vecTailEnd : pCnt;
            TaskTerm step =
                tail ? (scalarTail ? rb.oneIdx : rb.idx(tailVec)) : rb.oneIdx;
            int dim = tail ? int(tailVec) : (fixed ? int(fixed) : -3);
            Operation *pFor = makeOp(
                b, loc, "task_for", {lb.v, ub.v, step.v}, {},
                {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
            Block &pBb = pFor->getRegion(0).emplaceBlock();
            TaskTerm pIv{pBb.addArgument(b.getIndexType(), loc), "index"};
            {
              OpBuilder::InsertionGuard pg(b);
              b.setInsertionPointToEnd(&pBb);
              TaskTerm off =
                  tail ? pIv : rb.bin("mul", pIv, rb.panelWidth, "index");
              llvm::DenseMap<Value, TaskTerm> localMap;
              if (scalarTail && !sw.segmentLoop) {
                if (failed(evalScalarAt(subOps, prefix, pIv, nullptr, localMap)))
                  return failure();
                for (Operation *wPtr : sw.writes) {
                  auto vt = cast<MemRefType>(wPtr->getOperand(0).getType());
                  SmallVector<TaskTerm, 4> coords;
                  if (vt.getRank() >= 2)
                    coords.append(prefix.begin(), prefix.end());
                  coords.push_back(pIv);
                  TaskTerm val =
                      localMap.count(wPtr->getOperand(2))
                          ? localMap.at(wPtr->getOperand(2))
                          : rb.terms.lookup(wPtr->getOperand(2));
                  rb.commitOp(val, rb.scalarSlice(wPtr->getOperand(0), coords,
                                                  false));
                }
              } else {
                if (!tail && !cacheReadVals.empty()) {
                  std::string u64Ty = regionType("value", "ui64", 1, dim, axId);
                  std::string ixTy = regionType("value", "si64", 1, dim, axId);
                  std::string mskTy = regionType("value", "i1", 1, dim, axId);
                  Value lanesV =
                      makeOp(b, loc, "task_iota", {}, {b.getIndexType()},
                             {b.getNamedAttr("start", b.getI64IntegerAttr(0)),
                              b.getNamedAttr("end", b.getI64IntegerAttr(dim)),
                              b.getNamedAttr("val_type", b.getStringAttr(u64Ty))})
                          ->getResult(0);
                  TaskTerm addr64 =
                      rb.bin("add", rb.castOp({lanesV, u64Ty}, ixTy),
                             rb.castOp(off, "si64"), ixTy);
                  TaskTerm trueVec = rb.splatOp(rb.lit("true", "i1"), mskTy);
                  for (auto [cIdx, cv] : llvm::enumerate(cacheReadVals)) {
                    std::string eSp = cacheElemSps[cIdx];
                    std::string pTy = "!weft_kernel.pointer<" + eSp + ">";
                    std::string cValTy = regionType("value", eSp, 1, dim, axId);
                    TaskTerm zVec = rb.splatOp(rb.lit("0.0", eSp), cValTy);
                    Value ldV =
                        makeOp(b, loc, "task_ptr_load",
                               {cachePtrs[cIdx], addr64.v, trueVec.v, zVec.v},
                               {b.getIndexType()},
                               {b.getNamedAttr("ptr_type", b.getStringAttr(pTy)),
                                b.getNamedAttr("val_type",
                                               b.getStringAttr(cValTy))})
                            ->getResult(0);
                    localMap[cv] = {ldV, cValTy};
                  }
                }
                if (failed(rb.emitVectorPanelOps(
                        subOps, sw.writes, rowIv, prefix, off, rb.panelWidth,
                        scalarTail ? 1 : dim, baseDim, axId, !tail && fixed > 0,
                        localMap)))
                  return failure();
              }
              makeOp(b, loc, "task_yield", {}, {},
                     {b.getNamedAttr("val_types", b.getArrayAttr({}))});
            }
            if (splitScalarRem) {
              Operation *rFor = makeOp(
                  b, loc, "task_for", {vecTailEnd.v, colCnt.v, rb.oneIdx.v}, {},
                  {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
              Block &rBb = rFor->getRegion(0).emplaceBlock();
              TaskTerm rIv{rBb.addArgument(b.getIndexType(), loc), "index"};
              OpBuilder::InsertionGuard rg(b);
              b.setInsertionPointToEnd(&rBb);
              llvm::DenseMap<Value, TaskTerm> localMap;
              if (failed(evalScalarAt(subOps, prefix, rIv, nullptr, localMap)))
                return failure();
              for (Operation *wPtr : sw.writes) {
                auto vt = cast<MemRefType>(wPtr->getOperand(0).getType());
                SmallVector<TaskTerm, 4> coords;
                if (vt.getRank() >= 2)
                  coords.append(prefix.begin(), prefix.end());
                coords.push_back(rIv);
                TaskTerm val =
                    localMap.count(wPtr->getOperand(2))
                        ? localMap.at(wPtr->getOperand(2))
                        : rb.terms.lookup(wPtr->getOperand(2));
                rb.commitOp(val,
                            rb.scalarSlice(wPtr->getOperand(0), coords, false));
              }
              makeOp(b, loc, "task_yield", {}, {},
                     {b.getNamedAttr("val_types", b.getArrayAttr({}))});
            }
          }
          return success();
        };

        if (!vecOwnerInner)
          return emitStorePanels({rowIv}, rb.columns, panelCount, tailStart);

        Operation *oFor = makeOp(
            b, loc, "task_for", {rb.zeroIdx.v, outerTotal.v, rb.oneIdx.v}, {},
            {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
        Block &oBb = oFor->getRegion(0).emplaceBlock();
        TaskTerm outerIv{oBb.addArgument(b.getIndexType(), loc), "index"};
        {
          OpBuilder::InsertionGuard og(b);
          b.setInsertionPointToEnd(&oBb);
          auto prefix = buildOwnerPrefix(outerIv);
          if (failed(emitStorePanels(prefix, innerCols, innerPanelCount,
                                     innerTailStart)))
            return failure();
          makeOp(b, loc, "task_yield", {}, {},
                 {b.getNamedAttr("val_types", b.getArrayAttr({}))});
        }
        return success();
      };

      if (!cachedCut.empty()) {
        for (size_t sIdx = 0; sIdx + 1 < redStages.size(); ++sIdx) {
          if (failed(emitReductionStage(redStages[sIdx], {},
                                        staticExactInnerPanels)) ||
              failed(emitAvailableScalars()))
            return failure();
        }
        const auto &lastStage = redStages.back();
        if (staticExactInnerPanels && staticInnerCols > 0 &&
            staticInnerCols <= maxRowCacheCols) {
          if (failed(emitReductionStage(lastStage, cachedCut, true)) ||
              failed(emitAvailableScalars()) ||
              failed(emitStoreSubset(writes, true, cachedCut, true)))
            return failure();
        } else {
          TaskTerm fitsLen =
              rb.cmp("le", rb.columns, rb.idx(maxRowCacheCols), "i1");
          TaskTerm exactPanels = rb.cmp("eq", tailStart, rb.columns, "i1");
          TaskTerm canCache = rb.bin("and", fitsLen, exactPanels, "i1");
          auto savedTerms = rb.terms;
          auto savedRecords = rb.records;
          auto savedSummaries = summaryValues;

          Operation *ifCache =
              makeOp(b, loc, "task_if", {canCache.v}, {},
                     {b.getNamedAttr("res_types", b.getArrayAttr({}))}, 2);
          Block &thenBb = ifCache->getRegion(0).emplaceBlock();
          Block &elseBb = ifCache->getRegion(1).emplaceBlock();
          {
            OpBuilder::InsertionGuard tg(b);
            b.setInsertionPointToEnd(&thenBb);
            if (failed(emitReductionStage(lastStage, cachedCut, true)) ||
                failed(emitAvailableScalars()) ||
                failed(emitStoreSubset(writes, true, cachedCut, true)))
              return failure();
            makeOp(b, loc, "task_yield", {}, {},
                   {b.getNamedAttr("val_types", b.getArrayAttr({}))});
          }
          rb.terms = std::move(savedTerms);
          rb.records = std::move(savedRecords);
          summaryValues = std::move(savedSummaries);
          {
            OpBuilder::InsertionGuard eg(b);
            b.setInsertionPointToEnd(&elseBb);
            if (failed(emitReductionStage(lastStage, {}, false)) ||
                failed(emitAvailableScalars()) ||
                failed(emitStoreSubset(writes, true, {}, false)))
              return failure();
            makeOp(b, loc, "task_yield", {}, {},
                   {b.getNamedAttr("val_types", b.getArrayAttr({}))});
          }
        }
      } else {
        llvm::DenseSet<Operation *> doneReds, doneWrites;
        auto writeReadyAndUsesRed = [&](Operation *w, bool &usesAnyRed) {
          usesAnyRed = false;
          llvm::DenseSet<Operation *> seen;
          SmallVector<Operation *, 16> work;
          if (auto *def = w->getOperand(2).getDefiningOp();
              def && def->getBlock() == &bb)
            work.push_back(def);
          while (!work.empty()) {
            Operation *cur = work.pop_back_val();
            if (!seen.insert(cur).second)
              continue;
            if (isReductionOp(cur)) {
              usesAnyRed = true;
              if (!doneReds.contains(cur))
                return false;
              continue;
            }
            for (Value in : cur->getOperands())
              if (auto *def = in.getDefiningOp(); def && def->getBlock() == &bb)
                work.push_back(def);
          }
          return true;
        };

        const size_t maxFusedStores =
            allOps.size() > 128
                ? 1
                : ((grouped && allOps.size() > 32) ? 2 : 4);
        for (size_t sIdx = 0; sIdx < redStages.size(); ++sIdx) {
          const auto &stage = redStages[sIdx];
          if (failed(emitReductionStage(stage, {}, false)) ||
              failed(emitAvailableScalars()))
            return failure();
          for (Operation *ro : stage)
            doneReds.insert(ro);

          if (redStages.size() > 1 && writes.size() > maxFusedStores &&
              !hasSpecialSweep) {
            SmallVector<Operation *, 4> ready;
            for (Operation *w : writes) {
              if (doneWrites.contains(w))
                continue;
              bool usesRed = false;
              if (writeReadyAndUsesRed(w, usesRed) && usesRed)
                ready.push_back(w);
            }
            for (size_t i = 0; i < ready.size(); i += maxFusedStores) {
              auto chunk = ArrayRef<Operation *>(ready).slice(
                  i, std::min(maxFusedStores, ready.size() - i));
              if (failed(emitStoreSubset(chunk, false, {}, false)))
                return failure();
              for (Operation *cw : chunk)
                doneWrites.insert(cw);
            }
          }
        }

        SmallVector<Operation *> remWrites;
        for (Operation *w : writes)
          if (!doneWrites.contains(w))
            remWrites.push_back(w);
        if (remWrites.empty()) {
          if (failed(emitStoreSubset({}, true, {}, false)))
            return failure();
        } else if (remWrites.size() <= maxFusedStores || hasSpecialSweep) {
          if (failed(emitStoreSubset(remWrites, true, {}, false)))
            return failure();
        } else {
          if (failed(emitStoreSubset({}, true, {}, false)))
            return failure();
          for (size_t i = 0; i < remWrites.size(); i += maxFusedStores) {
            auto chunk = ArrayRef<Operation *>(remWrites).slice(
                i, std::min(maxFusedStores, remWrites.size() - i));
            if (failed(emitStoreSubset(chunk, false, {}, false)))
              return failure();
          }
        }
      }
    }

    makeOp(b, loc, "task_yield", {}, {},
           {b.getNamedAttr("val_types", b.getArrayAttr({}))});
  }
  b.setInsertionPointToEnd(&tkBb);
  makeOp(b, loc, "end");

  // ---------------- Region 1: host.c ----------------
  bool members = task.hasAttr("cpu.member_tasks");
  bool singleDispatcher = task.hasAttr("cpu.single_dispatcher");
  bool projected = false;
  for (const auto &sf : slotFacts)
    projected |= (sf.rank > 2);
  if (ownerAxis)
    projected = false;

  SmallVector<std::string> params;
  SmallVector<Attribute> strideExprs;
  for (auto [i, entry] : llvm::enumerate(abi)) {
    const auto *a = entry.getAsObject();
    std::string n = "a" + std::to_string(i);
    int64_t rank = slotFacts[i].rank;
    StringRef dt = *a->getString("dtype");
    std::string cType = dt == "f16"    ? "_Float16"
                        : dt == "bf16" ? "uint16_t"
                        : dt == "i32"  ? "int32_t"
                        : dt == "i64"  ? "int64_t"
                        : dt == "u64"  ? "uint64_t"
                                       : "float";
    params.push_back(cType + (rank ? " *" : " ") + n);
    if (rank > 2) {
      for (int64_t j = 0; j < rank - 1; ++j)
        params.push_back("int64_t " + n + "d" + std::to_string(j));
    }
    for (int64_t axis = 0; axis < rank; ++axis) {
      std::string stride = n + "s" + std::to_string(axis);
      params.push_back("int64_t " + stride);
      if (!slotFacts[i].contiguous && (ownerAxis || rank <= 2 || axis >= rank - 2)) {
        strideExprs.push_back(b.getStringAttr(
            slotFacts[i].swapAxes ? (n + "s" + std::to_string(1 - axis))
                                  : stride));
      }
    }
  }
  params.push_back(
      "int64_t rows, int64_t cols, int64_t grain, int64_t panel, int64_t "
      "workers");

  Block &hOuter = prog->getRegion(1).emplaceBlock();
  b.setInsertionPointToEnd(&hOuter);
  Operation *hFn = makeOp(
      b, loc, "host_row_fn", {}, {},
      {b.getNamedAttr("params", b.getStringAttr(llvm::join(params, ", ")))},
      1);
  makeOp(b, loc, "end");

  Block &hFnBb = hFn->getRegion(0).emplaceBlock();
  b.setInsertionPointToEnd(&hFnBb);
  auto emitRowGuard = [&](StringRef cond, int32_t code) {
    makeOp(b, loc, "host_guard_return", {}, {},
           {b.getNamedAttr("cond", b.getStringAttr(cond)),
            b.getNamedAttr("code", b.getI32IntegerAttr(code))});
  };
  auto emitRowLet = [&](StringRef name, StringRef expr, bool isMut = false) {
    makeOp(b, loc, "host_let_i64", {}, {},
           {b.getNamedAttr("name", b.getStringAttr(name)),
            b.getNamedAttr("expr", b.getStringAttr(expr)),
            b.getNamedAttr("is_mut", b.getBoolAttr(isMut))});
  };
  auto emitRowAssign = [&](StringRef dst, StringRef src, ValueRange vals = {}) {
    makeOp(b, loc, "host_elem_op", vals, {},
           {b.getNamedAttr("op", b.getStringAttr("assign")),
            b.getNamedAttr("dst_expr", b.getStringAttr(dst)),
            b.getNamedAttr("src_expr", b.getStringAttr(src)),
            b.getNamedAttr("aux_expr", b.getStringAttr("")),
            b.getNamedAttr("elem_bytes", b.getI64IntegerAttr(8))});
  };
  auto emitRowDef = [&](StringRef expr) -> Value {
    return makeOp(b, loc, "host_def_i64", {}, {b.getIndexType()},
                  {b.getNamedAttr("expr", b.getStringAttr(expr))})
        ->getResult(0);
  };
  emitRowGuard("rows < 0 || cols < 0 || grain <= 0 || panel < 0 || workers <= 0",
               1);
  emitRowGuard("!rows", 0);
  if (walk == "destination") {
    emitRowLet("t_rows", "rows");
    emitRowAssign("rows", "cols");
    emitRowAssign("cols", "t_rows");
    emitRowLet("t_a0s", "a0s0");
    emitRowAssign("a0s0", "a0s1");
    emitRowAssign("a0s1", "t_a0s");
    emitRowLet("t_a1s", "a1s0");
    emitRowAssign("a1s0", "a1s1");
    emitRowAssign("a1s1", "t_a1s");
  }
  if (members)
    emitRowAssign("rows", "cols");
  if (singleDispatcher)
    emitRowAssign("workers", "1");

  auto cPtrTypeFor = [&](unsigned i) -> std::string {
    const auto *a = abi[i].getAsObject();
    StringRef dt = *a->getString("dtype");
    std::string base = dt == "f16"    ? "_Float16"
                       : dt == "bf16" ? "uint16_t"
                       : dt == "i32"  ? "int32_t"
                       : dt == "i64"  ? "int64_t"
                       : dt == "u64"  ? "uint64_t"
                                      : "float";
    bool isConst = a->getString("direction") == "In";
    return (isConst ? "const " : "") + base + " *";
  };

  if (walk == "tile") {
    int64_t tileW = task.getAttrOfType<IntegerAttr>("cpu.tile").getInt();
    std::string st = std::to_string(tileW);
    emitRowLet("nt", "(cols - 1) / " + st + " + 1");
    emitRowLet("mt", "(rows - 1) / " + st + " + 1");
    Operation *pTiles =
        makeOp(b, loc, "host_omp_for", {}, {},
               {b.getNamedAttr("iv", b.getStringAttr("g")),
                b.getNamedAttr("lb", b.getStringAttr("0")),
                b.getNamedAttr("ub", b.getStringAttr("mt * nt")),
                b.getNamedAttr("step", b.getStringAttr("1")),
                b.getNamedAttr("parallel", b.getBoolAttr(true)),
                b.getNamedAttr("simd", b.getBoolAttr(false)),
                b.getNamedAttr("x86_simd", b.getBoolAttr(false)),
                b.getNamedAttr("schedule", b.getStringAttr("static")),
                b.getNamedAttr("num_threads", b.getStringAttr("workers")),
                b.getNamedAttr("if_cond", b.getStringAttr("workers > 1"))},
               1);
    makeOp(b, loc, "end");
    Block &pBb = pTiles->getRegion(0).emplaceBlock();
    b.setInsertionPointToEnd(&pBb);
    Value rV = emitRowDef("(g / nt) * " + st);
    Value cV = emitRowDef("(g % nt) * " + st);
    Value nrV = makeOp(b, loc, "host_def_i64", {rV}, {b.getIndexType()},
                       {b.getNamedAttr("expr",
                                       b.getStringAttr("rows - $0 < " + st +
                                                       " ? rows - $0 : " + st))})
                    ->getResult(0);
    Value ncV = makeOp(b, loc, "host_def_i64", {cV}, {b.getIndexType()},
                       {b.getNamedAttr("expr",
                                       b.getStringAttr("cols - $0 < " + st +
                                                       " ? cols - $0 : " + st))})
                    ->getResult(0);

    Value off0 =
        makeOp(b, loc, "host_row_offset", {rV, cV}, {b.getIndexType()},
               {b.getNamedAttr("mode", b.getStringAttr("tile2d")),
                b.getNamedAttr("dim_vars", b.getArrayAttr({})),
                b.getNamedAttr("stride_vars",
                               b.getArrayAttr({b.getStringAttr("a0s0"),
                                               b.getStringAttr("a0s1")}))})
            ->getResult(0);
    makeOp(b, loc, "host_view_ptr", {off0}, {},
           {b.getNamedAttr("index", b.getI64IntegerAttr(0)),
            b.getNamedAttr("c_ptr_type", b.getStringAttr(cPtrTypeFor(0))),
            b.getNamedAttr("base_expr", b.getStringAttr("a0")),
            b.getNamedAttr("byte_offset", b.getI64IntegerAttr(0))});
    Value off1 =
        makeOp(b, loc, "host_row_offset", {cV, rV}, {b.getIndexType()},
               {b.getNamedAttr("mode", b.getStringAttr("tile2d")),
                b.getNamedAttr("dim_vars", b.getArrayAttr({})),
                b.getNamedAttr("stride_vars",
                               b.getArrayAttr({b.getStringAttr("a1s0"),
                                               b.getStringAttr("a1s1")}))})
            ->getResult(0);
    makeOp(b, loc, "host_view_ptr", {off1}, {},
           {b.getNamedAttr("index", b.getI64IntegerAttr(1)),
            b.getNamedAttr("c_ptr_type", b.getStringAttr(cPtrTypeFor(1))),
            b.getNamedAttr("base_expr", b.getStringAttr("a1")),
            b.getNamedAttr("byte_offset", b.getI64IntegerAttr(0))});
    makeOp(b, loc, "host_call_row_task", {nrV, ncV}, {},
           {b.getNamedAttr("sym_name", b.getStringAttr("row_task")),
            b.getNamedAttr("slot_args",
                           b.getArrayAttr({b.getStringAttr("p0"),
                                           b.getStringAttr("p1")})),
            b.getNamedAttr(
                "shape_exprs",
                b.getArrayAttr({b.getStringAttr("$0"), b.getStringAttr("$1"),
                                b.getStringAttr("panel ? panel : $1")})),
            b.getNamedAttr(
                "stride_exprs",
                b.getArrayAttr(
                    {b.getStringAttr("a0s0"), b.getStringAttr("a0s1"),
                     b.getStringAttr("a1s1"), b.getStringAttr("a1s0")}))});
    makeOp(b, loc, "end");
  } else {
    auto emitDispatchBody = [&](Value startV, Value countV, Value loV,
                                Value hiV) {
      SmallVector<Attribute> slotArgs;
      for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
        std::string n = "a" + std::to_string(i);
        int64_t rank = slotFacts[i].rank;
        if (rank == 0) {
          slotArgs.push_back(b.getStringAttr(n));
          continue;
        }
        slotArgs.push_back(b.getStringAttr("p" + std::to_string(i)));
        SmallVector<Value, 1> ptrOff;
        if (slotFacts[i].tableRows || !slotFacts[i].indirectExtents.empty()) {
          // Base pointer unshifted
        } else if ((members && slotFacts[i].rank == 1) ||
                   slotFacts[i].taskIndices) {
          Value off =
              makeOp(b, loc, "host_row_offset", {startV}, {b.getIndexType()},
                     {b.getNamedAttr("mode", b.getStringAttr("linear")),
                      b.getNamedAttr("dim_vars", b.getArrayAttr({})),
                      b.getNamedAttr("stride_vars",
                                     b.getArrayAttr({b.getStringAttr(n + "s0")}))})
                  ->getResult(0);
          ptrOff.push_back(off);
        } else if (ownerAxis) {
          std::string stVar =
              n + "s" + std::to_string(rank == 1 ? 0 : *ownerAxis);
          Value off =
              makeOp(b, loc, "host_row_offset", {startV}, {b.getIndexType()},
                     {b.getNamedAttr("mode", b.getStringAttr("linear")),
                      b.getNamedAttr("dim_vars", b.getArrayAttr({})),
                      b.getNamedAttr("stride_vars",
                                     b.getArrayAttr({b.getStringAttr(stVar)}))})
                  ->getResult(0);
          ptrOff.push_back(off);
        } else if (rank > 2) {
          SmallVector<Attribute, 4> dVars, sVars;
          for (int64_t j = 0; j < rank - 1; ++j) {
            dVars.push_back(b.getStringAttr(n + "d" + std::to_string(j)));
            sVars.push_back(b.getStringAttr(n + "s" + std::to_string(j)));
          }
          Value off = makeOp(b, loc, "host_row_offset", {startV},
                             {b.getIndexType()},
                             {b.getNamedAttr("mode",
                                             b.getStringAttr("projected")),
                              b.getNamedAttr("dim_vars", b.getArrayAttr(dVars)),
                              b.getNamedAttr("stride_vars",
                                             b.getArrayAttr(sVars))})
                          ->getResult(0);
          ptrOff.push_back(off);
        } else if (rank == 2) {
          std::string stVar = n + (slotFacts[i].swapAxes ? "s1" : "s0");
          Value off =
              makeOp(b, loc, "host_row_offset", {startV}, {b.getIndexType()},
                     {b.getNamedAttr("mode", b.getStringAttr("linear")),
                      b.getNamedAttr("dim_vars", b.getArrayAttr({})),
                      b.getNamedAttr("stride_vars",
                                     b.getArrayAttr({b.getStringAttr(stVar)}))})
                  ->getResult(0);
          ptrOff.push_back(off);
        }
        makeOp(b, loc, "host_view_ptr", ptrOff, {},
               {b.getNamedAttr("index", b.getI64IntegerAttr(i)),
                b.getNamedAttr("c_ptr_type", b.getStringAttr(cPtrTypeFor(i))),
                b.getNamedAttr("base_expr", b.getStringAttr(n)),
                b.getNamedAttr("byte_offset", b.getI64IntegerAttr(0))});
      }

      SmallVector<Value, 4> callShapeVals{startV, countV};
      if (ownedKeys) {
        callShapeVals.push_back(loV);
        callShapeVals.push_back(hiV);
      }
      SmallVector<Attribute> shapeExprs;
      shapeExprs.push_back(b.getStringAttr(members ? "1" : "$1"));
      shapeExprs.push_back(b.getStringAttr(members ? "$1" : "cols"));
      shapeExprs.push_back(b.getStringAttr("panel ? panel : cols"));
      if (absolute)
        shapeExprs.push_back(b.getStringAttr("$0"));
      if (ownedKeys) {
        shapeExprs.push_back(b.getStringAttr("$2"));
        shapeExprs.push_back(b.getStringAttr("$3"));
      }
      if (ownerAxis) {
        std::string base = "a" + std::to_string(domainArg);
        std::string last = "cols";
        for (int64_t j = 0; j < rankDomain - 1; ++j) {
          if (j != *ownerAxis) {
            std::string dm = base + "d" + std::to_string(j);
            last = "(" + dm + " ? " + last + " / " + dm + " : 0)";
          }
        }
        for (int64_t j = 0; j < rankDomain; ++j) {
          shapeExprs.push_back(b.getStringAttr(
              j == *ownerAxis    ? "$1"
              : rankDomain == 2  ? "cols"
              : j + 1 == rankDomain ? last
                                 : (base + "d" + std::to_string(j))));
        }
      }
      makeOp(b, loc, "host_call_row_task", callShapeVals, {},
             {b.getNamedAttr("sym_name", b.getStringAttr("row_task")),
              b.getNamedAttr("slot_args", b.getArrayAttr(slotArgs)),
              b.getNamedAttr("shape_exprs", b.getArrayAttr(shapeExprs)),
              b.getNamedAttr("stride_exprs", b.getArrayAttr(strideExprs))});
      makeOp(b, loc, "end");
    };

    if (auto keysAttr = task.getAttrOfType<IntegerAttr>("cpu.key_extent")) {
      int64_t keys = keysAttr.getInt();
      emitRowLet("keys", std::to_string(keys));
      emitRowLet("lines", "(keys + 15) / 16");
      emitRowLet("n_thr", "workers < lines ? workers : lines");
      Operation *pKeys =
          makeOp(b, loc, "host_omp_for", {}, {},
                 {b.getNamedAttr("iv", b.getStringAttr("b")),
                  b.getNamedAttr("lb", b.getStringAttr("0")),
                  b.getNamedAttr("ub", b.getStringAttr("n_thr")),
                  b.getNamedAttr("step", b.getStringAttr("1")),
                  b.getNamedAttr("parallel", b.getBoolAttr(true)),
                  b.getNamedAttr("simd", b.getBoolAttr(false)),
                  b.getNamedAttr("x86_simd", b.getBoolAttr(false)),
                  b.getNamedAttr("schedule", b.getStringAttr("static")),
                  b.getNamedAttr("num_threads", b.getStringAttr("n_thr")),
                  b.getNamedAttr("if_cond", b.getStringAttr("n_thr > 1"))},
                 1);
      makeOp(b, loc, "end");
      Block &pBb = pKeys->getRegion(0).emplaceBlock();
      b.setInsertionPointToEnd(&pBb);
      Value startV = emitRowDef("0");
      Value countV = emitRowDef("rows");
      Value loV = emitRowDef("16 * (lines * b / n_thr)");
      Value hiV = emitRowDef("16 * (lines * (b + 1) / n_thr)");
      emitDispatchBody(startV, countV, loV, hiV);
    } else {
      std::string chunkExpr =
          projected ? ("a" + std::to_string(domainArg) + "d" +
                       std::to_string(slotFacts[domainArg].rank - 2))
                    : "rows";
      emitRowLet("groups", "1 + (rows - 1) / grain");
      emitRowLet("bins", "groups < workers ? groups : workers");
      auto emitChunkWalk = [&](StringRef startExpr, StringRef endExpr) {
        emitRowLet("cur", startExpr, /*isMut=*/true);
        Operation *wOp =
            makeOp(b, loc, "host_while", {}, {},
                   {b.getNamedAttr("cond",
                                   b.getStringAttr(("cur < " + endExpr).str()))},
                   1);
        Block &wBb = wOp->getRegion(0).emplaceBlock();
        OpBuilder::InsertionGuard wg(b);
        b.setInsertionPointToEnd(&wBb);
        Value startV = emitRowDef("cur");
        Value countV;
        if (projected) {
          emitRowLet("rem", chunkExpr + " - cur % " + chunkExpr);
          countV =
              emitRowDef((endExpr + " - cur < rem ? " + endExpr + " - cur : rem")
                             .str());
        } else {
          countV = emitRowDef((endExpr + " - cur").str());
        }
        emitRowAssign("cur", "cur + $0", {countV});
        emitDispatchBody(startV, countV, {}, {});
      };
      Operation *sIf =
          makeOp(b, loc, "host_if", {}, {},
                 {b.getNamedAttr("cond", b.getStringAttr("bins <= 1"))}, 1);
      {
        Block &sBb = sIf->getRegion(0).emplaceBlock();
        OpBuilder::InsertionGuard sg(b);
        b.setInsertionPointToEnd(&sBb);
        emitChunkWalk("0", "rows");
        makeOp(b, loc, "host_guard_return", {}, {},
               {b.getNamedAttr("cond", b.getStringAttr("1")),
                b.getNamedAttr("code", b.getI32IntegerAttr(0))});
        makeOp(b, loc, "end");
      }
      Operation *pRows =
          makeOp(b, loc, "host_omp_for", {}, {},
                 {b.getNamedAttr("iv", b.getStringAttr("b")),
                  b.getNamedAttr("lb", b.getStringAttr("0")),
                  b.getNamedAttr("ub", b.getStringAttr("bins")),
                  b.getNamedAttr("step", b.getStringAttr("1")),
                  b.getNamedAttr("parallel", b.getBoolAttr(true)),
                  b.getNamedAttr("simd", b.getBoolAttr(false)),
                  b.getNamedAttr("x86_simd", b.getBoolAttr(false)),
                  b.getNamedAttr("schedule", b.getStringAttr("static")),
                  b.getNamedAttr("num_threads", b.getStringAttr("bins")),
                  b.getNamedAttr("if_cond", b.getStringAttr("bins > 1"))},
                 1);
      makeOp(b, loc, "end");
      Block &pBb = pRows->getRegion(0).emplaceBlock();
      b.setInsertionPointToEnd(&pBb);
      emitRowLet("group",
                 "(groups / bins * b + (groups % bins * b) / bins) * grain");
      emitRowLet(
          "end",
          "b + 1 == bins ? rows : (groups / bins * (b + 1) + (groups % bins * "
          "(b + 1)) / bins) * grain");
      emitChunkWalk("group", "end");
      makeOp(b, loc, "end");
    }
  }

  SmallVector<Operation *> toErase;
  for (Operation &op : fn.front())
    if (&op != prog && op.getName().getStringRef() != "func.return")
      toErase.push_back(&op);
  for (Operation *op : llvm::reverse(toErase))
    op->erase();
  return success();
}

bool hasWeftRowProgram(ModuleOp m) {
  auto funcs = m.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs))
    return false;
  auto fn = *funcs.begin();
  return !fn.empty() && !fn.front().empty() &&
         fn.front().front().getName().getStringRef() == "weft_host.row_program";
}

llvm::Error describeWeftRowPackage(func::FuncOp fn, Operation &prog,
                                   llvm::raw_ostream &os) {
  auto abiAttr = fn->getAttrOfType<StringAttr>("kir.abi");
  auto parsedAbi = llvm::json::parse(abiAttr.getValue());
  if (!parsedAbi)
    return parsedAbi.takeError();

  llvm::json::Array rows;
  for (Attribute a : prog.getAttrOfType<ArrayAttr>("rows")) {
    auto r = cast<DictionaryAttr>(a);
    rows.push_back(llvm::json::Object{
        {"grain", r.getAs<IntegerAttr>("grain").getInt()},
        {"columns", r.getAs<IntegerAttr>("columns").getInt()}});
  }
  llvm::json::Array shapeParams;
  for (Attribute a : prog.getAttrOfType<ArrayAttr>("shape_parameters"))
    shapeParams.push_back(cast<StringAttr>(a).str());
  llvm::json::Array compact;
  for (int64_t c :
       prog.getAttrOfType<DenseI64ArrayAttr>("contiguous_slots").asArrayRef())
    compact.push_back(c);
  auto taskRanksAttr = prog.getAttrOfType<DenseI64ArrayAttr>("task_ranks");

  llvm::json::Object fields{{"version", 3},
                            {"shape_parameters", std::move(shapeParams)},
                            {"domain", intentdsl::publishDomain(fn)},
                            {"abi", std::move(*parsedAbi)},
                            {"rows", std::move(rows)}};
  if (!compact.empty())
    fields["contiguous_slots"] = std::move(compact);
  if (!taskRanksAttr.empty()) {
    llvm::json::Array tr;
    for (int64_t r : taskRanksAttr.asArrayRef())
      tr.push_back(r);
    fields["task_ranks"] = std::move(tr);
  }
  os << llvm::json::Value(std::move(fields));
  return llvm::Error::success();
}

LogicalResult verifyWeftRow(ModuleOp m, const TargetProfile &p, bool callable) {
  auto fn = *m.getOps<func::FuncOp>().begin();
  auto &prog = fn.front().front();
  if (prog.getNumRegions() != 2 || prog.getRegion(0).empty() ||
      prog.getRegion(1).empty())
    return prog.emitError("incomplete weft_host.row_program regions");

  bool badAddr = m.walk([&](Operation *op) {
                    if (op->getName().getStringRef() != "weft_host.address")
                      return WalkResult::advance();
                    auto bits = op->getAttrOfType<IntegerAttr>("index_bits");
                    if (!bits || (bits.getInt() != 32 && bits.getInt() != 64)) {
                      op->emitError(
                          "weft_host.address requires i32 or i64 index_bits");
                      return WalkResult::interrupt();
                    }
                    if (bits.getInt() == 32 && !canNarrowWeftAddress(*op)) {
                      op->emitError("32-bit Weft row address lacks a proven "
                                    "signed i32 range");
                      return WalkResult::interrupt();
                    }
                    return WalkResult::advance();
                  }).wasInterrupted();
  if (badAddr)
    return failure();

  if (failed(weftTerminalPrinter().verify(m, p)))
    return failure();

  if (callable) {
    if (fn->getAttrOfType<StringAttr>("weft_host.entry") != "intent_entry")
      return fn.emitError("missing synchronous Weft row host entry");
    llvm::SmallVector<char> expected;
    llvm::raw_svector_ostream os(expected);
    if (auto err = describeWeftRowPackage(fn, prog, os))
      return fn.emitError() << llvm::toString(std::move(err)), failure();
    auto pkg = m->getAttrOfType<StringAttr>("weft_host.package");
    if (!pkg || pkg.getValue() != StringRef(expected.data(), expected.size()))
      return fn.emitError("weft_host.package disagrees with verified row ABI"),
             failure();
  }
  return success();
}
} // namespace CI_UNIT_NAMESPACE
