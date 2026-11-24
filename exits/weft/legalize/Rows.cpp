#include "Rows.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
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
                       int columnDim = -2) {
  std::string shape =
      rank == 2 ? "[-1,-2],[1,2]" : "[" + std::to_string(columnDim) + "],[2]";
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
};

struct SweepDesc {
  std::string combine; // "store", "publish", "sum", "max_num", "welford"
  Operation *anchor = nullptr;
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

  TaskTerm emitUniform(TaskTerm seed, TaskTerm position, int dim) {
    std::string vecU64 = regionType("value", "ui64", 1, dim);
    auto c64 = [&](uint64_t val) {
      return lit(std::to_string(val), "ui64");
    };
    auto b64 = [&](StringRef kind, TaskTerm lhs, TaskTerm rhs) {
      return bin(kind, lhs, rhs, lhs.type);
    };
    TaskTerm mask = c64(UINT32_MAX), bits = c64(32), zero = c64(0);
    TaskTerm base = b64("shr", position, c64(2));
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
    TaskTerm word = b64("and", position, c64(3));
    std::string vecI1 = regionType("value", "i1", 1, dim);
    for (auto [index, val] :
         {std::pair{2, c}, std::pair{1, rb}, std::pair{0, a}}) {
      TaskTerm active = cmp("eq", word, c64(index), vecI1);
      d = sel(active, val, d, vecU64);
    }
    std::string vecF32 = regionType("value", "f32", 1, dim);
    TaskTerm f = castOp(b64("shr", d, c64(8)), vecF32);
    return bin("mul", f, lit("0x33800000", "f32"), vecF32);
  }

  void collectSubgraph(Operation *root,
                       const llvm::DenseSet<Value> &summarySet,
                       llvm::DenseSet<Operation *> &visited,
                       SmallVectorImpl<Operation *> &order) {
    if (!root || !visited.insert(root).second)
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

  LogicalResult emitVectorPanelOps(ArrayRef<Operation *> ops,
                                   ArrayRef<Operation *> writes, TaskTerm row,
                                   std::optional<TaskTerm> startOff,
                                   TaskTerm activeExt, int dim,
                                   llvm::DenseMap<Value, TaskTerm> &localMap) {
    auto lookup = [&](Value v) -> TaskTerm {
      auto it = localMap.find(v);
      if (it != localMap.end())
        return it->second;
      return terms.lookup(v);
    };

    for (Operation *opPtr : ops) {
      Operation &op = *opPtr;
      StringRef code = op.getName().stripDialect();
      if (code == "write")
        continue;
      if (code == "gather_row") {
        TaskTerm keys = lookup(op.getOperand(1));
        TaskTerm table = lookup(op.getOperand(0));
        std::string kElem = printMlirType(
            cast<MemRefType>(op.getOperand(1).getType()).getElementType());
        std::string kSp = kElem == "i64" ? "si64" : "si32";
        TaskTerm keyVal = scalarSlice(op.getOperand(1), {row}, true, kSp);
        TaskTerm srcIdx = castOp(keyVal, "index");
        std::string dElem = printMlirType(
            cast<MemRefType>(op.getOperand(0).getType()).getElementType());
        std::string dSp = weftSpelling(
            cast<MemRefType>(op.getOperand(0).getType()).getElementType());
        std::string baseSlTy = regionType("slice", dSp, 1, -2);
        if (!startOff) {
          std::string vTy = regionType("value", dSp, 1, -2);
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
          std::string vTy = regionType("value", dSp, 1, dim);
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
        std::string e = printMlirType(vt.getElementType());
        TaskTerm viewTerm = lookup(view);
        SmallVector<Value> slOps{viewTerm.v};
        SmallVector<Attribute> sels;
        if (vt.getRank() >= 2) {
          slOps.push_back(row.v);
          sels.push_back(b.getStringAttr("index"));
        }
        sels.push_back(b.getStringAttr("all"));
        std::string baseSlTy = regionType("slice", e, 1, -2);
        std::string vTy = regionType("value", e, 1, dim);
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
        TaskTerm rGlob = bin("add", row, origin, "index");
        TaskTerm off = bin("mul", rGlob, columns, "index");
        off = bin("add", off, *startOff, "index");
        std::string elem = wide ? "ui64" : "ui32";
        std::string ixTy = regionType("value", elem, 1, dim);
        TaskTerm offElem = castOp(off, elem);
        Value lanesV =
            makeOp(b, loc, "task_iota", {}, {b.getIndexType()},
                   {b.getNamedAttr("start", b.getI64IntegerAttr(0)),
                    b.getNamedAttr("end", b.getI64IntegerAttr(dim)),
                    b.getNamedAttr("val_type", b.getStringAttr(ixTy))})
                ->getResult(0);
        TaskTerm idxVec = bin("add", {lanesV, ixTy}, offElem, ixTy);
        localMap[op.getResult(0)] =
            wide ? idxVec : castOp(idxVec, regionType("value", "si32", 1, dim));
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
        std::string vTy = regionType("value", "f32", 1, dim);
        auto splatF32 = [&](TaskTerm sc) -> TaskTerm {
          Value sp =
              makeOp(b, loc, "task_new", {sc.v}, {b.getIndexType()},
                     {b.getNamedAttr("in_type", b.getStringAttr(sc.type)),
                      b.getNamedAttr("val_type", b.getStringAttr(vTy))})
                  ->getResult(0);
          return {sp, vTy};
        };
        if (code == "segment_length") {
          TaskTerm lenSc = castOp(bin("sub", hi, lo, "index"), "f32");
          localMap[op.getResult(0)] = splatF32(lenSc);
          continue;
        }
        TaskTerm zVec = splatF32(lit("0.0", "f32"));
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
                std::string baseSlTy = regionType("slice", e, 1, fullCols);
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
                std::string ldTy = regionType("value", e, 1, dim);
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
        localMap[op.getResult(0)] =
            emitUniform(lookup(op.getOperand(0)), lookup(op.getOperand(1)), dim);
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
              ? regionType(
                    "value",
                    weftSpelling(cast<RankedTensorType>(resType).getElementType()),
                    1, dim)
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
        localMap[op.getResult(0)] = castOp(lookup(op.getOperand(0)), outTy);
      } else if (code == "sqrt" || code == "rsqrt" || code == "exp") {
        localMap[op.getResult(0)] = unaryOp(code, lookup(op.getOperand(0)));
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
      std::string e = printMlirType(vt.getElementType());
      TaskTerm viewTerm = lookup(view);
      SmallVector<Value> slOps{viewTerm.v};
      SmallVector<Attribute> sels;
      if (vt.getRank() >= 2) {
        slOps.push_back(row.v);
        sels.push_back(b.getStringAttr("index"));
      }
      sels.push_back(b.getStringAttr("all"));
      std::string baseSlTy = regionType("slice", e, 1, -2);
      TaskTerm valTerm = lookup(wPtr->getOperand(wPtr->getNumOperands() - 1));
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
        std::string subSlTy = regionType("slice", e, 1, dim);
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
        forOperands.push_back(sw.combine == "sum" ? zeroF32.v : minusInfF32.v);
        iterTys.push_back(b.getStringAttr("f32"));
        resTypes.push_back(b.getIndexType());
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
          Operation *segFor = makeOp(
              b, loc, "task_for", {lo.v, hi.v, oneIdx.v, zeroF32.v},
              {b.getIndexType()},
              {b.getNamedAttr("iter_types",
                              b.getArrayAttr({b.getStringAttr("f32")}))},
              1);
          Block &sBb = segFor->getRegion(0).emplaceBlock();
          TaskTerm segMem{sBb.addArgument(b.getIndexType(), loc), "index"};
          TaskTerm segCarry{sBb.addArgument(b.getIndexType(), loc), "f32"};
          localMap[segBb.getArgument(0)] = segMem;
          localMap[segBb.getArgument(1)] = memberIdx;
          {
            OpBuilder::InsertionGuard sg(b);
            b.setInsertionPointToEnd(&sBb);
            for (Operation &scOp : segBb) {
              StringRef scCode = scOp.getName().stripDialect();
              if (scCode == "member_load") {
                SmallVector<TaskTerm, 4> coords;
                for (Value ci : scOp.getOperands().drop_front()) {
                  TaskTerm ct = lookup(ci);
                  coords.push_back(ct.type == "index" ? ct
                                                      : castOp(ct, "index"));
                }
                localMap[scOp.getResult(0)] =
                    readScalar(scOp.getOperand(0), coords);
              } else if (scCode == "term") {
                TaskTerm total =
                    bin("add", segCarry, lookup(scOp.getOperand(0)), "f32");
                makeOp(
                    b, loc, "task_yield", {total.v}, {},
                    {b.getNamedAttr("val_types",
                                    b.getArrayAttr({b.getStringAttr("f32")}))});
              } else if (scCode == "mul" || scCode == "add") {
                localMap[scOp.getResult(0)] =
                    bin(scCode, lookup(scOp.getOperand(0)),
                        lookup(scOp.getOperand(1)), "f32");
              } else {
                return scOp.emitError("illegal ordered member operation ")
                       << scCode;
              }
            }
          }
          localMap[resVal] = {segFor->getResult(0), "f32"};
        }
      } else if (n == "scan_sum") {
        TaskTerm nextC =
            bin("add", iterArgs[scanCursor++], lookup(op.getOperand(0)), ty);
        nextCarries.push_back(nextC);
        localMap[resVal] = nextC;
      } else if (scalarRed && n == sw.combine) {
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
      TaskTerm acc = iterArgs[0];
      TaskTerm val = lookup(sw.anchor->getResult(0));
      TaskTerm next =
          bin(sw.combine == "sum" ? "add" : "max", acc, val, "f32");
      makeOp(b, loc, "task_yield", {next.v}, {},
             {b.getNamedAttr("val_types",
                             b.getArrayAttr({b.getStringAttr("f32")}))});
      terms[sw.anchor->getResult(0)] = {forOp->getResult(0), "f32"};
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
                           int64_t tailVecWidth) {
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
        auto *shape = facts.arguments[i].getAsObject()->getArray("shape");
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
        auto *shape = facts.arguments[i].getAsObject()->getArray("shape");
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
      auto *record = facts.arguments[slot].getAsObject();
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
      auto *slot = facts.arguments[table].getAsObject();
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
  StringRef walkEarly = task.getAttrOfType<StringAttr>("cpu.walk").getValue();
  if (!ownerAxis && walkEarly != "tile" && walkEarly != "destination") {
    for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
      if (!isa<MemRefType>(fn.getArgument(i).getType()) ||
          slotFacts[i].swapAxes || slotFacts[i].rank > 2)
        continue;
      auto *a = facts.arguments[i].getAsObject();
      auto *shape = a->getArray("shape"), *strides = a->getArray("strides");
      bool dense = shape && strides && shape->size() == strides->size();
      int64_t step = 1;
      if (dense) {
        for (int k = int(shape->size()) - 1; k >= 0; --k) {
          auto n = (*shape)[k].getAsInteger(), s = (*strides)[k].getAsInteger();
          if (!n || !s || *n <= 0 || *s != step || step > INT64_MAX / *n) {
            dense = false;
            break;
          }
          step *= *n;
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
      SmallVector<std::string, 4> shape, axes;
      for (int j = 0; j < fullRank; ++j) {
        shape.push_back(j == *ownerAxis || fullRank == 1
                            ? "-1"
                            : std::to_string(-4 - absolute - j));
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

  bool hasWelford = llvm::any_of(bb, [](Operation &op) {
    return op.getName().stripDialect() == "welford";
  });
  bool scalarMembers = ownerAxis.has_value() || hasWelford;
  if (scalarMembers) {
    rb.zeroCount = rb.lit("0", "si32");
    rb.oneCount = rb.lit("1", "si32");
    rb.zeroF32 = rb.lit("0.0", "f32");
    rb.minusInfF32 = rb.lit("0xFF800000", "f32");
  }

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
      std::string elem = printMlirType(
          cast<MemRefType>(tOp.getOperand(0).getType()).getElementType());
      std::string baseSlTy = regionType("slice", elem, 1, -2);
      auto emitTilePass = [&](bool tail) {
        int64_t tailVec =
            fixed > 0 ? std::min<int64_t>(fixed, std::max<int64_t>(1, tailVecWidth))
                      : 1;
        TaskTerm lb = tail ? tailStart : rb.zeroIdx;
        TaskTerm ub = tail ? rb.columns : panelCount;
        TaskTerm step = tail ? rb.idx(tailVec) : rb.oneIdx;
        int dim = tail ? int(tailVec) : (fixed ? int(fixed) : -3);
        Operation *pFor = makeOp(
            b, loc, "task_for", {lb.v, ub.v, step.v}, {},
            {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
        Block &pBb = pFor->getRegion(0).emplaceBlock();
        TaskTerm pIv{pBb.addArgument(b.getIndexType(), loc), "index"};
        OpBuilder::InsertionGuard pg(b);
        b.setInsertionPointToEnd(&pBb);
        TaskTerm off =
            tail ? pIv : rb.bin("mul", pIv, rb.panelWidth, "index");
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
        SmallVector<Value, 3> inSub{inSl, off.v};
        if (dim <= 0)
          inSub.push_back(rb.panelWidth.v);
        Value val =
            makeOp(b, loc, "task_subview", inSub, {b.getIndexType()},
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
        SmallVector<Value, 3> outSub{outSl, off.v};
        if (dim <= 0)
          outSub.push_back(rb.panelWidth.v);
        Value win =
            makeOp(b, loc, "task_subview", outSub, {b.getIndexType()},
                   {b.getNamedAttr("static_extent", b.getI64IntegerAttr(dim)),
                    b.getNamedAttr("slice_type", b.getStringAttr(baseSlTy)),
                    b.getNamedAttr("val_type", b.getStringAttr(outSubSlTy)),
                    b.getNamedAttr("admit", b.getBoolAttr(false))})
                ->getResult(0);
        rb.commitOp({val, vTy}, {win, outSubSlTy});
        makeOp(b, loc, "task_yield", {}, {},
               {b.getNamedAttr("val_types", b.getArrayAttr({}))});
      };
      emitTilePass(false);
      emitTilePass(true);
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
              TaskTerm dst = rb.scalarSlice(v.getOperand(0), {index}, false);
              TaskTerm delta = a(v, 2);
              Value rmw =
                  makeOp(b, loc, "task_atomic_rmw", {delta.v, dst.v},
                         {b.getIndexType()},
                         {b.getNamedAttr("kind", b.getStringAttr("add")),
                          b.getNamedAttr("ordering", b.getStringAttr("relaxed")),
                          b.getNamedAttr("val_type", b.getStringAttr("si32")),
                          b.getNamedAttr("slice_type",
                                         b.getStringAttr(dst.type))})
                      ->getResult(0);
              rhs = {rmw, "si32"};
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
    } else if (whole && !scalarMembers) {
      SmallVector<Operation *> ops, writes;
      for (Operation &op : bb.without_terminator()) {
        if (op.getName().stripDialect() == "write")
          writes.push_back(&op);
        else
          ops.push_back(&op);
      }
      llvm::DenseMap<Value, TaskTerm> localMap;
      if (failed(rb.emitVectorPanelOps(ops, writes, rowIv, std::nullopt,
                                       rb.panelWidth, -2, localMap)))
        return failure();
    } else {
      // Decompose logical row body into sweeps (owner captures, reductions,
      // welford, private_bins/vertical_sum, segment/prefix sweeps, panel stores).
      SmallVector<Operation *> allOps, writes, ownedWrites;
      for (Operation &op : bb.without_terminator()) {
        allOps.push_back(&op);
        StringRef n = op.getName().stripDialect();
        if (n == "write")
          writes.push_back(&op);
        else if (n == "write_owner")
          ownedWrites.push_back(&op);
      }

      llvm::DenseSet<Value> summaryValues;
      for (Operation *op : allOps) {
        StringRef n = op->getName().stripDialect();
        if (n == "read_owner") {
          rb.terms[op->getResult(0)] = rb.memberSlice(
              op->getOperand(0), rowIv, {}, true, true,
              weftSpelling(op->getResult(0).getType()));
          summaryValues.insert(op->getResult(0));
          continue;
        }
        bool publish = (n == "write_owner");
        bool store = (n == "write" || publish);
        if (!scalarMembers && store &&
            op != (publish ? ownedWrites : writes).front())
          continue;
        if (!store && n != "sum" && n != "max_num" && n != "welford")
          continue;

        SweepDesc sw;
        sw.combine =
            publish ? "publish" : store ? "store" : std::string(n);
        sw.anchor = op;
        if (store) {
          if (scalarMembers)
            sw.writes.push_back(op);
          else
            sw.writes = publish ? ownedWrites : writes;
        }

        llvm::DenseSet<Operation *> visited;
        SmallVector<Operation *> subOps;
        if (store) {
          for (Operation *w : sw.writes)
            rb.collectSubgraph(w->getOperand(2).getDefiningOp(), summaryValues,
                               visited, subOps);
        } else if (n == "welford") {
          rb.collectSubgraph(op->getOperand(0).getDefiningOp(), summaryValues,
                             visited, subOps);
          subOps.push_back(op);
        } else {
          rb.collectSubgraph(op, summaryValues, visited, subOps);
        }

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
        } else if (scalarMembers || sw.prefixCarry ||
                   (sw.segmentLoop &&
                    (whole || sw.combine != "store" ||
                     !canVectorizeSegmentSweep(task, subOps)))) {
          if (!rb.zeroF32.v) {
            rb.zeroCount = rb.lit("0", "si32");
            rb.oneCount = rb.lit("1", "si32");
            rb.zeroF32 = rb.lit("0.0", "f32");
            rb.minusInfF32 = rb.lit("0xFF800000", "f32");
          }
          if (failed(rb.emitScalarSweep(sw, subOps, rowIv)))
            return failure();
        } else {
          bool isStore = (sw.combine == "store");
          TaskTerm prev;
          if (!isStore)
            prev = rb.lit(sw.combine == "sum" ? "0.0" : "0xFF800000", "f32");
          int64_t tailVec =
              fixed > 0
                  ? std::min<int64_t>(fixed, std::max<int64_t>(1, tailVecWidth))
                  : std::max<int64_t>(1, tailVecWidth);

          for (bool tail : {false, true}) {
            TaskTerm lb = tail ? tailStart : rb.zeroIdx;
            TaskTerm ub = tail ? rb.columns : panelCount;
            TaskTerm step = tail ? rb.idx(tailVec) : rb.oneIdx;
            int dim = tail ? int(tailVec) : (fixed ? int(fixed) : -3);
            SmallVector<Value, 4> fOps{lb.v, ub.v, step.v};
            SmallVector<Attribute, 1> iterTys;
            SmallVector<Type, 1> resTys;
            if (!isStore) {
              fOps.push_back(prev.v);
              iterTys.push_back(b.getStringAttr("f32"));
              resTys.push_back(b.getIndexType());
            }
            Operation *pFor = makeOp(
                b, loc, "task_for", fOps, resTys,
                {b.getNamedAttr("iter_types", b.getArrayAttr(iterTys))}, 1);
            Block &pBb = pFor->getRegion(0).emplaceBlock();
            TaskTerm pIv{pBb.addArgument(b.getIndexType(), loc), "index"};
            TaskTerm accTerm;
            if (!isStore)
              accTerm = {pBb.addArgument(b.getIndexType(), loc), "f32"};
            {
              OpBuilder::InsertionGuard pg(b);
              b.setInsertionPointToEnd(&pBb);
              TaskTerm off =
                  tail ? pIv : rb.bin("mul", pIv, rb.panelWidth, "index");
              llvm::DenseMap<Value, TaskTerm> localMap;
              if (failed(rb.emitVectorPanelOps(subOps, sw.writes, rowIv, off,
                                               rb.panelWidth, dim, localMap)))
                return failure();
              if (!isStore) {
                TaskTerm partial = localMap.at(sw.anchor->getResult(0));
                TaskTerm next = rb.bin(sw.combine == "sum" ? "add" : "max",
                                       accTerm, partial, "f32");
                makeOp(
                    b, loc, "task_yield", {next.v}, {},
                    {b.getNamedAttr("val_types",
                                    b.getArrayAttr({b.getStringAttr("f32")}))});
                prev = {pFor->getResult(0), "f32"};
              } else {
                makeOp(b, loc, "task_yield", {}, {},
                       {b.getNamedAttr("val_types", b.getArrayAttr({}))});
              }
            }
          }
          if (!isStore)
            rb.terms[sw.anchor->getResult(0)] = prev;
        }

        if (!store)
          summaryValues.insert(op->getResult(0));
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
      {b.getNamedAttr("params", b.getStringAttr(llvm::join(params, ", "))),
       b.getNamedAttr("walk", b.getStringAttr(walk)),
       b.getNamedAttr("member_tasks", b.getBoolAttr(members)),
       b.getNamedAttr("single_dispatcher", b.getBoolAttr(singleDispatcher))},
      1);
  makeOp(b, loc, "end");

  Block &hFnBb = hFn->getRegion(0).emplaceBlock();
  b.setInsertionPointToEnd(&hFnBb);

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
    Operation *pTiles =
        makeOp(b, loc, "host_parallel_transpose_tiles", {}, {},
               {b.getNamedAttr("tile", b.getI64IntegerAttr(tileW))}, 1);
    makeOp(b, loc, "end");
    Block &pBb = pTiles->getRegion(0).emplaceBlock();
    Value rV = pBb.addArgument(b.getIndexType(), loc);
    Value cV = pBb.addArgument(b.getIndexType(), loc);
    Value nrV = pBb.addArgument(b.getIndexType(), loc);
    Value ncV = pBb.addArgument(b.getIndexType(), loc);
    b.setInsertionPointToEnd(&pBb);

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
                                b.getStringAttr("panel ? panel : nc")})),
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
        } else if ((members && int64_t(i) == domainArg) ||
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
      Operation *pKeys = makeOp(
          b, loc, "host_parallel_key_ranges", {}, {},
          {b.getNamedAttr("key_extent", b.getI64IntegerAttr(keysAttr.getInt()))},
          1);
      makeOp(b, loc, "end");
      Block &pBb = pKeys->getRegion(0).emplaceBlock();
      Value startV = pBb.addArgument(b.getIndexType(), loc);
      Value countV = pBb.addArgument(b.getIndexType(), loc);
      Value loV = pBb.addArgument(b.getIndexType(), loc);
      Value hiV = pBb.addArgument(b.getIndexType(), loc);
      b.setInsertionPointToEnd(&pBb);
      emitDispatchBody(startV, countV, loV, hiV);
    } else {
      std::string chunkExpr =
          projected ? ("a" + std::to_string(domainArg) + "d" +
                       std::to_string(slotFacts[domainArg].rank - 2))
                    : "rows";
      Operation *pRows = makeOp(
          b, loc, "host_parallel_row_intervals", {}, {},
          {b.getNamedAttr("chunk_expr", b.getStringAttr(chunkExpr))}, 1);
      makeOp(b, loc, "end");
      Block &pBb = pRows->getRegion(0).emplaceBlock();
      Value startV = pBb.addArgument(b.getIndexType(), loc);
      Value countV = pBb.addArgument(b.getIndexType(), loc);
      b.setInsertionPointToEnd(&pBb);
      emitDispatchBody(startV, countV, {}, {});
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
