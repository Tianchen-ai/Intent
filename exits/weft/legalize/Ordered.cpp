#include "Ordered.h"
#include "Arithmetic.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"
#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "intentdsl/levels/execution/cpu/ir/OwnerTraversal.h"
#include "intentdsl/levels/execution/cpu/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include <map>

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;

bool canNarrowWeftAddress(Operation &op);
TerminalPrinter weftTerminalPrinter();

namespace {
struct MemberStore {
  Value vector, scale;
};

std::optional<MemberStore> scaledMemberStore(Operation *op) {
  if (op->getName().stripDialect() != "slice_write")
    return {};
  auto narrow = op->getOperand(0).getDefiningOp<arith::TruncFOp>();
  if (!narrow || !narrow.getResult().hasOneUse())
    return {};
  auto product = narrow.getIn().getDefiningOp<arith::MulFOp>();
  if (!product || !product.getResult().hasOneUse())
    return {};
  for (unsigned j : {0u, 1u}) {
    auto *def = product->getOperand(j).getDefiningOp();
    if (def && def->getName().stripDialect() == "lane_splat")
      return MemberStore{product->getOperand(1 - j), def->getOperand(0)};
  }
  return {};
}

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

struct TightenedLoopBound {
  Value cmpResult;
  Value boundVal;
  bool inclusive = false;
  bool hasBufferStore = false;
};

std::optional<int64_t> staticIndexVal(func::FuncOp fn, Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  if (auto d = v.getDefiningOp<memref::DimOp>())
    if (auto idx = d.getConstantIndex())
      if (auto ba = dyn_cast<BlockArgument>(d.getSource()))
        return fn.getArgAttrOfType<DenseI64ArrayAttr>(ba.getArgNumber(),
                                                      "cpu.shape")[*idx];
  return std::nullopt;
}

std::optional<int64_t> maxStaticIndexVal(func::FuncOp fn, Value v) {
  if (auto s = staticIndexVal(fn, v))
    return s;
  if (auto sel = v.getDefiningOp<arith::SelectOp>()) {
    if (auto cmp = sel.getCondition().getDefiningOp<arith::CmpIOp>()) {
      auto p = cmp.getPredicate();
      bool isMin = ((p == arith::CmpIPredicate::slt ||
                     p == arith::CmpIPredicate::sle ||
                     p == arith::CmpIPredicate::ult ||
                     p == arith::CmpIPredicate::ule) &&
                    cmp.getLhs() == sel.getTrueValue() &&
                    cmp.getRhs() == sel.getFalseValue()) ||
                   ((p == arith::CmpIPredicate::sgt ||
                     p == arith::CmpIPredicate::sge ||
                     p == arith::CmpIPredicate::ugt ||
                     p == arith::CmpIPredicate::uge) &&
                    cmp.getLhs() == sel.getFalseValue() &&
                    cmp.getRhs() == sel.getTrueValue());
      if (isMin) {
        auto a = maxStaticIndexVal(fn, sel.getTrueValue());
        auto b = maxStaticIndexVal(fn, sel.getFalseValue());
        if (a && b)
          return std::min(*a, *b);
        return a ? a : b;
      }
    }
  }
  return std::nullopt;
}

bool isStaticMultipleOf(func::FuncOp fn, Value v, int64_t m) {
  if (m <= 1)
    return true;
  if (auto s = staticIndexVal(fn, v))
    return *s >= 0 && *s % m == 0;
  if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
    auto a = staticIndexVal(fn, mul.getLhs());
    auto b = staticIndexVal(fn, mul.getRhs());
    return (a && *a >= 0 && *a % m == 0) || (b && *b >= 0 && *b % m == 0);
  }
  if (auto sel = v.getDefiningOp<arith::SelectOp>())
    return isStaticMultipleOf(fn, sel.getTrueValue(), m) &&
           isStaticMultipleOf(fn, sel.getFalseValue(), m);
  return false;
}

bool isZeroFloatConst(Value v) {
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return false;
  if (auto fa = dyn_cast<FloatAttr>(cst.getValue()))
    return fa.getValue().isZero();
  return false;
}

bool isOneFloatConst(Value v) {
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return false;
  if (auto fa = dyn_cast<FloatAttr>(cst.getValue()))
    return fa.getValue().isExactlyValue(1.0);
  return false;
}

bool isNegInfConst(Value v) {
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return false;
  if (auto fa = dyn_cast<FloatAttr>(cst.getValue()))
    return fa.getValue().isNegative() && fa.getValue().isInfinity();
  return false;
}

bool isNegInfWhenFalse(Value v, Value cond) {
  if (auto sel = v.getDefiningOp<arith::SelectOp>())
    return sel.getCondition() == cond && isNegInfConst(sel.getFalseValue());
  if (auto ifOp = v.getDefiningOp<scf::IfOp>())
    if (ifOp.getCondition() == cond && !ifOp.getElseRegion().empty() &&
        ifOp.getNumResults() == 1)
      return isNegInfConst(ifOp.elseYield().getOperand(0));
  return false;
}

bool isZeroWhenFalse(Value v, Value cond) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto fa = dyn_cast<FloatAttr>(cst.getValue()))
      return fa.getValue().isZero();
    return false;
  }
  if (auto sel = v.getDefiningOp<arith::SelectOp>())
    return sel.getCondition() == cond &&
           isZeroWhenFalse(sel.getFalseValue(), cond);
  if (auto ifOp = v.getDefiningOp<scf::IfOp>())
    if (ifOp.getCondition() == cond && !ifOp.getElseRegion().empty() &&
        ifOp.getNumResults() == 1)
      return isZeroWhenFalse(ifOp.elseYield().getOperand(0), cond);
  if (auto ext = v.getDefiningOp<arith::ExtFOp>())
    return isZeroWhenFalse(ext.getIn(), cond);
  if (auto tr = v.getDefiningOp<arith::TruncFOp>())
    return isZeroWhenFalse(tr.getIn(), cond);
  if (Operation *def = v.getDefiningOp()) {
    if (def->getName().stripDialect() == "lane_splat")
      return isZeroWhenFalse(def->getOperand(0), cond);
    if (auto mul = dyn_cast<arith::MulFOp>(def))
      return isZeroWhenFalse(mul.getLhs(), cond) ||
             isZeroWhenFalse(mul.getRhs(), cond);
  }
  return false;
}

bool preservesIterArgWhenFalse(Value yielded, Value iterArg, Value cond) {
  if (yielded == iterArg)
    return true;
  if (auto sel = yielded.getDefiningOp<arith::SelectOp>())
    if (sel.getCondition() == cond && sel.getFalseValue() == iterArg)
      return true;
  if (auto ifOp = yielded.getDefiningOp<scf::IfOp>()) {
    if (ifOp.getCondition() == cond && !ifOp.getElseRegion().empty()) {
      for (auto [res, elseVal] :
           llvm::zip(ifOp.getResults(), ifOp.elseYield().getOperands()))
        if (res == yielded && elseVal == iterArg)
          return true;
    }
  }
  if (Operation *def = yielded.getDefiningOp()) {
    StringRef n = def->getName().getStringRef();
    if (n == "arith.maximumf" || n == "arith.maxnumf") {
      if (def->getOperand(0) == iterArg &&
          isNegInfWhenFalse(def->getOperand(1), cond))
        return true;
      if (def->getOperand(1) == iterArg &&
          isNegInfWhenFalse(def->getOperand(0), cond))
        return true;
    }
    if (n == "arith.addf") {
      if (def->getOperand(0) == iterArg &&
          isZeroWhenFalse(def->getOperand(1), cond))
        return true;
      if (def->getOperand(1) == iterArg &&
          isZeroWhenFalse(def->getOperand(0), cond))
        return true;
    }
  }
  return false;
}

std::optional<TightenedLoopBound> analyzeCausalLoop(func::FuncOp fn,
                                                    scf::ForOp loop) {
  if (staticIndexVal(fn, loop.getStep()) != 1 || loop.getInitArgs().empty())
    return std::nullopt;
  bool hasBufStore = false;
  bool unsafeEffect =
      loop.getBody()
          ->walk([&](Operation *op) {
            if (op == loop.getBody()->getTerminator())
              return WalkResult::advance();
            StringRef n = op->getName().getStringRef();
            if (isa<scf::ForOp>(op))
              return WalkResult::interrupt();
            if (isa<scf::IfOp, scf::YieldOp>(op))
              return WalkResult::advance();
            if (n == "cpu.buffer_store") {
              hasBufStore = true;
              return WalkResult::advance();
            }
            if (n == "cpu.buffer_load" || isa<memref::LoadOp, memref::DimOp>(op) ||
                op->getName().stripDialect() == "slice_read" ||
                isMemoryEffectFree(op))
              return WalkResult::advance();
            return WalkResult::interrupt();
          })
          .wasInterrupted();
  if (unsafeEffect)
    return std::nullopt;

  Value iv = loop.getInductionVar();
  auto yieldOp = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  auto isLoopInvariant = [&](Value v) {
    if (auto ba = dyn_cast<BlockArgument>(v))
      return ba.getOwner() != loop.getBody();
    if (Operation *def = v.getDefiningOp())
      return !loop->isProperAncestor(def);
    return false;
  };

  for (auto cmpOp : loop.getBody()->getOps<arith::CmpIOp>()) {
    Value boundVal;
    bool inclusive = false;
    auto pred = cmpOp.getPredicate();
    if (cmpOp.getLhs() == iv && isLoopInvariant(cmpOp.getRhs())) {
      if (pred == arith::CmpIPredicate::sle) {
        boundVal = cmpOp.getRhs();
        inclusive = true;
      } else if (pred == arith::CmpIPredicate::slt) {
        boundVal = cmpOp.getRhs();
        inclusive = false;
      }
    } else if (cmpOp.getRhs() == iv && isLoopInvariant(cmpOp.getLhs())) {
      if (pred == arith::CmpIPredicate::sge) {
        boundVal = cmpOp.getLhs();
        inclusive = true;
      }
    }
    if (!boundVal || !boundVal.getType().isIndex())
      continue;
    bool allPreserved = true;
    for (auto [yVal, iterArg] :
         llvm::zip(yieldOp.getOperands(), loop.getRegionIterArgs())) {
      if (!preservesIterArgWhenFalse(yVal, iterArg, cmpOp.getResult())) {
        allPreserved = false;
        break;
      }
    }
    if (allPreserved)
      return TightenedLoopBound{cmpOp.getResult(), boundVal, inclusive,
                                hasBufStore};
  }
  return std::nullopt;
}

struct TiledTwoPassPlan {
  StringAttr scoreEngine, valueEngine;
  int64_t innerAx = -1;
  int64_t tokenTile = 0, valueWidth = 0;
  Operation *qReadOp = nullptr;
  Operation *qExtOp = nullptr;
  Value qMem;
  int64_t qOwnMemAx = -1, qVecMemAx = -1, D = 0, axOwn = -1, axD = -1;
  Value scaleVal;
  Value outerScaleVal;
  Value postScaleVal;
  Value biasSlopeVal;
  bool biasNegate = false;

  scf::ForOp loop1;
  Value kMem;
  SmallVector<Value, 4> kCoords;
  int64_t kTokCoordPos = -1, kTokMemAx = -1, kVecMemAx = -1, KExt = 0, axTok = -1;
  bool hasCausal = false;
  Value causalBoundVal;
  bool causalInclusive = true;
  bool hasScoreBuffer = false;
  int64_t scoreSlots = 0;
  bool windowGuarded = false;

  Operation *stage2Op = nullptr;
  scf::IfOp windowBranch;
  scf::ForOp replayLoop;
  scf::ForOp recomputeLoop;
  Value zeroAccInitVal;
  Value stage2MassRes;
  Value stage2AccRes;
  bool useExp2 = true;
  std::optional<Type> weightRoundType;
  Value vMem;
  SmallVector<Value, 4> vCoords;
  int64_t vTokCoordPos = -1, vTokMemAx = -1, vVecMemAx = -1, DV = 0, axDV = -1;

  Operation *invIfOrSelOp = nullptr;
  Value invCondVal;
  bool invZeroWhenTrue = true;
  bool hasDirectRecip = false;
  Operation *ySplatOp = nullptr;
  Operation *yMulOp = nullptr;
  Operation *yTruncOp = nullptr;
  Operation *yWriteOp = nullptr;
  Value yMem;
  int64_t yOwnMemAx = -1, yVecMemAx = -1;
};

struct OrderedLowerer {
  func::FuncOp fn;
  Operation &space;
  OpBuilder &b;
  Location loc;
  int64_t batch = 1;
  llvm::DenseMap<Value, SmallVector<int64_t>> axesMap;
  SmallVector<llvm::DenseMap<Value, TaskTerm>, 16> lanes;
  SmallVector<TaskTerm, 16> localBufs;
  TaskTerm tiledScoreBuf, tiledOutBuf;
  TaskTerm true1, ninfS, zeroS, oneS, ln2S, cB64;
  TaskTerm ninfV, ninfM, zeroV, zeroM, zeroO, trueM, trueB;
  TaskTerm scoreMask, scoreZero;
  llvm::DenseSet<Value> knownTrueConds;
  bool allowBufferStoreTightening = true;
  bool anyTightenedCausalLoop = false;
  bool isRvv = false;

  OrderedLowerer(func::FuncOp fn, Operation &space, OpBuilder &b, Location loc,
                 int64_t batch)
      : fn(fn), space(space), b(b), loc(loc), batch(batch) {}

  static std::string scalarType(Type t) {
    if (t.isIndex())
      return "index";
    if (t.isF32())
      return "f32";
    if (t.isF16())
      return "f16";
    if (t.isBF16())
      return "bf16";
    if (t.isInteger(1))
      return "i1";
    if (t.isUnsignedInteger(64))
      return "ui64";
    if (t.isInteger(64))
      return "si64";
    return "si32";
  }

  static std::string carrierType(Type t) {
    if (t.isInteger(32))
      return "i32";
    if (t.isInteger(64) && !t.isUnsignedInteger(64))
      return "i64";
    return scalarType(t);
  }

  int64_t dimOf(Value v, StringRef key, unsigned i) {
    return fn.getArgAttrOfType<DenseI64ArrayAttr>(
        cast<BlockArgument>(v).getArgNumber(), key)[i];
  }

  TaskTerm idx(int64_t n) {
    Value v =
        makeOp(b, loc, "task_const", {}, {b.getIndexType()},
               {b.getNamedAttr("lit", b.getStringAttr(std::to_string(n))),
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

  std::string vectorType(Value v) {
    auto [base, axis] = *intentdsl::kir::memberAxis(v);
    return "!weft_kernel.value<" +
           scalarType(cast<ShapedType>(v.getType()).getElementType()) + ",[" +
           std::to_string(dimOf(base, "cpu.shape", axis)) + "],[" +
           std::to_string(axesMap.at(base)[axis]) + "]>";
  }

  TaskTerm sliceView(int64_t l, Value view, ValueRange coords, int member,
                     bool admit, StringRef admitTy = "") {
    auto elem = carrierType(cast<MemRefType>(view.getType()).getElementType());
    TaskTerm viewTerm = lanes[l].at(view);
    SmallVector<Value> ops{viewTerm.v};
    SmallVector<Attribute> sels;
    for (Value cVal : coords) {
      ops.push_back(lanes[l].at(cVal).v);
      sels.push_back(b.getStringAttr("index"));
    }
    std::string extStr, axStr;
    if (member >= 0) {
      sels.insert(sels.begin() + member, b.getStringAttr("all"));
      extStr = std::to_string(dimOf(view, "cpu.shape", member));
      axStr = std::to_string(axesMap.at(view)[member]);
    }
    std::string slTy = "!weft_kernel.slice<!weft_kernel.encoding<\"" + elem +
                       "\",\"dense\",\"dense." + elem + "\",[]>,[" + extStr +
                       "],[" + axStr + "]>";
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

  bool operandsUniform(Operation &op) {
    if (batch <= 1)
      return false;
    for (Value v : op.getOperands()) {
      Value v0 = lanes[0].at(v).v;
      for (int64_t l = 1; l < batch; ++l)
        if (lanes[l].at(v).v != v0)
          return false;
    }
    return true;
  }

  TaskTerm newVec(TaskTerm scalar, StringRef vecTy) {
    Value v = makeOp(b, loc, "task_new", {scalar.v}, {b.getIndexType()},
                     {b.getNamedAttr("in_type", b.getStringAttr(scalar.type)),
                      b.getNamedAttr("val_type", b.getStringAttr(vecTy))})
                  ->getResult(0);
    return {v, vecTy.str()};
  }

  TaskTerm unaryOp(StringRef kind, TaskTerm in) {
    Value v = makeOp(b, loc, "task_unary", {in.v}, {b.getIndexType()},
                     {b.getNamedAttr("kind", b.getStringAttr(kind)),
                      b.getNamedAttr("val_type", b.getStringAttr(in.type))})
                  ->getResult(0);
    return {v, in.type};
  }

  TaskTerm reduceOp(TaskTerm in, StringRef kind, int64_t axis,
                    StringRef outTy) {
    Value v = makeOp(b, loc, "task_reduce", {in.v}, {b.getIndexType()},
                     {b.getNamedAttr("kind", b.getStringAttr(kind)),
                      b.getNamedAttr("axis", b.getI64IntegerAttr(axis)),
                      b.getNamedAttr("in_type", b.getStringAttr(in.type)),
                      b.getNamedAttr("val_type", b.getStringAttr(outTy))})
                  ->getResult(0);
    return {v, outTy.str()};
  }

  TaskTerm outerContractOp(TaskTerm lhs, TaskTerm rhs, int64_t overAx,
                           StringRef outTy, StringAttr engine = {}) {
    Value v =
        makeOp(
            b, loc, "task_contract", {lhs.v, rhs.v}, {b.getIndexType()},
            {b.getNamedAttr("ci.implementation",
                            engine
                                ? engine
                                : b.getStringAttr("exits/weft::weft-simd-fma")),
             b.getNamedAttr("kind", b.getStringAttr("outer_contract")),
             b.getNamedAttr("over", b.getDenseI64ArrayAttr({overAx})),
             b.getNamedAttr("acc_type", b.getStringAttr("f32")),
             b.getNamedAttr("engine",
                            b.getStringAttr(engine  ? "onednn-brgemm-bf16"
                                            : isRvv ? "vector"
                                                    : "auto")),
             b.getNamedAttr("allow_subnormal_ftz", b.getBoolAttr(bool(engine))),
             b.getNamedAttr("in_types",
                            b.getArrayAttr({b.getStringAttr(lhs.type),
                                            b.getStringAttr(rhs.type)})),
             b.getNamedAttr("val_type", b.getStringAttr(outTy))})
            ->getResult(0);
    return {v, outTy.str()};
  }

  TaskTerm slice2D(Value view, ValueRange idxCoords, int64_t allAx0,
                   int64_t allAx1) {
    int64_t rank = cast<MemRefType>(view.getType()).getRank();
    auto elem = carrierType(cast<MemRefType>(view.getType()).getElementType());
    TaskTerm viewTerm = lanes[0].at(view);
    SmallVector<Value, 6> ops{viewTerm.v};
    SmallVector<Attribute, 6> sels;
    unsigned cPos = 0;
    for (int64_t ax = 0; ax < rank; ++ax) {
      if (ax == allAx0 || ax == allAx1) {
        sels.push_back(b.getStringAttr("all"));
      } else {
        ops.push_back(lanes[0].at(idxCoords[cPos++]).v);
        sels.push_back(b.getStringAttr("index"));
      }
    }
    int64_t loAx = std::min(allAx0, allAx1), hiAx = std::max(allAx0, allAx1);
    std::string slTy = "!weft_kernel.slice<!weft_kernel.encoding<\"" + elem +
                       "\",\"dense\",\"dense." + elem + "\",[]>,[" +
                       std::to_string(dimOf(view, "cpu.shape", loAx)) + "," +
                       std::to_string(dimOf(view, "cpu.shape", hiAx)) + "],[" +
                       std::to_string(axesMap.at(view)[loAx]) + "," +
                       std::to_string(axesMap.at(view)[hiAx]) + "]>";
    Value res =
        makeOp(b, loc, "task_slice", ops, {b.getIndexType()},
               {b.getNamedAttr("view_type", b.getStringAttr(viewTerm.type)),
                b.getNamedAttr("selectors", b.getArrayAttr(sels)),
                b.getNamedAttr("slice_type", b.getStringAttr(slTy)),
                b.getNamedAttr("val_type", b.getStringAttr(slTy)),
                b.getNamedAttr("admit", b.getBoolAttr(false))})
            ->getResult(0);
    return {res, slTy};
  }

  TaskTerm subview2D(TaskTerm baseSl, Value off0, Value off1, int64_t ext0,
                     int64_t ext1, bool admit, StringRef valTy) {
    Value res =
        makeOp(b, loc, "task_subview", {baseSl.v, off0, off1},
               {b.getIndexType()},
               {b.getNamedAttr("static_extent", b.getI64IntegerAttr(ext0)),
                b.getNamedAttr("static_extents",
                               b.getDenseI64ArrayAttr({ext0, ext1})),
                b.getNamedAttr("slice_type", b.getStringAttr(baseSl.type)),
                b.getNamedAttr("val_type", b.getStringAttr(valTy)),
                b.getNamedAttr("admit", b.getBoolAttr(admit))})
            ->getResult(0);
    return {res, valTy.str()};
  }

  TaskTerm slice1DAtTok(Value view, ArrayRef<Value> origCoords,
                        int64_t tokCoordPos, Value tokIv, int64_t vecAx,
                        StringRef admitTy) {
    auto elem = carrierType(cast<MemRefType>(view.getType()).getElementType());
    TaskTerm viewTerm = lanes[0].at(view);
    SmallVector<Value, 6> ops{viewTerm.v};
    SmallVector<Attribute, 6> sels;
    for (auto [idx, cVal] : llvm::enumerate(origCoords)) {
      ops.push_back(int64_t(idx) == tokCoordPos ? tokIv : lanes[0].at(cVal).v);
      sels.push_back(b.getStringAttr("index"));
    }
    sels.insert(sels.begin() + vecAx, b.getStringAttr("all"));
    std::string slTy = "!weft_kernel.slice<!weft_kernel.encoding<\"" + elem +
                       "\",\"dense\",\"dense." + elem + "\",[]>,[" +
                       std::to_string(dimOf(view, "cpu.shape", vecAx)) + "],[" +
                       std::to_string(axesMap.at(view)[vecAx]) + "]>";
    Value res =
        makeOp(b, loc, "task_slice", ops, {b.getIndexType()},
               {b.getNamedAttr("view_type", b.getStringAttr(viewTerm.type)),
                b.getNamedAttr("selectors", b.getArrayAttr(sels)),
                b.getNamedAttr("slice_type", b.getStringAttr(slTy)),
                b.getNamedAttr("val_type", b.getStringAttr(admitTy)),
                b.getNamedAttr("admit", b.getBoolAttr(true))})
            ->getResult(0);
    return {res, admitTy.str()};
  }

  bool matchTiledTwoPassContraction(Block &srcBb, Operation *localBufOp,
                                    TiledTwoPassPlan &plan) {
    if (isRvv || batch < 2 || !llvm::isPowerOf2_64(uint64_t(batch)))
      return false;
    auto geometry = space.getAttrOfType<DictionaryAttr>("cpu.vector_panel");
    if (!geometry)
      return false;
    plan.tokenTile = geometry.getAs<IntegerAttr>("token_tile").getInt();
    plan.valueWidth = geometry.getAs<IntegerAttr>("value_width").getInt();
    auto innerAttr = space.getAttrOfType<IntegerAttr>("cpu.innermost_owner");
    auto axesAttr = space.getAttrOfType<DenseI64ArrayAttr>("axes");
    if (!innerAttr || !axesAttr)
      return false;
    auto radices = intentdsl::cpu::ownerRadices(space);
    int64_t innerAx = innerAttr.getInt();
    if (radices.empty() || int64_t(radices[0]) != innerAx)
      return false;
    int64_t fullOwnExt =
        dimOf(space.getOperand(innerAx), "cpu.shape", axesAttr[innerAx]);
    if (fullOwnExt < batch || fullOwnExt % batch != 0)
      return false;
    plan.innerAx = innerAx;
    BlockArgument innerArg = srcBb.getArgument(innerAx);

    auto isTopCommon = [&](auto &self, Value v) -> bool {
      if (v == innerArg)
        return false;
      if (auto ba = dyn_cast<BlockArgument>(v))
        return ba.getOwner() == &srcBb || ba.getOwner() == &fn.front();
      Operation *def = v.getDefiningOp();
      if (!def || def->getBlock() != &srcBb)
        return false;
      if (def->hasAttr("cpu.owner_common"))
        return true;
      if (isa<arith::ConstantOp, memref::DimOp>(def))
        return true;
      return isMemoryEffectFree(def) && def->getNumRegions() == 0 &&
             llvm::all_of(def->getOperands(),
                          [&](Value x) { return self(self, x); });
    };

    SmallVector<Operation *, 8> topLoopsAndBranches;
    for (Operation &op : srcBb.without_terminator()) {
      if (isa<scf::ForOp>(op) || op.hasAttr("cpu.window_branch"))
        topLoopsAndBranches.push_back(&op);
    }
    if (topLoopsAndBranches.size() != 2)
      return false;

    auto loop1 = dyn_cast<scf::ForOp>(topLoopsAndBranches[0]);
    if (!loop1 || !loop1->hasAttr("kir.reassociate") ||
        staticIndexVal(fn, loop1.getStep()) != 1 ||
        loop1.getNumResults() != 1 || !loop1.getResult(0).getType().isF32() ||
        !isNegInfConst(loop1.getInitArgs()[0]) ||
        !isTopCommon(isTopCommon, loop1.getLowerBound()) ||
        !isTopCommon(isTopCommon, loop1.getUpperBound()))
      return false;
    plan.loop1 = loop1;

    Operation *stage2 = topLoopsAndBranches[1];
    plan.stage2Op = stage2;
    for (Operation *cur = loop1->getNextNode(); cur && cur != stage2;
         cur = cur->getNextNode()) {
      StringRef cn = cur->getName().getStringRef();
      if (cn == "cpu.buffer_store" || cn == "cpu.buffer_load" ||
          cur->getName().stripDialect() == "slice_write" ||
          isa<memref::StoreOp>(cur) || cur->getNumRegions() != 0)
        return false;
    }

    if (auto lp2 = dyn_cast<scf::ForOp>(stage2)) {
      if (localBufOp)
        plan.replayLoop = lp2;
      else
        plan.recomputeLoop = lp2;
    } else if (auto wb = dyn_cast<scf::IfOp>(stage2)) {
      if (!wb->hasAttr("cpu.window_branch") || wb.getElseRegion().empty() ||
          wb.getNumResults() != 2)
        return false;
      plan.windowBranch = wb;
      for (Operation &top : wb.thenBlock()->without_terminator())
        if (auto f = dyn_cast<scf::ForOp>(top))
          plan.replayLoop = f;
        else
          return false;
      for (Operation &eop : wb.elseBlock()->without_terminator())
        if (auto f = dyn_cast<scf::ForOp>(eop))
          plan.recomputeLoop = f;
        else
          return false;
      if (!plan.replayLoop || !plan.recomputeLoop)
        return false;
      if (wb.thenYield().getOperands() != plan.replayLoop.getResults() ||
          wb.elseYield().getOperands() != plan.recomputeLoop.getResults())
        return false;
    } else {
      return false;
    }
    if (stage2->getNumResults() != 2 || !stage2->getResult(0).getType().isF32() ||
        !isa<RankedTensorType>(stage2->getResult(1).getType()))
      return false;
    plan.stage2MassRes = stage2->getResult(0);
    plan.stage2AccRes = stage2->getResult(1);

    auto extractCausalCond = [&](Value cond, Value iv, Value &bndOut,
                                 bool &inclOut, bool &trueInPrefix) -> bool {
      auto cmpOp = cond.getDefiningOp<arith::CmpIOp>();
      if (!cmpOp)
        return false;
      auto pred = cmpOp.getPredicate();
      Value lhs = cmpOp.getLhs(), rhs = cmpOp.getRhs();
      auto validBound = [&](Value bVal) {
        return bVal == innerArg || isTopCommon(isTopCommon, bVal);
      };
      if (lhs == iv && validBound(rhs)) {
        if (pred == arith::CmpIPredicate::sle) {
          bndOut = rhs;
          inclOut = true;
          trueInPrefix = true;
          return true;
        }
        if (pred == arith::CmpIPredicate::slt) {
          bndOut = rhs;
          inclOut = false;
          trueInPrefix = true;
          return true;
        }
      } else if (rhs == iv && validBound(lhs)) {
        if (pred == arith::CmpIPredicate::sge) {
          bndOut = lhs;
          inclOut = true;
          trueInPrefix = true;
          return true;
        }
        if (pred == arith::CmpIPredicate::sgt) {
          bndOut = lhs;
          inclOut = false;
          trueInPrefix = true;
          return true;
        }
      }
      return false;
    };

    // Match Pass 1 body
    Value iv1 = loop1.getInductionVar();
    Value iterPeak = loop1.getRegionIterArgs()[0];
    Value yPeak = cast<scf::YieldOp>(loop1.getBody()->getTerminator()).getOperand(0);
    Operation *maxOp = yPeak.getDefiningOp();
    if (!maxOp || (maxOp->getName().getStringRef() != "arith.maxnumf" &&
                   maxOp->getName().getStringRef() != "arith.maximumf"))
      return false;
    Value stepScore = maxOp->getOperand(0) == iterPeak
                          ? maxOp->getOperand(1)
                          : (maxOp->getOperand(1) == iterPeak
                                 ? maxOp->getOperand(0)
                                 : Value{});
    if (!stepScore)
      return false;
    Value rawScore = stepScore;
    Operation *pass1CausalIf = nullptr;
    if (auto selOp = stepScore.getDefiningOp<arith::SelectOp>()) {
      Value bnd;
      bool incl = true, inTrue = true;
      if (!extractCausalCond(selOp.getCondition(), iv1, bnd, incl, inTrue))
        return false;
      Value outVal = inTrue ? selOp.getFalseValue() : selOp.getTrueValue();
      if (!isNegInfConst(outVal))
        return false;
      rawScore = inTrue ? selOp.getTrueValue() : selOp.getFalseValue();
      plan.hasCausal = true;
      plan.causalBoundVal = bnd;
      plan.causalInclusive = incl;
    } else if (auto ifOp = stepScore.getDefiningOp<scf::IfOp>()) {
      if (ifOp->getBlock() != loop1.getBody() || ifOp.getNumResults() != 1 ||
          ifOp.getElseRegion().empty())
        return false;
      auto pureBlock = [](Block *bb) {
        return llvm::all_of(bb->without_terminator(), [](Operation &o) {
          return isMemoryEffectFree(&o) && o.getNumRegions() == 0;
        });
      };
      if (!pureBlock(ifOp.thenBlock()) || !pureBlock(ifOp.elseBlock()))
        return false;
      Value bnd;
      bool incl = true, inTrue = true;
      if (!extractCausalCond(ifOp.getCondition(), iv1, bnd, incl, inTrue))
        return false;
      Value tVal = ifOp.thenYield().getOperand(0);
      Value eVal = ifOp.elseYield().getOperand(0);
      Value outVal = inTrue ? eVal : tVal;
      if (!isNegInfConst(outVal))
        return false;
      rawScore = inTrue ? tVal : eVal;
      pass1CausalIf = ifOp.getOperation();
      plan.hasCausal = true;
      plan.causalBoundVal = bnd;
      plan.causalInclusive = incl;
    }

    auto isFloatOfIndex = [](Value v, Value targetIdx) {
      auto sit = v.getDefiningOp<arith::SIToFPOp>();
      if (!sit)
        return false;
      auto ic = sit.getIn().getDefiningOp<arith::IndexCastOp>();
      return ic && ic.getIn() == targetIdx;
    };
    auto matchScoreExpr = [&](Value expr, Value tokIv, Value &dotOut,
                              Value &preScaleOut, Value &outerScaleOut,
                              Value &slopeOut, bool &negOut) -> bool {
      Value cur = expr;
      preScaleOut = Value{};
      outerScaleOut = Value{};
      slopeOut = Value{};
      negOut = false;
      Value maybeOuter;
      if (auto mulOuter = cur.getDefiningOp<arith::MulFOp>()) {
        if (isTopCommon(isTopCommon, mulOuter.getRhs())) {
          maybeOuter = mulOuter.getRhs();
          cur = mulOuter.getLhs();
        } else if (isTopCommon(isTopCommon, mulOuter.getLhs())) {
          maybeOuter = mulOuter.getLhs();
          cur = mulOuter.getRhs();
        }
      }
      if (auto addBias = cur.getDefiningOp<arith::AddFOp>()) {
        auto tryBiasTerm = [&](Value candBias, Value candDot) -> bool {
          auto mBias = candBias.getDefiningOp<arith::MulFOp>();
          if (!mBias)
            return false;
          Value sl, df;
          if (isTopCommon(isTopCommon, mBias.getLhs())) {
            sl = mBias.getLhs();
            df = mBias.getRhs();
          } else if (isTopCommon(isTopCommon, mBias.getRhs())) {
            sl = mBias.getRhs();
            df = mBias.getLhs();
          } else {
            return false;
          }
          auto subPos = df.getDefiningOp<arith::SubFOp>();
          if (!subPos)
            return false;
          if (isFloatOfIndex(subPos.getLhs(), tokIv) &&
              isFloatOfIndex(subPos.getRhs(), innerArg)) {
            slopeOut = sl;
            negOut = false;
          } else if (isFloatOfIndex(subPos.getLhs(), innerArg) &&
                     isFloatOfIndex(subPos.getRhs(), tokIv)) {
            slopeOut = sl;
            negOut = true;
          } else {
            return false;
          }
          outerScaleOut = maybeOuter;
          cur = candDot;
          if (auto mulPre = cur.getDefiningOp<arith::MulFOp>()) {
            if (isTopCommon(isTopCommon, mulPre.getRhs())) {
              preScaleOut = mulPre.getRhs();
              cur = mulPre.getLhs();
            } else if (isTopCommon(isTopCommon, mulPre.getLhs())) {
              preScaleOut = mulPre.getLhs();
              cur = mulPre.getRhs();
            }
          }
          return true;
        };
        if (!tryBiasTerm(addBias.getRhs(), addBias.getLhs()) &&
            !tryBiasTerm(addBias.getLhs(), addBias.getRhs()))
          return false;
      } else {
        preScaleOut = maybeOuter;
      }
      dotOut = cur;
      return true;
    };

    Value dotVal;
    if (!matchScoreExpr(rawScore, iv1, dotVal, plan.scaleVal, plan.outerScaleVal,
                        plan.biasSlopeVal, plan.biasNegate))
      return false;
    Operation *laneSum = dotVal.getDefiningOp();
    if (!laneSum || laneSum->getName().stripDialect() != "lane_sum")
      return false;
    plan.scoreEngine = laneSum->getAttrOfType<StringAttr>("cpu.panel_engine");
    auto dotMul = laneSum->getOperand(0).getDefiningOp<arith::MulFOp>();
    if (!dotMul)
      return false;
    Value qVecVal, kVecVal;
    if (dotMul.getLhs().getParentBlock() == &srcBb &&
        dotMul.getRhs().getParentBlock() == loop1.getBody()) {
      qVecVal = dotMul.getLhs();
      kVecVal = dotMul.getRhs();
    } else if (dotMul.getRhs().getParentBlock() == &srcBb &&
               dotMul.getLhs().getParentBlock() == loop1.getBody()) {
      qVecVal = dotMul.getRhs();
      kVecVal = dotMul.getLhs();
    } else {
      return false;
    }

    // Match Query vector read (pre-loop)
    Value qRawVal = qVecVal;
    if (auto ext = qVecVal.getDefiningOp<arith::ExtFOp>()) {
      plan.qExtOp = ext.getOperation();
      qRawVal = ext.getIn();
    }
    Operation *qRead = qRawVal.getDefiningOp();
    if (!qRead || qRead->getBlock() != &srcBb ||
        qRead->getName().stripDialect() != "slice_read")
      return false;
    if (plan.qExtOp && !qRead->hasOneUse())
      return false;
    for (Operation *u : qVecVal.getUsers())
      if (!loop1->isAncestor(u) && !stage2->isAncestor(u))
        return false;
    plan.qReadOp = qRead;
    plan.qMem = qRead->getOperand(0);
    if (!isa<BlockArgument>(plan.qMem))
      return false;
    auto qMemTy = cast<MemRefType>(plan.qMem.getType());
    Type qElem = qMemTy.getElementType();
    if (!(qElem.isF32() || qElem.isBF16() || qElem.isF16()))
      return false;
    int64_t qRank = qMemTy.getRank();
    plan.qVecMemAx = qRead->getAttrOfType<IntegerAttr>("axis").getInt();
    if (qRank < 2 || plan.qVecMemAx != qRank - 1 ||
        dimOf(plan.qMem, "cpu.strides", plan.qVecMemAx) != 1)
      return false;
    plan.D = dimOf(plan.qMem, "cpu.shape", plan.qVecMemAx);
    if (plan.D < 16 || plan.D % 16 != 0)
      return false;
    int64_t qOwnPos = -1;
    for (auto [pos, c] : llvm::enumerate(qRead->getOperands().drop_front())) {
      if (c == innerArg) {
        if (qOwnPos >= 0)
          return false;
        qOwnPos = int64_t(pos);
      } else if (!isTopCommon(isTopCommon, c)) {
        return false;
      }
    }
    if (qOwnPos < 0)
      return false;
    plan.qOwnMemAx = qOwnPos;
    if (dimOf(plan.qMem, "cpu.shape", plan.qOwnMemAx) < batch)
      return false;
    plan.axOwn = axesMap.at(plan.qMem)[plan.qOwnMemAx];
    plan.axD = axesMap.at(plan.qMem)[plan.qVecMemAx];

    // Match Key vector read inside loop1
    auto matchKeyDot = [&](Value kVal, scf::ForOp parentLoop, bool recordPlan) -> bool {
      Value kRaw = kVal;
      if (auto ext = kVal.getDefiningOp<arith::ExtFOp>())
        kRaw = ext.getIn();
      Operation *kRead = kRaw.getDefiningOp();
      if (!kRead || kRead->getBlock() != parentLoop.getBody() ||
          kRead->getName().stripDialect() != "slice_read")
        return false;
      Value kMem = kRead->getOperand(0);
      if (!isa<BlockArgument>(kMem))
        return false;
      auto kMemTy = cast<MemRefType>(kMem.getType());
      Type kElem = kMemTy.getElementType();
      if (!(kElem.isF32() || kElem.isBF16() || kElem.isF16()))
        return false;
      int64_t kRank = kMemTy.getRank();
      int64_t kVecAx = kRead->getAttrOfType<IntegerAttr>("axis").getInt();
      if (kRank < 2 || kVecAx != kRank - 1 ||
          dimOf(kMem, "cpu.strides", kVecAx) != 1 ||
          dimOf(kMem, "cpu.shape", kVecAx) != plan.D ||
          axesMap.at(kMem)[kVecAx] != plan.axD)
        return false;
      int64_t tokPos = -1;
      SmallVector<Value, 4> coords;
      for (auto [pos, c] : llvm::enumerate(kRead->getOperands().drop_front())) {
        coords.push_back(c);
        if (c == parentLoop.getInductionVar()) {
          if (tokPos >= 0)
            return false;
          tokPos = int64_t(pos);
        } else if (!isTopCommon(isTopCommon, c)) {
          return false;
        }
      }
      if (tokPos < 0)
        return false;
      int64_t kExt = dimOf(kMem, "cpu.shape", tokPos);
      if (kExt < plan.tokenTile)
        return false;
      int64_t axTok = axesMap.at(kMem)[tokPos];
      if (axTok == plan.axOwn || axTok == plan.axD)
        return false;
      if (recordPlan) {
        plan.kMem = kMem;
        plan.kCoords = std::move(coords);
        plan.kTokCoordPos = tokPos;
        plan.kTokMemAx = tokPos;
        plan.kVecMemAx = kVecAx;
        plan.KExt = kExt;
        plan.axTok = axTok;
      } else {
        if (kMem != plan.kMem || tokPos != plan.kTokCoordPos ||
            kVecAx != plan.kVecMemAx || coords.size() != plan.kCoords.size())
          return false;
        for (size_t i = 0; i < coords.size(); ++i)
          if (int64_t(i) != tokPos && coords[i] != plan.kCoords[i])
            return false;
      }
      return true;
    };
    if (!matchKeyDot(kVecVal, loop1, true))
      return false;

    // Check score buffer store and side effects in loop1
    unsigned bufStores = 0;
    for (Operation &op : loop1.getBody()->without_terminator()) {
      StringRef n = op.getName().getStringRef();
      if (n == "cpu.buffer_store") {
        ++bufStores;
        if (!localBufOp || op.getOperand(0) != rawScore ||
            op.getOperand(1) != localBufOp->getResult(0))
          return false;
        Value off = op.getOperand(2);
        bool okOff = (off == iv1);
        if (auto sub = off.getDefiningOp<arith::SubIOp>())
          okOff = (sub.getLhs() == iv1 && sub.getRhs() == loop1.getLowerBound());
        if (!okOff)
          return false;
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        if (&op == pass1CausalIf)
          continue;
        if (!ifOp->hasAttr("cpu.window_guard") || !ifOp.getElseRegion().empty())
          return false;
        plan.windowGuarded = true;
        for (Operation &inner : ifOp.thenBlock()->without_terminator()) {
          if (inner.getName().getStringRef() != "cpu.buffer_store")
            return false;
          ++bufStores;
          if (!localBufOp || inner.getOperand(0) != rawScore ||
              inner.getOperand(1) != localBufOp->getResult(0))
            return false;
          auto sub = inner.getOperand(2).getDefiningOp<arith::SubIOp>();
          if (!sub || sub.getLhs() != iv1 ||
              sub.getRhs() != loop1.getLowerBound())
            return false;
        }
      } else if (op.getNumRegions() != 0 ||
                 (op.getName().stripDialect() != "slice_read" &&
                  op.getName().stripDialect() != "lane_sum" &&
                  !isMemoryEffectFree(&op))) {
        return false;
      }
    }
    if (localBufOp) {
      if (bufStores != 1)
        return false;
      plan.hasScoreBuffer = true;
      plan.scoreSlots = geometry.getAs<IntegerAttr>("score_slots").getInt();
      if (!plan.scoreSlots)
        plan.scoreSlots = localBufOp->getAttrOfType<IntegerAttr>("elements").getInt();
      if (plan.scoreSlots < plan.tokenTile ||
          plan.scoreSlots % plan.tokenTile != 0)
        return false;
    } else if (bufStores != 0) {
      return false;
    }

    // Match Pass 2 loop(s)
    auto matchPass2 = [&](scf::ForOp lp2, bool expectBufLoad,
                          bool recordV) -> bool {
      if (!lp2 || !lp2->hasAttr("kir.reassociate") ||
          staticIndexVal(fn, lp2.getStep()) != 1 || lp2.getNumResults() != 2)
        return false;
      if (lp2.getLowerBound() != loop1.getLowerBound() &&
          staticIndexVal(fn, lp2.getLowerBound()) !=
              staticIndexVal(fn, loop1.getLowerBound()))
        return false;
      if (lp2.getUpperBound() != loop1.getUpperBound())
        return false;
      if (!isZeroFloatConst(lp2.getInitArgs()[0]))
        return false;
      Operation *initSplat = lp2.getInitArgs()[1].getDefiningOp();
      if (!initSplat || initSplat->getName().stripDialect() != "lane_splat" ||
          !isZeroFloatConst(initSplat->getOperand(0)))
        return false;
      if (recordV)
        plan.zeroAccInitVal = lp2.getInitArgs()[1];

      Value iv2 = lp2.getInductionVar();
      Value iterMass = lp2.getRegionIterArgs()[0];
      Value iterAcc = lp2.getRegionIterArgs()[1];
      auto y2 = cast<scf::YieldOp>(lp2.getBody()->getTerminator());
      auto massAdd = y2.getOperand(0).getDefiningOp<arith::AddFOp>();
      if (!massAdd)
        return false;
      Value weightVal = massAdd.getLhs() == iterMass
                            ? massAdd.getRhs()
                            : (massAdd.getRhs() == iterMass ? massAdd.getLhs()
                                                             : Value{});
      if (!weightVal)
        return false;

      Value expVal = weightVal;
      Operation *pass2CausalIf = nullptr;
      if (plan.hasCausal) {
        if (auto sel2 = weightVal.getDefiningOp<arith::SelectOp>()) {
          Value bnd2;
          bool incl2 = true, inTrue2 = true;
          if (!extractCausalCond(sel2.getCondition(), iv2, bnd2, incl2,
                                 inTrue2) ||
              bnd2 != plan.causalBoundVal || incl2 != plan.causalInclusive)
            return false;
          Value zVal = inTrue2 ? sel2.getFalseValue() : sel2.getTrueValue();
          if (!isZeroFloatConst(zVal))
            return false;
          expVal = inTrue2 ? sel2.getTrueValue() : sel2.getFalseValue();
        } else if (auto if2 = weightVal.getDefiningOp<scf::IfOp>()) {
          if (if2->getBlock() != lp2.getBody() || if2.getNumResults() != 1 ||
              if2.getElseRegion().empty())
            return false;
          auto pureBlock = [](Block *bb) {
            return llvm::all_of(bb->without_terminator(), [](Operation &o) {
              return isMemoryEffectFree(&o) && o.getNumRegions() == 0;
            });
          };
          if (!pureBlock(if2.thenBlock()) || !pureBlock(if2.elseBlock()))
            return false;
          Value bnd2;
          bool incl2 = true, inTrue2 = true;
          if (!extractCausalCond(if2.getCondition(), iv2, bnd2, incl2,
                                 inTrue2) ||
              bnd2 != plan.causalBoundVal || incl2 != plan.causalInclusive)
            return false;
          Value tVal = if2.thenYield().getOperand(0);
          Value eVal = if2.elseYield().getOperand(0);
          Value zVal = inTrue2 ? eVal : tVal;
          if (!isZeroFloatConst(zVal))
            return false;
          expVal = inTrue2 ? tVal : eVal;
          pass2CausalIf = if2.getOperation();
        } else {
          return false;
        }
      } else if (isa_and_nonnull<arith::SelectOp, scf::IfOp>(
                     weightVal.getDefiningOp())) {
        return false;
      }

      Operation *expOp = expVal.getDefiningOp();
      if (!expOp || (expOp->getName().getStringRef() != "math.exp2" &&
                     expOp->getName().getStringRef() != "math.exp"))
        return false;
      bool isExp2 = expOp->getName().getStringRef() == "math.exp2";
      if (recordV)
        plan.useExp2 = isExp2;
      else if (plan.useExp2 != isExp2)
        return false;

      Value expArg = expOp->getOperand(0);
      Value postSc;
      if (auto mulPost = expArg.getDefiningOp<arith::MulFOp>()) {
        if (isTopCommon(isTopCommon, mulPost.getRhs())) {
          expArg = mulPost.getLhs();
          postSc = mulPost.getRhs();
        } else if (isTopCommon(isTopCommon, mulPost.getLhs())) {
          expArg = mulPost.getRhs();
          postSc = mulPost.getLhs();
        }
      }
      if (recordV)
        plan.postScaleVal = postSc;
      else if (plan.postScaleVal != postSc)
        return false;

      auto diffSub = expArg.getDefiningOp<arith::SubFOp>();
      if (!diffSub || diffSub.getRhs() != loop1.getResult(0))
        return false;
      Value score2 = diffSub.getLhs();
      if (expectBufLoad) {
        Operation *ld = score2.getDefiningOp();
        if (!ld || ld->getName().getStringRef() != "cpu.buffer_load" ||
            !localBufOp || ld->getOperand(0) != localBufOp->getResult(0))
          return false;
        Value off2 = ld->getOperand(1);
        bool okOff = (off2 == iv2);
        if (auto sub = off2.getDefiningOp<arith::SubIOp>())
          okOff = (sub.getLhs() == iv2 && sub.getRhs() == loop1.getLowerBound());
        if (!okOff)
          return false;
      } else {
        Value dot2, sc2, osc2, sl2;
        bool neg2 = false;
        if (!matchScoreExpr(score2, iv2, dot2, sc2, osc2, sl2, neg2) ||
            sc2 != plan.scaleVal || osc2 != plan.outerScaleVal ||
            sl2 != plan.biasSlopeVal || neg2 != plan.biasNegate)
          return false;
        Operation *ls2 = dot2.getDefiningOp();
        if (!ls2 || ls2->getName().stripDialect() != "lane_sum")
          return false;
        auto dm2 = ls2->getOperand(0).getDefiningOp<arith::MulFOp>();
        if (!dm2)
          return false;
        Value k2 = dm2.getLhs() == qVecVal
                       ? dm2.getRhs()
                       : (dm2.getRhs() == qVecVal ? dm2.getLhs() : Value{});
        if (!k2 || !matchKeyDot(k2, lp2, false))
          return false;
      }

      // Match weighted V accumulation
      auto accAdd = y2.getOperand(1).getDefiningOp<arith::AddFOp>();
      if (!accAdd)
        return false;
      if (auto engine = accAdd->getAttrOfType<StringAttr>("cpu.pv_engine"))
        plan.valueEngine = engine;
      Value termVec = accAdd.getLhs() == iterAcc
                          ? accAdd.getRhs()
                          : (accAdd.getRhs() == iterAcc ? accAdd.getLhs()
                                                         : Value{});
      auto vMul = termVec ? termVec.getDefiningOp<arith::MulFOp>()
                          : arith::MulFOp();
      if (!vMul)
        return false;
      Value vVecVal, splatVal;
      if (vMul.getLhs().getDefiningOp() &&
          vMul.getLhs().getDefiningOp()->getName().stripDialect() ==
              "lane_splat") {
        splatVal = vMul.getLhs();
        vVecVal = vMul.getRhs();
      } else if (vMul.getRhs().getDefiningOp() &&
                 vMul.getRhs().getDefiningOp()->getName().stripDialect() ==
                     "lane_splat") {
        splatVal = vMul.getRhs();
        vVecVal = vMul.getLhs();
      } else {
        return false;
      }
      Value factorVal = splatVal.getDefiningOp()->getOperand(0);
      std::optional<Type> roundTy;
      if (factorVal != weightVal) {
        auto ext = factorVal.getDefiningOp<arith::ExtFOp>();
        auto tr = ext ? ext.getIn().getDefiningOp<arith::TruncFOp>()
                      : arith::TruncFOp();
        if (!tr || tr.getIn() != weightVal)
          return false;
        Type rt = tr.getType();
        if (!rt.isBF16() && !rt.isF16())
          return false;
        roundTy = rt;
      }
      if (recordV)
        plan.weightRoundType = roundTy;
      else if (plan.weightRoundType != roundTy)
        return false;

      Value vRaw = vVecVal;
      if (auto ext = vVecVal.getDefiningOp<arith::ExtFOp>())
        vRaw = ext.getIn();
      Operation *vRead = vRaw.getDefiningOp();
      if (!vRead || vRead->getBlock() != lp2.getBody() ||
          vRead->getName().stripDialect() != "slice_read")
        return false;
      Value vMem = vRead->getOperand(0);
      if (!isa<BlockArgument>(vMem))
        return false;
      auto vMemTy = cast<MemRefType>(vMem.getType());
      Type vElem = vMemTy.getElementType();
      if (!(vElem.isF32() || vElem.isBF16() || vElem.isF16()))
        return false;
      int64_t vRank = vMemTy.getRank();
      int64_t vVecAx = vRead->getAttrOfType<IntegerAttr>("axis").getInt();
      if (vRank < 2 || vVecAx != vRank - 1 ||
          dimOf(vMem, "cpu.strides", vVecAx) != 1)
        return false;
      int64_t dv = dimOf(vMem, "cpu.shape", vVecAx);
      if (dv < 16 || dv % 16 != 0)
        return false;
      if (plan.valueWidth <= 0 || dv % plan.valueWidth != 0)
        return false;
      int64_t vTokPos = -1;
      SmallVector<Value, 4> vCoords;
      for (auto [pos, c] : llvm::enumerate(vRead->getOperands().drop_front())) {
        vCoords.push_back(c);
        if (c == iv2) {
          if (vTokPos >= 0)
            return false;
          vTokPos = int64_t(pos);
        } else if (!isTopCommon(isTopCommon, c)) {
          return false;
        }
      }
      if (vTokPos < 0 || dimOf(vMem, "cpu.shape", vTokPos) != plan.KExt ||
          axesMap.at(vMem)[vTokPos] != plan.axTok)
        return false;
      int64_t axDV = axesMap.at(vMem)[vVecAx];
      if (axDV == plan.axOwn || axDV == plan.axTok)
        return false;
      if (recordV) {
        plan.vMem = vMem;
        plan.vCoords = std::move(vCoords);
        plan.vTokCoordPos = vTokPos;
        plan.vTokMemAx = vTokPos;
        plan.vVecMemAx = vVecAx;
        plan.DV = dv;
        plan.axDV = axDV;
      } else {
        if (vMem != plan.vMem || vTokPos != plan.vTokCoordPos ||
            vVecAx != plan.vVecMemAx || vCoords.size() != plan.vCoords.size())
          return false;
        for (size_t i = 0; i < vCoords.size(); ++i)
          if (int64_t(i) != vTokPos && vCoords[i] != plan.vCoords[i])
            return false;
      }

      for (Operation &op : lp2.getBody()->without_terminator()) {
        StringRef n = op.getName().getStringRef();
        StringRef sn = op.getName().stripDialect();
        if (&op == pass2CausalIf)
          continue;
        if (op.getNumRegions() != 0)
          return false;
        if (n != "cpu.buffer_load" && sn != "slice_read" &&
            sn != "lane_splat" && sn != "lane_sum" && !isMemoryEffectFree(&op))
          return false;
      }
      return true;
    };

    if (plan.replayLoop && !matchPass2(plan.replayLoop, true, true))
      return false;
    if (plan.recomputeLoop &&
        !matchPass2(plan.recomputeLoop, false, !plan.replayLoop))
      return false;

    // Match publication of stage2AccRes
    if (!plan.stage2AccRes.hasOneUse())
      return false;
    Operation *accUser = *plan.stage2AccRes.getUsers().begin();
    Value pubVal = plan.stage2AccRes;
    if (auto mulOp = dyn_cast<arith::MulFOp>(accUser)) {
      if (mulOp->getBlock() != &srcBb || !mulOp.getResult().hasOneUse())
        return false;
      Value other = mulOp.getLhs() == plan.stage2AccRes ? mulOp.getRhs()
                                                        : mulOp.getLhs();
      Operation *sp = other.getDefiningOp();
      if (!sp || sp->getBlock() != &srcBb || !sp->hasOneUse() ||
          sp->getName().stripDialect() != "lane_splat")
        return false;
      plan.ySplatOp = sp;
      plan.yMulOp = mulOp.getOperation();
      pubVal = mulOp.getResult();

      Value invScalar = sp->getOperand(0);
      auto matchRecip = [&](Value v) {
        auto d = v.getDefiningOp<arith::DivFOp>();
        return d && isOneFloatConst(d.getLhs()) &&
               d.getRhs() == plan.stage2MassRes;
      };
      if (matchRecip(invScalar)) {
        plan.hasDirectRecip = true;
      } else if (auto ifInv = invScalar.getDefiningOp<scf::IfOp>()) {
        if (ifInv->getBlock() != &srcBb || ifInv.getNumResults() != 1 ||
            ifInv.getElseRegion().empty() ||
            !isTopCommon(isTopCommon, ifInv.getCondition()))
          return false;
        auto pureBlock = [](Block *bb) {
          return llvm::all_of(bb->without_terminator(), [](Operation &o) {
            return isMemoryEffectFree(&o) && o.getNumRegions() == 0;
          });
        };
        Value tVal = ifInv.thenYield().getOperand(0);
        Value eVal = ifInv.elseYield().getOperand(0);
        if (!pureBlock(ifInv.thenBlock()) || !pureBlock(ifInv.elseBlock()))
          return false;
        if (isZeroFloatConst(tVal) && matchRecip(eVal)) {
          plan.invIfOrSelOp = ifInv.getOperation();
          plan.invCondVal = ifInv.getCondition();
          plan.invZeroWhenTrue = true;
        } else if (matchRecip(tVal) && isZeroFloatConst(eVal)) {
          plan.invIfOrSelOp = ifInv.getOperation();
          plan.invCondVal = ifInv.getCondition();
          plan.invZeroWhenTrue = false;
        } else {
          return false;
        }
      } else if (auto selInv = invScalar.getDefiningOp<arith::SelectOp>()) {
        if (selInv->getBlock() != &srcBb ||
            !isTopCommon(isTopCommon, selInv.getCondition()))
          return false;
        if (isZeroFloatConst(selInv.getTrueValue()) &&
            matchRecip(selInv.getFalseValue())) {
          plan.invIfOrSelOp = selInv.getOperation();
          plan.invCondVal = selInv.getCondition();
          plan.invZeroWhenTrue = true;
        } else if (matchRecip(selInv.getTrueValue()) &&
                   isZeroFloatConst(selInv.getFalseValue())) {
          plan.invIfOrSelOp = selInv.getOperation();
          plan.invCondVal = selInv.getCondition();
          plan.invZeroWhenTrue = false;
        } else {
          return false;
        }
      } else {
        return false;
      }
    }

    if (pubVal != plan.stage2AccRes)
      accUser = *pubVal.getUsers().begin();
    if (auto trOp = dyn_cast<arith::TruncFOp>(accUser)) {
      if (trOp->getBlock() != &srcBb || !trOp.getResult().hasOneUse())
        return false;
      plan.yTruncOp = trOp.getOperation();
      pubVal = trOp.getResult();
      accUser = *pubVal.getUsers().begin();
    }
    if (!accUser || accUser->getBlock() != &srcBb ||
        accUser->getName().stripDialect() != "slice_write" ||
        accUser->getOperand(0) != pubVal ||
        accUser->hasAttr("weft_host.member_store"))
      return false;
    plan.yWriteOp = accUser;
    plan.yMem = accUser->getOperand(1);
    if (!isa<BlockArgument>(plan.yMem))
      return false;
    auto yMemTy = cast<MemRefType>(plan.yMem.getType());
    Type yElem = yMemTy.getElementType();
    if (!(yElem.isF32() || yElem.isBF16() || yElem.isF16()))
      return false;
    int64_t yRank = yMemTy.getRank();
    plan.yVecMemAx = accUser->getAttrOfType<IntegerAttr>("axis").getInt();
    if (yRank < 2 || plan.yVecMemAx != yRank - 1 ||
        dimOf(plan.yMem, "cpu.strides", plan.yVecMemAx) != 1 ||
        dimOf(plan.yMem, "cpu.shape", plan.yVecMemAx) != plan.DV ||
        axesMap.at(plan.yMem)[plan.yVecMemAx] != plan.axDV)
      return false;
    int64_t yOwnPos = -1;
    for (auto [pos, c] : llvm::enumerate(accUser->getOperands().drop_front(2))) {
      if (c == innerArg) {
        if (yOwnPos >= 0)
          return false;
        yOwnPos = int64_t(pos);
      } else if (!isTopCommon(isTopCommon, c)) {
        return false;
      }
    }
    if (yOwnPos < 0 || dimOf(plan.yMem, "cpu.shape", yOwnPos) < batch ||
        axesMap.at(plan.yMem)[yOwnPos] != plan.axOwn)
      return false;
    plan.yOwnMemAx = yOwnPos;
    return true;
  }

  LogicalResult lowerTiledTwoPassBody(Block &srcBb, TiledTwoPassPlan &plan,
                                      TaskTerm iotaOwn64, TaskTerm iotaTok64,
                                      TaskTerm qSlotOff, TaskTerm tileSlotOff) {
    const int64_t B = batch, T = plan.tokenTile;
    const std::string vecBTy = "!weft_kernel.value<f32,[" + std::to_string(B) +
                               "],[" + std::to_string(plan.axOwn) + "]>";
    const std::string maskBTy = "!weft_kernel.value<i1,[" + std::to_string(B) +
                                "],[" + std::to_string(plan.axOwn) + "]>";
    const std::string si64BTy = "!weft_kernel.value<si64,[" +
                                std::to_string(B) + "],[" +
                                std::to_string(plan.axOwn) + "]>";
    const std::string si64TTy = "!weft_kernel.value<si64,[" +
                                std::to_string(T) + "],[" +
                                std::to_string(plan.axTok) + "]>";
    const std::string vecTTy = "!weft_kernel.value<f32,[" + std::to_string(T) +
                               "],[" + std::to_string(plan.axTok) + "]>";
    const std::string scoreMatTy =
        "!weft_kernel.value<f32,[" + std::to_string(B) + "," +
        std::to_string(T) + "],[" + std::to_string(plan.axOwn) + "," +
        std::to_string(plan.axTok) + "]>";
    const std::string maskMatTy =
        "!weft_kernel.value<i1,[" + std::to_string(B) + "," +
        std::to_string(T) + "],[" + std::to_string(plan.axOwn) + "," +
        std::to_string(plan.axTok) + "]>";
    const std::string si64MatTy =
        "!weft_kernel.value<si64,[" + std::to_string(B) + "," +
        std::to_string(T) + "],[" + std::to_string(plan.axOwn) + "," +
        std::to_string(plan.axTok) + "]>";
    const std::string qElem =
        scalarType(cast<MemRefType>(plan.qMem.getType()).getElementType());
    const std::string qRawMatTy =
        "!weft_kernel.value<" + qElem + ",[" + std::to_string(B) + "," +
        std::to_string(plan.D) + "],[" + std::to_string(plan.axOwn) + "," +
        std::to_string(plan.axD) + "]>";
    const std::string qF32MatTy =
        "!weft_kernel.value<f32,[" + std::to_string(B) + "," +
        std::to_string(plan.D) + "],[" + std::to_string(plan.axOwn) + "," +
        std::to_string(plan.axD) + "]>";
    const std::string kElem =
        scalarType(cast<MemRefType>(plan.kMem.getType()).getElementType());
    const std::string kRawMatTy =
        "!weft_kernel.value<" + kElem + ",[" + std::to_string(T) + "," +
        std::to_string(plan.D) + "],[" + std::to_string(plan.axTok) + "," +
        std::to_string(plan.axD) + "]>";
    const std::string kF32MatTy =
        "!weft_kernel.value<f32,[" + std::to_string(T) + "," +
        std::to_string(plan.D) + "],[" + std::to_string(plan.axTok) + "," +
        std::to_string(plan.axD) + "]>";
    const std::string kRaw1DTy =
        "!weft_kernel.value<" + kElem + ",[" + std::to_string(plan.D) + "],[" +
        std::to_string(plan.axD) + "]>";
    const std::string kF321DTy =
        "!weft_kernel.value<f32,[" + std::to_string(plan.D) + "],[" +
        std::to_string(plan.axD) + "]>";
    const std::string vElem =
        scalarType(cast<MemRefType>(plan.vMem.getType()).getElementType());
    const std::string vRawMatTy =
        "!weft_kernel.value<" + vElem + ",[" + std::to_string(T) + "," +
        std::to_string(plan.DV) + "],[" + std::to_string(plan.axTok) + "," +
        std::to_string(plan.axDV) + "]>";
    const std::string vF32MatTy =
        "!weft_kernel.value<f32,[" + std::to_string(T) + "," +
        std::to_string(plan.DV) + "],[" + std::to_string(plan.axTok) + "," +
        std::to_string(plan.axDV) + "]>";
    const std::string vRaw1DTy =
        "!weft_kernel.value<" + vElem + ",[" + std::to_string(plan.DV) + "],[" +
        std::to_string(plan.axDV) + "]>";
    const std::string vF321DTy =
        "!weft_kernel.value<f32,[" + std::to_string(plan.DV) + "],[" +
        std::to_string(plan.axDV) + "]>";
    const std::string outMatTy =
        "!weft_kernel.value<f32,[" + std::to_string(B) + "," +
        std::to_string(plan.DV) + "],[" + std::to_string(plan.axOwn) + "," +
        std::to_string(plan.axDV) + "]>";
    const std::string yElem =
        scalarType(cast<MemRefType>(plan.yMem.getType()).getElementType());
    const std::string yMatTy =
        "!weft_kernel.value<" + yElem + ",[" + std::to_string(B) + "," +
        std::to_string(plan.DV) + "],[" + std::to_string(plan.axOwn) + "," +
        std::to_string(plan.axDV) + "]>";

    if (plan.hasCausal)
      anyTightenedCausalLoop = true;

    BlockArgument innerArg = srcBb.getArgument(plan.innerAx);
    llvm::DenseSet<Operation *> skipOps;
    skipOps.insert(plan.qReadOp);
    if (plan.qExtOp)
      skipOps.insert(plan.qExtOp);
    if (plan.zeroAccInitVal)
      if (Operation *zDef = plan.zeroAccInitVal.getDefiningOp())
        if (zDef->hasOneUse() ||
            llvm::all_of(zDef->getUsers(), [&](Operation *u) {
              return plan.stage2Op->isAncestor(u);
            }))
          skipOps.insert(zDef);
    if (plan.invIfOrSelOp && plan.invIfOrSelOp->hasOneUse())
      skipOps.insert(plan.invIfOrSelOp);
    if (plan.hasDirectRecip)
      if (Operation *dOp = plan.ySplatOp->getOperand(0).getDefiningOp();
          dOp && dOp->hasOneUse())
        skipOps.insert(dOp);
    if (plan.ySplatOp)
      skipOps.insert(plan.ySplatOp);
    if (plan.yMulOp)
      skipOps.insert(plan.yMulOp);
    if (plan.yTruncOp)
      skipOps.insert(plan.yTruncOp);

    TaskTerm qForContract, qRawSaved, scaleTerm, outerScaleTerm, postScaleTerm,
        effBiasSlopeTerm, qBiasTerm, qPosF32, peakVec, massVec, outMat;
    bool directSubCommitted = false;
    bool outInTiledBuf = false;
    const bool nativeQK =
        !isRvv && (qElem == "f32" || qElem == "bf16" || qElem == "f16") &&
        (kElem == "f32" || kElem == "bf16" || kElem == "f16");
    const bool nativeV =
        !isRvv && (vElem == "f32" || vElem == "bf16" || vElem == "f16");
    const int64_t subDV = plan.valueWidth;
    const bool useTiledOut = bool(tiledOutBuf.v);
    const std::string subOutTy =
        "!weft_kernel.value<f32,[" + std::to_string(B) + "," +
        std::to_string(subDV) + "],[" + std::to_string(plan.axOwn) + "," +
        std::to_string(plan.axDV) + "]>";
    const std::string subMaskOutTy =
        "!weft_kernel.value<i1,[" + std::to_string(B) + "," +
        std::to_string(subDV) + "],[" + std::to_string(plan.axOwn) + "," +
        std::to_string(plan.axDV) + "]>";
    const std::string subSi64OutTy =
        "!weft_kernel.value<si64,[" + std::to_string(B) + "," +
        std::to_string(subDV) + "],[" + std::to_string(plan.axOwn) + "," +
        std::to_string(plan.axDV) + "]>";
    const std::string vSubRawMatTy =
        "!weft_kernel.value<" + vElem + ",[" + std::to_string(T) + "," +
        std::to_string(subDV) + "],[" + std::to_string(plan.axTok) + "," +
        std::to_string(plan.axDV) + "]>";
    const std::string vSubF32MatTy =
        "!weft_kernel.value<f32,[" + std::to_string(T) + "," +
        std::to_string(subDV) + "],[" + std::to_string(plan.axTok) + "," +
        std::to_string(plan.axDV) + "]>";
    const std::string vSubRaw1DTy =
        "!weft_kernel.value<" + vElem + ",[" + std::to_string(subDV) + "],[" +
        std::to_string(plan.axDV) + "]>";
    const std::string vSubF321DTy =
        "!weft_kernel.value<f32,[" + std::to_string(subDV) + "],[" +
        std::to_string(plan.axDV) + "]>";
    const std::string ySubMatTy =
        "!weft_kernel.value<" + yElem + ",[" + std::to_string(B) + "," +
        std::to_string(subDV) + "],[" + std::to_string(plan.axOwn) + "," +
        std::to_string(plan.axDV) + "]>";
    TaskTerm subOutSlotOff, trueSubO, zeroSubO;
    if (useTiledOut) {
      std::string idxSubOnAxDVTy = "!weft_kernel.value<index,[" +
                                   std::to_string(subDV) + "],[" +
                                   std::to_string(plan.axDV) + "]>";
      std::string si64SubOnAxDVTy = "!weft_kernel.value<si64,[" +
                                    std::to_string(subDV) + "],[" +
                                    std::to_string(plan.axDV) + "]>";
      Value isAxDV =
          makeOp(b, loc, "task_iota", {}, {b.getIndexType()},
                 {b.getNamedAttr("start", b.getI64IntegerAttr(0)),
                  b.getNamedAttr("end", b.getI64IntegerAttr(subDV)),
                  b.getNamedAttr("val_type", b.getStringAttr(idxSubOnAxDVTy))})
              ->getResult(0);
      TaskTerm iotaSubAxDV64 =
          castOp({isAxDV, idxSubOnAxDVTy}, si64SubOnAxDVTy);
      TaskTerm cDV64 = lit(std::to_string(plan.DV), "si64");
      subOutSlotOff = bin("add", bin("mul", iotaOwn64, cDV64, si64BTy),
                          iotaSubAxDV64, subSi64OutTy);
      trueSubO = newVec(true1, subMaskOutTy);
      zeroSubO = newVec(zeroS, subOutTy);
    }
    auto applyScoreAffine2D = [&](TaskTerm sc, TaskTerm tIv) -> TaskTerm {
      if (scaleTerm.v)
        sc = bin("mul", sc, scaleTerm, scoreMatTy);
      if (effBiasSlopeTerm.v) {
        TaskTerm t64 = castOp(tIv, "si64");
        TaskTerm tPosF32 =
            castOp(bin("add", iotaTok64, t64, si64TTy), vecTTy);
        TaskTerm tBias = bin("mul", tPosF32, effBiasSlopeTerm, vecTTy);
        sc = bin("add", sc, qBiasTerm, scoreMatTy);
        sc = bin("sub", sc, tBias, scoreMatTy);
      }
      if (outerScaleTerm.v)
        sc = bin("mul", sc, outerScaleTerm, scoreMatTy);
      return sc;
    };
    auto applyScoreAffine1D = [&](TaskTerm sc1D, TaskTerm tIv) -> TaskTerm {
      if (scaleTerm.v)
        sc1D = bin("mul", sc1D, scaleTerm, vecBTy);
      if (effBiasSlopeTerm.v) {
        TaskTerm tPosS = castOp(castOp(tIv, "si64"), "f32");
        TaskTerm tBiasS = bin("mul", tPosS, effBiasSlopeTerm, "f32");
        sc1D = bin("add", sc1D, qBiasTerm, vecBTy);
        sc1D = bin("sub", sc1D, tBiasS, vecBTy);
      }
      if (outerScaleTerm.v)
        sc1D = bin("mul", sc1D, outerScaleTerm, vecBTy);
      return sc1D;
    };
    auto applyPostSubScale = [&](TaskTerm diff, StringRef ty) -> TaskTerm {
      if (postScaleTerm.v)
        diff = bin("mul", diff, postScaleTerm, ty);
      return diff;
    };
    const bool allowExpFtz = !isRvv && intentdsl::kir::allowsSubnormalFtz(fn);
    TaskTerm ftzMinS = allowExpFtz
                           ? lit(plan.useExp2 ? "-126.0" : "-87.33", "f32")
                           : TaskTerm{};
    auto emitExpWeight2D = [&](TaskTerm diff, TaskTerm tIv, TaskTerm boundVec64,
                               StringRef causalPred) -> TaskTerm {
      if (plan.useExp2 && isRvv)
        diff = bin("mul", diff, ln2S, scoreMatTy);
      TaskTerm keepMask;
      if (allowExpFtz) {
        keepMask = cmp("ge", diff, ftzMinS, maskMatTy);
        diff = bin("max", diff, ftzMinS, scoreMatTy);
      }
      TaskTerm ex = (plan.useExp2 && !isRvv) ? unaryOp("exp2", diff)
                                             : unaryOp("exp", diff);
      if (plan.hasCausal) {
        TaskTerm t64 = castOp(tIv, "si64");
        TaskTerm tVec64 = bin("add", iotaTok64, t64, si64TTy);
        TaskTerm cm = cmp(causalPred, boundVec64, tVec64, maskMatTy);
        keepMask = keepMask.v ? bin("and", cm, keepMask, maskMatTy) : cm;
      }
      return keepMask.v ? sel(keepMask, ex, zeroM, scoreMatTy) : ex;
    };
    auto emitExpWeight1D = [&](TaskTerm diff1D, TaskTerm tIv,
                               TaskTerm boundVec64,
                               StringRef causalPred) -> TaskTerm {
      if (plan.useExp2 && isRvv)
        diff1D = bin("mul", diff1D, ln2S, vecBTy);
      TaskTerm keepMask;
      if (allowExpFtz) {
        keepMask = cmp("ge", diff1D, ftzMinS, maskBTy);
        diff1D = bin("max", diff1D, ftzMinS, vecBTy);
      }
      TaskTerm ex1D = (plan.useExp2 && !isRvv) ? unaryOp("exp2", diff1D)
                                               : unaryOp("exp", diff1D);
      if (plan.hasCausal) {
        TaskTerm t64 = castOp(tIv, "si64");
        TaskTerm cm = cmp(causalPred, boundVec64, t64, maskBTy);
        keepMask = keepMask.v ? bin("and", cm, keepMask, maskBTy) : cm;
      }
      return keepMask.v ? sel(keepMask, ex1D, zeroV, vecBTy) : ex1D;
    };
    auto computeInvVec = [&](TaskTerm mv) -> TaskTerm {
      if (plan.hasDirectRecip)
        return bin("div", oneS, mv, vecBTy);
      std::optional<bool> staticCond;
      if (auto cmpC = plan.invCondVal.getDefiningOp<arith::CmpIOp>()) {
        auto lC = staticIndexVal(fn, cmpC.getLhs());
        auto rC = staticIndexVal(fn, cmpC.getRhs());
        if (lC && rC) {
          if (cmpC.getPredicate() == arith::CmpIPredicate::eq)
            staticCond = (*lC == *rC);
          else if (cmpC.getPredicate() == arith::CmpIPredicate::ne)
            staticCond = (*lC != *rC);
        }
      }
      if (staticCond) {
        bool useZero =
            *staticCond ? plan.invZeroWhenTrue : !plan.invZeroWhenTrue;
        return useZero ? zeroV : bin("div", oneS, mv, vecBTy);
      }
      TaskTerm condTerm = lanes[0].at(plan.invCondVal);
      ArrayAttr rTy = b.getArrayAttr({b.getStringAttr(vecBTy)});
      Operation *invIf =
          makeOp(b, loc, "task_if", {condTerm.v}, {b.getIndexType()},
                 {b.getNamedAttr("res_types", rTy)}, 2);
      {
        Block &tBb = invIf->getRegion(0).emplaceBlock();
        OpBuilder::InsertionGuard tg(b);
        b.setInsertionPointToEnd(&tBb);
        TaskTerm vT = plan.invZeroWhenTrue ? zeroV : bin("div", oneS, mv, vecBTy);
        makeOp(b, loc, "task_yield", {vT.v}, {},
               {b.getNamedAttr("val_types", rTy)});
      }
      {
        Block &eBb = invIf->getRegion(1).emplaceBlock();
        OpBuilder::InsertionGuard eg(b);
        b.setInsertionPointToEnd(&eBb);
        TaskTerm vE = plan.invZeroWhenTrue ? bin("div", oneS, mv, vecBTy) : zeroV;
        makeOp(b, loc, "task_yield", {vE.v}, {},
               {b.getNamedAttr("val_types", rTy)});
      }
      return {invIf->getResult(0), vecBTy};
    };
    auto loC = staticIndexVal(fn, plan.loop1.getLowerBound());
    auto hiMax = maxStaticIndexVal(fn, plan.loop1.getUpperBound());
    bool staticallyFitsWindow =
        plan.hasScoreBuffer &&
        (!plan.windowGuarded ||
         (loC && hiMax && (*hiMax - *loC) <= plan.scoreSlots));
    bool exactOrigTiles =
        isStaticMultipleOf(fn, plan.loop1.getLowerBound(), T) &&
        isStaticMultipleOf(fn, plan.loop1.getUpperBound(), T);
    bool exactTiles =
        exactOrigTiles &&
        (!plan.hasCausal ||
         (plan.causalBoundVal == innerArg && plan.causalInclusive &&
          B % T == 0));

    auto tightenCausalUb = [&](TaskTerm lbTerm, TaskTerm origUbTerm) -> TaskTerm {
      if (!plan.hasCausal)
        return origUbTerm;
      TaskTerm lb64 = castOp(lbTerm, "si64");
      TaskTerm ub64 = castOp(origUbTerm, "si64");
      TaskTerm lim64;
      if (plan.causalBoundVal == innerArg) {
        int64_t delta = (B - 1) + (plan.causalInclusive ? 1 : 0);
        lim64 = bin("add", castOp(lanes[0].at(innerArg), "si64"),
                    lit(std::to_string(delta), "si64"), "si64");
      } else {
        lim64 = castOp(lanes[0].at(plan.causalBoundVal), "si64");
        if (plan.causalInclusive)
          lim64 = bin("add", lim64, lit("1", "si64"), "si64");
      }
      TaskTerm clampedLo = sel(cmp("gt", lim64, lb64, "i1"), lim64, lb64, "si64");
      TaskTerm clampedUb = sel(cmp("lt", clampedLo, ub64, "i1"), clampedLo, ub64, "si64");
      return castOp(clampedUb, "index");
    };

    auto extractScalarLanes = [&](TaskTerm vec, Value ssaVal) {
      for (int64_t l = 0; l < B; ++l) {
        TaskTerm lIdx = idx(l);
        Value ex =
            makeOp(b, loc, "task_extract", {vec.v, lIdx.v}, {b.getIndexType()},
                   {b.getNamedAttr("selector", b.getStringAttr("index")),
                    b.getNamedAttr("in_type", b.getStringAttr(vec.type)),
                    b.getNamedAttr("point_type", b.getStringAttr("index")),
                    b.getNamedAttr("val_type", b.getStringAttr("f32"))})
                ->getResult(0);
        lanes[l][ssaVal] = {ex, "f32"};
      }
    };

    for (Operation &op : srcBb.without_terminator()) {
      if (op.getName().getStringRef() == "cpu.local_buffer" ||
          skipOps.contains(&op))
        continue;

      if (&op == plan.loop1.getOperation()) {
        // Admit 2D Q[B, D] and pre-apply scalar score scale once
        SmallVector<Value, 4> qOtherCoords;
        for (auto [pos, c] :
             llvm::enumerate(plan.qReadOp->getOperands().drop_front()))
          if (int64_t(pos) != plan.qOwnMemAx)
            qOtherCoords.push_back(c);
        TaskTerm qSl =
            slice2D(plan.qMem, qOtherCoords, plan.qOwnMemAx, plan.qVecMemAx);
        TaskTerm c0 = idx(0), c1 = idx(1), cT = idx(T);
        TaskTerm qRaw = subview2D(qSl, lanes[0].at(innerArg).v, c0.v, B, plan.D,
                                  true, qRawMatTy);
        qRawSaved = qRaw;
        qForContract =
            (nativeQK || qElem == "f32") ? qRaw : castOp(qRaw, qF32MatTy);
        scaleTerm = plan.scaleVal ? lanes[0].at(plan.scaleVal) : TaskTerm{};
        outerScaleTerm =
            plan.outerScaleVal ? lanes[0].at(plan.outerScaleVal) : TaskTerm{};
        postScaleTerm =
            plan.postScaleVal ? lanes[0].at(plan.postScaleVal) : TaskTerm{};
        if (plan.biasSlopeVal) {
          TaskTerm rawSlope = lanes[0].at(plan.biasSlopeVal);
          effBiasSlopeTerm = plan.biasNegate
                                 ? rawSlope
                                 : bin("sub", zeroS, rawSlope, "f32");
          if (outerScaleTerm.v) {
            if (scaleTerm.v)
              scaleTerm = bin("mul", scaleTerm, outerScaleTerm, "f32");
            else
              scaleTerm = outerScaleTerm;
            effBiasSlopeTerm =
                bin("mul", effBiasSlopeTerm, outerScaleTerm, "f32");
            outerScaleTerm = TaskTerm{};
          }
          TaskTerm qb64 = castOp(lanes[0].at(innerArg), "si64");
          qPosF32 = castOp(bin("add", iotaOwn64, qb64, si64BTy), vecBTy);
          qBiasTerm = bin("mul", qPosF32, effBiasSlopeTerm, vecBTy);
        }

        SmallVector<Value, 4> kOtherCoords;
        for (auto [pos, c] : llvm::enumerate(plan.kCoords))
          if (int64_t(pos) != plan.kTokCoordPos)
            kOtherCoords.push_back(c);
        TaskTerm kSl2D =
            slice2D(plan.kMem, kOtherCoords, plan.kTokMemAx, plan.kVecMemAx);

        TaskTerm lb = lanes[0].at(plan.loop1.getLowerBound());
        TaskTerm ub =
            tightenCausalUb(lb, lanes[0].at(plan.loop1.getUpperBound()));
        TaskTerm tMain = ub;
        if (!exactTiles) {
          TaskTerm span = bin("sub", ub, lb, "index");
          TaskTerm nTiles = bin("div", span, cT, "index");
          tMain = bin("add", lb, bin("mul", nTiles, cT, "index"), "index");
        }

        TaskTerm boundVec64;
        StringRef causalPred = plan.causalInclusive ? "ge" : "gt";
        if (plan.hasCausal) {
          if (plan.causalBoundVal == innerArg) {
            TaskTerm qb64 = castOp(lanes[0].at(innerArg), "si64");
            boundVec64 = bin("add", iotaOwn64, qb64, si64BTy);
          } else {
            TaskTerm bs64 = castOp(lanes[0].at(plan.causalBoundVal), "si64");
            boundVec64 = newVec(bs64, si64BTy);
          }
        }

        // Pass 1 main loop (step = T)
        Operation *p1Main = makeOp(
            b, loc, "task_for", {lb.v, tMain.v, cT.v, ninfV.v},
            {b.getIndexType()},
            {b.getNamedAttr("iter_types",
                            b.getArrayAttr({b.getStringAttr(vecBTy)}))},
            1);
        {
          Block &fBb = p1Main->getRegion(0).emplaceBlock();
          TaskTerm tIv{fBb.addArgument(b.getIndexType(), loc), "index"};
          TaskTerm pkIter{fBb.addArgument(b.getIndexType(), loc), vecBTy};
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToEnd(&fBb);
          TaskTerm kRaw =
              subview2D(kSl2D, tIv.v, c0.v, T, plan.D, true, kRawMatTy);
          TaskTerm kForC =
              (nativeQK || kElem == "f32") ? kRaw : castOp(kRaw, kF32MatTy);
          TaskTerm sc = outerContractOp(qForContract, kForC, plan.axD,
                                        scoreMatTy, plan.scoreEngine);
          sc = applyScoreAffine2D(sc, tIv);
          if (plan.hasScoreBuffer) {
            TaskTerm offIdx = loC && *loC == 0 ? tIv : bin("sub", tIv, lb, "index");
            TaskTerm off64 = castOp(offIdx, "si64");
            TaskTerm baseOff = bin("mul", off64, cB64, "si64");
            TaskTerm pOff = bin("add", tileSlotOff, baseOff, si64MatTy);
            auto emitStore = [&]() {
              makeOp(b, loc, "task_ptr_store",
                     {tiledScoreBuf.v, pOff.v, trueM.v, sc.v}, {},
                     {b.getNamedAttr(
                          "ptr_type",
                          b.getStringAttr("!weft_kernel.pointer<f32>")),
                      b.getNamedAttr("val_type", b.getStringAttr(scoreMatTy))});
            };
            if (staticallyFitsWindow) {
              emitStore();
            } else {
              TaskTerm inW = cmp("lt", off64,
                                 lit(std::to_string(plan.scoreSlots), "si64"),
                                 "i1");
              Operation *wIf = makeOp(
                  b, loc, "task_if", {inW.v}, {},
                  {b.getNamedAttr("res_types", b.getArrayAttr({}))}, 2);
              Block &wBb = wIf->getRegion(0).emplaceBlock();
              OpBuilder::InsertionGuard wg(b);
              b.setInsertionPointToEnd(&wBb);
              emitStore();
              makeOp(b, loc, "task_yield", {}, {},
                     {b.getNamedAttr("val_types", b.getArrayAttr({}))});
            }
          }
          TaskTerm msc = sc;
          if (plan.hasCausal) {
            TaskTerm t64 = castOp(tIv, "si64");
            TaskTerm tVec64 = bin("add", iotaTok64, t64, si64TTy);
            TaskTerm m = cmp(causalPred, boundVec64, tVec64, maskMatTy);
            msc = sel(m, sc, ninfM, scoreMatTy);
          }
          TaskTerm rmax = reduceOp(msc, "max", 1, vecBTy);
          TaskTerm npk = bin("max", pkIter, rmax, vecBTy);
          makeOp(b, loc, "task_yield", {npk.v}, {},
                 {b.getNamedAttr("val_types",
                                 b.getArrayAttr({b.getStringAttr(vecBTy)}))});
        }
        peakVec = {p1Main->getResult(0), vecBTy};

        if (!exactTiles) {
          TaskTerm hasTail =
              cmp("lt", castOp(tMain, "si64"), castOp(ub, "si64"), "i1");
          ArrayAttr pkResTy = b.getArrayAttr({b.getStringAttr(vecBTy)});
          Operation *tailIf =
              makeOp(b, loc, "task_if", {hasTail.v}, {b.getIndexType()},
                     {b.getNamedAttr("res_types", pkResTy)}, 2);
          {
            Block &tIfBb = tailIf->getRegion(0).emplaceBlock();
            OpBuilder::InsertionGuard ig(b);
            b.setInsertionPointToEnd(&tIfBb);
            TaskTerm qTailF32 =
                (qElem == "f32") ? qRawSaved : castOp(qRawSaved, qF32MatTy);
            Operation *p1Tail = makeOp(
                b, loc, "task_for", {tMain.v, ub.v, c1.v, peakVec.v},
                {b.getIndexType()},
                {b.getNamedAttr("iter_types", pkResTy)}, 1);
            {
              Block &fBb = p1Tail->getRegion(0).emplaceBlock();
              TaskTerm tIv{fBb.addArgument(b.getIndexType(), loc), "index"};
              TaskTerm pkIter{fBb.addArgument(b.getIndexType(), loc), vecBTy};
              OpBuilder::InsertionGuard g(b);
              b.setInsertionPointToEnd(&fBb);
              TaskTerm kRaw1D =
                  slice1DAtTok(plan.kMem, plan.kCoords, plan.kTokCoordPos,
                               tIv.v, plan.kVecMemAx, kRaw1DTy);
              TaskTerm kF1D =
                  (kElem == "f32") ? kRaw1D : castOp(kRaw1D, kF321DTy);
              TaskTerm prod = bin("mul", qTailF32, kF1D, qF32MatTy);
              TaskTerm sc1D = reduceOp(prod, "add", 1, vecBTy);
              sc1D = applyScoreAffine1D(sc1D, tIv);
              if (plan.hasScoreBuffer) {
                TaskTerm offIdx =
                    loC && *loC == 0 ? tIv : bin("sub", tIv, lb, "index");
                TaskTerm off64 = castOp(offIdx, "si64");
                TaskTerm baseOff = bin("mul", off64, cB64, "si64");
                TaskTerm pOff1D = bin("add", qSlotOff, baseOff, si64BTy);
                auto emitStore1D = [&]() {
                  makeOp(b, loc, "task_ptr_store",
                         {tiledScoreBuf.v, pOff1D.v, trueB.v, sc1D.v}, {},
                         {b.getNamedAttr(
                              "ptr_type",
                              b.getStringAttr("!weft_kernel.pointer<f32>")),
                          b.getNamedAttr("val_type", b.getStringAttr(vecBTy))});
                };
                if (staticallyFitsWindow) {
                  emitStore1D();
                } else {
                  TaskTerm inW = cmp("lt", off64,
                                     lit(std::to_string(plan.scoreSlots), "si64"),
                                     "i1");
                  Operation *wIf = makeOp(
                      b, loc, "task_if", {inW.v}, {},
                      {b.getNamedAttr("res_types", b.getArrayAttr({}))}, 2);
                  Block &wBb = wIf->getRegion(0).emplaceBlock();
                  OpBuilder::InsertionGuard wg(b);
                  b.setInsertionPointToEnd(&wBb);
                  emitStore1D();
                  makeOp(b, loc, "task_yield", {}, {},
                         {b.getNamedAttr("val_types", b.getArrayAttr({}))});
                }
              }
              TaskTerm msc1D = sc1D;
              if (plan.hasCausal) {
                TaskTerm t64 = castOp(tIv, "si64");
                TaskTerm m1D = cmp(causalPred, boundVec64, t64, maskBTy);
                msc1D = sel(m1D, sc1D, ninfV, vecBTy);
              }
              TaskTerm npk = bin("max", pkIter, msc1D, vecBTy);
              makeOp(b, loc, "task_yield", {npk.v}, {},
                     {b.getNamedAttr("val_types", pkResTy)});
            }
            makeOp(b, loc, "task_yield", {p1Tail->getResult(0)}, {},
                   {b.getNamedAttr("val_types", pkResTy)});
          }
          {
            Block &eIfBb = tailIf->getRegion(1).emplaceBlock();
            OpBuilder::InsertionGuard eg(b);
            b.setInsertionPointToEnd(&eIfBb);
            makeOp(b, loc, "task_yield", {peakVec.v}, {},
                   {b.getNamedAttr("val_types", pkResTy)});
          }
          peakVec = {tailIf->getResult(0), vecBTy};
        }
        if (llvm::any_of(plan.loop1.getResult(0).getUsers(), [&](Operation *u) {
              return !plan.stage2Op->isAncestor(u);
            }))
          extractScalarLanes(peakVec, plan.loop1.getResult(0));
        continue;
      }

      if (&op == plan.stage2Op) {
        TaskTerm c0 = idx(0), c1 = idx(1), cT = idx(T);
        TaskTerm lb = lanes[0].at(plan.loop1.getLowerBound());
        TaskTerm ub =
            tightenCausalUb(lb, lanes[0].at(plan.loop1.getUpperBound()));
        TaskTerm span = (!exactTiles || (plan.hasScoreBuffer && !staticallyFitsWindow))
                            ? bin("sub", ub, lb, "index")
                            : TaskTerm{};
        TaskTerm tMain = ub;
        if (!exactTiles) {
          TaskTerm nTiles = bin("div", span, cT, "index");
          tMain = bin("add", lb, bin("mul", nTiles, cT, "index"), "index");
        }

        SmallVector<Value, 4> kOtherCoords, vOtherCoords;
        for (auto [pos, c] : llvm::enumerate(plan.kCoords))
          if (int64_t(pos) != plan.kTokCoordPos)
            kOtherCoords.push_back(c);
        for (auto [pos, c] : llvm::enumerate(plan.vCoords))
          if (int64_t(pos) != plan.vTokCoordPos)
            vOtherCoords.push_back(c);
        TaskTerm vSl2D =
            slice2D(plan.vMem, vOtherCoords, plan.vTokMemAx, plan.vVecMemAx);

        TaskTerm boundVec64;
        StringRef causalPred = plan.causalInclusive ? "ge" : "gt";
        if (plan.hasCausal) {
          if (plan.causalBoundVal == innerArg) {
            TaskTerm qb64 = castOp(lanes[0].at(innerArg), "si64");
            boundVec64 = bin("add", iotaOwn64, qb64, si64BTy);
          } else {
            TaskTerm bs64 = castOp(lanes[0].at(plan.causalBoundVal), "si64");
            boundVec64 = newVec(bs64, si64BTy);
          }
        }

        std::string narrowScoreMatTy, narrowVecBTy;
        if (plan.weightRoundType) {
          std::string rElem = scalarType(*plan.weightRoundType);
          narrowScoreMatTy = "!weft_kernel.value<" + rElem + ",[" +
                             std::to_string(B) + "," + std::to_string(T) +
                             "],[" + std::to_string(plan.axOwn) + "," +
                             std::to_string(plan.axTok) + "]>";
          narrowVecBTy = "!weft_kernel.value<" + rElem + ",[" +
                         std::to_string(B) + "],[" +
                         std::to_string(plan.axOwn) + "]>";
        }

        auto emitPass2 =
            [&](bool fromBuffer) -> std::pair<TaskTerm, TaskTerm> {
          if (fromBuffer && useTiledOut) {
            ArrayAttr wIterTys = b.getArrayAttr({b.getStringAttr(vecBTy)});
            Operation *p2WMain = makeOp(
                b, loc, "task_for", {lb.v, tMain.v, cT.v, zeroV.v},
                {b.getIndexType()},
                {b.getNamedAttr("iter_types", wIterTys)}, 1);
            {
              Block &fBb = p2WMain->getRegion(0).emplaceBlock();
              TaskTerm tIv{fBb.addArgument(b.getIndexType(), loc), "index"};
              TaskTerm msIter{fBb.addArgument(b.getIndexType(), loc), vecBTy};
              OpBuilder::InsertionGuard g(b);
              b.setInsertionPointToEnd(&fBb);
              TaskTerm offIdx =
                  loC && *loC == 0 ? tIv : bin("sub", tIv, lb, "index");
              TaskTerm off64 = castOp(offIdx, "si64");
              TaskTerm baseOff = bin("mul", off64, cB64, "si64");
              TaskTerm pOff = bin("add", tileSlotOff, baseOff, si64MatTy);
              Value ld = makeOp(
                             b, loc, "task_ptr_load",
                             {tiledScoreBuf.v, pOff.v, trueM.v, zeroM.v},
                             {b.getIndexType()},
                             {b.getNamedAttr(
                                  "ptr_type",
                                  b.getStringAttr("!weft_kernel.pointer<f32>")),
                              b.getNamedAttr("val_type",
                                             b.getStringAttr(scoreMatTy))})
                             ->getResult(0);
              TaskTerm sc{ld, scoreMatTy};
              TaskTerm diff =
                  applyPostSubScale(bin("sub", sc, peakVec, scoreMatTy),
                                    scoreMatTy);
              TaskTerm wt = emitExpWeight2D(diff, tIv, boundVec64, causalPred);
              TaskTerm rsum = reduceOp(wt, "add", 1, vecBTy);
              TaskTerm nms = bin("add", msIter, rsum, vecBTy);
              TaskTerm wF = plan.weightRoundType
                                ? castOp(castOp(wt, narrowScoreMatTy), scoreMatTy)
                                : wt;
              makeOp(b, loc, "task_ptr_store",
                     {tiledScoreBuf.v, pOff.v, trueM.v, wF.v}, {},
                     {b.getNamedAttr(
                          "ptr_type",
                          b.getStringAttr("!weft_kernel.pointer<f32>")),
                      b.getNamedAttr("val_type", b.getStringAttr(scoreMatTy))});
              makeOp(b, loc, "task_yield", {nms.v}, {},
                     {b.getNamedAttr("val_types", wIterTys)});
            }
            TaskTerm msFinal{p2WMain->getResult(0), vecBTy};
            if (!exactTiles) {
              Operation *p2WTail = makeOp(
                  b, loc, "task_for", {tMain.v, ub.v, c1.v, msFinal.v},
                  {b.getIndexType()},
                  {b.getNamedAttr("iter_types", wIterTys)}, 1);
              {
                Block &fBb = p2WTail->getRegion(0).emplaceBlock();
                TaskTerm tIv{fBb.addArgument(b.getIndexType(), loc), "index"};
                TaskTerm msIter{fBb.addArgument(b.getIndexType(), loc), vecBTy};
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPointToEnd(&fBb);
                TaskTerm offIdx =
                    loC && *loC == 0 ? tIv : bin("sub", tIv, lb, "index");
                TaskTerm off64 = castOp(offIdx, "si64");
                TaskTerm baseOff = bin("mul", off64, cB64, "si64");
                TaskTerm pOff1D = bin("add", qSlotOff, baseOff, si64BTy);
                Value ld =
                    makeOp(b, loc, "task_ptr_load",
                           {tiledScoreBuf.v, pOff1D.v, trueB.v, zeroV.v},
                           {b.getIndexType()},
                           {b.getNamedAttr(
                                "ptr_type",
                                b.getStringAttr("!weft_kernel.pointer<f32>")),
                            b.getNamedAttr("val_type", b.getStringAttr(vecBTy))})
                        ->getResult(0);
                TaskTerm sc1D{ld, vecBTy};
                TaskTerm diff1D =
                    applyPostSubScale(bin("sub", sc1D, peakVec, vecBTy),
                                      vecBTy);
                TaskTerm wt1D =
                    emitExpWeight1D(diff1D, tIv, boundVec64, causalPred);
                TaskTerm nms = bin("add", msIter, wt1D, vecBTy);
                TaskTerm wF1D = plan.weightRoundType
                                    ? castOp(castOp(wt1D, narrowVecBTy), vecBTy)
                                    : wt1D;
                makeOp(b, loc, "task_ptr_store",
                       {tiledScoreBuf.v, pOff1D.v, trueB.v, wF1D.v}, {},
                       {b.getNamedAttr(
                            "ptr_type",
                            b.getStringAttr("!weft_kernel.pointer<f32>")),
                        b.getNamedAttr("val_type", b.getStringAttr(vecBTy))});
                makeOp(b, loc, "task_yield", {nms.v}, {},
                       {b.getNamedAttr("val_types", wIterTys)});
              }
              msFinal = {p2WTail->getResult(0), vecBTy};
            }

            ArrayAttr subIterTys = b.getArrayAttr({b.getStringAttr(subOutTy)});
            TaskTerm c0Idx = idx(0), cDvIdx = idx(plan.DV),
                     cSubDvIdx = idx(subDV);
            bool directCommitHere = staticallyFitsWindow;
            TaskTerm invVec =
                (directCommitHere && plan.yMulOp) ? computeInvVec(msFinal)
                                                  : TaskTerm{};
            TaskTerm ySl2D;
            std::string commitSubSlTy;
            if (directCommitHere) {
              SmallVector<Value, 4> yOtherCoords;
              for (auto [pos, c] :
                   llvm::enumerate(plan.yWriteOp->getOperands().drop_front(2)))
                if (int64_t(pos) != plan.yOwnMemAx)
                  yOtherCoords.push_back(c);
              ySl2D = slice2D(plan.yMem, yOtherCoords, plan.yOwnMemAx,
                              plan.yVecMemAx);
              commitSubSlTy = ySl2D.type;
              if (auto brk = commitSubSlTy.rfind(">,[");
                  brk != std::string::npos) {
                auto cls = commitSubSlTy.find(']', brk + 2);
                commitSubSlTy.replace(brk + 2, cls - (brk + 2) + 1,
                                      "[" + std::to_string(B) + "," +
                                          std::to_string(subDV) + "]");
              }
            }
            Operation *dvFor = makeOp(
                b, loc, "task_for", {c0Idx.v, cDvIdx.v, cSubDvIdx.v}, {},
                {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
            {
              Block &dvBb = dvFor->getRegion(0).emplaceBlock();
              TaskTerm dv0Idx{dvBb.addArgument(b.getIndexType(), loc), "index"};
              OpBuilder::InsertionGuard dvg(b);
              b.setInsertionPointToEnd(&dvBb);
              Operation *p2SubMain = makeOp(
                  b, loc, "task_for", {lb.v, tMain.v, cT.v, zeroSubO.v},
                  {b.getIndexType()},
                  {b.getNamedAttr("iter_types", subIterTys)}, 1);
              {
                Block &fBb = p2SubMain->getRegion(0).emplaceBlock();
                TaskTerm tIv{fBb.addArgument(b.getIndexType(), loc), "index"};
                TaskTerm subAcc{fBb.addArgument(b.getIndexType(), loc),
                                subOutTy};
                OpBuilder::InsertionGuard g(b);
                b.setInsertionPointToEnd(&fBb);
                TaskTerm offIdx =
                    loC && *loC == 0 ? tIv : bin("sub", tIv, lb, "index");
                TaskTerm off64 = castOp(offIdx, "si64");
                TaskTerm baseOff = bin("mul", off64, cB64, "si64");
                TaskTerm wOff = bin("add", tileSlotOff, baseOff, si64MatTy);
                Value wLd =
                    makeOp(b, loc, "task_ptr_load",
                           {tiledScoreBuf.v, wOff.v, trueM.v, zeroM.v},
                           {b.getIndexType()},
                           {b.getNamedAttr(
                                "ptr_type",
                                b.getStringAttr("!weft_kernel.pointer<f32>")),
                            b.getNamedAttr("val_type",
                                           b.getStringAttr(scoreMatTy))})
                        ->getResult(0);
                TaskTerm wFull{wLd, scoreMatTy};
                TaskTerm vRawSub = subview2D(vSl2D, tIv.v, dv0Idx.v, T, subDV,
                                             true, vSubRawMatTy);
                TaskTerm vForC = (nativeV || vElem == "f32")
                                     ? vRawSub
                                     : castOp(vRawSub, vSubF32MatTy);
                TaskTerm dSub =
                    outerContractOp(wFull, vForC, plan.axTok, subOutTy);
                TaskTerm nSub = bin("add", subAcc, dSub, subOutTy);
                makeOp(b, loc, "task_yield", {nSub.v}, {},
                       {b.getNamedAttr("val_types", subIterTys)});
              }
              TaskTerm subRes{p2SubMain->getResult(0), subOutTy};
              if (!exactTiles) {
                Operation *p2SubTail = makeOp(
                    b, loc, "task_for", {tMain.v, ub.v, c1.v, subRes.v},
                    {b.getIndexType()},
                    {b.getNamedAttr("iter_types", subIterTys)}, 1);
                {
                  Block &fBb = p2SubTail->getRegion(0).emplaceBlock();
                  TaskTerm tIv{fBb.addArgument(b.getIndexType(), loc), "index"};
                  TaskTerm subAcc{fBb.addArgument(b.getIndexType(), loc),
                                  subOutTy};
                  OpBuilder::InsertionGuard g(b);
                  b.setInsertionPointToEnd(&fBb);
                  TaskTerm offIdx =
                      loC && *loC == 0 ? tIv : bin("sub", tIv, lb, "index");
                  TaskTerm off64 = castOp(offIdx, "si64");
                  TaskTerm baseOff = bin("mul", off64, cB64, "si64");
                  TaskTerm pOff1D = bin("add", qSlotOff, baseOff, si64BTy);
                  Value wLd =
                      makeOp(b, loc, "task_ptr_load",
                             {tiledScoreBuf.v, pOff1D.v, trueB.v, zeroV.v},
                             {b.getIndexType()},
                             {b.getNamedAttr(
                                  "ptr_type",
                                  b.getStringAttr("!weft_kernel.pointer<f32>")),
                              b.getNamedAttr("val_type",
                                             b.getStringAttr(vecBTy))})
                          ->getResult(0);
                  TaskTerm wF1D{wLd, vecBTy};
                  // 1D slice at token tIv, then subview [subDV] at dv0Idx
                  std::string vSl1DTy =
                      "!weft_kernel.slice<!weft_kernel.encoding<\"" +
                      carrierType(
                          cast<MemRefType>(plan.vMem.getType()).getElementType()) +
                      "\",\"dense\",\"dense." +
                      carrierType(
                          cast<MemRefType>(plan.vMem.getType()).getElementType()) +
                      "\",[]>,[" + std::to_string(plan.DV) + "],[" +
                      std::to_string(plan.axDV) + "]>";
                  SmallVector<Value, 6> slOps{lanes[0].at(plan.vMem).v};
                  SmallVector<Attribute, 6> sels;
                  for (auto [idxC, cVal] : llvm::enumerate(plan.vCoords)) {
                    slOps.push_back(int64_t(idxC) == plan.vTokCoordPos
                                        ? tIv.v
                                        : lanes[0].at(cVal).v);
                    sels.push_back(b.getStringAttr("index"));
                  }
                  sels.insert(sels.begin() + plan.vVecMemAx,
                              b.getStringAttr("all"));
                  Value vSl1D =
                      makeOp(b, loc, "task_slice", slOps, {b.getIndexType()},
                             {b.getNamedAttr(
                                  "view_type",
                                  b.getStringAttr(lanes[0].at(plan.vMem).type)),
                              b.getNamedAttr("selectors", b.getArrayAttr(sels)),
                              b.getNamedAttr("slice_type",
                                             b.getStringAttr(vSl1DTy)),
                              b.getNamedAttr("val_type",
                                             b.getStringAttr(vSl1DTy)),
                              b.getNamedAttr("admit", b.getBoolAttr(false))})
                          ->getResult(0);
                  Value vRawSub1D =
                      makeOp(b, loc, "task_subview", {vSl1D, dv0Idx.v},
                             {b.getIndexType()},
                             {b.getNamedAttr("static_extent",
                                             b.getI64IntegerAttr(subDV)),
                              b.getNamedAttr("slice_type",
                                             b.getStringAttr(vSl1DTy)),
                              b.getNamedAttr("val_type",
                                             b.getStringAttr(vSubRaw1DTy)),
                              b.getNamedAttr("admit", b.getBoolAttr(true))})
                          ->getResult(0);
                  TaskTerm vFSub1D =
                      (vElem == "f32")
                          ? TaskTerm{vRawSub1D, vSubRaw1DTy}
                          : castOp({vRawSub1D, vSubRaw1DTy}, vSubF321DTy);
                  TaskTerm dSub = bin("mul", wF1D, vFSub1D, subOutTy);
                  TaskTerm nSub = bin("add", subAcc, dSub, subOutTy);
                  makeOp(b, loc, "task_yield", {nSub.v}, {},
                         {b.getNamedAttr("val_types", subIterTys)});
                }
                subRes = {p2SubTail->getResult(0), subOutTy};
              }
              if (directCommitHere) {
                TaskTerm scaledSub =
                    plan.yMulOp ? bin("mul", subRes, invVec, subOutTy) : subRes;
                TaskTerm finalSub = (yElem == "f32")
                                        ? scaledSub
                                        : castOp(scaledSub, ySubMatTy);
                TaskTerm ySubTile = subview2D(ySl2D, lanes[0].at(innerArg).v,
                                              dv0Idx.v, B, subDV, false,
                                              ySl2D.type);
                makeOp(b, loc, "task_commit", {finalSub.v, ySubTile.v}, {},
                       {b.getNamedAttr("val_type",
                                       b.getStringAttr(finalSub.type)),
                        b.getNamedAttr("slice_type",
                                       b.getStringAttr(commitSubSlTy))});
              } else {
                TaskTerm storeOff = bin("add", subOutSlotOff,
                                        castOp(dv0Idx, "si64"), subSi64OutTy);
                makeOp(b, loc, "task_ptr_store",
                       {tiledOutBuf.v, storeOff.v, trueSubO.v, subRes.v}, {},
                       {b.getNamedAttr(
                            "ptr_type",
                            b.getStringAttr("!weft_kernel.pointer<f32>")),
                        b.getNamedAttr("val_type", b.getStringAttr(subOutTy))});
              }
              makeOp(b, loc, "task_yield", {}, {},
                     {b.getNamedAttr("val_types", b.getArrayAttr({}))});
            }
            if (directCommitHere)
              directSubCommitted = true;
            else
              outInTiledBuf = true;
            return {msFinal, TaskTerm{}};
          }

          if (!fromBuffer && useTiledOut) {
            TaskTerm kSl2D = slice2D(plan.kMem, kOtherCoords, plan.kTokMemAx,
                                     plan.kVecMemAx);
            TaskTerm c0Idx = idx(0), cDvIdx = idx(plan.DV),
                     cSubDvIdx = idx(subDV);
            Operation *zeroDvFor = makeOp(
                b, loc, "task_for", {c0Idx.v, cDvIdx.v, cSubDvIdx.v}, {},
                {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
            {
              Block &zBb = zeroDvFor->getRegion(0).emplaceBlock();
              TaskTerm dv0Idx{zBb.addArgument(b.getIndexType(), loc), "index"};
              OpBuilder::InsertionGuard zg(b);
              b.setInsertionPointToEnd(&zBb);
              TaskTerm storeOff = bin("add", subOutSlotOff,
                                      castOp(dv0Idx, "si64"), subSi64OutTy);
              makeOp(b, loc, "task_ptr_store",
                     {tiledOutBuf.v, storeOff.v, trueSubO.v, zeroSubO.v}, {},
                     {b.getNamedAttr(
                          "ptr_type",
                          b.getStringAttr("!weft_kernel.pointer<f32>")),
                      b.getNamedAttr("val_type", b.getStringAttr(subOutTy))});
              makeOp(b, loc, "task_yield", {}, {},
                     {b.getNamedAttr("val_types", b.getArrayAttr({}))});
            }
            ArrayAttr msIterTys = b.getArrayAttr({b.getStringAttr(vecBTy)});
            Operation *p2Main = makeOp(
                b, loc, "task_for", {lb.v, tMain.v, cT.v, zeroV.v},
                {b.getIndexType()},
                {b.getNamedAttr("iter_types", msIterTys)}, 1);
            {
              Block &fBb = p2Main->getRegion(0).emplaceBlock();
              TaskTerm tIv{fBb.addArgument(b.getIndexType(), loc), "index"};
              TaskTerm msIter{fBb.addArgument(b.getIndexType(), loc), vecBTy};
              OpBuilder::InsertionGuard g(b);
              b.setInsertionPointToEnd(&fBb);
              TaskTerm kRaw =
                  subview2D(kSl2D, tIv.v, c0.v, T, plan.D, true, kRawMatTy);
              TaskTerm kForC =
                  (nativeQK || kElem == "f32") ? kRaw : castOp(kRaw, kF32MatTy);
              TaskTerm sc = outerContractOp(qForContract, kForC, plan.axD,
                                            scoreMatTy, plan.scoreEngine);
              sc = applyScoreAffine2D(sc, tIv);
              TaskTerm diff =
                  applyPostSubScale(bin("sub", sc, peakVec, scoreMatTy),
                                    scoreMatTy);
              TaskTerm wt = emitExpWeight2D(diff, tIv, boundVec64, causalPred);
              TaskTerm rsum = reduceOp(wt, "add", 1, vecBTy);
              TaskTerm nms = bin("add", msIter, rsum, vecBTy);
              bool passNarrowW =
                  !isRvv && plan.weightRoundType &&
                  (plan.weightRoundType->isBF16() ||
                   plan.weightRoundType->isF16());
              TaskTerm wForPV =
                  passNarrowW
                      ? castOp(wt, narrowScoreMatTy)
                      : (plan.weightRoundType
                             ? castOp(castOp(wt, narrowScoreMatTy), scoreMatTy)
                             : wt);
              Operation *dvFor = makeOp(
                  b, loc, "task_for", {c0Idx.v, cDvIdx.v, cSubDvIdx.v}, {},
                  {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
              {
                Block &dvBb = dvFor->getRegion(0).emplaceBlock();
                TaskTerm dv0Idx{dvBb.addArgument(b.getIndexType(), loc),
                                "index"};
                OpBuilder::InsertionGuard dvg(b);
                b.setInsertionPointToEnd(&dvBb);
                TaskTerm slotOff = bin("add", subOutSlotOff,
                                       castOp(dv0Idx, "si64"), subSi64OutTy);
                Value accLd =
                    makeOp(b, loc, "task_ptr_load",
                           {tiledOutBuf.v, slotOff.v, trueSubO.v, zeroSubO.v},
                           {b.getIndexType()},
                           {b.getNamedAttr(
                                "ptr_type",
                                b.getStringAttr("!weft_kernel.pointer<f32>")),
                            b.getNamedAttr("val_type",
                                           b.getStringAttr(subOutTy))})
                        ->getResult(0);
                TaskTerm vRawSub = subview2D(vSl2D, tIv.v, dv0Idx.v, T, subDV,
                                             true, vSubRawMatTy);
                TaskTerm vForC = (nativeV || vElem == "f32")
                                     ? vRawSub
                                     : castOp(vRawSub, vSubF32MatTy);
                TaskTerm dSub =
                    outerContractOp(wForPV, vForC, plan.axTok, subOutTy,
                                    plan.valueEngine);
                TaskTerm nSub = bin("add", {accLd, subOutTy}, dSub, subOutTy);
                makeOp(b, loc, "task_ptr_store",
                       {tiledOutBuf.v, slotOff.v, trueSubO.v, nSub.v}, {},
                       {b.getNamedAttr(
                            "ptr_type",
                            b.getStringAttr("!weft_kernel.pointer<f32>")),
                        b.getNamedAttr("val_type", b.getStringAttr(subOutTy))});
                makeOp(b, loc, "task_yield", {}, {},
                       {b.getNamedAttr("val_types", b.getArrayAttr({}))});
              }
              makeOp(b, loc, "task_yield", {nms.v}, {},
                     {b.getNamedAttr("val_types", msIterTys)});
            }
            TaskTerm msFinal{p2Main->getResult(0), vecBTy};
            if (!exactTiles) {
              TaskTerm hasTail =
                  cmp("lt", castOp(tMain, "si64"), castOp(ub, "si64"), "i1");
              Operation *tailIf =
                  makeOp(b, loc, "task_if", {hasTail.v}, {b.getIndexType()},
                         {b.getNamedAttr("res_types", msIterTys)}, 2);
              {
                Block &tIfBb = tailIf->getRegion(0).emplaceBlock();
                OpBuilder::InsertionGuard ig(b);
                b.setInsertionPointToEnd(&tIfBb);
                TaskTerm qTailF32 = (qElem == "f32")
                                        ? qRawSaved
                                        : castOp(qRawSaved, qF32MatTy);
                Operation *p2Tail = makeOp(
                    b, loc, "task_for", {tMain.v, ub.v, c1.v, msFinal.v},
                    {b.getIndexType()},
                    {b.getNamedAttr("iter_types", msIterTys)}, 1);
                {
                  Block &fBb = p2Tail->getRegion(0).emplaceBlock();
                  TaskTerm tIv{fBb.addArgument(b.getIndexType(), loc), "index"};
                  TaskTerm msIter{fBb.addArgument(b.getIndexType(), loc), vecBTy};
                  OpBuilder::InsertionGuard g(b);
                  b.setInsertionPointToEnd(&fBb);
                  TaskTerm kRaw1D =
                      slice1DAtTok(plan.kMem, plan.kCoords, plan.kTokCoordPos,
                                   tIv.v, plan.kVecMemAx, kRaw1DTy);
                  TaskTerm kF1D =
                      (kElem == "f32") ? kRaw1D : castOp(kRaw1D, kF321DTy);
                  TaskTerm prod = bin("mul", qTailF32, kF1D, qF32MatTy);
                  TaskTerm sc1D =
                      applyScoreAffine1D(reduceOp(prod, "add", 1, vecBTy), tIv);
                  TaskTerm diff1D =
                      applyPostSubScale(bin("sub", sc1D, peakVec, vecBTy),
                                        vecBTy);
                  TaskTerm wt1D =
                      emitExpWeight1D(diff1D, tIv, boundVec64, causalPred);
                  TaskTerm nms = bin("add", msIter, wt1D, vecBTy);
                  TaskTerm wF1D = plan.weightRoundType
                                      ? castOp(castOp(wt1D, narrowVecBTy), vecBTy)
                                      : wt1D;
                  std::string vSl1DTy =
                      "!weft_kernel.slice<!weft_kernel.encoding<\"" +
                      carrierType(
                          cast<MemRefType>(plan.vMem.getType()).getElementType()) +
                      "\",\"dense\",\"dense." +
                      carrierType(
                          cast<MemRefType>(plan.vMem.getType()).getElementType()) +
                      "\",[]>,[" + std::to_string(plan.DV) + "],[" +
                      std::to_string(plan.axDV) + "]>";
                  SmallVector<Value, 6> slOps{lanes[0].at(plan.vMem).v};
                  SmallVector<Attribute, 6> sels;
                  for (auto [idxC, cVal] : llvm::enumerate(plan.vCoords)) {
                    slOps.push_back(int64_t(idxC) == plan.vTokCoordPos
                                        ? tIv.v
                                        : lanes[0].at(cVal).v);
                    sels.push_back(b.getStringAttr("index"));
                  }
                  sels.insert(sels.begin() + plan.vVecMemAx,
                              b.getStringAttr("all"));
                  Value vSl1D =
                      makeOp(b, loc, "task_slice", slOps, {b.getIndexType()},
                             {b.getNamedAttr(
                                  "view_type",
                                  b.getStringAttr(lanes[0].at(plan.vMem).type)),
                              b.getNamedAttr("selectors", b.getArrayAttr(sels)),
                              b.getNamedAttr("slice_type",
                                             b.getStringAttr(vSl1DTy)),
                              b.getNamedAttr("val_type",
                                             b.getStringAttr(vSl1DTy)),
                              b.getNamedAttr("admit", b.getBoolAttr(false))})
                          ->getResult(0);
                  Operation *dvFor = makeOp(
                      b, loc, "task_for", {c0Idx.v, cDvIdx.v, cSubDvIdx.v}, {},
                      {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
                  {
                    Block &dvBb = dvFor->getRegion(0).emplaceBlock();
                    TaskTerm dv0Idx{dvBb.addArgument(b.getIndexType(), loc),
                                    "index"};
                    OpBuilder::InsertionGuard dvg(b);
                    b.setInsertionPointToEnd(&dvBb);
                    TaskTerm slotOff = bin("add", subOutSlotOff,
                                           castOp(dv0Idx, "si64"), subSi64OutTy);
                    Value accLd =
                        makeOp(b, loc, "task_ptr_load",
                               {tiledOutBuf.v, slotOff.v, trueSubO.v, zeroSubO.v},
                               {b.getIndexType()},
                               {b.getNamedAttr(
                                    "ptr_type",
                                    b.getStringAttr("!weft_kernel.pointer<f32>")),
                                b.getNamedAttr("val_type",
                                               b.getStringAttr(subOutTy))})
                            ->getResult(0);
                    Value vRawSub1D =
                        makeOp(b, loc, "task_subview", {vSl1D, dv0Idx.v},
                               {b.getIndexType()},
                               {b.getNamedAttr("static_extent",
                                               b.getI64IntegerAttr(subDV)),
                                b.getNamedAttr("slice_type",
                                               b.getStringAttr(vSl1DTy)),
                                b.getNamedAttr("val_type",
                                               b.getStringAttr(vSubRaw1DTy)),
                                b.getNamedAttr("admit", b.getBoolAttr(true))})
                            ->getResult(0);
                    TaskTerm vFSub1D =
                        (vElem == "f32")
                            ? TaskTerm{vRawSub1D, vSubRaw1DTy}
                            : castOp({vRawSub1D, vSubRaw1DTy}, vSubF321DTy);
                    TaskTerm dSub = bin("mul", wF1D, vFSub1D, subOutTy);
                    TaskTerm nSub =
                        bin("add", {accLd, subOutTy}, dSub, subOutTy);
                    makeOp(b, loc, "task_ptr_store",
                           {tiledOutBuf.v, slotOff.v, trueSubO.v, nSub.v}, {},
                           {b.getNamedAttr(
                                "ptr_type",
                                b.getStringAttr("!weft_kernel.pointer<f32>")),
                            b.getNamedAttr("val_type",
                                           b.getStringAttr(subOutTy))});
                    makeOp(b, loc, "task_yield", {}, {},
                           {b.getNamedAttr("val_types", b.getArrayAttr({}))});
                  }
                  makeOp(b, loc, "task_yield", {nms.v}, {},
                         {b.getNamedAttr("val_types", msIterTys)});
                }
                makeOp(b, loc, "task_yield", {p2Tail->getResult(0)}, {},
                       {b.getNamedAttr("val_types", msIterTys)});
              }
              {
                Block &eIfBb = tailIf->getRegion(1).emplaceBlock();
                OpBuilder::InsertionGuard eg(b);
                b.setInsertionPointToEnd(&eIfBb);
                makeOp(b, loc, "task_yield", {msFinal.v}, {},
                       {b.getNamedAttr("val_types", msIterTys)});
              }
              msFinal = {tailIf->getResult(0), vecBTy};
            }
            outInTiledBuf = true;
            return {msFinal, TaskTerm{}};
          }

          TaskTerm kSl2D =
              fromBuffer ? TaskTerm{}
                         : slice2D(plan.kMem, kOtherCoords, plan.kTokMemAx,
                                   plan.kVecMemAx);
          ArrayAttr p2IterTys = b.getArrayAttr(
              {b.getStringAttr(vecBTy), b.getStringAttr(outMatTy)});
          Operation *p2Main = makeOp(
              b, loc, "task_for",
              {lb.v, tMain.v, cT.v, zeroV.v, zeroO.v},
              {b.getIndexType(), b.getIndexType()},
              {b.getNamedAttr("iter_types", p2IterTys)}, 1);
          {
            Block &fBb = p2Main->getRegion(0).emplaceBlock();
            TaskTerm tIv{fBb.addArgument(b.getIndexType(), loc), "index"};
            TaskTerm msIter{fBb.addArgument(b.getIndexType(), loc), vecBTy};
            TaskTerm accIter{fBb.addArgument(b.getIndexType(), loc), outMatTy};
            OpBuilder::InsertionGuard g(b);
            b.setInsertionPointToEnd(&fBb);
            TaskTerm sc;
            if (fromBuffer) {
              TaskTerm offIdx =
                  loC && *loC == 0 ? tIv : bin("sub", tIv, lb, "index");
              TaskTerm off64 = castOp(offIdx, "si64");
              TaskTerm baseOff = bin("mul", off64, cB64, "si64");
              TaskTerm pOff = bin("add", tileSlotOff, baseOff, si64MatTy);
              Value ld = makeOp(
                             b, loc, "task_ptr_load",
                             {tiledScoreBuf.v, pOff.v, trueM.v, zeroM.v},
                             {b.getIndexType()},
                             {b.getNamedAttr(
                                  "ptr_type",
                                  b.getStringAttr("!weft_kernel.pointer<f32>")),
                              b.getNamedAttr("val_type",
                                             b.getStringAttr(scoreMatTy))})
                             ->getResult(0);
              sc = {ld, scoreMatTy};
            } else {
              TaskTerm kRaw =
                  subview2D(kSl2D, tIv.v, c0.v, T, plan.D, true, kRawMatTy);
              TaskTerm kForC =
                  (nativeQK || kElem == "f32") ? kRaw : castOp(kRaw, kF32MatTy);
              sc = outerContractOp(qForContract, kForC, plan.axD, scoreMatTy,
                                   plan.scoreEngine);
              sc = applyScoreAffine2D(sc, tIv);
            }
            TaskTerm diff =
                applyPostSubScale(bin("sub", sc, peakVec, scoreMatTy),
                                  scoreMatTy);
            TaskTerm wt = emitExpWeight2D(diff, tIv, boundVec64, causalPred);
            TaskTerm rsum = reduceOp(wt, "add", 1, vecBTy);
            TaskTerm nms = bin("add", msIter, rsum, vecBTy);
            bool passNarrowW =
                !isRvv && plan.weightRoundType &&
                (plan.weightRoundType->isBF16() ||
                 plan.weightRoundType->isF16());
            TaskTerm wForPV =
                passNarrowW
                    ? castOp(wt, narrowScoreMatTy)
                    : (plan.weightRoundType
                           ? castOp(castOp(wt, narrowScoreMatTy), scoreMatTy)
                           : wt);
            TaskTerm vRaw =
                subview2D(vSl2D, tIv.v, c0.v, T, plan.DV, true, vRawMatTy);
            TaskTerm vForC = (nativeV || vElem == "f32")
                                 ? vRaw
                                 : castOp(vRaw, vF32MatTy);
            TaskTerm dO = outerContractOp(wForPV, vForC, plan.axTok, outMatTy,
                                          plan.valueEngine);
            TaskTerm nacc = bin("add", accIter, dO, outMatTy);
            makeOp(b, loc, "task_yield", {nms.v, nacc.v}, {},
                   {b.getNamedAttr("val_types", p2IterTys)});
          }
          TaskTerm msMain{p2Main->getResult(0), vecBTy};
          TaskTerm accMain{p2Main->getResult(1), outMatTy};
          if (exactTiles)
            return {msMain, accMain};

          TaskTerm qTailF32 =
              fromBuffer ? TaskTerm{}
                         : ((qElem == "f32") ? qRawSaved
                                             : castOp(qRawSaved, qF32MatTy));
          Operation *p2Tail = makeOp(
              b, loc, "task_for",
              {tMain.v, ub.v, c1.v, msMain.v, accMain.v},
              {b.getIndexType(), b.getIndexType()},
              {b.getNamedAttr("iter_types", p2IterTys)}, 1);
          {
            Block &fBb = p2Tail->getRegion(0).emplaceBlock();
            TaskTerm tIv{fBb.addArgument(b.getIndexType(), loc), "index"};
            TaskTerm msIter{fBb.addArgument(b.getIndexType(), loc), vecBTy};
            TaskTerm accIter{fBb.addArgument(b.getIndexType(), loc), outMatTy};
            OpBuilder::InsertionGuard g(b);
            b.setInsertionPointToEnd(&fBb);
            TaskTerm sc1D;
            if (fromBuffer) {
              TaskTerm offIdx =
                  loC && *loC == 0 ? tIv : bin("sub", tIv, lb, "index");
              TaskTerm off64 = castOp(offIdx, "si64");
              TaskTerm baseOff = bin("mul", off64, cB64, "si64");
              TaskTerm pOff1D = bin("add", qSlotOff, baseOff, si64BTy);
              Value ld = makeOp(
                             b, loc, "task_ptr_load",
                             {tiledScoreBuf.v, pOff1D.v, trueB.v, zeroV.v},
                             {b.getIndexType()},
                             {b.getNamedAttr(
                                  "ptr_type",
                                  b.getStringAttr("!weft_kernel.pointer<f32>")),
                              b.getNamedAttr("val_type",
                                             b.getStringAttr(vecBTy))})
                             ->getResult(0);
              sc1D = {ld, vecBTy};
            } else {
              TaskTerm kRaw1D =
                  slice1DAtTok(plan.kMem, plan.kCoords, plan.kTokCoordPos,
                               tIv.v, plan.kVecMemAx, kRaw1DTy);
              TaskTerm kF1D =
                  (kElem == "f32") ? kRaw1D : castOp(kRaw1D, kF321DTy);
              TaskTerm prod = bin("mul", qTailF32, kF1D, qF32MatTy);
              sc1D =
                  applyScoreAffine1D(reduceOp(prod, "add", 1, vecBTy), tIv);
            }
            TaskTerm diff1D =
                applyPostSubScale(bin("sub", sc1D, peakVec, vecBTy), vecBTy);
            TaskTerm wt1D =
                emitExpWeight1D(diff1D, tIv, boundVec64, causalPred);
            TaskTerm nms = bin("add", msIter, wt1D, vecBTy);
            TaskTerm wF1D = plan.weightRoundType
                                ? castOp(castOp(wt1D, narrowVecBTy), vecBTy)
                                : wt1D;
            TaskTerm vRaw1D =
                slice1DAtTok(plan.vMem, plan.vCoords, plan.vTokCoordPos, tIv.v,
                             plan.vVecMemAx, vRaw1DTy);
            TaskTerm vF1D =
                (vElem == "f32") ? vRaw1D : castOp(vRaw1D, vF321DTy);
            TaskTerm dO1D = bin("mul", wF1D, vF1D, outMatTy);
            TaskTerm nacc = bin("add", accIter, dO1D, outMatTy);
            makeOp(b, loc, "task_yield", {nms.v, nacc.v}, {},
                   {b.getNamedAttr("val_types", p2IterTys)});
          }
          return {{p2Tail->getResult(0), vecBTy},
                  {p2Tail->getResult(1), outMatTy}};
        };

        if (plan.hasScoreBuffer && staticallyFitsWindow) {
          std::tie(massVec, outMat) = emitPass2(true);
        } else if (!plan.hasScoreBuffer) {
          std::tie(massVec, outMat) = emitPass2(false);
        } else if (useTiledOut) {
          TaskTerm fits =
              cmp("le", castOp(span, "si64"),
                  lit(std::to_string(plan.scoreSlots), "si64"), "i1");
          ArrayAttr resTys = b.getArrayAttr({b.getStringAttr(vecBTy)});
          Operation *wIf =
              makeOp(b, loc, "task_if", {fits.v}, {b.getIndexType()},
                     {b.getNamedAttr("res_types", resTys)}, 2);
          {
            Block &tBb = wIf->getRegion(0).emplaceBlock();
            OpBuilder::InsertionGuard tg(b);
            b.setInsertionPointToEnd(&tBb);
            auto [mT, _oT] = emitPass2(true);
            (void)_oT;
            makeOp(b, loc, "task_yield", {mT.v}, {},
                   {b.getNamedAttr("val_types", resTys)});
          }
          {
            Block &eBb = wIf->getRegion(1).emplaceBlock();
            OpBuilder::InsertionGuard eg(b);
            b.setInsertionPointToEnd(&eBb);
            auto [mE, _oE] = emitPass2(false);
            (void)_oE;
            makeOp(b, loc, "task_yield", {mE.v}, {},
                   {b.getNamedAttr("val_types", resTys)});
          }
          massVec = {wIf->getResult(0), vecBTy};
          outInTiledBuf = true;
        } else {
          TaskTerm fits =
              cmp("le", castOp(span, "si64"),
                  lit(std::to_string(plan.scoreSlots), "si64"), "i1");
          ArrayAttr resTys = b.getArrayAttr(
              {b.getStringAttr(vecBTy), b.getStringAttr(outMatTy)});
          Operation *wIf =
              makeOp(b, loc, "task_if", {fits.v},
                     {b.getIndexType(), b.getIndexType()},
                     {b.getNamedAttr("res_types", resTys)}, 2);
          {
            Block &tBb = wIf->getRegion(0).emplaceBlock();
            OpBuilder::InsertionGuard tg(b);
            b.setInsertionPointToEnd(&tBb);
            auto [mT, oT] = emitPass2(true);
            makeOp(b, loc, "task_yield", {mT.v, oT.v}, {},
                   {b.getNamedAttr("val_types", resTys)});
          }
          {
            Block &eBb = wIf->getRegion(1).emplaceBlock();
            OpBuilder::InsertionGuard eg(b);
            b.setInsertionPointToEnd(&eBb);
            auto [mE, oE] = emitPass2(false);
            makeOp(b, loc, "task_yield", {mE.v, oE.v}, {},
                   {b.getNamedAttr("val_types", resTys)});
          }
          massVec = {wIf->getResult(0), vecBTy};
          outMat = {wIf->getResult(1), outMatTy};
        }

        if (llvm::any_of(plan.stage2MassRes.getUsers(), [&](Operation *u) {
              return !skipOps.contains(u) &&
                     (!plan.invIfOrSelOp || !plan.invIfOrSelOp->isAncestor(u));
            }))
          extractScalarLanes(massVec, plan.stage2MassRes);
        continue;
      }

      if (&op == plan.yWriteOp) {
        if (directSubCommitted)
          continue;
        if (outInTiledBuf) {
          TaskTerm invVec = plan.yMulOp ? computeInvVec(massVec) : TaskTerm{};
          SmallVector<Value, 4> yOtherCoords;
          for (auto [pos, c] :
               llvm::enumerate(plan.yWriteOp->getOperands().drop_front(2)))
            if (int64_t(pos) != plan.yOwnMemAx)
              yOtherCoords.push_back(c);
          TaskTerm ySl2D =
              slice2D(plan.yMem, yOtherCoords, plan.yOwnMemAx, plan.yVecMemAx);
          std::string commitSubSlTy = ySl2D.type;
          if (auto brk = commitSubSlTy.rfind(">,[");
              brk != std::string::npos) {
            auto cls = commitSubSlTy.find(']', brk + 2);
            commitSubSlTy.replace(brk + 2, cls - (brk + 2) + 1,
                                  "[" + std::to_string(B) + "," +
                                      std::to_string(subDV) + "]");
          }
          TaskTerm c0Idx = idx(0), cDvIdx = idx(plan.DV),
                   cSubDvIdx = idx(subDV);
          Operation *dvFor = makeOp(
              b, loc, "task_for", {c0Idx.v, cDvIdx.v, cSubDvIdx.v}, {},
              {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
          {
            Block &dvBb = dvFor->getRegion(0).emplaceBlock();
            TaskTerm dv0Idx{dvBb.addArgument(b.getIndexType(), loc), "index"};
            OpBuilder::InsertionGuard dvg(b);
            b.setInsertionPointToEnd(&dvBb);
            TaskTerm slotOff = bin("add", subOutSlotOff,
                                   castOp(dv0Idx, "si64"), subSi64OutTy);
            Value accLd =
                makeOp(b, loc, "task_ptr_load",
                       {tiledOutBuf.v, slotOff.v, trueSubO.v, zeroSubO.v},
                       {b.getIndexType()},
                       {b.getNamedAttr(
                            "ptr_type",
                            b.getStringAttr("!weft_kernel.pointer<f32>")),
                        b.getNamedAttr("val_type", b.getStringAttr(subOutTy))})
                    ->getResult(0);
            TaskTerm scaledSub =
                plan.yMulOp ? bin("mul", {accLd, subOutTy}, invVec, subOutTy)
                            : TaskTerm{accLd, subOutTy};
            TaskTerm finalSub = (yElem == "f32")
                                    ? scaledSub
                                    : castOp(scaledSub, ySubMatTy);
            TaskTerm ySubTile = subview2D(ySl2D, lanes[0].at(innerArg).v,
                                          dv0Idx.v, B, subDV, false,
                                          ySl2D.type);
            makeOp(b, loc, "task_commit", {finalSub.v, ySubTile.v}, {},
                   {b.getNamedAttr("val_type", b.getStringAttr(finalSub.type)),
                    b.getNamedAttr("slice_type",
                                   b.getStringAttr(commitSubSlTy))});
            makeOp(b, loc, "task_yield", {}, {},
                   {b.getNamedAttr("val_types", b.getArrayAttr({}))});
          }
          continue;
        }
        TaskTerm scaledOut =
            plan.yMulOp ? bin("mul", outMat, computeInvVec(massVec), outMatTy)
                        : outMat;
        TaskTerm finalOut =
            (yElem == "f32") ? scaledOut : castOp(scaledOut, yMatTy);
        SmallVector<Value, 4> yOtherCoords;
        for (auto [pos, c] :
             llvm::enumerate(plan.yWriteOp->getOperands().drop_front(2)))
          if (int64_t(pos) != plan.yOwnMemAx)
            yOtherCoords.push_back(c);
        TaskTerm ySl2D =
            slice2D(plan.yMem, yOtherCoords, plan.yOwnMemAx, plan.yVecMemAx);
        TaskTerm c0 = idx(0);
        TaskTerm ySub = subview2D(ySl2D, lanes[0].at(innerArg).v, c0.v, B,
                                  plan.DV, false, ySl2D.type);
        std::string commitSlTy = ySl2D.type;
        auto brk = commitSlTy.rfind(">,[");
        if (brk != std::string::npos) {
          auto cls = commitSlTy.find(']', brk + 2);
          commitSlTy.replace(brk + 2, cls - (brk + 2) + 1,
                             "[" + std::to_string(B) + "," +
                                 std::to_string(plan.DV) + "]");
        }
        makeOp(b, loc, "task_commit", {finalOut.v, ySub.v}, {},
               {b.getNamedAttr("val_type", b.getStringAttr(finalOut.type)),
                b.getNamedAttr("slice_type", b.getStringAttr(commitSlTy))});
        continue;
      }

      if (failed(lowerSingleOpInBlock(op, B, 0)))
        return failure();
    }
    return success();
  }

  LogicalResult lowerSingleOpInBlock(Operation &op, int64_t activeLanes,
                                     int64_t laneStart) {
    return lowerOp(op, activeLanes, laneStart);
  }

  LogicalResult lowerBlock(Block &body, int64_t activeLanes, int64_t laneStart) {
    auto narrowed = [](Operation *o) {
      return isa<arith::TruncFOp>(o) && o->hasOneUse() &&
             (*o->getUsers().begin())->hasAttr("weft_host.member_store");
    };
    auto multiplied = [&](Operation *o) {
      return isa<arith::MulFOp>(o) && o->hasOneUse() &&
             narrowed(*o->getUsers().begin());
    };

    for (Operation &op : body) {
      if (narrowed(&op) || multiplied(&op))
        continue;
      if (failed(lowerOp(op, activeLanes, laneStart)))
        return failure();
    }
    return success();
  }

  bool tryLowerVectorizedInnerLoop(scf::ForOp loop, int64_t laneStart) {
    if (isRvv)
      return false;
    if (staticIndexVal(fn, loop.getStep()) != 1 ||
        loop.getNumRegionIterArgs() > 1 || analyzeCausalLoop(fn, loop))
      return false;

    const bool isRed = loop.getNumRegionIterArgs() == 1;
    bool isAdd = false;
    Operation *redOp = nullptr;
    Value iterArg, stepTerm;
    if (isRed) {
      if (!loop->hasAttr("kir.reassociate") ||
          !loop.getResult(0).getType().isF32())
        return false;
      iterArg = loop.getRegionIterArgs()[0];
      if (!iterArg.hasOneUse())
        return false;
      Value yVal = loop.getBody()->getTerminator()->getOperand(0);
      redOp = yVal.getDefiningOp();
      if (!redOp || redOp->getBlock() != loop.getBody())
        return false;
      if (auto add = dyn_cast<arith::AddFOp>(redOp)) {
        isAdd = true;
        stepTerm = add.getLhs() == iterArg
                       ? add.getRhs()
                       : (add.getRhs() == iterArg ? add.getLhs() : Value{});
      } else if (isa<arith::MaxNumFOp, arith::MaximumFOp>(redOp)) {
        isAdd = false;
        stepTerm = redOp->getOperand(0) == iterArg
                       ? redOp->getOperand(1)
                       : (redOp->getOperand(1) == iterArg ? redOp->getOperand(0)
                                                          : Value{});
      } else {
        return false;
      }
      if (!stepTerm || stepTerm == iterArg)
        return false;
    }

    auto vecElemOk = [](Type t) {
      return t.isF32() || t.isBF16() || t.isF16() || t.isIndex() ||
             t.isInteger(1) || t.isInteger(32) || t.isInteger(64) ||
             t.isUnsignedInteger(64);
    };
    constexpr int64_t W = 16;
    Value ivOrig = loop.getInductionVar();

    llvm::DenseSet<Value> writtenViews;
    for (Operation &bop : loop.getBody()->without_terminator()) {
      if (auto st = dyn_cast<memref::StoreOp>(bop)) {
        if (isRed || !isa<BlockArgument>(st.getMemref()))
          return false;
        writtenViews.insert(st.getMemref());
      }
    }
    if (!isRed && writtenViews.empty())
      return false;

    auto pureVecOpOk = [&](Operation &bop) -> bool {
      if (bop.getNumRegions() != 0 || bop.getNumResults() != 1 ||
          isa<ShapedType>(bop.getResult(0).getType()) ||
          !vecElemOk(bop.getResult(0).getType()))
        return false;
      if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
              arith::MaxNumFOp, arith::MaximumFOp, arith::MinNumFOp,
              arith::MinimumFOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
              arith::AndIOp, arith::OrIOp, arith::XOrIOp>(bop))
        return vecElemOk(bop.getOperand(0).getType()) &&
               vecElemOk(bop.getOperand(1).getType());
      if (isa<math::ExpOp, math::Exp2Op, math::LogOp, math::SqrtOp,
              math::RsqrtOp, arith::ExtFOp, arith::TruncFOp,
              arith::IndexCastOp, arith::SIToFPOp, arith::FPToSIOp>(bop))
        return vecElemOk(bop.getOperand(0).getType());
      if (auto ci = dyn_cast<arith::CmpIOp>(bop)) {
        auto p = ci.getPredicate();
        return vecElemOk(ci.getLhs().getType()) &&
               (p == arith::CmpIPredicate::eq ||
                p == arith::CmpIPredicate::ne ||
                p == arith::CmpIPredicate::slt ||
                p == arith::CmpIPredicate::sle ||
                p == arith::CmpIPredicate::sgt ||
                p == arith::CmpIPredicate::sge ||
                p == arith::CmpIPredicate::ult ||
                p == arith::CmpIPredicate::ule ||
                p == arith::CmpIPredicate::ugt ||
                p == arith::CmpIPredicate::uge);
      }
      if (auto cf = dyn_cast<arith::CmpFOp>(bop))
        return vecElemOk(cf.getLhs().getType());
      if (auto selOp = dyn_cast<arith::SelectOp>(bop))
        return selOp.getCondition().getType().isInteger(1) &&
               vecElemOk(selOp.getTrueValue().getType());
      return false;
    };

    llvm::DenseSet<Value> dependsOnIv{ivOrig};
    int64_t vecAxisId = -1;
    bool hasContigMem = false;
    bool needsIvIota = false;
    SmallVector<Operation *, 4> contigMemOps;

    for (Operation &bop : loop.getBody()->without_terminator()) {
      if (&bop == redOp)
        continue;
      if (isRed && llvm::is_contained(bop.getOperands(), iterArg))
        return false;
      if (auto ld = dyn_cast<memref::LoadOp>(bop)) {
        if (writtenViews.contains(ld.getMemref()))
          return false;
        bool anyIv = llvm::any_of(
            ld.getIndices(), [&](Value v) { return dependsOnIv.contains(v); });
        if (!anyIv)
          continue;
        Value mem = ld.getMemref();
        if (!isa<BlockArgument>(mem))
          return false;
        int64_t rank = cast<MemRefType>(mem.getType()).getRank();
        if (ld.getIndices().empty() || ld.getIndices().back() != ivOrig ||
            dimOf(mem, "cpu.strides", rank - 1) != 1 ||
            dimOf(mem, "cpu.shape", rank - 1) < W ||
            !vecElemOk(ld.getResult().getType()) ||
            llvm::any_of(ld.getIndices().drop_back(),
                         [&](Value v) { return dependsOnIv.contains(v); }))
          return false;
        int64_t ax = axesMap.at(mem)[rank - 1];
        if (vecAxisId < 0)
          vecAxisId = ax;
        else if (vecAxisId != ax)
          return false;
        hasContigMem = true;
        contigMemOps.push_back(&bop);
        dependsOnIv.insert(ld.getResult());
        continue;
      }
      if (auto st = dyn_cast<memref::StoreOp>(bop)) {
        Value mem = st.getMemref();
        int64_t rank = cast<MemRefType>(mem.getType()).getRank();
        if (st.getIndices().empty() || st.getIndices().back() != ivOrig ||
            dimOf(mem, "cpu.strides", rank - 1) != 1 ||
            dimOf(mem, "cpu.shape", rank - 1) < W ||
            !vecElemOk(st.getValue().getType()) ||
            llvm::any_of(st.getIndices().drop_back(),
                         [&](Value v) { return dependsOnIv.contains(v); }))
          return false;
        int64_t ax = axesMap.at(mem)[rank - 1];
        if (vecAxisId < 0)
          vecAxisId = ax;
        else if (vecAxisId != ax)
          return false;
        hasContigMem = true;
        contigMemOps.push_back(&bop);
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(bop)) {
        if (ifOp.getNumResults() != 1 || ifOp.getElseRegion().empty() ||
            isa<ShapedType>(ifOp.getResult(0).getType()) ||
            !vecElemOk(ifOp.getResult(0).getType()))
          return false;
        auto checkBranch = [&](Block *brBb) -> bool {
          for (Operation &sub : brBb->without_terminator()) {
            if (isRed && llvm::is_contained(sub.getOperands(), iterArg))
              return false;
            if (isa<arith::ConstantOp, arith::ConstantIndexOp, memref::DimOp>(
                    sub))
              continue;
            if (!pureVecOpOk(sub))
              return false;
            if (llvm::is_contained(sub.getOperands(), ivOrig))
              needsIvIota = true;
          }
          return true;
        };
        if (!checkBranch(ifOp.thenBlock()) || !checkBranch(ifOp.elseBlock()))
          return false;
        if (ifOp.getCondition() == ivOrig)
          needsIvIota = true;
        dependsOnIv.insert(ifOp.getResult(0));
        continue;
      }
      if (bop.getNumRegions() != 0)
        return false;
      if (isa<memref::DimOp, arith::ConstantOp, arith::ConstantIndexOp>(bop))
        continue;
      bool anyIv = llvm::any_of(
          bop.getOperands(), [&](Value v) { return dependsOnIv.contains(v); });
      if (!anyIv)
        continue;
      if (!pureVecOpOk(bop))
        return false;
      if (llvm::is_contained(bop.getOperands(), ivOrig))
        needsIvIota = true;
      dependsOnIv.insert(bop.getResult(0));
    }
    if (!hasContigMem || vecAxisId < 0)
      return false;

    auto vecTyStr = [&](StringRef elemSp) {
      return "!weft_kernel.value<" + elemSp.str() + ",[" + std::to_string(W) +
             "],[" + std::to_string(vecAxisId) + "]>";
    };
    const std::string f32VecTy = vecTyStr("f32");
    const std::string i1VecTy = vecTyStr("i1");
    const std::string si64VecTy = vecTyStr("si64");

    TaskTerm lbTerm = lanes[laneStart].at(loop.getLowerBound());
    TaskTerm ubTerm = lanes[laneStart].at(loop.getUpperBound());
    TaskTerm c0 = idx(0), c1 = idx(1), cW = idx(W);
    bool exact = isStaticMultipleOf(fn, loop.getLowerBound(), W) &&
                 isStaticMultipleOf(fn, loop.getUpperBound(), W);
    TaskTerm vecEnd = ubTerm;
    if (!exact) {
      TaskTerm span = bin("sub", ubTerm, lbTerm, "index");
      TaskTerm span64 = castOp(span, "si64");
      TaskTerm geW = cmp("ge", span64, lit(std::to_string(W), "si64"), "i1");
      TaskTerm aligned = bin("mul", bin("div", span, cW, "index"), cW, "index");
      TaskTerm vecSpan = sel(geW, aligned, c0, "index");
      vecEnd = bin("add", lbTerm, vecSpan, "index");
    }

    // Hoist loop-invariant 1D row slices (admit=false) for contiguous accesses.
    llvm::DenseMap<Operation *, std::pair<TaskTerm, std::string>> hoistedSlices;
    for (Operation *mOp : contigMemOps) {
      Value mem = isa<memref::LoadOp>(mOp) ? cast<memref::LoadOp>(mOp).getMemref()
                                           : cast<memref::StoreOp>(mOp).getMemref();
      ValueRange idxs = isa<memref::LoadOp>(mOp)
                            ? cast<memref::LoadOp>(mOp).getIndices()
                            : cast<memref::StoreOp>(mOp).getIndices();
      int64_t rank = cast<MemRefType>(mem.getType()).getRank();
      TaskTerm rowSl =
          sliceView(laneStart, mem, idxs.drop_back(), rank - 1, false);
      std::string subSlTy = rowSl.type;
      if (auto brk = subSlTy.rfind(">,["); brk != std::string::npos) {
        auto cls = subSlTy.find(']', brk + 2);
        subSlTy.replace(brk + 2, cls - (brk + 2) + 1,
                        "[" + std::to_string(W) + "]");
      }
      hoistedSlices[mOp] = {rowSl, subSlTy};
    }

    TaskTerm iotaW64;
    if (needsIvIota) {
      std::string idxVecTy = vecTyStr("index");
      Value io = makeOp(b, loc, "task_iota", {}, {b.getIndexType()},
                        {b.getNamedAttr("start", b.getI64IntegerAttr(0)),
                         b.getNamedAttr("end", b.getI64IntegerAttr(W)),
                         b.getNamedAttr("val_type", b.getStringAttr(idxVecTy))})
                     ->getResult(0);
      iotaW64 = castOp({io, idxVecTy}, si64VecTy);
    }

    SmallVector<Value> vForOps{lbTerm.v, vecEnd.v, cW.v};
    SmallVector<Attribute> vIterTys;
    SmallVector<Type> vResTys;
    if (isRed) {
      TaskTerm identS =
          isAdd ? lit("0.0", "f32") : lit("0xFF800000", "f32");
      TaskTerm identV = newVec(identS, f32VecTy);
      vForOps.push_back(identV.v);
      vIterTys.push_back(b.getStringAttr(f32VecTy));
      vResTys.push_back(b.getIndexType());
    }

    Operation *vFor =
        makeOp(b, loc, "task_for", vForOps, vResTys,
               {b.getNamedAttr("iter_types", b.getArrayAttr(vIterTys))}, 1);
    {
      Block &vBb = vFor->getRegion(0).emplaceBlock();
      TaskTerm ivVec{vBb.addArgument(b.getIndexType(), loc), "index"};
      TaskTerm vAccIter;
      if (isRed)
        vAccIter = {vBb.addArgument(b.getIndexType(), loc), f32VecTy};
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToEnd(&vBb);

      llvm::DenseMap<Value, TaskTerm> vMap;
      if (needsIvIota)
        vMap[ivOrig] = bin("add", iotaW64, castOp(ivVec, "si64"), si64VecTy);

      auto isVecTerm = [](const TaskTerm &t) {
        return StringRef(t.type).starts_with("!weft_kernel.value<");
      };
      auto getTerm = [&](Value orig) -> TaskTerm {
        if (auto it = vMap.find(orig); it != vMap.end())
          return it->second;
        return lanes[laneStart].at(orig);
      };
      auto asVec = [&](TaskTerm t, StringRef targetVecTy) -> TaskTerm {
        if (isVecTerm(t))
          return t;
        return newVec(t, targetVecTy);
      };
      auto elemOfVecTy = [](StringRef ty) -> StringRef {
        if (ty.starts_with("!weft_kernel.value<")) {
          auto rest = ty.drop_front(19);
          return rest.take_until([](char c) { return c == ','; });
        }
        return ty;
      };

      auto lowerVecPureOp = [&](Operation &bop,
                                std::optional<TaskTerm> guardMask) -> TaskTerm {
        StringRef opN = bop.getName().getStringRef();
        StringRef opK = bop.getName().stripDialect();
        if (auto d = dyn_cast<memref::DimOp>(bop))
          return idx(dimOf(d.getSource(), "cpu.shape", *d.getConstantIndex()));
        if (auto c = dyn_cast<arith::ConstantOp>(bop)) {
          if (c.getType().isIndex())
            return idx(cast<IntegerAttr>(c.getValue()).getInt());
          if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
            return lit(c.getType().isInteger(1)
                           ? (ia.getInt() ? "true" : "false")
                           : std::to_string(ia.getInt()),
                       scalarType(c.getType()));
          return lit(stripTypeSuffix(c.getValue()), scalarType(c.getType()));
        }
        std::string resElem = scalarType(bop.getResult(0).getType());
        if (resElem == "index")
          resElem = "si64";
        std::string resVTy = vecTyStr(resElem);
        if (isa<arith::ExtFOp, arith::TruncFOp, arith::IndexCastOp,
                arith::SIToFPOp, arith::FPToSIOp>(bop)) {
          TaskTerm in = getTerm(bop.getOperand(0));
          if (!isVecTerm(in))
            return castOp(in, scalarType(bop.getResult(0).getType()));
          if (in.type == resVTy)
            return in;
          return castOp(in, resVTy);
        }
        if (isa<math::ExpOp, math::Exp2Op, math::LogOp, math::SqrtOp,
                math::RsqrtOp>(bop)) {
          TaskTerm in = getTerm(bop.getOperand(0));
          if (guardMask && isVecTerm(in) &&
              isa<math::LogOp, math::SqrtOp, math::RsqrtOp>(bop)) {
            TaskTerm oneV = newVec(lit("1.0", "f32"), in.type);
            in = sel(*guardMask, in, oneV, in.type);
          }
          return unaryOp(opK, in);
        }
        if (auto selOp = dyn_cast<arith::SelectOp>(bop)) {
          TaskTerm cT = getTerm(selOp.getCondition());
          TaskTerm tT = getTerm(selOp.getTrueValue());
          TaskTerm fT = getTerm(selOp.getFalseValue());
          if (isVecTerm(cT) || isVecTerm(tT) || isVecTerm(fT))
            return sel(asVec(cT, i1VecTy), asVec(tT, resVTy), asVec(fT, resVTy),
                       resVTy);
          return sel(cT, tT, fT, scalarType(bop.getResult(0).getType()));
        }
        if (auto ci = dyn_cast<arith::CmpIOp>(bop)) {
          TaskTerm a = getTerm(ci.getLhs()), r = getTerm(ci.getRhs());
          bool vec = isVecTerm(a) || isVecTerm(r);
          if (vec) {
            if (a.type == "index")
              a = castOp(a, "si64");
            if (r.type == "index")
              r = castOp(r, "si64");
          } else if (a.type == "index" &&
                     ci.getPredicate() != arith::CmpIPredicate::eq &&
                     ci.getPredicate() != arith::CmpIPredicate::ne) {
            a = castOp(a, "si64");
            r = castOp(r, "si64");
          }
          StringRef pred = "eq";
          switch (ci.getPredicate()) {
          case arith::CmpIPredicate::eq: pred = "eq"; break;
          case arith::CmpIPredicate::ne: pred = "ne"; break;
          case arith::CmpIPredicate::slt:
          case arith::CmpIPredicate::ult: pred = "lt"; break;
          case arith::CmpIPredicate::sle:
          case arith::CmpIPredicate::ule: pred = "le"; break;
          case arith::CmpIPredicate::sgt:
          case arith::CmpIPredicate::ugt: pred = "gt"; break;
          case arith::CmpIPredicate::sge:
          case arith::CmpIPredicate::uge: pred = "ge"; break;
          }
          return cmp(pred, a, r, vec ? StringRef(i1VecTy) : StringRef("i1"));
        }
        if (auto cf = dyn_cast<arith::CmpFOp>(bop)) {
          TaskTerm a = getTerm(cf.getLhs()), r = getTerm(cf.getRhs());
          bool vec = isVecTerm(a) || isVecTerm(r);
          StringRef pred = "eq";
          switch (cf.getPredicate()) {
          case arith::CmpFPredicate::OEQ:
          case arith::CmpFPredicate::UEQ: pred = "eq"; break;
          case arith::CmpFPredicate::ONE:
          case arith::CmpFPredicate::UNE: pred = "ne"; break;
          case arith::CmpFPredicate::OLT:
          case arith::CmpFPredicate::ULT: pred = "lt"; break;
          case arith::CmpFPredicate::OLE:
          case arith::CmpFPredicate::ULE: pred = "le"; break;
          case arith::CmpFPredicate::OGT:
          case arith::CmpFPredicate::UGT: pred = "gt"; break;
          default: pred = "ge"; break;
          }
          return cmp(pred, a, r, vec ? StringRef(i1VecTy) : StringRef("i1"));
        }
        TaskTerm a = getTerm(bop.getOperand(0)), r = getTerm(bop.getOperand(1));
        bool vec = isVecTerm(a) || isVecTerm(r);
        if (vec) {
          if (a.type == "index")
            a = castOp(a, "si64");
          if (r.type == "index")
            r = castOp(r, "si64");
        }
        StringRef tok = llvm::StringSwitch<StringRef>(opN)
                            .Cases("arith.addf", "arith.addi", "add")
                            .Cases("arith.subf", "arith.subi", "sub")
                            .Cases("arith.mulf", "arith.muli", "mul")
                            .Case("arith.divf", "div")
                            .Cases("arith.maxnumf", "arith.maximumf", "max")
                            .Cases("arith.minnumf", "arith.minimumf", "min")
                            .Case("arith.andi", "and")
                            .Case("arith.ori", "or")
                            .Case("arith.xori", "xor")
                            .Default("add");
        std::string outTy =
            vec ? vecTyStr(elemOfVecTy(isVecTerm(a) ? a.type : r.type))
                : a.type;
        return lowerBinary(*this, bop, tok, a, r, outTy);
      };

      for (Operation &bop : loop.getBody()->without_terminator()) {
        if (&bop == redOp)
          continue;
        if (auto ld = dyn_cast<memref::LoadOp>(bop)) {
          if (auto it = hoistedSlices.find(&bop); it != hoistedSlices.end()) {
            std::string vTy = vecTyStr(scalarType(ld.getType()));
            Value val =
                makeOp(b, loc, "task_subview", {it->second.first.v, ivVec.v},
                       {b.getIndexType()},
                       {b.getNamedAttr("static_extent", b.getI64IntegerAttr(W)),
                        b.getNamedAttr("slice_type",
                                       b.getStringAttr(it->second.first.type)),
                        b.getNamedAttr("val_type", b.getStringAttr(vTy)),
                        b.getNamedAttr("admit", b.getBoolAttr(true))})
                    ->getResult(0);
            vMap[ld.getResult()] = {val, vTy};
          } else {
            vMap[ld.getResult()] = sliceView(laneStart, ld.getMemref(),
                                             ld.getIndices(), -1, true,
                                             scalarType(ld.getType()));
          }
          continue;
        }
        if (auto st = dyn_cast<memref::StoreOp>(bop)) {
          auto &[rowSl, subSlTy] = hoistedSlices.at(&bop);
          std::string vTy = vecTyStr(scalarType(st.getValue().getType()));
          TaskTerm valVec = asVec(getTerm(st.getValue()), vTy);
          Value subSl =
              makeOp(b, loc, "task_subview", {rowSl.v, ivVec.v},
                     {b.getIndexType()},
                     {b.getNamedAttr("static_extent", b.getI64IntegerAttr(W)),
                      b.getNamedAttr("slice_type", b.getStringAttr(rowSl.type)),
                      b.getNamedAttr("val_type", b.getStringAttr(subSlTy)),
                      b.getNamedAttr("admit", b.getBoolAttr(false))})
                  ->getResult(0);
          makeOp(b, loc, "task_commit", {valVec.v, subSl}, {},
                 {b.getNamedAttr("val_type", b.getStringAttr(valVec.type)),
                  b.getNamedAttr("slice_type", b.getStringAttr(subSlTy))});
          continue;
        }
        if (auto ifOp = dyn_cast<scf::IfOp>(bop)) {
          TaskTerm condV = asVec(getTerm(ifOp.getCondition()), i1VecTy);
          TaskTerm notCondV = sel(condV, newVec(lit("false", "i1"), i1VecTy),
                                  newVec(lit("true", "i1"), i1VecTy), i1VecTy);
          for (Operation &sub : ifOp.thenBlock()->without_terminator())
            vMap[sub.getResult(0)] = lowerVecPureOp(sub, condV);
          for (Operation &sub : ifOp.elseBlock()->without_terminator())
            vMap[sub.getResult(0)] = lowerVecPureOp(sub, notCondV);
          std::string resElem = scalarType(ifOp.getResult(0).getType());
          if (resElem == "index")
            resElem = "si64";
          std::string resVTy = vecTyStr(resElem);
          TaskTerm tRes = asVec(getTerm(ifOp.thenYield().getOperand(0)), resVTy);
          TaskTerm eRes = asVec(getTerm(ifOp.elseYield().getOperand(0)), resVTy);
          vMap[ifOp.getResult(0)] = sel(condV, tRes, eRes, resVTy);
          continue;
        }
        vMap[bop.getResult(0)] = lowerVecPureOp(bop, std::nullopt);
      }

      if (isRed) {
        TaskTerm stepV = asVec(getTerm(stepTerm), f32VecTy);
        TaskTerm nextV =
            bin(isAdd ? "add" : "max", vAccIter, stepV, f32VecTy);
        makeOp(b, loc, "task_yield", {nextV.v}, {},
               {b.getNamedAttr("val_types", b.getArrayAttr(vIterTys))});
      } else {
        makeOp(b, loc, "task_yield", {}, {},
               {b.getNamedAttr("val_types", b.getArrayAttr({}))});
      }
    }

    TaskTerm curScalar;
    if (isRed) {
      TaskTerm initS = lanes[laneStart].at(loop.getInitArgs()[0]);
      TaskTerm vFinal{vFor->getResult(0), f32VecTy};
      TaskTerm redS = reduceOp(vFinal, isAdd ? "add" : "max", 0, "f32");
      TaskTerm ran = cmp("gt", castOp(vecEnd, "si64"), castOp(lbTerm, "si64"),
                         "i1");
      TaskTerm combined = bin(isAdd ? "add" : "max", initS, redS, "f32");
      curScalar = sel(ran, combined, initS, "f32");
    }

    if (!exact) {
      SmallVector<Value> tForOps{vecEnd.v, ubTerm.v, c1.v};
      SmallVector<Attribute> tIterTys;
      SmallVector<Type> tResTys;
      if (isRed) {
        tForOps.push_back(curScalar.v);
        tIterTys.push_back(b.getStringAttr("f32"));
        tResTys.push_back(b.getIndexType());
      }
      Operation *tFor =
          makeOp(b, loc, "task_for", tForOps, tResTys,
                 {b.getNamedAttr("iter_types", b.getArrayAttr(tIterTys))}, 1);
      Block &tBb = tFor->getRegion(0).emplaceBlock();
      Value tIv = tBb.addArgument(b.getIndexType(), loc);
      lanes[laneStart][ivOrig] = {tIv, "index"};
      if (isRed)
        lanes[laneStart][iterArg] = {tBb.addArgument(b.getIndexType(), loc),
                                     "f32"};
      OpBuilder::InsertionGuard tg(b);
      b.setInsertionPointToEnd(&tBb);
      if (failed(lowerBlock(*loop.getBody(), 1, laneStart)))
        return false;
      if (tBb.empty() ||
          tBb.back().getName().getStringRef() != "weft_host.task_yield") {
        makeOp(b, loc, "task_yield", {}, {},
               {b.getNamedAttr("val_types", b.getArrayAttr({}))});
      }
      if (isRed)
        curScalar = {tFor->getResult(0), "f32"};
    }

    if (isRed)
      lanes[laneStart][loop.getResult(0)] = curScalar;
    return true;
  }

  LogicalResult lowerOp(Operation &op, int64_t activeLanes, int64_t laneStart) {
    StringRef name = op.getName().getStringRef();
    StringRef kind = op.getName().stripDialect();
    if (kind == "end" || name == "cpu.local_buffer")
      return success();

    if (name == "scf.yield") {
      SmallVector<Value> yVals;
      SmallVector<Attribute> yTys;
      for (int64_t l = laneStart; l < laneStart + activeLanes; ++l) {
        for (Value v : op.getOperands()) {
          TaskTerm t = lanes[l].at(v);
          yVals.push_back(t.v);
          yTys.push_back(b.getStringAttr(t.type));
        }
      }
      makeOp(b, loc, "task_yield", yVals, {},
             {b.getNamedAttr("val_types", b.getArrayAttr(yTys))});
      return success();
    }

      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        if (activeLanes == 1 && tryLowerVectorizedInnerLoop(loop, laneStart))
          return success();
        TaskTerm lbTerm = lanes[laneStart].at(loop.getLowerBound());
        TaskTerm ubTerm = lanes[laneStart].at(loop.getUpperBound());
        Value lb = lbTerm.v;
        Value ub = ubTerm.v;
        Value st = lanes[laneStart].at(loop.getStep()).v;

        Value simplifiedCmp;
        if (auto tight = analyzeCausalLoop(fn, loop)) {
          anyTightenedCausalLoop = true;
          bool ok = (!tight->hasBufferStore || allowBufferStoreTightening) &&
                    lanes[laneStart].count(tight->boundVal);
          if (ok) {
            Value b0 = lanes[laneStart].at(tight->boundVal).v;
            for (int64_t l = laneStart + 1; l < laneStart + activeLanes; ++l)
              if (!lanes[l].count(tight->boundVal) ||
                  lanes[l].at(tight->boundVal).v != b0)
                ok = false;
          }
          if (ok) {
            TaskTerm lb64 = castOp(lbTerm, "si64");
            TaskTerm ub64 = castOp(ubTerm, "si64");
            TaskTerm lim64 = castOp(lanes[laneStart].at(tight->boundVal), "si64");
            if (tight->inclusive)
              lim64 = bin("add", lim64, lit("1", "si64"), "si64");
            TaskTerm clampedLo =
                sel(cmp("gt", lim64, lb64, "i1"), lim64, lb64, "si64");
            TaskTerm clampedUb =
                sel(cmp("lt", clampedLo, ub64, "i1"), clampedLo, ub64, "si64");
            ub = castOp(clampedUb, "index").v;
            simplifiedCmp = tight->cmpResult;
            knownTrueConds.insert(simplifiedCmp);
          }
        }

        SmallVector<Value> forOps{lb, ub, st};
        SmallVector<Attribute> iterTys;
        SmallVector<Type> resTypes;
        for (int64_t l = laneStart; l < laneStart + activeLanes; ++l) {
          for (Value init : loop.getInitArgs()) {
            TaskTerm t = lanes[l].at(init);
            forOps.push_back(t.v);
            iterTys.push_back(b.getStringAttr(t.type));
            resTypes.push_back(b.getIndexType());
          }
        }
        Operation *forOp =
            makeOp(b, loc, "task_for", forOps, resTypes,
                   {b.getNamedAttr("iter_types", b.getArrayAttr(iterTys))}, 1);
        Block &fBb = forOp->getRegion(0).emplaceBlock();
        Value iv = fBb.addArgument(b.getIndexType(), loc);
        for (int64_t l = laneStart; l < laneStart + activeLanes; ++l) {
          lanes[l][loop.getInductionVar()] = {iv, "index"};
          for (auto [init, arg] :
               llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs())) {
            std::string ty = lanes[l].at(init).type;
            lanes[l][arg] = {fBb.addArgument(b.getIndexType(), loc), ty};
          }
        }
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToEnd(&fBb);
        LogicalResult bodyRes = lowerBlock(*loop.getBody(), activeLanes, laneStart);
        if (simplifiedCmp)
          knownTrueConds.erase(simplifiedCmp);
        if (failed(bodyRes))
          return failure();
        if (fBb.empty() ||
            fBb.back().getName().getStringRef() != "weft_host.task_yield") {
          makeOp(b, loc, "task_yield", {}, {},
                 {b.getNamedAttr("val_types", b.getArrayAttr({}))});
        }
        unsigned idxRes = 0;
        for (int64_t l = laneStart; l < laneStart + activeLanes; ++l) {
          for (auto [init, ret] :
               llvm::zip(loop.getInitArgs(), loop.getResults())) {
            lanes[l][ret] = {forOp->getResult(idxRes++),
                             lanes[l].at(init).type};
          }
        }
        return success();
      }

      if (auto branch = dyn_cast<scf::IfOp>(op)) {
        if (knownTrueConds.contains(branch.getCondition())) {
          auto yieldOp = branch.thenYield();
          yieldOp->remove();
          LogicalResult st =
              lowerBlock(*branch.thenBlock(), activeLanes, laneStart);
          branch.thenBlock()->push_back(yieldOp);
          if (failed(st))
            return failure();
          for (int64_t l = laneStart; l < laneStart + activeLanes; ++l) {
            for (auto [ret, yVal] :
                 llvm::zip(branch.getResults(), yieldOp.getOperands())) {
              lanes[l][ret] = lanes[l].at(yVal);
            }
          }
          return success();
        }
        if (branch->hasAttr("cpu.window_guard")) {
          if (auto pLoop = branch->getParentOfType<scf::ForOp>()) {
            auto loC = staticIndexVal(fn, pLoop.getLowerBound());
            auto hiC = staticIndexVal(fn, pLoop.getUpperBound());
            std::optional<int64_t> winC;
            if (auto cmpOp = branch.getCondition().getDefiningOp<arith::CmpIOp>())
              winC = staticIndexVal(fn, cmpOp.getRhs());
            if (loC && hiC && winC && (*hiC - *loC) <= *winC) {
              for (Operation &inner : branch.thenBlock()->without_terminator()) {
                if (inner.getName().getStringRef() == "cpu.buffer_store") {
                  for (int64_t l = laneStart; l < laneStart + activeLanes; ++l) {
                    TaskTerm val = lanes[l].at(inner.getOperand(0));
                    TaskTerm off = castOp(lanes[l].at(inner.getOperand(2)), "si64");
                    makeOp(b, loc, "task_ptr_store",
                           {localBufs[l].v, off.v, scoreMask.v, val.v}, {},
                           {b.getNamedAttr(
                                "ptr_type",
                                b.getStringAttr("!weft_kernel.pointer<f32>")),
                            b.getNamedAttr("val_type", b.getStringAttr("f32"))});
                  }
                }
              }
              return success();
            }
          }
        }
        if (branch->hasAttr("cpu.window_branch")) {
          if (auto cmpOp = branch.getCondition().getDefiningOp<arith::CmpIOp>()) {
            if (auto subOp = cmpOp.getLhs().getDefiningOp<arith::SubIOp>()) {
              auto hiC = staticIndexVal(fn, subOp.getLhs());
              auto loC = staticIndexVal(fn, subOp.getRhs());
              auto winC = staticIndexVal(fn, cmpOp.getRhs());
              if (loC && hiC && winC && (*hiC - *loC) <= *winC) {
                auto yieldOp = branch.thenYield();
                yieldOp->remove();
                LogicalResult st =
                    lowerBlock(*branch.thenBlock(), activeLanes, laneStart);
                branch.thenBlock()->push_back(yieldOp);
                if (failed(st))
                  return failure();
                for (int64_t l = laneStart; l < laneStart + activeLanes; ++l) {
                  for (auto [ret, yVal] :
                       llvm::zip(branch.getResults(), yieldOp.getOperands())) {
                    lanes[l][ret] = lanes[l].at(yVal);
                  }
                }
                return success();
              }
            }
          }
        }
        bool sharedCond = activeLanes > 1;
        Value c0 = lanes[laneStart].at(branch.getCondition()).v;
        for (int64_t l = laneStart + 1; l < laneStart + activeLanes; ++l)
          if (lanes[l].at(branch.getCondition()).v != c0)
            sharedCond = false;

        auto emitIfGroup = [&](int64_t grpStart,
                               int64_t grpCount) -> LogicalResult {
          SmallVector<Attribute> resTys;
          SmallVector<Type> outTypes;
          SmallVector<std::string> resTyStrs;
          for (int64_t l = grpStart; l < grpStart + grpCount; ++l) {
            for (Value r : branch.getResults()) {
              std::string ty = isa<RankedTensorType>(r.getType())
                                   ? vectorType(r)
                                   : scalarType(r.getType());
              resTys.push_back(b.getStringAttr(ty));
              resTyStrs.push_back(ty);
              outTypes.push_back(b.getIndexType());
            }
          }
          Value condVal = lanes[grpStart].at(branch.getCondition()).v;
          Operation *ifOp =
              makeOp(b, loc, "task_if", {condVal}, outTypes,
                     {b.getNamedAttr("res_types", b.getArrayAttr(resTys))}, 2);
          Block &thenBb = ifOp->getRegion(0).emplaceBlock();
          {
            OpBuilder::InsertionGuard g(b);
            b.setInsertionPointToEnd(&thenBb);
            if (failed(lowerBlock(*branch.thenBlock(), grpCount, grpStart)))
              return failure();
            if (thenBb.empty() || thenBb.back().getName().getStringRef() !=
                                      "weft_host.task_yield") {
              makeOp(b, loc, "task_yield", {}, {},
                     {b.getNamedAttr("val_types", b.getArrayAttr({}))});
            }
          }
          if (!branch.getElseRegion().empty()) {
            Block &elseBb = ifOp->getRegion(1).emplaceBlock();
            OpBuilder::InsertionGuard g(b);
            b.setInsertionPointToEnd(&elseBb);
            if (failed(lowerBlock(*branch.elseBlock(), grpCount, grpStart)))
              return failure();
            if (elseBb.empty() || elseBb.back().getName().getStringRef() !=
                                      "weft_host.task_yield") {
              makeOp(b, loc, "task_yield", {}, {},
                     {b.getNamedAttr("val_types", b.getArrayAttr({}))});
            }
          }
          unsigned idxRes = 0;
          for (int64_t l = grpStart; l < grpStart + grpCount; ++l) {
            for (Value r : branch.getResults()) {
              lanes[l][r] = {ifOp->getResult(idxRes), resTyStrs[idxRes]};
              ++idxRes;
            }
          }
          return success();
        };

        if (sharedCond) {
          if (failed(emitIfGroup(laneStart, activeLanes)))
            return failure();
        } else {
          for (int64_t l = laneStart; l < laneStart + activeLanes; ++l)
            if (failed(emitIfGroup(l, 1)))
              return failure();
        }
        return success();
      }

      if (name == "cpu.buffer_store") {
        for (int64_t l = laneStart; l < laneStart + activeLanes; ++l) {
          TaskTerm val = lanes[l].at(op.getOperand(0));
          TaskTerm off = castOp(lanes[l].at(op.getOperand(2)), "si64");
          makeOp(b, loc, "task_ptr_store",
                 {localBufs[l].v, off.v, scoreMask.v, val.v}, {},
                 {b.getNamedAttr("ptr_type",
                                 b.getStringAttr("!weft_kernel.pointer<f32>")),
                  b.getNamedAttr("val_type", b.getStringAttr("f32"))});
        }
        return success();
      }

      if (name == "cpu.buffer_load") {
        for (int64_t l = laneStart; l < laneStart + activeLanes; ++l) {
          TaskTerm off = castOp(lanes[l].at(op.getOperand(1)), "si64");
          Value ld =
              makeOp(b, loc, "task_ptr_load",
                     {localBufs[l].v, off.v, scoreMask.v, scoreZero.v},
                     {b.getIndexType()},
                     {b.getNamedAttr("ptr_type", b.getStringAttr(
                                                     "!weft_kernel.pointer<f32>")),
                      b.getNamedAttr("val_type", b.getStringAttr("f32"))})
                  ->getResult(0);
          lanes[l][op.getResult(0)] = {ld, "f32"};
        }
        return success();
      }

      if (kind == "slice_write") {
        for (int64_t l = laneStart; l < laneStart + activeLanes; ++l) {
          if (op.hasAttr("weft_host.member_store")) {
            auto plan = *scaledMemberStore(&op);
            TaskTerm lo = idx(0);
            TaskTerm hi = idx(
                intentdsl::kir::memberExtent(fn, plan.vector, "cpu.shape"));
            TaskTerm st = idx(1);
            Operation *fOp =
                makeOp(b, loc, "task_for", {lo.v, hi.v, st.v}, {},
                       {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
            Block &fBb = fOp->getRegion(0).emplaceBlock();
            Value iv = fBb.addArgument(b.getIndexType(), loc);
            {
              OpBuilder::InsertionGuard g(b);
              b.setInsertionPointToEnd(&fBb);
              TaskTerm vecT = lanes[l].at(plan.vector);
              Value ex =
                  makeOp(b, loc, "task_extract", {vecT.v, iv},
                         {b.getIndexType()},
                         {b.getNamedAttr("selector", b.getStringAttr("index")),
                          b.getNamedAttr("in_type", b.getStringAttr(vecT.type)),
                          b.getNamedAttr("point_type", b.getStringAttr("index")),
                          b.getNamedAttr("val_type", b.getStringAttr("f32"))})
                      ->getResult(0);
              TaskTerm scaled =
                  bin("mul", {ex, "f32"}, lanes[l].at(plan.scale), "f32");
              Value base = op.getOperand(1);
              std::string dstTy =
                  scalarType(cast<MemRefType>(base.getType()).getElementType());
              TaskTerm dst = castOp(scaled, dstTy);
              int64_t slotAx = op.getAttrOfType<IntegerAttr>("axis").getInt();
              SmallVector<Value> slOps{lanes[l].at(base).v};
              for (Value c : op.getOperands().drop_front(2))
                slOps.push_back(lanes[l].at(c).v);
              slOps.insert(slOps.begin() + 1 + slotAx, iv);
              std::string enc = carrierType(
                  cast<MemRefType>(base.getType()).getElementType());
              SmallVector<Attribute> sels(slOps.size() - 1,
                                          b.getStringAttr("index"));
              std::string slTy =
                  "!weft_kernel.slice<!weft_kernel.encoding<\"" + enc +
                  "\",\"dense\",\"dense." + enc + "\",[]>,[],[]>";
              Value addr =
                  makeOp(
                      b, loc, "task_slice", slOps, {b.getIndexType()},
                      {b.getNamedAttr("view_type",
                                      b.getStringAttr(lanes[l].at(base).type)),
                       b.getNamedAttr("selectors", b.getArrayAttr(sels)),
                       b.getNamedAttr("slice_type", b.getStringAttr(slTy)),
                       b.getNamedAttr("val_type", b.getStringAttr(slTy)),
                       b.getNamedAttr("admit", b.getBoolAttr(false))})
                      ->getResult(0);
              makeOp(b, loc, "task_commit", {dst.v, addr}, {},
                     {b.getNamedAttr("val_type", b.getStringAttr(dst.type)),
                      b.getNamedAttr("slice_type", b.getStringAttr(slTy))});
              makeOp(b, loc, "task_yield", {}, {},
                     {b.getNamedAttr("val_types", b.getArrayAttr({}))});
            }
            continue;
          }
          TaskTerm val = lanes[l].at(op.getOperand(0));
          TaskTerm sl = sliceView(
              l, op.getOperand(1), op.getOperands().drop_front(2),
              op.getAttrOfType<IntegerAttr>("axis").getInt(), false);
          makeOp(b, loc, "task_commit", {val.v, sl.v}, {},
                 {b.getNamedAttr("val_type", b.getStringAttr(val.type)),
                  b.getNamedAttr("slice_type", b.getStringAttr(sl.type))});
        }
        return success();
      }

      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        for (int64_t l = laneStart; l < laneStart + activeLanes; ++l) {
          TaskTerm val = lanes[l].at(store.getValue());
          TaskTerm sl =
              sliceView(l, store.getMemref(), store.getIndices(), -1, false);
          makeOp(b, loc, "task_commit", {val.v, sl.v}, {},
                 {b.getNamedAttr("val_type", b.getStringAttr(val.type)),
                  b.getNamedAttr("slice_type", b.getStringAttr(sl.type))});
        }
        return success();
      }

      bool isCommonQuotient = isa<arith::FloorDivSIOp>(op) &&
                              op.hasAttr("cpu.owner_common") &&
                              activeLanes == batch;
      bool emitOnce = activeLanes > 1 && op.hasAttr("cpu.owner_common") &&
                      (operandsUniform(op) || isCommonQuotient);
      int64_t count = emitOnce ? 1 : activeLanes;
      for (int64_t stepIdx = 0; stepIdx < count; ++stepIdx) {
        int64_t l = laneStart + stepIdx;
        auto val = [&](unsigned i) { return lanes[l].at(op.getOperand(i)); };
        TaskTerm res;
        if (kind == "slice_read") {
          std::string vTy = vectorType(op.getResult(0));
          res = sliceView(l, op.getOperand(0), op.getOperands().drop_front(),
                          op.getAttrOfType<IntegerAttr>("axis").getInt(), true,
                          vTy);
        } else if (kind == "lane_splat") {
          std::string vTy = vectorType(op.getResult(0));
          TaskTerm s = val(0);
          Value v = makeOp(b, loc, "task_new", {s.v}, {b.getIndexType()},
                           {b.getNamedAttr("in_type", b.getStringAttr(s.type)),
                            b.getNamedAttr("val_type", b.getStringAttr(vTy))})
                        ->getResult(0);
          res = {v, vTy};
        } else if (kind == "lane_sum") {
          TaskTerm in = val(0);
          Value v =
              makeOp(b, loc, "task_reduce", {in.v}, {b.getIndexType()},
                     {b.getNamedAttr("kind", b.getStringAttr("add")),
                      b.getNamedAttr("axis", b.getI64IntegerAttr(0)),
                      b.getNamedAttr("in_type", b.getStringAttr(in.type)),
                      b.getNamedAttr("val_type", b.getStringAttr("f32"))})
                  ->getResult(0);
          res = {v, "f32"};
        } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
          res = sliceView(l, load.getMemref(), load.getIndices(), -1, true,
                          scalarType(load.getType()));
        } else if (auto d = dyn_cast<memref::DimOp>(op)) {
          res = idx(dimOf(d.getSource(), "cpu.shape", *d.getConstantIndex()));
        } else if (auto c = dyn_cast<arith::ConstantOp>(op)) {
          if (c.getType().isIndex())
            res = idx(cast<IntegerAttr>(c.getValue()).getInt());
          else if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
            res = lit(c.getType().isInteger(1)
                          ? (ia.getInt() ? "true" : "false")
                          : std::to_string(ia.getInt()),
                      scalarType(c.getType()));
          else
            res = lit(stripTypeSuffix(c.getValue()), scalarType(c.getType()));
        } else if (name == "arith.extf" || name == "arith.truncf" ||
                   name == "arith.index_cast" || name == "arith.sitofp" ||
                   name == "arith.fptosi") {
          std::string dstTy = isa<RankedTensorType>(op.getResult(0).getType())
                                  ? vectorType(op.getResult(0))
                                  : scalarType(op.getResult(0).getType());
          res = castOp(val(0), dstTy);
        } else if (name == "arith.maxnumf" || name == "arith.maximumf") {
          res = bin("max", val(0), val(1), val(0).type);
        } else if (name == "arith.minnumf" || name == "arith.minimumf") {
          res = bin("min", val(0), val(1), val(0).type);
        } else if (name == "arith.select") {
          if (knownTrueConds.contains(op.getOperand(0))) {
            res = val(1);
          } else {
            TaskTerm cT = val(0), tT = val(1), fT = val(2);
            std::string dstTy = isa<RankedTensorType>(op.getResult(0).getType())
                                    ? vectorType(op.getResult(0))
                                    : scalarType(op.getResult(0).getType());
            res = sel(cT, tT, fT, dstTy);
          }
        } else if (name == "math.exp2" && isRvv) {
          TaskTerm in = val(0);
          TaskTerm scaled =
              bin("mul", in, lit("0.693147182", "f32"), in.type);
          Value v =
              makeOp(b, loc, "task_unary", {scaled.v}, {b.getIndexType()},
                     {b.getNamedAttr("kind", b.getStringAttr("exp")),
                      b.getNamedAttr("val_type", b.getStringAttr(in.type))})
                  ->getResult(0);
          res = {v, in.type};
        } else if (name == "math.exp" || name == "math.sqrt" ||
                   name == "math.rsqrt" || name == "math.exp2" ||
                   name == "math.log") {
          TaskTerm in = val(0);
          Value v =
              makeOp(b, loc, "task_unary", {in.v}, {b.getIndexType()},
                     {b.getNamedAttr("kind", b.getStringAttr(kind)),
                      b.getNamedAttr("val_type", b.getStringAttr(in.type))})
                  ->getResult(0);
          res = {v, in.type};
        } else if (auto cmpOp = dyn_cast<arith::CmpIOp>(op)) {
          if (knownTrueConds.contains(op.getResult(0))) {
            res = lit("true", "i1");
          } else {
            TaskTerm a = val(0), r = val(1);
            auto p = cmpOp.getPredicate();
            if (a.type == "index" && p != arith::CmpIPredicate::eq &&
                p != arith::CmpIPredicate::ne) {
              a = castOp(a, "si64");
              r = castOp(r, "si64");
            }
            StringRef pred = "eq";
            switch (p) {
            case arith::CmpIPredicate::eq:
              pred = "eq";
              break;
            case arith::CmpIPredicate::ne:
              pred = "ne";
              break;
            case arith::CmpIPredicate::slt:
            case arith::CmpIPredicate::ult:
              pred = "lt";
              break;
            case arith::CmpIPredicate::sle:
            case arith::CmpIPredicate::ule:
              pred = "le";
              break;
            case arith::CmpIPredicate::sgt:
            case arith::CmpIPredicate::ugt:
              pred = "gt";
              break;
            case arith::CmpIPredicate::sge:
            case arith::CmpIPredicate::uge:
              pred = "ge";
              break;
            }
            res = cmp(pred, a, r, "i1");
          }
        } else if (auto fcmp = dyn_cast<arith::CmpFOp>(op)) {
          StringRef pred = "eq";
          switch (fcmp.getPredicate()) {
          case arith::CmpFPredicate::OEQ:
          case arith::CmpFPredicate::UEQ:
            pred = "eq";
            break;
          case arith::CmpFPredicate::ONE:
          case arith::CmpFPredicate::UNE:
            pred = "ne";
            break;
          case arith::CmpFPredicate::OLT:
          case arith::CmpFPredicate::ULT:
            pred = "lt";
            break;
          case arith::CmpFPredicate::OLE:
          case arith::CmpFPredicate::ULE:
            pred = "le";
            break;
          case arith::CmpFPredicate::OGT:
          case arith::CmpFPredicate::UGT:
            pred = "gt";
            break;
          default:
            pred = "ge";
            break;
          }
          std::string cTy = isa<RankedTensorType>(op.getResult(0).getType())
                                ? vectorType(op.getResult(0))
                                : std::string("i1");
          res = cmp(pred, val(0), val(1), cTy);
        } else {
          auto tok = llvm::StringSwitch<StringRef>(name)
                         .Cases("arith.addf", "arith.addi", "add")
                         .Cases("arith.subf", "arith.subi", "sub")
                         .Cases("arith.mulf", "arith.muli", "mul")
                         .Cases("arith.divf", "arith.floordivsi",
                                "arith.divsi", "div")
                         .Case("arith.remsi", "mod")
                         .Case("arith.andi", "and")
                         .Case("arith.ori", "or")
                         .Case("arith.xori", "xor")
                         .Default("");
          if (tok.empty())
            return op.emitError("unhandled ordered Weft instruction ") << name;
          res = lowerBinary(*this, op, tok, val(0), val(1), val(0).type);
        }

        if (emitOnce) {
          for (int64_t dstL = laneStart; dstL < laneStart + activeLanes; ++dstL)
            lanes[dstL][op.getResult(0)] = res;
        } else {
          lanes[l][op.getResult(0)] = res;
        }
      }
    return success();
  }
};
} // namespace

LogicalResult bindWeftMemberStores(Operation &task, bool rvv, bool assign) {
  auto result = task.walk([&](Operation *op) {
    bool use = rvv && scaledMemberStore(op).has_value();
    auto attr = StringAttr::get(op->getContext(), "scale-narrow-store");
    if (assign && use)
      op->setAttr("weft_host.member_store", attr);
    if (use ? op->getAttr("weft_host.member_store") != attr
            : op->hasAttr("weft_host.member_store")) {
      op->emitError(
          "lane publication requires scaled and narrowed member values");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

LogicalResult lowerWeftOrdered(func::FuncOp fn,
                               const intentdsl::CallFacts &facts, bool rvv) {
  auto &task = fn.front().front();
  auto &space = task.getRegion(0).front().front();
  Location loc = fn.getLoc();
  OpBuilder b(&task);

  auto abiAttr = fn->getAttrOfType<StringAttr>("kir.abi");
  auto parsedAbi = llvm::json::parse(abiAttr.getValue());
  if (!parsedAbi)
    return fn.emitError() << llvm::toString(parsedAbi.takeError());
  const auto &abi = *parsedAbi->getAsArray();

  int64_t owners = space.getAttrOfType<IntegerAttr>("cpu.owners").getInt();
  int64_t grain = space.getAttrOfType<IntegerAttr>("cpu.grain").getInt();
  auto bAttr = space.getAttrOfType<IntegerAttr>("cpu.owner_batch");
  if (!bAttr || bAttr.getInt() < 1)
    return space.emitError("missing cpu.owner_batch decision from cpu@tasked");
  int64_t batch = bAttr.getInt();

  OrderedLowerer lowerer{fn, space, b, loc, batch};
  lowerer.isRvv = rvv;
  lowerer.lanes.resize(batch);
  {
    std::optional<TightenedLoopBound> stBound;
    space.walk([&](scf::ForOp loop) {
      auto t = analyzeCausalLoop(fn, loop);
      if (t && t->hasBufferStore)
        stBound = t;
    });
    if (stBound) {
      space.walk([&](Operation *op) {
        if (op->getName().getStringRef() != "cpu.buffer_load")
          return;
        auto ldLoop = op->getParentOfType<scf::ForOp>();
        auto ldTight = ldLoop ? analyzeCausalLoop(fn, ldLoop) : std::nullopt;
        if (!ldTight || ldTight->boundVal != stBound->boundVal ||
            ldTight->inclusive != stBound->inclusive)
          lowerer.allowBufferStoreTightening = false;
      });
    }
  }

  SmallVector<Attribute> viewTypes, argAccess, shapeSyms, slotArgs, strideExprs;
  SmallVector<int64_t> argAliases, stridedSlots, contiguousSlots, taskRanks;
  SmallVector<std::string> viewTypeStrs;
  bool emptyExtent = false;
  std::map<std::pair<std::string, unsigned>, int64_t> dimNames;

  for (auto [i, v] : llvm::enumerate(fn.getArguments())) {
    std::string aName = "a" + std::to_string(i);
    slotArgs.push_back(b.getStringAttr(aName));
    auto view = dyn_cast<MemRefType>(v.getType());
    int64_t rank = view ? view.getRank() : 0;
    taskRanks.push_back(rank);
    std::string elem = view ? OrderedLowerer::carrierType(view.getElementType())
                            : OrderedLowerer::scalarType(v.getType());
    auto dir = abi[i].getAsObject()->getString("direction");
    argAliases.push_back(rank ? 0 : -1);
    argAccess.push_back(b.getStringAttr(!rank            ? "none"
                                        : dir == "Out"   ? "write"
                                        : dir == "InOut" ? "readwrite"
                                                         : "read"));
    std::string ty = elem;
    if (rank) {
      auto shAttr = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "cpu.shape");
      auto stAttr = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "cpu.strides");
      bool dense = shAttr && stAttr && shAttr.size() == stAttr.size();
      int64_t step = 1;
      if (dense) {
        for (int k = int(shAttr.size()) - 1; k >= 0; --k) {
          int64_t n = shAttr[k], s = stAttr[k];
          if (n <= 0 || (n > 1 && s != step) || step > INT64_MAX / n) {
            dense = false;
            break;
          }
          step *= n;
        }
      }
      SmallVector<std::string> shape, axes;
      std::map<std::string, unsigned> occurrences;
      auto *labels = abi[i].getAsObject()->getArray("dims");
      for (int64_t j = 0; j < rank; ++j) {
        std::string label;
        llvm::raw_string_ostream(label) << (*labels)[j];
        auto [it, added] = dimNames.try_emplace({label, occurrences[label]++},
                                                int64_t(dimNames.size()) + 1);
        lowerer.axesMap[v].push_back(it->second);
        int64_t extent = lowerer.dimOf(v, "cpu.shape", j);
        emptyExtent |= extent == 0;
        shape.push_back(extent ? std::to_string(extent) : "-3");
        axes.push_back(std::to_string(it->second));
        if (!dense)
          strideExprs.push_back(
              b.getStringAttr(aName + "s" + std::to_string(j)));
      }
      ty = "!weft_kernel.view<!weft_kernel.encoding<\"" + elem +
           "\",\"dense\",\"dense." + elem + "\",[]>,[" +
           llvm::join(shape, ",") + "],[" + llvm::join(axes, ",") + "]>";
      if (dense)
        contiguousSlots.push_back(int64_t(i));
      else
        stridedSlots.push_back(int64_t(i));
    }
    viewTypes.push_back(b.getStringAttr(ty));
    viewTypeStrs.push_back(ty);
  }

  shapeSyms.push_back(b.getStringAttr("LO"));
  shapeSyms.push_back(b.getStringAttr("HI"));
  if (emptyExtent)
    shapeSyms.push_back(b.getStringAttr("EMPTY"));

  OperationState progSt(loc, "weft_host.owner_program");
  progSt.addAttribute("owners", b.getI64IntegerAttr(owners));
  progSt.addAttribute("grain", b.getI64IntegerAttr(grain));
  progSt.addAttribute("shape_parameters", b.getArrayAttr(shapeSyms));
  progSt.addAttribute("contiguous_slots",
                      b.getDenseI64ArrayAttr(contiguousSlots));
  progSt.addAttribute("task_ranks", b.getDenseI64ArrayAttr(taskRanks));
  progSt.addRegion();
  progSt.addRegion();
  Operation *prog = b.create(progSt);

  // Region 0: task.weftir
  Block &tOuter = prog->getRegion(0).emplaceBlock();
  b.setInsertionPointToEnd(&tOuter);
  Operation *tk = makeOp(
      b, loc, "task_kernel", {}, {},
      {b.getNamedAttr("sym_name", b.getStringAttr("owner_task")),
       b.getNamedAttr("source", b.getStringAttr("independent ordered owners")),
       b.getNamedAttr("view_types", b.getArrayAttr(viewTypes)),
       b.getNamedAttr("arg_access", b.getArrayAttr(argAccess)),
       b.getNamedAttr("arg_aliases", b.getDenseI64ArrayAttr(argAliases)),
       b.getNamedAttr("shape_symbols", b.getArrayAttr(shapeSyms)),
       b.getNamedAttr("strided_slots", b.getDenseI64ArrayAttr(stridedSlots))},
      1);
  makeOp(b, loc, "end");

  Block &tkBb = tk->getRegion(0).emplaceBlock();
  for (auto [i, v] : llvm::enumerate(fn.getArguments())) {
    Value bArg = tkBb.addArgument(b.getIndexType(), loc);
    for (int64_t l = 0; l < batch; ++l)
      lowerer.lanes[l][v] = {bArg, viewTypeStrs[i]};
  }
  Value symLO = tkBb.addArgument(b.getIndexType(), loc);
  Value symHI = tkBb.addArgument(b.getIndexType(), loc);
  if (emptyExtent)
    tkBb.addArgument(b.getIndexType(), loc);
  tkBb.addArgument(b.getIndexType(), loc); // root_domain

  b.setInsertionPointToEnd(&tkBb);
  TaskTerm stepTerm = lowerer.idx(batch);

  auto &srcBb = space.getRegion(0).front();
  Operation *localBufOp = nullptr;
  for (Operation &op : srcBb) {
    if (op.getName().getStringRef() == "cpu.local_buffer") {
      localBufOp = &op;
      break;
    }
  }
  TiledTwoPassPlan tiledPlan;
  bool useTiled2D =
      lowerer.matchTiledTwoPassContraction(srcBb, localBufOp, tiledPlan);
  TaskTerm iotaOwn64, iotaTok64, qSlotOff, tileSlotOff;
  if (useTiled2D) {
    if (tiledPlan.hasCausal)
      lowerer.anyTightenedCausalLoop = true;
    TaskTerm oneIdx = lowerer.idx(1);
    lowerer.true1 = lowerer.cmp("eq", oneIdx, oneIdx, "i1");
    lowerer.ninfS = lowerer.lit("0xFF800000", "f32");
    lowerer.zeroS = lowerer.lit("0.0", "f32");
    lowerer.oneS = lowerer.lit("1.0", "f32");
    lowerer.ln2S = lowerer.lit("0.693147182", "f32");
    lowerer.cB64 = lowerer.lit(std::to_string(batch), "si64");
    std::string vecBTy = "!weft_kernel.value<f32,[" + std::to_string(batch) +
                         "],[" + std::to_string(tiledPlan.axOwn) + "]>";
    std::string maskBTy = "!weft_kernel.value<i1,[" + std::to_string(batch) +
                          "],[" + std::to_string(tiledPlan.axOwn) + "]>";
    std::string scoreMatTy =
        "!weft_kernel.value<f32,[" + std::to_string(batch) + "," +
        std::to_string(tiledPlan.tokenTile) + "],[" +
        std::to_string(tiledPlan.axOwn) + "," +
        std::to_string(tiledPlan.axTok) + "]>";
    std::string maskMatTy =
        "!weft_kernel.value<i1,[" + std::to_string(batch) + "," +
        std::to_string(tiledPlan.tokenTile) + "],[" +
        std::to_string(tiledPlan.axOwn) + "," +
        std::to_string(tiledPlan.axTok) + "]>";
    std::string outMatTy =
        "!weft_kernel.value<f32,[" + std::to_string(batch) + "," +
        std::to_string(tiledPlan.DV) + "],[" +
        std::to_string(tiledPlan.axOwn) + "," +
        std::to_string(tiledPlan.axDV) + "]>";
    const bool needTiledOut = tiledPlan.valueWidth < tiledPlan.DV;
    lowerer.ninfV = lowerer.newVec(lowerer.ninfS, vecBTy);
    if (tiledPlan.hasCausal)
      lowerer.ninfM = lowerer.newVec(lowerer.ninfS, scoreMatTy);
    lowerer.zeroV = lowerer.newVec(lowerer.zeroS, vecBTy);
    lowerer.zeroM = lowerer.newVec(lowerer.zeroS, scoreMatTy);
    if (!needTiledOut)
      lowerer.zeroO = lowerer.newVec(lowerer.zeroS, outMatTy);
    if (tiledPlan.hasScoreBuffer) {
      lowerer.trueM = lowerer.newVec(lowerer.true1, maskMatTy);
      lowerer.trueB = lowerer.newVec(lowerer.true1, maskBTy);
    }
    std::string idxOwnTy =
        "!weft_kernel.value<index,[" + std::to_string(batch) + "],[" +
        std::to_string(tiledPlan.axOwn) + "]>";
    std::string si64OwnTy =
        "!weft_kernel.value<si64,[" + std::to_string(batch) + "],[" +
        std::to_string(tiledPlan.axOwn) + "]>";
    std::string idxTokTy =
        "!weft_kernel.value<index,[" + std::to_string(tiledPlan.tokenTile) +
        "],[" + std::to_string(tiledPlan.axTok) + "]>";
    std::string si64TokTy =
        "!weft_kernel.value<si64,[" + std::to_string(tiledPlan.tokenTile) +
        "],[" + std::to_string(tiledPlan.axTok) + "]>";
    Value ioIdx =
        makeOp(b, loc, "task_iota", {}, {b.getIndexType()},
               {b.getNamedAttr("start", b.getI64IntegerAttr(0)),
                b.getNamedAttr("end", b.getI64IntegerAttr(batch)),
                b.getNamedAttr("val_type", b.getStringAttr(idxOwnTy))})
            ->getResult(0);
    iotaOwn64 = lowerer.castOp({ioIdx, idxOwnTy}, si64OwnTy);
    Value itIdx =
        makeOp(b, loc, "task_iota", {}, {b.getIndexType()},
               {b.getNamedAttr("start", b.getI64IntegerAttr(0)),
                b.getNamedAttr("end", b.getI64IntegerAttr(tiledPlan.tokenTile)),
                b.getNamedAttr("val_type", b.getStringAttr(idxTokTy))})
            ->getResult(0);
    iotaTok64 = lowerer.castOp({itIdx, idxTokTy}, si64TokTy);
    if (tiledPlan.hasScoreBuffer) {
      int64_t allocSlots = tiledPlan.scoreSlots;
      auto loC = staticIndexVal(fn, tiledPlan.loop1.getLowerBound());
      auto hiMax = maxStaticIndexVal(fn, tiledPlan.loop1.getUpperBound());
      bool staticFit = loC && hiMax && (*hiMax - *loC) <= tiledPlan.scoreSlots;
      if (staticFit)
        allocSlots = std::max<int64_t>(tiledPlan.tokenTile, *hiMax - *loC);
      Value ptr =
          makeOp(b, loc, "task_local_alloc", {}, {b.getIndexType()},
                 {b.getNamedAttr("elements",
                                 b.getI64IntegerAttr(batch * allocSlots)),
                  b.getNamedAttr("alignment", b.getI64IntegerAttr(64)),
                  b.getNamedAttr("ptr_type",
                                 b.getStringAttr("!weft_kernel.pointer<f32>"))})
              ->getResult(0);
      lowerer.tiledScoreBuf = {ptr, "!weft_kernel.pointer<f32>"};
      qSlotOff = iotaOwn64;
      TaskTerm cT64 =
          lowerer.lit(std::to_string(tiledPlan.tokenTile), "si64");
      TaskTerm qTileOff = lowerer.bin("mul", iotaOwn64, cT64, si64OwnTy);
      std::string si64MatTy =
          "!weft_kernel.value<si64,[" + std::to_string(batch) + "," +
          std::to_string(tiledPlan.tokenTile) + "],[" +
          std::to_string(tiledPlan.axOwn) + "," +
          std::to_string(tiledPlan.axTok) + "]>";
      tileSlotOff = lowerer.bin("add", qTileOff, iotaTok64, si64MatTy);
    }
    if (needTiledOut) {
      Value outPtr =
          makeOp(b, loc, "task_local_alloc", {}, {b.getIndexType()},
                 {b.getNamedAttr("elements",
                                 b.getI64IntegerAttr(batch * tiledPlan.DV)),
                  b.getNamedAttr("alignment", b.getI64IntegerAttr(64)),
                  b.getNamedAttr("ptr_type",
                                 b.getStringAttr("!weft_kernel.pointer<f32>"))})
              ->getResult(0);
      lowerer.tiledOutBuf = {outPtr, "!weft_kernel.pointer<f32>"};
    }
  } else if (localBufOp) {
    int64_t slots =
        localBufOp->getAttrOfType<IntegerAttr>("elements").getInt();
    TaskTerm oneIdx = lowerer.idx(1);
    lowerer.scoreZero = lowerer.lit("0.0", "f32");
    lowerer.scoreMask = lowerer.cmp("eq", oneIdx, oneIdx, "i1");
    lowerer.localBufs.resize(batch);
    for (int64_t l = 0; l < batch; ++l) {
      Value ptr = makeOp(b, loc, "task_local_alloc", {}, {b.getIndexType()},
                         {b.getNamedAttr("elements", b.getI64IntegerAttr(slots)),
                          b.getNamedAttr("alignment", b.getI64IntegerAttr(64)),
                          b.getNamedAttr("ptr_type", b.getStringAttr(
                                                         "!weft_kernel.pointer<f32>"))})
                      ->getResult(0);
      lowerer.localBufs[l] = {ptr, "!weft_kernel.pointer<f32>"};
    }
  }

  Operation *ownerFor =
      makeOp(b, loc, "task_for", {symLO, symHI, stepTerm.v}, {},
             {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
  Block &oBb = ownerFor->getRegion(0).emplaceBlock();
  Value ownerIv = oBb.addArgument(b.getIndexType(), loc);
  b.setInsertionPointToEnd(&oBb);

  auto axesAttr = space.getAttrOfType<DenseI64ArrayAttr>("axes");
  TaskTerm cursor =
      batch > 1 ? lowerer.bin("div", {ownerIv, "index"}, stepTerm, "index")
                : TaskTerm{ownerIv, "index"};
  auto radices = intentdsl::cpu::ownerRadices(space);
  for (auto [pos, ax] : llvm::enumerate(radices)) {
    int64_t fullExt = std::max<int64_t>(
        1, lowerer.dimOf(space.getOperand(ax), "cpu.shape", axesAttr[ax]));
    int64_t radixExt = (pos == 0 && batch > 1) ? (fullExt / batch) : fullExt;
    TaskTerm d = lowerer.idx(radixExt);
    TaskTerm rem = lowerer.bin("mod", cursor, d, "index");
    cursor = lowerer.bin("div", cursor, d, "index");
    TaskTerm baseCoord =
        (pos == 0 && batch > 1) ? lowerer.bin("mul", rem, stepTerm, "index")
                                : rem;
    for (int64_t l = 0; l < batch; ++l) {
      TaskTerm laneCoord = (pos == 0 && batch > 1 && l > 0)
                               ? lowerer.bin("add", baseCoord, lowerer.idx(l),
                                             "index")
                               : baseCoord;
      lowerer.lanes[l][srcBb.getArgument(ax)] = laneCoord;
    }
  }

  LogicalResult bodyStatus =
      useTiled2D ? lowerer.lowerTiledTwoPassBody(srcBb, tiledPlan, iotaOwn64,
                                                 iotaTok64, qSlotOff,
                                                 tileSlotOff)
                 : lowerer.lowerBlock(srcBb, batch, 0);
  if (failed(bodyStatus))
    return failure();
  makeOp(b, loc, "task_yield", {}, {},
         {b.getNamedAttr("val_types", b.getArrayAttr({}))});
  b.setInsertionPointToEnd(&tkBb);
  makeOp(b, loc, "end");

  int64_t targetCausalGrain =
      std::max<int64_t>((owners >= 2048 ? 64 : 16), batch);
  int64_t hostGrain =
      (lowerer.anyTightenedCausalLoop && grain > targetCausalGrain &&
       grain % targetCausalGrain == 0)
          ? targetCausalGrain
          : grain;
  prog->setAttr("grain", b.getI64IntegerAttr(hostGrain));
  // Region 1: host.c
  Block &hOuter = prog->getRegion(1).emplaceBlock();
  b.setInsertionPointToEnd(&hOuter);
  Operation *hFn = makeOp(
      b, loc, "host_owner_fn", {}, {},
      {b.getNamedAttr("params",
                      b.getStringAttr(intentdsl::cpu::matrixHostParams(fn))),
       b.getNamedAttr("owners", b.getI64IntegerAttr(owners)),
       b.getNamedAttr("grain", b.getI64IntegerAttr(hostGrain))},
      1);
  makeOp(b, loc, "end");

  Block &hFnBb = hFn->getRegion(0).emplaceBlock();
  b.setInsertionPointToEnd(&hFnBb);
  bool dynamic = lowerer.anyTightenedCausalLoop;
  if (dynamic)
    makeOp(b, loc, "host_let_i64", {}, {},
           {b.getNamedAttr("name", b.getStringAttr("chunks")),
            b.getNamedAttr("expr",
                           b.getStringAttr("(count + tile - 1) / tile"))});
  StringRef parallelIf =
      dynamic ? "workers > 1 && chunks > 1" : "workers > 1 && count > tile";
  // The single-instance ABI bypasses Weft's grid launcher. End the same
  // thread-local oneDNN context before returning an OpenMP worker to its pool.
  bool managed = tkBb.walk([](Operation *op) {
    auto engine = op->getAttrOfType<StringAttr>("engine");
    return op->getName().getStringRef() == "weft_host.task_contract" &&
                   engine && engine.getValue() != "vector"
               ? WalkResult::interrupt() : WalkResult::advance();
  }).wasInterrupted();
  if (managed) {
    hFn->setAttr("checked_tasks", b.getUnitAttr());
    auto *team = makeOp(
        b, loc, "host_omp_parallel", {}, {},
        {b.getNamedAttr("release_hw_ctx", b.getBoolAttr(true)),
         b.getNamedAttr("num_threads", b.getStringAttr("workers")),
         b.getNamedAttr("if_cond", b.getStringAttr(parallelIf))},
        1);
    makeOp(b, loc, "end");
    b.setInsertionPointToStart(&team->getRegion(0).emplaceBlock());
  }
  Operation *pOwners = makeOp(
      b, loc, "host_omp_for", {}, {},
      {b.getNamedAttr("iv", b.getStringAttr(dynamic ? "ch_i" : "lo_i")),
       b.getNamedAttr("lb", b.getStringAttr("0")),
       b.getNamedAttr("ub", b.getStringAttr(dynamic ? "chunks" : "count")),
       b.getNamedAttr("step", b.getStringAttr(dynamic ? "1" : "tile")),
       b.getNamedAttr("parallel", b.getBoolAttr(!managed)),
       b.getNamedAttr("simd", b.getBoolAttr(false)),
       b.getNamedAttr("x86_simd", b.getBoolAttr(false)),
       b.getNamedAttr("schedule", b.getStringAttr(dynamic ? "dynamic, 1" : "static, 1")),
       b.getNamedAttr("num_threads", b.getStringAttr(managed ? "" : "workers")),
       b.getNamedAttr("if_cond", b.getStringAttr(managed ? "" : parallelIf))},
      1);
  if (managed)
    makeOp(b, loc, "host_collect_task_status");
  makeOp(b, loc, "end");

  Block &pBb = pOwners->getRegion(0).emplaceBlock();
  b.setInsertionPointToEnd(&pBb);
  if (lowerer.anyTightenedCausalLoop) {
    makeOp(b, loc, "host_let_i64", {}, {},
           {b.getNamedAttr("name", b.getStringAttr("lo_i")),
            b.getNamedAttr("expr", b.getStringAttr("ch_i * tile"))});
  }
  Value loVal =
      makeOp(b, loc, "host_def_i64", {}, {b.getIndexType()},
             {b.getNamedAttr("expr", b.getStringAttr("lo_i"))})
          ->getResult(0);
  Value hiVal =
      makeOp(b, loc, "host_def_i64", {}, {b.getIndexType()},
             {b.getNamedAttr(
                 "expr",
                 b.getStringAttr("count - lo_i < tile ? count : lo_i + tile"))})
          ->getResult(0);
  makeOp(b, loc, "host_call_owner_task", {loVal, hiVal}, {},
         {b.getNamedAttr("sym_name", b.getStringAttr("owner_task")),
          b.getNamedAttr("slot_args", b.getArrayAttr(slotArgs)),
          b.getNamedAttr("empty_extent", b.getBoolAttr(emptyExtent)),
          b.getNamedAttr("stride_exprs", b.getArrayAttr(strideExprs))});
  makeOp(b, loc, "end");

  SmallVector<Operation *> toErase;
  for (Operation &op : fn.front())
    if (&op != prog && op.getName().getStringRef() != "func.return")
      toErase.push_back(&op);
  for (Operation *op : llvm::reverse(toErase))
    op->erase();
  return success();
}

bool hasWeftOwnerProgram(ModuleOp m) {
  auto funcs = m.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs))
    return false;
  auto fn = *funcs.begin();
  return !fn.empty() && !fn.front().empty() &&
         fn.front().front().getName().getStringRef() ==
             "weft_host.owner_program";
}

llvm::Error describeWeftOrderedPackage(func::FuncOp fn, Operation &prog,
                                       llvm::raw_ostream &os) {
  auto abiAttr = fn->getAttrOfType<StringAttr>("kir.abi");
  auto parsedAbi = llvm::json::parse(abiAttr.getValue());
  if (!parsedAbi)
    return parsedAbi.takeError();
  llvm::json::Array shapeParams;
  for (Attribute a : prog.getAttrOfType<ArrayAttr>("shape_parameters"))
    shapeParams.push_back(cast<StringAttr>(a).str());
  llvm::json::Array ranks;
  for (int64_t r :
       prog.getAttrOfType<DenseI64ArrayAttr>("task_ranks").asArrayRef())
    ranks.push_back(r);
  llvm::json::Array compact;
  for (int64_t c :
       prog.getAttrOfType<DenseI64ArrayAttr>("contiguous_slots").asArrayRef())
    compact.push_back(c);

  int64_t grain = prog.getAttrOfType<IntegerAttr>("grain").getInt();
  llvm::json::Object obj{
      {"version", 3},
      {"abi", std::move(*parsedAbi)},
      {"native_symbol", "owner_task_instance"},
      {"shape_parameters", std::move(shapeParams)},
      {"task_ranks", std::move(ranks)},
      {"domain", intentdsl::publishDomain(fn)},
      {"rows",
       llvm::json::Array{llvm::json::Object{{"grain", grain}, {"columns", 1}}}}};
  if (!compact.empty())
    obj["contiguous_slots"] = std::move(compact);
  os << llvm::json::Value(std::move(obj));
  return llvm::Error::success();
}

LogicalResult verifyWeftOrdered(ModuleOp m, const TargetProfile &p,
                                bool callable) {
  auto fn = *m.getOps<func::FuncOp>().begin();
  auto &prog = fn.front().front();
  if (prog.getNumRegions() != 2 || prog.getRegion(0).empty() ||
      prog.getRegion(1).empty())
    return prog.emitError("incomplete weft_host.owner_program regions");

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
                      op->emitError("32-bit Weft ordered address lacks a "
                                    "proven signed i32 range");
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
      return fn.emitError("missing synchronous Weft ordered host entry");
    llvm::SmallVector<char> expected;
    llvm::raw_svector_ostream os(expected);
    if (auto err = describeWeftOrderedPackage(fn, prog, os))
      return fn.emitError() << llvm::toString(std::move(err)), failure();
    auto pkg = m->getAttrOfType<StringAttr>("weft_host.package");
    if (!pkg || pkg.getValue() != StringRef(expected.data(), expected.size()))
      return fn.emitError(
                 "weft_host.package disagrees with verified ordered ABI"),
             failure();
  }
  return success();
}
} // namespace CI_UNIT_NAMESPACE
