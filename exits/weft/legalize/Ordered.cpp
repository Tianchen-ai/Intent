#include "Ordered.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
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

struct OrderedLowerer {
  func::FuncOp fn;
  Operation &space;
  OpBuilder &b;
  Location loc;
  int64_t batch = 1;
  llvm::DenseMap<Value, SmallVector<int64_t>> axesMap;
  SmallVector<llvm::DenseMap<Value, TaskTerm>, 16> lanes;
  SmallVector<TaskTerm, 16> localBufs;
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
      StringRef name = op.getName().getStringRef();
      StringRef kind = op.getName().stripDialect();
      if (kind == "end" || name == "cpu.local_buffer")
        continue;
      if (narrowed(&op) || multiplied(&op))
        continue;

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
        continue;
      }

      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        TaskTerm lbTerm = lanes[laneStart].at(loop.getLowerBound());
        TaskTerm ubTerm = lanes[laneStart].at(loop.getUpperBound());
        Value lb = lbTerm.v;
        Value ub = ubTerm.v;
        Value st = lanes[laneStart].at(loop.getStep()).v;

        Value simplifiedCmp;
        if (auto tight = analyzeCausalLoop(fn, loop)) {
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
            anyTightenedCausalLoop = true;
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
        continue;
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
          continue;
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
              continue;
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
                continue;
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
        continue;
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
        continue;
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
        continue;
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
        continue;
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
        continue;
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
        } else if (name == "math.exp2" && !isRvv) {
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
          res = bin(tok, val(0), val(1), val(0).type);
        }

        if (emitOnce) {
          for (int64_t dstL = laneStart; dstL < laneStart + activeLanes; ++dstL)
            lanes[dstL][op.getResult(0)] = res;
        } else {
          lanes[l][op.getResult(0)] = res;
        }
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
  int64_t batch = 1;
  if (auto bAttr = space.getAttrOfType<IntegerAttr>("cpu.owner_batch")) {
    batch = bAttr.getInt();
  } else if (!rvv) {
    if (auto scAttr = space.getAttrOfType<IntegerAttr>("cpu.score_cache")) {
      if (auto reuse = intentdsl::cpu::sharedInterval(&space, facts)) {
        int64_t scSlots = scAttr.getInt();
        for (int64_t cand : {4, 2}) {
          if (reuse->group % cand == 0 && reuse->extent % cand == 0 &&
              grain % cand == 0 && owners % cand == 0 &&
              cand * scSlots * 4 <= 32768) {
            batch = cand;
            space.setAttr("cpu.innermost_owner",
                          b.getI64IntegerAttr(reuse->axis));
            space.walk([&](Operation *op) {
              if (op->getNumResults() == 1 &&
                  intentdsl::cpu::invariantOwnerValue(op->getResult(0), space,
                                                      *reuse))
                op->setAttr("cpu.owner_common", b.getUnitAttr());
            });
            break;
          }
        }
      }
    }
  }

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

  Operation *localBufOp = nullptr;
  for (Operation &op : space.getRegion(0).front()) {
    if (op.getName().getStringRef() == "cpu.local_buffer") {
      localBufOp = &op;
      break;
    }
  }
  if (localBufOp) {
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
  auto &srcBb = space.getRegion(0).front();
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

  if (failed(lowerer.lowerBlock(srcBb, batch, 0)))
    return failure();
  makeOp(b, loc, "task_yield", {}, {},
         {b.getNamedAttr("val_types", b.getArrayAttr({}))});
  b.setInsertionPointToEnd(&tkBb);
  makeOp(b, loc, "end");

  int64_t hostGrain =
      (lowerer.anyTightenedCausalLoop && grain > 16 && 16 % batch == 0)
          ? int64_t(16)
          : grain;
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
  Operation *pOwners = makeOp(b, loc, "host_parallel_owners", {}, {}, {}, 1);
  makeOp(b, loc, "end");

  Block &pBb = pOwners->getRegion(0).emplaceBlock();
  Value loVal = pBb.addArgument(b.getIndexType(), loc);
  Value hiVal = pBb.addArgument(b.getIndexType(), loc);
  b.setInsertionPointToEnd(&pBb);
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
