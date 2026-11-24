#include "intentdsl/levels/storage/dsa/ir/Owners.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
namespace intentdsl::dsa {
namespace {
bool dependsImpl(Value v, Value target,
                 llvm::SmallPtrSetImpl<Value> &visited) {
  if (!v)
    return false;
  if (v == target)
    return true;
  if (!visited.insert(v).second)
    return false;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
      if (arg == loop.getInductionVar())
        return dependsImpl(loop.getLowerBound(), target, visited) ||
               dependsImpl(loop.getUpperBound(), target, visited) ||
               dependsImpl(loop.getStep(), target, visited);
      unsigned idx = arg.getArgNumber() - 1;
      return dependsImpl(loop.getInitArgs()[idx], target, visited) ||
             dependsImpl(loop.getBody()->getTerminator()->getOperand(idx),
                         target, visited);
    }
    return false;
  }
  auto *op = v.getDefiningOp();
  if (!op)
    return false;
  if (auto loop = dyn_cast<scf::ForOp>(op)) {
    unsigned idx = cast<OpResult>(v).getResultNumber();
    return dependsImpl(loop.getLowerBound(), target, visited) ||
           dependsImpl(loop.getUpperBound(), target, visited) ||
           dependsImpl(loop.getInitArgs()[idx], target, visited) ||
           dependsImpl(loop.getBody()->getTerminator()->getOperand(idx), target,
                       visited);
  }
  if (auto iff = dyn_cast<scf::IfOp>(op)) {
    unsigned idx = cast<OpResult>(v).getResultNumber();
    if (dependsImpl(iff.getCondition(), target, visited))
      return true;
    if (!iff.getThenRegion().empty() &&
        dependsImpl(iff.thenBlock()->getTerminator()->getOperand(idx), target,
                    visited))
      return true;
    if (!iff.getElseRegion().empty() &&
        dependsImpl(iff.elseBlock()->getTerminator()->getOperand(idx), target,
                    visited))
      return true;
    return false;
  }
  return llvm::any_of(op->getOperands(), [&](Value u) {
    return dependsImpl(u, target, visited);
  });
}

struct IntervalAlloc {
  struct Slot {
    unsigned start, end;
    int64_t offset, size;
  };
  int64_t alignment;
  int64_t peak = 0;
  SmallVector<Slot> slots;

  explicit IntervalAlloc(int64_t align) : alignment(std::max<int64_t>(64, align)) {}

  int64_t alloc(unsigned start, unsigned end, int64_t bytes) {
    int64_t alignedBytes = llvm::alignTo(std::max<int64_t>(bytes, 64), alignment);
    int64_t offset = 0;
    for (const auto &s : slots) {
      if (s.end < start || end < s.start)
        continue;
      offset = llvm::alignTo(offset, alignment);
      if (offset + alignedBytes <= s.offset)
        break;
      offset = std::max<int64_t>(offset, llvm::alignTo(s.offset + s.size, alignment));
    }
    peak = std::max(peak, offset + alignedBytes);
    auto pos = llvm::lower_bound(slots, offset, [](const Slot &s, int64_t key) {
      return s.offset < key;
    });
    slots.insert(pos, {start, end, offset, alignedBytes});
    return offset;
  }
};

Operation *defWithName(Value v, StringRef stripped) {
  auto *op = v ? v.getDefiningOp() : nullptr;
  return op && op->getName().stripDialect() == stripped ? op : nullptr;
}

bool floatConstIs(Value v, float expected) {
  auto c = v ? v.getDefiningOp<arith::ConstantOp>() : arith::ConstantOp{};
  auto fa = c ? dyn_cast<FloatAttr>(c.getValue()) : FloatAttr{};
  return fa && fa.getValueAsDouble() == double(expected);
}

bool negInfConst(Value v) {
  auto c = v ? v.getDefiningOp<arith::ConstantOp>() : arith::ConstantOp{};
  auto fa = c ? dyn_cast<FloatAttr>(c.getValue()) : FloatAttr{};
  return fa && fa.getValue().isNegInfinity();
}
} // namespace

bool ownerValueDependsOn(Value v, Value target) {
  llvm::SmallPtrSet<Value, 16> visited;
  return dependsImpl(v, target, visited);
}

OwnerStreams ownerStreams(func::FuncOp fn,
                          const compilerinfra::TargetProfile &p) {
  auto &space = fn.front().front();
  if (!fn.getArgAttrOfType<DenseI64ArrayAttr>(0, "block.shape"))
    return {};
  auto abi = llvm::cantFail(
      llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi")));
  for (auto &arg : *abi.getAsArray())
    if (arg.getAsObject()->getString("direction") == "InOut")
      return {};

  auto caps = p.services.get<LocalSupply>();
  if (!caps)
    return {};
  int64_t limit = std::min(caps->budget, caps->capacity);

  auto layout = [&](Value v, StringRef k) {
    return fn.getArgAttrOfType<DenseI64ArrayAttr>(
        cast<BlockArgument>(v).getArgNumber(), k);
  };

  auto buildForPack = [&](int64_t pack) -> OwnerStreams {
    OwnerStreams result;
    result.pack = pack;
    auto &bb = space.getRegion(0).front();
    Value innerOwner = bb.getNumArguments() ? bb.getArguments().back() : Value{};
    if (pack > 1) {
      auto tileAttr = space.getAttrOfType<IntegerAttr>("block.owner_tile");
      auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
      if (!tileAttr || !axes || axes.empty() || !innerOwner)
        return {};
      int64_t tile = tileAttr.getInt();
      int64_t innerExt =
          layout(space.getOperand(axes.size() - 1),
                 "block.shape")[axes[axes.size() - 1]];
      int64_t stepTile = std::max<int64_t>(tile, pack);
      if ((tile % pack != 0 && pack % tile != 0) || innerExt < stepTile ||
          innerExt % stepTile != 0)
        return {};
      int loopCount = 0;
      int64_t commonK = -1;
      bool ok = true;
      for (auto &topOp : bb.without_terminator()) {
        if (auto loop = dyn_cast<scf::ForOp>(topOp)) {
          ++loopCount;
          if (ownerValueDependsOn(loop.getLowerBound(), innerOwner) ||
              ownerValueDependsOn(loop.getUpperBound(), innerOwner) ||
              ownerValueDependsOn(loop.getStep(), innerOwner))
            ok = false;
          for (Value init : loop.getInitArgs())
            if (!isa<RankedTensorType>(init.getType()) &&
                ownerValueDependsOn(init, innerOwner))
              ok = false;
        } else if (isa<scf::IfOp, memref::LoadOp, memref::StoreOp>(topOp) ||
                   topOp.getName().stripDialect() == "slice_read") {
          if (llvm::any_of(topOp.getOperands(), [&](Value v) {
                return ownerValueDependsOn(v, innerOwner);
              }))
            ok = false;
        } else if (topOp.getName().stripDialect() == "slice_write") {
          Value vec = topOp.getOperand(0), view = topOp.getOperand(1);
          auto vt = dyn_cast<RankedTensorType>(vec.getType());
          if (!vt || !vt.getElementType().isF32()) {
            ok = false;
            continue;
          }
          int64_t sAxis = topOp.getAttrOfType<IntegerAttr>("axis").getInt();
          SmallVector<Value> coords(topOp.getOperands().drop_front(2));
          coords.insert(coords.begin() + sAxis, Value{});
          auto st = layout(view, "block.strides");
          int fPos = -1;
          for (auto [ci, cv] : llvm::enumerate(coords)) {
            if (!cv)
              continue;
            if (cv == innerOwner) {
              if (fPos >= 0 || st[ci] != 1)
                ok = false;
              fPos = ci;
            } else if (ownerValueDependsOn(cv, innerOwner)) {
              ok = false;
            }
          }
          if (fPos < 0)
            ok = false;
        }
      }
      if (!ok || loopCount != 1)
        return {};
      space.walk([&](Operation *op) {
        if (isa<scf::IfOp>(op))
          ok = false;
        if (auto loop = dyn_cast<scf::ForOp>(op)) {
          if (op->getBlock() != &bb)
            ok = false;
        }
        if (op->getName().stripDialect() == "lane_sum") {
          int64_t k = kir::memberExtent(fn, op->getOperand(0), "block.shape");
          if (k < 32 || k % 32 != 0 || (commonK > 0 && commonK != k))
            ok = false;
          commonK = k;
        }
        for (Value r : op->getResults()) {
          if (auto rt = dyn_cast<RankedTensorType>(r.getType())) {
            if (!rt.getElementType().isF32())
              ok = false;
            int64_t k = kir::memberExtent(fn, r, "block.shape");
            if (k < 32 || k % 32 != 0 || (commonK > 0 && commonK != k))
              ok = false;
            commonK = k;
            StringRef sn = op->getName().stripDialect();
            StringRef fnName = op->getName().getStringRef();
            if (sn != "slice_read" && sn != "lane_splat" &&
                fnName != "scf.for" && fnName != "arith.addf" &&
                fnName != "arith.subf" && fnName != "arith.mulf")
              ok = false;
          }
        }
      });
      if (!ok || commonK <= 0)
        return {};
    }

    for (auto loop : space.getRegion(0).front().getOps<scf::ForOp>()) {
      auto bound = loop.getUpperBound().getDefiningOp<memref::DimOp>();
      if (!bound || !bound.getConstantIndex() ||
          !matchPattern(loop.getLowerBound(), m_Zero()) ||
          !matchPattern(loop.getStep(), m_One())) {
        if (pack > 1)
          return {};
        continue;
      }
      auto steps =
          layout(bound.getSource(), "block.shape")[*bound.getConstantIndex()];
      if (!steps || steps > INT_MAX) {
        if (pack > 1)
          return {};
        continue;
      }
      for (auto &op : *loop.getBody()) {
        SmallVector<Value> coords;
        Value view;
        int member = -1;
        if (auto l = dyn_cast<memref::LoadOp>(op)) {
          view = l.getMemref();
          llvm::append_range(coords, l.getIndices());
        } else if (auto s = dyn_cast<memref::StoreOp>(op)) {
          view = s.getMemref();
          llvm::append_range(coords, s.getIndices());
        } else if (op.getName().stripDialect() == "slice_read") {
          view = op.getOperand(0);
          llvm::append_range(coords, op.getOperands().drop_front());
          member = op.getAttrOfType<IntegerAttr>("axis").getInt();
          coords.insert(coords.begin() + member, Value{});
        } else if (op.getName().stripDialect() == "slice_write") {
          if (pack > 1)
            return {};
          continue;
        } else {
          if (pack > 1 && op.getName().getStringRef() != "scf.yield") {
            // Scalar values inside a packed loop that depend on innerOwner are
            // only supported via streamed loads/lane_splat/lane_sum->store.
            for (Value r : op.getResults()) {
              if (!isa<RankedTensorType>(r.getType()) &&
                  ownerValueDependsOn(r, innerOwner) &&
                  op.getName().stripDialect() != "lane_sum")
                return {};
            }
          }
          continue;
        }
        auto axis = llvm::find(coords, loop.getInductionVar()) - coords.begin();
        if (axis == int64_t(coords.size()) ||
            llvm::count(coords, loop.getInductionVar()) != 1) {
          if (pack > 1)
            return {};
          continue;
        }
        if (llvm::any_of(coords, [&](Value v) {
              return v && v != loop.getInductionVar() &&
                     !loop.isDefinedOutsideOfLoop(v);
            })) {
          if (pack > 1)
            return {};
          continue;
        }
        auto sh = layout(view, "block.shape"),
             stride = layout(view, "block.strides");
        if (steps > sh[axis] || (member >= 0 && stride[member] != 1)) {
          if (pack > 1)
            return {};
          continue;
        }
        if (isa<memref::StoreOp>(op)) {
          if (llvm::any_of(view.getUsers(), [&](Operation *other) {
                return other != &op && isa<memref::StoreOp>(other);
              })) {
            if (pack > 1)
              return {};
            continue;
          }
        }
        int64_t accessPack = 1;
        if (pack > 1) {
          auto elemTy = cast<MemRefType>(view.getType()).getElementType();
          if (!elemTy.isF32())
            return {};
          int fPos = -1;
          bool anyDep = false;
          for (auto [ci, cv] : llvm::enumerate(coords)) {
            if (!cv || int64_t(ci) == axis)
              continue;
            if (cv == innerOwner) {
              if (fPos >= 0 || member >= 0 || stride[ci] != 1)
                return {};
              fPos = ci;
              anyDep = true;
            } else if (ownerValueDependsOn(cv, innerOwner)) {
              return {};
            }
          }
          if (isa<memref::StoreOp>(op) && fPos < 0)
            return {};
          accessPack = anyDep ? pack : 1;
        }
        int64_t width = member >= 0 ? sh[member] : accessPack, bytes = 0;
        result.bytes = llvm::alignTo(result.bytes, 128);
        if (width <= 0 || __builtin_mul_overflow(width, steps, &bytes) ||
            __builtin_mul_overflow(bytes,
                                   (int64_t)cast<MemRefType>(view.getType())
                                           .getElementTypeBitWidth() /
                                       8,
                                   &bytes))
          return {};
        result.transfers.push_back(
            {&op, result.bytes, axis, steps, width, accessPack});
        if (__builtin_add_overflow(bytes, result.bytes, &result.bytes))
          return {};
      }
    }
    if (!result.bytes || result.bytes > limit)
      return {};
    return result;
  };

  auto tileAttr = space.getAttrOfType<IntegerAttr>("block.owner_tile");
  if (tileAttr && tileAttr.getInt() >= 1) {
    for (int64_t cand : {int64_t(8), int64_t(4), int64_t(2)}) {
      if (tileAttr.getInt() % cand == 0 || cand % tileAttr.getInt() == 0) {
        auto packed = buildForPack(cand);
        if (packed.bytes > 0)
          return packed;
      }
    }
  }
  return buildForPack(1);
}

std::optional<PanelFoldSupply> panelFoldSupply(
    func::FuncOp fn, const compilerinfra::TargetProfile &p,
    llvm::DenseMap<Operation *, SmallVector<int64_t>> *opOffsets) {
  auto caps = p.services.get<LocalSupply>();
  auto ms = p.services.get<MatrixStorage>();
  if (!caps || !ms || ms->wramBytes <= 0 ||
      !fn.getArgAttrOfType<DenseI64ArrayAttr>(0, "block.shape"))
    return {};
  auto &space = fn.front().front();
  auto tileAttr = space.getAttrOfType<IntegerAttr>("block.owner_tile");
  auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
  auto &bb = space.getRegion(0).front();
  if (!tileAttr || !axes || axes.empty() || bb.getNumArguments() == 0)
    return {};
  int64_t tile = tileAttr.getInt();
  auto layout = [&](Value v, StringRef k) {
    return fn.getArgAttrOfType<DenseI64ArrayAttr>(
        cast<BlockArgument>(v).getArgNumber(), k);
  };
  Value rowArg = bb.getArguments().back();
  int64_t rowExtent =
      layout(space.getOperand(axes.size() - 1),
             "block.shape")[axes[axes.size() - 1]];
  int64_t M = 0;
  for (int64_t cand : {int64_t(64), int64_t(32), int64_t(16)}) {
    int64_t stepTile = std::max<int64_t>(tile, cand);
    if ((tile % cand == 0 || cand % tile == 0) && rowExtent >= stepTile &&
        rowExtent % stepTile == 0) {
      M = cand;
      break;
    }
  }
  if (!M)
    return {};
  constexpr int64_t S = 64;

  auto rowInvariant = [&](Value v, Value loopIv = {}) {
    if (!ownerValueDependsOn(v, rowArg))
      return !loopIv || !ownerValueDependsOn(v, loopIv);
    return false;
  };

  SmallVector<scf::ForOp> loops;
  Operation *qSlice = nullptr;
  Operation *outWrite = nullptr;
  for (auto &op : bb.without_terminator()) {
    StringRef sn = op.getName().stripDialect();
    if (auto l = dyn_cast<scf::ForOp>(op)) {
      if (!l->hasAttr("kir.reassociate"))
        return {};
      loops.push_back(l);
    } else if (sn == "slice_read") {
      if (qSlice)
        return {};
      qSlice = &op;
    } else if (sn == "slice_write") {
      if (outWrite)
        return {};
      outWrite = &op;
    } else if (!isMemoryEffectFree(&op)) {
      return {};
    }
  }
  if (loops.size() != 2 || !qSlice || !outWrite)
    return {};

  auto checkRowOrTokSlice = [&](Operation *sl, Value lastCoord, Value loopIv,
                                Type &elemTy, int64_t &outerExt,
                                int64_t &featExt) -> bool {
    Value view = sl->getOperand(sl->getName().stripDialect() == "slice_write" ? 1 : 0);
    if (!isa<BlockArgument>(view))
      return false;
    auto mt = cast<MemRefType>(view.getType());
    elemTy = mt.getElementType();
    if (!elemTy.isBF16() && !elemTy.isF16())
      return false;
    int64_t rank = mt.getRank();
    auto axisAttr = sl->getAttrOfType<IntegerAttr>("axis");
    if (rank < 2 || !axisAttr || axisAttr.getInt() != rank - 1)
      return false;
    auto coords = sl->getOperands().drop_front(
        sl->getName().stripDialect() == "slice_write" ? 2 : 1);
    if (int64_t(coords.size()) != rank - 1 || coords.back() != lastCoord)
      return false;
    for (Value c : coords.drop_back())
      if (!rowInvariant(c, loopIv))
        return false;
    auto sh = layout(view, "block.shape"), st = layout(view, "block.strides");
    if (st[rank - 1] != 1)
      return false;
    featExt = sh[rank - 1];
    outerExt = sh[rank - 2];
    return featExt >= 32 && featExt <= 256 && featExt % 32 == 0;
  };

  Type halfTy;
  int64_t qRows = 0, D = 0;
  if (!checkRowOrTokSlice(qSlice, rowArg, Value{}, halfTy, qRows, D) ||
      !qSlice->hasOneUse())
    return {};
  Operation *qExt = *qSlice->getUsers().begin();
  if (qExt->getName().getStringRef() != "arith.extf")
    return {};
  Value qWide = qExt->getResult(0);

  scf::ForOp maxLoop = loops[0], sumLoop = loops[1];
  auto sameKBound = [&](scf::ForOp l, Value kView, int64_t tokAxis) {
    if (!matchPattern(l.getLowerBound(), m_Zero()) ||
        !matchPattern(l.getStep(), m_One()))
      return false;
    auto dim = l.getUpperBound().getDefiningOp<memref::DimOp>();
    return dim && dim.getSource() == kView &&
           dim.getConstantIndex() == size_t(tokAxis);
  };

  if (maxLoop.getNumResults() != 1 || !negInfConst(maxLoop.getInitArgs()[0]) ||
      sumLoop.getNumResults() != 2 || !floatConstIs(sumLoop.getInitArgs()[0], 0.0f))
    return {};
  auto *initVecOp = defWithName(sumLoop.getInitArgs()[1], "lane_splat");
  if (!initVecOp || !floatConstIs(initVecOp->getOperand(0), 0.0f))
    return {};

  auto matchDot = [&](Value scoreVal, Value iv, Operation *&kSl,
                      Operation *&dotOp, Value &gainVal) -> bool {
    auto mulScore = scoreVal.getDefiningOp<arith::MulFOp>();
    if (!mulScore)
      return false;
    for (int side : {0, 1}) {
      Operation *ls = defWithName(mulScore.getOperand(side), "lane_sum");
      Value g = mulScore.getOperand(1 - side);
      if (!ls || !ls->hasOneUse() || !rowInvariant(g, iv))
        continue;
      auto prod = ls->getOperand(0).getDefiningOp<arith::MulFOp>();
      if (!prod)
        continue;
      for (int pSide : {0, 1}) {
        if (prod.getOperand(pSide) != qWide)
          continue;
        auto extK = prod.getOperand(1 - pSide).getDefiningOp<arith::ExtFOp>();
        Operation *ks = extK ? defWithName(extK.getIn(), "slice_read") : nullptr;
        Type kElem;
        int64_t kTokens = 0, kD = 0;
        if (!ks || !checkRowOrTokSlice(ks, iv, iv, kElem, kTokens, kD) ||
            kElem != halfTy || kD != D)
          continue;
        kSl = ks;
        dotOp = ls;
        gainVal = g;
        return true;
      }
    }
    return false;
  };

  // Match maxLoop body.
  Value maxYield = maxLoop.getBody()->getTerminator()->getOperand(0);
  auto maxUpdate = maxYield.getDefiningOp<arith::MaxNumFOp>();
  if (!maxUpdate)
    return {};
  Value maxCand = maxUpdate.getLhs() == maxLoop.getRegionIterArgs()[0]
                      ? maxUpdate.getRhs()
                      : maxUpdate.getRhs() == maxLoop.getRegionIterArgs()[0]
                            ? maxUpdate.getLhs()
                            : Value{};
  if (!maxCand)
    return {};
  bool causal1 = false;
  Value rawScore1 = maxCand;
  if (auto br1 = maxCand.getDefiningOp<scf::IfOp>()) {
    auto cmp1 = br1.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp1 || cmp1.getPredicate() != arith::CmpIPredicate::sge ||
        cmp1.getLhs() != rowArg || cmp1.getRhs() != maxLoop.getInductionVar() ||
        br1.getElseRegion().empty() ||
        !negInfConst(br1.elseBlock()->getTerminator()->getOperand(0)))
      return {};
    causal1 = true;
    rawScore1 = br1.thenBlock()->getTerminator()->getOperand(0);
  }
  Operation *kSlice1 = nullptr, *dot1 = nullptr;
  Value gain1;
  if (!matchDot(rawScore1, maxLoop.getInductionVar(), kSlice1, dot1, gain1))
    return {};
  Value kView = kSlice1->getOperand(0);
  int64_t kRank = cast<MemRefType>(kView.getType()).getRank();
  if (!sameKBound(maxLoop, kView, kRank - 2) ||
      !sameKBound(sumLoop, kView, kRank - 2))
    return {};

  // Match sumLoop body.
  auto sumYields = sumLoop.getBody()->getTerminator()->getOperands();
  auto massUpdate = sumYields[0].getDefiningOp<arith::AddFOp>();
  auto vecUpdate = sumYields[1].getDefiningOp<arith::AddFOp>();
  if (!massUpdate || !vecUpdate)
    return {};
  Value weightVal = massUpdate.getLhs() == sumLoop.getRegionIterArgs()[0]
                        ? massUpdate.getRhs()
                        : massUpdate.getRhs() == sumLoop.getRegionIterArgs()[0]
                              ? massUpdate.getLhs()
                              : Value{};
  Value vecTerm = vecUpdate.getLhs() == sumLoop.getRegionIterArgs()[1]
                      ? vecUpdate.getRhs()
                      : vecUpdate.getRhs() == sumLoop.getRegionIterArgs()[1]
                            ? vecUpdate.getLhs()
                            : Value{};
  if (!weightVal || !vecTerm)
    return {};

  bool causal2 = false;
  Value expVal = weightVal;
  if (auto br2 = weightVal.getDefiningOp<scf::IfOp>()) {
    auto cmp2 = br2.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp2 || cmp2.getPredicate() != arith::CmpIPredicate::sge ||
        cmp2.getLhs() != rowArg || cmp2.getRhs() != sumLoop.getInductionVar() ||
        br2.getElseRegion().empty() ||
        !floatConstIs(br2.elseBlock()->getTerminator()->getOperand(0), 0.0f))
      return {};
    causal2 = true;
    expVal = br2.thenBlock()->getTerminator()->getOperand(0);
  }
  if (causal1 != causal2)
    return {};
  Operation *expOp = expVal.getDefiningOp();
  if (!expOp || (expOp->getName().getStringRef() != "math.exp2" &&
                 expOp->getName().getStringRef() != "math.exp"))
    return {};
  auto subPeak = expOp->getOperand(0).getDefiningOp<arith::SubFOp>();
  if (!subPeak || subPeak.getRhs() != maxLoop.getResult(0))
    return {};
  Operation *kSlice2 = nullptr, *dot2 = nullptr;
  Value gain2;
  if (!matchDot(subPeak.getLhs(), sumLoop.getInductionVar(), kSlice2, dot2,
                gain2))
    return {};
  if (kSlice2->getOperand(0) != kView ||
      kSlice2->getOperands().drop_back() != kSlice1->getOperands().drop_back())
    return {};

  auto vProd = vecTerm.getDefiningOp<arith::MulFOp>();
  if (!vProd)
    return {};
  Operation *vSlice = nullptr, *wTrunc = nullptr;
  int64_t DV = 0;
  for (int side : {0, 1}) {
    auto extV = vProd.getOperand(side).getDefiningOp<arith::ExtFOp>();
    Operation *vs = extV ? defWithName(extV.getIn(), "slice_read") : nullptr;
    Operation *sp = defWithName(vProd.getOperand(1 - side), "lane_splat");
    auto extW = sp ? sp->getOperand(0).getDefiningOp<arith::ExtFOp>()
                   : arith::ExtFOp{};
    auto trW = extW ? extW.getIn().getDefiningOp<arith::TruncFOp>()
                    : arith::TruncFOp{};
    Type vElem;
    int64_t vTokens = 0, vFeat = 0;
    if (vs && vs != kSlice2 && trW && trW.getIn() == weightVal &&
        trW.getType() == halfTy &&
        checkRowOrTokSlice(vs, sumLoop.getInductionVar(),
                           sumLoop.getInductionVar(), vElem, vTokens, vFeat) &&
        vElem == halfTy &&
        layout(vs->getOperand(0), "block.shape")[kRank - 2] ==
            layout(kView, "block.shape")[kRank - 2]) {
      vSlice = vs;
      wTrunc = trW.getOperation();
      DV = vFeat;
      break;
    }
  }
  if (!vSlice || !wTrunc)
    return {};

  // Match post-loop normalization and writeback.
  Type yElem;
  int64_t yRows = 0, yFeat = 0;
  if (!checkRowOrTokSlice(outWrite, rowArg, Value{}, yElem, yRows, yFeat) ||
      yElem != halfTy || yFeat != DV || yRows != qRows)
    return {};
  auto outTrunc = outWrite->getOperand(0).getDefiningOp<arith::TruncFOp>();
  auto normMul =
      outTrunc ? outTrunc.getIn().getDefiningOp<arith::MulFOp>() : arith::MulFOp{};
  if (!normMul)
    return {};
  Operation *invSplat = nullptr;
  for (int side : {0, 1}) {
    if (normMul.getOperand(side) == sumLoop.getResult(1))
      invSplat = defWithName(normMul.getOperand(1 - side), "lane_splat");
  }
  if (!invSplat)
    return {};
  Value invVal = invSplat->getOperand(0);
  auto checkRecip = [&](Value r) {
    auto dv = r.getDefiningOp<arith::DivFOp>();
    return dv && floatConstIs(dv.getLhs(), 1.0f) &&
           dv.getRhs() == sumLoop.getResult(0);
  };
  if (!checkRecip(invVal)) {
    auto safeIf = invVal.getDefiningOp<scf::IfOp>();
    if (!safeIf || safeIf.getElseRegion().empty())
      return {};
    auto eqZero = safeIf.getCondition().getDefiningOp<arith::CmpIOp>();
    auto dimOp = eqZero ? eqZero.getLhs().getDefiningOp<memref::DimOp>()
                        : memref::DimOp{};
    if (!eqZero || eqZero.getPredicate() != arith::CmpIPredicate::eq || !dimOp ||
        dimOp.getSource() != kView ||
        dimOp.getConstantIndex() != size_t(kRank - 2) ||
        !matchPattern(eqZero.getRhs(), m_Zero()) ||
        !floatConstIs(safeIf.thenBlock()->getTerminator()->getOperand(0),
                      0.0f) ||
        !checkRecip(safeIf.elseBlock()->getTerminator()->getOperand(0)))
      return {};
  }

  int64_t halfB = halfTy.getIntOrFloatBitWidth() / 8;
  int64_t wramBytes = 64 * std::max<int64_t>(D, S) * halfB;
  if (wramBytes > ms->wramBytes)
    return {};

  // Assign real NRAM fragment slots by interval liveness over actual SSA uses.
  llvm::DenseMap<Operation *, unsigned> startOrd, endOrd;
  unsigned nextOrd = 0;
  std::function<void(Block &)> numberBlock = [&](Block &blk) {
    for (auto &op : blk) {
      startOrd[&op] = nextOrd++;
      for (auto &reg : op.getRegions())
        for (auto &sub : reg)
          numberBlock(sub);
      endOrd[&op] = nextOrd++;
    }
  };
  numberBlock(bb);

  Operation *weightDef = weightVal.getDefiningOp();
  struct RoleSpec {
    Operation *op;
    StringRef role;
    SmallVector<int64_t, 2> slotBytes;
  };
  SmallVector<RoleSpec, 13> roleSpecs = {
      {qSlice, "query", {M * D * halfB}},
      {maxLoop.getOperation(), "peak_loop", {M * 4}},
      {kSlice1, "key_peak", {S * D * halfB}},
      {dot1, "score_peak", {M * S * 4}},
      {sumLoop.getOperation(), "sum_loop", {M * 4, M * DV * 4}},
      {kSlice2, "key_sum", {S * D * halfB}},
      {dot2, "score_sum", {M * S * 4}},
      {weightDef, "weight", {M * S * 4}},
      {wTrunc, "weight_narrow", {M * S * halfB}},
      {vSlice, "value", {S * DV * halfB, DV * S * halfB}},
      {normMul.getOperation(), "norm", {M * DV * 4}},
      {outTrunc.getOperation(), "output_narrow", {M * DV * halfB}},
      {outWrite, "output", {}},
  };

  llvm::DenseSet<Operation *> materialized;
  llvm::DenseMap<Operation *, StringRef> roleMap;
  for (const auto &rs : roleSpecs) {
    materialized.insert(rs.op);
    roleMap[rs.op] = rs.role;
  }

  // Compute the last program point where v's panel value (forwarded through
  // unmaterialized elementwise/cast/if bridge ops) reaches a materialized
  // consumer or enclosing loop latch.
  auto liveEndOrd = [&](Value v, Operation *defOp) {
    unsigned hi = endOrd.lookup(defOp);
    SmallVector<Value, 8> worklist = {v};
    llvm::DenseSet<Value> seen = {v};
    while (!worklist.empty()) {
      Value cur = worklist.pop_back_val();
      for (Operation *u : cur.getUsers()) {
        Operation *top = u;
        while (top->getBlock() != defOp->getBlock() && top->getParentOp())
          top = top->getParentOp();
        if (top->getBlock() == defOp->getBlock())
          hi = std::max(hi, endOrd.lookup(top));
        if (materialized.contains(u))
          continue;
        for (Value res : u->getResults())
          if (seen.insert(res).second)
            worklist.push_back(res);
        if (isa<scf::YieldOp>(u)) {
          Operation *parent = u->getParentOp();
          if (isa<scf::IfOp>(parent) && !materialized.contains(parent)) {
            for (Value res : parent->getResults())
              if (seen.insert(res).second)
                worklist.push_back(res);
          }
        }
      }
    }
    return hi;
  };

  llvm::sort(roleSpecs, [&](const RoleSpec &a, const RoleSpec &b) {
    return startOrd.lookup(a.op) < startOrd.lookup(b.op);
  });

  IntervalAlloc alloc(caps->alignment);
  llvm::DenseMap<Operation *, SmallVector<int64_t>> map;
  for (const auto &rs : roleSpecs) {
    if (rs.slotBytes.empty())
      continue;
    SmallVector<int64_t> offs;
    unsigned lo = startOrd.lookup(rs.op);
    for (unsigned i = 0; i < rs.slotBytes.size(); ++i) {
      Value tracked =
          rs.op->getResult(std::min<unsigned>(i, rs.op->getNumResults() - 1));
      unsigned hi = liveEndOrd(tracked, rs.op);
      offs.push_back(alloc.alloc(lo, hi, rs.slotBytes[i]));
    }
    map[rs.op] = std::move(offs);
  }

  // Derive panel-contraction / row-reduction scratch interval from the
  // top-level operations enclosing the contraction and reduction roles.
  unsigned scratchLo = ~0u, scratchHi = 0;
  for (Operation *u : {dot1, dot2, weightDef, vSlice}) {
    Operation *top = u;
    while (top->getBlock() != &bb && top->getParentOp())
      top = top->getParentOp();
    scratchLo = std::min(scratchLo, startOrd.lookup(top));
    scratchHi = std::max(scratchHi, endOrd.lookup(top));
  }
  int64_t magOff = alloc.alloc(scratchLo, scratchHi, 4096 * 2);
  int64_t flagOff = alloc.alloc(scratchLo, scratchHi, 4096 * 2);
  int64_t auxBytes = 512 + M * 64 * 4;
  int64_t auxOff = alloc.alloc(scratchLo, scratchHi, auxBytes);

  if (alloc.peak > std::min(caps->budget, caps->capacity))
    return {};

  if (opOffsets)
    *opOffsets = std::move(map);

  PanelFoldSupply res;
  res.rows = M;
  res.span = S;
  res.depth = D;
  res.features = DV;
  res.wramBytes = wramBytes;
  res.localBytes = alloc.peak;
  res.scratchOffsets = {magOff, flagOff, auxOff};
  res.roles = std::move(roleMap);
  return res;
}
} // namespace intentdsl::dsa
