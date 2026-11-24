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

  Type halfTy;
  auto checkSlice = [&](Operation *sl, Value lastCoord, Value loopIv,
                        int64_t &outerExt, int64_t &featExt) -> bool {
    bool isWrite = sl->getName().stripDialect() == "slice_write";
    Value view = sl->getOperand(isWrite ? 1 : 0);
    if (!isa<BlockArgument>(view))
      return false;
    auto mt = cast<MemRefType>(view.getType());
    Type et = mt.getElementType();
    if (!et.isBF16() && !et.isF16())
      return false;
    if (halfTy && et != halfTy)
      return false;
    int64_t rank = mt.getRank();
    auto axisAttr = sl->getAttrOfType<IntegerAttr>("axis");
    if (rank < 2 || !axisAttr || axisAttr.getInt() != rank - 1)
      return false;
    auto coords = sl->getOperands().drop_front(isWrite ? 2 : 1);
    if (int64_t(coords.size()) != rank - 1 || coords.back() != lastCoord)
      return false;
    for (Value c : coords.drop_back())
      if (!rowInvariant(c, loopIv))
        return false;
    auto sh = layout(view, "block.shape"), st = layout(view, "block.strides");
    if (!sh || !st || st[rank - 1] != 1)
      return false;
    featExt = sh[rank - 1];
    outerExt = sh[rank - 2];
    if (featExt < 32 || featExt > 256 || featExt % 32 != 0)
      return false;
    if (!halfTy)
      halfTy = et;
    return true;
  };

  auto isPureScalarArith = [](Operation &op) {
    if (op.getNumResults() != 1 ||
        isa<RankedTensorType>(op.getResult(0).getType()) ||
        !isMemoryEffectFree(&op) || isa<scf::ForOp, scf::IfOp>(op))
      return false;
    StringRef n = op.getName().getStringRef();
    return n == "arith.constant" || n == "memref.dim" ||
           n == "arith.index_cast" || n == "arith.index_castui" ||
           n == "arith.addi" || n == "arith.subi" || n == "arith.muli" ||
           n == "arith.divsi" || n == "arith.floordivsi" ||
           n == "arith.remsi" || n == "arith.minsi" || n == "arith.maxsi" ||
           n == "arith.addf" || n == "arith.subf" || n == "arith.mulf" ||
           n == "arith.divf" || n == "arith.maxnumf" || n == "arith.extf" ||
           n == "arith.truncf" || n == "arith.cmpi" || n == "arith.select" ||
           n == "math.sqrt" || n == "math.exp" || n == "math.exp2" ||
           n == "math.log";
  };

  auto isCausalCond = [&](Value cond, Value iv) {
    auto cmp = cond.getDefiningOp<arith::CmpIOp>();
    if (!cmp)
      return false;
    return (cmp.getPredicate() == arith::CmpIPredicate::sge &&
            cmp.getLhs() == rowArg && cmp.getRhs() == iv) ||
           (cmp.getPredicate() == arith::CmpIPredicate::sle &&
            cmp.getLhs() == iv && cmp.getRhs() == rowArg);
  };

  // Value-driven shape classification across the owner space and each
  // independent associative reduction pass (no whole-algorithm matching).
  enum class VKind {
    InvScalar,
    OwnerScalar,
    OwnerPanelHalf,
    OwnerPanelWidenRef,
    OwnerPanelF32,
    OwnerSplat,
    LoopScalarCarry,
    LoopPanelCarry,
    StripSliceHalf,
    StripSliceWidenRef,
    StripPanelF32,
    StripPanelHalf,
    StripWeightSplatRef,
    StripOuterTerm,
  };
  struct VInfo {
    VKind kind = VKind::InvScalar;
    int64_t feat = 0;
  };
  llvm::DenseMap<Value, VInfo> vals;
  for (Value arg : fn.getArguments())
    vals[arg] = {VKind::InvScalar, 0};
  for (Value arg : bb.getArguments().drop_back())
    vals[arg] = {VKind::InvScalar, 0};

  struct RoleSpec {
    Operation *op;
    StringRef role;
    SmallVector<int64_t, 2> slotBytes;
  };
  SmallVector<RoleSpec, 16> roleSpecs;
  SmallVector<Operation *, 8> scratchAnchors;
  int64_t wramBytes = 0;
  int64_t ownerTransferSites = 0, ownerTransferBytes = 0;
  int64_t stripTransferSites = 0, stripTransferBytes = 0;
  bool hasContraction = false;

  auto halfBytes = [&]() -> int64_t {
    return halfTy ? halfTy.getIntOrFloatBitWidth() / 8 : 2;
  };

  auto checkScalarSubregion = [&](Block &subBlk) -> bool {
    llvm::DenseSet<Value> localOk;
    for (Operation &inOp : subBlk.without_terminator()) {
      if (!isPureScalarArith(inOp))
        return false;
      for (Value u : inOp.getOperands()) {
        if (localOk.contains(u))
          continue;
        auto it = vals.find(u);
        if (it == vals.end() || (it->second.kind != VKind::InvScalar &&
                                 it->second.kind != VKind::OwnerScalar))
          return false;
      }
      localOk.insert(inOp.getResult(0));
    }
    auto *term = subBlk.getTerminator();
    if (term->getNumOperands() != 1)
      return false;
    Value y = term->getOperand(0);
    if (localOk.contains(y))
      return true;
    auto it = vals.find(y);
    return it != vals.end() && (it->second.kind == VKind::InvScalar ||
                                it->second.kind == VKind::OwnerScalar);
  };

  auto analyzeStripLoop = [&](scf::ForOp loop) -> bool {
    if (!loop->hasAttr("kir.reassociate") || loop.getNumResults() == 0 ||
        !matchPattern(loop.getLowerBound(), m_Zero()) ||
        !matchPattern(loop.getStep(), m_One()))
      return false;
    auto dimOp = loop.getUpperBound().getDefiningOp<memref::DimOp>();
    if (!dimOp || !isa<BlockArgument>(dimOp.getSource()) ||
        !dimOp.getConstantIndex())
      return false;
    auto boundType = cast<MemRefType>(dimOp.getSource().getType());
    if (*dimOp.getConstantIndex() != size_t(boundType.getRank() - 2))
      return false;
    int64_t K =
        layout(dimOp.getSource(), "block.shape")[*dimOp.getConstantIndex()];
    if (K < 0)
      return false;

    // Pre-scan strip_read element type if halfTy has not been latched yet.
    if (!halfTy) {
      loop.walk([&](Operation *sub) {
        if (sub->getName().stripDialect() == "slice_read" &&
            isa<BlockArgument>(sub->getOperand(0))) {
          Type et =
              cast<MemRefType>(sub->getOperand(0).getType()).getElementType();
          if (et.isBF16() || et.isF16())
            halfTy = et;
        }
      });
      if (!halfTy)
        return false;
    }

    SmallVector<int64_t, 2> carryBytes;
    for (unsigned k = 0; k < loop.getNumResults(); ++k) {
      Value initV = loop.getInitArgs()[k];
      Value iterV = loop.getRegionIterArgs()[k];
      Type resTy = loop.getResult(k).getType();
      auto it = vals.find(initV);
      if (it == vals.end())
        return false;
      if (resTy.isF32()) {
        if (it->second.kind != VKind::InvScalar &&
            it->second.kind != VKind::OwnerScalar)
          return false;
        carryBytes.push_back(M * 4);
        vals[iterV] = {VKind::LoopScalarCarry, 0};
      } else if (auto rt = dyn_cast<RankedTensorType>(resTy)) {
        if (!rt.getElementType().isF32())
          return false;
        int64_t F = kir::memberExtent(fn, loop.getResult(k), "block.shape");
        if (F < 32 || F > 256 || F % 32 != 0)
          return false;
        if ((it->second.kind != VKind::OwnerSplat &&
             it->second.kind != VKind::OwnerPanelF32) ||
            it->second.feat != F)
          return false;
        carryBytes.push_back(M * F * 4);
        vals[iterV] = {VKind::LoopPanelCarry, F};
      } else {
        return false;
      }
    }

    Block &body = *loop.getBody();
    Value iv = loop.getInductionVar();
    auto *yieldOp = body.getTerminator();

    llvm::DenseSet<Operation *> yieldCombOps;
    llvm::DenseSet<Operation *> dotMulOps;
    llvm::DenseSet<Operation *> outerMulOps;
    for (unsigned k = 0; k < loop.getNumResults(); ++k) {
      Operation *comb = yieldOp->getOperand(k).getDefiningOp();
      if (!comb || !comb->hasOneUse() || comb->getBlock() != &body)
        return false;
      Value iterV = loop.getRegionIterArgs()[k];
      if (vals[iterV].kind == VKind::LoopScalarCarry) {
        if (!isa<arith::MaxNumFOp, arith::AddFOp>(comb) ||
            (comb->getOperand(0) != iterV && comb->getOperand(1) != iterV))
          return false;
        yieldCombOps.insert(comb);
      } else {
        auto add = dyn_cast<arith::AddFOp>(comb);
        if (!add ||
            (add.getLhs() != iterV && add.getRhs() != iterV))
          return false;
        Value term = add.getLhs() == iterV ? add.getRhs() : add.getLhs();
        auto mul = term.getDefiningOp<arith::MulFOp>();
        if (!mul || !mul->hasOneUse() || mul->getBlock() != &body)
          return false;
        yieldCombOps.insert(comb);
        outerMulOps.insert(mul.getOperation());
      }
    }
    for (Operation &bop : body.without_terminator()) {
      if (bop.getName().stripDialect() == "lane_sum") {
        auto mul = bop.getOperand(0).getDefiningOp<arith::MulFOp>();
        if (!mul || !mul->hasOneUse() || mul->getBlock() != &body)
          return false;
        dotMulOps.insert(mul.getOperation());
      }
    }

    auto classifyStripElem = [&](Operation &eop, bool inThen) -> bool {
      StringRef en = eop.getName().getStringRef();
      if (en == "math.exp" || en == "math.exp2") {
        auto it = vals.find(eop.getOperand(0));
        if (it == vals.end() || it->second.kind != VKind::StripPanelF32)
          return false;
        roleSpecs.push_back(
            {&eop, inThen ? "strip_then_elem" : "strip_elem", {M * S * 4}});
        vals[eop.getResult(0)] = {VKind::StripPanelF32, S};
        return true;
      }
      if ((en == "arith.mulf" || en == "arith.addf" || en == "arith.subf") &&
          eop.getResult(0).getType().isF32()) {
        auto it0 = vals.find(eop.getOperand(0));
        auto it1 = vals.find(eop.getOperand(1));
        if (it0 == vals.end() || it1 == vals.end())
          return false;
        auto okKind = [](VKind k, Type t) {
          return k == VKind::StripPanelF32 || k == VKind::OwnerScalar ||
                 (k == VKind::InvScalar && t.isF32());
        };
        if (!okKind(it0->second.kind, eop.getOperand(0).getType()) ||
            !okKind(it1->second.kind, eop.getOperand(1).getType()) ||
            (it0->second.kind != VKind::StripPanelF32 &&
             it1->second.kind != VKind::StripPanelF32))
          return false;
        roleSpecs.push_back(
            {&eop, inThen ? "strip_then_elem" : "strip_elem", {M * S * 4}});
        vals[eop.getResult(0)] = {VKind::StripPanelF32, S};
        return true;
      }
      return false;
    };

    int64_t hb = halfBytes();
    bool loopHasContraction = false;
    for (Operation &bop : body.without_terminator()) {
      if (yieldCombOps.contains(&bop) || dotMulOps.contains(&bop))
        continue;
      StringRef sn = bop.getName().stripDialect();
      StringRef fnName = bop.getName().getStringRef();

      if (outerMulOps.contains(&bop)) {
        bool matched = false;
        for (int side : {0, 1}) {
          auto itW = vals.find(bop.getOperand(side));
          auto itV = vals.find(bop.getOperand(1 - side));
          if (itW != vals.end() && itV != vals.end() &&
              itW->second.kind == VKind::StripWeightSplatRef &&
              itV->second.kind == VKind::StripSliceWidenRef) {
            int64_t DV = itV->second.feat;
            roleSpecs.push_back({&bop, "strip_outer_acc", {}});
            vals[bop.getResult(0)] = {VKind::StripOuterTerm, DV};
            wramBytes =
                std::max<int64_t>(wramBytes, 64 * std::max<int64_t>(S, DV) * hb);
            loopHasContraction = true;
            hasContraction = true;
            matched = true;
            break;
          }
        }
        if (!matched)
          return false;
        continue;
      }

      if (sn == "slice_read") {
        int64_t kTok = 0, kFeat = 0;
        if (!checkSlice(&bop, iv, iv, kTok, kFeat) || kTok != K ||
            !bop.hasOneUse())
          return false;
        auto extOp = dyn_cast<arith::ExtFOp>(*bop.getUsers().begin());
        auto extRt =
            extOp ? dyn_cast<RankedTensorType>(extOp.getType()) : RankedTensorType{};
        if (!extRt || !extRt.getElementType().isF32() || !extOp->hasOneUse())
          return false;
        Operation *extUser = *extOp->getUsers().begin();
        if (dotMulOps.contains(extUser)) {
          roleSpecs.push_back({&bop, "strip_slice", {S * kFeat * hb}});
        } else if (outerMulOps.contains(extUser)) {
          roleSpecs.push_back({&bop,
                               "strip_transpose_slice",
                               {S * kFeat * hb, kFeat * S * hb}});
        } else {
          return false;
        }
        vals[bop.getResult(0)] = {VKind::StripSliceHalf, kFeat};
        vals[extOp.getResult()] = {VKind::StripSliceWidenRef, kFeat};
        int64_t panels = std::max<int64_t>(1, (K + S - 1) / S);
        stripTransferSites += panels;
        stripTransferBytes += K * kFeat * hb;
        continue;
      }

      if (fnName == "arith.extf") {
        if (vals.contains(bop.getResult(0)))
          continue;
        auto it = vals.find(bop.getOperand(0));
        if (it != vals.end() && it->second.kind == VKind::StripPanelHalf &&
            bop.getResult(0).getType().isF32()) {
          vals[bop.getResult(0)] = {VKind::StripWeightSplatRef, S};
          continue;
        }
      }

      if (sn == "lane_splat") {
        auto it = vals.find(bop.getOperand(0));
        if (it != vals.end() && it->second.kind == VKind::StripWeightSplatRef) {
          vals[bop.getResult(0)] = {VKind::StripWeightSplatRef, S};
          continue;
        }
      }

      if (sn == "lane_sum") {
        auto mul = bop.getOperand(0).getDefiningOp<arith::MulFOp>();
        if (!mul)
          return false;
        bool matched = false;
        for (int side : {0, 1}) {
          auto itQ = vals.find(mul.getOperand(side));
          auto itK = vals.find(mul.getOperand(1 - side));
          if (itQ != vals.end() && itK != vals.end() &&
              itQ->second.kind == VKind::OwnerPanelWidenRef &&
              itK->second.kind == VKind::StripSliceWidenRef &&
              itQ->second.feat == itK->second.feat) {
            int64_t D = itQ->second.feat;
            roleSpecs.push_back({&bop, "strip_dot", {M * S * 4}});
            vals[bop.getResult(0)] = {VKind::StripPanelF32, S};
            wramBytes =
                std::max<int64_t>(wramBytes, 64 * std::max<int64_t>(D, S) * hb);
            loopHasContraction = true;
            hasContraction = true;
            matched = true;
            break;
          }
        }
        if (!matched)
          return false;
        continue;
      }

      if (fnName == "arith.truncf") {
        auto it = vals.find(bop.getOperand(0));
        if (it != vals.end() && it->second.kind == VKind::StripPanelF32 &&
            bop.getResult(0).getType() == halfTy) {
          roleSpecs.push_back({&bop, "strip_narrow", {M * S * hb}});
          vals[bop.getResult(0)] = {VKind::StripPanelHalf, S};
          continue;
        }
      }

      if (isa<arith::CmpIOp>(bop) && isCausalCond(bop.getResult(0), iv))
        continue;

      if (auto br = dyn_cast<scf::IfOp>(bop)) {
        if (br.getNumResults() != 1 || !br.getResult(0).getType().isF32() ||
            br.getElseRegion().empty() ||
            !isCausalCond(br.getCondition(), iv))
          return false;
        for (Operation &eop : br.elseBlock()->without_terminator()) {
          if (!isa<arith::ConstantOp>(eop))
            return false;
          vals[eop.getResult(0)] = {VKind::InvScalar, 0};
        }
        Value padV = br.elseBlock()->getTerminator()->getOperand(0);
        auto itPad = vals.find(padV);
        if (itPad == vals.end() || itPad->second.kind != VKind::InvScalar)
          return false;
        for (Operation &inOp : br.thenBlock()->without_terminator())
          if (!classifyStripElem(inOp, /*inThen=*/true))
            return false;
        Value thenY = br.thenBlock()->getTerminator()->getOperand(0);
        auto itThen = vals.find(thenY);
        if (itThen == vals.end() || itThen->second.kind != VKind::StripPanelF32)
          return false;
        roleSpecs.push_back({&bop, "strip_causal_mask", {M * S * 4}});
        vals[br.getResult(0)] = {VKind::StripPanelF32, S};
        continue;
      }

      if (auto sel = dyn_cast<arith::SelectOp>(bop)) {
        if (sel.getType().isF32() && isCausalCond(sel.getCondition(), iv)) {
          auto itT = vals.find(sel.getTrueValue());
          auto itF = vals.find(sel.getFalseValue());
          if (itT != vals.end() && itF != vals.end() &&
              itT->second.kind == VKind::StripPanelF32 &&
              itF->second.kind == VKind::InvScalar) {
            roleSpecs.push_back({&bop, "strip_causal_mask", {M * S * 4}});
            vals[sel.getResult()] = {VKind::StripPanelF32, S};
            continue;
          }
        }
      }

      if (isPureScalarArith(bop) &&
          llvm::all_of(bop.getOperands(), [&](Value u) {
            auto it = vals.find(u);
            return it != vals.end() && it->second.kind == VKind::InvScalar;
          })) {
        vals[bop.getResult(0)] = {VKind::InvScalar, 0};
        continue;
      }

      if (classifyStripElem(bop, /*inThen=*/false))
        continue;

      return false;
    }

    if (!loopHasContraction)
      return false;

    for (unsigned k = 0; k < loop.getNumResults(); ++k) {
      Operation *comb = yieldOp->getOperand(k).getDefiningOp();
      Value iterV = loop.getRegionIterArgs()[k];
      Value rhs =
          comb->getOperand(0) == iterV ? comb->getOperand(1) : comb->getOperand(0);
      auto itR = vals.find(rhs);
      if (itR == vals.end())
        return false;
      if (vals[iterV].kind == VKind::LoopScalarCarry) {
        if (itR->second.kind != VKind::StripPanelF32)
          return false;
        vals[loop.getResult(k)] = {VKind::OwnerScalar, 0};
      } else {
        if (itR->second.kind != VKind::StripOuterTerm ||
            itR->second.feat != vals[iterV].feat)
          return false;
        vals[loop.getResult(k)] = {VKind::OwnerPanelF32, vals[iterV].feat};
      }
    }

    roleSpecs.push_back({loop.getOperation(), "strip_loop", std::move(carryBytes)});
    scratchAnchors.push_back(loop.getOperation());
    return true;
  };

  for (Operation &op : bb.without_terminator()) {
    StringRef sn = op.getName().stripDialect();
    StringRef fnName = op.getName().getStringRef();

    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (!analyzeStripLoop(loop))
        return {};
      continue;
    }

    if (sn == "slice_read") {
      int64_t qRows = 0, feat = 0;
      if (!checkSlice(&op, rowArg, Value{}, qRows, feat) || qRows != rowExtent)
        return {};
      int64_t hb = halfBytes();
      roleSpecs.push_back({&op, "owner_slice", {M * feat * hb}});
      vals[op.getResult(0)] = {VKind::OwnerPanelHalf, feat};
      ownerTransferSites += 1;
      ownerTransferBytes += M * feat * hb;
      continue;
    }

    if (sn == "slice_write") {
      int64_t yRows = 0, feat = 0;
      auto itSrc = vals.find(op.getOperand(0));
      if (itSrc == vals.end() || itSrc->second.kind != VKind::OwnerPanelHalf ||
          !checkSlice(&op, rowArg, Value{}, yRows, feat) ||
          yRows != rowExtent || feat != itSrc->second.feat)
        return {};
      int64_t hb = halfBytes();
      roleSpecs.push_back({&op, "owner_write", {}});
      ownerTransferSites += 1;
      ownerTransferBytes += M * feat * hb;
      continue;
    }

    if (fnName == "arith.extf" &&
        isa<RankedTensorType>(op.getResult(0).getType())) {
      auto it = vals.find(op.getOperand(0));
      if (it == vals.end() || it->second.kind != VKind::OwnerPanelHalf ||
          !cast<RankedTensorType>(op.getResult(0).getType())
               .getElementType()
               .isF32())
        return {};
      bool onlyDotUses = !op.use_empty() &&
                         llvm::all_of(op.getUsers(), [](Operation *u) {
                           return isa<arith::MulFOp>(u) && u->hasOneUse() &&
                                  u->getUsers().begin()->getName().stripDialect() ==
                                      "lane_sum";
                         });
      if (onlyDotUses) {
        vals[op.getResult(0)] = {VKind::OwnerPanelWidenRef, it->second.feat};
      } else {
        roleSpecs.push_back({&op, "owner_widen", {M * it->second.feat * 4}});
        vals[op.getResult(0)] = {VKind::OwnerPanelF32, it->second.feat};
      }
      continue;
    }

    if (fnName == "arith.truncf" &&
        isa<RankedTensorType>(op.getResult(0).getType())) {
      auto it = vals.find(op.getOperand(0));
      if (it == vals.end() || it->second.kind != VKind::OwnerPanelF32 ||
          !halfTy ||
          cast<RankedTensorType>(op.getResult(0).getType()).getElementType() !=
              halfTy)
        return {};
      int64_t hb = halfBytes();
      roleSpecs.push_back({&op, "owner_narrow", {M * it->second.feat * hb}});
      vals[op.getResult(0)] = {VKind::OwnerPanelHalf, it->second.feat};
      continue;
    }

    if (sn == "lane_splat") {
      auto rt = dyn_cast<RankedTensorType>(op.getResult(0).getType());
      auto it = vals.find(op.getOperand(0));
      if (!rt || !rt.getElementType().isF32() || it == vals.end() ||
          (it->second.kind != VKind::InvScalar &&
           it->second.kind != VKind::OwnerScalar))
        return {};
      int64_t feat = kir::memberExtent(fn, op.getResult(0), "block.shape");
      vals[op.getResult(0)] = {VKind::OwnerSplat, feat};
      continue;
    }

    if ((fnName == "arith.mulf" || fnName == "arith.addf" ||
         fnName == "arith.subf") &&
        isa<RankedTensorType>(op.getResult(0).getType())) {
      auto rt = cast<RankedTensorType>(op.getResult(0).getType());
      auto it0 = vals.find(op.getOperand(0));
      auto it1 = vals.find(op.getOperand(1));
      if (!rt.getElementType().isF32() || it0 == vals.end() ||
          it1 == vals.end())
        return {};
      int64_t feat = kir::memberExtent(fn, op.getResult(0), "block.shape");
      auto okOp = [&](const VInfo &vi) {
        return (vi.kind == VKind::OwnerPanelF32 || vi.kind == VKind::OwnerSplat) &&
               vi.feat == feat;
      };
      if (!okOp(it0->second) || !okOp(it1->second) ||
          (it0->second.kind != VKind::OwnerPanelF32 &&
           it1->second.kind != VKind::OwnerPanelF32))
        return {};
      roleSpecs.push_back({&op, "owner_vec", {M * feat * 4}});
      vals[op.getResult(0)] = {VKind::OwnerPanelF32, feat};
      continue;
    }

    if (auto sif = dyn_cast<scf::IfOp>(op)) {
      if (sif.getNumResults() != 1 || !sif.getResult(0).getType().isF32() ||
          sif.getElseRegion().empty())
        return {};
      auto itC = vals.find(sif.getCondition());
      if (itC == vals.end() || itC->second.kind != VKind::InvScalar ||
          !checkScalarSubregion(*sif.thenBlock()) ||
          !checkScalarSubregion(*sif.elseBlock()))
        return {};
      roleSpecs.push_back({&op, "owner_scalar", {M * 4}});
      vals[sif.getResult(0)] = {VKind::OwnerScalar, 0};
      continue;
    }

    if (isPureScalarArith(op)) {
      bool allInv = true, anyOwner = false;
      for (Value u : op.getOperands()) {
        if (u == rowArg) {
          allInv = false;
          anyOwner = true;
          continue;
        }
        auto it = vals.find(u);
        if (it == vals.end())
          return {};
        if (it->second.kind == VKind::OwnerScalar) {
          allInv = false;
          anyOwner = true;
        } else if (it->second.kind != VKind::InvScalar) {
          return {};
        }
      }
      if (allInv) {
        vals[op.getResult(0)] = {VKind::InvScalar, 0};
        continue;
      }
      if (anyOwner && op.getResult(0).getType().isF32()) {
        roleSpecs.push_back({&op, "owner_scalar", {M * 4}});
        vals[op.getResult(0)] = {VKind::OwnerScalar, 0};
        continue;
      }
    }

    return {};
  }

  if (!hasContraction || !halfTy || wramBytes <= 0 || wramBytes > ms->wramBytes)
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

  llvm::DenseSet<Operation *> materialized;
  llvm::DenseMap<Operation *, StringRef> roleMap;
  for (const auto &rs : roleSpecs) {
    materialized.insert(rs.op);
    roleMap[rs.op] = rs.role;
  }

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

  unsigned scratchLo = ~0u, scratchHi = 0;
  for (Operation *top : scratchAnchors) {
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
  res.wramBytes = wramBytes;
  res.localBytes = alloc.peak;
  res.ownerTransferSites = ownerTransferSites;
  res.ownerTransferBytes = ownerTransferBytes;
  res.stripTransferSites = stripTransferSites;
  res.stripTransferBytes = stripTransferBytes;
  res.scratchOffsets = {magOff, flagOff, auxOff};
  res.roles = std::move(roleMap);
  return res;
}
} // namespace intentdsl::dsa
