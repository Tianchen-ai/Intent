#include "intentdsl/levels/execution/block/ir/MatrixProgram.h"
#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/execution/block/ir/TileBuilder.h"
#include "llvm/Support/JSON.h"

namespace intentdsl::block {
using namespace mlir;

void encodedTiles(Tiles &, MatrixAccess &, ScaledPanelLowering,
                  MatrixPanelRead, WindowPanelLowering, MatrixPanelWrite,
                  bool flatRecordBitPlanes);
namespace {
// Lower the chosen panel loop with a loop-carried accumulator. The pointers
// for both transposed and contiguous operands use their verified ABI strides.
void matrixTiles(Tiles &t, intentdsl::block::MatrixAccess &p,
                 MatrixPanelRead native, MatrixPanelWrite writeNative,
                 Value splitWorkspace = {}) {
  auto &b = t.b;
  auto loc = t.loc;
  auto [m, n, k] = std::tuple{p.shape[0], p.shape[1], p.shape[2]};
  auto [bm, bn, bk] = std::tuple{p.tile[0], p.tile[1], p.tile[2]};
  Value gp = t.pid(2);
  Value first = b.create<arith::MulIOp>(loc, t.pid(0), t.integer(bm));
  Value row = t.stepped(first, bm),
        col = t.stepped(b.create<arith::MulIOp>(loc, t.pid(1), t.integer(bn)),
                        bn);
  Value end = t.integer(m), groupStart, live;
  auto less = [&](Value x, Value y) -> Value {
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, x,
                                   t.like(y, x.getType()));
  };
  bool segmented = !splitWorkspace && p.dot.getNumOperands() > 2;
  scf::IfOp active;
  if (segmented) {
    auto bounds = p.dot.getOperand(2);
    auto next = b.create<arith::AddIOp>(loc, gp, t.integer(1));
    auto at = [&](Value i) -> Value {
      auto raw = t.read(bounds, t.address(bounds, i, {-1}));
      return b.create<arith::ExtSIOp>(loc, b.getI64Type(), raw);
    };
    groupStart = at(gp);
    end = at(next);
    active = b.create<scf::IfOp>(
        loc, less(b.create<arith::AddIOp>(loc, first, groupStart), end), false);
    b.setInsertionPointToStart(active.thenBlock());
    row = b.create<arith::AddIOp>(loc, row, t.like(groupStart, row.getType()));
  }
  Value validRow = less(row, end), validCol = less(col, t.integer(n)),
        dst = row, src = row;
  Value multiplicity;
  bool isRouted = !splitWorkspace && p.routed();
  if (isRouted) {
    auto route = p.dot.getOperand(3), y = p.domain.getOperand(0);
    auto ids = t.read(route, t.address(route, row, {0}), validRow);
    dst = b.create<arith::ExtSIOp>(loc, row.getType(), ids);
    multiplicity = t.like(t.integer(t.shape(y)[1]), row.getType());
    src = b.create<arith::DivUIOp>(loc, dst, multiplicity);
  }
  Value a = p.dot.getOperand(0),
        rhs = splitWorkspace ? splitWorkspace : p.dot.getOperand(1);
  Type element = cast<MemRefType>(a.getType()).getElementType().isInteger(8)
                     ? Type(b.getI32Type())
                     : Type(b.getF32Type());
  auto mt = RankedTensorType::get({bm, bn}, element);
  auto zero = b.create<arith::ConstantOp>(
      loc, mt, DenseElementsAttr::get(mt, b.getZeroAttr(element)));
  auto bound = p.fold.getAttrOfType<IntegerAttr>("bound").getInt();
  auto step = p.fold.getAttrOfType<IntegerAttr>("step").getInt();
  auto limit =
      b.create<arith::ConstantIndexOp>(loc, llvm::divideCeil(bound, step));
  auto lo = b.create<arith::ConstantIndexOp>(loc, 0),
       one = b.create<arith::ConstantIndexOp>(loc, 1);
  auto loop = b.create<scf::ForOp>(loc, lo, limit, one, ValueRange{zero});
  loop->setAttr(t.symbol("runtime_range"), b.getUnitAttr());
  b.setInsertionPointToStart(loop.getBody());
  auto panel =
      b.create<arith::IndexCastOp>(loc, b.getI64Type(), loop.getInductionVar());
  auto member =
      t.stepped(b.create<arith::MulIOp>(loc, panel, t.integer(step)), bk);
  auto panelAddress = [&](Value base, Value free, int fa, int ka, int ba,
                          int dim) {
    SmallVector<Value> coords(p.rank(base));
    SmallVector<int64_t> axes(p.rank(base), -1);
    coords[fa] = free;
    axes[fa] = dim;
    coords[ka] = member;
    axes[ka] = 1 - dim;
    if (ba >= 0)
      coords[ba] = gp;
    return t.address(base, coords, axes);
  };
  auto mask = [&](Value rows, Value columns, int64_t nr, int64_t nc) -> Value {
    auto maskType = t.shaped({nr, nc}, b.getI1Type());
    return b.create<arith::AndIOp>(loc, t.like(rows, maskType, 0),
                                   t.like(columns, maskType, 1));
  };
  int ak = splitWorkspace ? 1 : p.axis("lhs_k"),
      rk = splitWorkspace ? 0 : p.axis("rhs_k"),
      ab = splitWorkspace ? -1 : p.axis("lhs_batch"),
      rb = splitWorkspace ? -1 : p.axis("rhs_batch");
  auto km = less(member, t.integer(k));
  auto left = native ? native(t, p, true, gp, panel) : Value();
  if (!left)
    left = t.read(a, panelAddress(a, src, p.freeAxis(a, ak, ab), ak, ab, 0),
                  mask(validRow, km, bm, bk));
  auto right = native ? native(t, p, false, gp, panel) : Value();
  if (!right)
    right =
        t.read(rhs, panelAddress(rhs, col, p.freeAxis(rhs, rk, rb), rk, rb, 1),
               mask(km, validCol, bk, bn));
  auto acc = t.dot(left, right, false, loop.getRegionIterArgs()[0]);
  b.create<scf::YieldOp>(loc, acc);
  b.setInsertionPointAfter(loop);
  IRMapping map;
  map.map(p.fold.getResult(0), loop.getResult(0));
  for (auto &op : p.domain.getRegion(0).front()) {
    auto name = op.getName().stripDialect();
    if (name == "add_columns") {
      auto bias = op.getOperand(1), v = map.lookup(op.getOperand(0));
      auto lane = t.read(bias, t.address(bias, col, {0}), validCol);
      Value values = t.like(lane, v.getType(), 1);
      Value added = getElementTypeOrSelf(v.getType()).isInteger(32)
                        ? Value(b.create<arith::AddIOp>(loc, v, values))
                        : Value(b.create<arith::AddFOp>(loc, v, values));
      map.map(op.getResult(0), added);
    } else if (name == "convert_matrix") {
      Value v = map.lookup(op.getOperand(0));
      auto to = mt.clone(getElementTypeOrSelf(op.getResult(0).getType()));
      if (v.getType() != to) {
        auto from = getElementTypeOrSelf(v.getType());
        if (from.isInteger(32))
          v = b.create<arith::SIToFPOp>(loc, to, v);
        else if (to.getElementType().isInteger(32))
          v = b.create<arith::FPToSIOp>(loc, to, v);
        else if (from.getIntOrFloatBitWidth() < to.getElementTypeBitWidth())
          v = b.create<arith::ExtFOp>(loc, to, v);
        else {
          if (from.getIntOrFloatBitWidth() == to.getElementTypeBitWidth())
            v = b.create<arith::ExtFOp>(loc, mt.clone(b.getF32Type()), v);
          v = b.create<arith::TruncFOp>(loc, to, v);
        }
      }
      map.map(op.getResult(0), v);
    } else if (name == "publish_matrix") {
      auto y = op.getOperand(0), v = map.lookup(op.getOperand(1));
      SmallVector<Value, 1> batches;
      if (!segmented && !isRouted && p.rank(y) == 3)
        batches.push_back(gp);
      if (!segmented && !isRouted && writeNative &&
          writeNative(t, p, y, v, batches))
        continue;
      Value addr;
      if (isRouted) {
        Value a = b.create<arith::DivUIOp>(loc, dst, multiplicity),
              r = b.create<arith::RemUIOp>(loc, dst, multiplicity);
        addr = t.address(y, {a, r, col}, {0, 0, 1});
      } else if (p.rank(y) == 3)
        addr = t.address(y, {gp, dst, col}, {-1, 0, 1});
      else
        addr = t.address(y, {dst, col}, {0, 1});
      t.write(y, addr, v, mask(validRow, validCol, bm, bn));
    }
  }
  if (active)
    b.setInsertionPointAfter(active);
}

Value buildLaneShift(OpBuilder &b, Location loc, Tiles &t, Value subLane,
                     const intentdsl::kir::QuantFormatSpec &spec) {
  auto lit = [&](int n, Type type) {
    auto e = t.elem(type);
    auto c = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(e, n));
    return t.like(c, type);
  };
  auto castTo = [&](StringRef operation, Value a, Type e) {
    auto type = a.getType();
    if (auto r = dyn_cast<RankedTensorType>(type))
      type = r.clone(e);
    else
      type = e;
    OperationState s(loc, operation);
    s.addOperands(a);
    s.addTypes(type);
    return b.create(s)->getResult(0);
  };
  Type shTy = cast<RankedTensorType>(subLane.getType()).clone(b.getI32Type());
  if (spec.packFactor <= 1)
    return lit(spec.laneBitShifts[0], shTy);
  bool affine = true;
  int step = spec.laneBitShifts[1] - spec.laneBitShifts[0];
  for (int idx = 1; idx < spec.packFactor; ++idx)
    if (spec.laneBitShifts[idx] - spec.laneBitShifts[idx - 1] != step)
      affine = false;
  if (affine) {
    Value sub32 = castTo("arith.trunci", subLane, b.getI32Type());
    Value scaled = b.create<arith::MulIOp>(loc, sub32, lit(step, shTy));
    return spec.laneBitShifts[0] == 0
               ? scaled
               : Value(b.create<arith::AddIOp>(loc, scaled,
                                               lit(spec.laneBitShifts[0], shTy)));
  }
  if (spec.packFactor == 8) {
    int base = spec.laneBitShifts[0];
    int s0 = spec.laneBitShifts[1] - base;
    int s1 = spec.laneBitShifts[2] - base;
    bool twoLevel = true;
    for (int idx = 0; idx < 8; ++idx)
      if (spec.laneBitShifts[idx] != base + (idx & 1) * s0 + (idx >> 1) * s1)
        twoLevel = false;
    if (twoLevel) {
      Value sub32 = castTo("arith.trunci", subLane, b.getI32Type());
      Value odd = b.create<arith::AndIOp>(loc, sub32, lit(1, shTy));
      Value hi = b.create<arith::ShRUIOp>(loc, sub32, lit(1, shTy));
      Value term0 = b.create<arith::MulIOp>(loc, odd, lit(s0, shTy));
      Value term1 = b.create<arith::MulIOp>(loc, hi, lit(s1, shTy));
      Value sum = b.create<arith::AddIOp>(loc, term0, term1);
      return base == 0 ? sum
                       : Value(b.create<arith::AddIOp>(loc, sum,
                                                       lit(base, shTy)));
    }
  }
  Value sh = lit(spec.laneBitShifts[0], shTy);
  for (int idx = 1; idx < spec.packFactor; ++idx) {
    Value eq = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, subLane,
                                       lit(idx, subLane.getType()));
    sh = b.create<arith::SelectOp>(loc, eq, lit(spec.laneBitShifts[idx], shTy),
                                   sh);
  }
  return sh;
}

Value decodeGroupedRhsValues(OpBuilder &b, Location loc, Tiles &t, Value wCode,
                             Value zCode, Value scVal, Type dstElem,
                             const intentdsl::kir::QuantFormatSpec &spec,
                             int bcastAxis) {
  auto lit = [&](int n, Type type) {
    auto e = t.elem(type);
    auto c = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(e, n));
    return t.like(c, type);
  };
  auto castTo = [&](StringRef operation, Value a, Type e) {
    auto type = a.getType();
    if (auto r = dyn_cast<RankedTensorType>(type))
      type = r.clone(e);
    else
      type = e;
    OperationState s(loc, operation);
    s.addOperands(a);
    s.addTypes(type);
    return b.create(s)->getResult(0);
  };
  auto dstTy = cast<RankedTensorType>(wCode.getType()).clone(dstElem);
  bool exactHalfBias =
      dstElem.isF16() && t.elem(scVal.getType()).isF16() &&
      spec.valueCode == intentdsl::kir::QuantValueCode::UnsignedAffine &&
      spec.packBits > 0 && spec.packBits <= 10 && spec.zeroBias >= 0 &&
      spec.zeroBias <= 1023 && !(spec.hasZeroView && spec.zeroBias != 0);
  if (exactHalfBias) {
    constexpr int32_t kFp16Exp1024 = 0x6400;
    auto biasToF16 = [&](Value uCode) -> Value {
      Value biased =
          b.create<arith::OrIOp>(loc, uCode, lit(kFp16Exp1024, uCode.getType()));
      Value i16 = castTo("arith.trunci", biased, b.getI16Type());
      auto f16Ty = cast<RankedTensorType>(i16.getType()).clone(b.getF16Type());
      return b.create<arith::BitcastOp>(loc, f16Ty, i16);
    };
    Value wF16 = biasToF16(wCode);
    Value zF16;
    if (spec.hasZeroView) {
      Value zRaw = biasToF16(zCode);
      zF16 = zRaw.getType() == dstTy ? zRaw : t.like(zRaw, dstTy, bcastAxis);
    } else {
      Value zConst = b.create<arith::ConstantOp>(
          loc, b.getF16Type(),
          b.getFloatAttr(b.getF16Type(), double(1024 + spec.zeroBias)));
      zF16 = t.like(zConst, dstTy);
    }
    Value diffF16 = b.create<arith::SubFOp>(loc, wF16, zF16);
    Value scUse =
        scVal.getType() == dstTy ? scVal : t.like(scVal, dstTy, bcastAxis);
    return b.create<arith::MulFOp>(loc, diffF16, scUse);
  }
  Type i32Ty = wCode.getType();
  if (spec.hasZeroView) {
    Value zUse = zCode.getType() == i32Ty ? zCode : t.like(zCode, i32Ty, bcastAxis);
    wCode = b.create<arith::SubIOp>(loc, wCode, zUse);
  }
  if (spec.zeroBias != 0)
    wCode = b.create<arith::SubIOp>(loc, wCode, lit(spec.zeroBias, i32Ty));
  Value wF32 = castTo("arith.sitofp", wCode, b.getF32Type());
  if (!t.elem(scVal.getType()).isF32())
    scVal = castTo("arith.extf", scVal, b.getF32Type());
  Value scUse = scVal.getType() == wF32.getType()
                    ? scVal
                    : t.like(scVal, wF32.getType(), bcastAxis);
  Value prod = b.create<arith::MulFOp>(loc, wF32, scUse);
  return dstElem.isF32() ? prod : castTo("arith.truncf", prod, dstElem);
}

void splitGroupedRhsProgram(func::FuncOp fn, intentdsl::block::MatrixAccess &p,
                            StringRef unit, MatrixPanelRead native,
                            MatrixPanelWrite writeNative, bool flatSplitDequant,
                            bool permitTma, StringRef libraryGemmImpl) {
  const auto *spec = intentdsl::kir::quantFormatSpec(p.dot);
  assert(spec &&
         spec->roleLayout == intentdsl::kir::QuantRoleLayout::GroupedWeightRhs);
  auto *ctx = fn.getContext();
  Builder fb(ctx);
  auto loc = fn.getLoc();
  int64_t K = p.shape[2], N = p.shape[1], G = spec->blockElements;
  int64_t packFactor = spec->packFactor;
  int64_t NP = N / packFactor;
  int64_t numGroups = K / G;
  Value lhsArg = p.dot.getOperand(0), qwOrig = p.dot.getOperand(1),
        scOrig = p.dot.getOperand(2), outArg = p.domain.getOperand(0);
  Value qzOrig = spec->hasZeroView ? p.dot.getOperand(3) : Value();
  Type lhsElem = cast<MemRefType>(lhsArg.getType()).getElementType();
  auto wsMemTy = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                 lhsElem);

  // 1. Build Stage 1 companion function @dequant_rhs before @kernel.
  SmallVector<Type, 4> dqArgTypes{qwOrig.getType(), scOrig.getType()};
  SmallVector<int64_t, 3> dqArgSlots{
      int64_t(cast<BlockArgument>(qwOrig).getArgNumber()),
      int64_t(cast<BlockArgument>(scOrig).getArgNumber())};
  if (spec->hasZeroView) {
    dqArgTypes.push_back(qzOrig.getType());
    dqArgSlots.push_back(int64_t(cast<BlockArgument>(qzOrig).getArgNumber()));
  }
  dqArgTypes.push_back(wsMemTy);

  OpBuilder modBuilder(fn);
  auto dqFn = modBuilder.create<func::FuncOp>(
      loc, "dequant_rhs", FunctionType::get(ctx, dqArgTypes, {}));
  dqFn->setAttr("block.entry", fb.getStringAttr("dequant_rhs"));
  dqFn->setAttr("kir.domain_arg", fb.getI64IntegerAttr(dqArgTypes.size() - 1));
  for (auto key : {"denormal-fp-math", "kir.fp_contract", "kir.matmul_precision",
                   "kir.normalization", "kir.division", "ci.semantics"})
    if (auto attr = fn->getAttr(key))
      dqFn->setAttr(key, attr);
  {
    llvm::json::Array dqAbi;
    auto pushSlot = [&](StringRef name, Type memTy, StringRef dir) {
      Type el = cast<MemRefType>(memTy).getElementType();
      StringRef dt = el.isF16()         ? "f16"
                     : el.isBF16()      ? "bf16"
                     : el.isInteger(32) ? "i32"
                     : el.isInteger(8)  ? "u8"
                                        : "f32";
      dqAbi.push_back(llvm::json::Object{
          {"name", name.str()},
          {"dtype", dt.str()},
          {"direction", dir.str()},
          {"dims", llvm::json::Array{"d0", "d1"}}});
    };
    pushSlot("qw", qwOrig.getType(), "In");
    pushSlot("sc", scOrig.getType(), "In");
    if (spec->hasZeroView)
      pushSlot("qz", qzOrig.getType(), "In");
    pushSlot("ws", wsMemTy, "Out");
    std::string abiStr;
    llvm::raw_string_ostream(abiStr) << llvm::json::Value(std::move(dqAbi));
    dqFn->setAttr("kir.abi", fb.getStringAttr(abiStr));
  }
  Block *dqEntry = dqFn.addEntryBlock();
  dqFn.setArgAttr(0, "block.shape", fb.getDenseI64ArrayAttr({K, NP}));
  dqFn.setArgAttr(0, "block.strides", fb.getDenseI64ArrayAttr({NP, 1}));
  dqFn.setArgAttr(1, "block.shape", fb.getDenseI64ArrayAttr({numGroups, N}));
  dqFn.setArgAttr(1, "block.strides", fb.getDenseI64ArrayAttr({N, 1}));
  unsigned wsIdx = 2;
  if (spec->hasZeroView) {
    dqFn.setArgAttr(2, "block.shape", fb.getDenseI64ArrayAttr({numGroups, NP}));
    dqFn.setArgAttr(2, "block.strides", fb.getDenseI64ArrayAttr({NP, 1}));
    wsIdx = 3;
  }
  dqFn.setArgAttr(wsIdx, "block.shape", fb.getDenseI64ArrayAttr({K, N}));
  dqFn.setArgAttr(wsIdx, "block.strides", fb.getDenseI64ArrayAttr({N, 1}));
  {
    OpBuilder rb(dqEntry, dqEntry->end());
    rb.create<func::ReturnOp>(loc);
  }

  Value dqQw = dqFn.getArgument(0), dqSc = dqFn.getArgument(1),
        dqQz = spec->hasZeroView ? dqFn.getArgument(2) : Value(),
        dqWs = dqFn.getArgument(wsIdx);

  bool isTriton = unit == "exits/triton";
  bool isCutile = unit == "exits/cutile";
  constexpr int64_t kFlatBlock = 1024;
  constexpr int64_t kTileK = 16, kTileN = 256;
  bool useFlat1D = isTriton && flatSplitDequant;
  bool can2DRect =
      !useFlat1D && K % kTileK == 0 && N % kTileN == 0 && G % kTileK == 0 &&
      kTileN % packFactor == 0 && llvm::isPowerOf2_64(kTileN / packFactor);
  // If 2D structured tile isn't divisible, fall back cleanly to masked 1D flat.
  if (!can2DRect)
    useFlat1D = true;

  StringRef dialectName = fn.front().front().getName().getDialectNamespace();
  Tiles dt(dqFn, "split_dequant", unit, dialectName, ValueRange{dqWs});
  dt.program->setAttr("block.dequant_extents", fb.getDenseI64ArrayAttr({K, N}));
  auto &db = dt.b;
  auto dlit = [&](int64_t n, Type type) {
    auto e = dt.elem(type);
    auto c = db.create<arith::ConstantOp>(loc, db.getIntegerAttr(e, n));
    return dt.like(c, type);
  };
  auto dcast = [&](StringRef operation, Value a, Type e) {
    auto type = a.getType();
    if (auto r = dyn_cast<RankedTensorType>(type))
      type = r.clone(e);
    else
      type = e;
    OperationState s(loc, operation);
    s.addOperands(a);
    s.addTypes(type);
    return db.create(s)->getResult(0);
  };
  auto readPacked = [&](Value view, Value ptr, Value mask, bool evictLast) {
    Value raw = mask ? dt.read(view, ptr, mask) : dt.read(view, ptr);
    if (evictLast && isTriton)
      raw.getDefiningOp()->setAttr("eviction_policy",
                                   db.getStringAttr("evict_last"));
    if (dt.elem(raw.getType()).isInteger(32))
      return raw;
    Value u = dcast("arith.extui", raw, db.getI32Type());
    return Value(db.create<arith::AndIOp>(loc, u, dlit(255, u.getType())));
  };
  int codeMaskVal = (1 << spec->packBits) - 1;
  int64_t preGridX = 1, preGridY = 1, preRows = 1, preWidth = 1;

  if (useFlat1D) {
    dt.program->setAttr("block.dequant_tile",
                        fb.getDenseI64ArrayAttr({kFlatBlock, 1}));
    int64_t total = K * N;
    preGridX = llvm::divideCeil(total, kFlatBlock);
    preGridY = 1;
    preRows = kFlatBlock;
    preWidth = 1;
    bool exactTotal = (total % kFlatBlock) == 0;
    Value base =
        db.create<arith::MulIOp>(loc, dt.pid(0), dt.integer(kFlatBlock));
    Value x = dt.stepped(base, kFlatBlock);
    Type xTy = x.getType();
    Value valid = exactTotal
                      ? Value()
                      : Value(db.create<arith::CmpIOp>(
                            loc, arith::CmpIPredicate::slt, x,
                            dt.like(dt.integer(total), xTy)));
    Value kIdx = db.create<arith::DivUIOp>(loc, x, dlit(N, xTy));
    Value nIdx = db.create<arith::RemUIOp>(loc, x, dlit(N, xTy));
    Value gIdx = db.create<arith::DivUIOp>(loc, kIdx, dlit(G, xTy));
    Value npIdx = db.create<arith::DivUIOp>(loc, nIdx, dlit(packFactor, xTy));
    Value lane = db.create<arith::RemUIOp>(loc, nIdx, dlit(packFactor, xTy));
    Value sh = buildLaneShift(db, loc, dt, lane, *spec);
    Value qwRaw = readPacked(dqQw, dt.address(dqQw, {kIdx, npIdx}, {0, 0}),
                             valid, /*evictLast=*/true);
    Value wCode = db.create<arith::AndIOp>(
        loc, db.create<arith::ShRUIOp>(loc, qwRaw, sh),
        dlit(codeMaskVal, qwRaw.getType()));
    Value zCode;
    if (spec->hasZeroView) {
      Value qzRaw = readPacked(dqQz, dt.address(dqQz, {gIdx, npIdx}, {0, 0}),
                               valid, /*evictLast=*/true);
      zCode = db.create<arith::AndIOp>(
          loc, db.create<arith::ShRUIOp>(loc, qzRaw, sh),
          dlit(codeMaskVal, qzRaw.getType()));
    }
    Value scAddr = dt.address(dqSc, {gIdx, nIdx}, {0, 0});
    Value scVal = valid ? dt.read(dqSc, scAddr, valid) : dt.read(dqSc, scAddr);
    if (isTriton)
      scVal.getDefiningOp()->setAttr("eviction_policy",
                                     db.getStringAttr("evict_last"));
    Value decoded = decodeGroupedRhsValues(db, loc, dt, wCode, zCode, scVal,
                                           lhsElem, *spec, /*bcastAxis=*/0);
    dt.write(dqWs, dt.address(dqWs, {kIdx, nIdx}, {0, 0}), decoded, valid);
  } else {
    dt.program->setAttr("block.dequant_tile",
                        fb.getDenseI64ArrayAttr({kTileK, kTileN}));
    int64_t bnPack = kTileN / packFactor;
    preGridX = K / kTileK;
    preGridY = N / kTileN;
    preRows = kTileK;
    preWidth = kTileN;
    Value pk = dt.pid(0), pn = dt.pid(1);
    Value kBase = db.create<arith::MulIOp>(loc, pk, dt.integer(kTileK));
    Value gScalar = db.create<arith::DivUIOp>(loc, kBase, dt.integer(G));
    auto reshapeT = [&](Value v, ArrayRef<int64_t> sh) {
      return dt.op("reshape", v, dt.shaped(sh, dt.elem(v.getType())))
          ->getResult(0);
    };
    auto expandAx = [&](Value v, ArrayRef<int64_t> sh, int64_t ax) {
      return dt
          .op("expand_scale", v, dt.shaped(sh, dt.elem(v.getType())),
              {db.getNamedAttr("axis", db.getI64IntegerAttr(ax))})
          ->getResult(0);
    };
    Value qwCompact, qzCompact1D, sc1D;
    if (isCutile) {
      Type qwEl = cast<MemRefType>(dqQw.getType()).getElementType();
      qwCompact =
          dt.op("rect_read", {dqQw, pk, pn}, dt.shaped({kTileK, bnPack}, qwEl),
                {db.getNamedAttr("allow_tma", db.getBoolAttr(permitTma))})
              ->getResult(0);
      if (!qwEl.isInteger(32)) {
        Value u = dcast("arith.extui", qwCompact, db.getI32Type());
        qwCompact = db.create<arith::AndIOp>(loc, u, dlit(255, u.getType()));
      }
      if (spec->hasZeroView) {
        Type qzEl = cast<MemRefType>(dqQz.getType()).getElementType();
        Value qz2D =
            dt.op("rect_read", {dqQz, gScalar, pn}, dt.shaped({1, bnPack}, qzEl),
                  {db.getNamedAttr("allow_tma", db.getBoolAttr(false))})
                ->getResult(0);
        qzCompact1D = reshapeT(qz2D, {bnPack});
        if (!qzEl.isInteger(32)) {
          Value u = dcast("arith.extui", qzCompact1D, db.getI32Type());
          qzCompact1D =
              db.create<arith::AndIOp>(loc, u, dlit(255, u.getType()));
        }
      }
      Type scEl = cast<MemRefType>(dqSc.getType()).getElementType();
      Value sc2D =
          dt.op("rect_read", {dqSc, gScalar, pn}, dt.shaped({1, kTileN}, scEl),
                {db.getNamedAttr("allow_tma", db.getBoolAttr(false))})
              ->getResult(0);
      sc1D = reshapeT(sc2D, {kTileN});
    } else {
      Value kOffs = dt.stepped(kBase, kTileK);
      Value npBase = db.create<arith::MulIOp>(loc, pn, dt.integer(bnPack));
      Value npOffs = dt.stepped(npBase, bnPack);
      Value nBase = db.create<arith::MulIOp>(loc, pn, dt.integer(kTileN));
      Value nOffs = dt.stepped(nBase, kTileN);
      qwCompact = readPacked(dqQw, dt.address(dqQw, {kOffs, npOffs}, {0, 1}),
                             Value(), /*evictLast=*/true);
      if (spec->hasZeroView)
        qzCompact1D =
            readPacked(dqQz, dt.address(dqQz, {gScalar, npOffs}, {-1, 0}),
                       Value(), /*evictLast=*/true);
      sc1D = dt.read(dqSc, dt.address(dqSc, {gScalar, nOffs}, {-1, 0}));
      sc1D.getDefiningOp()->setAttr("eviction_policy",
                                    db.getStringAttr("evict_last"));
    }
    Value qwRaw = reshapeT(
        expandAx(qwCompact, {kTileK, bnPack, packFactor}, 2), {kTileK, kTileN});
    Value colLanes = dt.lanes(kTileN);
    Value subLane =
        db.create<arith::RemUIOp>(loc, colLanes, dlit(packFactor, colLanes.getType()));
    Value sh1D = buildLaneShift(db, loc, dt, subLane, *spec);
    Type rhsI32Ty = dt.shaped({kTileK, kTileN}, db.getI32Type());
    Value wCode = db.create<arith::AndIOp>(
        loc, db.create<arith::ShRUIOp>(loc, qwRaw, dt.like(sh1D, rhsI32Ty, 1)),
        dlit(codeMaskVal, rhsI32Ty));
    Value zCode1D;
    if (spec->hasZeroView) {
      Value qz1D = reshapeT(expandAx(qzCompact1D, {bnPack, packFactor}, 1),
                            {kTileN});
      zCode1D = db.create<arith::AndIOp>(
          loc, db.create<arith::ShRUIOp>(loc, qz1D, sh1D),
          dlit(codeMaskVal, qz1D.getType()));
    }
    Value decoded = decodeGroupedRhsValues(db, loc, dt, wCode, zCode1D, sc1D,
                                           lhsElem, *spec, /*bcastAxis=*/1);
    if (isCutile) {
      dt.op("rect_write", {dqWs, decoded, pk, pn});
    } else {
      Value kOffs = dt.stepped(kBase, kTileK);
      Value nOffs =
          dt.stepped(db.create<arith::MulIOp>(loc, pn, dt.integer(kTileN)),
                     kTileN);
      dt.write(dqWs, dt.address(dqWs, {kOffs, nOffs}, {0, 1}), decoded);
    }
  }
  dt.op("end");

  // 2. Append workspace argument to @kernel and lower Stage 2 GEMM.
  unsigned mainWsSlot = fn.getNumArguments();
  fn.insertArgument(mainWsSlot, wsMemTy, {}, loc);
  fn.setArgAttr(mainWsSlot, "block.shape", fb.getDenseI64ArrayAttr({K, N}));
  fn.setArgAttr(mainWsSlot, "block.strides", fb.getDenseI64ArrayAttr({N, 1}));
  Value mainWs = fn.getArgument(mainWsSlot);

  Tiles tiles(fn, "matrix", unit);
  if (p.domain.hasAttr("block.split_library")) {
    tiles.program->setAttr("block.split_library", fb.getUnitAttr());
    fn->setAttr("block.library_gemm", fb.getUnitAttr());
    SmallVector<NamedAttribute, 1> implAttrs;
    if (!libraryGemmImpl.empty())
      implAttrs.push_back(
          fb.getNamedAttr("ci.implementation", fb.getStringAttr(libraryGemmImpl)));
    tiles.op("library_gemm", {lhsArg, mainWs, outArg}, {}, implAttrs);
  } else {
    matrixTiles(tiles, p, native, writeNative, mainWs);
  }
  tiles.op("end");

  // 3. Stamp pre-stage manifest JSON metadata on @kernel.
  llvm::json::Array slotArr;
  for (int64_t s : dqArgSlots)
    slotArr.push_back(s);
  llvm::json::Object preNative =
      isCutile ? llvm::json::Object{{"num_ctas", 1}, {"occupancy", 4}, {"opt_level", 3}}
               : llvm::json::Object{{"num_warps", 4}, {"num_stages", 1}};
  StringRef wsDtype = lhsElem.isF16()    ? "f16"
                      : lhsElem.isBF16() ? "bf16"
                                         : "f32";
  llvm::json::Object preObj{
      {"entry", "dequant_rhs"},
      {"workspace",
       llvm::json::Object{{"shape", llvm::json::Array{K, N}},
                          {"dtype", wsDtype.str()}}},
      {"arg_slots", std::move(slotArr)},
      {"grid", llvm::json::Array{preGridX, preGridY, 1}},
      {"rows", preRows},
      {"width", preWidth},
      {"native", std::move(preNative)},
      {"lhs_slot", int64_t(cast<BlockArgument>(lhsArg).getArgNumber())},
      {"out_slot", int64_t(cast<BlockArgument>(outArg).getArgNumber())}};
  std::string preStr;
  llvm::raw_string_ostream(preStr) << llvm::json::Value(std::move(preObj));
  fn->setAttr("block.pre_stage", fb.getStringAttr(preStr));
  fn->setAttr("block.split_dequant", fb.getUnitAttr());
}

} // namespace
void expandMatrixProgram(func::FuncOp fn, StringRef unit,
                         ScaledPanelLowering scaled, MatrixPanelRead native,
                         WindowPanelLowering window, MatrixPanelWrite write,
                         bool flatRecordBitPlanes, bool flatSplitDequant,
                         bool permitTma, StringRef libraryGemmImpl) {
  fn.getContext()
      ->loadDialect<arith::ArithDialect, scf::SCFDialect, math::MathDialect>();
  auto &root = fn.front().front();
  if (intentdsl::kir::matrixDomain(fn)) {
    intentdsl::block::MatrixAccess p(fn);
    if (p.domain.hasAttr("block.split_dequant")) {
      splitGroupedRhsProgram(fn, p, unit, native, write, flatSplitDequant,
                             permitTma, libraryGemmImpl);
    } else {
      Tiles tiles(fn, "matrix", unit);
      if (p.dot.getName().stripDialect() == "contract")
        matrixTiles(tiles, p, native, write);
      else
        encodedTiles(tiles, p, scaled, native, window, write,
                     flatRecordBitPlanes);
      tiles.op("end");
    }
    root.erase();
  }
}
} // namespace intentdsl::block
