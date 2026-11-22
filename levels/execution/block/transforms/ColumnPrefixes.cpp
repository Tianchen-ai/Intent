#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
using namespace mlir;
namespace intentdsl::block {
namespace {
bool indexes(ValueRange actual, ArrayRef<Value> expected) {
  return actual == ValueRange(expected);
}
SmallVector<Operation *> arithmetic(Block &bb) {
  SmallVector<Operation *> ops;
  for (auto &o : bb.without_terminator())
    if (!isa<arith::ConstantOp, arith::AddIOp, memref::DimOp>(o))
      ops.push_back(&o);
  return ops;
}
bool sizeOf(Value n, Value matrix) {
  auto dim = n.getDefiningOp<memref::DimOp>();
  return dim && dim.getSource() == matrix && dim.getConstantIndex() == 1;
}
bool prefix(scf::ForOp sum, Value view, Value batch, Value row, Value col) {
  if (!sum || !matchPattern(sum.getLowerBound(), m_Zero()) ||
      !matchPattern(sum.getStep(), m_One()) || sum.getUpperBound() != col ||
      sum.getNumRegionIterArgs() != 1 ||
      !matchPattern(sum.getInitArgs()[0], m_PosZeroFloat()))
    return false;
  Value k = sum.getInductionVar(), carry = sum.getRegionIterArgs()[0];
  auto ops = arithmetic(*sum.getBody());
  if (ops.size() != (row == col ? 3 : 4))
    return false;
  auto add = dyn_cast<arith::AddFOp>(ops.back());
  auto mul = dyn_cast<arith::MulFOp>(ops[ops.size() - 2]);
  if (!add || !mul || add.getLhs() != carry ||
      add.getRhs() != mul.getResult() ||
      sum.getBody()->getTerminator()->getOperand(0) != add.getResult())
    return false;
  auto l = mul.getLhs().getDefiningOp<memref::LoadOp>(),
       r = mul.getRhs().getDefiningOp<memref::LoadOp>();
  return l && r && l.getMemref() == view && r.getMemref() == view &&
         indexes(l.getIndices(), {batch, row, k}) &&
         indexes(r.getIndices(), {batch, col, k});
}
} // namespace
bool columnPrefixes(Operation *space, const compilerinfra::TargetProfile &p) {
  auto machine = p.services.get<OwnerMachine>();
  if (!machine || !machine->residentElements || space->getNumOperands() != 1 ||
      space->getParentOfType<func::FuncOp>().getNumArguments() != 1 ||
      space->getAttrOfType<DenseI64ArrayAttr>("axes")[0] != 0)
    return false;
  auto view = space->getOperand(0);
  auto shape = p.services.get<CallFacts>()->shapeOf(view);
  if (!shape || shape->size() != 3 ||
      !cast<MemRefType>(view.getType()).getElementType().isF32())
    return false;
  auto n = (*shape)[1].getAsInteger().value_or(0);
  if (n <= 0 || (*shape)[2].getAsInteger() != n ||
      n > machine->residentElements / n || !llvm::isPowerOf2_64(n))
    return false;
  auto &body = space->getRegion(0).front();
  auto outer = arithmetic(body);
  if (outer.size() != 1)
    return false;
  auto columns = dyn_cast<scf::ForOp>(outer[0]);
  if (!columns || columns.getNumResults() ||
      !matchPattern(columns.getLowerBound(), m_Zero()) ||
      !matchPattern(columns.getStep(), m_One()) ||
      !sizeOf(columns.getUpperBound(), view))
    return false;
  Value c = columns.getInductionVar(), batch = body.getArgument(0);
  auto steps = arithmetic(*columns.getBody());
  if (steps.size() != 6)
    return false;
  auto diagSum = dyn_cast<scf::ForOp>(steps[0]);
  auto read = dyn_cast<memref::LoadOp>(steps[1]);
  auto sub = dyn_cast<arith::SubFOp>(steps[2]);
  auto root = dyn_cast<math::SqrtOp>(steps[3]);
  auto diag = dyn_cast<memref::StoreOp>(steps[4]);
  auto rows = dyn_cast<scf::ForOp>(steps[5]);
  if (!prefix(diagSum, view, batch, c, c) || !read ||
      read.getMemref() != view || !indexes(read.getIndices(), {batch, c, c}) ||
      !sub || sub.getLhs() != read.getResult() ||
      sub.getRhs() != diagSum.getResult(0) || !root ||
      root.getOperand() != sub.getResult() || !diag ||
      diag.getMemref() != view || !indexes(diag.getIndices(), {batch, c, c}) ||
      diag.getValue() != root.getResult() || !rows || rows.getNumResults() ||
      !sizeOf(rows.getUpperBound(), view) ||
      !matchPattern(rows.getStep(), m_One()))
    return false;
  auto begin = rows.getLowerBound().getDefiningOp<arith::AddIOp>();
  if (!begin || begin.getLhs() != c || !matchPattern(begin.getRhs(), m_One()))
    return false;
  auto r = rows.getInductionVar();
  auto row = arithmetic(*rows.getBody());
  if (row.size() != 6)
    return false;
  auto sum = dyn_cast<scf::ForOp>(row[0]);
  auto load = dyn_cast<memref::LoadOp>(row[1]);
  auto diff = dyn_cast<arith::SubFOp>(row[2]);
  auto div = dyn_cast<arith::DivFOp>(row[3]);
  auto lower = dyn_cast<memref::StoreOp>(row[4]),
       upper = dyn_cast<memref::StoreOp>(row[5]);
  return prefix(sum, view, batch, r, c) && load && load.getMemref() == view &&
         indexes(load.getIndices(), {batch, r, c}) && diff &&
         diff.getLhs() == load.getResult() &&
         diff.getRhs() == sum.getResult(0) && div &&
         div.getLhs() == diff.getResult() && div.getRhs() == root.getResult() &&
         lower && lower.getMemref() == view &&
         indexes(lower.getIndices(), {batch, r, c}) &&
         lower.getValue() == div.getResult() && upper &&
         upper.getMemref() == view &&
         indexes(upper.getIndices(), {batch, c, r}) &&
         matchPattern(upper.getValue(), m_PosZeroFloat());
}
void realizeColumns(func::FuncOp fn) {
  auto &space = fn.front().front();
  auto view = space.getOperand(0);
  auto &bb = space.getRegion(0).front();
  auto batch = bb.getArgument(0);
  int64_t n = fn.getArgAttrOfType<DenseI64ArrayAttr>(0, "block.shape")[1];
  auto *end = &bb.back();
  while (auto *op = end->getPrevNode())
    op->erase();
  OpBuilder b(end);
  auto loc = space.getLoc();
  auto f = b.getF32Type();
  auto mat = RankedTensorType::get({n, n}, f),
       col = RankedTensorType::get({n}, f);
  auto inst = [&](StringRef name, ValueRange args, TypeRange ts) {
    OperationState s(loc, "block." + name.str());
    s.addOperands(args);
    s.addTypes(ts);
    return b.create(s);
  };
  auto data = inst("read_owner_matrix", {view, batch}, mat)->getResult(0);
  auto init = b.create<arith::ConstantOp>(
      loc, mat, DenseElementsAttr::get(mat, b.getF32FloatAttr(0)));
  auto lo = b.create<arith::ConstantIndexOp>(loc, 0),
       hi = b.create<arith::ConstantIndexOp>(loc, n),
       one = b.create<arith::ConstantIndexOp>(loc, 1);
  auto loop = b.create<scf::ForOp>(loc, lo, hi, one, ValueRange{init});
  b.setInsertionPointToStart(loop.getBody());
  auto c = loop.getInductionVar();
  auto a = inst("column", {data, c}, col)->getResult(0),
       prefix =
           inst("column", {loop.getRegionIterArgs()[0], c}, col)->getResult(0);
  auto residual = b.create<arith::SubFOp>(loc, a, prefix);
  auto pivot = b.create<math::SqrtOp>(
      loc, inst("element", {residual, c}, f)->getResult(0));
  auto values = inst("lower_column", {residual, pivot, c}, col)->getResult(0);
  inst("store_column", {view, batch, c, values}, {});
  auto update = inst("outer_product", {values, values}, mat)->getResult(0);
  auto sum = b.create<arith::AddFOp>(loc, loop.getRegionIterArgs()[0], update);
  b.create<scf::YieldOp>(loc, ValueRange{sum});
}
LogicalResult verifyColumns(func::FuncOp fn, StringRef ns) {
  auto &space = fn.front().front();
  auto &bb = space.getRegion(0).front();
  auto abi = fn->getAttrOfType<StringAttr>("kir.abi");
  if (!abi)
    return space.emitError("missing column ABI");
  auto description = llvm::json::parse(abi);
  if (!description) {
    llvm::consumeError(description.takeError());
    return space.emitError("malformed column ABI");
  }
  auto *slots = description->getAsArray();
  if (!slots || slots->size() != 1 || !slots->front().getAsObject() ||
      slots->front().getAsObject()->getString("direction") != "InOut" ||
      slots->front().getAsObject()->getString("dtype") != "f32")
    return space.emitError("column updates require one writable f32 matrix");
  if (space.getName().getDialectNamespace() != ns ||
      fn.getNumArguments() != 1 || bb.getNumArguments() != 1 ||
      space.getNumOperands() != 1 || space.getOperand(0) != fn.getArgument(0))
    return space.emitError("invalid triangular owner");
  auto n = fn.getArgAttrOfType<DenseI64ArrayAttr>(0, "block.shape");
  if (!n || n.size() != 3 || n[1] != n[2])
    return failure();
  for (auto &op : bb) {
    if (op.getName().stripDialect() == "read_owner_matrix" &&
        (!indexes(op.getOperands(), {fn.getArgument(0), bb.getArgument(0)}) ||
         cast<RankedTensorType>(op.getResult(0).getType()).getShape() !=
             ArrayRef<int64_t>{n[1], n[1]}))
      return op.emitError("incorrect resident matrix projection");
  }
  auto steps = arithmetic(bb);
  if (steps.size() != 2 ||
      steps[0]->getName().stripDialect() != "read_owner_matrix")
    return space.emitError("missing matrix snapshot");
  auto loop = dyn_cast<scf::ForOp>(steps[1]);
  if (!loop || !matchPattern(loop.getLowerBound(), m_Zero()) ||
      !matchPattern(loop.getStep(), m_One()) ||
      !loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>() ||
      loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>().value() !=
          n[1] ||
      loop.getInitArgs().size() != 1 ||
      !matchPattern(loop.getInitArgs()[0], m_AnyZeroFloat()))
    return space.emitError("incorrect prefix traversal");
  auto ops = arithmetic(*loop.getBody());
  SmallVector<StringRef> expected = {
      "column",       "column",       "subf",          "element", "sqrt",
      "lower_column", "store_column", "outer_product", "addf"};
  if (ops.size() != expected.size())
    return loop.emitError("invalid prefix update");
  for (auto [o, name] : llvm::zip(ops, expected))
    if (o->getName().stripDialect() != name)
      return o->emitError("incorrect pivot update");
  auto f = Float32Type::get(fn.getContext());
  auto mat = RankedTensorType::get({n[1], n[1]}, f),
       col = RankedTensorType::get({n[1]}, f);
  if (steps[0]->getResult(0).getType() != mat ||
      loop.getRegionIterArgs()[0].getType() != mat ||
      ops[0]->getResult(0).getType() != col ||
      ops[1]->getResult(0).getType() != col ||
      ops[5]->getResult(0).getType() != col ||
      ops[7]->getResult(0).getType() != mat)
    return space.emitError("incorrect column or outer-product shape");
  auto r = [&](int i) { return ops[i]->getResult(0); };
  Value ix = loop.getInductionVar(), carry = loop.getRegionIterArgs()[0];
  if (!indexes(ops[0]->getOperands(), {steps[0]->getResult(0), ix}) ||
      !indexes(ops[1]->getOperands(), {carry, ix}) ||
      !indexes(ops[2]->getOperands(), {r(0), r(1)}) ||
      !indexes(ops[3]->getOperands(), {r(2), ix}) ||
      ops[4]->getOperand(0) != r(3) ||
      !indexes(ops[5]->getOperands(), {r(2), r(4), ix}) ||
      !indexes(ops[6]->getOperands(),
               {fn.getArgument(0), bb.getArgument(0), ix, r(5)}) ||
      !indexes(ops[7]->getOperands(), {r(5), r(5)}) ||
      !indexes(ops[8]->getOperands(), {carry, r(7)}) ||
      loop.getBody()->getTerminator()->getOperand(0) != r(8))
    return loop.emitError("prefix dependence or column publication differs");
  return success();
}
} // namespace intentdsl::block
