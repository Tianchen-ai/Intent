#include "Intent/Target/DSA/BangC/Microkernels.h"
#include "../Transforms/PassDetail.h"

using namespace mlir;
namespace intent::bangc {
void realizeRowSumChannels(func::FuncOp function, physical::ConfigurationAttr config) {
  SmallVector<physical::ReduceOp> operations;
  function.walk([&](physical::ReduceOp reduce) {
    auto input = cast<MemRefType>(reduce.getInput().getType());
    if (reduce.getKind() == BinaryOperator::Add && reduce.getAxis() == 1 &&
        input.getDimSize(0) >= 32 && input.getDimSize(0) % 32 == 0 && input.getDimSize(1) >= 32)
      operations.push_back(reduce);
  });
  for (auto reduce : operations) {
    auto input = cast<MemRefType>(reduce.getInput().getType());
    OpBuilder b(reduce);
    Location loc = reduce.getLoc();
    Value storage = allocate(b, loc, input.getElementType(),
        {input.getDimSize(1), input.getDimSize(0)}, physical::nramSpace);
    Value rows = b.create<arith::ConstantIndexOp>(loc, input.getDimSize(0));
    auto transpose = b.create<physical::TransposeOp>(loc, reduce.getInput(), storage, rows, reduce.getCount());
    auto replacement = b.create<physical::ReduceOp>(loc, storage, reduce.getOutput(), reduce.getScratch(),
        reduce.getCount(), reduce.getIdentity(), reduce.getKindAttr(), b.getI64IntegerAttr(0));
    int64_t nram = 0, wram = 0;
    measureStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > capabilities(function).getNramBytes()) {
      replacement.erase(); transpose.erase(); storage.getDefiningOp()->erase();
    } else reduce.erase();
  }
}
}
