#include "Quantization.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <functional>

using namespace mlir;
namespace wk = ::weft::kernel;
namespace intent::weft_provider {
namespace {

IntegerType integer(OpBuilder &b, unsigned bits, bool sign) {
  return IntegerType::get(b.getContext(), bits, sign ? IntegerType::Signed : IntegerType::Unsigned);
}

struct Microprogram {
  OpBuilder &b;
  Location loc;
  int64_t &nextAxis;

  DenseI64ArrayAttr array(ArrayRef<int64_t> values) { return b.getDenseI64ArrayAttr(values); }
  Value index(int64_t value) { return b.create<arith::ConstantIndexOp>(loc, value); }
  Value fp(double value) { return b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(value)); }
  Value integerConstant(unsigned bits, bool sign, int64_t value) {
    Type type = integer(b, bits, sign);
    return b.create<wk::ConstantOp>(loc, type, b.getIntegerAttr(type, value));
  }
  wk::EncodingType dense(Type element) {
    std::string family;
    if (auto type = dyn_cast<IntegerType>(element))
      family = (type.isUnsigned() ? "u" : "i") + std::to_string(type.getWidth());
    else { llvm::raw_string_ostream stream(family); element.print(stream); }
    return wk::EncodingType::get(b.getContext(), family, "dense", "dense." + family, array({}));
  }
  Type valueType(Type element, ArrayRef<int64_t> shape, ArrayRef<int64_t> axes) {
    if (shape.empty()) return element;
    return wk::ValueType::get(b.getContext(), element, array(shape), array(axes));
  }
  Type converted(Type type, Type element) {
    if (auto value = dyn_cast<wk::ValueType>(type))
      return valueType(element, value.getShape(), value.getAxisIds());
    return element;
  }
  Value convert(Value value, Type element) {
    Type inputElement = value.getType();
    if (auto shaped = dyn_cast<wk::ValueType>(inputElement)) inputElement = shaped.getElementType();
    auto from = dyn_cast<IntegerType>(inputElement), to = dyn_cast<IntegerType>(element);
    if (from && to && to.getWidth() > from.getWidth())
      return b.create<wk::WidenOp>(loc, converted(value.getType(), element), value);
    return b.create<wk::CastOp>(loc, converted(value.getType(), element), value);
  }
  Value binary(Value lhs, Value rhs, StringRef kind) {
    Type type = isa<wk::ValueType>(lhs.getType()) ? lhs.getType() : rhs.getType();
    return b.create<wk::BinaryOp>(loc, type, lhs, rhs, kind);
  }
  Value record(Value view, Value number) {
    auto shape = isa<wk::ViewType>(view.getType())
        ? cast<wk::ViewType>(view.getType()).getShape() : cast<wk::SliceType>(view.getType()).getShape();
    auto axes = isa<wk::ViewType>(view.getType())
        ? cast<wk::ViewType>(view.getType()).getAxisIds() : cast<wk::SliceType>(view.getType()).getAxisIds();
    auto encoding = isa<wk::ViewType>(view.getType())
        ? cast<wk::ViewType>(view.getType()).getEncoding() : cast<wk::SliceType>(view.getType()).getEncoding();
    SmallVector<int64_t> selectedShape(shape.asArrayRef()), selectedAxes(axes.asArrayRef());
    unsigned recordAxis = shape.size() - 2;
    selectedShape.erase(selectedShape.begin() + recordAxis);
    selectedAxes.erase(selectedAxes.begin() + recordAxis);
    SmallVector<Attribute> selectors(shape.size(), b.getStringAttr("all"));
    selectors[recordAxis] = b.getStringAttr("index");
    return b.create<wk::SliceOp>(loc,
        wk::SliceType::get(b.getContext(), encoding, array(selectedShape), array(selectedAxes)),
        view, ValueRange{number}, b.getArrayAttr(selectors));
  }
  Value load(Value region, Type element) {
    auto type = cast<wk::SliceType>(region.getType());
    return b.create<wk::AdmitOp>(loc, valueType(element, type.getShape(), type.getAxisIds()), region);
  }
  Value field(Value owner, StringRef name, Type element, int64_t count) {
    auto ownerShape = isa<wk::ValueType>(owner.getType())
        ? cast<wk::ValueType>(owner.getType()).getShape() : cast<wk::SliceType>(owner.getType()).getShape();
    auto ids = isa<wk::ValueType>(owner.getType())
        ? cast<wk::ValueType>(owner.getType()).getAxisIds() : cast<wk::SliceType>(owner.getType()).getAxisIds();
    SmallVector<int64_t> shape(ownerShape.asArrayRef().drop_back()), axes(ids.asArrayRef().drop_back());
    if (count) {
      shape.push_back(count);
      axes.push_back(ids.asArrayRef().back());
    }
    Type type = isa<wk::ValueType>(owner.getType()) ? valueType(element, shape, axes)
        : Type(wk::SliceType::get(b.getContext(), dense(element), array(shape), array(axes)));
    return b.create<wk::FieldOp>(loc, type, owner, name);
  }
  Value project(Value value, Value point, int64_t extent, StringRef selector) {
    bool memory = isa<wk::SliceType>(value.getType());
    auto ids = memory ? cast<wk::SliceType>(value.getType()).getAxisIds()
                      : cast<wk::ValueType>(value.getType()).getAxisIds();
    auto sourceShape = memory ? cast<wk::SliceType>(value.getType()).getShape()
                             : cast<wk::ValueType>(value.getType()).getShape();
    SmallVector<int64_t> shape(sourceShape.asArrayRef().drop_back()), axes(ids.asArrayRef().drop_back());
    if (extent) { shape.push_back(extent); axes.push_back(ids.asArrayRef().back()); }
    SmallVector<Attribute> entries(ids.size(), b.getStringAttr("all"));
    entries.back() = b.getStringAttr(selector);
    auto selectors = b.getArrayAttr(entries);
    if (memory) return b.create<wk::SliceOp>(loc,
        wk::SliceType::get(b.getContext(), cast<wk::SliceType>(value.getType()).getEncoding(), array(shape), array(axes)),
        value, ValueRange{point}, selectors);
    return b.create<wk::ExtractOp>(loc,
        valueType(cast<wk::ValueType>(value.getType()).getElementType(), shape, axes), value, ValueRange{point}, selectors);
  }
  Value gather(Value value, Value indices) {
    auto indexType = cast<wk::ValueType>(indices.getType());
    auto source = cast<wk::ValueType>(value.getType());
    SmallVector<int64_t> shape(source.getShape().asArrayRef().drop_back()), axes(source.getAxisIds().asArrayRef().drop_back());
    llvm::append_range(shape, indexType.getShape().asArrayRef());
    llvm::append_range(axes, indexType.getAxisIds().asArrayRef());
    SmallVector<Attribute> selectors(source.getShape().size(), b.getStringAttr("all"));
    selectors.back() = b.getStringAttr("gather");
    return b.create<wk::ExtractOp>(loc, valueType(source.getElementType(), shape, axes),
        value, ValueRange{indices}, b.getArrayAttr(selectors));
  }
  Value rootDomain() {
    Operation *owner = b.getInsertionBlock()->getParentOp();
    while (!isa<wk::KernelOp>(owner)) owner = owner->getParentOp();
    return *cast<wk::KernelOp>(owner).getBody().front().getOps<wk::RootDomainOp>().begin();
  }
  SmallVector<Value> level(Value parent, int64_t axis, int64_t extent, int64_t partition,
      ValueRange initial, std::function<SmallVector<Value>(Value, Value, ValueRange)> body) {
    Value size = index(extent), block = index(partition);
    Value count = b.create<arith::CeilDivUIOp>(loc, size, block);
    int64_t id = nextAxis++;
    auto parentType = cast<wk::DomainType>(parent.getType());
    auto type = wk::DomainType::get(b.getContext(), "q." + std::to_string(id), id,
        parentType.getDomainId(), axis, parentType.getDomainId() == 0 ? "blocks" : "subs", "exact");
    Value domain = b.create<wk::DomainOp>(loc, type, parent, size, block, count);
    auto level = b.create<wk::LevelOp>(loc, TypeRange(initial), domain, initial);
    auto pointType = wk::PointType::get(b.getContext(), type);
    OpBuilder::InsertionGuard guard(b);
    for (Region *region : {&level.getStateBirths(), &level.getStagedBirths()}) {
      auto *births = new Block;
      region->push_back(births);
      births->addArgument(pointType, loc);
      b.setInsertionPointToStart(births);
      b.create<wk::BirthsYieldOp>(loc, ValueRange{});
    }
    auto *computation = new Block;
    level.getBody().push_back(computation);
    Value point = computation->addArgument(pointType, loc);
    for (Value input : initial) computation->addArgument(input.getType(), loc);
    b.setInsertionPointToStart(computation);
    b.create<wk::HandoffOp>(loc, body(domain, point, computation->getArguments().drop_front()));
    return SmallVector<Value>(level.getResults().begin(), level.getResults().end());
  }
  Value reduce(Value value, StringRef kind, int64_t axis) {
    auto type = cast<wk::ValueType>(value.getType());
    auto shape = llvm::to_vector(type.getShape().asArrayRef());
    auto axes = llvm::to_vector(type.getAxisIds().asArrayRef());
    shape.erase(shape.begin() + axis);
    axes.erase(axes.begin() + axis);
    return b.create<wk::ReduceOp>(loc, valueType(type.getElementType(), shape, axes), value, kind, axis);
  }
};

// Each of the eight scale groups has its own exact i32 dot. Keep the group
// and member axes visible together so the CPU backend can map either to
// vector lanes. Floats still enter only after the record's integer reduction.
Value vectorScaledDot(Microprogram &p, Value weight, Value activation) {
  auto &b = p.b;
  Type u8 = integer(b, 8, false), i32 = integer(b, 32, true);
  int64_t group = p.nextAxis++, member = p.nextAxis++;
  auto axes = [&](ArrayRef<int64_t> shape, ArrayRef<int64_t> ids) {
    return cast<wk::ValueType>(p.valueType(u8, shape, ids));
  };
  Value groups = b.create<wk::IotaOp>(p.loc, axes({8}, {group}), 0, 8);
  Value lanes = b.create<wk::IotaOp>(p.loc, axes({32}, {member}), 0, 32);
  Value begin = p.binary(groups, p.integerConstant(8, false, 32), "mul");
  Value coordinates = b.create<wk::BinaryOp>(p.loc, axes({8, 32}, {group, member}),
                                              begin, lanes, "add");
  Value lhs = p.gather(p.field(weight, "q", integer(b, 4, false), 256), coordinates);
  Value rhs = p.gather(p.field(activation, "q", integer(b, 8, true), 256), coordinates);
  Value partial = b.create<wk::ContractOp>(p.loc, p.valueType(i32, {8}, {group}),
      lhs, rhs, p.array({member}), TypeAttr::get(i32), StringAttr());
  Value scales = p.convert(p.gather(p.field(weight, "sc", integer(b, 6, false), 8),
                                    groups), i32);
  return p.reduce(p.binary(partial, scales, "mul"), "add", 0);
}

Value matrixScaledDot(Microprogram &p, Value weight, Value activation, int64_t columns,
                      int64_t rowAxis, int64_t member, int64_t depth) {
  auto &b = p.b;
  Type i32 = integer(b, 32, true);
  Type output = p.valueType(i32, {columns}, {rowAxis});
  Value initial = b.create<wk::NewOp>(p.loc, output, p.integerConstant(32, true, 0), true);
  return p.level(p.rootDomain(), member, 256, 32, ValueRange{initial},
      [&](Value domain, Value point, ValueRange carried) {
        Value wq = p.project(p.field(weight, "q", integer(b, 4, false), 256), point, 32, "domain");
        Value xq = p.project(p.field(activation, "q", integer(b, 8, true), 256), point, 32, "domain");
        int64_t mAxis = p.nextAxis++;
        Type matrix = p.valueType(i32, {1, columns}, {mAxis, rowAxis});
        Value zero = b.create<wk::NewOp>(p.loc, matrix, p.integerConstant(32, true, 0), true);
        auto product = p.level(domain, member, 32, depth, ValueRange{zero},
            [&](Value, Value inner, ValueRange accumulator) {
              Value a = p.project(xq, inner, depth, "domain");
              Value weights = p.convert(p.project(wq, inner, depth, "domain"), integer(b, 8, true));
              Value left = b.create<wk::ReshapeOp>(p.loc,
                  p.valueType(integer(b, 8, true), {1, depth}, {mAxis, member}), a, p.array({member}));
              Value term = b.create<wk::OuterContractOp>(p.loc, matrix, left, weights,
                  p.array({member}), TypeAttr::get(i32), b.getStringAttr("matrix"));
              return SmallVector<Value>{p.binary(accumulator[0], term, "add")};
            });
        Value partial = b.create<wk::ReshapeOp>(p.loc, output, product[0], p.array({mAxis, rowAxis}));
        Value sc = p.convert(p.project(p.field(weight, "sc", integer(b, 6, false), 8),
                                        point, 0, "group_index"), i32);
        return SmallVector<Value>{p.binary(carried[0], p.binary(partial, sc, "mul"), "add")};
      })[0];
}

}

wk::EncodingType quantEncoding(OpBuilder &b, intent::QuantFormat format) {
  StringRef name = format == intent::QuantFormat::Q4K ? "Q4_K" : "Q8_K";
  return wk::EncodingType::get(b.getContext(), name, "base", name, b.getDenseI64ArrayAttr({}));
}

void declareQuantEncodings(ModuleOp module) {
  OpBuilder b(module.getContext());
  b.setInsertionPointToStart(module.getBody());
  auto natural = b.getArrayAttr({b.getDictionaryAttr({b.getNamedAttr("kind", b.getStringAttr("natural"))})});
  auto layout = [&](StringRef kind, int64_t size) {
    return b.getDictionaryAttr({b.getNamedAttr("kind", b.getStringAttr(kind)),
        b.getNamedAttr("size", b.getI64IntegerAttr(size)),
        b.getNamedAttr("order", b.getStringAttr("lo_first"))});
  };
  auto joined = [&](int64_t role) {
    return b.getArrayAttr({b.getDictionaryAttr({
        b.getNamedAttr("kind", b.getStringAttr("joined")),
        b.getNamedAttr("size", b.getI64IntegerAttr(4)),
        b.getNamedAttr("fields", b.getI64IntegerAttr(2)),
        b.getNamedAttr("low_bits", b.getI64IntegerAttr(4)),
        b.getNamedAttr("role", b.getI64IntegerAttr(role)),
        b.getNamedAttr("order", b.getStringAttr("lo_first"))})});
  };
  auto strings = [&](ArrayRef<StringRef> names) {
    SmallVector<Attribute> result;
    for (StringRef name : names) result.push_back(b.getStringAttr(name));
    return b.getArrayAttr(result);
  };
  auto types = [&](ArrayRef<Type> entries) {
    SmallVector<Attribute> result;
    for (Type type : entries) result.push_back(TypeAttr::get(type));
    return b.getArrayAttr(result);
  };
  auto dims = [&](ArrayRef<int64_t> extents) {
    SmallVector<Attribute> result;
    for (int64_t extent : extents) result.push_back(b.getDenseI64ArrayAttr(
        extent ? ArrayRef<int64_t>{extent} : ArrayRef<int64_t>{}));
    return b.getArrayAttr(result);
  };
  b.create<wk::EncodingDeclOp>(module.getLoc(), "Q4_K", "base", "Q4_K", "lsb_first", "little",
      2, 256, 1152, strings({"d", "dmin", "sc", "m", "q"}),
      types({b.getF16Type(), b.getF16Type(), integer(b, 6, false), integer(b, 6, false), integer(b, 4, false)}),
      dims({0, 0, 8, 8, 256}), b.getArrayAttr({natural, natural, joined(0), joined(1),
          b.getArrayAttr({layout("grouped", 64), layout("layered", 32)})}),
      b.getDenseI64ArrayAttr({0, 16, 32, 32, 128}), b.getDenseI64ArrayAttr({16, 16, 96, 96, 1024}),
      b.getDenseI64ArrayAttr({}));
  b.create<wk::EncodingDeclOp>(module.getLoc(), "Q8_K", "base", "Q8_K", "lsb_first", "little",
      4, 256, 2336, strings({"ds", "q", "bsum"}),
      types({b.getF32Type(), integer(b, 8, true), integer(b, 16, true)}), dims({0, 256, 16}),
      b.getArrayAttr({natural, natural, natural}), b.getDenseI64ArrayAttr({0, 32, 2080}),
      b.getDenseI64ArrayAttr({32, 2048, 256}), b.getDenseI64ArrayAttr({}));
}

LogicalResult expandQuantize(cpu::QuantizeOp operation,
                            cpu::ImplementationExpansion &expansion) {
  OpBuilder &b = expansion.builder();
  int64_t &nextAxis = expansion.nextAxis();
  auto inputView = expansion.view(operation.getInput());
  auto outputView = expansion.view(operation.getOutput());
  if (failed(inputView) || failed(outputView)) return failure();
  Value input = *inputView, output = *outputView;
  auto binding = operation->getAttrOfType<cpu::ImplementationAttr>("intent_cpu.implementation");
  if (!binding)
    return operation.emitError("Q8_K expansion requires its selected Weft implementation");
  int64_t chunk = cpu::implementationParameter(binding, "chunk");
  if (chunk != 32 && chunk != 64 && chunk != 256)
    return operation.emitError("Q8_K chunk must be 32, 64 or one full record (256)");
  Microprogram p{b, operation.getLoc(), nextAxis};
  Value groups = b.create<wk::ExtentOp>(p.loc, b.getIndexType(), input, 0);
  auto records = b.create<scf::ForOp>(p.loc, p.index(0), groups, p.index(1));
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(records.getBody());
  Value source = p.record(input, records.getInductionVar());
  Value destination = p.record(output, records.getInductionVar());
  Value values = p.load(source, b.getF32Type());
  Value maximum = p.reduce(values, "max", 0), minimum = p.reduce(values, "min", 0);
  Value a = b.create<wk::UnaryOp>(p.loc, b.getF32Type(), maximum, "abs");
  Value c = b.create<wk::UnaryOp>(p.loc, b.getF32Type(), minimum, "abs");
  Value greater = b.create<wk::CompareOp>(p.loc, b.getI1Type(), a, c, "gt");
  auto extreme = b.create<scf::IfOp>(p.loc, TypeRange{b.getF32Type()}, greater, true);
  {
    OpBuilder::InsertionGuard nested(b);
    b.setInsertionPointToStart(extreme.thenBlock()); b.create<scf::YieldOp>(p.loc, maximum);
    b.setInsertionPointToStart(extreme.elseBlock()); b.create<scf::YieldOp>(p.loc, minimum);
  }
  Value nonzero = b.create<wk::CompareOp>(p.loc, b.getI1Type(), extreme.getResult(0), p.fp(0), "ne");
  auto scale = b.create<scf::IfOp>(p.loc, TypeRange{b.getF32Type(), b.getF32Type()}, nonzero, true);
  {
    OpBuilder::InsertionGuard nested(b);
    b.setInsertionPointToStart(scale.thenBlock());
    Value inverse = p.binary(p.fp(-127), extreme.getResult(0), "div");
    b.create<scf::YieldOp>(p.loc, ValueRange{inverse, p.binary(p.fp(1), inverse, "div")});
    b.setInsertionPointToStart(scale.elseBlock());
    Value zero = p.fp(0); b.create<scf::YieldOp>(p.loc, ValueRange{zero, zero});
  }
  int64_t axis = cast<wk::SliceType>(source.getType()).getAxisIds()[0];
  Value codes = p.field(destination, "q", integer(b, 8, true), 256);
  Type sumType = integer(b, 16, true);
  Value sums = p.field(destination, "bsum", sumType, 16);
  auto encode = [&](Value current, Value output) -> Value {
    Value scaled = p.binary(current, scale.getResult(0), "mul");
    // Every finite input is bounded by the selected absolute extreme. The
    // inverse of a nonzero block is normal on the defined Q8_K domain, so the
    // two f32 roundings give |scaled| <= 127*(1+2^-24)^2 < 127.5.
    // Nearest-even conversion therefore needs no saturation.
    Value q = b.create<wk::NarrowOp>(p.loc,
        p.converted(scaled.getType(), integer(b, 8, true)), scaled, "rne", false);
    b.create<wk::CommitOp>(p.loc, q, output);
    return q;
  };
  if (chunk == 256) {
    Value q = encode(values, codes);
    // The stored sums are 16 independent reductions of 16 signed codes.
    Type grouped = p.valueType(integer(b, 8, true), {16, 16}, {axis, nextAxis++});
    Value rows = b.create<wk::ReshapeOp>(p.loc, grouped, q, p.array({axis}));
    Value totals = p.reduce(p.convert(rows, sumType), "add", 1);
    b.create<wk::CommitOp>(p.loc, totals, sums);
  } else {
    p.level(p.rootDomain(), axis, 256, chunk, {}, [&](Value domain, Value point, ValueRange) {
      Value current = p.load(p.project(source, point, chunk, "domain"), b.getF32Type());
      Value q = encode(current, p.project(codes, point, chunk, "domain"));
      p.level(domain, axis, chunk, 16, {}, [&](Value, Value group, ValueRange) {
        Value groupCodes = p.project(q, group, 16, "domain");
        Value sum = p.reduce(p.convert(groupCodes, sumType), "add", 0);
        b.create<wk::CommitOp>(p.loc, sum, p.project(sums, group, 0, "group_index"));
        return SmallVector<Value>{};
      });
      return SmallVector<Value>{};
    });
  }
  b.create<wk::CommitOp>(p.loc, scale.getResult(1), p.field(destination, "ds", b.getF32Type(), 0));
  return success();
}

LogicalResult expandQuantizedDot(cpu::QuantizedDotOp operation,
                                cpu::ImplementationExpansion &expansion) {
  OpBuilder &b = expansion.builder();
  int64_t &nextAxis = expansion.nextAxis();
  auto lhsView = expansion.view(operation.getLhs());
  auto rhsView = expansion.view(operation.getRhs());
  if (failed(lhsView) || failed(rhsView)) return failure();
  Value lhs = *lhsView, rhs = *rhsView;
  Microprogram p{b, operation.getLoc(), nextAxis};
  auto output = cast<MemRefType>(operation.getOutput().getType());
  bool grouped = output.getRank() != 0;
  int64_t columns = grouped ? output.getDimSize(0) : 1;
  auto lhsAxes = isa<wk::ViewType>(lhs.getType()) ? cast<wk::ViewType>(lhs.getType()).getAxisIds()
                                               : cast<wk::SliceType>(lhs.getType()).getAxisIds();
  int64_t rowAxis = grouped ? lhsAxes[0] : 0;
  int64_t microDepth = 0;
  if (grouped) {
    auto binding = operation->getAttrOfType<cpu::ImplementationAttr>("intent_cpu.implementation");
    if (!binding || !binding.getParameters().get("columns") ||
        !binding.getParameters().get("micro_k") ||
        cpu::implementationParameter(binding, "columns") != columns)
      return operation.emitError("grouped quantized dots require their selected matrix geometry");
    microDepth = cpu::implementationParameter(binding, "micro_k");
    if (columns <= 0 || microDepth <= 0 || 32 % microDepth)
      return operation.emitError("quantized matrix depth must partition one scale group");
  }
  auto state = [&](Type element, Value initial) -> Value {
    return grouped ? Value(b.create<wk::NewOp>(p.loc, p.valueType(element, {columns}, {rowAxis}), initial, true)) : initial;
  };
  Value groups = b.create<wk::ExtentOp>(p.loc, b.getIndexType(), lhs, grouped ? 1 : 0);
  auto records = b.create<scf::ForOp>(p.loc, p.index(0), groups, p.index(1),
      ValueRange{state(b.getF32Type(), p.fp(0))});
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(records.getBody());
    Value w = p.load(p.record(lhs, records.getInductionVar()), quantEncoding(b, intent::QuantFormat::Q4K));
    Value x = p.load(p.record(rhs, records.getInductionVar()), quantEncoding(b, intent::QuantFormat::Q8K));
    int64_t axis = cast<wk::ValueType>(w.getType()).getAxisIds().asArrayRef().back();
    Value scaled = grouped ? matrixScaledDot(p, w, x, columns, rowAxis, axis, microDepth)
                           : vectorScaledDot(p, w, x);
    Value groups = b.create<wk::IotaOp>(p.loc,
        cast<wk::ValueType>(p.valueType(integer(b, 32, false), {8}, {nextAxis++})), 0, 8);
    Value m = p.convert(p.gather(p.field(w, "m", integer(b, 6, false), 8), groups), integer(b, 32, true));
    Value bsum = p.field(x, "bsum", integer(b, 16, true), 16);
    Value even = p.binary(groups, p.integerConstant(32, false, 2), "mul");
    Value odd = p.binary(even, p.integerConstant(32, false, 1), "add");
    Value sums = p.binary(p.convert(p.gather(bsum, even), integer(b, 32, true)),
        p.convert(p.gather(bsum, odd), integer(b, 32, true)), "add");
    Value correction = p.reduce(p.binary(m, sums, "mul"), "add", grouped ? 1 : 0);
    Value positive = p.binary(p.convert(p.field(w, "d", b.getF16Type(), 0), b.getF32Type()),
        p.convert(scaled, b.getF32Type()), "mul");
    Value negative = p.binary(p.convert(p.field(w, "dmin", b.getF16Type(), 0), b.getF32Type()),
        p.convert(correction, b.getF32Type()), "mul");
    Value term = p.binary(p.field(x, "ds", b.getF32Type(), 0), p.binary(positive, negative, "sub"), "mul");
    b.create<scf::YieldOp>(p.loc, p.binary(records.getRegionIterArgs()[0], term, "add"));
  }
  return expansion.write(operation.getOutput(), records.getResult(0));
}

}
