#include "Intent/Dialect/GPU/Serialization/PythonEmitter.h"
#include "Intent/Dialect/GPU/Analysis/ProgramInterface.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace intent::gpu {

PythonEmitter::PythonEmitter(func::FuncOp kernel, raw_ostream &output,
                             PythonScalarSyntax scalarSyntax,
                             PythonExpressionSyntax expressionSyntax,
                             StringRef constexprAnnotation)
    : SourceEmitter(output), kernel(kernel), scalarSyntax(scalarSyntax),
      expressionSyntax(expressionSyntax), constexprAnnotation(constexprAnnotation) {}

void PythonEmitter::addCommonOperations(Emitters &emitters) {
  auto valid = [](auto) { return success(); };
  emitters.add<arith::ConstantOp>([](arith::ConstantOp op) -> LogicalResult {
    if (!isa<IntegerAttr, FloatAttr>(op.getValue()))
      return op.emitOpError("Python GPU constants require a scalar integer or floating value");
    return success();
  }, [](arith::ConstantOp op, PythonEmitter &e) {
    e.emitConstant(op); return success();
  });
  emitters.add<ParameterOp>(valid, [](ParameterOp op, PythonEmitter &e) {
    e.bind(op.getResult(), op.getReference().getName().getValue());
    e.constexprValues.insert(op.getResult()); return success();
  });
  emitters.add<PhysicalExprOp>(valid, [](PhysicalExprOp op, PythonEmitter &e) {
    e.assign(op.getResult(), e.expressionString(op.getExpression()),
             isConstexprPhysicalExpression(op.getExpression())); return success();
  });
  emitters.add<ProgramIdOp>(valid, [](ProgramIdOp op, PythonEmitter &e) {
    e.assign(op.getResult(), e.programId(op)); return success();
  });
  emitters.add<WorksetCoordinateOp>(valid, [](WorksetCoordinateOp op, PythonEmitter &e) {
    e.bind(op.getResult(), e.valueString(op.getCoordinate())); return success();
  });
  emitters.add<DimOp>(valid, [](DimOp op, PythonEmitter &e) {
    auto extent = cast<PhysicalExprAttr>(op.getView().getType().getLayout().getExtents()[op.getAxis()]);
    e.assign(op.getResult(), e.expressionString(extent), isConstexprPhysicalExpression(extent));
    return success();
  });
  emitters.add<RangeOp>(valid, [](RangeOp op, PythonEmitter &e) {
    e.assign(op.getResult(), e.tuple(op->getOperands())); return success();
  });
  emitters.add<RangeBoundOp>(valid, [](RangeBoundOp op, PythonEmitter &e) {
    e.assign(op.getResult(), e.valueString(op.getRange()) + "[" + std::to_string(op.getBound()) + "]");
    return success();
  });
  emitters.add<DelinearizeOp>(valid, [](DelinearizeOp op, PythonEmitter &e) {
    std::string remaining = e.valueString(op.getLinear());
    for (int64_t axis = static_cast<int64_t>(op.getCoordinates().size()) - 1; axis >= 0; --axis) {
      std::string extent = e.valueString(op.getExtents()[axis]);
      std::string quotient = e.newName();
      e.line(quotient + " = " + e.expressionSyntax.integerDivision(
          PhysicalExprKind::FloorDiv, remaining, extent));
      e.assign(op.getCoordinates()[axis],
               "(" + remaining + " - " + quotient + " * " + extent + ")");
      remaining = quotient;
    }
    return success();
  });
  emitters.add<MakeRangeOp>(valid, [](MakeRangeOp op, PythonEmitter &e) {
    e.assign(op.getResult(), e.makeRange(op)); return success();
  });
  emitters.add<SplatOp>(valid, [](SplatOp op, PythonEmitter &e) {
    e.assign(op.getResult(), e.splat(op)); return success();
  });
  emitters.add<BroadcastOp>(valid, [](BroadcastOp op, PythonEmitter &e) {
    e.assign(op.getResult(), e.broadcastValue(op.getValue(), op.getResult().getType()));
    return success();
  });
  emitters.add<ReshapeOp>(valid, [](ReshapeOp op, PythonEmitter &e) {
    e.assign(op.getResult(), e.reshape(op)); return success();
  });
  emitters.add<TransposeOp>(valid, [](TransposeOp op, PythonEmitter &e) {
    e.assign(op.getResult(), e.permute(op.getValue(), op.getPermutation())); return success();
  });
  emitters.add<JoinOp>(valid, [](JoinOp op, PythonEmitter &e) {
    e.assign(op.getResult(), e.join(op)); return success();
  });
  emitters.add<MakeRecordOp>(valid, [](MakeRecordOp op, PythonEmitter &e) {
    auto expression = e.tuple(op.getFields(), true);
    if (e.inlineRecords()) e.bind(op.getResult(), expression);
    else e.assign(op.getResult(), expression);
    return success();
  });
  emitters.add<ExtractOp>(valid, [](ExtractOp op, PythonEmitter &e) {
    e.assign(op.getResult(), e.valueString(op.getRecord()) + "[" + std::to_string(op.getField()) + "]");
    return success();
  });
  emitters.add<scf::ForOp>(valid, [](scf::ForOp op, PythonEmitter &e) {
    e.emitFor(op); return success();
  });
  emitters.add<scf::IfOp>(valid, [](scf::IfOp op, PythonEmitter &e) {
    e.emitIf(op); return success();
  });
  emitters.add<scf::WhileOp>(valid, [](scf::WhileOp op, PythonEmitter &e) {
    e.emitWhile(op); return success();
  });
  // Function/terminator roles are emitted by their owning region handler.
  auto regionOwned = [](auto, PythonEmitter &) { return success(); };
  emitters.add<func::FuncOp>(valid, regionOwned);
  emitters.add<func::ReturnOp>(valid, regionOwned);
  emitters.add<scf::YieldOp>(valid, regionOwned);
  emitters.add<scf::ConditionOp>(valid, regionOwned);
  emitters.add<YieldOp>(valid, regionOwned);
  emitters.add<ViewOverlapOp>(valid, regionOwned);
}

void PythonEmitter::emitOperation(Operation &operation) {
  if (mlir::failed(operationEmitters().emit(&operation, *this))) failed = true;
}

void PythonEmitter::assignYield(ValueRange yielded, ArrayRef<std::string> results) {
  if (results.empty()) return;
  SmallVector<std::string> expressions;
  for (Value value : yielded) expressions.push_back(controlValueString(value));
  if (results.size() == 1) line(results.front() + " = " + expressions.front());
  else line(stringTuple(results) + " = " + stringTuple(expressions));
}

void PythonEmitter::emitBlock(Block &block, ArrayRef<std::string> results) {
  auto begin = output.tell();
  for (Operation &operation : block.without_terminator()) emitOperation(operation);
  if (auto yield = dyn_cast<scf::YieldOp>(block.getTerminator()))
    assignYield(yield.getOperands(), results);
  if (output.tell() == begin) line("pass");
}

void PythonEmitter::emitFor(scf::ForOp loop) {
  SmallVector<std::string> results;
  for (auto [result, initial] : llvm::zip(loop.getResults(), loop.getInitArgs())) {
    std::string name = newName();
    bind(result, name);
    line(name + " = " + loopInitialValue(initial));
    results.push_back(name);
  }
  std::string induction = newName("iv");
  bind(loop.getInductionVar(), induction);
  for (auto [argument, name] : llvm::zip(loop.getRegionIterArgs(), results)) bind(argument, name);
  line("for " + induction + " in " + forRange(loop) + ":");
  ++indent; emitBlock(*loop.getBody(), results); --indent;
}

void PythonEmitter::emitIf(scf::IfOp branch) {
  SmallVector<std::string> results;
  for (Value result : branch.getResults()) {
    auto name = newName(); bind(result, name); results.push_back(name);
  }
  line("if " + valueString(branch.getCondition()) + ":");
  ++indent; emitBlock(branch.getThenRegion().front(), results); --indent;
  if (!branch.getElseRegion().empty()) {
    line("else:");
    ++indent; emitBlock(branch.getElseRegion().front(), results); --indent;
  }
}

void PythonEmitter::emitWhile(scf::WhileOp loop) {
  SmallVector<std::string> carries;
  for (Value initial : loop.getInits()) {
    auto name = newName();
    line(name + " = " + loopInitialValue(initial)); carries.push_back(name);
  }
  SmallVector<std::string> results;
  for (Value result : loop.getResults()) {
    auto name = newName(); bind(result, name); results.push_back(name);
  }
  Block &before = loop.getBefore().front();
  for (auto [argument, name] : llvm::zip(before.getArguments(), carries)) bind(argument, name);
  std::string active = newName();
  auto condition = cast<scf::ConditionOp>(before.getTerminator());
  auto emitBefore = [&] {
    for (Operation &operation : before.without_terminator()) emitOperation(operation);
    // The condition's forwarded values are both exit results and after-region
    // arguments. Initialize them before the Python loop so provider frontends
    // recognize its complete carried schema, including a false first condition.
    assignYield(condition.getArgs(), results);
    line(active + " = " + controlValueString(condition.getCondition()));
  };
  emitBefore();
  line("while " + active + ":");
  ++indent;
  Block &after = loop.getAfter().front();
  for (auto [argument, result] : llvm::zip(after.getArguments(), results))
    bind(argument, result);
  emitBlock(after, carries);
  emitBefore();
  --indent;
}

void PythonEmitter::emitHelper(Operation *owner, Region &region, StringRef name,
                               StringRef decorator, StringRef argumentPrefix) {
  helperNames[owner] = name.str();
  ScopedValues scope(*this);
  auto savedConstexpr = std::move(constexprValues);
  unsigned savedIndent = indent;
  bool savedHelper = emittingHelper;
  constexprValues.clear(); emittingHelper = true;
  output << decorator << "\ndef " << name << "(";
  Block &block = region.front();
  for (auto [index, argument] : llvm::enumerate(block.getArguments())) {
    if (index) output << ", ";
    std::string argumentName = (argumentPrefix + Twine(index)).str();
    output << argumentName; bind(argument, argumentName);
  }
  output << "):\n"; indent = 1;
  for (Operation &operation : block.without_terminator()) emitOperation(operation);
  auto yield = cast<YieldOp>(block.getTerminator());
  line("return " + (yield.getValues().size() == 1 ? valueString(yield.getValues().front()) : tuple(yield.getValues())));
  output << "\n";
  constexprValues = std::move(savedConstexpr); indent = savedIndent; emittingHelper = savedHelper;
}

std::string PythonEmitter::expressionString(PhysicalExprAttr expression) {
  return pythonExpression(expression, expressionSyntax, [&](PhysicalExprAttr leaf) {
    if (leaf.getKind() == PhysicalExprKind::Parameter)
      return leaf.getParameterReference().getName().getValue().str();
    return valueString(resolveArgument(kernel, leaf.getArgumentReference()));
  });
}

std::string PythonEmitter::fragmentShape(FragmentType type) {
  SmallVector<std::string> extents;
  for (Attribute extent : type.getShape()) extents.push_back(expressionString(cast<PhysicalExprAttr>(extent)));
  return stringTuple(extents);
}

std::string PythonEmitter::pythonType(Type type) const { return pythonScalarType(type, scalarSyntax); }
std::string PythonEmitter::literal(Attribute value) {
  if (auto expression = dyn_cast<PhysicalExprAttr>(value)) return expressionString(expression);
  return pythonLiteral(value);
}
Type PythonEmitter::elementType(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type)) return fragment.getElementType();
  return type;
}
std::string PythonEmitter::stringTuple(ArrayRef<std::string> values) {
  std::string result = "(";
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index) result += ", "; result += value;
  }
  if (values.size() == 1) result += ",";
  return result + ")";
}
std::string PythonEmitter::stringList(ArrayRef<std::string> values) {
  std::string result = "[";
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index) result += ", "; result += value;
  }
  return result + "]";
}
std::string PythonEmitter::axisTuple(ArrayRef<int64_t> axes) {
  SmallVector<std::string> values;
  for (int64_t axis : axes) values.push_back(std::to_string(axis));
  return stringTuple(values);
}
std::string PythonEmitter::tuple(ValueRange values, bool control) {
  SmallVector<std::string> expressions;
  for (Value value : values) expressions.push_back(control ? controlValueString(value) : valueString(value));
  return stringTuple(expressions);
}
std::string PythonEmitter::helperName(Operation *owner) const { return helperNames.lookup(owner); }
void PythonEmitter::assign(Value value, StringRef expression, bool compileTime) {
  auto name = newName(); bind(value, name);
  if (compileTime) constexprValues.insert(value);
  line(name + (compileTime ? constexprAnnotation : "") + " = " + expression.str());
}
void PythonEmitter::assignResults(ResultRange results, StringRef expression) {
  SmallVector<std::string> names;
  for (Value result : results) { auto name = newName(); bind(result, name); names.push_back(name); }
  std::string lhs;
  for (auto [index, name] : llvm::enumerate(names)) { if (index) lhs += ", "; lhs += name; }
  line(lhs + " = " + expression.str());
}
std::string PythonEmitter::controlValueString(Value value) { return valueString(value); }
std::string PythonEmitter::loopInitialValue(Value value) { return controlValueString(value); }

} // namespace intent::gpu
