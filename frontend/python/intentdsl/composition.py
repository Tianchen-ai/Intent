"""Static host call capture. Only scalar arguments and tensor dimensions can drive sizes."""

import ast
import inspect
import operator
from dataclasses import dataclass
from textwrap import dedent

from . import Definition
from .signature import signature, spelling


@dataclass(frozen=True)
class Program:
    name: str
    signature: inspect.Signature
    steps: tuple
    returns: tuple
    multiple: bool


_arithmetic = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.FloorDiv: operator.floordiv}


def _scalar(expr, known, owner):
    if isinstance(expr, ast.Constant) and type(expr.value) in (int, float):
        return lambda values: expr.value
    if isinstance(expr, ast.Name) and expr.id in known:
        def scalar(values):
            value = values[expr.id]
            if type(value) not in (int, float):
                raise TypeError(f"{expr.id} is not a host scalar")
            return value
        return scalar
    if isinstance(expr, ast.Call) and not expr.keywords:
        fn = expr.func
        if (isinstance(fn, ast.Attribute) and fn.attr == "dim"
                and isinstance(fn.value, ast.Name) and fn.value.id in known
                and len(expr.args) == 1):
            axis = ast.literal_eval(expr.args[0])
            if type(axis) is not int:
                raise ValueError("dim requires an integer axis")
            return lambda values: values[fn.value.id].shape[axis]
        if spelling(fn, owner) == "ceildiv" and len(expr.args) == 2:
            left, right = (_scalar(x, known, owner) for x in expr.args)
            return lambda values: -(-left(values) // right(values))
    if isinstance(expr, ast.BinOp) and type(expr.op) in _arithmetic:
        left, right = (_scalar(x, known, owner) for x in (expr.left, expr.right))
        return lambda values: _arithmetic[type(expr.op)](left(values), right(values))
    raise ValueError(f"unsupported host scalar expression: {ast.unparse(expr)}")


def capture_program(fn):
    tree, = ast.parse(dedent(inspect.getsource(fn))).body
    decorator, = tree.decorator_list
    owner = decorator.value.id
    sig = inspect.signature(fn)
    captures = inspect.getclosurevars(fn)
    symbols = {**captures.globals, **captures.nonlocals}
    known = set(sig.parameters)
    directions = {
        arg.arg: spelling(arg.annotation.value, owner) for arg in tree.args.args
        if isinstance(arg.annotation, ast.Subscript)
    }
    allocated = {}
    ready = {name for name in known if directions.get(name) != "Out"}
    steps = []

    def view(expr):
        if not isinstance(expr, ast.Name) or expr.id not in known:
            raise ValueError("expected a declared program view")
        return expr.id

    for stmt in tree.body[:-1]:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            dst = stmt.targets[0]
            call = stmt.value
            if (not isinstance(dst, ast.Name) or dst.id in known
                    or not isinstance(call, ast.Call) or call.keywords or len(call.args) != 2):
                raise ValueError("expected a fresh workspace or output declaration")
            kind = spelling(call.func, owner)
            dtype = spelling(call.args[0], owner)
            shape = call.args[1]
            if (kind not in ("workspace", "output") or dtype not in ("f16","bf16","f32","i32")
                    or not isinstance(shape, ast.Tuple) or not shape.elts):
                raise ValueError("invalid program allocation")
            extents = tuple(_scalar(x, known, owner) for x in shape.elts)
            known.add(dst.id)
            allocated[dst.id] = kind
            steps.append(("allocate", dst.id, dtype, extents))
            continue
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            raise ValueError("program control and kernel call order must be static")
        call = stmt.value
        if call.keywords:
            raise ValueError("use declaration order for program calls")
        if isinstance(call.func, ast.Attribute) and spelling(call.func, owner) == "fill":
            if len(call.args) != 2:
                raise ValueError("fill requires a view and a scalar")
            dst = view(call.args[0])
            if directions.get(dst) == "In":
                raise ValueError(f"fill cannot write read-only input {dst}")
            ready.add(dst)
            steps.append(("fill", dst, _scalar(call.args[1], known, owner)))
            continue
        callee = symbols.get(call.func.id) if isinstance(call.func, ast.Name) else None
        if not isinstance(callee, Definition):
            raise ValueError("program call requires a statically named author kernel")
        formals = signature(ast.parse(callee.source).body[0])
        if len(formals) != len(call.args):
            raise ValueError(f"{callee.name}: kernel call arity mismatch")
        operands, writes = [], []
        for spec, arg in zip(formals, call.args):
            direction = spec["direction"]
            if direction == "Scalar":
                operands.append(_scalar(arg, known, owner))
                continue
            slot = view(arg)
            if direction != "Out" and slot not in ready:
                raise ValueError(f"{callee.name}: read before initialization of {slot}")
            if direction != "In":
                if directions.get(slot) == "In":
                    raise ValueError(f"{callee.name}: cannot write read-only input {slot}")
                writes.append(slot)
            operands.append(slot)
        steps.append(("call", callee, tuple(operands)))
        ready.update(writes)

    last = tree.body[-1]
    if not isinstance(last, ast.Return):
        raise ValueError("a program must end by returning its output views")
    multiple = isinstance(last.value, ast.Tuple)
    returns = tuple(view(e) for e in (last.value.elts if multiple else (last.value,)))
    if not returns or any(r not in ready or allocated.get(r) == "workspace" for r in returns):
        raise ValueError("return requires initialized outputs or caller views")
    return Program(fn.__name__, sig, tuple(steps), returns, multiple)


class ProgramKernel:
    def __init__(self, compiler, definition, backend):
        from .compiler import Kernel

        self.definition = definition
        self.views = compiler.runtime("kir", "binding.views")
        self.sequence = compiler.runtime("kir", "binding.sequence")
        self.kernels = {s[1]: Kernel(compiler, s[1], backend)
                        for s in definition.steps if s[0] == "call"}

    def prepare(self, *args, **kwargs):
        bound = self.definition.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        inputs = [v for v in values.values() if type(v) not in (int, float)]
        if not inputs:
            raise ValueError("program requires a view to identify the execution device")
        device = self.views.inspect(inputs[0]).device
        if any(self.views.inspect(v).device != device for v in inputs[1:]):
            raise ValueError("program inputs span devices")
        allocator = self.views.adapter(inputs[0])
        calls = []
        labels = []
        for step in self.definition.steps:
            if step[0] == "allocate":
                _, name, dtype, extents = step
                shape = tuple(get(values) for get in extents)
                if any(type(n) is not int or n < 0 for n in shape):
                    raise ValueError(f"{name}: invalid workspace or output extent {shape}")
                values[name] = allocator.allocate(shape, dtype, device)
            elif step[0] == "fill":
                calls.append(self.sequence.Fill(values[step[1]], step[2](values)))
                labels.append(step[1])
            else:
                kernel = self.kernels[step[1]]
                inputs, outputs = [], []
                for operand, spec in zip(step[2], kernel.args):
                    value = values[operand] if isinstance(operand, str) else operand(values)
                    (outputs if spec["direction"] == "Out" else inputs).append(value)
                calls.append(kernel.prepare(*inputs, outputs=outputs))
                labels.append(step[1].name)
        result = tuple(values[n] for n in self.definition.returns)
        return self.sequence.PreparedSequence(calls, labels, result if self.definition.multiple else result[0])

    def run(self, *args, **kwargs):
        prepared = self.prepare(*args, **kwargs)
        prepared.launch()
        return prepared.result()
