"""Capture rectangular iteration and affine subscripts as typed KIR argument maps."""

import ast
import json
from .intentdsl.signature import argument_type, spelling


class IndexedCapture:
    def __init__(self, capture, loop):
        self.capture = capture
        self.public = list(capture.args)
        self.positions = {a["name"]: i for i, a in enumerate(self.public)}
        self.axes = [item.id for item in loop.target.elts]
        extents = loop.iter.args[0].elts
        if not 1 <= len(extents) == len(self.axes) <= 4 or len(set(self.axes)) != len(
            self.axes
        ):
            raise ValueError("expected one to four distinct parallel axes")
        self.bounds = [self.extent(n) for n in extents]
        self.static = {}
        self.slots = {}
        self.written = set()
        capture.args = []
        capture.values = {}
        capture.access = self.access
        self.body(loop.body)

    def extent(self, n):
        if isinstance(n, ast.Constant) and type(n.value) is int and n.value > 0:
            return [-1, -1, 1, n.value]
        divisor = 1
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.FloorDiv):
            divisor = ast.literal_eval(n.right)
            n = n.left
        if not isinstance(n, ast.Call) or n.func.attr != "dim" or divisor <= 0:
            raise ValueError("expected a dimension, exact quotient, or constant extent")
        slot = self.positions[n.func.value.id]
        axis = ast.literal_eval(n.args[0])
        if not 0 <= axis < len(self.public[slot]["dims"]):
            raise ValueError("dimension outside view")
        return [slot, axis, divisor, 0]

    def affine(self, n):
        zero = (0,) * (len(self.axes) + 1)
        if isinstance(n, ast.Constant):
            return (*zero[:-1], int(n.value))
        if isinstance(n, ast.Name):
            if n.id in self.static:
                return (*zero[:-1], self.static[n.id])
            i = self.axes.index(n.id)
            return tuple(int(j == i) for j in range(len(zero)))
        if isinstance(n, ast.BinOp):
            a, b = self.affine(n.left), self.affine(n.right)
            if isinstance(n.op, ast.Add):
                return tuple(x + y for x, y in zip(a, b))
            if isinstance(n.op, ast.Sub):
                return tuple(x - y for x, y in zip(a, b))
            if isinstance(n.op, ast.Mult):
                if not any(b[:-1]):
                    return tuple(x * b[-1] for x in a)
                if not any(a[:-1]):
                    return tuple(x * a[-1] for x in b)
            if isinstance(n.op, ast.Mod) and not any(a[:-1] + b[:-1]) and b[-1] > 0:
                return (*zero[:-1], a[-1] % b[-1])
        raise ValueError("subscript is outside the affine projection subset")

    def access(self, node, store=False):
        src = self.positions[node.value.id]
        actual = self.public[src]
        if (
            actual["direction"] != ("Out" if store else "In")
            and actual["direction"] != "InOut"
        ):
            raise ValueError("access violates public buffer direction")
        indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if len(indices) != len(actual["dims"]):
            raise ValueError("subscript rank differs from view")
        matrix = tuple(self.affine(x) for x in indices)
        key = (src, matrix)
        if key not in self.slots:
            i = len(self.slots)
            self.slots[key] = i
            desc = dict(
                name="view" + str(i),
                direction=actual["direction"],
                dtype=actual["dtype"],
                dims=["axis" + str(j) for j in range(len(self.axes))],
            )
            self.capture.args.append(desc)
        if store:
            self.written.add(src)
        d = self.capture.args[self.slots[key]]
        return [
            ("%" + d["name"], argument_type(d)),
            ("%row", "index"),
        ], f'tensor<?x{actual["dtype"]}>'

    def body(self, statements):
        c = self.capture
        for stmt in statements:
            c.line = stmt.lineno
            if (
                isinstance(stmt, ast.For)
                and spelling(stmt.iter.func, c.namespace) == "static_range"
                and not stmt.orelse
            ):
                count = ast.literal_eval(stmt.iter.args[0])
                if type(count) is not int or count < 0:
                    raise ValueError("static range requires a nonnegative integer")
                for i in range(count):
                    self.static[stmt.target.id] = i
                    self.body(stmt.body)
                self.static.pop(stmt.target.id, None)
                continue
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                raise ValueError("expected a pointwise assignment")
            value = c.expr(stmt.value)
            dest = stmt.targets[0]
            if isinstance(dest, ast.Name):
                c.values[dest.id] = value
            else:
                args, t = self.access(dest, True)
                if t != value[1]:
                    raise ValueError("write dtype differs from destination")
                c.emit("write", [*args, value])

    def module(self):
        c = self.capture
        if self.written != {
            i for i, a in enumerate(self.public) if a["direction"] in ("Out", "InOut")
        }:
            raise ValueError("indexed kernel leaves an output undefined")
        formals = []
        for (src, matrix), d in zip(self.slots, c.args, strict=True):
            axes = ",".join("d" + str(j) for j in range(len(self.axes)))
            exprs = []
            for row in matrix:
                terms = [f"d{j} * {k}" for j, k in enumerate(row[:-1]) if k]
                exprs.append(" + ".join([*terms, str(row[-1])]))
            layout = "affine_map<(" + axes + ")->(" + ",".join(exprs) + ")>"
            formals.append(
                f'%{d["name"]}: {argument_type(d)} {{kir.source = {src} : i64, kir.indexing = {layout}}}'
            )
        string = lambda x: json.dumps(json.dumps(x, separators=(",", ":")))
        bounds = ", ".join(str(x) for row in self.bounds for x in row)
        first = c.args[0]
        return (
            f"module {{ func.func @{c.fn.name}("
            + ", ".join(formals)
            + f") attributes {{kir.abi={string(c.args)}, kir.public_abi={string(self.public)}, kir.iteration=array<i64:{bounds}>}} {{\n"
            + f'"kir.parallel"(%{first["name"]}) ({{^bb0(%row:index):\n'
            + "\n".join(c.ops)
            + '\n"kir.end"():()->()\n}):('
            + argument_type(first)
            + ")->()\nreturn\n} }"
        ).encode()
