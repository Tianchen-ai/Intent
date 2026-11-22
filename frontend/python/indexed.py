"""Capture rectangular iteration and affine subscripts as typed KIR argument maps."""

import ast
import json
from .intentdsl.signature import argument_type, spelling


class IndexedCapture:
    def __init__(self, capture, loop):
        self.capture = capture
        self.public = list(capture.args)
        self.positions = {a["name"]: i for i, a in enumerate(self.public)}
        targets=loop.target.elts if isinstance(loop.target,ast.Tuple) else [loop.target]
        self.axes = [item.id for item in targets]
        owners=len(self.axes)
        extents = list(loop.iter.args[0].elts) if isinstance(loop.iter.args[0],ast.Tuple) else list(loop.iter.args)
        statements = list(loop.body)
        # A named member domain is a tensor axis inside each parallel owner.
        while statements and isinstance(statements[0], ast.Assign):
            decl = statements[0]
            if (len(decl.targets)==1 and isinstance(decl.targets[0], ast.Name)
                    and isinstance(decl.value, ast.Call)
                    and ast.unparse(decl.value.func)==capture.namespace+".domain"):
                begin,stop=decl.value.args
                if ast.literal_eval(begin)!=0:
                    raise ValueError("an affine member axis must start at zero")
                self.axes.append(decl.targets[0].id)
                extents.append(stop)
                statements=statements[1:]
            else:
                break
        self.members=len(self.axes)-owners
        self.owner_axis=0 if self.members>1 and owners==1 else None
        if self.members>1 and self.owner_axis is None:
            raise ValueError("multiple member axes require a single owner axis")
        self.owner_bindings=set()
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
        # Scalar captures keep one public slot, independent of the access maps.
        # Append them after projected views so the root domain has a view owner.
        scalars=[(i,a) for i,a in enumerate(self.public) if a["direction"]=="Scalar"]
        for _,a in scalars:
            capture.values[a["name"]]=("%"+a["name"],argument_type(a))
        self.body(statements)
        for i,a in scalars:
            self.slots[(i,None)]=len(capture.args)
            capture.args.append(a)

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

    def access(self, node, store=False, scalar_result=False):
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
        owned = scalar_result and store and self.members>0
        if owned and any(any(row[-1-self.members:-1]) for row in matrix):
            raise ValueError("a reduced scalar can write only an owner coordinate")
        key = (src, matrix)
        if key not in self.slots:
            i = len(self.slots)
            self.slots[key] = i
            desc = dict(
                name="view" + str(i),
                direction=actual["direction"],
                dtype=actual["dtype"],
                dims=["axis0"] if owned else ["axis" + str(j) for j in range(len(self.axes))],
            )
            self.capture.args.append(desc)
        if store:
            self.written.add(src)
        if owned:self.owner_bindings.add(key)
        d = self.capture.args[self.slots[key]]
        return [
            ("%" + d["name"], argument_type(d)),
            ("%row", "index"),
        ], actual["dtype"] if owned else f'tensor<?x{actual["dtype"]}>'

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
                args, t = self.access(dest, True, not value[1].startswith("tensor"))
                if t != value[1]:
                    raise ValueError("write dtype differs from destination")
                c.emit("write" if t.startswith("tensor") else "write_owner", [*args, value])

    def module(self):
        c = self.capture
        if self.written != {
            i for i, a in enumerate(self.public) if a["direction"] in ("Out", "InOut")
        }:
            raise ValueError("indexed kernel leaves an output undefined")
        formals = []
        for (src, matrix), d in zip(self.slots, c.args, strict=True):
            if matrix is None:
                formals.append(f'%{d["name"]}: {argument_type(d)} {{kir.source = {src} : i64}}')
                continue
            axes = ",".join("d" + str(j) for j in range(len(self.axes)))
            exprs = []
            for row in matrix:
                terms = [f"d{j} * {k}" for j, k in enumerate(row[:-1]) if k]
                exprs.append(" + ".join([*terms, str(row[-1])]))
            layout = "affine_map<(" + axes + ")->(" + ",".join(exprs) + ")>"
            owner=", kir.owner_projection" if (src,matrix) in self.owner_bindings else ""
            formals.append(f'%{d["name"]}: {argument_type(d)} {{kir.source = {src} : i64, kir.indexing = {layout}{owner}}}')
        string = lambda x: json.dumps(json.dumps(x, separators=(",", ":")))
        bounds = ", ".join(str(x) for row in self.bounds for x in row)
        first = c.args[0]
        owner=", kir.owner_axis=0:i64" if self.owner_axis==0 else ""
        return (
            f"module {{ func.func @{c.fn.name}("
            + ", ".join(formals)
            + f") attributes {{kir.abi={string(c.args)}, kir.public_abi={string(self.public)}, kir.iteration=array<i64:{bounds}>{owner}}} {{\n"
            + f'"kir.parallel"(%{first["name"]}) ({{^bb0(%row:index):\n'
            + "\n".join(c.ops)
            + '\n"kir.end"():()->()\n}):('
            + argument_type(first)
            + ")->()\nreturn\n} }"
        ).encode()
