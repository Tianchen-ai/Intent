import ast
import json

from .intentdsl.signature import argument_type, signature, spelling, namespace


class Capture:
    def __init__(self, fn):
        self.fn = fn
        self.namespace = namespace(fn)
        self.args = signature(fn)
        self.values = {
            a["name"]: ("%" + a["name"], argument_type(a)) for a in self.args
        }
        self.ops = []
        self.serial = 0
        self.line = fn.lineno

    def emit(self, op, args, ty=None, attrs=""):
        self.serial += 1
        name = f"%v{self.serial}"
        vals = ", ".join(v for v, _ in args)
        types = ", ".join(t for _, t in args)
        self.ops.append(
            f'{(name+" = ") if ty else ""}"kir.{op}"({vals}) {attrs} : ({types}) -> {ty or "()"} loc("{self.fn.name}":{self.line}:1)'
        )
        return name, ty

    def access(self, node, store=False):
        view = next(a for a in self.args if a["name"] == node.value.id)
        indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if len(self.axes) == 1:
            if (
                view["dims"] != self.axes
                or len(indices) != 1
                or ast.unparse(indices[0]) != self.row
            ):
                raise ValueError("flat access must follow its parallel member")
            direction = view["direction"]
            if direction != "InOut" and direction != ("Out" if store else "In"):
                raise ValueError("access violates argument direction")
            return [
                self.values[view["name"]],
                ("%row", "index"),
            ], f'tensor<?x{view["dtype"]}>'
        if (
            len(indices) != len(view["dims"])
            or len(indices) not in (1, 2)
            or ast.unparse(indices[-1]) != ":"
        ):
            raise ValueError("expected a full final axis")
        if (
            view["dims"][-1] != self.axes[-1]
            or len(indices) == 2
            and (ast.unparse(indices[0]) != self.row or view["dims"][0] != self.axes[0])
        ):
            raise ValueError("unrelated row or column domain")
        if view["direction"] != "InOut" and store != (view["direction"] == "Out"):
            raise ValueError("access violates argument direction")
        args = [self.values[view["name"]]]
        if len(indices) == 2:
            args.append(("%row", "index"))
        return args, f'tensor<?x{view["dtype"]}>'

    def expr(self, n):
        if isinstance(n, ast.Name):
            return self.values[n.id]
        if isinstance(n, ast.Subscript):
            args, ty = self.access(n)
            return self.emit("read", args, ty)
        if (
            isinstance(n, ast.Compare)
            and len(n.ops) == 1
            and isinstance(n.ops[0], ast.Gt)
        ):
            a, b = self.expr(n.left), self.expr(n.comparators[0])
            ty = (
                "tensor<?xi1>"
                if any(t.startswith("tensor") for _, t in (a, b))
                else "i1"
            )
            return self.emit("gt", [a, b], ty)
        if isinstance(n, ast.BinOp):
            args = [self.expr(n.left), self.expr(n.right)]
            op = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.Div: "div",
                ast.BitXor: "xor",
                ast.BitAnd: "and",
                ast.LShift: "shl",
                ast.RShift: "shr",
            }[type(n.op)]
            ty = next((t for _, t in args if t.startswith("tensor")), args[0][1])
            return self.emit(op, args, ty)
        if isinstance(n, ast.Call):
            fn = spelling(n.func, self.namespace)
            if (
                fn in ("linear_index", "member_counter")
                and len(n.args) == 2
                and isinstance(n.args[0], ast.Name)
                and n.args[0].id == self.owner
                and isinstance(n.args[1], ast.Name)
                and n.args[1].id == self.row
            ):
                return self.emit(
                    "ordinal",
                    [self.values[self.owner], ("%row", "index")],
                    "tensor<?xi32>" if fn == "linear_index" else "tensor<?xui64>",
                )
            if fn == "uniform" and len(n.args) == 2:
                return self.emit(
                    "uniform", [self.expr(a) for a in n.args], "tensor<?xf32>"
                )
            if fn == "where" and len(n.args) == 3:
                mask, a, b = (self.expr(v) for v in n.args)
                return self.emit(
                    "select",
                    [mask, a, b],
                    next((t for _, t in (a, b) if t.startswith("tensor")), a[1]),
                )
            if fn == "maximum" and len(n.args) == 2 and not n.keywords:
                args = [self.expr(v) for v in n.args]
                return self.emit(
                    "maximum",
                    args,
                    next((ty for _, ty in args if ty.startswith("tensor")), args[0][1]),
                )
            if (
                fn == "cast"
                and len(n.args) == 2
                and isinstance(n.args[0], ast.Constant)
            ):
                ty = spelling(n.args[1], self.namespace)
                lit = ast.literal_eval(n.args[0])
                if ty == "i32" and type(lit) is int and -(1 << 31) <= lit < (1 << 31):
                    return self.emit("constant", [], ty, f"{{value={lit} : i32}}")
                if ty not in ("f16", "bf16", "f32") or type(lit) not in (float, int):
                    raise ValueError("expected typed floating literal")
                return self.emit(
                    "constant", [], ty, f"{{value={float(lit):.17e} : {ty}}}"
                )
            count = 2 if fn == "cast" else 1
            if len(n.args) != count or (fn not in ("sum", "reduce") and n.keywords):
                raise ValueError(f"unsupported arguments for {fn}")
            v = self.expr(n.args[0])
            if fn in ("sum", "reduce"):
                kw = {k.arg: k.value for k in n.keywords}
                if ast.literal_eval(kw["axis"]) != 0 or v[1] != "tensor<?xf32>":
                    raise ValueError("expected f32 row reduction on axis 0")
                if fn == "reduce" and (
                    set(kw) != {"axis", "combine", "init"}
                    or spelling(kw["combine"], self.namespace) != "max_num"
                    or not isinstance(kw["init"], ast.Call)
                    or spelling(kw["init"].func, self.namespace) != "neg_inf"
                    or len(kw["init"].args) != 1
                    or spelling(kw["init"].args[0], self.namespace) != "f32"
                ):
                    raise ValueError("unsupported reduction algebra")
                if fn == "sum" and (
                    set(kw) != {"axis", "acc"}
                    or spelling(kw["acc"], self.namespace) != "f32"
                ):
                    raise ValueError("expected an explicit f32 accumulator")
                return self.emit("sum" if fn == "sum" else "max_num", [v], "f32")
            if fn == "cast":
                dst = spelling(n.args[1], self.namespace)
                return self.emit(
                    "cast",
                    [v],
                    f"tensor<?x{dst}>" if v[1].startswith("tensor") else dst,
                )
            if fn in ("exp", "rsqrt", "sqrt"):
                return self.emit(fn, [v], v[1])
        raise ValueError(
            f"unsupported author expression at line {n.lineno}: {ast.unparse(n)}"
        )

    def capture(self):
        (loop,) = self.fn.body
        if (
            not isinstance(loop, ast.For)
            or loop.orelse
            or spelling(loop.iter.func, self.namespace) != "parallel"
        ):
            raise ValueError("expected an independent parallel row domain")
        if isinstance(loop.target, ast.Tuple):
            from .indexed import IndexedCapture
            return IndexedCapture(self, loop).module()
        (dim,) = loop.iter.args
        if (
            not isinstance(dim, ast.Call)
            or dim.func.attr != "dim"
            or ast.literal_eval(dim.args[0]) != 0
        ):
            raise ValueError("expected a logical leading dimension")
        owner = next(a for a in self.args if a["name"] == dim.func.value.id)
        self.owner = owner["name"]
        self.axes = owner["dims"]
        if len(self.axes) not in (1, 2):
            raise ValueError("expected matrix for parallel row domain")
        self.row = loop.target.id
        outputs = set()
        for stmt in loop.body:
            self.line = stmt.lineno
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                raise ValueError(f"unsupported row statement at line {stmt.lineno}")
            target = stmt.targets[0]
            v = self.expr(stmt.value)
            if isinstance(target, ast.Name):
                self.values[target.id] = v
            else:
                transposed = isinstance(target.slice, ast.Tuple) and [
                    ast.unparse(x) for x in target.slice.elts
                ] == [":", self.row]
                if transposed:
                    view = next(a for a in self.args if a["name"] == target.value.id)
                    if view["direction"] != "Out" or view["dims"] != list(
                        reversed(self.axes)
                    ):
                        raise ValueError("transposed store must cover the output axes")
                    args = [self.values[target.value.id], ("%row", "index")]
                    ty = f'tensor<?x{view["dtype"]}>'
                else:
                    args, ty = self.access(target, True)
                if len(args) != 2 or ty != v[1]:
                    raise ValueError("output type or ownership differs from row")
                outputs.add(target.value.id)
                self.emit(
                    "write",
                    [*args, v],
                    attrs="{axes=array<i64:1,0>}" if transposed else "",
                )
        if outputs != {
            a["name"] for a in self.args if a["direction"] in ("Out", "InOut")
        }:
            raise ValueError("missing output definition")
        sig = ", ".join("%" + a["name"] + ": " + argument_type(a) for a in self.args)
        abi = json.dumps(json.dumps(self.args, separators=(",", ":")))
        return (
            f"module {{ func.func @{self.fn.name}({sig}) attributes {{kir.abi = {abi}}} {{\n"
            + f'"kir.parallel"(%{owner["name"]}) ({{^bb0(%row: index):\n'
            + "\n".join(self.ops)
            + '\n"kir.end"() : () -> ()\n}) : ('
            + argument_type(owner)
            + ") -> ()\n return\n} }"
        ).encode()
