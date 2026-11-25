import ast
import json

from .intentdsl.signature import (
    argument_type,
    format_fastmath_attr,
    kernel_math_policy,
    namespace,
    signature,
    spelling,
)


class Capture:
    def __init__(self, fn):
        self.fn = fn
        self.namespace = namespace(fn)
        self.default_flags, self.denormal_mode = kernel_math_policy(fn)
        self.args = signature(fn)
        self.values = {
            a["name"]: ("%" + a["name"], argument_type(a)) for a in self.args
        }
        self.ops = []
        self.updated=set()
        self.serial = 0
        self.line = fn.lineno

    def fm_attr(self, extra=()):
        fm = format_fastmath_attr((*self.default_flags, *extra))
        return f"{{fastmath = {fm}}}" if fm else ""

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
        if self.owner_axis is not None:
            owner = self.axes[self.owner_axis]
            scalar = view["dims"] == [owner] and len(indices)==1 and ast.unparse(indices[0])==self.row
            slab = view["dims"] == self.axes and len(indices)==len(self.axes) and all(
                ast.unparse(index)==(self.row if i==self.owner_axis else ":")
                for i,index in enumerate(indices))
            if not (scalar or slab):
                raise ValueError("access must project the owning axis or its complete member domain")
            if view["direction"] not in (("Out","InOut") if store else ("In","InOut")):
                raise ValueError("access violates view direction")
            element=view["dtype"]
            return [self.values[view["name"]],("%row","index")], element if scalar else f"tensor<?x{element}>"
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
        if isinstance(n, ast.Attribute) and n.attr in ("count", "mean", "m2"):
            state = self.expr(n.value)
            if state[1] != "tuple<i32, f32, f32>":
                raise ValueError("expected a Welford summary")
            return self.field(state, ("count", "mean", "m2").index(n.attr))
        if isinstance(n, ast.Subscript):
            args, ty = self.access(n)
            return self.emit("read" if ty.startswith("tensor") else "read_owner", args, ty)
        if (
            isinstance(n, ast.Compare)
            and len(n.ops) == 1
            and isinstance(n.ops[0], (ast.Gt,ast.NotEq))
        ):
            a, b = self.expr(n.left), self.expr(n.comparators[0])
            ty = (
                "tensor<?xi1>"
                if any(t.startswith("tensor") for _, t in (a, b))
                else "i1"
            )
            return self.emit("gt" if isinstance(n.ops[0],ast.Gt) else "ne", [a, b], ty)
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
            elem = ty.removeprefix("tensor<?x").removesuffix(">")
            is_fp = elem in ("f16", "bf16", "f32")
            return self.emit(op, args, ty, self.fm_attr() if is_fp else "")
        if isinstance(n, ast.Call):
            fn = spelling(n.func, self.namespace)
            if fn in ("atomic_add", "gather"):
                from .effects import indexed_expression
                return indexed_expression(self,n,fn)
            if fn=="column_index" and [ast.unparse(a) for a in n.args]==[self.owner,self.row]:
                return self.emit("member_index",[self.values[self.owner],("%row","index")],"tensor<?xi32>")
            if fn=="reduce_by_key":
                from .keyed import capture
                return capture(self,n)
            if fn=="sum_columns" and len(n.args)==1 and not n.keywords:
                return self.emit("column_sum",[self.values[n.args[0].id],("%row","index")],"tensor<?xi32>")
            if fn in ("ordered_sum", "segment_sum"):
                from .segments import fold
                return fold(self, n, reassoc=(fn == "segment_sum"))
            if fn=="segment_length":
                from .segments import bounds
                return self.emit("segment_length",bounds(self,n),"tensor<?xf32>")
            if fn=="gather_row":
                if len(n.args)!=3 or len(n.keywords)!=1 or n.keywords[0].arg!="in_bounds" or ast.literal_eval(n.keywords[0].value) is not True:
                    raise ValueError("row gather requires a table, row indices and an explicit bounds promise")
                table,ids,row=n.args
                slots={a["name"]:a for a in self.args}
                t,ix=slots[table.id],slots[ids.id]
                if (row.id!=self.row or len(self.axes)!=2 or t["direction"]!="In" or ix["direction"]!="In" or
                    len(t["dims"])!=2 or t["dims"][1]!=self.axes[1] or ix["dims"]!=self.axes[:1] or ix["dtype"] not in ("i32","i64")):
                    raise ValueError("gather index and feature dimensions do not match the member domain")
                return self.emit("gather_row",[self.values[table.id],self.values[ids.id],("%row","index")],
                                 f'tensor<?x{t["dtype"]}>','{in_bounds}')
            if fn == "cumsum":
                if len(n.args)!=1 or len(n.keywords)!=1 or n.keywords[0].arg!="axis" or ast.literal_eval(n.keywords[0].value)!=0:
                    raise ValueError("cumsum requires one member vector and axis=0")
                v=self.expr(n.args[0])
                if v[1] not in ("tensor<?xf32>","tensor<?xi32>"):
                    raise ValueError("inclusive sum requires f32 or i32 members")
                return self.emit("scan_sum",[v],v[1])
            if fn == "welford" and len(n.args) == 1 and not n.keywords:
                samples = self.expr(n.args[0])
                if samples[1] != "tensor<?xf32>":
                    raise ValueError("welford requires one f32 member vector")
                return self.emit("welford", [samples], "tuple<i32, f32, f32>")
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
            if fn == "fdiv" and len(n.args) == 2:
                kw = {k.arg: k.value for k in n.keywords}
                if len(kw) != len(n.keywords) or kw.keys() - {"approximate", "flush_to_zero"}:
                    raise ValueError("fdiv accepts only approximate and flush_to_zero")
                approx = ast.literal_eval(kw["approximate"]) if "approximate" in kw else False
                ftz = ast.literal_eval(kw["flush_to_zero"]) if "flush_to_zero" in kw else False
                if type(approx) is not bool or type(ftz) is not bool:
                    raise ValueError("fdiv flags must be boolean literals")
                if ftz and self.denormal_mode == "ieee":
                    self.denormal_mode = "preserve-sign"
                args = [self.expr(v) for v in n.args]
                ty = next((t for _, t in args if t.startswith("tensor")), args[0][1])
                extra = ("afn", "arcp") if approx else ()
                return self.emit("div", args, ty, self.fm_attr(extra))
            count = 2 if fn == "cast" else 1
            if len(n.args) != count or (fn not in ("sum", "reduce", "exp") and n.keywords):
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
                return self.emit(
                    "sum" if fn == "sum" else "max_num",
                    [v],
                    "f32",
                    self.fm_attr(("reassoc",)),
                )
            if fn == "cast":
                dst = spelling(n.args[1], self.namespace)
                return self.emit(
                    "cast",
                    [v],
                    f"tensor<?x{dst}>" if v[1].startswith("tensor") else dst,
                )
            if fn in ("exp", "rsqrt", "sqrt"):
                kw = {k.arg: k.value for k in n.keywords}
                if len(kw) != len(n.keywords) or kw.keys() - {"approximate", "flush_to_zero"}:
                    raise ValueError(f"unsupported keywords for {fn}")
                approx = ast.literal_eval(kw["approximate"]) if "approximate" in kw else False
                ftz = ast.literal_eval(kw["flush_to_zero"]) if "flush_to_zero" in kw else False
                if type(approx) is not bool or type(ftz) is not bool:
                    raise ValueError("math flags must be boolean literals")
                if ftz and self.denormal_mode == "ieee":
                    self.denormal_mode = "preserve-sign"
                extra = ("afn",) if approx else ()
                return self.emit(fn, [v], v[1], self.fm_attr(extra))
        raise ValueError(
            f"unsupported author expression at line {n.lineno}: {ast.unparse(n)}"
        )

    def field(self, state, position):
        return self.emit("field", [state], "i32" if position == 0 else "f32",
                         f"{{position={position} : i64}}")

    def capture(self):
        (loop,) = self.fn.body
        if (
            not isinstance(loop, ast.For)
            or loop.orelse
            or spelling(loop.iter.func, self.namespace) != "parallel"
        ):
            raise ValueError("expected an independent parallel row domain")
        first=loop.body[0] if loop.body else None
        named_domain=(isinstance(first,ast.Assign) and isinstance(first.value,ast.Call) and
                      ast.unparse(first.value.func)==self.namespace+".domain")
        if isinstance(loop.target, ast.Tuple) or named_domain:
            from .indexed import IndexedCapture
            return IndexedCapture(self, loop).module()
        (dim,) = loop.iter.args
        if (
            not isinstance(dim, ast.Call)
            or dim.func.attr != "dim"
            or not isinstance(dim.args[0],ast.Constant)
        ):
            raise ValueError("expected a logical leading dimension")
        owner = next(a for a in self.args if a["name"] == dim.func.value.id)
        self.owner = owner["name"]
        self.axes = owner["dims"]
        axis = ast.literal_eval(dim.args[0])
        if type(axis) is not int or not 0 <= axis < len(self.axes) <= 4:
            raise ValueError("owner axis outside the declared view")
        self.owner_axis = axis if len(self.axes)>2 or axis != 0 else None
        self.row = loop.target.id
        outputs = set()
        for stmt in loop.body:
            self.line = stmt.lineno
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                from .effects import statement
                statement(self,stmt.value,outputs)
                continue
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                raise ValueError(f"unsupported row statement at line {stmt.lineno}")
            target = stmt.targets[0]
            v = self.expr(stmt.value)
            if isinstance(target, ast.Name):
                self.values[target.id] = v
            elif isinstance(target, ast.Tuple):
                if (v[1] != "tuple<i32, f32, f32>" or len(target.elts) != 3
                        or not all(isinstance(part, ast.Name) for part in target.elts)):
                    raise ValueError("tuple assignment requires one name per summary field")
                for position, part in enumerate(target.elts):
                    self.values[part.id] = self.field(v, position)
            else:
                transposed = self.owner_axis is None and isinstance(target.slice, ast.Tuple) and [
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
                    "write" if ty.startswith("tensor") else "write_owner",
                    [*args, v],
                    attrs="{axes=array<i64:1,0>}" if transposed else "",
                )
        if outputs|self.updated != {
            a["name"] for a in self.args if a["direction"] in ("Out", "InOut")
        }:
            raise ValueError("missing output definition")
        sig = ", ".join("%" + a["name"] + ": " + argument_type(a) for a in self.args)
        abi = json.dumps(json.dumps(self.args, separators=(",", ":")))
        return (
            f'module {{ func.func @{self.fn.name}({sig}) attributes {{kir.abi = {abi}, "denormal-fp-math" = "{self.denormal_mode}"'
            + (f", kir.owner_axis = {axis} : i64" if self.owner_axis is not None else "")
            + "} {\n"
            + f'"kir.parallel"(%{owner["name"]}) ({{^bb0(%row: index):\n'
            + "\n".join(self.ops)
            + '\n"kir.end"() : () -> ()\n}) : ('
            + argument_type(owner)
            + ") -> ()\n return\n} }"
        ).encode()
