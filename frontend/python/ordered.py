"""Capture independent owners and strictly ordered scalar loops into MLIR SSA."""

import ast
import json
from dataclasses import dataclass

from .intentdsl.signature import argument_type, namespace, signature


@dataclass(frozen=True)
class Value:
    name: str
    type: str


class Ordered:
    def __init__(self, fn):
        self.fn = fn
        self.ns = namespace(fn)
        self.args = signature(fn)
        self.slots = {a["name"]: a for a in self.args}
        self.env = {
            a["name"]: Value("%" + a["name"], argument_type(a)) for a in self.args
        }
        self.lines = []
        self.seq = 0
        self.lanes = {}

    def fresh(self):
        self.seq += 1
        return f"%v{self.seq}"

    def emit(self, text, ty):
        v = Value(self.fresh(), ty)
        self.lines.append(f"{v.name} = {text}")
        return v

    def literal(self, value, ty):
        return self.emit(f"arith.constant {value} : {ty}", ty)

    def dim(self, node):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "dim"
            and isinstance(node.func.value, ast.Name)
            and len(node.args) == 1
            and not node.keywords
        ):
            raise ValueError("parallel bounds must be dimensions of captured views")
        view = self.slots[node.func.value.id]
        axis = ast.literal_eval(node.args[0])
        if type(axis) is not int or not 0 <= axis < len(view["dims"]):
            raise ValueError("invalid view dimension")
        return view, axis

    def access(self, node):
        if not isinstance(node.value, ast.Name) or node.value.id not in self.slots:
            raise ValueError("expected a captured buffer")
        slot = self.slots[node.value.id]
        coords = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        slices = [i for i, v in enumerate(coords) if isinstance(v, ast.Slice)]
        if len(slices) > 1 or any(ast.unparse(coords[i]) != ":" for i in slices):
            raise ValueError("expected a single whole member axis")
        indices = [self.expr(i) for i in coords if not isinstance(i, ast.Slice)]
        if len(coords) != len(slot["dims"]) or any(i.type != "index" for i in indices):
            raise ValueError("ordered access requires one index per axis")
        return slot, indices, slices

    def member_op(self, op, args, result, axis=None):
        attrs = "" if axis is None else f" {{axis={axis}:i64}}"
        text = (
            f'"kir.{op}"('
            + ", ".join(a.name for a in args)
            + ")"
            + attrs
            + " : ("
            + ", ".join(a.type for a in args)
            + ")->"
        )
        if result is None:
            self.lines.append(text + "()")
            return
        return self.emit(text + result, result)

    def splat(self, scalar, slot, axis):
        t = f"tensor<?x{scalar.type}>"
        v = self.member_op("lane_splat", [scalar, self.env[slot["name"]]], t, axis)
        self.lanes[v] = (slot, axis)
        return v

    def expr(self, node):
        if isinstance(node, ast.Name):
            return self.env[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (float, int):
            return self.literal(
                node.value, "index" if type(node.value) is int else "f32"
            )
        if isinstance(node, ast.Subscript):
            slot, coords, axes = self.access(node)
            view = self.env[slot["name"]]
            if axes:
                v = self.member_op(
                    "slice_read", [view, *coords], f'tensor<?x{slot["dtype"]}>', axes[0]
                )
                self.lanes[v] = (slot, axes[0])
                return v
            at = view.name + "[" + ", ".join(c.name for c in coords) + "]"
            return self.emit(f"memref.load {at} : {view.type}", slot["dtype"])
        if isinstance(node, ast.BinOp):
            a, b = self.expr(node.left), self.expr(node.right)
            if a in self.lanes and b not in self.lanes:
                b = self.splat(b, *self.lanes[a])
            if b in self.lanes and a not in self.lanes:
                a = self.splat(a, *self.lanes[b])
            if a.type != b.type:
                raise ValueError("mixed operands require an explicit cast")
            integer = a.type in ("index", "i32", "i64")
            ops = (
                {
                    ast.Add: "addi",
                    ast.Sub: "subi",
                    ast.Mult: "muli",
                    ast.FloorDiv: "floordivsi",
                    ast.Mod: "remsi",
                }
                if integer
                else {
                    ast.Add: "addf",
                    ast.Sub: "subf",
                    ast.Mult: "mulf",
                    ast.Div: "divf",
                }
            )
            op = ops.get(type(node.op))
            if not op:
                raise ValueError("unsupported arithmetic on " + a.type)
            result = self.emit(f"arith.{op} {a.name}, {b.name} : {a.type}", a.type)
            if a in self.lanes:
                av, ax = self.lanes[a]
                bv, bx = self.lanes[b]
                if av["dims"][ax] != bv["dims"][bx]:
                    raise ValueError("member extents differ")
                self.lanes[result] = (av, ax)
            return result
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            a, b = self.expr(node.left), self.expr(node.comparators[0])
            pred = {ast.GtE: "sge", ast.Lt: "slt", ast.Eq: "eq"}[type(node.ops[0])]
            if a.type != b.type or a.type != "index":
                raise ValueError("expected an index predicate")
            return self.emit(f"arith.cmpi {pred}, {a.name}, {b.name} : index", "i1")
        if isinstance(node, ast.IfExp):
            pred = self.expr(node.test)
            if pred.type != "i1":
                raise ValueError("conditional requires a boolean")
            at = len(self.lines)
            result = self.fresh()
            a = self.expr(node.body)
            self.lines.insert(at, f"{result} = scf.if {pred.name} -> ({a.type}) {{")
            self.lines.extend([f"scf.yield {a.name} : {a.type}", "} else {"])
            b = self.expr(node.orelse)
            if b.type != a.type:
                raise ValueError("conditional results disagree")
            self.lines.extend([f"scf.yield {b.name} : {b.type}", "}"])
            return Value(result, a.type)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "dim":
                slot, axis = self.dim(node)
                v = self.literal(axis, "index")
                return self.emit(
                    f'memref.dim %{slot["name"]}, {v.name} : {argument_type(slot)}',
                    "index",
                )
            name = ast.unparse(node.func).removeprefix(self.ns + ".")
            if (
                name == "zeros"
                and len(node.args) == 2
                and ast.unparse(node.args[1]) == self.ns + ".f32"
            ):
                return self.splat(self.literal(0.0, "f32"), *self.dim(node.args[0]))
            if name == "sum" and len(node.args) == 1:
                a = self.expr(node.args[0])
                if a.type != "tensor<?xf32>":
                    raise ValueError("expected f32 members")
                return self.member_op("lane_sum", [a], "f32")
            if name in ("f32", "i32") and len(node.args) == 1:
                value = ast.literal_eval(node.args[0])
                return self.literal(float(value) if name == "f32" else int(value), name)
            if name == "cast":
                a = self.expr(node.args[0])
                ty = ast.unparse(node.args[1]).removeprefix(self.ns + ".")
                src = a.type.removeprefix("tensor<?x").removesuffix(">")
                if ty == src:
                    return a
                if (src, ty) not in {
                    ("f16", "f32"),
                    ("bf16", "f32"),
                    ("f32", "f16"),
                    ("f32", "bf16"),
                }:
                    raise ValueError("unsupported ordered cast")
                op = "extf" if ty == "f32" else "truncf"
                if a in self.lanes:
                    ty = f"tensor<?x{ty}>"
                result = self.emit(f"arith.{op} {a.name} : {a.type} to {ty}", ty)
                if a in self.lanes:
                    self.lanes[result] = self.lanes[a]
                return result
            if name in {"sqrt", "exp"} and len(node.args) == 1:
                a = self.expr(node.args[0])
                if a.type != "f32":
                    raise ValueError("expected f32 operand")
                return self.emit(f"math.{name} {a.name} : f32", "f32")
        raise ValueError("unsupported ordered expression: " + ast.unparse(node))

    def statements(self, stmts):
        for stmt in stmts:
            if isinstance(stmt, ast.For):
                self.loop(stmt)
                continue
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                raise ValueError(
                    "expected a local assignment, buffer write or ordered loop"
                )
            dest = stmt.targets[0]
            v = self.expr(stmt.value)
            if isinstance(dest, ast.Name):
                self.env[dest.id] = v
            elif isinstance(dest, ast.Subscript):
                slot, coords, axes = self.access(dest)
                view = self.env[slot["name"]]
                ty = f'tensor<?x{slot["dtype"]}>' if axes else slot["dtype"]
                if slot["direction"] not in ("Out", "InOut") or ty != v.type:
                    raise ValueError("invalid ordered store")
                if axes:
                    self.member_op("slice_write", [v, view, *coords], None, axes[0])
                else:
                    self.lines.append(
                        f"memref.store {v.name}, {view.name}["
                        + ", ".join(c.name for c in coords)
                        + f"] : {view.type}"
                    )
            else:
                raise ValueError("unsupported assignment target")

    def loop(self, stmt):
        it = stmt.iter
        if (
            not isinstance(stmt.target, ast.Name)
            or not isinstance(it, ast.Call)
            or ast.unparse(it.func) != self.ns + ".ordered"
            or it.keywords
            or len(it.args) not in (1, 2)
            or stmt.orelse
        ):
            raise ValueError("expected I.ordered(stop) or I.ordered(start,stop)")
        lo = self.literal(0, "index") if len(it.args) == 1 else self.expr(it.args[0])
        hi = self.expr(it.args[-1])
        step = self.literal(1, "index")
        if hi.type != "index" or lo.type != "index":
            raise ValueError("ordered bounds require indices")
        saved = self.env.copy()
        assigns = {
            a.id
            for s in stmt.body
            for a in ast.walk(s)
            if isinstance(a, ast.Name) and isinstance(a.ctx, ast.Store)
        }
        carry = [k for k in saved if k in assigns and k not in self.slots]
        iv = self.fresh()
        self.env[stmt.target.id] = Value(iv, "index")
        args = []
        types = []
        for k in carry:
            v = saved[k]
            name = self.fresh()
            self.env[k] = Value(name, v.type)
            if v in self.lanes:
                self.lanes[self.env[k]] = self.lanes[v]
            args.append(name + " = " + v.name)
            types.append(v.type)
        results = [self.fresh() for k in carry]
        self.lines.append(
            (", ".join(results) + " = " if carry else "")
            + f"scf.for {iv} = {lo.name} to {hi.name} step {step.name}"
            + (
                " iter_args(" + ", ".join(args) + ") -> (" + ", ".join(types) + ")"
                if carry
                else ""
            )
            + " {"
        )
        self.statements(stmt.body)
        if any(self.env[k].type != saved[k].type for k in carry):
            raise ValueError("loop carry type changed")
        if carry:
            self.lines.append(
                "scf.yield "
                + ", ".join(self.env[k].name for k in carry)
                + " : "
                + ", ".join(types)
            )
        self.lines.append("}")
        self.env = saved.copy()
        self.env.update((k, Value(r, t)) for k, r, t in zip(carry, results, types))
        for k in carry:
            if saved[k] in self.lanes:
                self.lanes[self.env[k]] = self.lanes[saved[k]]

    def capture(self):
        body = self.fn.body
        views = []
        axes = []
        ivs = []
        while len(body) == 1 and isinstance(body[0], ast.For):
            stmt = body[0]
            it = stmt.iter
            if (
                not isinstance(it, ast.Call)
                or ast.unparse(it.func) != self.ns + ".parallel"
            ):
                break
            if (
                it.keywords
                or len(it.args) != 1
                or stmt.orelse
                or not isinstance(stmt.target, ast.Name)
            ):
                raise ValueError("invalid parallel owner")
            slot, axis = self.dim(it.args[0])
            views.append(slot)
            axes.append(axis)
            iv = self.fresh()
            ivs.append(iv + ": index")
            self.env[stmt.target.id] = Value(iv, "index")
            body = stmt.body
        if not views:
            raise ValueError(
                "ordered kernel needs an enclosing independent owner domain"
            )
        self.statements(body)
        params = ", ".join(f'%{a["name"]}: {argument_type(a)}' for a in self.args)
        abi = json.dumps(json.dumps(self.args, separators=(",", ":")))
        bounds = ", ".join("%" + s["name"] for s in views)
        types = ", ".join(argument_type(s) for s in views)
        return (
            f"module {{func.func @{self.fn.name}({params}) attributes {{kir.abi={abi}}} {{\n"
            f'"kir.owner_space"({bounds}) ({{\n^bb0('
            + ", ".join(ivs)
            + "):\n"
            + "\n".join(self.lines)
            + '\n"kir.end"():()->()\n}) {axes=array<i64: '
            + ", ".join(map(str, axes))
            + f">}} : ({types})->()\nreturn\n}}}}"
        ).encode()
