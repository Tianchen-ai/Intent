"""Capture independent owners and strictly ordered scalar loops into MLIR SSA."""

import ast
import json
from dataclasses import dataclass

from .intentdsl.signature import (
    argument_type,
    format_fastmath_attr,
    format_fastmath_inline,
    kernel_math_policy,
    namespace,
    signature,
)


@dataclass(frozen=True)
class Value:
    name: str
    type: str


class Ordered:
    def __init__(self, fn):
        self.fn = fn
        self.ns = namespace(fn)
        self.default_flags, self.denormal_mode = kernel_math_policy(fn)
        self.kernel_denormal_mode = self.denormal_mode
        self.args = signature(fn)
        self.slots = {a["name"]: a for a in self.args}
        self.env = {
            a["name"]: Value("%" + a["name"], argument_type(a)) for a in self.args
        }
        self.lines = []
        self.seq = 0
        self.lanes = {}
        self.fold_carries = set()
        self.ordered_carries = set()
        self.fold_depth = 0

    def fm_inline(self, extra=(), *, strip_reassoc=False):
        flags = [*self.default_flags, *extra]
        if strip_reassoc:
            if "fast" in flags:
                flags = ["nnan", "ninf", "nsz", "arcp", "contract", "afn"]
            else:
                flags = [f for f in flags if f != "reassoc"]
        return format_fastmath_inline(flags)

    def fm_attr(self, extra=()):
        return format_fastmath_attr((*self.default_flags, *extra))

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

    def member_op(self, op, args, result, axis=None, fastmath=""):
        items = []
        if axis is not None:
            items.append(f"axis={axis}:i64")
        if fastmath:
            items.append(f"fastmath={fastmath}")
        if op == "lane_sum":
            # Per-expression FTZ on exp/div does not license the dot reduction.
            items.append(f'"denormal-fp-math"="{self.kernel_denormal_mode}"')
        attrs = f' {{{", ".join(items)}}}' if items else ""
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
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if ast.unparse(node.operand) == self.ns + ".inf":
                return self.literal("0xFF800000", "f32")
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
            fm = ""
            if not integer:
                is_fold_carry = a in self.fold_carries
                is_ordered_carry = a in self.ordered_carries
                fm = self.fm_inline(
                    ("reassoc",) if is_fold_carry and op == "addf" else (),
                    strip_reassoc=is_ordered_carry,
                )
            result = self.emit(f"arith.{op} {a.name}, {b.name}{fm} : {a.type}", a.type)
            if a in self.lanes:
                av, ax = self.lanes[a]
                bv, bx = self.lanes[b]
                if av["dims"][ax] != bv["dims"][bx]:
                    raise ValueError("member extents differ")
                self.lanes[result] = (av, ax)
            return result
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            a, b = self.expr(node.left), self.expr(node.comparators[0])
            if a.type != b.type:
                raise ValueError("comparison requires matching scalar types")
            op_ty = type(node.ops[0])
            if a.type in ("index", "i32", "i64"):
                preds = {
                    ast.Gt: "sgt",
                    ast.GtE: "sge",
                    ast.Lt: "slt",
                    ast.LtE: "sle",
                    ast.Eq: "eq",
                    ast.NotEq: "ne",
                }
                if op_ty not in preds:
                    raise ValueError("unsupported integer comparison")
                return self.emit(
                    f"arith.cmpi {preds[op_ty]}, {a.name}, {b.name} : {a.type}", "i1"
                )
            if a.type == "f32":
                preds = {
                    ast.Gt: "ogt",
                    ast.GtE: "oge",
                    ast.Lt: "olt",
                    ast.LtE: "ole",
                    ast.Eq: "oeq",
                    ast.NotEq: "une",
                }
                if op_ty not in preds:
                    raise ValueError("unsupported floating comparison")
                return self.emit(
                    f"arith.cmpf {preds[op_ty]}, {a.name}, {b.name} : f32", "i1"
                )
            raise ValueError("unsupported comparison operand type: " + a.type)
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
            if name == "sum" and len(node.args) == 1 and not node.keywords:
                a = self.expr(node.args[0])
                if a.type != "tensor<?xf32>":
                    raise ValueError("expected f32 members")
                return self.member_op(
                    "lane_sum", [a], "f32", fastmath=self.fm_attr(("reassoc",))
                )
            if name == "fdiv" and len(node.args) == 2:
                kw = {k.arg: k.value for k in node.keywords}
                if len(kw) != len(node.keywords) or kw.keys() - {"approximate", "flush_to_zero"}:
                    raise ValueError("fdiv accepts only approximate and flush_to_zero")
                approx = ast.literal_eval(kw["approximate"]) if "approximate" in kw else False
                ftz = ast.literal_eval(kw["flush_to_zero"]) if "flush_to_zero" in kw else False
                if type(approx) is not bool or type(ftz) is not bool:
                    raise ValueError("fdiv flags must be boolean literals")
                if ftz and self.denormal_mode == "ieee":
                    self.denormal_mode = "preserve-sign"
                a, b = self.expr(node.args[0]), self.expr(node.args[1])
                if a in self.lanes and b not in self.lanes:
                    b = self.splat(b, *self.lanes[a])
                if b in self.lanes and a not in self.lanes:
                    a = self.splat(a, *self.lanes[b])
                if a.type != b.type or a.type in ("index", "i32", "i64"):
                    raise ValueError("fdiv requires matching floating operands")
                extra = ("afn", "arcp") if approx else ()
                result = self.emit(
                    f"arith.divf {a.name}, {b.name}{self.fm_inline(extra)} : {a.type}",
                    a.type,
                )
                if a in self.lanes:
                    self.lanes[result] = self.lanes[a]
                return result
            if name in ("f32", "i32") and len(node.args) == 1:
                value = ast.literal_eval(node.args[0])
                return self.literal(float(value) if name == "f32" else int(value), name)
            if name == "cast":
                a = self.expr(node.args[0])
                ty = ast.unparse(node.args[1]).removeprefix(self.ns + ".")
                src = a.type.removeprefix("tensor<?x").removesuffix(">")
                if ty == src:
                    return a
                if a not in self.lanes:
                    if (src, ty) in {
                        ("i32", "index"),
                        ("i64", "index"),
                        ("index", "i32"),
                        ("index", "i64"),
                    }:
                        return self.emit(
                            f"arith.index_cast {a.name} : {src} to {ty}", ty
                        )
                    if src in ("i32", "i64") and ty == "f32":
                        return self.emit(f"arith.sitofp {a.name} : {src} to f32", "f32")
                    if src == "index" and ty == "f32":
                        wide = self.emit(
                            f"arith.index_cast {a.name} : index to i64", "i64"
                        )
                        return self.emit(f"arith.sitofp {wide.name} : i64 to f32", "f32")
                    if src == "f32" and ty == "i32":
                        return self.emit(f"arith.fptosi {a.name} : f32 to i32", "i32")
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
            if name == "maximum_num" and len(node.args) == 2 and not node.keywords:
                a, b = (self.expr(arg) for arg in node.args)
                if a.type != "f32" or b.type != "f32":
                    raise ValueError("maximum_num requires f32 scalar operands")
                fm = self.fm_inline(
                    ("reassoc",) if a in self.fold_carries else (),
                    strip_reassoc=(a in self.ordered_carries),
                )
                return self.emit(f"arith.maxnumf {a.name}, {b.name}{fm} : f32", "f32")
            if name in {"sqrt", "exp", "exp2", "log"} and len(node.args) == 1:
                kw = {k.arg: k.value for k in node.keywords}
                if len(kw) != len(node.keywords) or kw.keys() - {"approximate", "flush_to_zero"}:
                    raise ValueError(f"unsupported keywords for {name}")
                approx = ast.literal_eval(kw["approximate"]) if "approximate" in kw else False
                ftz = ast.literal_eval(kw["flush_to_zero"]) if "flush_to_zero" in kw else False
                if type(approx) is not bool or type(ftz) is not bool:
                    raise ValueError("math flags must be boolean literals")
                if ftz and self.denormal_mode == "ieee":
                    self.denormal_mode = "preserve-sign"
                a = self.expr(node.args[0])
                if a.type != "f32":
                    raise ValueError("expected f32 operand")
                extra = ("afn",) if approx else ()
                return self.emit(
                    f"math.{name} {a.name}{self.fm_inline(extra)} : f32", "f32"
                )
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
            if (
                self.fold_depth > 0
                and isinstance(dest, ast.Subscript)
                and isinstance(stmt.value, ast.BinOp)
                and isinstance(stmt.value.op, ast.Add)
                and ast.unparse(stmt.value.left) == ast.unparse(dest)
            ):
                slot, coords, axes = self.access(dest)
                if not axes and slot["direction"] == "InOut":
                    view = self.env[slot["name"]]
                    at = view.name + "[" + ", ".join(c.name for c in coords) + "]"
                    lhs = self.emit(f"memref.load {at} : {view.type}", slot["dtype"])
                    rhs = self.expr(stmt.value.right)
                    if lhs.type != rhs.type or rhs in self.lanes:
                        raise ValueError("mixed operands require an explicit cast")
                    integer = lhs.type in ("index", "i32", "i64")
                    op = "addi" if integer else "addf"
                    fm = "" if integer else self.fm_inline(("reassoc",))
                    v = self.emit(
                        f"arith.{op} {lhs.name}, {rhs.name}{fm} : {lhs.type}",
                        lhs.type,
                    )
                    self.lines.append(
                        f"memref.store {v.name}, {view.name}["
                        + ", ".join(c.name for c in coords)
                        + f"] : {view.type}"
                    )
                    continue
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
            or ast.unparse(it.func) not in (self.ns + ".ordered", self.ns + ".fold")
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
        is_fold = ast.unparse(it.func) == self.ns + ".fold"
        saved_fold = self.fold_carries.copy()
        saved_ord = self.ordered_carries.copy()
        for k in carry:
            v = saved[k]
            name = self.fresh()
            arg_val = Value(name, v.type)
            self.env[k] = arg_val
            if v in self.lanes:
                self.lanes[arg_val] = self.lanes[v]
            (self.fold_carries if is_fold else self.ordered_carries).add(arg_val)
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
        if is_fold:
            self.fold_depth += 1
        try:
            self.statements(stmt.body)
        finally:
            if is_fold:
                self.fold_depth -= 1
        self.fold_carries = saved_fold
        self.ordered_carries = saved_ord
        if any(self.env[k].type != saved[k].type for k in carry):
            raise ValueError("loop carry type changed")
        if carry:
            self.lines.append(
                "scf.yield "
                + ", ".join(self.env[k].name for k in carry)
                + " : "
                + ", ".join(types)
            )
        self.lines.append("} {kir.reassociate}" if is_fold else "}")
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
            f'module {{func.func @{self.fn.name}({params}) attributes {{kir.abi={abi}, "denormal-fp-math"="{self.denormal_mode}"}} {{\n'
            f'"kir.owner_space"({bounds}) ({{\n^bb0('
            + ", ".join(ivs)
            + "):\n"
            + "\n".join(self.lines)
            + '\n"kir.end"():()->()\n}) {axes=array<i64: '
            + ", ".join(map(str, axes))
            + f">}} : ({types})->()\nreturn\n}}}}"
        ).encode()
