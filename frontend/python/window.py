"""Scalar SSA map over a projected right-hand contraction panel."""

import ast
from .intentdsl.signature import spelling


def capture_value_map(fn, api):
    if (
        not isinstance(fn, ast.Lambda)
        or len(fn.args.args) != 4
        or fn.args.defaults
        or fn.args.vararg
        or fn.args.kwarg
        or fn.args.kwonlyargs
        or fn.args.posonlyargs
    ):
        raise ValueError("expected a four-argument scalar RHS map")
    types = ("bf16", "f32", "f32", "f32")
    args = {a.arg: (f"%rhs{i}", t) for i, (a, t) in enumerate(zip(fn.args.args, types))}
    code = ["^bb0(" + ",".join(f"{v}:{t}" for v, t in args.values()) + "):"]

    def emit(expr, dtype):
        value = f"%map{len(code)}"
        code.append(value + "=" + expr)
        return value, dtype

    def lower(n):
        if isinstance(n, ast.Name):
            return args[n.id]
        if isinstance(n, ast.BinOp):
            (a, t), (b, u) = lower(n.left), lower(n.right)
            op = {ast.Add: "addf", ast.Sub: "subf", ast.Mult: "mulf"}.get(type(n.op))
            if t == u == "f32" and op:
                return emit(f"arith.{op} {a},{b}:f32", t)
        if isinstance(n, ast.Call) and not n.keywords:
            op = spelling(n.func, api)
            if op == "f32" and len(n.args) == 1:
                return emit(
                    f"arith.constant {float(ast.literal_eval(n.args[0]))}:f32", "f32"
                )
            if op == "cast" and len(n.args) == 2:
                a, t = lower(n.args[0])
                u = spelling(n.args[1], api)
                if (t, u) in (("bf16", "f32"), ("f32", "bf16")):
                    return emit(
                        f'arith.{"extf" if u=="f32" else "truncf"} {a}:{t} to {u}', u
                    )
            if op == "minimum" and len(n.args) == 2:
                (a, t), (b, u) = map(lower, n.args)
                if t == u == "f32":
                    return emit(f"arith.minimumf {a},{b}:f32", "f32")
            if op == "exp" and len(n.args) == 1:
                a, t = lower(n.args[0])
                if t == "f32":
                    return emit(f"math.exp {a}:f32", "f32")
        raise ValueError("unsupported scalar panel expression " + ast.unparse(n))

    result, dtype = lower(fn.body)
    if dtype != "bf16":
        raise ValueError("the right operand must round to BF16 before contraction")
    code.append(f'"kir.panel_value"({result}):(bf16)->()')
    return "\n".join(code)
