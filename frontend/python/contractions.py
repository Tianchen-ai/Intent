"""Capture whole-view contractions and their epilogues in SSA order."""
import ast
import json

from .intentdsl.signature import (
    argument_type,
    element_type,
    format_fastmath_attr,
    kernel_math_policy,
    kernel_numeric_attributes,
    namespace,
    signature,
    spelling,
)


def capture(fn):
    api = namespace(fn)
    default_flags, _ = kernel_math_policy(fn)
    args = signature(fn)
    slots = {a["name"]: a for a in args}
    outputs = [a for a in args if a["direction"] == "Out"]
    if len(outputs) != 1:
        raise ValueError("a contraction region requires one destination")
    out, = outputs
    rank = len(out["dims"])
    values, lines = {}, []

    def tensor(n, dtype):
        return f'tensor<{"?x"*n}{element_type(dtype)}>'

    def view(node, direction="In"):
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
            raise ValueError("expected a whole buffer view")
        slot = slots[node.value.id]
        axes = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
        if (slot["direction"] != direction or len(axes) != len(slot["dims"]) or
                any(not isinstance(i, ast.Slice) or i.lower or i.upper or i.step for i in axes)):
            raise ValueError("expected one full slice per view axis")
        return "%" + slot["name"], argument_type(slot)

    def emit(name, inputs, result, attributes=""):
        operands = ", ".join(x for x, _ in inputs)
        types = ", ".join(t for _, t in inputs)
        ssa = f"%v{len(lines)}"
        assignment = f"{ssa} = " if result else ""
        lines.append(f'{assignment}"kir.{name}"({operands}) {attributes} : ({types}) -> {result or "()"}')
        return ssa, result

    def expr(node):
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            acc = expr(node.left)
            return emit("add_columns", [acc, view(node.right)], acc[1])
        if not isinstance(node, ast.Call):
            raise ValueError("unsupported contraction expression")
        name = spelling(node.func, api)
        kw = {k.arg: k.value for k in node.keywords}
        if len(kw) != len(node.keywords):
            raise ValueError("duplicate contraction keyword")
        if name == "cast" and len(node.args) == 2 and not kw:
            return emit("convert_matrix", [expr(node.args[0])], tensor(rank, spelling(node.args[1], api)))
        if name == "quantize_q8k" and len(node.args) == 1 and not kw:
            src = view(node.args[0])
            dims = len(slots[node.args[0].value.id]["dims"])
            if dims not in (2, 3):
                raise ValueError("quantize_q8k expects [G,256] or [B,G,256] view")
            res_ty = "tensor<?x292xi8>" if dims == 2 else "tensor<?x?x292xi8>"
            return emit("quantize_q8k", [src], res_ty)
        if name == "window_contract":
            from .window import capture_value_map
            if len(node.args)!=4 or set(kw)!={"head_group","rhs","acc"} or spelling(kw["acc"],api)!="f32":
                raise ValueError("window_contract needs four views, a head group, an RHS map and an f32 accumulator")
            group=ast.literal_eval(kw["head_group"])
            if type(group) is not int or group<1:raise ValueError("expected a positive group size")
            inputs=[view(a) for a in node.args]
            ssa=f"%v{len(lines)}";result=tensor(rank,"f32")
            region=capture_value_map(kw["rhs"],api)
            wfm = format_fastmath_attr((*default_flags, "reassoc", "contract"))
            lines.append(f'{ssa} = "kir.window_contract"('+', '.join(a for a,_ in inputs)+') ({'+region+
                f'}}) {{fastmath={wfm}, head_group={group}:i64}} : ('+', '.join(t for _,t in inputs)+f')->{result}')
            return ssa,result
        if name == "q4k_dot" and len(node.args) == 2 and not kw:
            qfm = format_fastmath_attr((*default_flags, "reassoc", "contract"))
            act = view(node.args[1]) if isinstance(node.args[1], ast.Subscript) else expr(node.args[1])
            return emit("q4k_dot", [view(node.args[0]), act], tensor(rank, "f32"), f"{{fastmath={qfm}}}")
        if name == "quant_contract" and len(node.args) in (2, 3, 4) and set(kw) == {"format", "acc"}:
            fmt = ast.literal_eval(kw["format"])
            if type(fmt) is not str or not fmt:
                raise ValueError("quant_contract requires a string format name")
            acc_ty = spelling(kw["acc"], api)
            qfm = format_fastmath_attr((*default_flags, "reassoc", "contract"))
            inputs = []
            for idx, arg_node in enumerate(node.args):
                if isinstance(arg_node, ast.Subscript):
                    inputs.append(view(arg_node))
                elif idx == 3 and isinstance(arg_node, ast.Call) and spelling(arg_node.func, api) == "transpose":
                    if len(arg_node.args) != 2 or arg_node.keywords or ast.literal_eval(arg_node.args[1]) != (1, 0):
                        raise ValueError("right scale transpose must swap (1, 0)")
                    inputs.append(emit("transpose_scale", [view(arg_node.args[0])], tensor(2, "u8")))
                else:
                    inputs.append(expr(arg_node))
            return emit("quant_contract", inputs, tensor(rank, acc_ty), f'{{fastmath={qfm}, format={json.dumps(fmt)}}}')
        if name == "scaled_contract" and len(node.args) == 4 and set(kw) == {"acc"}:
            sfm = format_fastmath_attr((*default_flags, "reassoc", "contract"))
            transpose = node.args[3]
            if isinstance(transpose,ast.Subscript):
                return emit("scaled_contract",[view(a) for a in node.args],tensor(rank,spelling(kw["acc"],api)),f"{{fastmath={sfm}}}")
            if (not isinstance(transpose, ast.Call) or spelling(transpose.func, api) != "transpose" or
                    len(transpose.args) != 2 or transpose.keywords or ast.literal_eval(transpose.args[1]) != (1, 0)):
                raise ValueError("right scale requires an explicit [G,N] to [N,G] transpose")
            scale = emit("transpose_scale", [view(transpose.args[0])], tensor(2, "u8"))
            return emit("scaled_contract", [view(a) for a in node.args[:3]] + [scale],
                        tensor(rank, spelling(kw["acc"], api)), f"{{fastmath={sfm}}}")
        if name != "contract" or len(node.args) != 2 or not {"axes","acc"} <= kw.keys() or kw.keys()-{"axes","acc","batch","offsets","routes","allow_subnormal_ftz"}:
            raise ValueError("expected contract(lhs, rhs, axes=..., acc=...)")
        lhs, rhs = (view(a) for a in node.args)
        axes = ast.literal_eval(kw["axes"])
        batch = ast.literal_eval(kw["batch"]) if "batch" in kw else ((), ())
        ftz = ast.literal_eval(kw["allow_subnormal_ftz"]) if "allow_subnormal_ftz" in kw else False
        if type(ftz) is not bool:
            raise ValueError("allow_subnormal_ftz must be a boolean literal")
        if len(axes)!=2 or any(len(x)!=1 for x in axes) or len(batch)!=2 or any(len(x)>1 for x in batch):
            raise ValueError("expected paired contraction axes and at most one batch axis per view")
        if "routes" in kw and "offsets" not in kw:
            raise ValueError("route indices require segment offsets")
        attrs = dict(lhs_k=axes[0][0], rhs_k=axes[1][0],
                     lhs_batch=next(iter(batch[0]),-1), rhs_batch=next(iter(batch[1]),-1))
        if any(type(x) is not int for x in attrs.values()):
            raise ValueError("axis indices must be integers")
        acc_ty = spelling(kw["acc"], api)
        attr_items = []
        if "allow_subnormal_ftz" in kw:
            if acc_ty == "i32":
                raise ValueError("allow_subnormal_ftz requires floating-point operands")
            attr_items.append('"denormal-fp-math"="preserve-sign"' if ftz else '"denormal-fp-math"="ieee"')
        if acc_ty != "i32":
            cfm = format_fastmath_attr((*default_flags, "reassoc", "contract"))
            attr_items.append(f"fastmath={cfm}")
        attr_items.extend(f"{k}={v}:i64" for k, v in attrs.items())
        attr = "{"+", ".join(attr_items)+"}"
        extras = [view(kw[k]) for k in ("offsets","routes") if k in kw]
        return emit("contract",[lhs,rhs,*extras],tensor(rank,acc_ty),attr)

    for stmt in fn.body:
        if not isinstance(stmt,ast.Assign) or len(stmt.targets)!=1:
            raise ValueError("expected a contraction expression assignment")
        target, = stmt.targets
        value=expr(stmt.value)
        if isinstance(target,ast.Name):
            values[target.id]=value
        else:
            emit("publish_matrix",[view(target,"Out"),value],None)
    sig=", ".join(f'%{a["name"]}: {argument_type(a)}' for a in args)
    abi=json.dumps(json.dumps(args,separators=(",",":")))
    return (f'module {{func.func @{fn.name}({sig}) attributes {{kir.abi={abi}{kernel_numeric_attributes(fn)}}} {{\n'
            f'"kir.matrix_domain"(%{out["name"]}) ({{\n'+'\n'.join(lines)+
            f'\n"kir.end"():()->()\n}}):({argument_type(out)})->()\nreturn\n}}}}').encode()
