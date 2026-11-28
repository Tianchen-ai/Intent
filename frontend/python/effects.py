"""Capture the authored indexed reads, writes and atomic side effects."""
import ast

from .intentdsl.signature import spelling


def indexed_expression(capture, call, name):
    atomic = name == "atomic_add"
    options = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    expected = {"order": "relaxed", "in_bounds": True} if atomic else {"in_bounds": True}
    if options != expected or len(call.args) != (3 if atomic else 2):
        raise ValueError("indexed access needs its bounds promise and memory order")
    target = call.args[0].id
    slot = next(arg for arg in capture.args if arg["name"] == target)
    if (slot["direction"] != ("InOut" if atomic else "In")
            or slot["dtype"] != "i32" or len(slot["dims"]) != 1):
        raise ValueError("expected a one-dimensional i32 buffer for indexed access")
    args = [capture.values[target], capture.expr(call.args[1])]
    if atomic:
        args.append(capture.expr(call.args[2]))
        capture.updated.add(target)
    return capture.emit(name if atomic else "index_read", args, "tensor<?xi32>",
                        '{in_bounds,ordering="relaxed"}' if atomic else "{in_bounds}")


def statement(capture, call, outputs):
    name = spelling(call.func, capture.namespace)
    options = {kw.arg: kw.value for kw in call.keywords}
    row = ("%row", "index")
    view = capture.values
    if name == "atomic_add":
        indexed_expression(capture, call, name)
    elif name == "fill_ranges":
        dest, offsets, owner = call.args
        if (set(options) != {"unit", "disjoint_in_bounds"}
                or ast.literal_eval(options["disjoint_in_bounds"]) is not True
                or offsets.id != capture.owner or owner.id != capture.row):
            raise ValueError("range stores require disjoint in-bounds owner intervals")
        unit = ast.literal_eval(options["unit"])
        if type(unit) is not int or unit <= 0:
            raise ValueError("range unit must be a positive integer")
        capture.emit("fill_ranges", [view[dest.id], view[offsets.id], row],
                     attrs=f"{{unit={unit}:i64,disjoint_in_bounds}}")
        capture.updated.add(dest.id)
    elif name == "write_offsets":
        offsets, total, prefix = call.args
        if options:
            raise ValueError("write_offsets takes positional arguments")
        capture.emit("write_offsets", [view[offsets.id], view[total.id], row, capture.expr(prefix)])
        outputs.update((offsets.id, total.id))
    elif name == "encode_q8k":
        if options or len(call.args) != 3:
            raise ValueError("encode_q8k expects positional (out, row, values)")
        dest, owner_row, values_expr = call.args
        if (not isinstance(dest, ast.Name)
                or not isinstance(owner_row, ast.Name)
                or owner_row.id != capture.row):
            raise ValueError("encode_q8k must target a buffer by the parallel row variable")
        capture.emit("encode_q8k", [view[dest.id], row, capture.expr(values_expr)])
        outputs.add(dest.id)
    elif name == "scatter_unique" and len(call.args) == 3:
        dest, indices, payload = call.args
        if isinstance(dest, ast.Name):
            if set(options) != {"in_bounds"} or ast.literal_eval(options["in_bounds"]) is not True:
                raise ValueError("indexed scatter requires an explicit bounds promise")
            capture.emit("unique_put", [view[dest.id], capture.expr(indices), capture.expr(payload)],
                         attrs="{in_bounds}")
            capture.updated.add(dest.id)
            return
        if set(options) != {"valid", "fill", "count"}:
            raise ValueError("row scatter requires a predicate, fill and count destination")
        count = options["count"]
        if (not isinstance(dest, ast.Subscript) or not isinstance(dest.slice, ast.Tuple)
                or [ast.unparse(i) for i in dest.slice.elts] != [capture.row, ":"]
                or not isinstance(count, ast.Subscript) or ast.unparse(count.slice) != capture.row):
            raise ValueError("the scatter and its count must belong to the same row")
        fill = ast.literal_eval(options["fill"])
        if type(fill) is not int or not -(2**31) <= fill < 2**31:
            raise ValueError("expected an i32 scatter fill")
        capture.emit("prefix_scatter", [view[dest.value.id], view[count.value.id], row,
                     capture.expr(options["valid"]), capture.expr(indices), capture.expr(payload)],
                     attrs=f"{{fill={fill}:i32}}")
        outputs.update((dest.value.id, count.value.id))
    else:
        raise ValueError(f"unsupported effect statement {name}")
