import ast


def namespace(fn):
    for decorator in fn.decorator_list:
        if (
            isinstance(decorator, ast.Attribute)
            and decorator.attr == "kernel"
            and isinstance(decorator.value, ast.Name)
        ):
            return decorator.value.id
    raise ValueError("expected a qualified kernel decorator")


def spelling(node, owner):
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == owner
    ):
        return node.attr
    raise ValueError(f"expected an Intent DSL name, got {ast.unparse(node)}")


def signature(fn):
    owner = namespace(fn)
    args = []
    for param in fn.args.args:
        ann = param.annotation
        if isinstance(ann, ast.Subscript):
            dtype, shape = ann.slice.elts
            direction = spelling(ann.value, owner)
            dims = list(ast.literal_eval(shape))
            if (
                direction not in ("In", "Out", "InOut")
                or not dims
                or not all(isinstance(x,str) or type(x) is int and x>=0 for x in dims)
            ):
                raise ValueError(f"unsupported view annotation: {ast.unparse(ann)}")
        else:
            dtype, dims, direction = ann, [], "Scalar"
        dtype = spelling(dtype, owner)
        if dtype not in ("f16", "bf16", "f32", "i32", "u64") and not (dtype=="i64" and dims):
            raise ValueError(f"unsupported element type: {dtype}")
        args.append(dict(name=param.arg, dtype=dtype, dims=dims, direction=direction))
    return args


def argument_type(arg):
    rank = len(arg["dims"])
    element = "ui64" if arg["dtype"] == "u64" else arg["dtype"]
    if not rank:
        return element
    strides = ", ".join(["?"] * rank)
    return f'memref<{"?x"*rank}{element}, strided<[{strides}], offset: ?>>'
