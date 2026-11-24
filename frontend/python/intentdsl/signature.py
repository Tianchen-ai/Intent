import ast


def _kernel_decorator(fn):
    for decorator in fn.decorator_list:
        if (
            isinstance(decorator, ast.Attribute)
            and decorator.attr == "kernel"
            and isinstance(decorator.value, ast.Name)
        ):
            return decorator.value.id, "source"
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "kernel"
            and isinstance(decorator.func.value, ast.Name)
        ):
            if decorator.args or any(k.arg != "numerics" for k in decorator.keywords):
                raise ValueError("kernel decorator accepts only keyword numerics")
            kw = {k.arg: k.value for k in decorator.keywords}
            if len(kw) != len(decorator.keywords):
                raise ValueError("duplicate kernel decorator keyword")
            val = ast.literal_eval(kw["numerics"]) if "numerics" in kw else "source"
            if val not in ("source", "relaxed_normalization"):
                raise ValueError(f"unsupported kernel numerics: {val}")
            return decorator.func.value.id, val
    raise ValueError("expected a qualified kernel decorator")


def namespace(fn):
    return _kernel_decorator(fn)[0]


def kernel_numerics(fn):
    return _kernel_decorator(fn)[1]


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
        if dtype not in ("f16", "bf16", "f32", "i32", "u64") and not (dtype in ("i64","i8","u8","f8e4m3fn") and dims):
            raise ValueError(f"unsupported element type: {dtype}")
        args.append(dict(name=param.arg, dtype=dtype, dims=dims, direction=direction))
    return args


def element_type(dtype):
    return {"u64":"ui64","u8":"i8","f8e4m3fn":"f8E4M3FN"}.get(dtype,dtype)

def argument_type(arg):
    rank = len(arg["dims"])
    element = element_type(arg["dtype"])
    if not rank:
        return element
    strides = ", ".join(["?"] * rank)
    return f'memref<{"?x"*rank}{element}, strided<[{strides}], offset: ?>>'
