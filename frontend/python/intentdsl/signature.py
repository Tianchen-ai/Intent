import ast


_DENORMAL_MODES = ("ieee", "preserve-sign", "positive-zero", "dynamic")


def valid_denormal_mode(mode):
    if not isinstance(mode, str):
        return False
    parts = mode.split(",")
    return len(parts) <= 2 and all(part in _DENORMAL_MODES for part in parts)


def _kernel_decorator(fn):
    for decorator in fn.decorator_list:
        if (
            isinstance(decorator, ast.Attribute)
            and decorator.attr == "kernel"
            and isinstance(decorator.value, ast.Name)
        ):
            return decorator.value.id, ("source", False, "ieee")
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "kernel"
            and isinstance(decorator.func.value, ast.Name)
        ):
            allowed = {"numerics", "fast_math", "denormal_fp_math"}
            if decorator.args or any(k.arg not in allowed for k in decorator.keywords):
                raise ValueError(
                    "kernel decorator accepts only numerics, fast_math, and denormal_fp_math"
                )
            kw = {k.arg: k.value for k in decorator.keywords}
            if len(kw) != len(decorator.keywords):
                raise ValueError("duplicate kernel decorator keyword")
            val = ast.literal_eval(kw["numerics"]) if "numerics" in kw else "source"
            if val not in ("source", "relaxed_normalization"):
                raise ValueError(f"unsupported kernel numerics: {val}")
            fm = ast.literal_eval(kw["fast_math"]) if "fast_math" in kw else False
            if type(fm) is not bool:
                raise ValueError("fast_math must be a boolean literal")
            dm = (
                ast.literal_eval(kw["denormal_fp_math"])
                if "denormal_fp_math" in kw
                else ("preserve-sign" if fm else "ieee")
            )
            if not valid_denormal_mode(dm):
                raise ValueError(f"unsupported denormal_fp_math: {dm}")
            return decorator.func.value.id, (val, fm, dm)
    raise ValueError("expected a qualified kernel decorator")


def kernel_numeric_attributes(fn):
    """Only write author overrides; target defaults are bound in canonical KIR."""
    for decorator in fn.decorator_list:
        if not isinstance(decorator, ast.Call) or getattr(decorator.func, "attr", None) != "kernel":
            continue
        keys = {kw.arg for kw in decorator.keywords}
        numerics, fast, mode = _kernel_decorator(fn)[1]
        attrs = []
        if "denormal_fp_math" in keys or fast:
            attrs.append(f'"denormal-fp-math"="{mode}"')
        if "fast_math" in keys:
            attrs.append(f'kir.fp_contract="{"on" if fast else "off"}"')
            attrs.append("kir.author_fastmath")
        if "numerics" in keys:
            attrs.append(f'kir.normalization="{"online-eligible" if numerics == "relaxed_normalization" else "source"}"')
        return "".join(", " + attr for attr in attrs)
    return ""


def expression_denormal(keywords, kernel_mode):
    if "flush_to_zero" not in keywords:
        return ""
    enabled = ast.literal_eval(keywords["flush_to_zero"])
    mode = ("preserve-sign" if kernel_mode == "ieee" else kernel_mode) if enabled else "ieee"
    return f'"denormal-fp-math"="{mode}"'


def namespace(fn):
    return _kernel_decorator(fn)[0]


def kernel_numerics(fn):
    return _kernel_decorator(fn)[1][0]


def kernel_math_policy(fn):
    numerics, fast_math, denormal = _kernel_decorator(fn)[1]
    flags = ()
    if fast_math:
        flags = ("fast",)
    return flags, denormal


def format_fastmath_attr(flags):
    if not flags:
        return ""
    if "fast" in flags:
        return "#arith.fastmath<fast>"
    ordered = [
        f
        for f in ("reassoc", "nnan", "ninf", "nsz", "arcp", "contract", "afn")
        if f in flags
    ]
    return f'#arith.fastmath<{",".join(ordered)}>' if ordered else ""


def format_fastmath_inline(flags):
    if not flags:
        return ""
    if "fast" in flags:
        return " fastmath<fast>"
    ordered = [
        f
        for f in ("reassoc", "nnan", "ninf", "nsz", "arcp", "contract", "afn")
        if f in flags
    ]
    return f' fastmath<{",".join(ordered)}>' if ordered else ""


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
