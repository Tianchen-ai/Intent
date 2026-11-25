import ast

from .capture import Capture


def root_ir(source: bytes) -> bytes:
    (definition,) = ast.parse(source.decode()).body
    if not isinstance(definition, ast.FunctionDef):
        raise ValueError("expected one kernel definition")
    nested = definition.body
    parallel_depth = 0
    while (len(nested)==1 and isinstance(nested[0],ast.For) and
           isinstance(nested[0].iter,ast.Call) and isinstance(nested[0].iter.func,ast.Attribute)
           and nested[0].iter.func.attr=="parallel"):
        parallel_depth += 1
        nested=nested[0].body
    if parallel_depth>1 or any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
           n.func.attr in ("ordered","fold") for n in ast.walk(definition)):
        from .ordered import Ordered
        return Ordered(definition).capture()
    if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
           n.func.attr in {"contract","window_contract","scaled_contract","q4k_dot","quant_contract"} for n in ast.walk(definition)):
        from .contractions import capture
        return capture(definition)
    return Capture(definition).capture()
