import ast

from .capture import Capture


def root_ir(source: bytes) -> bytes:
    (definition,) = ast.parse(source.decode()).body
    if not isinstance(definition, ast.FunctionDef):
        raise ValueError("expected one kernel definition")
    if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
           n.func.attr=="ordered" for n in ast.walk(definition)):
        from .ordered import Ordered
        return Ordered(definition).capture()
    if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
           n.func.attr in {"contract","window_contract","scaled_contract","q4k_dot"} for n in ast.walk(definition)):
        from .contractions import capture
        return capture(definition)
    return Capture(definition).capture()
