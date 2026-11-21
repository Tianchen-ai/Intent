import ast

from .capture import Capture


def root_ir(source: bytes) -> bytes:
    (definition,) = ast.parse(source.decode()).body
    if not isinstance(definition, ast.FunctionDef):
        raise ValueError("expected one kernel definition")
    return Capture(definition).capture()
