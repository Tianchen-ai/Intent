from __future__ import annotations

import ast

from .control import lower_control_intrinsic
from .tensor import lower_tensor_intrinsic
from .structured import lower_structured_intrinsic
from .matrix import lower_matrix_intrinsic
from .memory import lower_memory_intrinsic


def lower_intrinsic(lowerer: object, name: str, node: ast.Call) -> object:
    for handler in (
        lower_control_intrinsic,
        lower_tensor_intrinsic,
        lower_matrix_intrinsic,
        lower_structured_intrinsic,
        lower_memory_intrinsic,
    ):
        result = handler(lowerer, name, node)
        if result is not NotImplemented:
            return result
    lowerer.error(node, f"Intent intrinsic I.{name} is not implemented by the frontend")


__all__ = ["lower_intrinsic"]
