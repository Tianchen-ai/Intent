"""Dispatch structured intrinsic calls to their semantic family."""
from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from .collectives import lower_arg_reduce_max, lower_histogram, lower_reduce, lower_scan
from .region_operations import lower_region_fold, lower_region_scan
from .contractions import (
    lower_contract,
    lower_scaled_contract,
    lower_sparse_contract,
    lower_sparse_contract_2to4,
    lower_sparse_format,
)

if TYPE_CHECKING:
    from ..ast.context import FunctionLowerer

def lower_structured_intrinsic(
    lowerer: FunctionLowerer,
    name: str,
    node: ast.Call,
) -> object:
    if name in (
        "reduce",
        "reduce.max",
        "reduce.sum",
        "reduce.any",
        "reduce.all",
    ):
        return lower_reduce(lowerer, name, node)
    if name == "arg_reduce.max":
        return lower_arg_reduce_max(lowerer, node)
    if name in ("scan", "cumsum", "cummax"):
        return lower_scan(lowerer, node, name)
    if name == "region_fold":
        return lower_region_fold(lowerer, node)
    if name == "region_scan":
        return lower_region_scan(lowerer, node)
    if name == "contract":
        return lower_contract(lowerer, node)
    if name in ("quantize", "quantized_dot"):
        from .quantization import lower_quantization

        return lower_quantization(lowerer, name, node)
    if name == "scaled_contract":
        return lower_scaled_contract(lowerer, node)
    if name in ("sparse.one_of_two", "sparse.two_of_four"):
        return lower_sparse_format(lowerer, name, node)
    if name == "sparse_contract":
        return lower_sparse_contract(lowerer, node)
    if name == "sparse_contract_2to4":
        return lower_sparse_contract_2to4(lowerer, node)
    if name == "histogram":
        return lower_histogram(lowerer, node)
    return NotImplemented
