from intent.frontend.semantics import OperationKind, StaticDim, TensorType
from intent.frontend.semantics.operations import QuantFormatKind
from intent.frontend.semantics.types import dims_compatible
from intent.language import f32, u8
from intent.language.builtins import QuantFormat

from .common import bind_declared_call, require_dtype


def _format(lowerer, node):
    value = lowerer.lower_expression(node)
    if not isinstance(value, QuantFormat) or value.name not in ("q4_k", "q8_k"):
        lowerer.error(node, "quantization requires I.quant.q4_k or I.quant.q8_k")
    return QuantFormatKind[value.name.upper()]


def _tensor(lowerer, node, dtype, columns):
    value = lowerer.read_value(lowerer.lower_expression(node), node)
    if (not isinstance(value.type, TensorType) or value.type.rank != 2
            or value.type.dtype != dtype or value.type.shape[1] != StaticDim(columns)):
        lowerer.error(node, f"quantization operand requires {dtype} [G,{columns}]")
    return value


def lower_quantization(lowerer, name, node):
    bound = bind_declared_call(lowerer, node, name)
    if name == "quantize":
        format = _format(lowerer, bound["format"])
        if format != QuantFormatKind.Q8_K:
            lowerer.error(node, "quantize only defines the Q8_K numerical operation")
        value = _tensor(lowerer, bound["values"], f32, 256)
        return lowerer.emit(
            OperationKind.QUANTIZE, lowerer.location(node), operands=(value,),
            result_types=(TensorType(u8, (value.type.shape[0], StaticDim(292))),),
            attributes={"format": format},
        ).results[0]
    lhs_format = _format(lowerer, bound["lhs_format"])
    rhs_format = _format(lowerer, bound["rhs_format"])
    if (lhs_format != QuantFormatKind.Q4_K or rhs_format != QuantFormatKind.Q8_K
            or require_dtype(lowerer, bound["acc_dtype"]) != f32):
        lowerer.error(node, "quantized_dot requires Q4_K x Q8_K with f32 accumulation")
    lhs = _tensor(lowerer, bound["lhs"], u8, 144)
    rhs = _tensor(lowerer, bound["rhs"], u8, 292)
    if not dims_compatible(lhs.type.shape[0], rhs.type.shape[0]):
        lowerer.error(node, "quantized_dot requires matching record extents")
    return lowerer.emit(
        OperationKind.QUANTIZED_DOT, lowerer.location(node), operands=(lhs, rhs),
        result_types=(TensorType(f32, ()),),
        attributes={"lhs_format": lhs_format, "rhs_format": rhs_format},
    ).results[0]
