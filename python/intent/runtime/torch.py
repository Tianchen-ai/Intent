from __future__ import annotations

from inspect import Parameter, Signature
from typing import Protocol, runtime_checkable

from ..language.dtypes import DType, DTypeCategory
from .interface import ScalarParameter, ViewParameter


@runtime_checkable
class TorchOutputInference(Protocol):
    """Infer Torch outputs from the public interface without native execution."""

    def infer_outputs(self, arguments: tuple) -> object: ...


def torch_dtype(element: DType):
    import torch

    names = {"index": "int64", "bool": "bool", "i8": "int8", "i16": "int16",
             "i32": "int32", "i64": "int64", "u8": "uint8", "u16": "uint16",
             "u32": "uint32", "u64": "uint64", "f16": "float16", "bf16": "bfloat16",
             "f32": "float32", "f64": "float64", "f8e4m3fn": "float8_e4m3fn", "f8e5m2": "float8_e5m2"}
    return getattr(torch, names[element.name])


def register_operator(artifact, name: str):
    """Register the public invocation's outputs and mutations with PyTorch.

    Device execution is opaque to PyTorch. The fake implementation consumes the
    same exported interface as ordinary calls and never runs provider code.
    Mutable inputs must not share Torch storage with another input: automatic
    functionalization may clone those arguments independently of read-only ones.
    """
    import torch

    interface = artifact.interface
    if artifact.device_type not in {"cpu", "cuda"} or not isinstance(artifact.runtime, TorchOutputInference):
        raise NotImplementedError("PyTorch registration requires a CPU or CUDA tensor runtime with output inference")
    inputs = interface.inputs
    views = tuple((index, parameter) for index, parameter in enumerate(inputs)
                  if isinstance(parameter, ViewParameter))
    if not views:
        raise NotImplementedError("PyTorch registration requires an input tensor to determine device dispatch")
    interface.binding_relations(explicit_outputs=False).require_output_allocation()
    pieces = name.split("::")
    if len(pieces) != 2 or not all(piece.isidentifier() for piece in pieces):
        raise ValueError("operator name must be 'your_namespace::your_operator'")

    mutable_names = tuple(f"arg{index}" for index, parameter in views if parameter.writable)
    mutable_alias_pairs = tuple((left_index, right_index, left.name, right.name)
                               for index, (left_index, left) in enumerate(views)
                               for right_index, right in views[index + 1:]
                               if left.writable or right.writable)
    parameters = []
    for index, parameter in enumerate(inputs):
        if isinstance(parameter, ViewParameter):
            scalar = f"Tensor(a{index}!)" if parameter.writable else "Tensor"
        elif isinstance(parameter, ScalarParameter):
            if parameter.dtype.category is DTypeCategory.BOOL:
                scalar = "bool"
            elif parameter.dtype.category in {DTypeCategory.SIGNED_INTEGER, DTypeCategory.UNSIGNED_INTEGER, DTypeCategory.INDEX}:
                scalar = "SymInt"
            elif parameter.dtype.category in {DTypeCategory.FLOAT, DTypeCategory.BFLOAT}:
                scalar = "float"
            else:
                raise NotImplementedError(f"PyTorch scalar schema for {parameter.dtype}")
        parameters.append(f"{scalar} arg{len(parameters)}")
    returns = "Tensor" if len(interface.outputs) == 1 else (
        "(" + ", ".join("Tensor" for _ in interface.outputs) + ")")
    schema = f"({', '.join(parameters)}) -> {returns}"
    signature = Signature(Parameter(f"arg{index}", Parameter.POSITIONAL_OR_KEYWORD)
                          for index in range(len(parameters)))

    def function(invoke):
        def operator(*arguments, **keywords):
            arguments = signature.bind(*arguments, **keywords).args
            for left, right, left_name, right_name in mutable_alias_pairs:
                # Storage identity is available for both real and FakeTensor
                # inputs, without reading a pointer or tensor contents. The
                # fake check sees aliases before graph functionalization.
                if torch._C._is_alias_of(arguments[left], arguments[right]):
                    raise NotImplementedError(
                        f"PyTorch mutable calls require independent input storage: "
                        f"{left_name} and {right_name} share storage; use ordinary artifact calls for this binding")
            return invoke(*arguments)

        return operator

    operator = torch.library.custom_op(name, function(artifact.run), mutates_args=mutable_names,
                                       device_types=artifact.device_type, schema=schema)
    operator.register_fake(function(lambda *arguments: artifact.runtime.infer_outputs(arguments)))
    return operator
