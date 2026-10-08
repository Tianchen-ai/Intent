from __future__ import annotations

import builtins as python_builtins
from dataclasses import dataclass
from enum import Enum as PythonEnum
from enum import IntEnum
from typing import Any

from .dtypes import DType


ShapeDim = int | str
StrideDim = int | str | None


def _normalize_shape(shape: object) -> tuple[ShapeDim, ...]:
    if not isinstance(shape, (tuple, list)):
        raise TypeError("tensor shape annotation must be a tuple or list")

    normalized: list[ShapeDim] = []
    for dim in shape:
        if isinstance(dim, python_builtins.bool):
            raise TypeError("boolean is not a tensor shape dimension")
        if isinstance(dim, int):
            if dim < 0:
                raise ValueError("static tensor dimensions must be non-negative")
            normalized.append(dim)
            continue
        if isinstance(dim, str):
            if not dim or not dim.isidentifier():
                raise ValueError(f"invalid symbolic dimension: {dim!r}")
            normalized.append(dim)
            continue
        raise TypeError(f"unsupported tensor shape dimension: {dim!r}")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class ViewConstraints:
    strides: tuple[StrideDim, ...] | None = None
    alias: str | None = None
    noalias: bool = False

    def __post_init__(self) -> None:
        if self.strides is not None:
            normalized_strides = tuple(self.strides)
            for stride in normalized_strides:
                if stride is None:
                    continue
                if isinstance(stride, python_builtins.bool):
                    raise TypeError("boolean is not a stride constraint")
                if isinstance(stride, int):
                    continue
                if isinstance(stride, str) and stride and stride.isidentifier():
                    continue
                raise TypeError(f"unsupported stride constraint: {stride!r}")
            object.__setattr__(self, "strides", normalized_strides)
        if self.alias is not None and not self.alias:
            raise ValueError("alias group must not be empty")
        if self.alias is not None and self.noalias:
            raise ValueError("alias and noalias cannot be specified together")


def constraints(
    *,
    strides: tuple[StrideDim, ...] | None = None,
    alias: str | None = None,
    noalias: bool = False,
) -> ViewConstraints:
    return ViewConstraints(
        strides=strides,
        alias=alias,
        noalias=noalias,
    )


class ViewKind(IntEnum):
    IN = 0
    OUT = 1
    INOUT = 2


@dataclass(frozen=True, slots=True)
class ViewSpec:
    kind: ViewKind
    dtype: DType
    shape: tuple[ShapeDim, ...]
    constraints: ViewConstraints

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ViewKind):
            raise TypeError("view kind must be a ViewKind")
        if not isinstance(self.dtype, DType):
            raise TypeError("view element type must be an Intent dtype")
        if not isinstance(self.constraints, ViewConstraints):
            raise TypeError("view constraints must be a ViewConstraints")
        normalized = _normalize_shape(self.shape)
        object.__setattr__(self, "shape", normalized)
        if (
            self.constraints.strides is not None
            and len(self.constraints.strides) != len(normalized)
        ):
            raise ValueError("stride constraint rank must match tensor rank")


class _ViewAnnotation:
    kind: ViewKind

    @classmethod
    def __class_getitem__(cls, parameters: object) -> ViewSpec:
        if not isinstance(parameters, tuple):
            raise TypeError("view annotation expects dtype and shape")
        if len(parameters) not in (2, 3):
            raise TypeError("view annotation expects dtype, shape, and optional constraints")

        element_type = parameters[0]
        shape = parameters[1]
        view_constraints = parameters[2] if len(parameters) == 3 else ViewConstraints()
        if not isinstance(view_constraints, ViewConstraints):
            raise TypeError("third view annotation argument must be I.constraints(...)")
        return ViewSpec(
            kind=cls.kind,
            dtype=element_type,
            shape=_normalize_shape(shape),
            constraints=view_constraints,
        )


class In(_ViewAnnotation):
    kind = ViewKind.IN


class Out(_ViewAnnotation):
    """Output view with no defined incoming elements.

    Define an element before reading it in a subsequent ordered operation.
    For example, a full-domain copy may initialize an output before an ordered
    factorization loop reads and updates it.
    """

    kind = ViewKind.OUT


class InOut(_ViewAnnotation):
    kind = ViewKind.INOUT


@dataclass(frozen=True, slots=True)
class ConstexprSpec:
    value_type: Any


class Constexpr:
    @classmethod
    def __class_getitem__(cls, value_type: Any) -> ConstexprSpec:
        if value_type is None:
            raise TypeError("Constexpr requires a value type")
        return ConstexprSpec(value_type=value_type)


class Enum(PythonEnum):
    """Base class for user specialization enums."""
