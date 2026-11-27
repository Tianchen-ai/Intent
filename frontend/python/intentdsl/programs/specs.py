"""Lightweight pure-Python kernel and multi-kernel program descriptors."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelSpec:
    """Identifies a program's underlying kernel and mutable InOut parameter indices."""

    name: str
    mutable_inputs: tuple[int, ...] = ()


@dataclass
class Histogram:
    """Multi-kernel descriptor for histogram partials + merge."""

    partials: object
    merge: object
    chunk: int = 8192
    mutable_inputs: tuple[int, ...] = ()


@dataclass
class GroupNormBackward:
    """Multi-kernel descriptor for GroupNorm dx + weight/bias gradients."""

    dx: object
    affine: object
    mutable_inputs: tuple[int, ...] = ()


@dataclass
class PagedDecode:
    """Multi-kernel descriptor for split-K paged decode partials + merge."""

    partials: object
    merge: object
    mutable_inputs: tuple[int, ...] = ()


@dataclass
class MoEAlignment:
    """Multi-kernel descriptor for count/prefix/scatter/mark MoE alignment."""

    count: object
    prefix: object
    scatter: object
    mark: object
    mutable_inputs: tuple[int, ...] = ()
