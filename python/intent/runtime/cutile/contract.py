"""cuTile host recipes read without importing cuTile or a device runtime."""
from __future__ import annotations

from dataclasses import dataclass

from ..contract import integer_field, name_field, object_field, sequence_field
from ..gpu.contract import optional_name, require_references, view_binding, native_names
from ..gpu.expressions import Expression, read_expressions
from ..gpu.interface import PublicArgument


@dataclass(frozen=True, slots=True)
class ArrayView:
    name: str
    eligible: str
    base: int
    group_ends: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ArrayIndexBound:
    array: str
    bounds: tuple[Expression, ...]
    eligible: str | None


@dataclass(frozen=True, slots=True)
class CuTileFacts:
    kernel: str
    narrow_kernel: str | None
    index_tile_bounds: tuple[ArrayIndexBound, ...] | None
    array_views: tuple[ArrayView, ...]
    compiler_hints: tuple[tuple[str, str], ...]
    inferred_worker_warps: int
    tuning_key_scalars: tuple[str, ...]
    kernel_arguments: tuple[str, ...]

    @classmethod
    def read(cls, entry, interface):
        entry = object_field(entry, "cuTile facts")
        arrays = []
        for array in sequence_field(entry["array_views"], "cuTile array views"):
            view = view_binding(array["base"], interface)
            rank = len(view.parameter.shape) if isinstance(view, PublicArgument) else len(view.shape)
            ends = sequence_field(array["group_ends"], "array view groups")
            if not ends or any(type(end) is not int or not 0 < end <= rank for end in ends) or any(a >= b for a, b in zip(ends, ends[1:])) or ends[-1] != rank:
                raise ValueError("cuTile array groups must partition their source view axes")
            arrays.append(ArrayView(name_field(array["name"], "array view name"),
                                    name_field(array["eligible"], "array eligibility name"), array["base"], ends))
        narrow = optional_name(entry["narrow_kernel"], "cuTile narrow kernel")
        bounds = entry["index_tile_bounds"]
        if bounds is not None:
            expected = {
                view.kernel_name: (len(view.parameter.shape) if isinstance(view, PublicArgument)
                                   else len(view.shape), None)
                for view in interface.views
            }
            for array in arrays:
                if array.name in expected:
                    raise ValueError("cuTile array aliases require distinct native names")
                expected[array.name] = (len(array.group_ends), array.eligible)
            parsed = []
            for value in sequence_field(bounds, "cuTile index bounds"):
                value = object_field(value, "cuTile array index bound")
                array = name_field(value["array"], "bounded native array")
                eligible = optional_name(value["eligible"], "bounded array eligibility")
                extents = read_expressions(sequence_field(value["bounds"], "array tile bounds"))
                if array not in expected or expected.pop(array) != (len(extents), eligible):
                    raise ValueError("cuTile index bounds must identify each native array and its axes")
                require_references(extents, interface)
                parsed.append(ArrayIndexBound(array, extents, eligible))
            if expected:
                raise ValueError("cuTile index bounds must cover every native array")
            bounds = tuple(parsed)
        if (bounds is None) != (narrow is None):
            raise ValueError("cuTile narrow kernel requires its index-bound recipes")
        hints = object_field(entry["compiler_hints"], "cuTile compiler hints")
        if set(hints) - {"occupancy", "num_ctas", "num_worker_warps"}:
            raise NotImplementedError("cuTile compiler hints are not supported by this runtime")
        if any(parameter not in interface.configuration_space.bound_names for parameter in hints.values()):
            raise ValueError("cuTile compiler hint must reference a bound parameter")
        available = set(interface.by_name) | {parameter.name for parameter in interface.configuration_space.parameters}
        available.update(overlap.name for overlap in interface.overlaps)
        available.update(name for array in arrays for name in (array.name, array.eligible))
        keys = native_names(entry["tuning_key_scalars"], (scalar.kernel_name for scalar in interface.scalars), "cuTile scalar tuning key")
        arguments = native_names(entry["kernel_arguments"], available, "cuTile kernel arguments")
        return cls(name_field(entry["kernel"], "cuTile kernel"), narrow, bounds, tuple(arrays), tuple(hints.items()),
                   integer_field(entry["inferred_worker_warps"], "inferred cuTile worker warps"), keys, arguments)
