"""Triton host recipes read without importing Triton or a device runtime."""
from __future__ import annotations

from dataclasses import dataclass

from ..contract import integer_field, name_field, object_field, sequence_field
from ..gpu.contract import optional_name, require_references, view_binding, native_names
from ..gpu.expressions import Expression, read_expressions
from ..gpu.interface import PublicArgument


@dataclass(frozen=True, slots=True)
class Descriptor:
    name: str
    base: int
    rank: int
    shape: tuple[Expression, ...]
    strides: tuple[Expression, ...]
    block_shape: tuple[Expression, ...]
    require_positive_shape: bool
    require_positive_strides: bool
    alignment: int
    maximum_shape_extent: int
    padding: str
    aligned_stride_axes: tuple[int, ...]
    unit_stride_axes: tuple[int, ...]

    @classmethod
    def read(cls, entry, interface):
        entry = object_field(entry, "Triton descriptor")
        view = view_binding(entry["base"], interface)
        rank = integer_field(entry["rank"], "descriptor rank", minimum=1)
        view_rank = len(view.parameter.shape) if isinstance(view, PublicArgument) else len(view.shape)
        if rank != view_rank:
            raise ValueError("descriptor rank disagrees with its view")
        expressions = tuple(read_expressions(entry[field]) for field in ("shape", "strides", "block_shape"))
        for values in expressions:
            if len(values) != rank:
                raise ValueError("descriptor shape and stride recipes must match its rank")
            require_references(values, interface)
        flags = tuple(entry[field] for field in ("require_positive_shape", "require_positive_strides"))
        if any(type(flag) is not bool for flag in flags):
            raise TypeError("descriptor requirements must be boolean")
        axes = []
        for field in ("aligned_stride_axes", "unit_stride_axes"):
            values = sequence_field(entry[field], field)
            if any(type(axis) is not int or not 0 <= axis < rank for axis in values):
                raise ValueError("descriptor stride requirement references an invalid axis")
            axes.append(values)
        return cls(name_field(entry["name"], "descriptor name"), entry["base"], rank, *expressions,
                   flags[0], flags[1], integer_field(entry["alignment"], "descriptor alignment", minimum=1),
                   integer_field(entry["maximum_shape_extent"], "maximum descriptor extent", minimum=1),
                   name_field(entry["padding"], "descriptor padding"), *axes)


@dataclass(frozen=True, slots=True)
class DescriptorChoice:
    config: str
    eligibility: str


@dataclass(frozen=True, slots=True)
class TritonFacts:
    kernel: str
    kernel_parameters: tuple[str, ...]
    native_options: tuple[tuple[str, str], ...]
    metadata_argument: str | None
    overlap_argument: str | None
    descriptor_choice: DescriptorChoice | None
    descriptors: tuple[Descriptor, ...]
    allocator: bool
    kernel_arguments: tuple[str, ...]
    autotune_key: tuple[str, ...]

    @classmethod
    def read(cls, entry, interface):
        entry = object_field(entry, "Triton facts")
        parameters = {parameter.name for parameter in interface.configuration_space.parameters}
        kernel_parameters = native_names(entry["kernel_parameters"], parameters, "Triton kernel parameters")
        options = object_field(entry["native_options"], "Triton native options")
        if set(options) != {"num_warps", "num_stages", "num_ctas"}:
            raise ValueError("Triton native options must bind warps, stages and CTAs")
        for parameter in options.values():
            if parameter not in interface.configuration_space.bound_names:
                raise ValueError("Triton native option must reference a bound configuration parameter")
        if (set(kernel_parameters) - set(interface.configuration_space.coverage_names)) | set(options.values()) != interface.configuration_space.bound_names:
            raise ValueError("Triton kernel and native options do not bind the complete configuration")
        metadata = optional_name(entry["metadata_argument"], "Triton metadata argument")
        overlap = optional_name(entry["overlap_argument"], "Triton overlap argument")
        if bool(interface.metadata) != (metadata is not None) or bool(interface.overlaps) != (overlap is not None):
            raise ValueError("Triton packed arguments disagree with the physical interface")
        choice = entry["descriptor_choice"]
        if choice is not None:
            choice = object_field(choice, "Triton descriptor choice")
            if choice["config"] not in interface.configuration_space.bound_names:
                raise ValueError("Triton descriptor choice has no configuration binding")
            choice = DescriptorChoice(choice["config"], name_field(choice["eligibility"], "descriptor eligibility"))
        descriptors = tuple(Descriptor.read(value, interface) for value in sequence_field(entry["descriptors"], "descriptors"))
        allocator = entry["allocator"]
        if allocator is not None:
            if allocator != {"size_argument": 0, "alignment_argument": 1, "stream_argument": 2,
                              "lifetime": "launch", "implementation": "torch_cuda_current_device"}:
                raise NotImplementedError("Triton descriptor allocator ABI is not supported by this runtime")
        if bool(descriptors) != (choice is not None) or bool(descriptors) != (allocator is not None):
            raise ValueError("Triton descriptors require their choice and allocator declarations")
        available = set(interface.by_name) | parameters
        available.update(name for name in (metadata, overlap, choice.eligibility if choice else None) if name is not None)
        arguments = native_names(entry["kernel_arguments"], available, "Triton kernel arguments")
        key = native_names(entry["autotune_key"], available, "Triton tuning key")
        return cls(name_field(entry["kernel"], "Triton kernel"), kernel_parameters, tuple(options.items()),
                   metadata, overlap, choice, descriptors, allocator is not None, arguments, key)
