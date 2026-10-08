from __future__ import annotations

from collections.abc import Callable, Sequence
import ctypes
from dataclasses import dataclass
from math import prod

from ..language.dtypes import DType, dtype
from .interface import AliasCheck, PublicInterface, ScalarParameter, ViewParameter
from .invocation import ViewFacts, build_invocation_binders
from .native_requirements import NativeRequirements
from .diagnostics import ConfigurationAssessment, NativeObservation, invocation_arguments, resolve_observation


_CARRIERS = {"ptr": ctypes.c_void_p, "bool": ctypes.c_bool,
             "i8": ctypes.c_int8, "i16": ctypes.c_int16, "i32": ctypes.c_int32,
             "i64": ctypes.c_int64, "f32": ctypes.c_float, "f64": ctypes.c_double}

@dataclass(slots=True)
class BoundArguments:
    arguments: tuple[object, ...]
    native_arguments: tuple[object, ...]
    outputs: tuple[object, ...]
    key: tuple[object, ...]
    views: tuple[ViewFacts | None, ...]


@dataclass(frozen=True, slots=True)
class NativeSlot:
    parameter: ScalarParameter | ViewParameter
    role: str
    carrier: str
    axis: int | None = None
    element: DType | None = None

    @property
    def name(self) -> str:
        name = f"a{self.parameter.position}"
        return name if self.axis is None else f"{name}_{'d' if self.role == 'extent' else 's'}{self.axis}"


@dataclass(frozen=True, slots=True)
class NativeTrialRegion:
    """Contiguous readable state preserved around native benchmark invocations."""

    pointer: NativeSlot
    extents: tuple[NativeSlot, ...]
    element_bytes: int

    def byte_count(self, shape: tuple[int, ...]) -> int:
        if len(shape) != len(self.extents):
            raise ValueError("native trial region shape disagrees with its extent slots")
        return prod(shape) * self.element_bytes


class NativeABI:
    """CPU/DSA pointer, extent and stride slots over the public declaration."""

    def __init__(self, interface: PublicInterface, slots: tuple[NativeSlot, ...],
                 requirements: NativeRequirements) -> None:
        self.interface = interface
        self.slots = slots
        self.requirements = requirements
        self.pointer_slots = {slot.parameter.position: slot for slot in slots if slot.role == "pointer"}

    @classmethod
    def read(cls, metadata: dict) -> NativeABI:
        interface = PublicInterface.read(metadata["interface"])
        slots, seen = [], set()
        for entry in metadata["native"]["slots"]:
            position = entry["parameter"]
            if type(position) is not int or not 0 <= position < len(interface.parameters):
                raise ValueError("native slot references an unknown public parameter")
            parameter, role, carrier = interface.parameters[position], entry["role"], entry["carrier"]
            axis, element = None, None
            if ("axis" in entry) != (role in {"extent", "stride"}) or ("element" in entry) != (role == "pointer"):
                raise ValueError("native slot fields disagree with its role")
            if carrier not in _CARRIERS:
                raise NotImplementedError(f"native scalar carrier {carrier!r} is unsupported")
            if role == "scalar":
                if not isinstance(parameter, ScalarParameter) or carrier == "ptr":
                    raise ValueError("native scalar slot requires a scalar public parameter and carrier")
            elif isinstance(parameter, ViewParameter):
                if role == "pointer":
                    if carrier != "ptr":
                        raise ValueError("native pointer slot requires a pointer carrier")
                    element = dtype(entry["element"])
                elif role in {"extent", "stride"}:
                    axis = entry["axis"]
                    if type(axis) is not int or not 0 <= axis < len(parameter.shape) or carrier != "i64":
                        raise ValueError("native metadata slot requires a view axis and signed 64-bit carrier")
                else:
                    raise ValueError("unknown native view slot role")
            else:
                raise ValueError("native view slot requires a public view parameter")
            identity = position, role, axis
            if identity in seen:
                raise ValueError("native ABI has duplicate parameter slots")
            seen.add(identity)
            slots.append(NativeSlot(parameter, role, carrier, axis, element))
        for parameter in interface.parameters:
            required = "pointer" if isinstance(parameter, ViewParameter) else "scalar"
            if (parameter.position, required, None) not in seen:
                raise ValueError("native ABI does not bind every public parameter")
            if isinstance(parameter, ViewParameter):
                for axis in range(len(parameter.shape)):
                    if any((parameter.position, role, axis) not in seen for role in ("extent", "stride")):
                        raise ValueError("native ABI must bind the extent and stride of every view axis")
        requirements = NativeRequirements.read(metadata, interface)
        return cls(interface, tuple(slots), requirements)

    def argument_types(self) -> tuple[object, ...]:
        return tuple(_CARRIERS[slot.carrier] for slot in self.slots)

    def trial_regions(self) -> tuple[NativeTrialRegion, ...]:
        """Derive readable mutable byte ranges from this entry's actual ABI.

        Out-only views need no initial content. Arguments keep their original
        pointers, so restoring these ranges preserves all accepted view aliases.
        """
        regions = []
        for parameter in self.interface.mutable_inputs:
            if self.requirements.views[parameter.position].layout != "contiguous":
                raise NotImplementedError("native benchmark state requires contiguous InOut views")
            pointer = self.pointer_slots[parameter.position]
            bits = pointer.element.bits
            if bits is None:
                raise NotImplementedError("native benchmark state requires a known pointee byte width")
            extents = tuple(sorted(
                (slot for slot in self.slots
                 if slot.parameter.position == parameter.position and slot.role == "extent"),
                key=lambda slot: slot.axis,
            ))
            regions.append(NativeTrialRegion(pointer, extents, (bits + 7) // 8))
        return tuple(regions)

    def binders(
        self, *,
        observe_view: Callable[[object, ViewParameter, object], ViewFacts],
        allocate_output: Callable[[object, ViewParameter, tuple[int, ...]], tuple[object, ViewFacts]],
        check_alias: Callable[[AliasCheck, ViewFacts, ViewFacts], None],
        view_key: Callable[[ViewFacts, int], object] | None = None,
        scalar_key: Callable[[ScalarParameter, object], object] | None = None,
        scalar_argument: Callable[[ScalarParameter, object], object] | None = None,
        check_view_geometry: Callable[[object, ViewParameter, ViewFacts], None] | None = None,
        check_view_storage: Callable[[object, ViewParameter, ViewFacts], None] | None = None,
        check_dimensions: Callable[[object, dict[int, int]], None] | None = None,
    ) -> tuple[Callable[[object, tuple[object, ...]], BoundArguments], ...]:
        """Compose the common invocation binder with native slots and tuning keys.

        Both functions are compiled once from the declared interface. Each call
        observes fresh facts, then packs only the compiler's explicit native slots.
        check_alias enforces family entry requirements; the common binder already
        checks the author's alias and noalias contract.
        """
        if (view_key is None) != (scalar_key is None):
            raise ValueError("a native tuning key requires both view and scalar components")

        def tuple_expression(entries: Sequence[str]) -> str:
            return "(" + ", ".join(entries) + ("," if entries else "") + ")"

        public_binders = build_invocation_binders(
            self.interface, observe_view=observe_view, allocate_output=allocate_output,
            check_view_geometry=check_view_geometry, check_view_storage=check_view_storage,
            check_alias_requirements=check_alias, check_dimensions=check_dimensions,
        )
        result = []
        for public_bind in public_binders:
            namespace = {"_bind_public": public_bind, "_BoundArguments": BoundArguments,
                         "_view_key": view_key, "_scalar_key": scalar_key,
                         "_scalar_argument": scalar_argument}
            for parameter in self.interface.parameters:
                namespace[f"_p{parameter.position}"] = parameter
            lines = [
                "def bind(program, arguments):",
                "    public = _bind_public(program, arguments)",
            ]
            for parameter in self.interface.parameters:
                index = parameter.position
                if isinstance(parameter, ViewParameter):
                    lines.append(f"    f{index} = public.views[{index}]")
                else:
                    lines.append(f"    v{index} = public.arguments[{index}]")
            native, key = [], []
            if view_key is not None:
                lines.append("    groups = {}")
            for slot in self.slots:
                index = slot.parameter.position
                if slot.role == "pointer":
                    native.append(f"f{index}.pointer")
                elif slot.role == "scalar":
                    native.append(f"v{index}" if scalar_argument is None else f"_scalar_argument(_p{index}, v{index})")
                else:
                    native.append(f"f{index}.{'shape' if slot.role == 'extent' else 'strides'}[{slot.axis}]")
            for parameter in self.interface.parameters:
                index = parameter.position
                value, view = f"v{index}", f"f{index}"
                if isinstance(parameter, ViewParameter):
                    if view_key is not None:
                        lines.append(f"    g{index} = groups.setdefault({view}.allocation_identity, len(groups))")
                        key.append(f"_view_key({view}, g{index})")
                else:
                    if scalar_key is not None:
                        key.append(f"_scalar_key(_p{index}, {value})")
            lines.append(f"    return _BoundArguments(public.arguments, {tuple_expression(native)}, "
                         f"public.outputs, {tuple_expression(key)}, public.views)")
            code = compile("\n".join(lines) + "\n", "<intent.native.abi>", "exec", dont_inherit=True)
            exec(code, namespace)
            result.append(namespace["bind"])
        return tuple(result)


class NativePreparedRuntime:
    """Public preparation capability delegating to a native runtime's binder.

    The existing tuple-based prepare method remains the provider entry for
    run/launch and native benchmark callers. Neither preparation path executes.
    """

    interface: PublicInterface
    _observation: NativeObservation | None = None

    @property
    def observation(self) -> NativeObservation | None:
        """Latest native call snapshot; reading does not choose or execute a candidate."""
        return resolve_observation(self._observation)

    def describe_arguments(self, bound: BoundArguments, device: str):
        return invocation_arguments(self.interface, bound.arguments, bound.views, device)

    def inspect_configurations(self):
        """Read a CPU portfolio already accepted by this entry's binder."""
        return tuple(ConfigurationAssessment(description, ()) for description in self.configuration_descriptions)

    def run(self, *arguments):
        call = self.prepare(arguments)
        call.launch()
        return call.result()

    def launch(self, *arguments) -> None:
        self.prepare(arguments, explicit_outputs=True).launch()

    def prepare_call(self, arguments: tuple, *, outputs: tuple | None = None):
        if outputs is None:
            return self.prepare(arguments)
        return self.prepare(self.interface.explicit_arguments(arguments, outputs), explicit_outputs=True)
