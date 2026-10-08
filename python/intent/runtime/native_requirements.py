from __future__ import annotations

from dataclasses import dataclass

from .interface import AliasCheck, PublicInterface, ViewParameter, byte_spans_overlap
from .invocation import ViewFacts


def _check_geometry_relation(actual, expected, message: str) -> None:
    if actual != expected:
        raise NotImplementedError(message)


@dataclass(frozen=True, slots=True)
class NativeViewRequirements:
    layout: str
    alignment: int


@dataclass(frozen=True, slots=True)
class NativeRequirements:
    """Physical entry eligibility, separate from author shape and alias contracts."""

    interface: PublicInterface
    views: tuple[NativeViewRequirements | None, ...]
    disjoint: frozenset[tuple[int, int]]

    @classmethod
    def read(cls, metadata: dict, interface: PublicInterface) -> NativeRequirements:
        try:
            requirements = metadata["native"]["requirements"]
        except KeyError as error:
            raise ValueError(
                "native artifact lacks explicit entry requirements; regenerate it from the original Intent definition or KIR"
            ) from error
        views = [None] * len(interface.parameters)
        for entry in requirements["views"]:
            parameter = entry["parameter"]
            if (type(parameter) is not int or not 0 <= parameter < len(views) or
                    not isinstance(interface.parameters[parameter], ViewParameter)):
                raise ValueError("native view requirement references an unknown public view")
            if views[parameter] is not None:
                raise ValueError("native entry has duplicate view requirements")
            layout, alignment = entry["layout"], entry["alignment"]
            if layout not in {"strided", "contiguous"}:
                raise ValueError("native entry has an unknown view layout requirement")
            if type(alignment) is not int or alignment <= 0:
                raise ValueError("native entry alignment must be a positive byte count")
            views[parameter] = NativeViewRequirements(layout, alignment)
        if any(views[parameter.position] is None for parameter in interface.views):
            raise ValueError("native entry must declare requirements for every public view")
        disjoint = set()
        for entry in requirements["disjoint"]:
            left, right = entry["left"], entry["right"]
            if (type(left) is not int or type(right) is not int or
                    not 0 <= left < right < len(views) or
                    views[left] is None or views[right] is None):
                raise ValueError("native disjointness requirement must name two ordered public views")
            if not (interface.parameters[left].writable or interface.parameters[right].writable):
                raise ValueError("native disjointness requires a writable view")
            if (left, right) in disjoint:
                raise ValueError("native entry has duplicate disjointness requirements")
            disjoint.add((left, right))
        return cls(interface, tuple(views), frozenset(disjoint))

    def check_geometry(self, parameter: ViewParameter, facts: ViewFacts, *,
                       check_relation=_check_geometry_relation) -> None:
        """Check address-independent entry requirements, including abstract views."""
        requirements = self.views[parameter.position]
        if requirements.layout == "contiguous":
            expected_stride = 1
            for extent, stride in zip(reversed(facts.shape), reversed(facts.strides), strict=True):
                check_relation(stride, expected_stride,
                               f"{parameter.name}: native entry requires canonical contiguous element strides")
                expected_stride *= extent

    def check_storage(self, parameter: ViewParameter, facts: ViewFacts) -> None:
        """Check actual pointer requirements; abstract views have no addresses."""
        requirements = self.views[parameter.position]
        if facts.pointer % requirements.alignment:
            raise ValueError(
                f"{parameter.name}: native entry requires {requirements.alignment}-byte pointer alignment")

    def check_pair(self, check: AliasCheck, left: ViewFacts, right: ViewFacts) -> None:
        if (check.left, check.right) not in self.disjoint:
            return
        if byte_spans_overlap((left.begin, left.end), (right.begin, right.end)):
            left_name = self.interface.parameters[check.left].name
            right_name = self.interface.parameters[check.right].name
            raise NotImplementedError(
                f"{left_name} and {right_name}: native entry requires nonoverlapping accessed byte spans")
