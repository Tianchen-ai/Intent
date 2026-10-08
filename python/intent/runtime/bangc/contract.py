"""Read the compiler's complete BANG C candidates without loading NeuWare."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..contract import integer_field, name_field, object_field, sequence_field
from ..interface import ViewParameter
from ..native import NativeABI


CONFIGURATION_COLUMNS = ("tile", "tile_m", "tile_n", "tile_k", "region_tile", "tasks", "local_bytes")


@dataclass(frozen=True, slots=True)
class BangCCandidate:
    entry: str
    configuration: tuple[tuple[str, int], ...]
    full_extent_dimensions: tuple[int, ...]
    source: str = field(repr=False)

    @property
    def tile(self) -> int:
        return dict(self.configuration)["tile"]

    def metadata(self) -> dict:
        return {"entry": self.entry, **dict(self.configuration)}


@dataclass(frozen=True, slots=True)
class BangCFacts:
    candidates: tuple[BangCCandidate, ...]

    @classmethod
    def read(cls, source: str, metadata: dict, abi: NativeABI) -> BangCFacts:
        encoded = source.encode("utf-8")
        end = integer_field(metadata["source_prelude_end"], "BANG C source prelude end")
        if end > len(encoded):
            raise ValueError("BANG C source prelude exceeds the UTF-8 source length")
        prelude = encoded[:end].decode("utf-8")
        known = {identity for parameter in abi.interface.parameters if isinstance(parameter, ViewParameter)
                 for identity in parameter.dimensions if identity > 0}
        candidates, entries, configurations, ranges = [], set(), set(), []
        for raw in sequence_field(metadata["candidates"], "BANG C candidates"):
            candidate = object_field(raw, "BANG C candidate")
            entry = name_field(candidate["entry"], "BANG C native entry")
            if entry in entries:
                raise ValueError("BANG C native candidate entries must be distinct")
            entries.add(entry)
            values = object_field(candidate["configuration"], "BANG C configuration")
            if set(values) != set(CONFIGURATION_COLUMNS):
                raise ValueError("BANG C configuration must bind every declared column")
            configuration = tuple((name, integer_field(values[name], f"BANG C {name}", minimum=1))
                                  for name in CONFIGURATION_COLUMNS)
            if values["tile"] % 64:
                raise ValueError("BANG C vector tiles must be divisible by 64")
            if configuration in configurations:
                raise ValueError("BANG C configurations must be distinct")
            configurations.add(configuration)
            dimensions = tuple(integer_field(value, "BANG C full-extent dimension", minimum=1)
                               for value in sequence_field(candidate["full_extent_dimensions"],
                                                           "BANG C full-extent dimensions"))
            if len(set(dimensions)) != len(dimensions) or not set(dimensions).issubset(known):
                raise ValueError("BANG C full-extent obligations must name distinct public dimensions")
            extent = sequence_field(candidate["source_range"], "BANG C candidate source range")
            if len(extent) != 2:
                raise ValueError("BANG C candidate source range requires two UTF-8 byte offsets")
            begin, stop = (integer_field(value, "BANG C source offset") for value in extent)
            if not end <= begin < stop <= len(encoded):
                raise ValueError("BANG C candidate source range is outside its source or overlaps the prelude")
            ranges.append((begin, stop))
            candidates.append(BangCCandidate(entry, configuration, dimensions,
                                             prelude + encoded[begin:stop].decode("utf-8")))
        if not candidates:
            raise ValueError("BANG C generation requires at least one candidate")
        ordered = sorted(ranges)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("BANG C candidate source ranges must not overlap")
        return cls(tuple(candidates))
