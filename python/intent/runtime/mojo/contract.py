"""Pure interpretation of the compiler's Mojo compilation units."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..contract import CPUImplementation, integer_field, name_field, names_field, object_field, sequence_field
from ..native import NativeABI


@dataclass(frozen=True, slots=True)
class MojoCandidate:
    entry: str
    values: tuple[int, ...]
    implementations: tuple[CPUImplementation, ...]
    source: str = field(repr=False)

    def metadata(self) -> dict:
        return {"entry": self.entry, "values": list(self.values),
                "implementations": [value.metadata() for value in self.implementations]}


@dataclass(frozen=True, slots=True)
class MojoFacts:
    candidates: tuple[MojoCandidate, ...]
    native_dependencies: tuple[str, ...]

    @classmethod
    def read(cls, source: str, metadata: dict, abi: NativeABI) -> MojoFacts:
        encoded = source.encode("utf-8")
        end = integer_field(metadata["source_prelude_end"], "Mojo source prelude end")
        if end > len(encoded):
            raise ValueError("Mojo source prelude exceeds the UTF-8 source length")
        prelude = encoded[:end].decode("utf-8")
        dependencies = names_field(metadata["native_dependencies"], "Mojo native dependencies")
        candidates, entries, ranges = [], set(), []
        for raw in sequence_field(metadata["candidates"], "Mojo candidates"):
            candidate = object_field(raw, "Mojo candidate")
            entry = name_field(candidate["entry"], "Mojo candidate entry")
            if entry in entries:
                raise ValueError("Mojo candidate entries must be distinct")
            entries.add(entry)
            extent = sequence_field(candidate["source_range"], "Mojo candidate source range")
            if len(extent) != 2:
                raise ValueError("Mojo candidate source range needs two UTF-8 byte offsets")
            begin, stop = (integer_field(value, "Mojo candidate source offset") for value in extent)
            if not end <= begin < stop <= len(encoded):
                raise ValueError("Mojo candidate source range is outside its source or overlaps the prelude")
            ranges.append((begin, stop))
            values = tuple(integer_field(value, "CPU configuration value", minimum=-(1 << 63))
                           for value in sequence_field(candidate["values"], "CPU configuration values"))
            implementations = tuple(CPUImplementation.read(value) for value in
                                    sequence_field(candidate["implementations"], "CPU implementations"))
            candidates.append(MojoCandidate(entry, values, implementations,
                                            prelude + encoded[begin:stop].decode("utf-8")))
        if not candidates:
            raise ValueError("Mojo generation requires at least one candidate")
        ordered = sorted(ranges)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("Mojo candidate source ranges must not overlap")
        return cls(tuple(candidates), dependencies)
