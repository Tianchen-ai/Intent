"""MLU370 facts and BANG C specialization, not a definition of all DSAs."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field, fields
import json


@dataclass(frozen=True, slots=True)
class BangCCapabilities:
    clusters: int
    cores_per_cluster: int
    nram_bytes: int
    wram_bytes: int
    sram_bytes: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("BANG C capabilities require positive integer hardware capacities")


@dataclass(frozen=True, slots=True)
class BangCCompilationTarget:
    architecture: str
    shapes: tuple[tuple[str, tuple[int, ...]], ...] = field(default=(), compare=False)
    strides: tuple[tuple[str, tuple[int, ...]], ...] = field(default=(), compare=False)
    capabilities: BangCCapabilities | None = None

    def __post_init__(self):
        if not isinstance(self.capabilities, BangCCapabilities):
            raise TypeError("BANG C compilation requires device capabilities; local JIT discovers them")
        if self.architecture != "mtp_372":
            raise NotImplementedError("BANG C currently provides an MLU370 implementation profile")
        if any(not isinstance(name, str) or any(type(extent) is not int or extent < -1 for extent in shape)
               for name, shape in self.shapes):
            raise ValueError("BANG C shapes require integer extents, with -1 for dynamic axes")
        if any(not isinstance(name, str) or any(type(value) is not int or not -(1 << 63) <= value < (1 << 63)
               for value in strides) for name, strides in self.strides):
            raise ValueError("BANG C strides must be signed 64-bit element strides")

    @property
    def provider(self):
        return "bangc"

    def resolve(self):
        return self

    @property
    def compiler_options(self):
        options = asdict(self)
        options.update(shapes=dict(self.shapes), strides=dict(self.strides))
        return ("--target=bangc", f"--target-options={json.dumps(options)}")

    @property
    def compiler_role(self):
        return "Intent BANG C compiler"


def read_bangc_target(provider, values):
    expected = {item.name for item in fields(BangCCompilationTarget)} - {"shapes", "strides"}
    if values.keys() != expected:
        raise ValueError(f"BANG C target requires exactly these fields: {', '.join(sorted(expected))}")
    values["capabilities"] = BangCCapabilities(**values["capabilities"])
    return BangCCompilationTarget(**values)
