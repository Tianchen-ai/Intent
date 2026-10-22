from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetProfile:
    march: str
    abi: str
    vlen_bits: int
    cpus: tuple[int, ...]
    matrix_extension: str | None = None
    required_extensions: tuple[str, ...] = ()
    private_stack_bytes: int = 65536

    def __post_init__(self) -> None:
        if not isinstance(self.march, str) or not isinstance(self.abi, str):
            raise TypeError("Weft march and ABI must be strings")
        if type(self.vlen_bits) is not int or type(self.private_stack_bytes) is not int:
            raise TypeError("Weft VLEN and private stack budget must be integers")
        if not isinstance(self.cpus, (tuple, list)) or any(type(cpu) is not int for cpu in self.cpus):
            raise TypeError("Weft execution CPUs must be a sequence of integers")
        if not isinstance(self.required_extensions, (tuple, list)) or any(
                not isinstance(extension, str) or not extension for extension in self.required_extensions):
            raise TypeError("Weft required extensions must be a sequence of names")
        if self.matrix_extension is not None and not isinstance(self.matrix_extension, str):
            raise TypeError("Weft matrix extension must be a name or None")
        object.__setattr__(self, "cpus", tuple(self.cpus))
        object.__setattr__(self, "required_extensions", tuple(sorted(set(self.required_extensions))))
        if not self.march.startswith("rv64") or self.abi != "lp64d":
            raise NotImplementedError("native Weft currently requires RV64/lp64d")
        if self.vlen_bits <= 0 or not self.cpus or any(cpu < 0 for cpu in self.cpus) or self.private_stack_bytes <= 0:
            raise ValueError("native Weft requires an explicit VLEN and CPU execution set")
        if any(extension != self.matrix_extension for extension in self.required_extensions):
            raise ValueError("required matrix extension is absent from the target profile")

    @classmethod
    def from_deployment(cls, deployment: dict) -> TargetProfile:
        return cls(**{name: deployment[name] for name in cls.__dataclass_fields__ if name in deployment})
