"""CPU hardware facts and independently chosen execution budgets."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VectorCapabilities:
    minimum_bits: int
    scalable: bool

    def __post_init__(self):
        if type(self.minimum_bits) is not int or self.minimum_bits < 32 or self.minimum_bits % 32:
            raise ValueError("CPU vector minimum must be a positive multiple of 32 bits")
        if type(self.scalable) is not bool:
            raise TypeError("CPU scalable-vector capability must be boolean")


@dataclass(frozen=True, slots=True)
class ContractionCapability:
    lhs: str
    rhs: str
    accumulator: str
    flush_subnormals: bool

    def __post_init__(self):
        from intent.language.dtypes import dtype
        for value in (self.lhs, self.rhs, self.accumulator):
            dtype(value)
        if type(self.flush_subnormals) is not bool:
            raise TypeError("CPU contraction subnormal behavior must be explicit")


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    workers: int
    private_bytes: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in (self.workers, self.private_bytes)):
            raise ValueError("CPU worker and private-storage budgets must be positive integers")
