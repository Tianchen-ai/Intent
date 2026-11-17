from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True, slots=True)
class CompileOptions:
    """Numerical permissions and optimization controls for one compilation.

    ``source`` preserves the language contract, including its existing FMA and
    parallel-reduction permissions; it does not promise bitwise reproducibility.
    ``relaxed_normalization`` additionally permits normalized-summary segment
    rescaling, including changes to the reference and placement of low-precision
    weight casts. Scores of valid members must be finite. All value elements
    actually entering the moment contraction must be finite, including elements
    multiplied by zero weights: zero times infinity is not an inactive access.
    Masked accesses retain their declared fill semantics. Restructuring can change rounding, underflow and
    overflow, while empty/all-inactive inputs retain their identity. This is not
    global fast math: input precision, FTZ, effects and the ABI are unchanged.
    ``optimizations`` selects registered optional transformations by MLIR name;
    enabling one never grants additional numerical permission. Remarks report
    compiler decisions and do not select an algorithm or a tuning configuration.
    """

    numerics: Literal["source", "relaxed_normalization"] = "source"
    optimizations: Mapping[str, bool] = field(default_factory=dict)
    optimization_remarks: bool = False

    def __post_init__(self) -> None:
        if self.numerics not in ("source", "relaxed_normalization"):
            raise ValueError("numerics must be 'source' or 'relaxed_normalization'")
        if not isinstance(self.optimizations, Mapping) or any(
            not isinstance(name, str) or not name or type(enabled) is not bool
            for name, enabled in self.optimizations.items()
        ):
            raise TypeError("optimizations must map registered component names to booleans")
        if type(self.optimization_remarks) is not bool:
            raise TypeError("optimization_remarks must be a boolean")
        object.__setattr__(self, "optimizations", MappingProxyType(dict(sorted(self.optimizations.items()))))

    @classmethod
    def read(cls, metadata: dict) -> CompileOptions:
        """Read the complete policy emitted by the compiler, without defaults."""
        if not isinstance(metadata, dict) or set(metadata) != {
            "numerics", "optimizations", "optimization_remarks"
        }:
            raise ValueError("compiler options require numerics, optimizations and optimization_remarks")
        return cls(**metadata)

    @property
    def compiler_arguments(self) -> tuple[str, ...]:
        return (f"--numerics={self.numerics}",
                f"--optimization-remarks={str(self.optimization_remarks).lower()}",
                *(f"--optimization={name}={str(enabled).lower()}" for name, enabled in self.optimizations.items()))

    def metadata(self) -> dict:
        return {"numerics": self.numerics, "optimizations": dict(self.optimizations),
                "optimization_remarks": self.optimization_remarks}


def compilation_arguments(options: CompileOptions | None) -> tuple[str, ...]:
    if options is None:
        return ()
    if not isinstance(options, CompileOptions):
        raise TypeError("options must be an intent.CompileOptions or None")
    return options.compiler_arguments
