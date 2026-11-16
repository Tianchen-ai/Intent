from dataclasses import asdict, dataclass
import json
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class AscendCapabilities:
    architecture: str
    cube_cores: int
    vector_cores: int

    def __post_init__(self):
        if not isinstance(self.architecture, str) or not self.architecture:
            raise ValueError("Ascend requires its actual architecture")
        if type(self.cube_cores) is not int or self.cube_cores < 0:
            raise ValueError("Ascend cube_cores must be nonnegative")
        if type(self.vector_cores) is not int or self.vector_cores <= 0:
            raise ValueError("Ascend vector_cores must be positive")


@dataclass(frozen=True, slots=True)
class AscendCompilationTarget:
    capabilities: AscendCapabilities
    provider: ClassVar[str] = "triton_ascend"

    def resolve(self):
        return self

    @property
    def compiler_role(self):
        return "Intent Triton-Ascend compiler"

    @property
    def compiler_options(self):
        return (f"--target={self.provider}", f"--target-options={json.dumps(asdict(self.capabilities))}")
