"""Local, explicitly selected provider JIT over the ordinary Intent compiler."""
from .kernel import JITKernel, jit

__all__ = ["JITKernel", "jit"]
