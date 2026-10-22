"""Installed author algorithms and programs; independent of Bench."""
from .registry import PROGRAMS, load_program
__all__ = ["PROGRAMS", "load_program"]
