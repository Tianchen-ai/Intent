from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def lowering_scope(lowerer, **bindings):
    """Temporarily enter a lexical lowering context, including failed lowering."""
    previous = {name: getattr(lowerer, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(lowerer, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(lowerer, name, value)
