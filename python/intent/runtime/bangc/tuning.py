"""Invocation-private trial storage using the existing CNRT buffer ownership."""
from __future__ import annotations

from .buffer import DeviceBuffer, DeviceView, ELEMENT_BYTES


class TrialState:
    def __init__(self, program, arguments):
        self._copies = []
        self._owners = []
        shadows = {}
        try:
            for parameter in program.interface.views:
                if not parameter.writable:
                    continue
                value = arguments[parameter.position]
                owner = value.owner if isinstance(value, DeviceView) else value
                if owner in shadows:
                    continue
                snapshot = owner.to_host()
                shadow = DeviceBuffer(owner.shape, owner.dtype, device=owner.device,
                                      neuware=program.target.neuware)
                self._owners.append(shadow)
                shadows[owner] = shadow
                self._copies.append((shadow, snapshot))
            trial_arguments = list(arguments)
            for parameter in program.interface.views:
                value = arguments[parameter.position]
                owner = value.owner if isinstance(value, DeviceView) else value
                shadow = shadows.get(owner)
                if shadow is None:
                    continue
                trial_arguments[parameter.position] = (
                    shadow.view(value.shape, value.strides,
                                offset=value.byte_offset // ELEMENT_BYTES[value.dtype])
                    if isinstance(value, DeviceView) else shadow)
            bound = program._binders[True](program, tuple(trial_arguments))
            self.native_arguments = bound.native_arguments
        except Exception:
            self.close()
            raise

    def reset(self):
        for shadow, snapshot in self._copies:
            shadow.copy_from_host(snapshot)

    def close(self):
        for shadow in self._owners:
            shadow.close()
        self._owners.clear()
        self._copies.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
