"""Checked native task entries exported by Triton–Weft JIT."""
import ctypes as ct
import json
from pathlib import Path


class Error(ct.Structure):
    _fields_ = [("status", ct.c_int), ("message", ct.c_char * 1024)]


class Entry(ct.Structure):
    _fields_ = [("handle", ct.c_void_p), ("function", ct.c_void_p)]


class CallState(ct.Structure):
    _fields_ = [("entries", ct.POINTER(Entry)), ("status", ct.c_int),
                ("message", ct.c_char * 1024)]

    def check(self):
        if self.status:
            raise RuntimeError(bytes(self.message).decode("utf-8", errors="replace"))


class NativeEntries:
    def __init__(self, artifacts):
        from weft_core import sdk_path
        self.sdk = ct.CDLL(str(Path(sdk_path()) / "lib/libweft_runtime.so"))
        ptr = ct.c_void_p
        for name, arguments, result in (
            ("context_create", [ct.POINTER(ptr), ct.POINTER(Error)], ct.c_int),
            ("context_release", [ptr], None),
            ("kernel_open", [ptr, ct.c_char_p, ct.POINTER(ptr), ct.POINTER(Error)], ct.c_int),
            ("kernel_get_entry", [ptr, ct.POINTER(ptr), ct.POINTER(Error)], ct.c_int),
            ("kernel_release", [ptr], None),
        ):
            function = getattr(self.sdk, "weft_" + name)
            function.argtypes, function.restype = arguments, result
        self.entries = (Entry * len(artifacts))()
        context, error = ptr(), Error()
        self._check(self.sdk.weft_context_create(ct.byref(context), ct.byref(error)), error)
        try:
            for index, directory in enumerate(artifacts):
                handle, function = ptr(), ptr()
                self._check(self.sdk.weft_kernel_open(context, str(directory).encode(),
                            ct.byref(handle), ct.byref(error)), error)
                self.entries[index].handle = handle
                self._check(self.sdk.weft_kernel_get_entry(handle, ct.byref(function), ct.byref(error)), error)
                self.entries[index].function = function
        except Exception:
            self.close()
            raise
        finally:
            self.sdk.weft_context_release(context)

    @staticmethod
    def _check(status, error):
        if status:
            raise RuntimeError(bytes(error.message).decode("utf-8", errors="replace"))

    def state(self):
        return CallState(self.entries, 0, b"")

    def close(self):
        for entry in self.entries:
            if entry.handle:
                self.sdk.weft_kernel_release(entry.handle)
                entry.handle = None


def bridge_source(tasks, native_abis):
    """Spell exact task signatures; compilation already chose every task and ABI."""
    sections = ['''#include <weft/runtime.h>
#include <stdatomic.h>
#include <string.h>
struct intent_entry { void *handle; void *function; };
struct intent_call { struct intent_entry *entries; atomic_int status; char message[1024]; };
''']
    for index, (task, abi) in enumerate(zip(tasks, native_abis, strict=True)):
        task.abi.verify_native(abi)
        arguments = abi["arguments"]
        types = [value["c_type"] for value in arguments]
        types += ["size_t"] * (len(abi["shape_parameters"]) +
                              sum(len(value["shape"]) for value in arguments if value["strided"]))
        signature = ", ".join(f"{kind} a{i}" for i, kind in enumerate(types))
        values = ", ".join(f"a{i}" for i in range(len(types)))
        sections.append(f'''
void {task.abi.symbol}(void *opaque{', ' if signature else ''}{signature}) {{
  struct intent_call *call = opaque;
  if (atomic_load(&call->status)) return;
  struct intent_entry entry = call->entries[{index}];
  char message[1024] = {{0}};
  int status = weft_kernel_check_worker(entry.handle, message, sizeof(message));
  if (status) {{
    int expected = 0;
    if (atomic_compare_exchange_strong(&call->status, &expected, status))
      memcpy(call->message, message, sizeof(message));
    return;
  }}
  ((void (*)({', '.join(types) or 'void'}))entry.function)({values});
}}
''')
    return "".join(sections)
