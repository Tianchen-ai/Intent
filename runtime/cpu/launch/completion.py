"""Notify host callers when the entire native CPU entry has returned."""

from concurrent.futures import ThreadPoolExecutor
import os
import select
from threading import Lock


# Native libraries may retain OpenMP teams between calls. Reusing a dispatch
# thread across CPU sets would leave those team members on the old cores.
_queues = {}
_lock = Lock()


def submit(enqueue, device):
    affinity = frozenset(os.sched_getaffinity(0))
    with _lock:
        queue = _queues.get(affinity)
        if queue is None:
            queue = ThreadPoolExecutor(max_workers=1, thread_name_prefix="intent-cpu")
            _queues[affinity] = queue
    signal = os.fdopen(os.eventfd(0, os.EFD_CLOEXEC), "rb", buffering=0)
    ready = select.poll()
    ready.register(signal, select.POLLIN)

    def invoke():
        try:
            if os.sched_getaffinity(0) != affinity:
                os.sched_setaffinity(0, affinity)
            enqueue()
        finally:
            os.eventfd_write(signal.fileno(), 1)

    future = queue.submit(invoke)

    def wait():
        ready.poll()
        future.result()

    def done():
        # Keep the descriptor alive while it is registered with poll.
        return not signal.closed and bool(ready.poll(0))

    return wait, done
