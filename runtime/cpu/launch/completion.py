"""Notify host callers when the entire native CPU entry has returned."""

from concurrent.futures import ThreadPoolExecutor
import os
import select


# Each candidate owns its dispatch queue. The native host entry still owns
# task partitioning, worker threads and joins.
_queue = ThreadPoolExecutor(max_workers=1, thread_name_prefix="intent-cpu")


def submit(enqueue, device):
    affinity = os.sched_getaffinity(0)
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

    future = _queue.submit(invoke)

    def wait():
        ready.poll()
        future.result()

    def done():
        # Keep the descriptor alive while it is registered with poll.
        return not signal.closed and bool(ready.poll(0))

    return wait, done
