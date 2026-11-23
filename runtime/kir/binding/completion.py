"""Submission and result semantics shared by installed runtime modules."""

from concurrent.futures import ThreadPoolExecutor


_retire = ThreadPoolExecutor(max_workers=1, thread_name_prefix="intent-event")


def _retire_after(event, owner):
    # Keep external tensor owners and the native module alive even when the
    # caller drops both the prepared object and its handle immediately.
    event.synchronize()


def submit_device(enqueue, device):
    import torch

    api = getattr(torch, device.type)
    with api.device(device):
        stream = api.current_stream(device)
        capturing = getattr(api, "is_current_stream_capturing", lambda: False)()
        event = api.Event(external=True) if capturing and device.type == "cuda" else api.Event()
        error = None
        try:
            enqueue()
        except Exception as cause:
            error = cause
        event.record(stream)
    # Graph capture records future work. The graph's owner keeps its bound
    # buffers alive through replays. Never synchronize during capture.
    if not capturing:
        _retire.submit(_retire_after, event, enqueue)

    def wait():
        event.synchronize()
        if error is not None:
            raise error

    return wait, event.query


class PreparedLaunch:
    previous = None

    def launch(self):
        wait, poll = self.submit(self.enqueue, self.device)
        result = self.completion_type(wait, poll, self)
        if str(self.device) == "cpu":
            self.previous = result
        return result

    def wait_previous(self):
        previous = self.previous
        if previous is not None:
            previous.wait()
            if previous is self.previous:
                self.previous = None

    def execute(self):
        """Ordinary call semantics: CPU returns after native completion."""
        self.wait_previous()
        self.enqueue()
