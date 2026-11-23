class Completion:
    """Completion of one submitted kernel or author program."""

    def __init__(self, wait, poll, owner):
        self._wait = wait
        self._poll = poll
        self._owner = owner

    def wait(self):
        """Wait for this invocation and propagate its execution error."""
        self._wait()
        self._owner = None

    def done(self):
        """Query completion without blocking. Use wait() to retrieve an error."""
        if self._poll():
            self._owner = None
            return True
        return False

    poll = done
