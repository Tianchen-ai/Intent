from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import sqlite3
from uuid import UUID, uuid4


def cache_root() -> Path:
    configured = os.environ.get("INTENT_CACHE_DIR")
    if configured is not None:
        if not configured:
            raise ValueError("INTENT_CACHE_DIR must name a directory")
        return Path(configured).expanduser().resolve()
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return (base / "intentdsl").expanduser().resolve()


def file_identity(path: str | Path) -> tuple[str, tuple[int, int, int, int, int]]:
    resolved = Path(path).expanduser().resolve()
    status = resolved.stat()
    return str(resolved), (status.st_dev, status.st_ino, status.st_size,
                           status.st_mtime_ns, status.st_ctime_ns)


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """An entry used while holding its lock; consumers validate their outputs.

    Native attempts have permanent, distinct paths. Publishing only replaces
    the ready pointer: a published attempt must never be rewritten or removed
    by a caller, including when a later attempt is needed.
    """

    directory: Path
    key: str

    def create_attempt(self) -> Path:
        attempts = self.directory / "attempts"
        attempts.mkdir(exist_ok=True)
        attempt = attempts / uuid4().hex
        attempt.mkdir()
        return attempt

    def _attempt(self, identity: str) -> Path:
        try:
            valid = UUID(identity).hex == identity
        except (ValueError, AttributeError) as error:
            raise ValueError(f"invalid cache attempt identity in {self.directory}") from error
        if not valid:
            raise ValueError(f"invalid cache attempt identity in {self.directory}: {identity!r}")
        attempt = self.directory / "attempts" / identity
        if attempt.is_symlink() or not attempt.is_dir():
            raise ValueError(f"cache attempt is not an existing directory: {attempt}")
        return attempt

    def ready_attempt(self) -> Path | None:
        ready = self.directory / "ready"
        try:
            text = ready.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            record = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid cache ready record: {ready}") from error
        if not isinstance(record, dict) or set(record) != {"attempt"} or not isinstance(record["attempt"], str):
            raise ValueError(f"invalid cache ready record: {ready}")
        return self._attempt(record["attempt"])

    def publish(self, attempt: Path) -> None:
        attempt = Path(attempt)
        if attempt != self._attempt(attempt.name):
            raise ValueError(f"attempt does not belong to cache entry {self.directory}: {attempt}")
        temporary = self.directory / f".ready-{uuid4().hex}"
        try:
            with temporary.open("x", encoding="utf-8") as output:
                json.dump({"attempt": attempt.name}, output)
                output.write("\n")
            temporary.replace(self.directory / "ready")
        finally:
            temporary.unlink(missing_ok=True)

    def invalidate_ready(self) -> None:
        (self.directory / "ready").unlink(missing_ok=True)


@contextmanager
def locked_cache_entry(namespace: str, fullkey: str):
    if not namespace or Path(namespace).name != namespace or namespace in {".", ".."}:
        raise ValueError("cache namespace must be one directory name")
    if not isinstance(fullkey, str):
        raise TypeError("cache key must contain the complete serialized inputs")
    root = cache_root() / namespace
    root.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(root / "entries.sqlite3", timeout=60)) as database:
        with database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS entries "
                "(directory TEXT PRIMARY KEY, compile_inputs TEXT NOT NULL UNIQUE)"
            )
            database.execute("INSERT OR IGNORE INTO entries (directory, compile_inputs) VALUES (?, ?)",
                             (uuid4().hex, fullkey))
            identity, = database.execute(
                "SELECT directory FROM entries WHERE compile_inputs = ?", (fullkey,)
            ).fetchone()
    directory = root / identity
    directory.mkdir(exist_ok=True)
    # The database transaction only resolves a key. Independent entries may
    # compile concurrently; the stable entry lock covers validation/publication.
    with (directory / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield CacheEntry(directory, fullkey)


def _compilation_key(executable: Path, module_text: str,
                     options: tuple[str, ...], resources: tuple[tuple[str, str], ...]) -> str:
    resolved, identity = file_identity(executable)
    # Compare the complete compile inputs. File identity invalidates entries
    # when the compiler is rebuilt or replaced, without checksumming artifacts.
    return json.dumps((resolved, identity,
                       os.environ.get("LD_LIBRARY_PATH", ""),
                       options, resources, module_text), ensure_ascii=False)


@contextmanager
def compilation_directory(executable: Path, module_text: str,
                          options: tuple[str, ...], resources: tuple[tuple[str, str], ...]):
    key = _compilation_key(executable, module_text, options, resources)
    with locked_cache_entry("compiler", key) as entry:
        yield entry.directory, key
