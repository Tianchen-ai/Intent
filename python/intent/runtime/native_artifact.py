"""Immutable native build outputs and process-local library loading.

Provider drivers own their commands and dependency closure. This module owns
publication, file identity checks and library handles, never tuning or devices.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import ctypes
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
from threading import Lock
import time

from ..compiler.cache import file_identity, locked_cache_entry
from ..compiler.toolchain import CompilationStageError


@dataclass(frozen=True, slots=True)
class FileDependency:
    path: Path
    identity: tuple[str, tuple[int, int, int, int, int]]

    @classmethod
    def capture(cls, path: Path) -> FileDependency:
        # Retain the lookup path: changing a symlink must invalidate the input
        # even when its previous target still exists.
        path = Path(path).absolute()
        return cls(path, file_identity(path))

    @classmethod
    def read(cls, record: object) -> FileDependency:
        if not isinstance(record, dict) or set(record) != {"path", "identity"}:
            raise ValueError("invalid native file dependency")
        path, identity = record["path"], record["identity"]
        if (not isinstance(path, str) or not Path(path).is_absolute() or
                not isinstance(identity, list) or len(identity) != 2 or
                not isinstance(identity[0], str) or
                not isinstance(identity[1], list) or len(identity[1]) != 5 or
                any(type(value) is not int for value in identity[1])):
            raise ValueError("invalid native file identity")
        return cls(Path(path), (identity[0], tuple(identity[1])))

    def record(self) -> dict:
        return {"path": str(self.path), "identity": self.identity}

    def current(self) -> bool:
        try:
            return file_identity(self.path) == self.identity
        except FileNotFoundError:
            return False


@dataclass(frozen=True, slots=True)
class NativeBuildResult:
    library: Path
    dependencies: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class NativeArtifact:
    directory: Path
    library: Path
    dependencies: tuple[FileDependency, ...]
    cache_hit: bool
    cache_reason: str | None

    @classmethod
    def read(cls, directory: Path, *, cache_hit: bool = False,
             cache_reason: str | None = None) -> NativeArtifact:
        directory = Path(directory).absolute()
        manifest = directory / "native.json"
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or set(record) != {"library", "dependencies"}:
                raise ValueError("unsupported native artifact manifest")
            if not isinstance(record["library"], str) or not isinstance(record["dependencies"], list):
                raise ValueError("invalid native artifact manifest")
            relative = Path(record["library"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("native library must belong to its immutable attempt")
            library = directory / relative
            dependencies = tuple(FileDependency.read(item) for item in record["dependencies"])
            if sum(item.path == library for item in dependencies) != 1:
                raise ValueError("native manifest must identify its library exactly once")
        except (OSError, ValueError, TypeError) as error:
            raise CompilationStageError(
                "native_artifact_lookup", f"Cannot read native artifact {manifest}: {error}",
                cache_directory=directory, artifacts={"manifest": manifest},
            ) from error
        return cls(directory, library, dependencies, cache_hit, cache_reason)

    @property
    def identity(self) -> tuple:
        return next(item.identity for item in self.dependencies if item.path == self.library)

    def current(self) -> bool:
        return all(dependency.current() for dependency in self.dependencies)

    def validate(self) -> None:
        changed = [str(item.path) for item in self.dependencies if not item.current()]
        if changed:
            raise CompilationStageError(
                "native_artifact_validation",
                "Native artifact dependencies changed; compile the original program again: " + ", ".join(changed),
                cache_directory=self.directory, artifacts={"library": self.library},
            )


def write_source(path: Path, source: str) -> None:
    """Keep stable source mtimes for the provider's own compiler cache."""
    if not path.exists() or path.read_text(encoding="utf-8") != source:
        path.write_text(source, encoding="utf-8")


def run_native_command(command: Sequence[str], directory: Path, stage: str, *,
                       environment: Mapping[str, str] | None = None,
                       source: str | None = None) -> str:
    directory = Path(directory)
    command = tuple(str(part) for part in command)
    (directory / f"{stage}.command").write_text(shlex.join(command) + "\n", encoding="utf-8")
    started = time.monotonic()
    try:
        result = subprocess.run(command, input=source, env=environment,
                                capture_output=True, text=True, check=False)
    except OSError as error:
        (directory / f"{stage}.stdout").write_text("", encoding="utf-8")
        (directory / f"{stage}.stderr").write_text(str(error) + "\n", encoding="utf-8")
        (directory / f"{stage}.status").write_text("process not started\n", encoding="utf-8")
        raise CompilationStageError(
            stage, f"Cannot start native compiler: {error}\nNative compiler artifacts: {directory}",
            cache_directory=directory,
        ) from error
    finally:
        (directory / f"{stage}.seconds").write_text(str(time.monotonic() - started) + "\n", encoding="utf-8")
    (directory / f"{stage}.stdout").write_text(result.stdout, encoding="utf-8")
    (directory / f"{stage}.stderr").write_text(result.stderr, encoding="utf-8")
    (directory / f"{stage}.status").write_text(str(result.returncode) + "\n", encoding="utf-8")
    if result.returncode:
        raise CompilationStageError(
            stage, f"Native compiler exited with code {result.returncode}:\n"
            f"{result.stderr}{result.stdout}\nNative compiler artifacts: {directory}",
            cache_directory=directory,
        )
    return result.stdout


def build_native_artifact(namespace: str, key: str,
                          build: Callable[[Path], NativeBuildResult], *,
                          reusable: bool, cache_reason: str | None = None,
                          validate_cached: Callable[[NativeArtifact], bool] | None = None) -> NativeArtifact:
    """Compile into a fresh attempt and publish only complete native outputs.

    `reusable` asserts the provider's key and dependency checks cover its native
    compiler inputs. Unknown toolchain closure permits building, not cache reuse.
    """
    identity = json.dumps((key, file_identity(Path(__file__))))
    with locked_cache_entry(namespace, identity) as cache:
        previous = cache.ready_attempt() if reusable else None
        if previous is not None:
            artifact = NativeArtifact.read(previous, cache_hit=True)
            if artifact.current() and (validate_cached is None or validate_cached(artifact)):
                return artifact
        attempt = cache.create_attempt()
        try:
            result = build(attempt)
            library = Path(result.library).absolute()
            relative = library.relative_to(attempt)
            if not library.is_file() or library.stat().st_size == 0:
                raise ValueError("native compiler did not produce a nonempty library")
            paths = dict.fromkeys((library, *(Path(path).absolute() for path in result.dependencies)))
            dependencies = tuple(FileDependency.capture(path) for path in paths)
            (attempt / "native.json").write_text(json.dumps({
                "library": str(relative), "dependencies": [item.record() for item in dependencies],
            }), encoding="utf-8")
            artifact = NativeArtifact(attempt, library, dependencies, False, cache_reason)
            artifact.validate()
            if reusable:
                cache.publish(attempt)
            return artifact
        except CompilationStageError:
            raise
        except Exception as error:
            raise CompilationStageError(
                "provider_native_compilation", f"Native artifact construction failed: {error}\n"
                f"Native compiler artifacts: {attempt}", cache_directory=attempt,
            ) from error


class _LibraryImage:
    def __init__(self, artifact: NativeArtifact):
        self.artifact = artifact
        self.handle = ctypes.CDLL(str(artifact.library))
        self.functions: dict[str, tuple[tuple, object]] = {}
        self.lock = Lock()

    def bind(self, symbol: str, argtypes: Sequence, restype):
        signature = (tuple(argtypes), restype)
        with self.lock:
            existing = self.functions.get(symbol)
            if existing is not None:
                if existing[0] != signature:
                    raise ValueError(f"Native symbol {symbol!r} was bound with a different ABI")
                return existing[1]
            function = getattr(self.handle, symbol)
            function.argtypes = signature[0]
            function.restype = restype
            self.functions[symbol] = (signature, function)
            return function


class _BoundNativeFunction:
    def __init__(self, owner: LoadedNativeLibrary, function):
        self.owner = owner
        self.function = function

    def __call__(self, *arguments):
        self.owner._require_open()
        return self.function(*arguments)


class LoadedNativeLibrary:
    """One runtime's lease on an immutable, process-lifetime library image."""

    def __init__(self, image: _LibraryImage):
        self._image: _LibraryImage | None = image

    def _require_open(self) -> _LibraryImage:
        if self._image is None:
            raise CompilationStageError("native_loading", "native library is closed")
        return self._image

    @property
    def handle(self):
        return self._require_open().handle

    def bind(self, symbol: str, argtypes: Sequence, restype):
        image = self._require_open()
        try:
            function = image.bind(symbol, argtypes, restype)
        except (AttributeError, ValueError) as error:
            raise CompilationStageError(
                "native_loading", f"Cannot bind native symbol {symbol!r}: {error}",
                cache_directory=image.artifact.directory,
                artifacts={"library": image.artifact.library},
            ) from error
        return _BoundNativeFunction(self, function)

    def close(self) -> None:
        # Other runtimes may still use this image. Never mutate or unload their
        # handle, and keep the immutable backing path for the process lifetime.
        self._image = None


_images: dict[tuple, _LibraryImage] = {}
_image_lock = Lock()


def load_native_library(artifact: NativeArtifact) -> LoadedNativeLibrary:
    artifact.validate()
    with _image_lock:
        image = _images.get(artifact.identity)
        if image is None:
            try:
                image = _LibraryImage(artifact)
            except OSError as error:
                raise CompilationStageError(
                    "native_loading", f"Cannot load native library {artifact.library}: {error}",
                    cache_directory=artifact.directory, artifacts={"library": artifact.library},
                ) from error
            artifact.validate()
            _images[artifact.identity] = image
    return LoadedNativeLibrary(image)
