"""Synchronous CLI and asynchronous MCP transport for the same tool worker."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory


def _command(result: Path) -> list[str]:
    return [sys.executable, "-m", "intent.tools.worker", "--result", str(result)]


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    return environment


def _encode_argument(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"tool request cannot encode {type(value).__name__}")


def _payload(action: str, arguments: dict) -> bytes:
    return json.dumps({"action": action, "arguments": arguments},
                      default=_encode_argument).encode("utf-8")


def _stop(process_id: int) -> None:
    try:
        # Include a compiler spawned by the worker, even if the worker itself
        # has already exited while that child still owns an output pipe.
        os.killpg(process_id, signal.SIGKILL)
    except ProcessLookupError:
        pass  # The entire request process group has already exited.


def _result(path: Path, returncode: int, stdout: str, stderr: str) -> dict:
    if returncode != 0 or not path.is_file():
        result = {
            "status": "error", "stage": "tool_worker", "tool_invoked_kernel": False,
            "diagnostic": {"type": "ToolWorkerError",
                           "message": f"Tool request process exited with status {returncode} without completing its response",
                           "stdout": stdout, "stderr": stderr},
        }
    else:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise TypeError("tool worker response must be an object")
    if stdout:
        result["program_stdout"] = result.get("program_stdout", "") + stdout
    if stderr:
        result["program_stderr"] = stderr
    return result


def request(action: str, arguments: dict) -> dict:
    payload = _payload(action, arguments)
    with TemporaryDirectory(prefix="intent-request-") as directory:
        result = Path(directory) / "result.json"
        process = subprocess.Popen(
            _command(result), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=_environment(), start_new_session=True)
        try:
            stdout, stderr = process.communicate(payload)
        except BaseException:
            _stop(process.pid)
            process.communicate()
            raise
        return _result(result, process.returncode, stdout.decode("utf-8", errors="replace"),
                       stderr.decode("utf-8", errors="replace"))


async def request_async(action: str, arguments: dict) -> dict:
    # Only the MCP transport uses AnyIO; ordinary CLI/import stays stdlib-only.
    from anyio import CancelScope
    from anyio.lowlevel import checkpoint

    payload = _payload(action, arguments)
    with TemporaryDirectory(prefix="intent-request-") as directory:
        result = Path(directory) / "result.json"
        with CancelScope(shield=True):
            process = await asyncio.create_subprocess_exec(
                *_command(result), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, env=_environment(), start_new_session=True)
        try:
            # Observe cancellation pending during creation before sending work.
            await checkpoint()
            stdout, stderr = await process.communicate(payload)
        except BaseException:
            with CancelScope(shield=True):
                _stop(process.pid)
                await process.communicate()
            raise
        return _result(result, process.returncode, stdout.decode("utf-8", errors="replace"),
                       stderr.decode("utf-8", errors="replace"))
