"""Install one backend's declared Python dependencies in the active environment."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys

from .backends import backend


def setup_backend(target: str, *, torch_index_url: str | None = None) -> dict:
    """Install the selected public dependency route, without a source checkout.

    This explicitly invokes pip in this Python environment. External compilers,
    drivers and devices are not installed or checked; doctor owns that inspection.
    Package-manager output goes to stderr, preserving the CLI's JSON transport.
    """
    from .compilation import _failure

    result = {"target": target, "python": sys.executable, "tool_invoked_kernel": False,
              "commands": [], "scope": "Python dependency installation only; run intent doctor for compiler, SDK and device availability"}
    stage = "dependency_selection"
    try:
        selected = backend(target)
        if sys.platform != "linux":
            raise NotImplementedError("the public backend dependency routes currently support Linux")
        if not (3, 10) <= sys.version_info[:2] <= selected["python_max"]:
            raise NotImplementedError("the public backend dependency routes currently require Python 3.10–3.12")
        if torch_index_url is not None and (not torch_index_url.strip() or selected["torch"] is None):
            raise ValueError("torch_index_url requires a nonempty index URL and a backend that installs PyTorch")
        requirements = list(selected["requirements"])
        if selected["torch"] is not None:
            requirements.insert(0, f"torch=={selected['torch']}")
        steps = []
        if requirements:
            command = [sys.executable, "-m", "pip", "install", *requirements]
            if selected["torch"] is not None:
                command.extend(("--extra-index-url", torch_index_url or selected["torch_index"]))
            steps.append(("dependency_installation", command))
        steps.append(("dependency_consistency", [sys.executable, "-m", "pip", "check"]))
        result.update(commands=[command for _, command in steps], toolchain=selected["toolchain"])
        environment = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME"):
            environment.pop(name, None)
        environment["PYTHONNOUSERSITE"] = "1"
        for stage, command in steps:
            print("+ " + shlex.join(command), file=sys.stderr, flush=True)
            subprocess.run(command, check=True, env=environment, stdout=sys.stderr, stderr=sys.stderr)
        result["status"] = "installed"
    except Exception as error:
        _failure(result, error, stage)
    return result
