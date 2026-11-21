"""Publish an immutable native artifact directory only after a successful build."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def build_directory(exit, ingredients, build):
    root = (
        Path(os.environ.get("INTENT_CACHE_DIR", Path.home() / ".cache" / "intentdsl"))
        / exit
    )
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(ingredients, sort_keys=True).encode()).hexdigest()
    path = root / key
    with (root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_dir():
            return path, True
        stage = Path(tempfile.mkdtemp(dir=root))
        try:
            build(stage)
            os.replace(stage, path)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return path, False
