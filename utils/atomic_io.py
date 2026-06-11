import json
import os
import tempfile
from typing import Any


def atomic_write_json(path: str, data: dict[str, Any]) -> None:
    """
    Atomically write JSON by writing to a temp file and replacing target.
    This prevents partial/corrupt files if process exits mid-write.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_",
        suffix=".json",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
