"""
Legacy compatibility module.

Canonical backend entrypoint is `main_gr.py`.
This module re-exports `app` so existing imports continue to work.
"""

from main_gr import app


if __name__ == "__main__":
    import os
    import subprocess
    import sys

    print("[Deprecated] `main_multi.py` is a compatibility wrapper.")
    print("[Deprecated] Use `python main_gr.py --mode full` instead.")
    script = os.path.join(os.path.dirname(__file__), "main_gr.py")
    raise SystemExit(subprocess.call([sys.executable, script, "--mode", "full"]))
