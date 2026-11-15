"""Where one dubbing job keeps its files.

By default that is this folder, exactly as before. dub.py sets the DUB_WORKDIR
environment variable so every stage reads and writes inside the folder of one
video instead of next to the scripts.
"""

import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = Path(os.environ.get("DUB_WORKDIR") or SCRIPT_DIR).resolve()
