"""Work around ONNX Runtime refusing Hugging Face cache symlinks.

Recent ONNX Runtime versions reject a model whose external-data file (the big
.data next to the .onnx) is a symlink leading outside the model's own folder.
Recent huggingface_hub versions store every cached file as a symlink into a
shared blob store (blobs/<xx>/<hash>), so VieNeu fails to load with
"External data path escapes model directory".

This module does two things:
  1. Importing it sets HF_HUB_DISABLE_SYMLINKS=1, so files downloaded from now
     on are stored as real files. Import it BEFORE anything that imports
     huggingface_hub (including vieneu).
  2. materialize_symlinks() turns the symlinks of files that are already cached
     into real copies, so no re-download is needed.

Run it once by hand with:  python hf_cache_fix.py
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

from pathlib import Path  # noqa: E402  (after the environment variable on purpose)
import shutil  # noqa: E402
import sys  # noqa: E402


MODEL_REPOS = [
    "pnnbao-ump/VieNeu-TTS-v3-Turbo",
    "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX",
]


def hub_cache_dir():
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def materialize_symlinks(repos=MODEL_REPOS):
    """Replace every symlink in the cached snapshots of `repos` with a real copy.

    Returns how many files were converted (0 when nothing needed fixing).
    """
    cache = hub_cache_dir()
    converted = 0
    for repo in repos:
        snapshots = cache / f"models--{repo.replace('/', '--')}" / "snapshots"
        if not snapshots.is_dir():
            continue
        for path in list(snapshots.rglob("*")):
            if not path.is_symlink():
                continue
            target = path.resolve()
            if not target.is_file():
                continue  # dangling link: leave it for huggingface_hub to repair
            temporary = path.with_name(path.name + ".tmp")
            shutil.copyfile(target, temporary)
            path.unlink()
            temporary.replace(path)
            converted += 1
    return converted


if __name__ == "__main__":
    repos = sys.argv[1:] or MODEL_REPOS
    count = materialize_symlinks(repos)
    print(f"Converted {count} cached file(s) from symlinks to real files "
          f"in {hub_cache_dir()}")