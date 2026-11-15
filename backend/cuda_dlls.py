"""Make pip's CUDA libraries visible to Windows before faster-whisper loads.

pip's nvidia-cublas-cu12 / nvidia-cudnn-cu12 packages keep their DLLs in
site-packages/nvidia/<library>/bin, a folder Windows does not search, which ends in
"Library cublas64_12.dll is not found or cannot be loaded". Importing this module
adds those folders to the DLL search path. It does nothing on other systems.
"""

import importlib.util
import os
from pathlib import Path


def nvidia_bin_dirs(roots):
    """Every <root>/<library>/bin folder that holds DLLs."""
    return [
        bin_dir
        for root in roots
        for bin_dir in sorted(Path(root).glob("*/bin"))
        if any(bin_dir.glob("*.dll"))
    ]


def add_nvidia_dll_dirs():
    if os.name != "nt":
        return []
    spec = importlib.util.find_spec("nvidia")
    folders = nvidia_bin_dirs(spec.submodule_search_locations or []) if spec else []
    for folder in folders:
        os.add_dll_directory(str(folder))  # for Python's own extension loading
    if folders:
        # CTranslate2 loads cuBLAS/cuDNN itself, and that lookup only follows PATH.
        paths = [str(folder) for folder in folders]
        os.environ["PATH"] = os.pathsep.join([*paths, os.environ.get("PATH", "")])
    return folders


add_nvidia_dll_dirs()
