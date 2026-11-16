"""Copy finished videos to another folder, e.g. one that Google Drive / OneDrive uploads.

    EXPORT_DIR=D:\\Google Drive\\Dubbed   # in backend/.env: every finished dubbed.mp4 is copied
                                         # there as <video id>.mp4 when its job ends
    python sync_copy.py                  # copy the finished videos already on disk (skips copies
                                         # that are already there)

With a synced folder the video is in the cloud without another step, so the job folder on this
computer can be cleaned (CLEANUP) or deleted with the "Xóa" button; the copy stays.
"""

import os
from pathlib import Path
import shutil

from dotenv import load_dotenv

from workdir import SCRIPT_DIR, replace_file


load_dotenv(SCRIPT_DIR / ".env")

JOBS_DIR = SCRIPT_DIR / "dub_jobs_output"
RESULT = Path("dub_output") / "dubbed.mp4"


def export_dir():
    """EXPORT_DIR as a Path, or None when it is not set."""
    text = (os.getenv("EXPORT_DIR") or "").strip().strip('"')
    return Path(text).expanduser() if text else None


def settings_problem(folder=None):
    """None when EXPORT_DIR is unset or an existing folder; otherwise what to fix."""
    folder = folder if folder is not None else export_dir()
    if folder is not None and not folder.is_dir():
        return f"EXPORT_DIR {folder} is not a folder (create it, or remove EXPORT_DIR from backend/.env)"
    return None


def copy_path(video_id, folder=None):
    """Where the video of `video_id` is (or would be) copied; None without EXPORT_DIR."""
    folder = folder if folder is not None else export_dir()
    return folder / f"{video_id}.mp4" if folder is not None else None


def existing_copy(video_id, folder=None):
    """The copy of `video_id` as a string when it exists, else ''."""
    path = copy_path(video_id, folder)
    return str(path) if path is not None and path.is_file() else ""


def export(job, folder=None):
    """Copy the job's dubbed.mp4 into EXPORT_DIR; return the copy, or None when it is not set."""
    target = copy_path(Path(job).name, folder)
    if target is None:
        return None
    # Written under another name first, so a syncing app never uploads half a video.
    temporary = target.with_name(target.name + ".part")
    shutil.copyfile(Path(job) / RESULT, temporary)
    replace_file(temporary, target)
    return target


def run_local_checks():
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        job = root / "jobs" / "abc"
        (job / "dub_output").mkdir(parents=True)
        (job / RESULT).write_bytes(b"video")
        out = root / "sync"
        assert settings_problem(out) is not None  # not created yet
        out.mkdir()
        assert settings_problem(out) is None
        assert existing_copy("abc", out) == ""
        copy = export(job, out)
        assert copy == out / "abc.mp4" and copy.read_bytes() == b"video"
        assert not list(out.glob("*.part")) and existing_copy("abc", out) == str(copy)
        (job / RESULT).write_bytes(b"newer")  # a redone video replaces the old copy
        assert export(job, out).read_bytes() == b"newer"


def main():
    run_local_checks()
    folder = export_dir()
    if folder is None:
        print("EXPORT_DIR is not set in backend/.env: nothing to copy.")
        return
    problem = settings_problem(folder)
    if problem:
        raise SystemExit(problem)
    jobs = sorted(job for job in JOBS_DIR.iterdir() if (job / RESULT).is_file()) if JOBS_DIR.is_dir() else []
    copied = 0
    for job in jobs:
        if existing_copy(job.name, folder):
            print(f"  {job.name}: already copied")
            continue
        print(f"  {job.name}: copied to {export(job, folder)}")
        copied += 1
    print(f"Copied {copied} video(s) to {folder}.")


if __name__ == "__main__":
    main()
