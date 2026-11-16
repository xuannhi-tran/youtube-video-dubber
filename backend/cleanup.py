"""Free disk space after a job: remove the big intermediate files, or a whole job.

    CLEANUP=clips   # in .env: after the video is made, drop the voice clips and the audio of the source
    CLEANUP=all     # ...and also the source video and the dub track (only dubbed.mp4 + small files stay)
    python cleanup.py JOB_ID [--level clips|all]   # clean one finished job by hand (--all-jobs: every job)

What stays with "clips": dubbed.mp4, the translation and shortened text, the source video and the
dub track, so --original-volume and --subs still re-mux in seconds. With "all" dub.py gets
what it needs again when you ask for something (see dub.py `restore`): it downloads the video
again and reads the voice again, but never translates again.
"""

import argparse
from pathlib import Path
import shutil

from workdir import SCRIPT_DIR

JOBS_DIR = SCRIPT_DIR / "dub_jobs_output"
LEVELS = ("none", "clips", "all")


def read_level(value):
    level = (value or "").strip().lower() or "none"
    if level not in LEVELS:
        raise ValueError(f"CLEANUP must be one of {', '.join(LEVELS)}, got {value!r}")
    return level


def files_to_remove(job, level):
    """Files `level` removes from the job folder (only those that exist)."""
    job = Path(job)
    found = []
    if level in ("clips", "all"):
        found += [job / "source.wav", job / "dub_output" / "preview_mix.wav"]
        found += sorted((job / "block_tts_output").glob("*.wav"))
        found += sorted((job / "dub_output" / "blocks").glob("*.wav"))
    if level == "all":
        found += [job / "source.mp4", job / "dub_output" / "dub_track.wav"]
    return [path for path in found if path.is_file()]


def clean_job(job, level):
    """Remove the files; returns the bytes freed. The final video is never touched."""
    if level == "none":
        return 0
    if not (Path(job) / "dub_output" / "dubbed.mp4").is_file():
        return 0  # nothing was finished: keep every file a resume needs
    freed = 0
    for path in files_to_remove(job, level):
        size = path.stat().st_size
        path.unlink()
        freed += size
    return freed


def folder_size(folder):
    return sum(path.stat().st_size for path in Path(folder).rglob("*") if path.is_file())


def job_folder(job_id, jobs_dir=JOBS_DIR):
    """The job folder for a video id, only if it really is a direct child of the jobs folder."""
    if not job_id or job_id in (".", "..") or "/" in job_id or "\\" in job_id:
        raise ValueError(f"Not a job id: {job_id!r}")
    folder = Path(jobs_dir) / job_id
    if folder.resolve().parent != Path(jobs_dir).resolve() or not folder.is_dir():
        raise ValueError(f"No such job: {job_id}")
    return folder


def delete_job(job_id, jobs_dir=JOBS_DIR):
    """Delete a whole job folder (video included); returns the bytes freed."""
    folder = job_folder(job_id, jobs_dir)
    size = folder_size(folder)
    shutil.rmtree(folder)
    return size


def megabytes(count):
    return f"{count / 1_000_000:.0f} MB"


def run_local_checks():
    import tempfile

    assert read_level(None) == "none" and read_level(" Clips ") == "clips" and read_level("all") == "all"
    try:
        read_level("everything")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown level must be rejected")

    with tempfile.TemporaryDirectory() as folder:
        jobs = Path(folder) / "jobs"

        def make_job(name, finished=True):
            job = jobs / name
            for relative in ("source.mp4", "source.wav", "translate_cache.json", "dub_output/dubbed.mp4",
                             "dub_output/dub_track.wav", "dub_output/dub_timeline.json",
                             "dub_output/preview_mix.wav", "dub_output/blocks/block_001.wav",
                             "dub_output/blocks/block_001_trim.wav", "block_tts_output/block_001.wav",
                             "block_tts_output/cache_metadata.json"):
                if relative == "dub_output/dubbed.mp4" and not finished:
                    continue
                path = job / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x" * 100)
            return job

        job = make_job("abc")
        assert files_to_remove(job, "none") == []
        names = {path.relative_to(job).as_posix() for path in files_to_remove(job, "clips")}
        assert names == {"source.wav", "dub_output/preview_mix.wav", "dub_output/blocks/block_001.wav",
                         "dub_output/blocks/block_001_trim.wav", "block_tts_output/block_001.wav"}
        assert clean_job(job, "clips") == 500
        assert (job / "source.mp4").exists() and (job / "dub_output" / "dub_track.wav").exists()
        assert (job / "block_tts_output" / "cache_metadata.json").exists() and not (job / "source.wav").exists()
        assert clean_job(job, "all") == 200  # source.mp4 + dub_track.wav
        kept = sorted(path.relative_to(job).as_posix() for path in job.rglob("*") if path.is_file())
        assert kept == ["block_tts_output/cache_metadata.json", "dub_output/dub_timeline.json",
                        "dub_output/dubbed.mp4", "translate_cache.json"], kept
        assert clean_job(job, "all") == 0  # nothing left to free

        unfinished = make_job("half", finished=False)  # no dubbed.mp4: a resume still needs the files
        assert clean_job(unfinished, "all") == 0 and (unfinished / "source.mp4").exists()

        for bad in ("", "..", "../abc", "a/b", "missing", "a\\b"):
            try:
                job_folder(bad, jobs)
            except ValueError:
                continue
            raise AssertionError(f"accepted job id {bad!r}")
        assert job_folder("abc", jobs) == jobs / "abc"
        assert delete_job("half", jobs) > 0
        assert not (jobs / "half").exists() and (jobs / "abc").exists()


def main():
    run_local_checks()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", nargs="?", help="video id (folder name under dub_jobs_output)")
    parser.add_argument("--level", choices=LEVELS[1:], default="clips")
    parser.add_argument("--all-jobs", action="store_true", help="clean every finished job")
    args = parser.parse_args()
    if args.all_jobs:
        jobs = sorted(path for path in JOBS_DIR.iterdir() if path.is_dir()) if JOBS_DIR.is_dir() else []
    elif args.job:
        jobs = [job_folder(args.job)]
    else:
        parser.error("give a JOB id or --all-jobs")
    total = sum(clean_job(job, args.level) for job in jobs)
    print(f"Cleaned {len(jobs)} job(s) with level {args.level}: freed {megabytes(total)}")


if __name__ == "__main__":
    main()
