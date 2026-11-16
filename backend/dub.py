"""Dub a YouTube video from a link: Chinese speech in, Vietnamese voice-over out.

    python dub.py https://www.youtube.com/watch?v=...
    python dub.py URL --max-seconds 120    # try only the first 2 minutes
    python dub.py URL --redo speech        # redo this stage and everything after it
    python dub.py URL --glossary-only      # prepare the job folder, then stop
    python dub.py URL --original-volume 0.04   # mix the original voice lower (0 = drop it); redoes only the last stage
    python dub.py URL --subs               # burn Vietnamese subtitles into a black bar under the video
    python dub.py URL --cleanup clips      # afterwards delete the voice clips and audio (or set CLEANUP in .env)
                                           # EXPORT_DIR in .env also copies the result to that folder

Each video gets its own folder, dub_jobs_output/<video id>/, and the result is
dub_jobs_output/<video id>/dub_output/dubbed.mp4. Finished stages are skipped on
a re-run, so after an interruption (or an LLM error) just run the same command.

Translation and speech run at the same time: speech.py starts reading blocks
as soon as pipeline.py has translated them.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

import cleanup
import llm
import sync_copy
from workdir import SCRIPT_DIR, replace_file


JOBS_DIR = SCRIPT_DIR / "dub_jobs_output"  # matches "*_output/" in .gitignore
STATE_FILE = "dub_state.json"
STATE_VERSION = 2  # 1 had the stages blocks, tts, fit and tts_fitted
REQUIRED_MODULES = {
    "yt_dlp": "yt-dlp",
    "faster_whisper": "faster-whisper",
    "vieneu": "vieneu",
    "httpx": "httpx",
}

# (name, what it does) in running order.
STAGES = [
    ("download", "download the video and extract its audio"),
    ("transcribe", "Whisper transcript + LLM translation"),
    ("speech", "shorten blocks that will not fit (LLM) and read them (VieNeu)"),
    ("assemble", "place the clips on the original timeline"),
    ("mux", "put the dub track into the video"),
]
STAGE_NAMES = [name for name, _ in STAGES]


# ---------------------------------------------------------
# Pure helpers (covered by run_local_checks)
# ---------------------------------------------------------

def safe_id(video_id):
    """A folder name that cannot escape JOBS_DIR."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", video_id.strip()).strip("_")
    if not cleaned:
        raise ValueError(f"Cannot use {video_id!r} as a folder name")
    return cleaned


def forget_from(done, stage):
    """Stages still counted as done after redoing `stage` (it and later ones are dropped)."""
    cut = STAGE_NAMES.index(stage)
    return [name for name in done if name in STAGE_NAMES and STAGE_NAMES.index(name) < cut]


def done_prefix(done):
    """Finished stages, cut at the first unfinished one (a later stage built on
    an older run of an earlier stage would be stale)."""
    prefix = []
    for name in STAGE_NAMES:
        if name not in done:
            break
        prefix.append(name)
    return prefix


def reconcile_state(state, max_seconds):
    """Drop everything after the download when the time limit changed.

    The transcript depends on --max-seconds, so a different limit means redoing it.
    """
    done = state.get("done", [])
    done = done if isinstance(done, list) else []
    if state.get("version") != STATE_VERSION:
        # Older jobs: a finished one stays finished; otherwise keep what still applies.
        done = STAGE_NAMES if "mux" in done else [name for name in done if name in ("download", "transcribe")]
    done = done_prefix(done)
    if state.get("max_seconds") != max_seconds:
        done = forget_from(done, "transcribe")
    return {"version": STATE_VERSION, "done": done, "max_seconds": max_seconds}


def missing_modules(available):
    """pip names of the modules in REQUIRED_MODULES that `available(module)` rejects."""
    return [package for module, package in REQUIRED_MODULES.items() if not available(module)]


def run_local_checks():
    assert safe_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert safe_id("../a b/c") == "a_b_c"
    try:
        safe_id("///")
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted an empty folder name")

    done = ["download", "transcribe", "speech"]
    assert forget_from(done, "speech") == ["download", "transcribe"]
    assert forget_from(done, "download") == []
    assert forget_from(["bogus", "download"], "transcribe") == ["download"]
    assert done_prefix(["download", "speech"]) == ["download"]
    assert done_prefix(["mux", "download"]) == ["download"]

    v = STATE_VERSION
    state = {"version": v, "done": done, "max_seconds": None}
    assert reconcile_state(state, None) == {"version": v, "done": done, "max_seconds": None}
    assert reconcile_state(state, 120) == {"version": v, "done": ["download"], "max_seconds": 120}
    assert reconcile_state({}, None) == {"version": v, "done": [], "max_seconds": None}
    assert reconcile_state({"done": "x"}, None)["done"] == []
    # Version 1 jobs: finished stays finished, unfinished keeps download + transcribe.
    old_done = ["download", "transcribe", "blocks", "tts", "fit", "tts_fitted", "assemble", "mux"]
    assert reconcile_state({"done": old_done, "max_seconds": None}, None)["done"] == STAGE_NAMES
    old_running = ["download", "transcribe", "blocks", "tts", "fit"]
    assert reconcile_state({"done": old_running, "max_seconds": None}, None)["done"] == ["download", "transcribe"]
    assert reconcile_state({"done": ["download", "assemble"], "max_seconds": None}, None)["done"] == ["download"]

    assert mux_volume_args(argparse.Namespace(original_volume=None)) == []
    assert mux_volume_args(argparse.Namespace(original_volume=0.04)) == ["--original-volume", 0.04]
    quiet = dict(subs=False, sub_font=None, sub_font_size=None, sub_bar=None, redo=None, original_volume=None)
    assert mux_subs_args(argparse.Namespace(**quiet)) == []
    loud = argparse.Namespace(**{**quiet, "subs": True, "sub_font": "Tahoma", "sub_bar": 120})
    assert mux_subs_args(loud) == ["--subs", "--sub-font", "Tahoma", "--sub-bar", 120]
    assert stage_to_redo(argparse.Namespace(**quiet), STAGE_NAMES, True) is None
    assert stage_to_redo(loud, STAGE_NAMES, True) == "mux"
    assert stage_to_redo(loud, STAGE_NAMES, False) == "assemble"
    assert stage_to_redo(argparse.Namespace(**{**quiet, "original_volume": 0.1}), STAGE_NAMES, False) == "mux"
    assert stage_to_redo(argparse.Namespace(**{**quiet, "subs": True, "redo": "speech"}), STAGE_NAMES, True) == "speech"
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        job = Path(folder)
        every = ["download", "transcribe", "speech", "assemble"]
        assert stage_for_missing_files(job, every + ["mux"]) is None
        results = job / "block_tts_duration_results_fitted.json"
        (job / "block_tts_output").mkdir()
        (job / "block_tts_output" / "block_001.wav").write_bytes(b"x")
        results.write_text(json.dumps([{"audio_file": "block_tts_output/block_001.wav"}]))
        assert clips_present(job)
        assert stage_for_missing_files(job, every) == "assemble"      # no dub_track, clips exist
        (job / "block_tts_output" / "block_001.wav").unlink()
        assert not clips_present(job)
        assert stage_for_missing_files(job, every) == "speech"        # nothing left: read the voice again
        assert stage_for_missing_files(job, every[:3]) == "speech"    # assemble pending, no clips
        (job / "dub_output").mkdir()
        (job / "dub_output" / "dub_track.wav").write_bytes(b"x")
        assert stage_for_missing_files(job, every) is None            # track present: mux alone is enough
        assert stage_for_missing_files(job, every[:2]) is None        # speech not done: nothing to restore
    assert missing_modules(lambda module: module != "vieneu") == ["vieneu"]
    assert missing_modules(lambda module: True) == []


# ---------------------------------------------------------
# Job folder and state
# ---------------------------------------------------------

def load_state(job, max_seconds):
    path = job / STATE_FILE
    state = {}
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            print("Invalid job state file; starting this job from the beginning")
        if not isinstance(state, dict):
            state = {}
    return reconcile_state(state, max_seconds)


def save_state(job, state):
    path = job / STATE_FILE
    temporary_file = path.with_suffix(".json.tmp")
    temporary_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    replace_file(temporary_file, path)


# ---------------------------------------------------------
# Running things
# ---------------------------------------------------------

def preflight():
    problems = []
    for package in missing_modules(lambda module: importlib.util.find_spec(module) is not None):
        problems.append(f"Python package missing: pip install {package}")
    if shutil.which("ffmpeg") is None:
        problems.append("ffmpeg was not found on PATH")
    for role in ("translate", "fit"):
        try:
            llm.settings(role=role)
        except (RuntimeError, ValueError) as error:
            problems.append(f"{error} (backend/.env)")
            continue
        problem = llm.check_local_server(role)
        if problem:
            problems.append(problem)
    try:
        cleanup.read_level(os.getenv("CLEANUP"))
    except ValueError as error:
        problems.append(f"{error} (backend/.env)")
    problem = sync_copy.settings_problem()
    if problem:
        problems.append(problem)
    try:
        import vieneu_engine  # reads TTS_DEVICE; does not load the voice model
        problem = vieneu_engine.device_problem()
    except ValueError as error:
        problem = f"{error} (backend/.env)"
    if problem:
        problems.append(problem)
    try:
        fallback = llm.fallback_settings("fit")
    except (RuntimeError, ValueError) as error:
        problems.append(f"{error} (FIT_FALLBACK_*, backend/.env)")
        fallback = None
    if fallback:
        problem = llm.check_local_server("fit", fallback=True)
        if problem:  # only a warning: the main LLM may not run out
            print(f"Warning: the shortening fallback is not ready. {problem}")
        else:
            print(f"Shortening falls back to {fallback[0]}:{fallback[1]} if the main LLM runs out of quota.")
    if problems:
        raise SystemExit("Cannot start:\n- " + "\n- ".join(problems))


def script_command(job, script, *script_args):
    environment = {
        **os.environ,
        "DUB_WORKDIR": str(job),
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
    }
    command = [sys.executable, str(SCRIPT_DIR / script), *map(str, script_args)]
    print(f"\n>>> {script} {' '.join(map(str, script_args))}".rstrip(), flush=True)
    return command, environment


def failed(script, code):
    return SystemExit(
        f"\n{script} failed (exit code {code}). "
        "Fix the problem and run the same command again: finished work is kept."
    )


def run_script(job, script, *script_args):
    """Run one pipeline script with its files redirected into the job folder."""
    command, environment = script_command(job, script, *script_args)
    result = subprocess.run(command, cwd=job, env=environment)
    if result.returncode != 0:
        raise failed(script, result.returncode)


def start_script(job, label, script, *script_args):
    """Start a script whose output lines are printed with `label` in front."""
    command, environment = script_command(job, script, *script_args)
    process = subprocess.Popen(
        command, cwd=job, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )

    def relay():
        for line in process.stdout:
            print(f"{label} | {line.rstrip()}", flush=True)

    process.relay = threading.Thread(target=relay, daemon=True)
    process.relay.start()
    return process


def run_side_by_side(job, transcribe_args, on_translated):
    """pipeline.py and speech.py --follow at the same time.

    `on_translated` runs as soon as the translation succeeds; if either script
    fails, the other one is stopped.
    """
    # A stale file from an earlier run would look like a finished translation.
    (job / "translated_partial.json").unlink(missing_ok=True)
    translate = start_script(job, "translate", "pipeline.py", *transcribe_args)
    speech = start_script(job, "speech   ", "speech.py", "--follow")
    running = {"pipeline.py": translate, "speech.py": speech}
    try:
        while running:
            for script, process in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                process.relay.join()
                del running[script]
                if code != 0:
                    raise failed(script, code)
                if process is translate:
                    on_translated()
            time.sleep(0.5)
    finally:
        for process in running.values():
            process.terminate()
        for process in running.values():
            process.wait()
            process.relay.join()


def fetch_video_id(url):
    command = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--print", "id", url]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not lines:
        raise SystemExit(f"yt-dlp could not read that link:\n{result.stderr.strip()}")
    return safe_id(lines[-1])


def download(url, job, max_height):
    video = job / "source.mp4"
    height = f"[height<={max_height}]"
    formats = (
        f"bv*{height}[vcodec^=avc1]+ba[ext=m4a]/"
        f"bv*{height}[ext=mp4]+ba[ext=m4a]/"
        f"b{height}/b"
    )
    command = [
        sys.executable, "-m", "yt_dlp", "--no-playlist",
        "-f", formats, "--merge-output-format", "mp4",
        "-o", str(job / "source.%(ext)s"), url,
    ]
    print("\n>>> yt-dlp", flush=True)
    if subprocess.run(command).returncode != 0 or not video.exists():
        raise SystemExit("Download failed. Update yt-dlp (pip install -U yt-dlp) and try again.")

    audio = job / "source.wav"
    print("\n>>> ffmpeg (extract audio)", flush=True)
    extract = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", str(audio),
    ]
    if subprocess.run(extract).returncode != 0 or not audio.exists():
        raise SystemExit("Could not extract the audio from the downloaded video.")


def transcribe_args(job, args):
    result = ["--audio", job / "source.wav"]
    if args.max_seconds is not None:
        result += ["--max-seconds", args.max_seconds]
    return result


def mux_volume_args(args):
    """--original-volume for mux_dub.py; without the option its own default applies."""
    if args.original_volume is None:
        return []
    return ["--original-volume", args.original_volume]


def mux_subs_args(args):
    """Subtitle options for mux_dub.py; without --subs nothing is passed (video stays as is)."""
    if not args.subs:
        return []
    result = ["--subs"]
    if args.sub_font:
        result += ["--sub-font", args.sub_font]
    if args.sub_font_size is not None:
        result += ["--sub-font-size", args.sub_font_size]
    if args.sub_bar is not None:
        result += ["--sub-bar", args.sub_bar]
    return result


def stage_to_redo(args, done, has_timeline):
    """Which stage to rewind to, or None. --redo wins; --subs/--original-volume only redo mux."""
    if args.redo:
        return args.redo
    if args.subs and not has_timeline:
        return "assemble"  # jobs from before subtitles existed have no dub_timeline.json
    if args.subs or args.original_volume is not None:
        return "mux"  # only the last stage depends on them
    return None


def clips_present(job):
    """True when every voice clip listed in the results file is still on disk."""
    rows = read_json_file(job / "block_tts_duration_results_fitted.json")
    if not isinstance(rows, list) or not rows:
        return False
    return all(isinstance(row, dict) and (job / str(row.get("audio_file", ""))).is_file() for row in rows)


def read_json_file(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def stage_for_missing_files(job, done):
    """Stage to rewind to when a later stage must run but CLEANUP removed what it reads.

    Only the voice (speech stage) is read again, from the saved translation and shortened
    text; nothing is translated or shortened again. None when nothing is missing.
    """
    if "mux" in done:
        return None
    clips = clips_present(job)
    if "assemble" in done and not (job / "dub_output" / "dub_track.wav").is_file():
        return "assemble" if clips else "speech"
    if "assemble" not in done and "speech" in done and not clips:
        return "speech"
    return None


def stage_actions(url, job, args):
    return {
        "download": lambda: download(url, job, args.max_height),
        "transcribe": lambda: run_script(job, "pipeline.py", *transcribe_args(job, args)),
        "speech": lambda: run_script(job, "speech.py", *(["--fresh"] if args.redo == "speech" else [])),
        "assemble": lambda: run_script(job, "assemble_dub.py"),
        "mux": lambda: run_script(job, "mux_dub.py", "--video", job / "source.mp4",
                                       *mux_volume_args(args), *mux_subs_args(args)),
    }


# ---------------------------------------------------------
# Main flow
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", help="YouTube link")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="only dub the first N seconds (default: the whole video)")
    parser.add_argument("--max-height", type=int, default=720,
                        help="largest video height to download (default 720)")
    parser.add_argument("--redo", choices=STAGE_NAMES,
                        help="run this stage again, and every stage after it")
    parser.add_argument("--original-volume", type=float, default=None,
                        help="volume of the original audio under the dub, 0 to 1 (default: mux_dub.py's 0.02, "
                             "0 drops it); changing it on a finished job only redoes the mux stage")
    parser.add_argument("--subs", action="store_true",
                        help="burn Vietnamese subtitles into a black bar added below the video (the original "
                             "Chinese text stays visible); on a finished job only the last stage is redone")
    parser.add_argument("--sub-font", default=None,
                        help="font with Vietnamese diacritics (default: SUB_FONT in .env, else Arial)")
    parser.add_argument("--sub-font-size", type=int, default=None,
                        help="subtitle size in pixels of the video (default: automatic, or SUB_FONT_SIZE)")
    parser.add_argument("--sub-bar", type=int, default=None,
                        help="height of the black bar in pixels (default: automatic, or SUB_BAR)")
    parser.add_argument("--cleanup", choices=cleanup.LEVELS, default=None,
                        help="after the video is made: clips = delete the voice clips and audio files, all = also "
                             "the source video and dub track (default: CLEANUP in .env, else none)")
    parser.add_argument("--glossary-only", action="store_true",
                        help="create the job folder and glossary.json, then stop")
    args = parser.parse_args()
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if args.max_height <= 0:
        parser.error("--max-height must be positive")
    if args.original_volume is not None and not 0 <= args.original_volume <= 1:
        parser.error("--original-volume must be between 0 and 1")
    for name in ("sub_font_size", "sub_bar"):
        if getattr(args, name) is not None and getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be 0 or more")
    return args


def main():
    run_local_checks()
    args = parse_args()
    preflight()

    video_id = fetch_video_id(args.url)
    job = JOBS_DIR / video_id
    job.mkdir(parents=True, exist_ok=True)
    glossary = job / "glossary.json"
    if not glossary.exists():
        # Empty on purpose: the default list holds terms of one specific video.
        glossary.write_text("[]\n", encoding="utf-8")
    print(f"Video {video_id}\nJob folder: {job}")
    print(f"Glossary (edit before the transcribe stage to pin names): {glossary}")
    if args.glossary_only:
        return

    state = load_state(job, args.max_seconds)
    redo = stage_to_redo(args, state["done"], (job / "dub_output" / "dub_timeline.json").is_file())
    if redo:
        state["done"] = forget_from(state["done"], redo)
    missing = stage_for_missing_files(job, state["done"])
    if missing:
        print(f"Files from an earlier CLEANUP are missing: redoing the {missing} stage "
              "(no translation or shortening is repeated)")
        state["done"] = forget_from(state["done"], missing)
    save_state(job, state)
    if "mux" not in state["done"] and not (job / "source.mp4").is_file() and (job / "transcript_raw.json").is_file():
        print("The source video was removed by CLEANUP: downloading it again")
        download(args.url, job, args.max_height)

    def finish(name):
        state["done"].append(name)
        save_state(job, state)

    actions = stage_actions(args.url, job, args)
    ran_alongside = set()
    for number, (name, description) in enumerate(STAGES, start=1):
        if name in ran_alongside:
            continue
        label = f"[{number}/{len(STAGES)}] {name}: {description}"
        if name in state["done"]:
            print(f"\n{label} -- already done, skipping")
            continue
        print(f"\n{label}")
        if name == "transcribe":
            # Speech reads blocks while the rest of the video is still being translated.
            print(f"[{number + 1}/{len(STAGES)}] speech: {STAGES[number][1]} -- at the same time")
            run_side_by_side(job, transcribe_args(job, args), lambda: finish("transcribe"))
            finish("speech")
            ran_alongside.add("speech")
            continue
        actions[name]()
        finish(name)

    result = job / "dub_output" / "dubbed.mp4"
    copy = sync_copy.export(job)  # before the cleanup; it never removes dubbed.mp4 anyway
    if copy:
        print(f"\nCopied to {copy} (EXPORT_DIR)")
    level = args.cleanup or cleanup.read_level(os.getenv("CLEANUP"))
    freed = cleanup.clean_job(job, level)
    if freed:
        print(f"\nCleanup ({level}): freed {cleanup.megabytes(freed)}")
    print("\n================================")
    print(f"Done: {result}")
    print("================================")


if __name__ == "__main__":
    main()
