"""Dub a YouTube video from a link: Chinese speech in, Vietnamese voice-over out.

    python dub.py https://www.youtube.com/watch?v=...
    python dub.py URL --max-seconds 120    # try only the first 2 minutes
    python dub.py URL --redo speech        # redo this stage and everything after it
    python dub.py URL --glossary-only      # prepare the job folder, then stop

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

import llm
from workdir import SCRIPT_DIR


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
    temporary_file.replace(path)


# ---------------------------------------------------------
# Running things
# ---------------------------------------------------------

def preflight():
    problems = []
    for package in missing_modules(lambda module: importlib.util.find_spec(module) is not None):
        problems.append(f"Python package missing: pip install {package}")
    if shutil.which("ffmpeg") is None:
        problems.append("ffmpeg was not found on PATH")
    try:
        llm.settings()
    except (RuntimeError, ValueError) as error:
        problems.append(f"{error} (backend/.env)")
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


def stage_actions(url, job, args):
    return {
        "download": lambda: download(url, job, args.max_height),
        "transcribe": lambda: run_script(job, "pipeline.py", *transcribe_args(job, args)),
        "speech": lambda: run_script(job, "speech.py", *(["--fresh"] if args.redo == "speech" else [])),
        "assemble": lambda: run_script(job, "assemble_dub.py"),
        "mux": lambda: run_script(job, "mux_dub.py", "--video", job / "source.mp4"),
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
    parser.add_argument("--glossary-only", action="store_true",
                        help="create the job folder and glossary.json, then stop")
    args = parser.parse_args()
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if args.max_height <= 0:
        parser.error("--max-height must be positive")
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
    if args.redo:
        state["done"] = forget_from(state["done"], args.redo)
    save_state(job, state)

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
    print("\n================================")
    print(f"Done: {result}")
    print("================================")


if __name__ == "__main__":
    main()
