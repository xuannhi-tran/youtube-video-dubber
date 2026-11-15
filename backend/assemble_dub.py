"""Place the Vietnamese clips on the video's timeline into dub_output/dub_track.wav.

The dub follows the original loosely, the way a voice-over does:
- a block starts at its original time when it can, at least PAUSE after the
  previous one;
- when that would leave a long silence, it starts earlier (at most MAX_LEAD
  ahead of the original), and pauses between sentences inside the previous
  clip are lengthened a little to fill what remains;
- a clip is read at normal speed; once the dub is more than SOFT_LAG behind
  the original it is read slightly faster (at most SOFT_MAX_SPEED), and only
  a dub about to fall more than MAX_LAG behind is sped up further (at most
  --max-speed);
- every clip is brought to the same loudness.

Local only (ffmpeg + numpy); writes only inside dub_output/.
"""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import wave

import numpy as np

from analyze_tts_silence import analyze_wav
from workdir import WORKDIR


BASE_DIR = WORKDIR
INPUT_FILE = BASE_DIR / "block_tts_duration_results_fitted.json"
OUTPUT_DIR = BASE_DIR / "dub_output"
BLOCKS_DIR = OUTPUT_DIR / "blocks"
TRACK_FILE = OUTPUT_DIR / "dub_track.wav"
PREVIEW_FILE = OUTPUT_DIR / "preview_mix.wav"
TIMELINE_FILE = OUTPUT_DIR / "dub_timeline.json"
EXPECTED_FORMAT = (1, 2, 48000)  # channels, sample width in bytes, sample rate
TRIM_PADDING = 0.040

# Timeline (seconds)
PAUSE = 0.25          # shortest pause between two blocks
MAX_GAP = 0.8         # longer silences than this are closed up when possible
MAX_LEAD = 3.0        # how far the dub may run ahead of the original
SOFT_LAG = 1.5        # behind by more than this: catch up gently ...
SOFT_MAX_SPEED = 1.1  # ... reading at most this much faster
MAX_LAG = 3.0         # never further behind than this, unless --max-speed is not enough
# Pauses inside a clip (between sentences) that may be lengthened
INNER_PAUSE_MIN = 0.20
EXTRA_PER_PAUSE = 0.5
# Loudness
TARGET_DBFS = -20.0   # speech level (RMS of the voiced parts)
MAX_GAIN_DB = 12.0
PEAK_LIMIT = 0.89     # about -1 dBFS
SILENCE_DBFS = -40.0
WINDOW = 0.020


# ---------------------------------------------------------
# Pure helpers (covered by run_local_checks)
# ---------------------------------------------------------

def check_format(channels, width, rate, comptype="NONE"):
    if comptype != "NONE" or (channels, width, rate) != EXPECTED_FORMAT:
        raise ValueError(
            f"Expected 48 kHz mono 16-bit PCM, got {rate} Hz, {channels} ch, "
            f"{8 * width}-bit, {comptype}"
        )


def trim_range(lead_frames, active_frames, total_frames, pad_frames):
    """Active speech plus padding on each side, clamped to the existing audio."""
    start = max(0, lead_frames - pad_frames)
    end = min(total_frames, lead_frames + active_frames + pad_frames)
    return start, end


def next_start(cursor, natural, first=False):
    """Where a block starts, given where the previous one ended (cursor).

    Its original time if that leaves a pause between PAUSE and MAX_GAP; never
    sooner than PAUSE after the previous block; and when the original time
    would leave a longer silence, earlier, but at most MAX_LEAD before it.
    """
    earliest = cursor if first else cursor + PAUSE
    if natural <= cursor + MAX_GAP:
        return max(natural, earliest)
    return max(cursor + MAX_GAP, natural - MAX_LEAD)


def plan_timeline(actives, inner_pauses, starts, ends, max_speed):
    """Decide start, speed and added pause time for every block.

    actives: seconds of speech per clip; inner_pauses: lengthenable pauses per clip.
    Returns dicts with start, speed, extra (seconds added inside), end, lag.
    """
    plan, cursor = [], 0.0
    count = len(actives)
    for i in range(count):
        start = max(0.0, next_start(cursor, starts[i], first=(i == 0)))
        available = ends[i] + MAX_LAG - start
        needed = actives[i] / available if available > 0 else math.inf
        soft_available = ends[i] + SOFT_LAG - start
        gentle = actives[i] / soft_available if soft_available > 0 else math.inf
        speed = min(max_speed, max(1.0, min(SOFT_MAX_SPEED, gentle), needed))
        end = start + actives[i] / speed
        extra = 0.0
        if i + 1 < count:
            gap = next_start(end, starts[i + 1]) - end
            if gap > MAX_GAP:
                extra = min(gap - PAUSE, inner_pauses[i] * EXTRA_PER_PAUSE)
        end += extra
        plan.append({"start": start, "speed": speed, "extra": extra, "end": end,
                     "lag": start - starts[i], "capped": needed > max_speed})
        cursor = end
    return plan


def loudness_gain(rms_dbfs, peak):
    """dB to bring voiced RMS to TARGET_DBFS, limited so the peak stays below PEAK_LIMIT."""
    if rms_dbfs is None or peak <= 0:
        return 0.0
    gain = max(-MAX_GAIN_DB, min(MAX_GAIN_DB, TARGET_DBFS - rms_dbfs))
    return min(gain, 20 * math.log10(PEAK_LIMIT / peak))


def window_levels(samples, rate):
    """RMS (linear, 0..1) of each WINDOW-long piece of int16 samples."""
    size = max(1, round(rate * WINDOW))
    usable = len(samples) // size * size
    if usable == 0:
        return np.zeros(0)
    pieces = samples[:usable].astype(np.float64).reshape(-1, size) / 32768.0
    return np.sqrt((pieces ** 2).mean(axis=1))


def voiced_level(samples, rate):
    """(RMS dBFS of the voiced windows or None, peak 0..1)."""
    levels = window_levels(samples, rate)
    voiced = levels[levels > 10 ** ((SILENCE_DBFS - 5) / 20)]
    peak = float(np.abs(samples.astype(np.int32)).max()) / 32768.0 if len(samples) else 0.0
    if len(voiced) == 0:
        return None, peak
    return 20 * math.log10(float(np.sqrt((voiced ** 2).mean()))), peak


def find_pauses(samples, rate, min_seconds=INNER_PAUSE_MIN):
    """(first, last) sample index of silent runs inside the clip, edges excluded."""
    size = max(1, round(rate * WINDOW))
    silent = window_levels(samples, rate) <= 10 ** (SILENCE_DBFS / 20)
    runs, run_start = [], None
    for index, quiet in enumerate(silent):
        if quiet and run_start is None:
            run_start = index
        elif not quiet and run_start is not None:
            if run_start > 0 and (index - run_start) * WINDOW >= min_seconds:
                runs.append((run_start * size, index * size))
            run_start = None
    return runs  # a run still open at the end is the clip's tail, not an inner pause


def lengthen_pauses(samples, rate, extra_seconds):
    """Insert silence in the middle of the clip's inner pauses, spread evenly."""
    pauses = find_pauses(samples, rate)
    total = round(extra_seconds * rate)
    if not pauses or total <= 0:
        return samples
    pieces, previous = [], 0
    for number, (first, last) in enumerate(pauses):
        middle = (first + last) // 2
        share = total // len(pauses) + (1 if number < total % len(pauses) else 0)
        pieces += [samples[previous:middle], np.zeros(share, dtype=samples.dtype)]
        previous = middle
    pieces.append(samples[previous:])
    return np.concatenate(pieces)


# ---------------------------------------------------------
# WAV / ffmpeg work
# ---------------------------------------------------------

def run_ffmpeg(args):
    command = ["ffmpeg", "-y", "-loglevel", "error", *args]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"ffmpeg failed: {error.stderr.strip()}") from error


def load_blocks(path=INPUT_FILE):
    blocks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Expected a non-empty result list")
    for block in blocks:
        if not 0 <= block["start"] < block["end"]:
            raise ValueError(f"Block {block['block_id']}: invalid timestamps")
    return blocks


def read_samples(path):
    with wave.open(str(path), "rb") as wav:
        check_format(wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype())
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()


def write_samples(path, samples):
    channels, width, rate = EXPECTED_FORMAT
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(samples.astype("<i2").tobytes())


def trim_block(source, target):
    """Keep active speech plus TRIM_PADDING each side; returns the kept samples."""
    rate = EXPECTED_FORMAT[2]
    measurements = analyze_wav(source)
    samples = read_samples(source)
    lead = round(measurements["leading_silence"] * rate)
    active = round(measurements["active_speech_duration"] * rate)
    if active <= 0:
        raise ValueError(f"{source.name}: no active speech found")
    start, end = trim_range(lead, active, len(samples), round(TRIM_PADDING * rate))
    kept = samples[start:end]
    write_samples(target, kept)
    return kept


def render(source, target, speed, gain_db):
    run_ffmpeg([
        "-i", str(source), "-filter:a", f"atempo={speed:.6f},volume={gain_db:.2f}dB",
        "-ar", str(EXPECTED_FORMAT[2]), "-ac", str(EXPECTED_FORMAT[0]),
        "-c:a", "pcm_s16le", str(target),
    ])


def make_preview(original, track, volume, duration):
    run_ffmpeg([
        "-i", str(original), "-i", str(track),
        "-filter_complex",
        f"[0:a]volume={volume},apad[o];[o][1:a]amix=inputs=2:duration=shortest:normalize=0[m]",
        "-map", "[m]", "-t", f"{duration:.6f}",
        "-ar", str(EXPECTED_FORMAT[2]), "-ac", str(EXPECTED_FORMAT[0]),
        "-c:a", "pcm_s16le", str(PREVIEW_FILE),
    ])


# ---------------------------------------------------------
# Main flow
# ---------------------------------------------------------

def measure_blocks(blocks):
    """Trim every clip and measure what planning needs."""
    rate = EXPECTED_FORMAT[2]
    BLOCKS_DIR.mkdir(parents=True, exist_ok=True)
    measured = []
    for block in blocks:
        source = BASE_DIR / block["audio_file"]
        trimmed = BLOCKS_DIR / f"block_{block['block_id']:03d}_trim.wav"
        samples = trim_block(source, trimmed)
        rms, peak = voiced_level(samples, rate)
        measured.append({
            "block": block, "trimmed": trimmed,
            "active": len(samples) / rate,
            "pauses": len(find_pauses(samples, rate)),
            "gain": loudness_gain(rms, peak), "rms": rms,
        })
    return measured


def render_blocks(measured, plan):
    rate = EXPECTED_FORMAT[2]
    for item, step in zip(measured, plan):
        block_id = item["block"]["block_id"]
        final = BLOCKS_DIR / f"block_{block_id:03d}.wav"
        render(item["trimmed"], final, step["speed"], item["gain"])
        samples = read_samples(final)
        if step["extra"] > 0:
            samples = lengthen_pauses(samples, rate, step["extra"])
            write_samples(final, samples)
        item["samples"] = samples


def assemble(measured, plan):
    rate = EXPECTED_FORMAT[2]
    pieces, cursor, timeline = [], 0, []
    for item, step in zip(measured, plan):
        start = max(round(step["start"] * rate), cursor)  # never overlap the previous clip
        pieces += [np.zeros(start - cursor, dtype="<i2"), item["samples"]]
        cursor = start + len(item["samples"])
        timeline.append({
            "block_id": item["block"]["block_id"],
            "start": round(start / rate, 3), "end": round(cursor / rate, 3),
            "text": item["block"].get("translated_text", ""),
        })
    OUTPUT_DIR.mkdir(exist_ok=True)
    write_samples(TRACK_FILE, np.concatenate(pieces))
    # Where each spoken line really landed; subtitles are timed from this.
    TIMELINE_FILE.write_text(json.dumps(timeline, ensure_ascii=False, indent=1), encoding="utf-8")
    return cursor / rate


def print_report(measured, plan, track_length):
    print(f"\n{'id':>3} {'start':>8} {'orig':>8} {'lag':>6} {'speed':>6} {'+pause':>6} {'gain':>6}")
    for item, step in zip(measured, plan):
        block = item["block"]
        print(f"{block['block_id']:>3} {step['start']:>7.2f}s {block['start']:>7.2f}s "
              f"{step['lag']:>+5.2f}s {step['speed']:>5.2f}x {step['extra']:>5.2f}s {item['gain']:>+5.1f}dB")
    lags = [step["lag"] for step in plan]
    speeds = [step["speed"] for step in plan]
    gaps = [plan[i + 1]["start"] - plan[i]["end"] for i in range(len(plan) - 1)]
    last_end = measured[-1]["block"]["end"]
    print("\n--------------------------------")
    print(f"Blocks: {len(plan)} | sped up: {sum(s > 1.001 for s in speeds)} "
          f"(max {max(speeds):.2f}x) | could not catch up: {sum(step['capped'] for step in plan)}")
    print(f"Dub vs original: from {min(lags):+.2f}s (ahead) to {max(lags):+.2f}s (behind)")
    if gaps:
        print(f"Silences between blocks: longest {max(gaps):.2f}s, "
              f"over 1s: {sum(g > 1 for g in gaps)}, over 3s: {sum(g > 3 for g in gaps)}")
    print(f"Pause time added inside clips: {sum(step['extra'] for step in plan):.1f}s")
    print(f"Loudness gains: {min(i['gain'] for i in measured):+.1f} to {max(i['gain'] for i in measured):+.1f} dB "
          f"(target {TARGET_DBFS:.0f} dBFS)")
    print(f"Total track length: {track_length:.2f}s (original timeline ends at {last_end:.2f}s)")
    print(f"Saved: {TRACK_FILE.relative_to(BASE_DIR)}")
    print("--------------------------------")


def run_local_checks():
    for args in [(1, 2, 24000), (2, 2, 48000), (1, 3, 48000), (1, 2, 48000, "ULAW")]:
        try:
            check_format(*args)
        except ValueError:
            continue
        raise AssertionError(f"Accepted bad format {args}")
    check_format(1, 2, 48000)

    assert round(TRIM_PADDING * 48000) == 1920
    assert trim_range(5000, 10000, 20000, 1920) == (3080, 16920)
    assert trim_range(500, 10000, 10800, 1920) == (0, 10800)

    # Starts: on time when the pause is natural, never closer than PAUSE,
    # pulled earlier (at most MAX_LEAD) instead of a long silence.
    assert next_start(10.0, 10.5) == 10.5
    assert next_start(10.0, 10.1) == 10.0 + PAUSE
    assert next_start(10.0, 9.0) == 10.0 + PAUSE
    assert next_start(10.0, 12.0) == 10.0 + MAX_GAP
    assert next_start(10.0, 20.0) == 20.0 - MAX_LEAD
    assert next_start(0.0, 0.0, first=True) == 0.0

    # Plan: a clip that fits plays at 1.0x on time.
    plan = plan_timeline([2.0, 2.0], [0, 0], [0.0, 2.5], [2.5, 5.0], 1.3)
    assert [round(p["start"], 2) for p in plan] == [0.0, 2.5] and all(p["speed"] == 1.0 for p in plan)
    # Overrunning a little only makes the dub lag; no speed-up within MAX_LAG.
    plan = plan_timeline([4.0, 2.0, 2.0], [0, 0, 0], [0, 3, 6], [3, 6, 9], 1.3)
    assert plan[0]["speed"] == 1.0 and math.isclose(plan[1]["lag"], 1.0 + PAUSE)
    # Behind by more than SOFT_LAG: read a little faster, never above SOFT_MAX_SPEED.
    plan = plan_timeline([4.8, 2.0], [0, 0], [0, 3], [3, 6], 1.3)
    assert math.isclose(plan[0]["speed"], 4.8 / (3 + SOFT_LAG))
    plan = plan_timeline([5.8, 2.0], [0, 0], [0, 3], [3, 6], 1.3)
    assert plan[0]["speed"] == SOFT_MAX_SPEED
    # Falling too far behind is caught up with speed, at most max_speed.
    plan = plan_timeline([7.0, 2.0], [0, 0], [0, 3], [3, 6], 1.3)
    assert math.isclose(plan[0]["speed"], 7 / (3 + MAX_LAG)) and not plan[0]["capped"]
    plan = plan_timeline([20.0, 2.0], [0, 0], [0, 3], [3, 6], 1.3)
    assert plan[0]["speed"] == 1.3 and plan[0]["capped"]
    # A short clip in a long slot: next block pulled earlier, pauses lengthened.
    plan = plan_timeline([10.0, 2.0], [4, 0], [0, 20], [20, 22], 1.3)
    gap_before = (20 - MAX_LEAD) - 10.0
    assert math.isclose(plan[0]["extra"], min(gap_before - PAUSE, 4 * EXTRA_PER_PAUSE))
    assert math.isclose(plan[1]["start"], 20 - MAX_LEAD) and plan[1]["lag"] < 0
    # No pauses to lengthen: nothing added.
    assert plan_timeline([10.0, 2.0], [0, 0], [0, 20], [20, 22], 1.3)[0]["extra"] == 0
    for steps in (plan,):
        for left, right in zip(steps, steps[1:]):
            assert right["start"] >= left["end"] + PAUSE - 1e-9

    # Loudness: brought to target, never past the peak limit or MAX_GAIN_DB.
    assert math.isclose(loudness_gain(-26.0, 0.1), 6.0)
    assert math.isclose(loudness_gain(-14.0, 0.5), -6.0)
    assert loudness_gain(-40.0, 0.01) == MAX_GAIN_DB
    assert math.isclose(loudness_gain(-30.0, 0.5), 20 * math.log10(PEAK_LIMIT / 0.5))
    assert loudness_gain(None, 0.5) == 0.0

    # Pauses: found inside the clip only, lengthened in their middle.
    rate = 1000
    tone = (np.sin(np.arange(300) / 3) * 8000).astype("<i2")
    gap = np.zeros(300, dtype="<i2")
    clip = np.concatenate([gap[:100], tone, gap, tone, gap])
    pauses = find_pauses(clip, rate)
    assert len(pauses) == 1 and pauses[0][0] >= 400 and pauses[0][1] <= 700
    longer = lengthen_pauses(clip, rate, 0.5)
    assert len(longer) == len(clip) + 500
    assert np.array_equal(longer[:400], clip[:400]) and np.array_equal(longer[-600:], clip[-600:])
    assert len(lengthen_pauses(tone, rate, 0.5)) == len(tone)
    rms, peak = voiced_level(tone, rate)
    assert rms is not None and -16.5 < rms < -14 and 0.2 < peak < 0.3  # sine at 0.244: RMS -15.2 dBFS
    assert voiced_level(gap, rate)[0] is None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-speed", type=float, default=1.3,
                        help="fastest reading speed when the dub must catch up (default 1.3)")
    parser.add_argument("--original", type=Path, help="original audio for preview_mix.wav")
    parser.add_argument("--original-volume", type=float, default=0.02)
    args = parser.parse_args()
    if not (math.isfinite(args.max_speed) and 1.0 <= args.max_speed <= 2.0):
        parser.error("need 1.0 <= --max-speed <= 2.0")
    if args.original and not args.original.exists():
        parser.error(f"--original not found: {args.original}")
    return args


def main():
    run_local_checks()
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg was not found on PATH. Install it and try again.")
    blocks = load_blocks()
    measured = measure_blocks(blocks)
    plan = plan_timeline(
        [item["active"] for item in measured], [item["pauses"] for item in measured],
        [item["block"]["start"] for item in measured], [item["block"]["end"] for item in measured],
        args.max_speed,
    )
    render_blocks(measured, plan)
    track_length = assemble(measured, plan)
    print_report(measured, plan, track_length)
    if args.original:
        make_preview(args.original, TRACK_FILE, args.original_volume, track_length)
        print(f"Saved: {PREVIEW_FILE.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
