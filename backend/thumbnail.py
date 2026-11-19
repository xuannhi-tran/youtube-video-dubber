"""A vertical (9:16) thumbnail picked from the video itself.

    python thumbnail.py VIDEO_ID                 # make dub_output/thumbnail.jpg for a finished job
    python thumbnail.py VIDEO_ID --force         # make it again (it is kept otherwise)
    python thumbnail.py VIDEO_ID --title "Bốn kiểu tâm lý tiền bạc"   # with a big title on top

dub.py makes one after every finished video (THUMBNAIL=0 in backend/.env or --no-thumbnail turns it
off) and EXPORT_DIR gets a copy next to the video as <video id>.jpg.

How the frame is chosen: about a dozen frames are taken between 8% and 92% of the video, and the one
that is sharpest and neither dark nor washed out wins (black fades, blur and flat frames score low).
A horizontal video becomes 1080x1920 with the frame in the middle over a blurred, enlarged copy of itself;
a vertical video is cropped to fill the picture.

Local only: ffmpeg/ffprobe and numpy, no API.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import mux_dub
import subtitles
from workdir import SCRIPT_DIR


WIDTH, HEIGHT = 1080, 1920
OUTPUT_NAME = Path("dub_output") / "thumbnail.jpg"
JOBS_DIR = SCRIPT_DIR / "dub_jobs_output"
CANDIDATES = 12
FIRST, LAST = 0.08, 0.92       # share of the duration the candidates are taken from
SAMPLE = 256                    # candidates are scored on a SAMPLE x SAMPLE gray picture
TOO_DARK, TOO_BRIGHT, FLAT = 35.0, 225.0, 14.0   # mean brightness limits and the least contrast (std) of a usable frame
VERTICAL_FROM = 1.5             # height / width from which a video is cropped instead of padded
BLUR = "boxblur=luma_radius=40:luma_power=3"
TITLE_FILE = "thumbnail_title.ass"
TITLE_FONT_SIZE = 120
TITLE_LINE_CHARS = 16


def enabled():
    """THUMBNAIL in .env: on unless it is 0 / no / false / off."""
    return (os.getenv("THUMBNAIL") or "1").strip().lower() not in ("0", "no", "false", "off")


def candidate_times(duration, count=CANDIDATES):
    """Evenly spaced moments (seconds) between FIRST and LAST of the duration."""
    if not duration or duration <= 0:
        return [0.0]
    if duration < 3 or count < 2:
        return [duration / 2]
    start, end = duration * FIRST, duration * LAST
    return [round(start + (end - start) * i / (count - 1), 3) for i in range(count)]


def sharpness(gray):
    """Variance of the Laplacian: high for a picture with fine, crisp detail, low for blur."""
    g = gray.astype(np.float64)
    laplacian = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(laplacian.var())


def frame_score(gray):
    """How good a thumbnail this gray picture would be (0 = unusable): sharp, neither dark nor washed out."""
    mean, spread = float(gray.mean()), float(gray.std())
    if mean < TOO_DARK or mean > TOO_BRIGHT or spread < FLAT:
        return 0.0
    # a middle brightness is best; a frame at the edge of the allowed range counts for little
    middle = 1.0 - abs(mean - 128.0) / 128.0
    return sharpness(gray) * (0.3 + 0.7 * middle)


def best_time(scored):
    """The moment with the highest score from [(seconds, score)]; the middle one when none is usable."""
    if not scored:
        return 0.0
    best = max(scored, key=lambda item: item[1])
    if best[1] <= 0:
        return scored[len(scored) // 2][0]
    return best[0]


def grab_gray(video, seconds, run=subprocess.run):
    """One frame at `seconds` as a SAMPLE x SAMPLE gray numpy array; None when ffmpeg gives nothing."""
    command = ["ffmpeg", "-loglevel", "error", "-ss", f"{seconds:.3f}", "-i", str(video), "-frames:v", "1",
               "-vf", f"scale={SAMPLE}:{SAMPLE},format=gray", "-f", "rawvideo", "-"]
    result = run(command, capture_output=True)
    if result.returncode != 0 or len(result.stdout) != SAMPLE * SAMPLE:
        return None
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(SAMPLE, SAMPLE)


def choose_time(video, duration, grab=grab_gray, times=None):
    """(seconds of the best frame, how many candidates could be read)."""
    scored = []
    for seconds in (times if times is not None else candidate_times(duration)):
        gray = grab(video, seconds)
        if gray is not None:
            scored.append((seconds, frame_score(gray)))
    return best_time(scored), len(scored)


def is_vertical(width, height):
    return bool(width and height) and height / width >= VERTICAL_FROM


def build_filter(width, height, title=False):
    """The -filter_complex that turns one frame into the WIDTH x HEIGHT picture."""
    if is_vertical(width, height):
        graph = (f"[0:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
                 f"crop={WIDTH}:{HEIGHT},setsar=1")
    else:
        graph = (f"[0:v]split[a][b];"
                 f"[a]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},{BLUR}[bg];"
                 f"[b]scale={WIDTH}:-2[fg];"
                 f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")
    return graph + (f",ass={TITLE_FILE}" if title else "") + "[out]"


def title_ass(title, font=subtitles.DEFAULT_FONT):
    """An ASS file with `title` big and centred in the upper part of the picture (wrapped to a few lines)."""
    lines = subtitles.wrap(" ".join(title.split()), TITLE_LINE_CHARS)
    text = "\\N".join(subtitles.ass_escape(line) for line in lines)
    return (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {WIDTH}\nPlayResY: {HEIGHT}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Title,{font},{TITLE_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,8,2,8,60,60,200,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        f"Dialogue: 0,0:00:00.00,0:00:10.00,Title,,0,0,0,,{text}\n"
    )


def build_command(video, seconds, out, width, height, title=False):
    return ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{seconds:.3f}", "-i", str(video),
            "-filter_complex", build_filter(width, height, title), "-map", "[out]",
            "-frames:v", "1", "-q:v", "2", str(out)]


def source_video(job):
    """The picture to take frames from: the original (no burned-in subtitle bar), else the dubbed video."""
    job = Path(job)
    for path in (job / "source.mp4", job / "dub_output" / "dubbed.mp4"):
        if path.is_file():
            return path
    return None


def make(job, title=None, force=False):
    """Make dub_output/thumbnail.jpg for a finished job; returns its path (the existing one unless `force`).

    Raises RuntimeError when no video is there or ffmpeg fails.
    """
    job = Path(job)
    out = job / OUTPUT_NAME
    if out.is_file() and not force:
        return out
    video = source_video(job)
    if video is None:
        raise RuntimeError(f"No video in {job} to take a thumbnail from")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    info, _ = mux_dub.probe(video)
    size = info.get("video") or {}
    width, height = size.get("width"), size.get("height")
    seconds, readable = choose_time(video, info.get("duration"))
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name("thumbnail.part.jpg")
    with tempfile.TemporaryDirectory() as work:
        if title:
            (Path(work) / TITLE_FILE).write_text(title_ass(title), encoding="utf-8")
        result = subprocess.run(build_command(video, seconds, temporary.resolve(), width, height, bool(title)),
                                cwd=work, capture_output=True, text=True)
    if result.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg could not make the thumbnail: {result.stderr.strip()[-300:]}")
    temporary.replace(out)
    print(f"Thumbnail: {out} (frame at {seconds:.1f}s, best of {readable})", flush=True)
    return out


def run_local_checks():
    # moments
    times = candidate_times(100)
    assert len(times) == CANDIDATES and times[0] == 8.0 and times[-1] == 92.0
    assert candidate_times(0) == [0.0] and candidate_times(2) == [1.0]
    # scoring: a crisp picture beats a blurred one, black / white / flat frames are unusable
    rng = np.random.default_rng(0)
    crisp = rng.integers(60, 200, (SAMPLE, SAMPLE), dtype=np.uint8)
    ramp = np.tile(np.linspace(40, 220, SAMPLE).astype(np.uint8), (SAMPLE, 1))  # smooth: contrast, no detail
    assert frame_score(crisp) > frame_score(ramp) >= 0
    assert frame_score(np.zeros((SAMPLE, SAMPLE), dtype=np.uint8)) == 0
    assert frame_score(np.full((SAMPLE, SAMPLE), 250, dtype=np.uint8)) == 0
    assert frame_score(np.full((SAMPLE, SAMPLE), 128, dtype=np.uint8)) == 0  # no contrast
    assert best_time([(10.0, 5.0), (20.0, 90.0), (30.0, 40.0)]) == 20.0
    assert best_time([(10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]) == 20.0  # nothing usable: the middle
    assert best_time([]) == 0.0
    frames = {10.0: np.zeros((SAMPLE, SAMPLE), dtype=np.uint8), 20.0: crisp, 30.0: None}
    assert choose_time("v.mp4", 40, grab=lambda video, seconds: frames[seconds], times=[10.0, 20.0, 30.0]) == (20.0, 2)
    # layout
    assert is_vertical(1080, 1920) and not is_vertical(1920, 1080) and not is_vertical(None, None)
    horizontal = build_filter(1920, 1080)
    assert "split" in horizontal and "overlay" in horizontal and horizontal.endswith("[out]")
    assert f"scale={WIDTH}:-2" in horizontal and "ass=" not in horizontal
    vertical = build_filter(1080, 1920)
    assert "split" not in vertical and f"crop={WIDTH}:{HEIGHT}" in vertical
    assert build_filter(1920, 1080, title=True).endswith(f",ass={TITLE_FILE}[out]")
    command = build_command("v.mp4", 12.5, "o.jpg", 1920, 1080)
    assert command[:2] == ["ffmpeg", "-y"] and command[command.index("-ss") + 1] == "12.500"
    assert command.index("-ss") < command.index("-i") and command[-1] == "o.jpg"
    # title
    script = title_ass("Bốn kiểu tâm lý tiền bạc, bạn là kiểu nào?")
    assert f"PlayResY: {HEIGHT}" in script and "Dialogue:" in script and "\\N" in script
    assert "Bốn" in script
    # source preference and the files
    with tempfile.TemporaryDirectory() as root:
        job = Path(root)
        assert source_video(job) is None
        (job / "dub_output").mkdir()
        (job / "dub_output" / "dubbed.mp4").write_bytes(b"x")
        assert source_video(job) == job / "dub_output" / "dubbed.mp4"
        (job / "source.mp4").write_bytes(b"x")
        assert source_video(job) == job / "source.mp4"  # no subtitle bar in the original
        (job / OUTPUT_NAME).write_bytes(b"jpg")
        assert make(job) == job / OUTPUT_NAME           # kept, ffmpeg not needed
        empty = Path(root) / "empty"
        empty.mkdir()
        try:
            make(empty)
        except RuntimeError:
            pass
        else:
            raise AssertionError("a job without a video must be refused")
    os.environ["THUMBNAIL"] = "0"
    assert not enabled()
    os.environ.pop("THUMBNAIL")
    assert enabled()
    if shutil.which("ffmpeg"):
        real_check()


def real_check():
    """With ffmpeg on this computer: a small test video becomes a 1080x1920 JPEG."""
    with tempfile.TemporaryDirectory() as root:
        job = Path(root)
        (job / "dub_output").mkdir()
        made = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=10",
             "-t", "6", "-pix_fmt", "yuv420p", str(job / "source.mp4")], capture_output=True)
        if made.returncode != 0:
            return
        out = make(job)
        info, _ = mux_dub.probe(out)
        size = info["video"]
        assert (size["width"], size["height"]) == (WIDTH, HEIGHT), size
        titled = make(job, title="Tiêu đề thử", force=True)
        assert titled.is_file() and titled.stat().st_size > 0


def main():
    parser = argparse.ArgumentParser(description="Make a vertical thumbnail for a finished video.")
    parser.add_argument("video_id")
    parser.add_argument("--title", help="a big title written on the top of the picture")
    parser.add_argument("--force", action="store_true", help="make it again even if there is one")
    args = parser.parse_args()
    folder = JOBS_DIR / args.video_id
    if not folder.is_dir():
        raise SystemExit(f"No job folder {folder}")
    try:
        make(folder, args.title, args.force or bool(args.title))
    except RuntimeError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        run_local_checks()
        print("thumbnail local checks passed")
