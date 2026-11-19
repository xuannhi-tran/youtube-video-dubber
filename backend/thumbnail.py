"""A vertical (9:16) thumbnail picked from the video itself.

    python thumbnail.py VIDEO_ID                 # make dub_output/thumbnail.jpg for a finished job
    python thumbnail.py VIDEO_ID --force         # make it again with a new frame and headline (kept otherwise)
    python thumbnail.py VIDEO_ID --title "Bốn kiểu tâm lý tiền bạc"   # your own title instead of the written one
    python thumbnail.py VIDEO_ID --no-title      # the picture only

The title: a short Vietnamese headline (4 to 8 words, white with a black outline) that an LLM writes from the
video's title, its brief and the start of the Vietnamese translation. It is written once and kept in
dub_output/thumbnail_title.txt and reused (--title gives your own, --force writes a new one). THUMBNAIL_TITLE=0 turns the
written title off; when the LLM fails the picture is made without a title.

dub.py makes one after every finished video (THUMBNAIL=0 in backend/.env or --no-thumbnail turns it
off) and EXPORT_DIR gets a copy next to the video as <video id>.jpg.

How the frame is chosen: about a dozen frames are taken between 8% and 92% of the video, and the one
that is sharpest and neither dark nor washed out wins (black fades, blur and flat frames score low).
A horizontal video becomes 1080x1920 with the frame in the middle over a blurred, enlarged copy of itself;
a vertical video is cropped to fill the picture.

Local only: ffmpeg/ffprobe and numpy, no API.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import llm
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
TITLE_TEXT = Path("dub_output") / "thumbnail_title.txt"   # the written title, kept so a remake does not ask again
TITLE_FONT_SIZE = 120
TITLE_LINE_CHARS = 16
TITLE_MAX_WORDS = 9
TITLE_MAX_LINES = 3
CJK = re.compile(r"[\u3400-\u9fff]")
SAMPLE_CHARS = 1500            # how much of the translation the title writer reads


def enabled():
    """THUMBNAIL in .env: on unless it is 0 / no / false / off."""
    return (os.getenv("THUMBNAIL") or "1").strip().lower() not in ("0", "no", "false", "off")


def title_enabled():
    """THUMBNAIL_TITLE in .env: a written title is on unless it is 0 / no / false / off."""
    return (os.getenv("THUMBNAIL_TITLE") or "1").strip().lower() not in ("0", "no", "false", "off")


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def title_context(job):
    """What the title writer reads: {"title", "channel", "summary", "excerpt"} of a job (missing parts are "")."""
    job = Path(job)
    info = read_json(job / "video_info.json", {})
    brief = read_json(job / "video_brief.json", {}).get("brief") or {}
    segments = read_json(job / "translated_segments.json", [])
    lines, size = [], 0
    for segment in segments if isinstance(segments, list) else []:
        text = str(segment.get("translated_text") or "").strip() if isinstance(segment, dict) else ""
        if text:
            lines.append(text)
            size += len(text) + 1
            if size >= SAMPLE_CHARS:
                break
    return {"title": str(info.get("title") or ""), "channel": str(info.get("channel") or ""),
            "summary": str(brief.get("summary") or ""), "excerpt": " ".join(lines)[:SAMPLE_CHARS]}


def build_title_prompt(context):
    parts = [f"{label}: {context[key]}" for key, label in
             (("title", "Original title (Chinese)"), ("channel", "Channel"), ("summary", "Summary"),
              ("excerpt", "Start of the Vietnamese voice-over")) if context.get(key)]
    return f"""
You write the headline on the thumbnail of a Vietnamese short video (a dub of a Chinese video).
Write ONE headline in Vietnamese:
- 4 to 8 words, in the viewer's own language, natural and easy to read at a glance;
- it makes people curious about this video's real content, but never promises something the video does not say;
- no hashtags, emoji, quotation marks, brackets or Chinese characters, and no full stop at the end;
- start with a capital letter; a "?" or "!" only if it helps.
Answer with ONLY the headline, nothing else.

{chr(10).join(parts)}
"""


def clean_title(answer):
    """The headline inside the model's answer, or "" when it is not usable (empty, Chinese, too long)."""
    text = re.sub(r"<(think|thought)>.*?</\1>", "", answer or "", flags=re.S)
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    title = re.sub(r"^(tiêu đề|headline|title)\s*:\s*", "", lines[0], flags=re.IGNORECASE)
    title = re.sub(r"[#\"“”«»'`*_\[\]()<>{}]", "", title)
    title = " ".join(title.split()).rstrip(".,;:…-– ")
    if not title or CJK.search(title) or len(title.split()) > TITLE_MAX_WORDS:
        return ""
    return title[0].upper() + title[1:]


def title_writers():
    """The LLM calls to try in turn: THUMBNAIL_LLM_PROVIDER/MODEL if set, else translation, then review."""
    provider = (os.getenv("THUMBNAIL_LLM_PROVIDER") or "").strip().lower() or None
    model = (os.getenv("THUMBNAIL_LLM_MODEL") or "").strip() or None
    if provider:
        return [lambda prompt: llm.generate(prompt, provider=provider, model=model)]
    writers = [lambda prompt: llm.generate(prompt, role="translate")]
    if llm.role_configured("review"):
        writers.append(lambda prompt: llm.generate(prompt, role="review"))
    return writers


def write_title(context, writers=None):
    """A headline from the first LLM that gives a usable one; "" when none does (the picture then has no title)."""
    if not any(context.values()):
        return ""
    prompt = build_title_prompt(context)
    for writer in writers if writers is not None else title_writers():
        try:
            title = clean_title(writer(prompt))
        except (llm.QuotaExhausted, llm.LLMError, RuntimeError, ValueError) as error:
            print(f"Thumbnail title: the LLM failed ({str(error)[:120]})", flush=True)
            continue
        if title:
            return title
        print("Thumbnail title: the answer was not usable", flush=True)
    return ""


def job_title(job, new=False, writers=None):
    """The job's thumbnail title: the saved one, else a new one that is saved. "" when it cannot be written."""
    path = Path(job) / TITLE_TEXT
    if not new and path.is_file():
        return path.read_text(encoding="utf-8").strip()
    title = write_title(title_context(job), writers)
    if title:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(title + "\n", encoding="utf-8")
    return title


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


def title_place(width, height, lines):
    """(y of the middle of the title, font size). On a horizontal video the title sits in the blurred band
    above the picture, so it never covers the speaker; on a vertical one it goes near the top."""
    if is_vertical(width, height) or not (width and height):
        return 330, TITLE_FONT_SIZE
    band = (HEIGHT - round(WIDTH * height / width)) / 2    # the blurred strip above the picture
    size = int(max(50, min(TITLE_FONT_SIZE, band * 0.8 / (max(lines, 1) * 1.15))))
    return int(max(band / 2, size * lines * 0.6)), size


def title_ass(title, font=subtitles.DEFAULT_FONT, width=None, height=None):
    """An ASS file with `title` big and centred on the picture, white with a black outline (wrapped to a few lines)."""
    lines = subtitles.wrap(" ".join(title.split()), TITLE_LINE_CHARS)
    y, size = title_place(width, height, len(lines))
    text = "{\\an5\\pos(" + f"{WIDTH // 2},{y}" + ")}" + "\\N".join(subtitles.ass_escape(line) for line in lines)
    return (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {WIDTH}\nPlayResY: {HEIGHT}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Title,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,8,2,8,60,60,200,1\n\n"
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


def make(job, title=None, force=False, written_title=None):
    """Make dub_output/thumbnail.jpg for a finished job; returns its path (the existing one unless `force`).

    `title` is the text to write on it. Without one, the job's written title is used (made by an LLM the first
    time), unless THUMBNAIL_TITLE is off or `written_title` is False; `force` also writes a new title.

    Raises RuntimeError when no video is there or ffmpeg fails.
    """
    job = Path(job)
    out = job / OUTPUT_NAME
    if out.is_file() and not force:
        return out
    video = source_video(job)
    if video is None:
        raise RuntimeError(f"No video in {job} to take a thumbnail from")
    if title:
        (job / TITLE_TEXT).parent.mkdir(parents=True, exist_ok=True)
        (job / TITLE_TEXT).write_text(title + "\n", encoding="utf-8")
    elif written_title is not False and title_enabled():
        title = job_title(job, new=force)
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
            (Path(work) / TITLE_FILE).write_text(title_ass(title, width=width, height=height), encoding="utf-8")
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
    # title layout: in the blurred band above a 16:9 picture, near the top of a vertical one
    y, size = title_place(1920, 1080, 3)
    assert 0 < y < 656 and size <= TITLE_FONT_SIZE and y + size * 1.5 < (HEIGHT - 608) / 2 + 40
    assert title_place(1080, 1920, 3) == (330, TITLE_FONT_SIZE) and title_place(None, None, 2)[0] == 330
    script = title_ass("Bốn kiểu tâm lý tiền bạc, bạn là kiểu nào?", width=1920, height=1080)
    assert f"PlayResY: {HEIGHT}" in script and "Dialogue:" in script and "\\N" in script
    assert "Bốn" in script and "\\an5\\pos(540," in script and "&H00FFFFFF" in script
    # the written title: cleaning, prompt, order of the LLMs, saved text
    assert clean_title('"Bốn kiểu tâm lý tiền bạc."') == "Bốn kiểu tâm lý tiền bạc"
    assert clean_title("Tiêu đề: bạn thuộc kiểu nào?\nGiải thích dài dòng") == "Bạn thuộc kiểu nào?"
    assert clean_title("<think>hmm</think>Tiền bạc nói gì về bạn") == "Tiền bạc nói gì về bạn"
    assert clean_title("") == "" and clean_title("四种金钱心理") == "" and clean_title("a b c d e f g h i j") == ""
    context = {"title": "4种金钱心理模型", "channel": "C", "summary": "Nói về tiền", "excerpt": "Hôm nay"}
    prompt = build_title_prompt(context)
    assert "4种金钱心理模型" in prompt and "Nói về tiền" in prompt and "ONLY the headline" in prompt
    assert write_title({"title": "", "channel": "", "summary": "", "excerpt": ""}, [lambda p: "x"]) == ""
    asked = []
    def broken(prompt):
        asked.append("first")
        raise llm.LLMError("down")
    assert write_title(context, [broken, lambda p: asked.append("second") or "Tiền bạc nói gì về bạn?"]) == \
        "Tiền bạc nói gì về bạn?" and asked == ["first", "second"]
    assert write_title(context, [broken, lambda p: "四种"]) == ""  # nothing usable: no title, no failure
    with tempfile.TemporaryDirectory() as root:
        job = Path(root)
        (job / "video_info.json").write_text(json.dumps({"title": "T", "channel": "K"}), encoding="utf-8")
        (job / "translated_segments.json").write_text(json.dumps(
            [{"translated_text": "Câu một."}, {"translated_text": "Câu hai."}]), encoding="utf-8")
        found = title_context(job)
        assert found["title"] == "T" and found["excerpt"] == "Câu một. Câu hai." and found["summary"] == ""
        assert job_title(job, writers=[lambda p: "Một tiêu đề hay"]) == "Một tiêu đề hay"
        assert job_title(job, writers=[broken]) == "Một tiêu đề hay"          # saved: nobody is asked again
        assert job_title(job, new=True, writers=[lambda p: "Tiêu đề mới"]) == "Tiêu đề mới"
        (job / TITLE_TEXT).unlink()
        assert job_title(job, writers=[broken]) == ""
    os.environ["THUMBNAIL_TITLE"] = "off"
    assert not title_enabled()
    os.environ.pop("THUMBNAIL_TITLE")
    assert title_enabled()
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
        out = make(job, written_title=False)
        info, _ = mux_dub.probe(out)
        size = info["video"]
        assert (size["width"], size["height"]) == (WIDTH, HEIGHT), size
        titled = make(job, title="Tiêu đề thử", force=True)
        assert titled.is_file() and titled.stat().st_size > 0


def main():
    parser = argparse.ArgumentParser(description="Make a vertical thumbnail for a finished video.")
    parser.add_argument("video_id")
    parser.add_argument("--title", help="write this title on the picture (default: a headline an LLM writes)")
    parser.add_argument("--no-title", action="store_true", help="the picture without any title")
    parser.add_argument("--force", action="store_true", help="make it again even if there is one")
    args = parser.parse_args()
    folder = JOBS_DIR / args.video_id
    if not folder.is_dir():
        raise SystemExit(f"No job folder {folder}")
    try:
        make(folder, args.title, args.force or bool(args.title), written_title=False if args.no_title else None)
    except RuntimeError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        run_local_checks()
        print("thumbnail local checks passed")
