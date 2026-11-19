"""A vertical (9:16) cover picture made from the video itself.

    python thumbnail.py VIDEO_ID                 # make dub_output/thumbnail.jpg for a finished job
    python thumbnail.py VIDEO_ID --force         # make it again: a new frame and new texts (kept otherwise)
    python thumbnail.py VIDEO_ID --redraw        # draw it again with the saved texts (edit thumbnail_text.json first)
    python thumbnail.py VIDEO_ID --title "Bốn kiểu tâm lý tiền bạc"   # your own title, the other texts stay
    python thumbnail.py VIDEO_ID --no-title      # the picture only

The look follows the usual TikTok cover: the frame darkened (darker towards the bottom), a big white
UPPER-CASE title in the middle and a small upper-case topic line under it, inside TikTok's safe area
(THUMBNAIL_HOOK=1 adds a gold hook sentence lower down; it is off by default). An LLM writes the three texts from the video's title, its brief and the start
of the Vietnamese translation; they are kept in dub_output/thumbnail_text.json and reused.
THUMBNAIL_TITLE=0 leaves the text off; when the LLM fails the picture is made without text.

dub.py makes one after every finished video (THUMBNAIL=0 in backend/.env or --no-thumbnail turns it
off) and EXPORT_DIR gets a copy next to the video as <video id>.jpg.

How the frame is chosen: about a dozen frames are taken between 8% and 92% of the video, and the one
that is sharpest and neither dark nor washed out wins (black fades, blur and flat frames score low).
A horizontal video becomes 1080x1920 with the frame in the middle over a blurred, enlarged copy of itself;
a vertical video is cropped to fill the picture.

Needs ffmpeg and numpy; only the texts use an LLM.
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
import video_brief
from workdir import SCRIPT_DIR


WIDTH, HEIGHT = 1080, 1920
OUTPUT_NAME = Path("dub_output") / "thumbnail.jpg"
JOBS_DIR = SCRIPT_DIR / "dub_jobs_output"
CANDIDATES = 12
FIRST, LAST = 0.08, 0.92       # share of the duration the candidates are taken from
SAMPLE = 256                    # candidates are scored on a SAMPLE x SAMPLE gray picture
TOO_DARK, TOO_BRIGHT, FLAT = 35.0, 225.0, 14.0   # mean brightness limits and the least contrast (std) of a usable frame
VERTICAL_FROM = 1.5             # height / width from which a video is cropped instead of padded
BLUR = "boxblur=luma_radius=28:luma_power=3"   # light, so the speaker can still be made out in the background
TEXT_ASS = "thumbnail_text.ass"
TEXT_FILE = Path("dub_output") / "thumbnail_text.json"    # the written texts, kept so a remake does not ask again
TEXT_LIMITS = {"title": 9, "topic": 6, "hook": 22}         # most words of each text
CJK = re.compile(r"[\u3400-\u9fff]")
SAMPLE_CHARS = 1500            # how much of the translation the text writer reads
SIDE_MARGIN = 60               # px left and right of the texts
CENTER_Y = 980                 # middle of the title + topic block: the middle of the picture, as on the TikTok covers
SAFE_BOTTOM = 1560             # the bottom 380 px belong to TikTok's caption: the hook ends above this
HOOK_GAP = 330                 # px between the topic line and the hook (less when the hook would pass SAFE_BOTTOM)
FONTS_DIR = SCRIPT_DIR / "fonts"   # Be Vietnam Pro and Lora (SIL Open Font License, with the Vietnamese diacritics)
TITLE_SIZES, TOPIC_SIZES, HOOK_SIZES = (112, 104, 96, 88, 80, 72, 64), (42, 38), (72, 66, 60, 54, 48, 44)
TITLE_LINES, TOPIC_LINES, HOOK_LINES = 3, 2, 3
TITLE_TWO_LINES_FROM, HOOK_TWO_LINES_FROM = 80, 54   # a text is kept to 2 lines unless that needs a smaller font
GOLD, SOFT_WHITE = "&H004AA0E8", "&H00D8D8D8"   # ASS colours are BBGGRR: orange-gold E8A04A and light grey
DIM = "eq=brightness=-0.10:saturation=0.90"      # the picture is darkened so white text reads on any frame
SHADE = "clip(8+125*pow(Y/H,3),0,135)"          # extra darkness (alpha), strongest at the bottom


def font(name, default):
    return (os.getenv(name) or "").strip() or default


def enabled():
    """THUMBNAIL in .env: on unless it is 0 / no / false / off."""
    return (os.getenv("THUMBNAIL") or "1").strip().lower() not in ("0", "no", "false", "off")


def title_enabled():
    """THUMBNAIL_TITLE in .env: the written texts are on unless it is 0 / no / false / off."""
    return (os.getenv("THUMBNAIL_TITLE") or "1").strip().lower() not in ("0", "no", "false", "off")


def hook_enabled():
    """THUMBNAIL_HOOK in .env: the gold hook sentence at the bottom is off unless it is 1 / yes / true / on."""
    return (os.getenv("THUMBNAIL_HOOK") or "0").strip().lower() in ("1", "yes", "true", "on")


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def title_context(job):
    """What the text writer reads: {"title", "channel", "summary", "excerpt"} of a job (missing parts are "")."""
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
    hook_rule = ('- "hook": ONE sentence of 8 to 16 words in simple spoken Vietnamese, taken from the video or the key idea\n'
                 '  of it, that makes the viewer want to hear the rest;\n') if hook_enabled() else ""
    return f"""
You write the texts on the cover (thumbnail) of a Vietnamese short video (a dub of a Chinese video).
Answer with ONE JSON object, in Vietnamese:

{{"title": "...", "topic": "..."{', "hook": "..."' if hook_enabled() else ""}}}

- "title": 4 to 8 words, a question or a bold statement that makes people curious about this video's
  real content, never a promise the video does not keep;
- "topic": 2 to 5 words naming the subject area, like a section label (e.g. "Tâm lý học", "Lịch sử Trung Hoa");
{hook_rule}- natural Vietnamese a viewer reads at a glance; no hashtags, emoji, quotation marks, brackets or Chinese
  characters inside the values; no full stop at the end of "title" and "topic".
Return ONLY the JSON object.

{chr(10).join(parts)}
"""


def clean_text(value, max_words, capital=True):
    """One cover text made tidy; "" when it is empty, has Chinese characters or is too long."""
    text = re.sub(r"[#\"“”«»`*_\[\]<>{}]", "", str(value or ""))
    text = " ".join(text.split()).strip(" ,;:…-–")
    if not text or CJK.search(text) or len(text.split()) > max_words:
        return ""
    return text[0].upper() + text[1:] if capital else text


def parse_texts(answer):
    """{"title", "topic", "hook"} from the model's answer (a field it got wrong is ""), or {} without a usable title."""
    text = re.sub(r"<(think|thought)>.*?</\1>", "", answer or "", flags=re.S)
    try:
        data = video_brief._json_object(text)
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    texts = {key: clean_text(data.get(key), limit) for key, limit in TEXT_LIMITS.items()}
    texts["title"] = texts["title"].rstrip(".")
    texts["topic"] = texts["topic"].rstrip(".")
    return texts if texts["title"] else {}


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


def write_texts(context, writers=None):
    """The cover texts from the first LLM that gives usable ones; {} when none does (the picture then has no text)."""
    if not any(context.values()):
        return {}
    prompt = build_title_prompt(context)
    for writer in writers if writers is not None else title_writers():
        try:
            texts = parse_texts(writer(prompt))
        except (llm.QuotaExhausted, llm.LLMError, RuntimeError, ValueError) as error:
            print(f"Thumbnail text: the LLM failed ({str(error)[:120]})", flush=True)
            continue
        if texts:
            return texts
        print("Thumbnail text: the answer was not usable", flush=True)
    return {}


def job_texts(job, new=False, writers=None):
    """The job's cover texts: the saved ones, else new ones that are saved. {} when they cannot be written."""
    path = Path(job) / TEXT_FILE
    saved = read_json(path, {})
    if not new and isinstance(saved, dict) and saved.get("title"):
        return {key: str(saved.get(key) or "") for key in TEXT_LIMITS}
    texts = write_texts(title_context(job), writers)
    if texts:
        save_texts(job, texts)
    return texts


def save_texts(job, texts):
    path = Path(job) / TEXT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({key: texts.get(key, "") for key in TEXT_LIMITS}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


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


def grab_gray(video, seconds, run=subprocess.run, crop=None):
    """One frame at `seconds` as a SAMPLE x SAMPLE gray numpy array; None when ffmpeg gives nothing.
    `crop` keeps only the top `crop` pixel rows (the picture above a burned-in subtitle bar)."""
    cut = f"crop=iw:{crop}:0:0," if crop else ""
    command = ["ffmpeg", "-loglevel", "error", "-ss", f"{seconds:.3f}", "-i", str(video), "-frames:v", "1",
               "-vf", f"{cut}scale={SAMPLE}:{SAMPLE},format=gray", "-f", "rawvideo", "-"]
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


def build_filter(width, height, styled=False, crop=None):
    """The -filter_complex that turns one frame into the WIDTH x HEIGHT picture. `styled` darkens it
    (more towards the bottom) and writes the cover texts of thumbnail_text.ass on it. `crop` keeps only the
    top `crop` rows of the frame (`height` is then that cropped height)."""
    start = f"[0:v]crop=iw:{crop}:0:0[src];[src]" if crop else "[0:v]"
    if is_vertical(width, height):
        graph = (f"{start}scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
                 f"crop={WIDTH}:{HEIGHT},setsar=1")
    else:
        graph = (f"{start}split[a][b];"
                 f"[a]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},{BLUR}[bg];"
                 f"[b]scale={WIDTH}:-2[fg];"
                 f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")
    if not styled:
        return graph + "[out]"
    return (graph + f",{DIM}[dim];"
            f"color=c=black:s={WIDTH}x{HEIGHT}:d=1,format=rgba,geq=r=0:g=0:b=0:a='{SHADE}'[shade];"
            f"[dim][shade]overlay,ass={TEXT_ASS}:fontsdir=.[out]")


def balanced_wrap(text, limit):
    """Wrap `text` at `limit` characters, then shrink the limit while the line count stays the same, so the
    lines come out about equally long (no lone word on the last line)."""
    lines = subtitles.wrap(text, limit)
    count = len(lines)
    best = limit
    while best > 4 and len(subtitles.wrap(text, best - 1)) == count:
        best -= 1
    return subtitles.wrap(text, best)


def fit_lines(text, sizes, width_factor, max_lines, two_lines_from=None):
    """(lines, font size): the biggest size from `sizes` whose wrapped text fits in few lines. A text goes on
    2 lines whenever that needs no font smaller than `two_lines_from`; only then are up to `max_lines` allowed."""
    room = WIDTH - 2 * SIDE_MARGIN
    for cap in range(2 if two_lines_from else max_lines, max_lines + 1):
        for size in sizes:
            if cap < max_lines and size < two_lines_from:
                continue
            limit = max(int(room / (width_factor * size)), 4)
            if len(subtitles.wrap(text, limit)) <= cap:
                return balanced_wrap(text, limit), size
    size = sizes[-1]
    lines = subtitles.wrap(text, max(int(room / (width_factor * size)), 4))
    return lines[:max_lines], size   # never more lines than allowed (a text of up to 22 words always fits)


def text_ass(texts):
    """An ASS file with the cover texts in the layout of the TikTok covers: a big white upper-case title and a
    small upper-case topic line, the two together in the middle of the picture, and a gold hook sentence lower down."""
    sans = font("THUMBNAIL_FONT", "Be Vietnam Pro")    # fonts with all the Vietnamese diacritics (bundled in fonts/)
    sans_medium = font("THUMBNAIL_TOPIC_FONT", "Be Vietnam Pro Medium")
    serif = font("THUMBNAIL_HOOK_FONT", "Lora")
    events, styles = [], []

    def put(name, y, size, bold, colour, face, outline, spacing, lines, box=False):
        # box: a translucent dark box behind the text (BorderStyle 3), so it stays readable over a busy frame
        edge = "&H80000000" if box else "&H00000000"
        styles.append(f"Style: {name},{face},{size},{colour},{colour},{edge},&H78000000,{-1 if bold else 0},0,0,0,"
                      f"100,100,{spacing},0,{3 if box else 1},{outline},{0 if box else 2},8,{SIDE_MARGIN},{SIDE_MARGIN},0,1")
        body = "\\N".join(subtitles.ass_escape(line) for line in lines)
        events.append(f"Dialogue: 0,0:00:00.00,0:00:10.00,{name},,0,0,0,,{{\\pos({WIDTH // 2},{int(y)})}}{body}")

    title = topic = None
    if texts.get("title"):
        lines, size = fit_lines(texts["title"].upper(), TITLE_SIZES, 0.47, TITLE_LINES, TITLE_TWO_LINES_FROM)
        title = (lines, size, len(lines) * size * 1.22)
    if texts.get("topic"):
        lines, size = fit_lines(texts["topic"].upper(), TOPIC_SIZES, 0.55, TOPIC_LINES)
        topic = (lines, size, len(lines) * size * 1.35)
    gap = 36 if title and topic else 0
    block = sum(part[2] for part in (title, topic) if part) + gap
    top = CENTER_Y - block / 2
    if title:
        put("Title", top, title[1], True, "&H00FFFFFF", sans, 2, 0, title[0])
        top += title[2] + gap
    if topic:
        put("Topic", top, topic[1], False, SOFT_WHITE, sans_medium, 10, 5, topic[0], box=True)
        top += topic[2]
    if texts.get("hook") and hook_enabled():
        lines, size = fit_lines(texts["hook"], HOOK_SIZES, 0.40, HOOK_LINES, HOOK_TWO_LINES_FROM)
        height = len(lines) * size * 1.4
        put("Hook", min(top + HOOK_GAP, SAFE_BOTTOM - height), size, False, GOLD, serif, 1, 0, lines)
    return (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {WIDTH}\nPlayResY: {HEIGHT}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        + "\n".join(styles) + "\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        + "\n".join(events) + "\n"
    )


def build_command(video, seconds, out, width, height, styled=False, crop=None):
    return ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{seconds:.3f}", "-i", str(video),
            "-filter_complex", build_filter(width, height, styled, crop), "-map", "[out]",
            "-frames:v", "1", "-q:v", "2", str(out)]


def source_video(job):
    """The picture to take frames from: the original (no burned-in subtitle bar), else the dubbed video."""
    job = Path(job)
    for path in (job / "source.mp4", job / "dub_output" / "dubbed.mp4"):
        if path.is_file():
            return path
    return None


def subtitle_bar_crop(job, video, video_height):
    """Rows of the picture above the burned-in subtitle bar, when `video` is the dubbed video of a --subs job
    (its dub_subs.ass says how tall the original was); None when there is no bar to cut off."""
    if Path(video).name != "dubbed.mp4":
        return None   # the original video has no bar
    try:
        text = (Path(job) / "dub_output" / "dub_subs.ass").read_text(encoding="utf-8-sig")
    except OSError:
        return None
    total, centre = re.search(r"PlayResY:\s*(\d+)", text), re.search(r"\\pos\(\d+,(\d+)\)", text)
    if not total or not centre or int(total.group(1)) != video_height:
        return None   # no cues, or the video was made again without subtitles since
    height = 2 * int(centre.group(1)) - int(total.group(1))   # the bar is centred: centre = height + bar / 2
    height -= height % 2
    return height if 0 < height < video_height else None


def make(job, title=None, force=False, written_title=None, new_text=None):
    """Make dub_output/thumbnail.jpg for a finished job; returns its path (the existing one unless `force`).

    The picture carries the job's cover texts (title, topic, hook), written by an LLM the first time and kept in
    dub_output/thumbnail_text.json. `title` replaces the title; `written_title=False` or THUMBNAIL_TITLE=0 leaves
    all text off; `force` also writes new texts unless `new_text` is False (draw again with the saved ones).

    Raises RuntimeError when no video is there or ffmpeg fails.
    """
    job = Path(job)
    out = job / OUTPUT_NAME
    if out.is_file() and not force:
        return out
    video = source_video(job)
    if video is None:
        raise RuntimeError(f"No video in {job} to take a thumbnail from")
    texts = {}
    if title:
        saved = read_json(job / TEXT_FILE, {})
        texts = {**{key: str(saved.get(key) or "") for key in TEXT_LIMITS if isinstance(saved, dict)}, "title": title}
        save_texts(job, texts)
    elif written_title is not False and title_enabled():
        texts = job_texts(job, new=force if new_text is None else new_text)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    info, _ = mux_dub.probe(video)
    size = info.get("video") or {}
    width, height = size.get("width"), size.get("height")
    crop = subtitle_bar_crop(job, video, height)
    if crop:
        height = crop
    seconds, readable = choose_time(video, info.get("duration"), grab=lambda v, t: grab_gray(v, t, crop=crop))
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name("thumbnail.part.jpg")
    with tempfile.TemporaryDirectory() as work:
        if texts:
            (Path(work) / TEXT_ASS).write_text(text_ass(texts), encoding="utf-8")
            for face in sorted(FONTS_DIR.glob("*.ttf")):   # libass reads the bundled fonts from the working folder
                shutil.copy2(face, work)
        result = subprocess.run(build_command(video, seconds, temporary.resolve(), width, height, bool(texts), crop),
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
    command = build_command("v.mp4", 12.5, "o.jpg", 1920, 1080)
    assert command[:2] == ["ffmpeg", "-y"] and command[command.index("-ss") + 1] == "12.500"
    assert command.index("-ss") < command.index("-i") and command[-1] == "o.jpg"
    # a --subs video made by dub.py: the bar below the picture is cut off when the frame comes from dubbed.mp4
    with tempfile.TemporaryDirectory() as root:
        job = Path(root)
        (job / "dub_output").mkdir()
        (job / "dub_output" / "dub_subs.ass").write_text(
            "[Script Info]\nPlayResX: 1280\nPlayResY: 844\n\n[Events]\n"
            "Dialogue: 0,0:00:00.00,0:00:02.00,Viet,,0,0,0,,{\\an5\\pos(640,782)}Xin chào\n", encoding="utf-8-sig")
        dubbed, original = job / "dub_output" / "dubbed.mp4", job / "source.mp4"
        assert subtitle_bar_crop(job, dubbed, 844) == 720                 # 844 = 720 + a 124 px bar
        assert subtitle_bar_crop(job, original, 844) is None              # the original never has the bar
        assert subtitle_bar_crop(job, dubbed, 720) is None                # made again without subtitles
        assert subtitle_bar_crop(job / "none", dubbed, 844) is None
    cropped = build_filter(1280, 720, crop=720)
    assert cropped.startswith("[0:v]crop=iw:720:0:0[src];[src]split[a][b];") and cropped.endswith("[out]")
    assert "crop=iw:" not in build_filter(1280, 720)
    # layout of the texts
    horizontal = build_filter(1920, 1080, styled=True)
    assert f"ass={TEXT_ASS}:fontsdir=.[out]" in horizontal and DIM in horizontal and "geq=" in horizontal
    texts = {"title": "Bạn có đang tự dằn nhân rối loạn?", "topic": "Tâm lý khoa học khác thường",
             "hook": "Hôm nay tôi sẽ chỉ cho bạn một cách để thoát khỏi vòng lặp của sự lo âu."}
    assert not hook_enabled() and text_ass(texts).count("Dialogue:") == 2 and "hook" not in build_title_prompt({"title": "T"})
    os.environ["THUMBNAIL_HOOK"] = "1"
    assert hook_enabled() and '"hook"' in build_title_prompt({"title": "T"})
    script = text_ass(texts)
    os.environ.pop("THUMBNAIL_HOOK")
    assert f"PlayResY: {HEIGHT}" in script and script.count("Dialogue:") == 3
    assert "BẠN CÓ ĐANG" in script and "TÂM LÝ KHOA HỌC" in script and "\\N" in script
    assert "&H00FFFFFF" in script and GOLD in script and f"\\pos({WIDTH // 2}," in script
    assert text_ass({"title": "Chỉ có tiêu đề"}).count("Dialogue:") == 1
    long_lines, long_size = fit_lines("MỘT TIÊU ĐỀ RẤT RẤT DÀI " * 6, TITLE_SIZES, 0.47, TITLE_LINES, TITLE_TWO_LINES_FROM)
    assert len(long_lines) == TITLE_LINES and long_size == TITLE_SIZES[-1]    # never more lines than allowed
    assert fit_lines("NGẮN GỌN", TITLE_SIZES, 0.47, TITLE_LINES, TITLE_TWO_LINES_FROM) == (["NGẮN GỌN"], TITLE_SIZES[0])
    even, size = fit_lines("BẠN CÓ ĐANG TỰ DẰN NHÂN RỐI LOẠN?", TITLE_SIZES, 0.47, TITLE_LINES, TITLE_TWO_LINES_FROM)
    assert len(even) == 2 and size >= TITLE_TWO_LINES_FROM and min(map(len, even)) >= 12   # two even lines, not three
    # the title and topic are centred on the picture, the hook stays above TikTok's caption area
    middle = [int(line.split("pos(540,")[1].split(")")[0]) for line in script.splitlines() if "pos(540," in line]
    assert len(middle) == 3 and middle[0] < CENTER_Y < middle[1] + 60 and middle[2] > middle[1]
    # the written texts: parsing, order of the LLMs, saved file
    good = '{"title": "Bạn thuộc kiểu nào?", "topic": "Tâm lý học.", "hook": "Câu hook dài vừa phải thôi nhé."}'
    assert parse_texts(good) == {"title": "Bạn thuộc kiểu nào?", "topic": "Tâm lý học",
                                 "hook": "Câu hook dài vừa phải thôi nhé."}
    assert parse_texts("<think>x</think>```json\n" + good + "\n```")["title"] == "Bạn thuộc kiểu nào?"
    assert parse_texts('{"title": "Ok title", "topic": "四种", "hook": ""}') == {"title": "Ok title", "topic": "", "hook": ""}
    assert parse_texts('{"title": "四种金钱心理", "topic": "x"}') == {} and parse_texts("no json") == {}
    assert parse_texts('{"title": "a b c d e f g h i j", "topic": "x"}') == {}   # too many words
    context = {"title": "4种金钱心理模型", "channel": "C", "summary": "Nói về tiền", "excerpt": "Hôm nay"}
    prompt = build_title_prompt(context)
    assert "4种金钱心理模型" in prompt and "Nói về tiền" in prompt and "ONE JSON object" in prompt
    assert write_texts({"title": "", "channel": "", "summary": "", "excerpt": ""}, [lambda p: good]) == {}
    asked = []
    def broken(prompt):
        asked.append("first")
        raise llm.LLMError("down")
    assert write_texts(context, [broken, lambda p: asked.append("second") or good])["title"] == "Bạn thuộc kiểu nào?" \
        and asked == ["first", "second"]
    assert write_texts(context, [broken, lambda p: "không phải json"]) == {}   # nothing usable: no text, no failure
    with tempfile.TemporaryDirectory() as root:
        job = Path(root)
        (job / "video_info.json").write_text(json.dumps({"title": "T", "channel": "K"}), encoding="utf-8")
        (job / "translated_segments.json").write_text(json.dumps(
            [{"translated_text": "Câu một."}, {"translated_text": "Câu hai."}]), encoding="utf-8")
        found = title_context(job)
        assert found["title"] == "T" and found["excerpt"] == "Câu một. Câu hai." and found["summary"] == ""
        assert job_texts(job, writers=[lambda p: good])["topic"] == "Tâm lý học"
        assert job_texts(job, writers=[broken])["title"] == "Bạn thuộc kiểu nào?"       # saved: nobody is asked again
        assert job_texts(job, new=True, writers=[lambda p: '{"title": "Tiêu đề mới"}'])["title"] == "Tiêu đề mới"
        (job / TEXT_FILE).unlink()
        assert job_texts(job, writers=[broken]) == {}
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
        (job / TEXT_FILE).write_text(json.dumps({"title": "Bạn có đang tự dằn nhân rối loạn?", "topic": "Tâm lý",
                                                  "hook": "Hôm nay tôi chỉ cho bạn một cách."}), encoding="utf-8")
        titled = make(job, force=True, new_text=False)
        assert titled.is_file() and titled.stat().st_size > out.stat().st_size // 4
        replaced = make(job, title="Tiêu đề thử", force=True)
        assert replaced.is_file() and json.loads((job / TEXT_FILE).read_text(encoding="utf-8"))["topic"] == "Tâm lý"


def main():
    parser = argparse.ArgumentParser(description="Make a vertical thumbnail for a finished video.")
    parser.add_argument("video_id")
    parser.add_argument("--title", help="write this title on the picture (default: a headline an LLM writes)")
    parser.add_argument("--no-title", action="store_true", help="the picture without any text")
    parser.add_argument("--redraw", action="store_true",
                        help="draw it again with the texts saved in dub_output/thumbnail_text.json (edit them first)")
    parser.add_argument("--force", action="store_true", help="make it again even if there is one")
    args = parser.parse_args()
    folder = JOBS_DIR / args.video_id
    if not folder.is_dir():
        raise SystemExit(f"No job folder {folder}")
    try:
        make(folder, args.title, args.force or args.redraw or bool(args.title),
             written_title=False if args.no_title else None, new_text=False if args.redraw or args.title else None)
    except RuntimeError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        run_local_checks()
        print("thumbnail local checks passed")
