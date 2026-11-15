"""Vietnamese subtitles for the dubbed video.

Builds an ASS file from dub_output/dub_timeline.json (what the voice really says,
at the time it really plays) for a black bar that mux_dub.py adds *below* the
video, so the original hard-coded Chinese subtitles stay visible.

Pure helpers only: no ffmpeg, no network.
"""

import json
import math
import re

BAR_RATIO = 0.17        # auto bar height as a share of the video height
FONT_BAR_RATIO = 0.30   # auto font size as a share of the bar height
FONT_WIDTH_DIV = 20     # ...but never wider than width / this (vertical videos)
CHAR_WIDTH = 0.52       # average glyph width as a share of the font size
LINES_PER_CUE = 2
MIN_CUE = 0.8           # seconds; shorter cues are unreadable
DEFAULT_FONT = "Arial"  # has full Vietnamese diacritics on Windows/macOS


def even(value):
    return max(2, int(math.ceil(value / 2)) * 2)


def layout(width, height, bar=0, font_size=0):
    """Return (bar_height, font_size) in pixels; 0 means automatic."""
    if width <= 0 or height <= 0:
        raise ValueError("video size must be positive")
    bar = even(bar) if bar > 0 else even(height * BAR_RATIO)
    if font_size <= 0:
        font_size = min(bar * FONT_BAR_RATIO, width / FONT_WIDTH_DIV)
    return bar, max(8, int(round(font_size)))


def chars_per_line(width, font_size):
    return max(8, int(width * 0.92 / (font_size * CHAR_WIDTH)))


def wrap(text, limit):
    """Greedy word wrap; a word longer than the limit stays on its own line."""
    lines, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > limit:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def make_cues(timeline, limit, lines_per_cue=LINES_PER_CUE):
    """One block -> one or more cues of at most `lines_per_cue` lines.

    A long sentence is split across consecutive cues, sharing the block's time
    in proportion to the characters shown, so it never grows past two lines.
    """
    cues = []
    for item in timeline:
        text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
        start, end = float(item["start"]), float(item["end"])
        if not text or end <= start:
            continue
        lines = wrap(text, limit)
        groups = [lines[i:i + lines_per_cue] for i in range(0, len(lines), lines_per_cue)]
        weights = [sum(len(line) for line in group) for group in groups]
        total, cursor = sum(weights), start
        for index, (group, weight) in enumerate(zip(groups, weights)):
            stop = end if index == len(groups) - 1 else cursor + (end - start) * weight / total
            cues.append({"start": cursor, "end": max(stop, cursor + MIN_CUE) if len(groups) == 1 else stop,
                         "lines": group})
            cursor = stop
    # A padded short cue must not run into the next one.
    for current, following in zip(cues, cues[1:]):
        current["end"] = min(current["end"], following["start"]) if following["start"] > current["start"] else current["end"]
    return cues


def ass_time(seconds):
    centis = max(0, int(round(seconds * 100)))
    return f"{centis // 360000}:{centis // 6000 % 60:02d}:{centis // 100 % 60:02d}.{centis % 100:02d}"


def ass_escape(line):
    return line.replace("\\", "").replace("{", "(").replace("}", ")")


def build_ass(timeline, width, height, bar=0, font_size=0, font=DEFAULT_FONT):
    """ASS script for a video of width x height that gets `bar` px added below."""
    bar, size = layout(width, height, bar, font_size)
    total = height + bar
    cues = make_cues(timeline, chars_per_line(width, size))
    outline = max(1, round(size / 18))
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {total}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Viet,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
        f"0,0,0,0,100,100,0,0,1,{outline},0,5,10,10,10,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    centre_y = height + bar // 2
    events = []
    for cue in cues:
        text = "\\N".join(ass_escape(line) for line in cue["lines"])
        events.append(f"Dialogue: 0,{ass_time(cue['start'])},{ass_time(cue['end'])},Viet,,0,0,0,,"
                      f"{{\\an5\\pos({width // 2},{centre_y})}}{text}")
    return header + "\n".join(events) + "\n", len(cues)


def load_timeline(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list")
    return data


def run_local_checks():
    assert even(121) == 122 and even(0) == 2
    assert layout(1280, 720) == (124, 37)
    bar, size = layout(1080, 1920)
    assert bar == 328 and size == 54, (bar, size)  # width-limited on vertical video
    assert layout(1280, 720, bar=100, font_size=30) == (100, 30)
    for bad in ((0, 720), (1280, -1)):
        try:
            layout(*bad)
        except ValueError:
            continue
        raise AssertionError("accepted bad size")

    assert wrap("một hai ba bốn năm", 10) == ["một hai ba", "bốn năm"]
    assert wrap("khôngngắtđược từ", 5) == ["khôngngắtđược", "từ"]
    assert wrap("", 10) == []

    text = "Xin chào các bạn, hôm nay chúng ta sẽ cùng nhau đi tìm hiểu một câu chuyện rất dài và thú vị"
    timeline = [{"block_id": 1, "start": 1.0, "end": 9.0, "text": text},
                {"block_id": 2, "start": 9.0, "end": 9.2, "text": "Ừ."},
                {"block_id": 3, "start": 12.0, "end": 12.0, "text": "bị bỏ"},
                {"block_id": 4, "start": 13.0, "end": 15.0, "text": "  "}]
    cues = make_cues(timeline, 20)
    assert all(len(c["lines"]) <= 2 for c in cues) and len(cues) >= 3
    first = [c for c in cues if c["start"] < 9.0]
    assert abs(first[0]["start"] - 1.0) < 1e-9 and abs(first[-1]["end"] - 9.0) < 1e-9
    for a, b in zip(first, first[1:]):
        assert abs(a["end"] - b["start"]) < 1e-9
    short = cues[-1]
    assert short["lines"] == ["Ừ."] and short["start"] == 9.0
    assert len(cues) == len(first) + 1  # empty and zero-length blocks dropped

    assert ass_time(0) == "0:00:00.00" and ass_time(3725.5) == "1:02:05.50"
    assert ass_escape("a{b}\\c") == "a(b)c"

    script, count = build_ass(timeline, 1280, 720)
    assert count >= 2 and "PlayResY: 844" in script and "Fontname" in script
    assert "\\pos(640,782)" in script  # centre of the 124px bar below the 720px video
    assert "Style: Viet,Arial,37," in script
    assert script.count("Dialogue:") == count
    assert "tìm hiểu" in script  # Vietnamese diacritics survive untouched
    assert build_ass([], 1280, 720, font="Tahoma")[0].count("Dialogue:") == 0
