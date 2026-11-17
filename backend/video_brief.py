"""What the translator should know about the whole video before it translates a line.

Two files in the job folder:

    video_info.json   written by dub.py: title, description and channel from YouTube, and the
                      speaker ("male", "female" or "" for unknown) chosen with --speaker
    video_brief.json  written by pipeline.py after one LLM call that reads the whole transcript:
                      a short summary, the speaker as heard, how the audience is addressed, the
                      names and terms with their Vietnamese, and the words speech recognition
                      clearly got wrong (fixed in the transcript before it is translated)

Each 20-second passage is then translated and reviewed knowing what the video is about, who
speaks and how names are written, instead of guessing from the passage alone.
VIDEO_BRIEF=0 in .env turns the LLM call off (the title and the speaker are still used).
"""

import hashlib
import json
import os
import re

from workdir import WORKDIR


INFO_FILE = WORKDIR / "video_info.json"
BRIEF_FILE = WORKDIR / "video_brief.json"
BRIEF_VERSION = 1  # part of the brief's cache key: raise it when the prompt below changes
SPEAKERS = ("male", "female")
MAX_TRANSCRIPT_CHARS = 24000  # about 25 minutes of speech; longer videos are sampled evenly
MAX_DESCRIPTION_CHARS = 800
MAX_TERMS = 60
MAX_FIXES = 40
DEFAULT_WORDS_PER_SECOND = 4.0  # Vietnamese read by VieNeu; replaced by the calibrated rate when known
CJK = re.compile(r"[㐀-䶿一-鿿]")


# ---------------------------------------------------------
# Files
# ---------------------------------------------------------

def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def load_info(path=INFO_FILE):
    """video_info.json as a dict with the known keys (missing ones empty)."""
    data = read_json(path, {})
    data = data if isinstance(data, dict) else {}
    info = {key: str(data.get(key) or "").strip() for key in ("title", "description", "channel", "speaker")}
    if info["speaker"] not in SPEAKERS:
        info["speaker"] = ""
    return info


def info_from_ytdlp(data, speaker=None, previous=None):
    """The video_info.json content from yt-dlp's JSON; `speaker` None keeps the previous choice."""
    previous = previous or {}
    chosen = previous.get("speaker", "") if speaker is None else ("" if speaker == "auto" else speaker)
    return {
        "title": str(data.get("title") or "").strip(),
        "description": str(data.get("description") or "").strip()[:MAX_DESCRIPTION_CHARS],
        "channel": str(data.get("channel") or data.get("uploader") or "").strip(),
        "speaker": chosen if chosen in SPEAKERS else "",
    }


def enabled():
    return (os.getenv("VIDEO_BRIEF") or "1").strip().lower() not in ("0", "no", "false", "off")


# ---------------------------------------------------------
# The brief (one LLM call over the whole transcript)
# ---------------------------------------------------------

def transcript_text(segments, limit=MAX_TRANSCRIPT_CHARS):
    """One line per segment; a transcript longer than `limit` is sampled evenly so the call stays small."""
    lines = [segment["source_text"] for segment in segments if segment.get("source_text")]
    text = "\n".join(lines)
    if len(text) <= limit or not lines:
        return text
    average = len(text) / len(lines)
    keep = max(1, int(limit / average))
    step = len(lines) / keep
    return "\n".join(lines[int(i * step)] for i in range(keep))


def build_brief_prompt(info, segments):
    about = "\n".join(f"{label}: {info[key]}" for key, label in
                      (("title", "Title"), ("channel", "Channel"), ("description", "Description")) if info.get(key))
    return f"""
You prepare a Chinese video for a Vietnamese voice-over. Read the whole speech-recognition
transcript below (one line per piece of speech) and answer with ONE JSON object:

{{
  "summary": "2 or 3 sentences in Vietnamese: what the video is about and its tone",
  "speaker": "male" | "female" | "unknown",
  "address": "how the speaker should address the audience in Vietnamese, e.g. bạn, các bạn, mọi người",
  "terms": [{{"zh": "内啡肽", "vi": "endorphin"}}],
  "asr_fixes": [{{"wrong": "内非胎", "right": "内啡肽"}}]
}}

- "speaker": only from clear evidence (the speaker names herself or himself, says 我是女生, is
  called 姐/哥, talks about her husband...); otherwise "unknown".
- "terms": names of people, books, places, brands and the key terms of the topic that recur or
  matter, at most {MAX_TERMS}. "zh" is exactly as it should be written in Chinese, "vi" is how the
  Vietnamese voice-over should say it: Chinese personal names in their Sino-Vietnamese (Hán Việt)
  reading, foreign names and brands in their original spelling, book titles by their usual
  Vietnamese title, technical terms by the usual Vietnamese word.
- "asr_fixes": only words the speech recognition CLEARLY got wrong, where the context (and the
  title) shows what was said: "wrong" exactly as it appears in the transcript (2 or more Chinese
  characters), "right" the intended word. Leave out anything you are not sure about. At most {MAX_FIXES}.
- Return ONLY the JSON object.

{about}

Transcript:
{transcript_text(segments)}
"""


def _json_object(text):
    text = re.sub(r"```[a-zA-Z]*", "", text or "")
    text = re.sub(r"<(think|thought)>.*?</\1>", "", text, flags=re.S)
    first, last = text.find("{"), text.rfind("}")
    if first == -1 or last <= first:
        raise ValueError("no JSON object in the answer")
    return json.loads(text[first:last + 1])


def parse_brief(answer, segments):
    """The usable part of the model's answer; ValueError when there is no JSON object at all."""
    data = _json_object(answer)
    if not isinstance(data, dict):
        raise ValueError("the answer is not a JSON object")
    transcript = "\n".join(segment["source_text"] for segment in segments)
    terms, fixes = [], []
    for item in data.get("terms") or []:
        if isinstance(item, dict):
            zh, vi = str(item.get("zh") or "").strip(), str(item.get("vi") or "").strip()
            if zh and vi and CJK.search(zh) and not CJK.search(vi) and len(terms) < MAX_TERMS:
                terms.append({"zh": zh, "vi": vi})
    for item in data.get("asr_fixes") or []:
        if isinstance(item, dict):
            wrong, right = str(item.get("wrong") or "").strip(), str(item.get("right") or "").strip()
            # Only a real Chinese word that is in the transcript, replaced by another Chinese word.
            if (len(wrong) >= 2 and right and wrong != right and wrong in transcript
                    and CJK.search(wrong) and CJK.search(right) and len(fixes) < MAX_FIXES):
                fixes.append({"wrong": wrong, "right": right})
    speaker = str(data.get("speaker") or "").strip().lower()
    return {
        "summary": " ".join(str(data.get("summary") or "").split())[:600],
        "speaker": speaker if speaker in SPEAKERS else "",
        "address": " ".join(str(data.get("address") or "").split())[:80],
        "terms": terms,
        "asr_fixes": fixes,
    }


def brief_key(info, segments, model_name):
    request = {"info": info, "transcript": [s["source_text"] for s in segments],
               "model": model_name, "version": BRIEF_VERSION}
    return hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def make_brief(info, segments, generate, model_name, path=BRIEF_FILE):
    """The saved brief when its inputs did not change, else a new one; {} when it cannot be made."""
    key = brief_key(info, segments, model_name)
    saved = read_json(path, {})
    if isinstance(saved, dict) and saved.get("key") == key and isinstance(saved.get("brief"), dict):
        print("Using the saved video brief")
        return saved["brief"]
    prompt = build_brief_prompt(info, segments)
    for attempt in (1, 2):
        try:
            brief = parse_brief(generate(prompt if attempt == 1 else prompt + "\n\nAnswer with ONLY the JSON object."),
                                segments)
            break
        except ValueError as error:
            print(f"Video brief not usable ({error}). Attempt {attempt}/2")
    else:
        return {}
    path.write_text(json.dumps({"key": key, "brief": brief}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return brief


def apply_asr_fixes(segments, fixes):
    """(segments with the recognition errors replaced, number of segments changed). Input untouched."""
    result, changed = [], 0
    for segment in segments:
        text = segment["source_text"]
        for fix in fixes:
            text = text.replace(fix["wrong"], fix["right"])
        changed += text != segment["source_text"]
        result.append({**segment, "source_text": text})
    return result, changed


# ---------------------------------------------------------
# What every translation / review prompt gets
# ---------------------------------------------------------

SPEAKER_RULES = {
    "female": ("The speaker is a woman. She calls herself \"tôi\" (or \"mình\" in a casual video); "
               "when someone talks about or to her, use female forms (\"cô ấy\", \"chị\", \"bà\" for an older "
               "woman), never \"anh\" or \"ông\"."),
    "male": ("The speaker is a man. He calls himself \"tôi\" (or \"mình\" in a casual video); when "
             "someone talks about or to him, use male forms (\"anh ấy\", \"anh\", \"ông\" for an older man)."),
}


def speaker_of(info, brief):
    """The speaker chosen in dub.py wins over the one the brief heard."""
    return info.get("speaker") or (brief or {}).get("speaker") or ""


def context_block(info, brief, chunk_text):
    """'About this video' lines for a prompt: title, summary, speaker, address and the terms in this passage."""
    brief = brief or {}
    lines = []
    if info.get("title"):
        lines.append(f"Title: {info['title']}")
    if brief.get("summary"):
        lines.append(f"Topic: {brief['summary']}")
    speaker = speaker_of(info, brief)
    if speaker:
        lines.append(SPEAKER_RULES[speaker])
    if brief.get("address"):
        lines.append(f"The speaker addresses the audience as \"{brief['address']}\".")
    terms = [term for term in brief.get("terms", []) if term["zh"] in chunk_text]
    if terms:
        lines.append("Names and terms in this passage, write them exactly like this: "
                     + "; ".join(f"{term['zh']} = {term['vi']}" for term in terms))
    if not lines:
        return ""
    return "About this video (for understanding only, do not translate this):\n" + "\n".join(lines) + "\n\n"


def calibrated_words_per_second(path=None):
    """The speech rate speech.py measured on earlier jobs (median of the saved voices), else the default."""
    path = path or WORKDIR.parent / "tts_calibration.json"
    saved = read_json(path, {})
    rates = sorted(value for value in (saved.values() if isinstance(saved, dict) else [])
                   if isinstance(value, (int, float)) and 1.5 < value < 9)
    return rates[len(rates) // 2] if rates else DEFAULT_WORDS_PER_SECOND


def duration_rule(chunk, words_per_second):
    """The time the passage takes, and the Vietnamese length that fits it when read aloud."""
    seconds = max(0.0, chunk[-1]["end"] - chunk[0]["start"])
    if seconds < 3:
        return ""
    budget = int(seconds * words_per_second)
    return (f"- Timing: this passage is {seconds:.0f} seconds of speech and the Vietnamese voice reads about "
            f"{words_per_second:.1f} words per second, so the whole Vietnamese passage should be about "
            f"{budget} words or fewer. Say it concisely the way a Vietnamese speaker would (drop filler "
            "and repetition, never drop a fact) so it fits without being cut later.\n")


# ---------------------------------------------------------
# Checks
# ---------------------------------------------------------

def run_local_checks():
    import tempfile
    from pathlib import Path

    data = {"title": " 每个人都有病 ", "description": "x" * 2000, "uploader": "某频道"}
    info = info_from_ytdlp(data, "female")
    assert info == {"title": "每个人都有病", "description": "x" * MAX_DESCRIPTION_CHARS,
                    "channel": "某频道", "speaker": "female"}
    assert info_from_ytdlp(data, None, {"speaker": "male"})["speaker"] == "male"  # kept
    assert info_from_ytdlp(data, "auto", {"speaker": "male"})["speaker"] == ""   # auto clears it
    assert info_from_ytdlp(data, "robot")["speaker"] == ""

    segments = [{"start": 0.0, "end": 4.0, "source_text": "神经地质不平衡"},
                {"start": 4.0, "end": 9.0, "source_text": "内非胎给你提供愉悦感"},
                {"start": 9.0, "end": 25.0, "source_text": "吃掉那只青蛙"}]
    answer = """```json
    {"summary": "Video nói về trì hoãn.", "speaker": "Female", "address": "các bạn",
     "terms": [{"zh": "吃掉那只青蛙", "vi": "Ăn con ếch đó"}, {"zh": "x", "vi": "y"}, {"zh": "内啡肽", "vi": "内啡肽"}],
     "asr_fixes": [{"wrong": "内非胎", "right": "内啡肽"}, {"wrong": "神经地质", "right": "神经递质"},
                   {"wrong": "不存在", "right": "存在"}, {"wrong": "胎", "right": "肽"}]}
    ```"""
    brief = parse_brief(answer, segments)
    assert brief["speaker"] == "female" and brief["address"] == "các bạn"
    assert brief["terms"] == [{"zh": "吃掉那只青蛙", "vi": "Ăn con ếch đó"}]  # no Chinese-only or Chinese-in-vi terms
    assert [fix["wrong"] for fix in brief["asr_fixes"]] == ["内非胎", "神经地质"]  # absent and 1-character fixes dropped
    fixed, count = apply_asr_fixes(segments, brief["asr_fixes"])
    assert count == 2 and fixed[1]["source_text"] == "内啡肽给你提供愉悦感" and segments[1]["source_text"].startswith("内非胎")
    try:
        parse_brief("Xin lỗi.", segments)
    except ValueError:
        pass
    else:
        raise AssertionError("an answer without JSON must be rejected")

    block = context_block(info, brief, "吃掉那只青蛙 很难")
    assert "Title: 每个人都有病" in block and "woman" in block and "Ăn con ếch đó" in block and "các bạn" in block
    assert "Ăn con ếch đó" not in context_block(info, brief, "别的")  # only the terms in this passage
    assert context_block({"speaker": ""}, {}, "x") == ""
    assert speaker_of({"speaker": "male"}, {"speaker": "female"}) == "male"
    assert speaker_of({"speaker": ""}, {"speaker": "female"}) == "female"

    assert "25 seconds" in duration_rule(segments, 4.0) and "100 words" in duration_rule(segments, 4.0)
    assert duration_rule([{"start": 0.0, "end": 2.0, "source_text": "a"}], 4.0) == ""  # too short to matter

    long = [{"start": i, "end": i + 1, "source_text": "字" * 100} for i in range(1000)]
    assert len(transcript_text(long, 5000)) <= 5100

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "video_brief.json"
        calls = []
        def generate(prompt):
            calls.append(prompt)
            return answer
        assert make_brief(info, segments, generate, "m", path) == brief
        assert make_brief(info, segments, generate, "m", path) == brief and len(calls) == 1  # saved
        make_brief(info, segments, generate, "other", path)
        assert len(calls) == 2  # another model: asked again
        assert make_brief(info, segments, lambda prompt: "không", "x", path) == {}  # unusable: no brief
        rates = Path(folder) / "tts_calibration.json"
        rates.write_text(json.dumps({"a|b": 4.6, "c|d": 3.9, "e|f": 4.2, "bad": "x"}))
        assert calibrated_words_per_second(rates) == 4.2
        assert calibrated_words_per_second(Path(folder) / "none.json") == DEFAULT_WORDS_PER_SECOND
        info_path = Path(folder) / "video_info.json"
        info_path.write_text(json.dumps({"title": "t", "speaker": "robot"}))
        assert load_info(info_path) == {"title": "t", "description": "", "channel": "", "speaker": ""}
        assert load_info(Path(folder) / "missing.json")["title"] == ""


if __name__ == "__main__":
    run_local_checks()
    print("video_brief local checks passed")
