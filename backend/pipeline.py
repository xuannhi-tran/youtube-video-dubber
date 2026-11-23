import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import cuda_dlls  # keep before faster_whisper: it adds pip's CUDA DLLs to the search path
from faster_whisper import WhisperModel

import llm
import glossary
from glossary import GLOSSARY, prompt_lines, relevant, whisper_hotwords
import video_brief
from workdir import WORKDIR

OUTPUT_FILE = WORKDIR / "translated_segments.json"
# Kept so an interrupted run does not redo Whisper or already-translated chunks.
RAW_FILE = WORKDIR / "transcript_raw.json"
CACHE_FILE = WORKDIR / "translate_cache.json"
# Grows after every chunk so speech.py can start reading before the whole video
# is translated: {"complete": bool, "segments": [...]}.
PARTIAL_FILE = WORKDIR / "translated_partial.json"
ANSWER_ATTEMPTS = 3  # fresh requests when the answer is not usable JSON
# Cache entry (not a chunk) listing the models that filled the cache, so their
# translations are reused after switching model; see find_cached().
MODELS_ENTRY = "_models"
# Part of every cache key: raise it when the translation prompt changes, so
# chunks translated with an older prompt are translated again.
PROMPT_VERSION = 7
# Same for the review pass (see review_chunk).
REVIEW_PROMPT_VERSION = 7
# Whisper is told the video title (first characters only: a long prompt can make it repeat itself).
WHISPER_TITLE_CHARS = 60
# A review that changes more lines than this share of a chunk (and more than 3) is rewriting rather
# than correcting; it is still used, but the log says so (see review_warning).
REVIEW_WARN_SHARE = 0.25
# Requests the review pass sends at once (REVIEW_WORKERS in .env).
DEFAULT_REVIEW_WORKERS = 4
# Seconds a chunk waits for its review, counted from when its draft is ready (REVIEW_TIMEOUT in
# .env, 0 = no limit). After that the draft is used, so a slow or failing reviewer never holds up
# the dub; the review is not saved, so a later run tries again.
DEFAULT_REVIEW_TIMEOUT = 150
# Reviews in a row that time out or fail before the review pass is switched off for this run.
REVIEW_GIVE_UP_AFTER = 3
# A corrected line may be at most this much longer than the draft: the dub has to fit the same time.
REVIEW_MAX_GROWTH = 1.3
# Translated segments from the end of the previous chunk shown as context.
CONTEXT_SEGMENTS = 8
# Chinese segments from the start of the next chunk shown as context (not translated).
FOLLOWING_SEGMENTS = 3

# Translation gets surrounding context rather than isolated Whisper segments.
TARGET_CHUNK_SECONDS = 20


# ---------------------------------------------------------
# 1. TRANSCRIBE
# ---------------------------------------------------------

def load_whisper():
    """Prefer the local copy: asking Hugging Face first can hang for minutes."""
    settings = {"device": "cuda", "compute_type": "float16"}
    try:
        return WhisperModel("large-v3", local_files_only=True, **settings)
    except OSError:  # not in the local cache yet
        print("Whisper large-v3 is not downloaded yet; downloading (about 3 GB)...")
        return WhisperModel("large-v3", **settings)


def transcribe_audio(audio_file, max_seconds, hotwords=None, initial_prompt=None):
    print("\n[1/3] Loading Whisper large-v3...")

    model = load_whisper()

    print("Transcribing..." + (f" (expecting: {hotwords})" if hotwords else "")
          + (f" (title: {initial_prompt})" if initial_prompt else ""))

    start_time = time.perf_counter()

    segments, info = model.transcribe(
        str(audio_file),
        language="zh",
        beam_size=5,
        vad_filter=True,
        hotwords=hotwords,
        # The title tells Whisper the topic's words (names, terms) before it hears them.
        initial_prompt=initial_prompt,
    )

    result = []

    for segment in segments:
        if max_seconds is not None and segment.start >= max_seconds:
            break

        result.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "source_text": segment.text.strip(),
        })

    elapsed = time.perf_counter() - start_time

    print(
        f"Whisper finished: {len(result)} segments "
        f"in {elapsed:.2f}s"
    )

    return result


def write_json_atomic(path, data):
    temporary_file = path.with_suffix(path.suffix + ".tmp")
    temporary_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # On Windows the replace fails while another process (speech.py) is reading
    # the file; that read takes milliseconds, so try again briefly.
    for attempt in range(50):
        try:
            temporary_file.replace(path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.1)


def whisper_prompt(title):
    """The start of the video title for Whisper, or None."""
    title = " ".join((title or "").split())[:WHISPER_TITLE_CHARS]
    return title or None


def load_or_transcribe(audio_file, max_seconds, title=None):
    """Whisper is the slow stage, so keep its output with the other job files."""
    hotwords = whisper_hotwords(GLOSSARY)
    prompt = whisper_prompt(title)
    stamp = {
        "audio": audio_file.name,
        "bytes": audio_file.stat().st_size,
        "max_seconds": max_seconds,
        "hotwords": hotwords,
    }
    if prompt:  # only then, so transcripts made before titles were used stay valid
        stamp["title"] = prompt
    if RAW_FILE.exists():
        saved = json.loads(RAW_FILE.read_text(encoding="utf-8"))
        old = saved.get("stamp")
        # A transcript made before the title was known is still kept (the brief's fixes cover it).
        if old == stamp or (isinstance(old, dict) and "title" not in old and old == {
                key: value for key, value in stamp.items() if key != "title"}):
            print(f"Using saved transcript ({len(saved['segments'])} segments)")
            return saved["segments"]
    segments = transcribe_audio(audio_file, max_seconds, hotwords, prompt)
    write_json_atomic(RAW_FILE, {"stamp": stamp, "segments": segments})
    return segments


# ---------------------------------------------------------
# 2. GROUP SEGMENTS INTO CONTEXT WINDOWS
# ---------------------------------------------------------

def create_chunks(segments):
    chunks = []
    current_chunk = []

    if not segments:
        return chunks

    chunk_start = segments[0]["start"]

    for segment in segments:
        current_chunk.append(segment)

        duration = segment["end"] - chunk_start

        if duration >= TARGET_CHUNK_SECONDS:
            chunks.append(current_chunk)
            current_chunk = []

            chunk_start = (
                segment["end"]
            )

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ---------------------------------------------------------
# 3. TRANSLATE ONE CONTEXT WINDOW
# ---------------------------------------------------------

TEXT_KEYS = ("text", "translation", "translated_text", "vietnamese", "vi")


def _loads_lenient(text):
    """json.loads, then again without trailing commas and with typographic quotes fixed."""
    try:
        return json.loads(text)
    except ValueError:
        pass
    repaired = re.sub(r",\s*([\]}])", r"\1", text)
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    return json.loads(repaired)


def _first_list(value):
    """A list as it is, or the first list inside an object like {"segments": [...]}."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for item in value.values():
            if isinstance(item, list):
                return item
    return None


def extract_json_list(text):
    """The JSON array in a model answer.

    Small local models often wrap it in Markdown fences or chatter, leave a
    trailing comma, wrap it in an object, or stop before the closing bracket;
    all of those are accepted here. Raises ValueError when nothing usable is left.
    """
    text = llm.strip_thinking(text)
    text = re.sub(r"```[a-zA-Z]*", "", text).strip()
    candidates = [text]
    first, last = text.find("["), text.rfind("]")
    if first != -1 and last > first:
        candidates.append(text[first:last + 1])
    for candidate in candidates:
        try:
            found = _first_list(_loads_lenient(candidate))
        except ValueError:
            continue
        if found is not None:
            return found
    # Last resort: the flat {...} objects one by one (e.g. an answer cut off before "]").
    objects = []
    for match in re.finditer(r"\{[^{}\[\]]*\}", text):
        try:
            objects.append(_loads_lenient(match.group(0)))
        except ValueError:
            continue
    if objects:
        return objects
    raise ValueError("no JSON array in the answer")


def normalize_translation(translated):
    """Accept plain strings and other text keys from small models: [{"text": ...}, ...]."""
    if not isinstance(translated, list):
        return translated
    result = []
    for item in translated:
        if isinstance(item, str):
            item = {"text": item}
        elif isinstance(item, dict) and not isinstance(item.get("text"), str):
            for key in TEXT_KEYS[1:]:
                if isinstance(item.get(key), str):
                    item = {**item, "text": item[key]}
                    break
        result.append(item)
    return result


# Chinese characters and CJK / full-width punctuation: the Vietnamese voice cannot read them.
CJK = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")


SENTENCE_END = (".", "?", "!", "…", "。", "？", "！")
MIN_SEGMENTS_FOR_SENTENCE_CHECK = 6


class NoSentenceEnds(ValueError):
    """A long passage without a single full stop: the dub cannot be cut into sentences."""


class ChineseLeft(ValueError):
    """The 'Vietnamese' still contains Chinese characters."""


def strip_cjk(text):
    return " ".join(CJK.sub(" ", text).split())


def check_translation(translated, expected):
    if not isinstance(translated, list) or len(translated) != expected:
        count = len(translated) if isinstance(translated, list) else "no list"
        raise ValueError(f"expected {expected} segments, got {count}")
    for item in translated:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError("a segment has no text")
    if expected >= MIN_SEGMENTS_FOR_SENTENCE_CHECK and not any(
        item["text"].rstrip().rstrip("\"'”’»)]}").endswith(SENTENCE_END) for item in translated
    ):
        raise NoSentenceEnds("no segment ends a sentence with . ? or !")
    for item in translated:
        if CJK.search(item["text"]):
            raise ChineseLeft(f"Chinese characters left in the translation: {CJK.search(item['text']).group(0)!r}")


def is_usable(translated, count):
    try:
        check_translation(translated, count)
    except ValueError:
        return False
    return True


def context_lines(context):
    """The end of the previous chunk, as 'Chinese => Vietnamese' lines."""
    return "\n".join(f"{source} => {translation}" for source, translation in context)


TRANSLATION_RULES = """- First understand the whole passage, then translate it as fluent, natural
  spoken Vietnamese (not word-for-word, not subtitle style).
- Write for a listener who hears each sentence once and cannot read it again: say
  the idea the way a Vietnamese speaker would explain it, so it is understood at once.
  When the speaker counts off cases or types ("第一种…", "第二种…", "一是…二是…"),
  say what the case IS in plain words ("Kiểu thứ hai là những người hay lo lắng, và
  cách này giúp họ bớt lo"), never a literal frame such as "Dạng thứ hai giải quyết
  nỗi lo của bạn" that leaves the listener guessing what is meant.
- Read each complete Vietnamese sentence back as a native speaker would say it.
  Remove duplicated meaning, repeated synonyms and stiff Chinese-derived phrasing;
  for example, say "sâu thẳm trong lòng" once instead of combining it with
  "trong tiềm thức" for the same idea.
- Then split your Vietnamese back over the same segments, in order: each
  segment holds the Vietnamese for its part. Inside a sentence you may move a
  word or two across a segment boundary so the Vietnamese word order is natural.
- Put . ? ! only where a sentence really ends, and ALWAYS put one there: a
  passage of this length contains several sentences, so several segments must
  end with . ? or !. A segment that continues a sentence ends with no
  punctuation or a comma, and the next segment starts in lowercase.
- Keep the speaker's meaning. The Chinese comes from speech recognition and may
  contain recognition errors: correct them only when the context clearly shows
  what was meant (a glossary term often is).
- Do not invent facts when the transcript is uncertain.
- Preserve the speaker's key information and use natural connecting words.
  Do not make the Vietnamese abrupt just to shorten it; a later speech-fitting
  step handles duration limits separately and may shorten lines if needed.
- Never translate word for word. Spoken Chinese has structures that sound foreign in
  Vietnamese: "我们" in front of a name (我们天才尼采) is only familiar speech, so write
  "thiên tài Nietzsche", never "của chúng ta"; avoid chains of "của" ("thiên tài của
  Nietzsche" becomes "thiên tài Nietzsche" or "Nietzsche, một thiên tài"); leave out
  pronouns and fillers Vietnamese does not need.
- Call a historical figure, a thinker or an older person "ông" or "bà" (or by name);
  "anh ấy" and "cô ấy" are for young people only.
- Choose the pronoun for 他/她 from WHO it is, never by habit: a child, a baby or a
  son or daughter (also a child someone may have in the future) is "con", "đứa con",
  "đứa trẻ" or "bé", never "ông ấy", "bà ấy" or "anh ấy"; "ông"/"bà" are only for a
  real older person or a historical figure.
- Forms of address: the speaker talks to the audience as "bạn" or "chúng ta".
  Never call a person, a group of people or a personality type "nó"; use their
  name, "người này", "những người này" or "họ". Refer to the same person the
  same way throughout.
- Write only what a voice-over can read: no parentheses, no quotation marks,
  no brackets, no symbols.
"""


def build_translation_prompt(chunk, context=(), brief=None, following=()):
    """`brief` = {"about": 'About this video' text, "timing": duration rule} (both optional).
    `following` = the Chinese of the next few segments, so a sentence that runs on is understood."""
    brief = brief or {}
    input_segments = [
        {
            "start": item["start"],
            "end": item["end"],
            "text": item["source_text"],
        }
        for item in chunk
    ]

    glossary_lines = prompt_lines(GLOSSARY, chunk_source_text(chunk))
    glossary_rule = (
        "- Glossary: use these terms exactly as written, but only where the speaker really says "
        "that word or name. Never put a glossary term in place of a different word, and write every "
        "other Chinese personal name in its Sino-Vietnamese (Hán Việt) reading; foreign names "
        "keep their original spelling:\n"
        f"{glossary_lines}\n"
        if glossary_lines else
        "- Write Chinese personal names in their Sino-Vietnamese (Hán Việt) reading; foreign names "
        "keep their original spelling.\n"
    )
    context_block = (
        "Previous part of the video, already translated (for continuity only, do "
        "NOT translate it again; keep its forms of address and its terms):\n"
        f"{context_lines(context)}\n\n"
        if context else ""
    )
    following_block = (
        "What the speaker says right after this passage (Chinese, for understanding only, do NOT "
        "translate it; it shows how the last sentence continues):\n"
        + "\n".join(following) + "\n\n"
        if following else ""
    )

    return f"""
You are the translation component of a video dubbing system. The Vietnamese
text is read aloud by a voice-over, so it must sound like natural speech.

The segments below are consecutive pieces of ONE passage of speech. Speech
recognition cut it into short pieces, so a sentence often runs over several
segments.

How to translate:
{TRANSLATION_RULES}{brief.get("timing", "")}{glossary_rule}- Preserve every start and end timestamp exactly.
- Return exactly the same number of segments as the input, in the same order.
- Return ONLY valid JSON.

Required format:

[
  {{
    "start": 0.0,
    "end": 0.0,
    "text": "Vietnamese translation"
  }}
]

{brief.get("about", "")}{context_block}Chinese transcript to translate:

{json.dumps(input_segments, ensure_ascii=False)}

{following_block}"""


def retry_note(problem, expected):
    """Told to the model after an unusable answer, so the next one is fixed."""
    return (
        f"\n\nYour previous answer could not be used ({problem}). Write every segment in Vietnamese "
        "only: translate or transliterate Chinese words, never leave Chinese characters, and end each "
        "sentence with . ? or ! (segments that continue a sentence end with a comma or nothing). Answer again with ONLY "
        f"a JSON array of exactly {expected} objects, one per input segment, each with "
        '"start", "end" and "text". No other text before or after the array.'
    )


def translate_chunk(chunk, model=None, context=(), generate=None, brief=None, following=()):
    """Vietnamese for every segment of `chunk`, as [{"start", "end", "text"}, ...].

    An unusable answer is asked again (telling the model what was wrong). If a
    chunk still fails, its two halves are translated separately: fewer segments
    are easier for a small local model to count.
    """
    generate = generate or (lambda text: llm.generate(text, model=model, role="translate"))
    prompt = build_translation_prompt(chunk, context, brief, following)

    last_problem = None
    for attempt in range(1, ANSWER_ATTEMPTS + 1):
        request = prompt if last_problem is None else prompt + retry_note(last_problem, len(chunk))
        answer = generate(request)
        try:
            translated = normalize_translation(extract_json_list(answer))
            check_translation(translated, len(chunk))
            return translated
        except ValueError as error:  # includes json.JSONDecodeError
            last_problem = error
            print(f"Bad answer from the model ({error}). Attempt {attempt}/{ANSWER_ATTEMPTS}")
    if len(chunk) == 1 and isinstance(last_problem, ChineseLeft):
        # One segment, still some Chinese after retries: drop those characters
        # (the voice would skip or garble them) rather than stop the whole video.
        print("Removing the Chinese characters the model left in this segment.")
        item = normalize_translation(extract_json_list(answer))[0]
        text = strip_cjk(item["text"])
        if text:
            return [{**item, "text": text}]
    if len(chunk) == 1:
        raise RuntimeError(f"The model never returned a usable translation: {last_problem}")
    middle = len(chunk) // 2
    print(f"Translating this chunk in two halves ({middle} + {len(chunk) - middle} segments)...")
    # The halves keep the video context, but their timing rule would describe the whole chunk.
    halves = {"about": (brief or {}).get("about", "")}
    ahead = ([item["source_text"] for item in chunk[middle:]] + list(following))[:FOLLOWING_SEGMENTS]
    first = translate_chunk(chunk[:middle], model, context, generate, halves, ahead)
    joined = list(context) + [(source["source_text"], item["text"]) for source, item in zip(chunk[:middle], first)]
    second = translate_chunk(chunk[middle:], model, joined[-CONTEXT_SEGMENTS:], generate, halves, following)
    return first + second



# ---------------------------------------------------------
# REVIEW (optional second pass)
# ---------------------------------------------------------
# A second model reads the Chinese next to the draft and returns ONLY the lines that
# are wrong or sound like a word-for-word translation. Off unless REVIEW_LLM_PROVIDER is
# set in .env; the draft stays whenever the review fails.

def review_workers():
    value = os.getenv("REVIEW_WORKERS")
    if not value:
        return DEFAULT_REVIEW_WORKERS
    if not value.isdigit() or not 1 <= int(value) <= 16:
        raise ValueError(f"REVIEW_WORKERS must be a whole number from 1 to 16, got {value!r}")
    return int(value)


def review_timeout():
    """REVIEW_TIMEOUT in seconds, or None for no limit."""
    value = (os.getenv("REVIEW_TIMEOUT") or "").strip()
    if not value:
        return DEFAULT_REVIEW_TIMEOUT
    if not value.isdigit():
        raise ValueError(f"REVIEW_TIMEOUT must be a whole number of seconds (0 = no limit), got {value!r}")
    return int(value) or None


class ReviewTracker:
    """Decides what happens to each review: use it, or give up and keep the draft.

    A review that fails (server error, quota) or is not back `timeout` seconds after it was
    submitted is dropped and the draft is used. After `give_up_after` such drops in a row, or
    at once on a quota stop, the review pass is switched off for the rest of the run, so a
    reviewer that is down costs a few minutes at most instead of stopping the job.
    """

    def __init__(self, timeout, give_up_after=REVIEW_GIVE_UP_AFTER, clock=time.monotonic):
        self.timeout, self.give_up_after, self.clock = timeout, give_up_after, clock
        self.failures_in_row = 0
        self.off = False
        self.kept_drafts = 0

    def overdue(self, submitted):
        return self.timeout is not None and self.clock() - submitted > self.timeout

    def failed(self, quota=False):
        self.kept_drafts += 1
        self.failures_in_row += 1
        if quota or self.failures_in_row >= self.give_up_after:
            self.off = True

    def succeeded(self):
        self.failures_in_row = 0


def review_warning(changed, count):
    """A note for the log when a review changed suspiciously many lines, else ''."""
    if changed > 3 and changed > count * REVIEW_WARN_SHARE:
        return (f" (many: the reviewer rewrote {changed * 100 // count}% of this chunk; "
                "if this happens often, it is polishing rather than fixing errors)")
    return ""


def build_review_prompt(chunk, draft, brief=None):
    about = (brief or {}).get("about", "")
    lines = "\n".join(
        f"{number}. {source['source_text']}  =>  {item['text']}"
        for number, (source, item) in enumerate(zip(chunk, draft), start=1)
    )
    glossary_lines = prompt_lines(GLOSSARY, chunk_source_text(chunk))
    glossary_rule = (
        f"- Glossary, keep these terms exactly as written where they appear:\n{glossary_lines}\n"
        if glossary_lines else ""
    )
    return f"""
You are the Vietnamese-language editor of a Chinese to Vietnamese video dub. Below are numbered lines:
the Chinese original, then "=>", then a Vietnamese draft that a voice-over will read.
The lines are consecutive pieces of ONE passage of speech, so a sentence can run over several lines.
Your main goal is idiomatic, fluent Vietnamese that sounds natural when spoken aloud.
A draft can be understandable and still sound translated or awkward; correct those lines.

Change a line when it has one of these problems:
- the meaning is wrong, missing or invented compared with the Chinese;
- it reads like a word-for-word translation: 我们 + a name is just the name ("thiên tài
  Nietzsche", never "của chúng ta"), chains of "của", a Chinese word order;
- it is noticeably unnatural in Vietnamese: awkward word order or word choice,
  duplicated meaning, repeated synonyms, a stiff expression or a phrase a native
  Vietnamese speaker would not use in conversation;
- a SPECIFIC person (named, or clearly the same person as in the lines around it) is called
  "nó" or "anh ấy" where "ông", "bà", the name or "họ" is right (historical figures, thinkers
  and older people are "ông" or "bà");
- a child, a baby or a son or daughter (also a future child) is called "ông ấy", "bà ấy" or
  "anh ấy": use "con", "đứa con", "đứa trẻ" or "bé";
- it is said to the wrong person: who says what to whom must match the Chinese
  (the speaker "tôi", the audience "bạn"/"chúng ta", other people "họ");
- a Chinese name or term is written inconsistently;
- a listener hearing it once would not understand it at once: a literal frame like
  "Dạng thứ hai giải quyết nỗi lo của bạn" for "第二种…" must say plainly what the case is
  ("Kiểu thứ hai là ..."), using what the Chinese says about it.
Leave a line alone when it is already idiomatic and natural. Do not make cosmetic
paraphrases for taste alone, but do fix a noticeably awkward line even if its meaning
is understandable. Make as many corrections as the passage needs.
Do NOT add a pronoun or a person ("ông", "bà", "anh", "chị", "họ") that the Chinese line
does not have: a general statement about "a person", "someone" or "people" stays general.
Do NOT replace the key words of the topic with milder or vaguer ones (if the speaker says
"bệnh", keep "bệnh").
Do NOT drop a detail the Chinese line has: when you make a line more natural, keep every
idea, image and qualifier of the draft that is in the Chinese (盲选 keeps "nhắm mắt", 山洪 stays
"lũ", not a vaguer word).
Do NOT change the kind of sentence: a statement stays a statement (never add "liệu",
"phải chăng" or a question mark), and a question stays a question.
Leave every line you do not change out of your answer.

Rules for a corrected line:
- idiomatic spoken Vietnamese, concise and easy to read aloud (never more than a third longer);
- it keeps the line's place in the sentence: same split between lines, same . ? ! or comma at the end;
- no Chinese characters, parentheses, quotation marks or symbols.
{glossary_rule}
Return ONLY a JSON array of the lines you fix, for example
[{{"n": 3, "text": "corrected Vietnamese"}}]
and [] if no line needs a change.

{about}{lines}
"""


def parse_review(answer, count):
    """{line number (1-based): corrected text} from the model's answer; ValueError if it is not a list."""
    corrections = {}
    for item in extract_json_list(answer):
        if not isinstance(item, dict):
            continue
        number = item.get("n", item.get("line"))
        if isinstance(number, str) and number.strip().isdigit():
            number = int(number)
        text = next((item[key] for key in TEXT_KEYS if isinstance(item.get(key), str)), None)
        if isinstance(number, int) and 1 <= number <= count and text and text.strip():
            corrections[number] = " ".join(text.split())
    return corrections


def apply_review(draft, corrections, skipped=None):
    """(reviewed translation, number of lines changed). A correction that is unusable is skipped;
    when `skipped` is a list, "line N: reason" is added to it for each one."""
    skipped = [] if skipped is None else skipped
    result = [dict(item) for item in draft]
    changed = 0
    for number, text in sorted(corrections.items()):
        old = result[number - 1]["text"]
        limit = max(len(old.split()) * REVIEW_MAX_GROWTH, len(old.split()) + 3)
        if CJK.search(text):
            skipped.append(f"line {number}: has Chinese characters")
        elif text == old:
            skipped.append(f"line {number}: same as the draft")
        elif len(text.split()) > limit:
            skipped.append(f"line {number}: too long ({len(text.split())} words, draft {len(old.split())})")
        else:
            result[number - 1]["text"] = text
            changed += 1
    if changed:
        try:
            check_translation(result, len(draft))
        except ValueError as error:
            skipped.append(f"all fixes dropped: {error}")
            return [dict(item) for item in draft], 0
    return result, changed


def review_chunk(chunk, draft, generate=None, brief=None):
    """The draft with the model's corrections; the draft itself when the answer is unusable."""
    generate = generate or (lambda text: llm.generate(text, role="review"))
    prompt = build_review_prompt(chunk, draft, brief)
    for attempt in (1, 2):
        request = prompt if attempt == 1 else prompt + (
            "\n\nYour previous answer could not be read. Answer with ONLY a JSON array like "
            '[{"n": 3, "text": "..."}], or [].')
        try:
            answer = generate(request)
            corrections, skipped = parse_review(answer, len(draft)), []
            result, changed = apply_review(draft, corrections, skipped)
            if not changed:  # nothing to show for this chunk: print what the model said, to tell "all fine" from "ignored"
                print(f"  review: the model proposed {len(corrections)} fixes, none used"
                      + (f" ({'; '.join(skipped)})" if skipped else "")
                      + f". Its answer: {' '.join(str(answer).split())[:300]!r}", flush=True)
            return result
        except ValueError as error:
            print(f"Review answer not usable ({error}). Attempt {attempt}/2")
    print("Keeping the draft translation for this chunk.")
    return draft


def review_key(chunk, draft, model_name, brief=None):
    request = {
        "review": model_name,
        "chunk": chunk,
        "draft": [item["text"] for item in draft],
        "glossary": relevant(GLOSSARY, chunk_source_text(chunk)),
        "prompt": REVIEW_PROMPT_VERSION,
    }
    if (brief or {}).get("about"):  # only then, so a job without video context keeps its saved reviews
        request["about"] = brief["about"]
    return hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def chunk_source_text(chunk):
    return " ".join(item["source_text"] for item in chunk)


def chunk_key(chunk, model_name=None, brief=None):
    """Identifies a chunk's translation request, so a changed input is re-translated.

    Only the glossary entries that apply to this chunk are part of the key, so
    adding a name that is not in a chunk does not translate that chunk again.
    The video context and timing rule are part of it when there are any.
    """
    request = {
        "chunk": chunk,
        "glossary": relevant(GLOSSARY, chunk_source_text(chunk)),
        "model": model_name or llm.describe(role="translate"),
        "prompt": PROMPT_VERSION,
    }
    for part in ("about", "timing"):
        if (brief or {}).get(part):
            request[part] = brief[part]
    encoded = json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def earlier_models(cache, current):
    """Models whose translations may be in the cache, besides `current`.

    Caches written before MODELS_ENTRY existed were filled by the default
    Groq or Gemini model, so those are always tried too.
    """
    names = cache.get(MODELS_ENTRY, [])
    names = names if isinstance(names, list) else []
    defaults = [f"{provider}:{llm.DEFAULT_MODELS[provider]}" for provider in ("groq", "gemini")]
    return [name for name in dict.fromkeys(names + defaults) if isinstance(name, str) and name != current]


def find_cached(cache, chunk, current, reuse_other_models=True, brief=None):
    """(translation, model name) of a saved translation of `chunk`, or (None, None).

    Prefers the current model. Unless told otherwise, a translation another
    model made of the same chunk (same glossary, same prompt version) is kept:
    switching from Groq to a local model must not throw away finished work.
    """
    def usable(translated):
        return is_usable(translated, len(chunk))  # one with Chinese left in it is translated again

    key = chunk_key(chunk, current, brief)
    if key in cache and usable(cache[key]):
        return cache[key], current
    if reuse_other_models:
        for name in earlier_models(cache, current):
            other = chunk_key(chunk, name, brief)
            if other in cache and usable(cache[other]):
                return cache[other], name
    return None, None


def remember_model(cache, name):
    names = cache.get(MODELS_ENTRY)
    names = names if isinstance(names, list) else []
    if name not in names:
        cache[MODELS_ENTRY] = names + [name]


def load_cache():
    if not CACHE_FILE.exists():
        return {}
    try:
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        print("Invalid translation cache; translating everything again")
        return {}
    return cache if isinstance(cache, dict) else {}


# ---------------------------------------------------------
# WHOLE-VIDEO TRANSLATION (optional, one request per video)
# ---------------------------------------------------------
# A model with a tiny daily request quota but a huge token budget (Gemini 3.8 Flash: 20 requests a
# day) translates the whole transcript in ONE request, as JSON keyed by segment number. Every number
# must come back; missing ones are asked again (only those). Chunks it could not deliver in a usable
# form, or all of them when the model is out of quota, go through the normal chunked path.

WHOLE_MAX_TOKENS = 32000  # answer length allowed for one whole-video request
WHOLE_BATCH_SEGMENTS = 400  # a longer transcript is sent in several consecutive requests
WHOLE_RETRY_ROUNDS = 2  # extra requests per batch for numbers that did not come back


def whole_settings():
    """(provider, model) of TRANSLATE_WHOLE_MODEL when TRANSLATE_WHOLE=1, else None."""
    if (os.getenv("TRANSLATE_WHOLE") or "0").strip().lower() not in ("1", "yes", "true", "on"):
        return None
    provider = (os.getenv("TRANSLATE_WHOLE_PROVIDER") or "gemini").strip().lower()
    model = (os.getenv("TRANSLATE_WHOLE_MODEL") or "").strip()
    if not model:
        raise ValueError("TRANSLATE_WHOLE=1 needs TRANSLATE_WHOLE_MODEL (see: python llm.py --list gemini)")
    llm.settings(provider, model)  # a missing key stops here
    return provider, model


def whole_batch_size():
    value = (os.getenv("TRANSLATE_WHOLE_BATCH") or "").strip()
    if not value:
        return WHOLE_BATCH_SEGMENTS
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"TRANSLATE_WHOLE_BATCH must be a whole number of segments, got {value!r}")
    return int(value)


def build_whole_prompt(items, wanted, about="", context=()):
    """`items` = [(number, text)] of one batch; `wanted` = the numbers to translate now (the rest is context)."""
    source = "\n".join(f"{number}: {text}" for number, text in items)
    glossary_lines = prompt_lines(GLOSSARY, " ".join(text for _, text in items))
    glossary_rule = (
        "- Glossary: use these terms exactly as written, but only where the speaker really says "
        "that word or name. Write every other Chinese personal name in its Sino-Vietnamese (Hán Việt) "
        f"reading; foreign names keep their original spelling:\n{glossary_lines}\n"
        if glossary_lines else
        "- Write Chinese personal names in their Sino-Vietnamese (Hán Việt) reading; foreign names "
        "keep their original spelling.\n"
    )
    previous = (
        "The part of the video before this transcript, already translated (continuity only, do NOT "
        f"translate it again):\n{context_lines(context)}\n\n" if context else ""
    )
    only = ""
    if len(wanted) != len(items):
        only = (f"- Translate ONLY the segments numbered {', '.join(map(str, wanted))}; the other "
                "segments are there for context and already done.\n")
    return f"""
You are the translation component of a video dubbing system. The Vietnamese
text is read aloud by a voice-over, so it must sound like natural speech.

Below is the transcript of one video, cut by speech recognition into numbered
segments. A sentence often runs over several segments. Read the whole transcript
first, then translate it.

How to translate:
{TRANSLATION_RULES}{glossary_rule}{only}- Answer with ONLY one JSON object: every segment number is a key and its
  Vietnamese text is the value, for example {{"1": "...", "2": "..."}}. Include every
  segment you were asked for, each exactly once, with the number it has below.

{about}{previous}Chinese transcript (number: text):

{source}
"""


def parse_whole_answer(text):
    """{segment number: Vietnamese text} from a model answer; a cut-off answer still gives its complete entries."""
    text = re.sub(r"```[a-zA-Z]*", "", llm.strip_thinking(text)).strip()
    found = {}
    first, last = text.find("{"), text.rfind("}")
    try:
        value = _loads_lenient(text[first:last + 1]) if first != -1 and last > first else None
    except ValueError:
        value = None
    if isinstance(value, dict):
        found = {str(key).strip(): item for key, item in value.items()}
    else:
        for match in re.finditer(r'"(\d+)"\s*:\s*("(?:[^"\\]|\\.)*")', text):
            try:
                found[match.group(1)] = json.loads(match.group(2))
            except ValueError:
                continue
    return {key: item.strip() for key, item in found.items()
            if isinstance(item, str) and item.strip() and key.isdigit()}


def translate_whole(texts, generate, about="", batch_size=WHOLE_BATCH_SEGMENTS):
    """(done, error): {number: Vietnamese} for the transcript `texts` (numbers start at 1), and the
    exception that stopped it early (quota, server) or None. What was translated before an error is kept.

    One request per batch, then up to WHOLE_RETRY_ROUNDS requests for the numbers still missing.
    """
    done = {}
    try:
        for begin in range(0, len(texts), batch_size):
            numbers = range(begin + 1, min(begin + batch_size, len(texts)) + 1)
            items = [(number, texts[number - 1]) for number in numbers]
            context = [(texts[number - 1], done[number]) for number in range(begin - CONTEXT_SEGMENTS + 1, begin + 1)
                       if number in done]
            for attempt in range(1 + WHOLE_RETRY_ROUNDS):
                wanted = [number for number in numbers if number not in done]
                if not wanted:
                    break
                if attempt:
                    print(f"{len(wanted)} segments missing from the answer; asking for them again "
                          f"({attempt}/{WHOLE_RETRY_ROUNDS})", flush=True)
                answer = parse_whole_answer(generate(build_whole_prompt(items, wanted, about, context)))
                for number in wanted:
                    text = answer.get(str(number))
                    if text and not CJK.search(text):
                        done[number] = text
    except (llm.QuotaExhausted, llm.LLMError, RuntimeError) as error:
        return done, error
    return done, None


def whole_chunks(chunks, done):
    """{chunk index: [{"start","end","text"}]} for the chunks `done` translated completely and usably."""
    result = {}
    number = 0
    for index, chunk in enumerate(chunks):
        numbers = range(number + 1, number + len(chunk) + 1)
        number += len(chunk)
        if not all(item in done for item in numbers):
            continue
        translated = [{"start": source["start"], "end": source["end"], "text": done[item]}
                      for source, item in zip(chunk, numbers)]
        if is_usable(translated, len(chunk)):
            result[index] = translated
    return result


def run_whole_translation(chunks, about, provider, model):
    """The chunks' translations from the whole-video request(s); {} for chunks it could not deliver."""
    flat = [source["source_text"] for chunk in chunks for source in chunk]
    print(f"\nTranslating the whole video ({len(flat)} segments) in one request with {provider}:{model}...", flush=True)
    done, error = translate_whole(
        flat,
        lambda prompt: llm.generate(prompt, provider=provider, model=model, max_tokens=WHOLE_MAX_TOKENS),
        about, whole_batch_size(),
    )
    if error:
        print(f"Whole-video translation stopped ({str(error)[:150]}); the chunks it did not "
              "deliver are translated chunk by chunk.", flush=True)
    result = whole_chunks(chunks, done)
    print(f"Whole-video translation delivered {len(result)} of {len(chunks)} chunks.", flush=True)
    return result


def run_local_checks():
    glossary.run_local_checks()
    fenced = '<think>x</think>Here you go:\n```json\n[{"start": 0, "end": 1, "text": "Chào",},]\n```'
    assert extract_json_list(fenced) == [{"start": 0, "end": 1, "text": "Chào"}]
    assert extract_json_list('{"segments": [{"text": "a"}]}') == [{"text": "a"}]
    assert extract_json_list('[{"text": "a"}, {"text": "b"}') == [{"text": "a"}, {"text": "b"}]  # cut off
    assert extract_json_list('Kết quả [ghi chú] như sau: [{"text": "a"}]') == [{"text": "a"}]
    try:
        extract_json_list("Xin lỗi, tôi không thể dịch.")
    except ValueError:
        pass
    else:
        raise AssertionError("an answer without JSON must be rejected")
    try:
        check_translation([{"text": "đừng quá用力"}], 1)
    except ChineseLeft:
        pass
    else:
        raise AssertionError("Chinese characters in the translation must be rejected")
    check_translation([{"text": "Huang Xiaoming và Wu Tianle, 100%"}], 1)
    flat = [{"text": f"đoạn {i}"} for i in range(6)]
    try:
        check_translation(flat, 6)
    except NoSentenceEnds:
        pass
    else:
        raise AssertionError("a long passage with no sentence end must be rejected")
    check_translation(flat[:5], 5)  # short windows may have no full stop
    check_translation(flat[:5] + [{"text": 'Anh ấy nói "xong rồi."'}], 6)
    assert strip_cjk("đừng quá用力 nhé。") == "đừng quá nhé"
    assert normalize_translation(["a", {"translation": "b"}]) == [{"text": "a"}, {"translation": "b", "text": "b"}]

    chunk = [{"start": float(i), "end": i + 1.0, "source_text": f"句{i}"} for i in range(3)]
    def answer(texts):
        return json.dumps([{"start": 0, "end": 0, "text": text} for text in texts], ensure_ascii=False)

    asked = []
    replies = iter(["không phải JSON", answer(["a", "b", "c"])])
    def generate(prompt):
        asked.append(prompt)
        return next(replies)
    assert [item["text"] for item in translate_chunk(chunk, generate=generate)] == ["a", "b", "c"]
    assert "exactly 3 objects" in asked[1] and "exactly 3" not in asked[0]

    # Always the wrong count for 3 segments: split into 1 + 2.
    def by_size(prompt):
        size = prompt.split("Chinese transcript to translate:")[1].count('"text"')
        return answer(["x"] * (size + 1)) if size == 3 else answer([f"t{size}"] * size)
    result = translate_chunk(chunk, generate=by_size)
    assert [item["text"] for item in result] == ["t1", "t2", "t2"]

    # Chinese left in an answer is asked again; if it never goes away in a single segment, it is removed.
    one = chunk[:1]
    replies = iter([answer(["quá用力"]), answer(["quá nhiều"])])
    assert translate_chunk(one, generate=lambda prompt: next(replies))[0]["text"] == "quá nhiều"
    assert translate_chunk(one, generate=lambda prompt: answer(["quá用力"]))[0]["text"] == "quá"

    # Glossary pairs: only the entries whose source occurs in the passage reach the prompt and the cache key.
    global GLOSSARY
    saved_glossary = GLOSSARY
    try:
        GLOSSARY = ["黄晓明 => Hoàng Hiểu Minh", "态度 => thái độ", "khí chất"]
        named = [{"start": 0.0, "end": 1.0, "source_text": "黄晓明很帅"}]
        other = [{"start": 0.0, "end": 1.0, "source_text": "没有名字"}]
        prompt = build_translation_prompt(named)
        assert '"黄晓明" must be written as "Hoàng Hiểu Minh"' in prompt and "thái độ" not in prompt
        assert "khí chất" in prompt and "Hán Việt" in prompt
        assert "thái độ" not in build_translation_prompt(other) and "Hoàng" not in build_translation_prompt(other)
        key_before = chunk_key(other, "x:y")
        GLOSSARY = GLOSSARY + ["吴亦凡 => Ngô Diệc Phàm"]
        assert chunk_key(other, "x:y") == key_before  # a name that is not in the chunk changes nothing
        assert chunk_key(named, "x:y") != chunk_key(other, "x:y")
        GLOSSARY = []
        assert "Glossary" not in build_translation_prompt(named) and "Hán Việt" in build_translation_prompt(named)
    finally:
        GLOSSARY = saved_glossary

    # Video context and timing reach the prompts and the cache keys only when there are any.
    extra = {"about": "About this video (for understanding only, do not translate this):\nTitle: X\n\n",
             "timing": "- Timing: this passage is 20 seconds of speech...\n"}
    prompt = build_translation_prompt(chunk, brief=extra)
    assert "Title: X" in prompt and "- Timing:" in prompt
    assert prompt.index("Title: X") < prompt.index("Chinese transcript to translate:")
    assert "About this video" not in build_translation_prompt(chunk) and "Timing" not in build_translation_prompt(chunk)
    assert chunk_key(chunk, "x:y") == chunk_key(chunk, "x:y", {"about": "", "timing": ""})
    assert chunk_key(chunk, "x:y") != chunk_key(chunk, "x:y", extra)
    assert "Title: X" in build_review_prompt(chunk, [{"text": "a"}] * 3, extra)
    assert review_key(chunk, [{"text": "a"}] * 3, "m") != review_key(chunk, [{"text": "a"}] * 3, "m", extra)
    seen = []
    translate_chunk(chunk, generate=lambda p: seen.append(p) or answer(["a", "b", "c"]), brief=extra)
    assert "Title: X" in seen[0] and "- Timing:" in seen[0]
    assert "right after this passage" not in seen[0]
    seen.clear()
    translate_chunk(chunk, generate=lambda p: seen.append(p) or answer(["a", "b", "c"]), following=["下一句"])
    assert "下一句" in seen[0] and seen[0].index("Chinese transcript to translate:") < seen[0].index("下一句")
    assert whisper_prompt("  每个人  都有病 ") == "每个人 都有病" and whisper_prompt("") is None
    assert len(whisper_prompt("字" * 200)) == WHISPER_TITLE_CHARS

    current, old = "ollama:qwen3:8b", f"groq:{llm.DEFAULT_MODELS['groq']}"
    groq_text = [{"text": f"từ groq {i}"} for i in range(3)]
    qwen_text = [{"text": f"từ qwen {i}"} for i in range(3)]
    cache = {chunk_key(chunk, old): groq_text}
    assert find_cached(cache, chunk, current) == (groq_text, old)
    assert find_cached(cache, chunk, current, reuse_other_models=False) == (None, None)
    cache[chunk_key(chunk, current)] = qwen_text
    assert find_cached(cache, chunk, current) == (qwen_text, current)
    cache[chunk_key(chunk, current)] = [{"text": "quá用力"}] * 3
    assert find_cached(cache, chunk, current) == (groq_text, old)  # the one with Chinese is skipped
    remember_model(cache, current)
    remember_model(cache, current)
    assert cache[MODELS_ENTRY] == [current]
    other = [{"text": "z"}] * 3
    assert find_cached({chunk_key(chunk, "x:y"): other, MODELS_ENTRY: ["x:y"]}, chunk, current) == (other, "x:y")

    # Review pass: only the lines the model names change, and only when the fix is usable.
    draft = [{"text": "Thiên tài Nietzsche của chúng ta cũng từng nói"}, {"text": "mọi thứ do bạn."}, {"text": "Ông ấy rất giỏi."}]
    assert parse_review('[{"n": 1, "text": "Thiên tài Nietzsche cũng từng nói"}, {"n": 9, "text": "x"}, "bad"]', 3) == {
        1: "Thiên tài Nietzsche cũng từng nói"}
    assert parse_review("[]", 3) == {} and parse_review('[{"n": "2", "translation": "ổn"}]', 3) == {2: "ổn"}
    fixed, count = apply_review(draft, {1: "Thiên tài Nietzsche cũng từng nói", 2: "quá用力", 3: "Ông ấy rất giỏi."})
    assert count == 1 and fixed[0]["text"] == "Thiên tài Nietzsche cũng từng nói" and fixed[1] == draft[1]
    assert apply_review(draft, {3: "Ông ấy rất giỏi " + "nữa " * 9})[1] == 0  # far longer than the draft
    reasons = []
    apply_review(draft, {2: "quá用力", 3: "Ông ấy rất giỏi.", 1: "x " * 40}, reasons)
    assert len(reasons) == 3 and "Chinese" in reasons[1] and "same as" in reasons[2] and "too long" in reasons[0]
    sentences = [{"text": f"câu {i}."} for i in range(6)]
    assert apply_review(sentences, {number: "chưa hết câu" for number in range(1, 7)}) == (sentences, 0)  # no sentence end left
    assert draft[0]["text"].startswith("Thiên tài Nietzsche của")  # the draft itself is never edited
    review_source = [{"start": float(i), "end": i + 1.0, "source_text": f"句{i}"} for i in range(3)]
    assert "1. 句0  =>  Thiên tài" in build_review_prompt(review_source, draft)
    assert review_warning(2, 15) == "" and review_warning(3, 4) == ""  # a few fixes, or a tiny chunk
    assert "40%" in review_warning(6, 15)
    replies = iter(["không phải JSON", '[{"n": 1, "text": "Thiên tài Nietzsche cũng từng nói"}]'])
    assert review_chunk(review_source, draft, generate=lambda prompt: next(replies))[0]["text"] == "Thiên tài Nietzsche cũng từng nói"
    assert review_chunk(review_source, draft, generate=lambda prompt: "xin lỗi") == draft
    assert review_key(review_source, draft, "a:b") == review_key(review_source, draft, "a:b")
    assert review_key(review_source, draft, "a:b") != review_key(review_source, draft, "c:d")
    assert review_key(review_source, draft, "a:b") != review_key(review_source, draft[:2] + [{"text": "khác."}], "a:b")
    os.environ["REVIEW_WORKERS"] = "0"
    try:
        review_workers()
    except ValueError:
        pass
    else:
        raise AssertionError("REVIEW_WORKERS=0 must be rejected")
    del os.environ["REVIEW_WORKERS"]
    assert review_workers() == DEFAULT_REVIEW_WORKERS

    # Review timeout and giving up: a slow or failing reviewer never stops the dub.
    saved_timeout = os.environ.pop("REVIEW_TIMEOUT", None)  # .env may set it; put back below
    assert review_timeout() == DEFAULT_REVIEW_TIMEOUT
    os.environ["REVIEW_TIMEOUT"] = "0"
    assert review_timeout() is None
    os.environ["REVIEW_TIMEOUT"] = "soon"
    try:
        review_timeout()
    except ValueError:
        pass
    else:
        raise AssertionError("REVIEW_TIMEOUT=soon must be rejected")
    del os.environ["REVIEW_TIMEOUT"]
    if saved_timeout is not None:
        os.environ["REVIEW_TIMEOUT"] = saved_timeout
    now = [0.0]
    tracker = ReviewTracker(10, give_up_after=3, clock=lambda: now[0])
    assert not tracker.overdue(0.0)
    now[0] = 10.5
    assert tracker.overdue(0.0) and not ReviewTracker(None, clock=lambda: 1e9).overdue(0.0)
    tracker.failed(); tracker.failed(); tracker.succeeded(); tracker.failed()
    assert not tracker.off and tracker.kept_drafts == 3  # a success resets the count in a row
    tracker.failed(); tracker.failed()
    assert tracker.off
    quota = ReviewTracker(10)
    quota.failed(quota=True)
    assert quota.off  # out of quota: no point asking again in this run

    # Whole-video translation: one request, JSON keyed by number, only the missing numbers asked again.
    assert parse_whole_answer('```json\n{"1": "a", "2": "b",}\n```') == {"1": "a", "2": "b"}
    assert parse_whole_answer('{"1": "a", "2": "b", "3": "cut o') == {"1": "a", "2": "b"}  # cut off
    assert parse_whole_answer('{"1": "", "x": "y", "2": 5, "3": " ok "}') == {"3": "ok"}
    prompt = build_whole_prompt([(1, "句一"), (2, "句二")], [2], context=[("上", "trên")])
    assert "1: 句一" in prompt and "ONLY the segments numbered 2" in prompt and "上 => trên" in prompt
    assert "ONLY the segments numbered" not in build_whole_prompt([(1, "句一")], [1])
    texts = [f"句{i}" for i in range(1, 8)]
    prompts = []
    def whole_model(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:  # forgets 3 and leaves Chinese in 5
            return json.dumps({"1": "một.", "2": "hai", "4": "bốn.", "5": "năm 句", "6": "sáu", "7": "bảy."}, ensure_ascii=False)
        return json.dumps({"3": "ba", "5": "năm"})
    done, error = translate_whole(texts, whole_model)
    assert error is None and sorted(done) == [1, 2, 3, 4, 5, 6, 7] and len(prompts) == 2
    assert "ONLY the segments numbered 3, 5" in prompts[1] and "ONLY the segments numbered" not in prompts[0]
    # Segments that never come back are asked 1 + WHOLE_RETRY_ROUNDS times, then left to the chunked path.
    prompts.clear()
    done, error = translate_whole(texts, lambda prompt: prompts.append(prompt) or '{"1": "một"}')
    assert error is None and done == {1: "một"} and len(prompts) == 1 + WHOLE_RETRY_ROUNDS
    # A quota stop keeps what came before it; batches are consecutive and carry the previous lines.
    prompts.clear()
    def quota_after_first(prompt):
        prompts.append(prompt)
        if len(prompts) > 1:
            raise llm.QuotaExhausted("daily limit")
        return json.dumps({"1": "một", "2": "hai"})
    done, error = translate_whole(texts, quota_after_first, batch_size=2)
    assert done == {1: "một", 2: "hai"} and isinstance(error, llm.QuotaExhausted)
    assert "句2 => hai" in prompts[1] and "3: 句3" in prompts[1] and "5: 句5" not in prompts[1]
    two = [[{"start": 0.0, "end": 1.0, "source_text": "句1"}, {"start": 1.0, "end": 2.0, "source_text": "句2"}],
           [{"start": 2.0, "end": 3.0, "source_text": "句3"}]]
    assert whole_chunks(two, {1: "một.", 2: "hai.", 3: "ba."}).keys() == {0, 1}
    assert whole_chunks(two, {1: "một.", 2: "hai."}).keys() == {0}  # a chunk missing a number is left out
    saved_whole = {name: os.environ.pop(name, None) for name in ("TRANSLATE_WHOLE", "TRANSLATE_WHOLE_MODEL", "TRANSLATE_WHOLE_PROVIDER")}
    assert whole_settings() is None
    os.environ["TRANSLATE_WHOLE"] = "1"
    try:
        whole_settings()
    except ValueError:
        pass
    else:
        raise AssertionError("TRANSLATE_WHOLE=1 without a model must be rejected")
    del os.environ["TRANSLATE_WHOLE"]
    for name, value in saved_whole.items():
        if value is not None:
            os.environ[name] = value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe (Whisper) and translate (Groq, Gemini or a local LLM) one audio file."
    )
    parser.add_argument(
        "--audio", type=Path, default=WORKDIR / "xiaowulang_original.mp3",
        help="audio or video file to transcribe (default: xiaowulang_original.mp3)",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=None,
        help="only process the first N seconds (default: the whole file)",
    )
    parser.add_argument(
        "--retranslate", action="store_true",
        help="translate chunks again that another model already translated "
             "(default: keep them, translate only what is missing)",
    )
    args = parser.parse_args()
    if not args.audio.exists():
        parser.error(f"--audio not found: {args.audio}")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    return args


def timing_enabled():
    """TRANSLATE_TIMING=1 asks for a translation that fits the passage's time. Off by default: in a test
    it made the text 7% shorter but dropped details and flattened metaphors (盲选 lost "nhắm mắt",
    山洪 became "khó khăn"), which the speech stage's shortening does more carefully, line by line."""
    return (os.getenv("TRANSLATE_TIMING") or "0").strip().lower() in ("1", "yes", "true", "on")


def prepare_brief(info, segments, current):
    """(brief, segments with recognition errors fixed). One LLM call; a failure only costs the brief."""
    if not video_brief.enabled():
        return {}, segments
    print("\nReading the whole transcript for the video brief (topic, speaker, names, misheard words)...")
    try:
        brief = video_brief.make_brief(info, segments, lambda text: llm.generate(text, role="translate"), current)
    except (llm.QuotaExhausted, llm.LLMError, RuntimeError) as error:
        print(f"No video brief ({str(error)[:150]}); translating without it.")
        return {}, segments
    if not brief:
        print("No usable video brief; translating without it.")
        return {}, segments
    fixed, changed = video_brief.apply_asr_fixes(segments, brief.get("asr_fixes", []))
    speaker = video_brief.speaker_of(info, brief) or "unknown"
    print(f"Video brief: speaker {speaker}, {len(brief.get('terms', []))} names/terms, "
          f"{len(brief.get('asr_fixes', []))} misheard words fixed in {changed} segments")
    for fix in brief.get("asr_fixes", [])[:10]:
        print(f"  heard {fix['wrong']} -> {fix['right']}")
    return brief, fixed


def chunk_brief(info, brief, chunk, words_per_second):
    """The per-chunk prompt additions: the video context and the timing rule."""
    return {
        "about": video_brief.context_block(info, brief, chunk_source_text(chunk)),
        "timing": video_brief.duration_rule(chunk, words_per_second) if timing_enabled() else "",
    }


def main():
    run_local_checks()
    video_brief.run_local_checks()
    args = parse_args()
    info = video_brief.load_info()
    segments = load_or_transcribe(args.audio, args.max_seconds, info["title"])
    if not segments:
        raise SystemExit("Whisper found no speech in this audio.")

    current = llm.describe(role="translate")
    brief, segments = prepare_brief(info, segments, current)
    words_per_second = video_brief.calibrated_words_per_second()

    print("\n[2/3] Creating translation chunks...")

    chunks = create_chunks(segments)
    briefs = [chunk_brief(info, brief, chunk, words_per_second) for chunk in chunks]

    print(f"Created {len(chunks)} context chunks.")

    cache = load_cache()
    final_segments = []
    context = []
    write_json_atomic(PARTIAL_FILE, {"complete": False, "segments": []})

    reviewer = llm.describe(role="review") if llm.role_configured("review") else None
    pool = None
    tracker = None
    if reviewer:
        llm.settings(role="review")  # a missing key stops here, not after the first chunk
        pool = ThreadPoolExecutor(max_workers=review_workers())
        tracker = ReviewTracker(review_timeout())
    whole = whole_settings()
    whole_name = f"{whole[0]}:{whole[1]}" if whole else None
    whole_done = {}
    if whole and any(find_cached(cache, chunk, current, not args.retranslate, briefs[index])[0] is None
                     for index, chunk in enumerate(chunks)):
        all_text = " ".join(chunk_source_text(chunk) for chunk in chunks)
        whole_done = run_whole_translation(chunks, video_brief.context_block(info, brief, all_text), *whole)
    print(f"\n[3/3] Translating with {current}" + (f", checking with {reviewer}..." if reviewer else "..."))
    reused = 0
    ready = {}      # chunk index -> its final translation (reviewed when a review model is set)
    pending = {}    # chunk index -> (future, draft, key, submitted at) of a review still running
    abandoned = []  # reviews given up while their request was still running
    published = 0   # chunks already written to final_segments, always a prefix

    def stop(error, what):
        saved = sum(key != MODELS_ENTRY for key in cache)
        if isinstance(error, llm.QuotaExhausted):
            raise SystemExit(
                f"\nQuota reached ({what}): {error}\n"
                f"{saved} saved translations; run the same command later to continue."
            ) from error
        raise SystemExit(
            f"\n{what.capitalize()} stopped: {error}\n"
            f"{saved} saved translations; fix the problem and run the same command again."
        ) from error

    def keep_draft(index, draft, reason, quota=False):
        was_on = not tracker.off
        tracker.failed(quota)
        print(f"Chunk {index + 1}/{len(chunks)}: {reason}; keeping the draft translation", flush=True)
        if was_on and tracker.off:
            print(f"Review switched off for the rest of this run ({reviewer} is not answering); "
                  "the remaining chunks keep their draft.", flush=True)
        ready[index] = draft

    def collect(wait=False):
        """Move finished (or given-up) reviews into `ready`; with wait=True, until none is left."""
        while True:
            for index in sorted(pending):
                future, draft, key, submitted = pending[index]
                if not future.done():
                    if tracker.overdue(submitted) or tracker.off:
                        if not future.cancel():  # not started yet: never sent; running: its answer is ignored
                            abandoned.append(future)
                        del pending[index]
                        keep_draft(index, draft, "review too slow" if not tracker.off else "review skipped")
                    continue
                del pending[index]
                try:
                    reviewed = future.result()
                except llm.QuotaExhausted as error:
                    keep_draft(index, draft, f"review quota reached ({str(error)[:100]})", quota=True)
                    continue
                except (llm.LLMError, RuntimeError) as error:
                    keep_draft(index, draft, f"review failed ({str(error)[:100]})")
                    continue
                tracker.succeeded()
                changed = sum(a["text"] != b["text"] for a, b in zip(draft, reviewed))
                print(f"Chunk {index + 1}/{len(chunks)}: review changed {changed} of {len(draft)} lines"
                      + review_warning(changed, len(draft)))
                cache[key] = reviewed
                write_json_atomic(CACHE_FILE, cache)
                ready[index] = reviewed
            if not (wait and pending):
                return
            publish()  # chunks whose review is settled can already be read aloud
            time.sleep(0.5)

    def publish():
        nonlocal published
        while published in ready:
            chunk, translated = chunks[published], ready.pop(published)
            for source, translation in zip(chunk, translated):
                final_segments.append({
                    "start": source["start"],
                    "end": source["end"],
                    "source_text": source["source_text"],
                    "translated_text": translation["text"],
                })
            published += 1
        write_json_atomic(PARTIAL_FILE, {"complete": False, "segments": final_segments})

    try:
        for index, chunk in enumerate(chunks):
            start = chunk[0]["start"]
            end = chunk[-1]["end"]
            translated, made_by = find_cached(cache, chunk, current, not args.retranslate, briefs[index])

            if translated is not None:
                by = "" if made_by == current else f" by {made_by}"
                reused += made_by != current
                print(
                    f"Chunk {index + 1}/{len(chunks)} "
                    f"({start:.2f}s -> {end:.2f}s): saved translation{by}"
                )
            elif index in whole_done:
                translated, made_by = whole_done[index], whole_name
                print(f"Chunk {index + 1}/{len(chunks)} ({start:.2f}s -> {end:.2f}s): from the whole-video request")
                cache[chunk_key(chunk, made_by, briefs[index])] = translated
                remember_model(cache, made_by)
                write_json_atomic(CACHE_FILE, cache)
            else:
                print(
                    f"Chunk {index + 1}/{len(chunks)} "
                    f"({start:.2f}s -> {end:.2f}s)..."
                )
                try:
                    following = [item["source_text"] for item in chunks[index + 1][:FOLLOWING_SEGMENTS]] \
                        if index + 1 < len(chunks) else []
                    translated = translate_chunk(chunk, context=context, brief=briefs[index], following=following)
                except (llm.QuotaExhausted, llm.LLMError, RuntimeError) as error:
                    stop(error, "translation")
                made_by = llm.active_name("translate")  # the fallback LLM once the main one ran out
                cache[chunk_key(chunk, made_by, briefs[index])] = translated
                remember_model(cache, made_by)
                write_json_atomic(CACHE_FILE, cache)

            # The next draft is translated with the previous drafts as context, so it never waits for a review.
            context = (context + [(source["source_text"], item["text"]) for source, item in zip(chunk, translated)])[-CONTEXT_SEGMENTS:]

            if not reviewer:
                ready[index] = translated
            else:
                key = review_key(chunk, translated, reviewer, briefs[index])
                if key in cache and is_usable(cache[key], len(chunk)):
                    ready[index] = cache[key]
                elif tracker.off:
                    ready[index] = translated
                else:
                    pending[index] = (pool.submit(review_chunk, chunk, translated, None, briefs[index]),
                                      translated, key, time.monotonic())
                collect()
            publish()

        collect(wait=True)
        publish()
    finally:
        if pool:
            pool.shutdown(wait=False, cancel_futures=True)

    write_json_atomic(OUTPUT_FILE, final_segments)
    write_json_atomic(PARTIAL_FILE, {"complete": True, "segments": final_segments})

    if reused:
        print(f"\nKept {reused} chunks translated by an earlier model "
              "(--retranslate translates them again with the current one).")
    if tracker and tracker.kept_drafts:
        print(f"\n{tracker.kept_drafts} chunks kept their draft because the review failed or was too slow "
              "(REVIEW_TIMEOUT); running the same command again reviews them.")
    print("\n--------------------------------")
    print("Pipeline complete.")
    print(f"Segments: {len(final_segments)}")
    print(f"Saved: {OUTPUT_FILE.name}")
    print("--------------------------------")
    if any(not future.done() for future in abandoned):
        # A review that was given up may still be waiting on the server (up to MAX_ATTEMPTS x
        # REQUEST_TIMEOUT in llm.py), and Python waits for its thread before exiting. Everything
        # is saved, so leave now instead of holding up the rest of the dub.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
