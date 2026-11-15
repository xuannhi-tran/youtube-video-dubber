import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import cuda_dlls  # keep before faster_whisper: it adds pip's CUDA DLLs to the search path
from faster_whisper import WhisperModel

import llm
from glossary import GLOSSARY, prompt_lines, whisper_hotwords
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
PROMPT_VERSION = 2
# Translated segments from the end of the previous chunk shown as context.
CONTEXT_SEGMENTS = 8

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


def transcribe_audio(audio_file, max_seconds, hotwords=None):
    print("\n[1/3] Loading Whisper large-v3...")

    model = load_whisper()

    print("Transcribing..." + (f" (expecting: {hotwords})" if hotwords else ""))

    start_time = time.perf_counter()

    segments, info = model.transcribe(
        str(audio_file),
        language="zh",
        beam_size=5,
        vad_filter=True,
        hotwords=hotwords,
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


def load_or_transcribe(audio_file, max_seconds):
    """Whisper is the slow stage, so keep its output with the other job files."""
    hotwords = whisper_hotwords(GLOSSARY)
    stamp = {
        "audio": audio_file.name,
        "bytes": audio_file.stat().st_size,
        "max_seconds": max_seconds,
        "hotwords": hotwords,
    }
    if RAW_FILE.exists():
        saved = json.loads(RAW_FILE.read_text(encoding="utf-8"))
        if saved.get("stamp") == stamp:
            print(f"Using saved transcript ({len(saved['segments'])} segments)")
            return saved["segments"]
    segments = transcribe_audio(audio_file, max_seconds, hotwords)
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
    for item in translated:
        if CJK.search(item["text"]):
            raise ChineseLeft(f"Chinese characters left in the translation: {CJK.search(item['text']).group(0)!r}")


def context_lines(context):
    """The end of the previous chunk, as 'Chinese => Vietnamese' lines."""
    return "\n".join(f"{source} => {translation}" for source, translation in context)


def build_translation_prompt(chunk, context=()):
    input_segments = [
        {
            "start": item["start"],
            "end": item["end"],
            "text": item["source_text"],
        }
        for item in chunk
    ]

    glossary_rule = (
        "- Glossary: use these terms exactly as written:\n"
        f"{prompt_lines(GLOSSARY)}\n"
        if GLOSSARY else ""
    )
    context_block = (
        "Previous part of the video, already translated (for continuity only, do "
        "NOT translate it again; keep its forms of address and its terms):\n"
        f"{context_lines(context)}\n\n"
        if context else ""
    )

    return f"""
You are the translation component of a video dubbing system. The Vietnamese
text is read aloud by a voice-over, so it must sound like natural speech.

The segments below are consecutive pieces of ONE passage of speech. Speech
recognition cut it into short pieces, so a sentence often runs over several
segments.

How to translate:
- First understand the whole passage, then translate it as fluent, natural
  spoken Vietnamese (not word-for-word, not subtitle style).
- Then split your Vietnamese back over the same segments, in order: each
  segment holds the Vietnamese for its part. Inside a sentence you may move a
  word or two across a segment boundary so the Vietnamese word order is natural.
- Put . ? ! only where a sentence really ends. A segment that continues a
  sentence ends with no punctuation or a comma, and the next segment starts in
  lowercase.
- Keep the speaker's meaning. The Chinese comes from speech recognition and may
  contain recognition errors: correct them only when the context clearly shows
  what was meant (a glossary term often is).
- Do not invent facts when the transcript is uncertain.
- Keep it concise: it will be spoken in the same time as the original.
- Forms of address: the speaker talks to the audience as "bạn" or "chúng ta".
  Never call a person, a group of people or a personality type "nó"; use their
  name, "người này", "những người này" or "họ". Refer to the same person the
  same way throughout.
- Write only what a voice-over can read: no parentheses, no quotation marks,
  no brackets, no symbols.
{glossary_rule}- Preserve every start and end timestamp exactly.
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

{context_block}Chinese transcript to translate:

{json.dumps(input_segments, ensure_ascii=False)}
"""


def retry_note(problem, expected):
    """Told to the model after an unusable answer, so the next one is fixed."""
    return (
        f"\n\nYour previous answer could not be used ({problem}). Write every segment in Vietnamese "
        "only: translate or transliterate Chinese words, never leave Chinese characters. Answer again with ONLY "
        f"a JSON array of exactly {expected} objects, one per input segment, each with "
        '"start", "end" and "text". No other text before or after the array.'
    )


def translate_chunk(chunk, model=None, context=(), generate=None):
    """Vietnamese for every segment of `chunk`, as [{"start", "end", "text"}, ...].

    An unusable answer is asked again (telling the model what was wrong). If a
    chunk still fails, its two halves are translated separately: fewer segments
    are easier for a small local model to count.
    """
    generate = generate or (lambda text: llm.generate(text, model=model, role="translate"))
    prompt = build_translation_prompt(chunk, context)

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
    first = translate_chunk(chunk[:middle], model, context, generate)
    joined = list(context) + [(source["source_text"], item["text"]) for source, item in zip(chunk[:middle], first)]
    second = translate_chunk(chunk[middle:], model, joined[-CONTEXT_SEGMENTS:], generate)
    return first + second


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def chunk_key(chunk, model_name=None):
    """Identifies a chunk's translation request, so a changed input is re-translated."""
    request = {
        "chunk": chunk,
        "glossary": GLOSSARY,
        "model": model_name or llm.describe(role="translate"),
        "prompt": PROMPT_VERSION,
    }
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


def find_cached(cache, chunk, current, reuse_other_models=True):
    """(translation, model name) of a saved translation of `chunk`, or (None, None).

    Prefers the current model. Unless told otherwise, a translation another
    model made of the same chunk (same glossary, same prompt version) is kept:
    switching from Groq to a local model must not throw away finished work.
    """
    def usable(translated):
        # A saved translation with Chinese left in it is translated again.
        try:
            check_translation(translated, len(chunk))
        except ValueError:
            return False
        return True

    key = chunk_key(chunk, current)
    if key in cache and usable(cache[key]):
        return cache[key], current
    if reuse_other_models:
        for name in earlier_models(cache, current):
            other = chunk_key(chunk, name)
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


def run_local_checks():
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


def main():
    run_local_checks()
    args = parse_args()
    segments = load_or_transcribe(args.audio, args.max_seconds)
    if not segments:
        raise SystemExit("Whisper found no speech in this audio.")

    print("\n[2/3] Creating translation chunks...")

    chunks = create_chunks(segments)

    print(f"Created {len(chunks)} context chunks.")

    cache = load_cache()
    final_segments = []
    context = []
    write_json_atomic(PARTIAL_FILE, {"complete": False, "segments": []})
    current = llm.describe(role="translate")

    print(f"\n[3/3] Translating with {current}...")
    reused = 0

    for index, chunk in enumerate(chunks, start=1):
        start = chunk[0]["start"]
        end = chunk[-1]["end"]
        translated, made_by = find_cached(cache, chunk, current, not args.retranslate)

        if translated is not None:
            by = "" if made_by == current else f" by {made_by}"
            reused += made_by != current
            print(
                f"Chunk {index}/{len(chunks)} "
                f"({start:.2f}s -> {end:.2f}s): saved translation{by}"
            )
        else:
            print(
                f"Chunk {index}/{len(chunks)} "
                f"({start:.2f}s -> {end:.2f}s)..."
            )
            saved = sum(key != MODELS_ENTRY for key in cache)
            try:
                translated = translate_chunk(chunk, context=context)
            except llm.QuotaExhausted as error:
                raise SystemExit(
                    f"\nQuota reached: {error}\n"
                    f"{saved} translated chunks are saved; run the same command later to continue."
                ) from error
            except (llm.LLMError, RuntimeError) as error:
                raise SystemExit(
                    f"\nTranslation stopped: {error}\n"
                    f"{saved} translated chunks are saved; fix the problem and run the same command again."
                ) from error
            cache[chunk_key(chunk, current)] = translated
            remember_model(cache, current)
            write_json_atomic(CACHE_FILE, cache)

        for source, translation in zip(chunk, translated):
            final_segments.append({
                "start": source["start"],
                "end": source["end"],
                "source_text": source["source_text"],
                "translated_text": translation["text"],
            })
        context = [(item["source_text"], item["translated_text"])
                   for item in final_segments[-CONTEXT_SEGMENTS:]]
        write_json_atomic(PARTIAL_FILE, {"complete": False, "segments": final_segments})

    write_json_atomic(OUTPUT_FILE, final_segments)
    write_json_atomic(PARTIAL_FILE, {"complete": True, "segments": final_segments})

    if reused:
        print(f"\nKept {reused} chunks translated by an earlier model "
              "(--retranslate translates them again with the current one).")
    print("\n--------------------------------")
    print("Pipeline complete.")
    print(f"Segments: {len(final_segments)}")
    print(f"Saved: {OUTPUT_FILE.name}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
