"""Turn the translated blocks into timed Vietnamese speech, reading each block once.

For every dub block:
  1. predict how long it will sound (words per second, learned while running);
     if it will not fit its time slot, shorten it with the LLM *before* reading it;
  2. when a review model is configured, check shortened text against the Chinese;
  3. read the final text with VieNeu and measure it;
  4. if it is still too long, shorten it once more and read it again.

The LLM and VieNeu work at the same time (one worker thread each), so a rewrite
costs no waiting while other blocks are being read. With --follow, blocks are
taken from translated_partial.json while pipeline.py is still translating, so
speech starts long before the translation is finished.

    python speech.py             # job already translated (translated_segments.json)
    python speech.py --follow    # run next to pipeline.py (dub.py does this)

Before reading, text is made speakable: a parenthesis that only repeats what the
sentence already says is dropped, quotation marks go, and words listed in
pronunciations.json (backend folder, or the job folder to override) are replaced
by how they should sound, e.g. {"ENFJ": "e n f j"}.

Writes dub_blocks_fitted.json and block_tts_duration_results_fitted.json, which
assemble_dub.py reads. An interrupted run resumes: rewrites are kept in
speech_state.json and audio in block_tts_output/ (reused when the text matches).
"""

import argparse
import concurrent.futures as futures
import heapq
import json
import math
import re
import statistics
import time
import wave

import dub_block_builder as builder
import fit_blocks as fit
import llm
import test_tts_duration as tts
from vieneu_engine import BATCH, MODEL, VOICE, Narrator
from workdir import SCRIPT_DIR, WORKDIR


SEGMENTS_FILE = WORKDIR / "translated_segments.json"
PARTIAL_FILE = WORKDIR / "translated_partial.json"
BLOCKS_FILE = WORKDIR / "dub_blocks.json"
STATE_FILE = WORKDIR / "speech_state.json"
FITTED_FILE = WORKDIR / "dub_blocks_fitted.json"
RESULTS_FILE = WORKDIR / "block_tts_duration_results_fitted.json"
AUDIO_DIR = WORKDIR / "block_tts_output"
METADATA_FILE = AUDIO_DIR / "cache_metadata.json"
# Shared by all jobs, so the next video starts from the rate measured on this one.
CALIBRATION_FILE = SCRIPT_DIR / "dub_jobs_output" / "tts_calibration.json"

PRONUNCIATION_FILES = [SCRIPT_DIR / "pronunciations.json", WORKDIR / "pronunciations.json"]

DEFAULT_WORDS_PER_SECOND = 4.5  # measured on the first test video
MIN_SAMPLES = 3                 # measured clips needed before trusting them
MAX_REFITS = 1                  # extra rewrites after a clip measured too long
POST_FIT_REVIEW_VERSION = 1     # raise to retry saved fitted blocks with a new review prompt
POLL_SECONDS = 1.0


# ---------------------------------------------------------
# Pure helpers (covered by run_local_checks)
# ---------------------------------------------------------

_PARENTHESIS = re.compile(r"\s*[(\[]([^()\[\]]*)[)\]]")
_LATIN_ONLY = re.compile(r"^[A-Za-z0-9.+\-,\s]+$")
_QUOTES = re.compile(r'[“”„"«»]')


def speakable(text, seen=None):
    """Text as the voice-over should read it (parentheses are read aloud otherwise).

    A parenthesis is dropped when it repeats words already in the sentence, was
    already said in an earlier block (`seen`, updated here), or only holds Latin
    abbreviations; any other one becomes a short aside between commas.
    """
    outside = _PARENTHESIS.sub(" ", text).lower()
    seen = set() if seen is None else seen

    def aside(match):
        inner = match[1].strip()
        key = inner.lower()
        if not inner or key in outside or key in seen or _LATIN_ONLY.match(inner):
            return ""
        seen.add(key)
        return f", {inner},"

    text = _PARENTHESIS.sub(aside, text)
    text = _QUOTES.sub("", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",\s*([,.;:!?])", r"\1", text)
    return text.strip(" ,")


def pronounce(text, pronunciations):
    """Replace whole words by how they should sound (pronunciations.json)."""
    for written, spoken in pronunciations.items():
        text = re.sub(rf"(?<!\w){re.escape(written)}(?!\w)", spoken, text)
    return text


def words_per_second(samples, fallback):
    """Median of the measured rates once there are enough, else `fallback`."""
    return statistics.median(samples) if len(samples) >= MIN_SAMPLES else fallback


def new_record(text):
    """What speech_state.json keeps per block."""
    return {"input": text, "text": text, "fitted": False, "refits": 0,
            "status": "unchanged", "post_fit_review_version": 0}


def restore_record(saved, text):
    """The saved record for a block, unless its translation changed since then."""
    if isinstance(saved, dict) and saved.get("input") == text and isinstance(saved.get("text"), str):
        return {**new_record(text), **saved}
    return new_record(text)


def same_block(left, right):
    keys = ("start", "end", "translated_text", "segment_indices")
    return all(left.get(key) == right.get(key) for key in keys)


def make_result(block, text, audio_file, duration, fingerprint):
    """One row of block_tts_duration_results_fitted.json (as assemble_dub.py expects)."""
    slot = block["end"] - block["start"]
    ratio = duration / slot
    return {
        **block,
        "translated_text": text,
        "slot_duration": slot,
        "tts_duration": duration,
        "duration_ratio": ratio,
        "required_speed_factor": max(1.0, ratio),
        "classification": tts.classify(ratio),
        "audio_file": audio_file.relative_to(WORKDIR).as_posix(),
        "fingerprint": fingerprint,
    }


def load_pronunciations(paths=PRONUNCIATION_FILES):
    """{written: spoken} from pronunciations.json files; later files override earlier ones."""
    merged = {}
    for path in paths:
        data = read_json(path, None)
        if data is None:
            continue
        if not isinstance(data, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in data.items()):
            raise SystemExit(f"{path} must be a JSON object of word -> pronunciation strings")
        merged.update({key.strip(): value.strip() for key, value in data.items() if key.strip()})
    return merged


def calibration_key():
    return f"{MODEL}|{VOICE}"


def run_local_checks():
    assert words_per_second([4.0, 5.0], 4.5) == 4.5
    assert words_per_second([4.0, 5.0, 3.0], 4.5) == 4.0

    record = new_record("a b")
    assert record == {"input": "a b", "text": "a b", "fitted": False, "refits": 0,
                      "status": "unchanged", "post_fit_review_version": 0}
    saved = {"input": "a b", "text": "a", "fitted": True, "refits": 1, "status": "rewritten"}
    assert restore_record(saved, "a b") == {**new_record("a b"), **saved}
    assert restore_record(saved, "a b c") == new_record("a b c")
    assert restore_record(None, "x") == new_record("x")
    assert restore_record({"input": "x", "text": None}, "x") == new_record("x")

    block = {"block_id": 1, "start": 0, "end": 2, "translated_text": "a", "segment_indices": [1]}
    assert same_block(block, dict(block))
    assert not same_block(block, {**block, "end": 3})

    assert speakable("Fe (cảm xúc hướng ngoại) chính là cảm xúc hướng ngoại.") == \
        "Fe chính là cảm xúc hướng ngoại."
    assert speakable("Vì vậy ENFJ (MBTI) thường được gọi là “nhân vật chính”.") == \
        "Vì vậy ENFJ thường được gọi là nhân vật chính."
    assert speakable("Chức năng Fe (cảm xúc hướng ngoại) rất mạnh.") == \
        "Chức năng Fe, cảm xúc hướng ngoại, rất mạnh."
    assert speakable("Kết thúc (cảm xúc hướng ngoại).") == "Kết thúc, cảm xúc hướng ngoại."
    seen = set()
    assert speakable("Là Fe (cảm xúc hướng ngoại) đó.", seen) == "Là Fe, cảm xúc hướng ngoại, đó."
    assert speakable("Vì Fe (cảm xúc hướng ngoại) mạnh.", seen) == "Vì Fe mạnh."
    assert seen == {"cảm xúc hướng ngoại"}
    assert speakable("Không có gì đặc biệt.") == "Không có gì đặc biệt."
    assert pronounce("ENFJ và INFJ, ENFJs", {"ENFJ": "e n f j"}) == "e n f j và INFJ, ENFJs"
    assert pronounce("Fe là Fe.", {"Fe": "ép e"}) == "ép e là ép e."

    row = make_result(block, "b c", WORKDIR / "block_tts_output" / "block_001.wav", 2.5, "f")
    assert row["translated_text"] == "b c" and row["duration_ratio"] == 1.25
    assert row["classification"] == "TOO_LONG" and row["required_speed_factor"] == 1.25
    assert row["audio_file"] == "block_tts_output/block_001.wav"


# ---------------------------------------------------------
# Files
# ---------------------------------------------------------

def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (ValueError, UnicodeError, PermissionError):
        return default


def load_fallback_rate():
    saved = read_json(CALIBRATION_FILE, {})
    rate = saved.get(calibration_key()) if isinstance(saved, dict) else None
    if isinstance(rate, (int, float)) and math.isfinite(rate) and rate > 0:
        return rate, "measured on an earlier job"
    return DEFAULT_WORDS_PER_SECOND, "default"


def save_fallback_rate(rate):
    saved = read_json(CALIBRATION_FILE, {})
    saved = saved if isinstance(saved, dict) else {}
    saved[calibration_key()] = round(rate, 3)
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    tts.write_json_atomic(CALIBRATION_FILE, saved)


# ---------------------------------------------------------
# The run
# ---------------------------------------------------------

class Speech:
    def __init__(self, follow, target, fresh):
        self.follow = follow
        self.target = target
        saved = {} if fresh else read_json(STATE_FILE, {})
        self.saved = saved.get("blocks", {}) if isinstance(saved, dict) else {}
        self.records = {}   # block_id -> record (see new_record)
        self.blocks = {}    # block_id -> block from dub_block_builder
        self.results = {}   # block_id -> result row of the latest clip
        self.complete = False
        self.waiting_shown = False
        self.review_enabled = llm.role_configured("review")
        self.review_off = False

        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        metadata = read_json(METADATA_FILE, {})
        self.metadata = metadata if isinstance(metadata, dict) else {}

        self.seen_asides = set()  # explanations in parentheses already said once
        self.pronunciations = load_pronunciations()
        if self.pronunciations:
            print(f"Pronunciations: {', '.join(f'{k} -> {v}' for k, v in self.pronunciations.items())}")
        self.fallback, source = load_fallback_rate()
        self.samples = []   # words per second of every measured clip
        print(f"Speech rate to start with: {self.fallback:.2f} words/s ({source}); "
              "updated from the clips as they are read")

        self.llm_queue = []  # heap of (block_id, kind) with kind "fit" or "refit"
        self.tts_queue = []  # heap of block_id
        self.llm_future = None
        self.tts_future = None
        self.llm_off = False
        self.narrator = None
        self.batch_size = BATCH  # blocks read in one pass (TTS_BATCH, default 1)
        self.busy = {"llm": 0.0, "tts": 0.0}
        self.counts = {"rewrites": 0, "refits": 0, "polishes": 0, "clips": 0, "cached": 0}

    # -- rate -------------------------------------------------

    def rate(self):
        return words_per_second(self.samples, self.fallback)

    def budget(self, block_id):
        block = self.blocks[block_id]
        return fit.budget_words(block["end"] - block["start"], self.rate(), self.target)

    def too_long(self, block_id):
        return fit.word_count(self.records[block_id]["text"]) > self.budget(block_id)

    def needs_post_fit_review(self, record):
        version = record.get("post_fit_review_version", 0)
        version = version if isinstance(version, int) else 0
        return (self.review_enabled and record["status"] in ("rewritten", "still_over_budget")
                and version < POST_FIT_REVIEW_VERSION)

    # -- new blocks -------------------------------------------

    def poll_source(self):
        if self.complete:
            return
        if not self.follow:
            segments = builder.load_segments(SEGMENTS_FILE)
            self.take_blocks(builder.final_blocks(segments, complete=True), complete=True)
            return
        partial = read_json(PARTIAL_FILE, None)
        if not isinstance(partial, dict) or not isinstance(partial.get("segments"), list):
            if not self.waiting_shown:
                print("Waiting for the first translated chunk...", flush=True)
                self.waiting_shown = True
            return
        segments = partial["segments"]
        builder.validate_segments(segments)
        complete = partial.get("complete") is True
        self.take_blocks(builder.final_blocks(segments, complete), complete)

    def take_blocks(self, blocks, complete):
        for block in blocks:
            block_id = block["block_id"]
            known = self.blocks.get(block_id)
            if known is not None:
                if not same_block(known, block):
                    raise RuntimeError(f"Block {block_id} changed after it was settled")
                continue
            self.blocks[block_id] = block
            # Blocks arrive in order, so `seen` is the same on every run (resume stays valid).
            spoken = speakable(block["translated_text"], self.seen_asides)
            record = restore_record(self.saved.get(str(block_id)), spoken)
            self.records[block_id] = record
            if not record["fitted"] and not self.too_long(block_id):
                record["fitted"] = True  # fits already: no need to wait for the LLM
            if record["fitted"]:
                if self.needs_post_fit_review(record):
                    heapq.heappush(self.llm_queue, (block_id, "polish"))
                else:
                    heapq.heappush(self.tts_queue, block_id)
            else:
                # Checked again when its turn comes, with the rate known by then.
                heapq.heappush(self.llm_queue, (block_id, "fit"))
        if complete and not self.complete:
            self.complete = True
            all_blocks = [self.blocks[key] for key in sorted(self.blocks)]
            if not all_blocks:
                raise SystemExit("No speech to dub: the translation is empty.")
            builder.save_blocks(all_blocks, BLOCKS_FILE)
            print(f"Translation complete: {len(all_blocks)} blocks", flush=True)

    # -- dispatching work -------------------------------------

    def dispatch(self, llm_pool, tts_pool):
        # Audio first: reused clips update the speech rate before any rewrite is sized.
        self.dispatch_tts(tts_pool)
        self.dispatch_llm(llm_pool)
        self.dispatch_tts(tts_pool)

    def dispatch_llm(self, llm_pool):
        while self.llm_future is None and self.llm_queue:
            block_id, kind = heapq.heappop(self.llm_queue)
            record, block = self.records[block_id], self.blocks[block_id]
            words = fit.word_count(record["text"])
            if kind == "fit":
                budget = self.budget(block_id)
                if self.llm_off or words <= budget:
                    record["fitted"] = True
                    if words > budget:
                        record["status"] = "still_over_budget"
                    heapq.heappush(self.tts_queue, block_id)
                    continue
            elif kind == "polish":
                if self.review_off:
                    heapq.heappush(self.tts_queue, block_id)
                    continue
                budget = self.budget(block_id)
            else:
                budget = record["refit_budget"]
                if self.llm_off:
                    self.finish_block(block_id)
                    continue
            label = {"fit": "shorten", "refit": "shorten again", "polish": "review fitted text"}[kind]
            print(f"[{block_id:03d}] {label}: {words} -> <= {budget} words", flush=True)
            request = {**block, "translated_text": record["text"]}
            self.llm_future = llm_pool.submit(self.rewrite, request, budget, kind)
            self.llm_future.job = (block_id, kind)
        self.save_state()

    def dispatch_tts(self, tts_pool):
        """Send up to BATCH waiting blocks to the voice in one go (reused clips are skipped)."""
        while self.tts_future is None and self.tts_queue:
            jobs = []
            while self.tts_queue and len(jobs) < self.batch_size:
                block_id = heapq.heappop(self.tts_queue)
                text = self.records[block_id]["text"]
                spoken = pronounce(text, self.pronunciations)
                audio_file = AUDIO_DIR / f"block_{block_id:03d}.wav"
                fingerprint = tts.content_fingerprint(spoken)
                duration = self.cached_duration(audio_file, fingerprint)
                if duration is not None:
                    self.counts["cached"] += 1
                    self.clip_done(block_id, text, audio_file, duration, fingerprint, cached=True)
                    continue
                # Invalidate first, so an interrupted write is never taken for a good clip.
                self.metadata.pop(audio_file.name, None)
                jobs.append((block_id, text, audio_file, fingerprint, spoken))
            if jobs:
                tts.write_json_atomic(METADATA_FILE, self.metadata)
                self.tts_future = tts_pool.submit(self.synthesize, [(job[4], job[2]) for job in jobs])
                self.tts_future.job = jobs

    def cached_duration(self, audio_file, fingerprint):
        entry = self.metadata.get(audio_file.name)
        if not audio_file.exists() or not isinstance(entry, dict) or entry.get("fingerprint") != fingerprint:
            return None
        try:
            return tts.measure_wav(str(audio_file))
        except (OSError, EOFError, wave.Error, ValueError):
            return None

    # -- worker threads ---------------------------------------

    def rewrite(self, block, budget, kind):
        started = time.perf_counter()
        try:
            if kind == "polish":
                return fit.review_fitted_block(block, budget), "polished"
            return fit.fit_block(block, budget)
        finally:
            self.busy["llm"] += time.perf_counter() - started

    def synthesize(self, items):
        if self.narrator is None:  # created in the TTS thread, the only one that uses it
            started = time.perf_counter()
            self.narrator = Narrator()
            print(f"VieNeu loaded in {time.perf_counter() - started:.1f}s", flush=True)
        started = time.perf_counter()
        try:
            return self.narrator.synthesize_many(items)
        finally:
            self.busy["tts"] += time.perf_counter() - started

    # -- finished work ----------------------------------------

    def collect(self):
        if self.llm_future is not None and self.llm_future.done():
            future, self.llm_future = self.llm_future, None
            self.rewrite_done(future)
        if self.tts_future is not None and self.tts_future.done():
            future, self.tts_future = self.tts_future, None
            durations = future.result()  # a TTS failure stops the run
            for (block_id, text, audio_file, fingerprint, _), duration in zip(future.job, durations):
                self.counts["clips"] += 1
                self.metadata[audio_file.name] = {"fingerprint": fingerprint}
            tts.write_json_atomic(METADATA_FILE, self.metadata)
            for (block_id, text, audio_file, fingerprint, _), duration in zip(future.job, durations):
                self.clip_done(block_id, text, audio_file, duration, fingerprint, cached=False)

    def rewrite_done(self, future):
        block_id, kind = future.job
        record = self.records[block_id]
        try:
            text, status = future.result()
        except (llm.QuotaExhausted, llm.LLMError, RuntimeError, ValueError) as error:
            if kind != "polish" and not isinstance(error, (llm.QuotaExhausted, llm.LLMError)):
                raise
            if kind == "polish":
                print(f"[{block_id:03d}] post-fit review failed ({error}); keeping fitted text", flush=True)
                self.review_off = True
                heapq.heappush(self.tts_queue, block_id)
                self.save_state()
                return
            print(f"LLM stopped ({error}). The remaining blocks keep their text; "
                  "too-long ones are sped up a little more when assembled.", flush=True)
            self.llm_off = True
            text, status = None, "still_over_budget"
        if kind == "polish":
            self.counts["polishes"] += 1
            if text:
                record["text"] = speakable(text, self.seen_asides)
            record["post_fit_review_version"] = POST_FIT_REVIEW_VERSION
            heapq.heappush(self.tts_queue, block_id)
            self.save_state()
            return
        if kind == "refit":
            record["refits"] += 1
            self.counts["refits"] += 1
        else:
            self.counts["rewrites"] += 1
        if text:
            record["text"] = speakable(text, self.seen_asides)
        record["status"] = status
        record["fitted"] = True
        record.pop("refit_budget", None)
        record["post_fit_review_version"] = 0
        if text and self.needs_post_fit_review(record):
            heapq.heappush(self.llm_queue, (block_id, "polish"))
        else:
            heapq.heappush(self.tts_queue, block_id)
        self.save_state()

    def clip_done(self, block_id, text, audio_file, duration, fingerprint, cached):
        record, block = self.records[block_id], self.blocks[block_id]
        row = make_result(block, text, audio_file, duration, fingerprint)
        self.results[block_id] = row
        if duration > 0:
            self.samples.append(fit.word_count(text) / duration)
        ratio = row["duration_ratio"]
        note = " (cached)" if cached else ""
        print(f"[{block_id:03d}] {duration:.2f}s for a {row['slot_duration']:.2f}s slot "
              f"= {ratio:.2f} {row['classification']}{note}", flush=True)
        if ratio > self.target and record["refits"] < MAX_REFITS and not self.llm_off:
            record["refit_budget"] = fit.measured_budget(fit.word_count(text), ratio, self.target)
            heapq.heappush(self.llm_queue, (block_id, "refit"))
        else:
            self.finish_block(block_id)
        self.save_state()

    def finish_block(self, block_id):
        record = self.records[block_id]
        record.pop("refit_budget", None)
        tts.write_json_atomic(RESULTS_FILE, [self.results[key] for key in sorted(self.results)])

    # -- state ------------------------------------------------

    def save_state(self):
        blocks = {**self.saved, **{str(key): value for key, value in self.records.items()}}
        tts.write_json_atomic(STATE_FILE, {"blocks": blocks})

    def idle(self):
        return (self.llm_future is None and self.tts_future is None
                and not self.llm_queue and not self.tts_queue)

    # -- main loop --------------------------------------------

    def run(self):
        started = time.perf_counter()
        with futures.ThreadPoolExecutor(1, thread_name_prefix="llm") as llm_pool, \
                futures.ThreadPoolExecutor(1, thread_name_prefix="tts") as tts_pool:
            try:
                while True:
                    self.poll_source()
                    self.dispatch(llm_pool, tts_pool)
                    if self.idle():
                        if self.complete:
                            break
                        time.sleep(POLL_SECONDS)  # waiting for more translation
                        continue
                    running = [f for f in (self.llm_future, self.tts_future) if f is not None]
                    if running:
                        futures.wait(running, timeout=POLL_SECONDS,
                                     return_when=futures.FIRST_COMPLETED)
                    self.collect()
            finally:
                self.save_state()
                for future in (self.llm_future, self.tts_future):
                    if future is not None:
                        future.cancel()
        self.write_outputs()
        self.print_summary(time.perf_counter() - started)

    def write_outputs(self):
        rows = [self.results[key] for key in sorted(self.blocks)]
        tts.write_json_atomic(RESULTS_FILE, rows)
        rate = self.rate()
        fitted = []
        for block_id in sorted(self.blocks):
            block, record = self.blocks[block_id], self.records[block_id]
            fitted.append({
                **block,
                "original_translated_text": block["translated_text"],
                "translated_text": record["text"],
                "word_count": fit.word_count(record["text"]),
                "budget_words": self.budget(block_id),
                "status": record["status"],
            })
        tts.write_json_atomic(FITTED_FILE, fitted)
        if len(self.samples) >= MIN_SAMPLES:
            save_fallback_rate(rate)

    def print_summary(self, elapsed):
        rows = [self.results[key] for key in sorted(self.results)]
        print("\n----------------------------------")
        print(f"Speech done: {len(rows)} blocks in {elapsed / 60:.1f} min")
        print(f"Clips read: {self.counts['clips']} (+{self.counts['cached']} reused) | "
              f"VieNeu busy {self.busy['tts'] / 60:.1f} min")
        print(f"Shortened: {self.counts['rewrites']} before reading, "
              f"{self.counts['refits']} again after measuring | LLM busy {self.busy['llm'] / 60:.1f} min")
        print(f"Post-fit reviews: {self.counts['polishes']}")
        for classification in ("FIT", "SPEED_UP", "TOO_LONG"):
            print(f"{classification}: {sum(row['classification'] == classification for row in rows)}")
        print(f"Speech rate measured: {self.rate():.2f} words/s")
        print(f"Saved: {FITTED_FILE.name}, {RESULTS_FILE.name}")
        print("----------------------------------")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--follow", action="store_true",
                        help="read blocks from translated_partial.json while pipeline.py is still translating")
    parser.add_argument("--target", type=float, default=fit.TARGET,
                        help=f"longest allowed clip/slot ratio before shortening (default {fit.TARGET})")
    parser.add_argument("--fresh", action="store_true",
                        help="forget earlier rewrites (audio for unchanged text is still reused)")
    args = parser.parse_args()
    if not math.isfinite(args.target) or args.target <= 0:
        parser.error("--target must be positive")
    return args


def main():
    run_local_checks()
    args = parse_args()
    reviewer = None
    if llm.role_configured("review"):
        try:
            reviewer = llm.describe(role="review")
        except (RuntimeError, ValueError) as error:
            print(f"Post-fit review is misconfigured ({error}); continuing without it.", flush=True)
    print(f"Model: {MODEL} | Voice: {VOICE} | Batch: {BATCH} | LLM (shorten): {llm.describe(role="fit")}" +
          (f" | post-fit review: {reviewer}" if reviewer else ""))
    Speech(args.follow, args.target, args.fresh).run()


if __name__ == "__main__":
    main()
