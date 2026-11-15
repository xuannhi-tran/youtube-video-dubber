"""Benchmark translated segments with local VieNeu TTS (no API quota)."""

import hashlib
import json
import math
from pathlib import Path
import wave

from vieneu_engine import MODEL, VOICE, Narrator


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "translated_segments.json"
OUTPUT_DIR = BASE_DIR / "tts_output"
RESULTS_FILE = BASE_DIR / "tts_duration_results.json"


def content_fingerprint(text):
    content = {
        "translated_text": text,
        "model": MODEL,
        "voice": VOICE,
    }
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path, data):
    temporary_file = path.with_suffix(path.suffix + ".tmp")
    temporary_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_file.replace(path)


def make_result(segment, audio_file, duration, fingerprint):
    slot_duration = segment["end"] - segment["start"]
    ratio = duration / slot_duration
    return {
        **segment,
        "slot_duration": slot_duration,
        "tts_duration": duration,
        "duration_ratio": ratio,
        "classification": classify(ratio),
        "audio_file": audio_file.relative_to(BASE_DIR).as_posix(),
        "fingerprint": fingerprint,
    }


def classify(ratio):
    if ratio <= 1.00:
        return "FIT"
    if ratio <= 1.20:
        return "SPEED_UP"
    return "TOO_LONG"


def measure_wav(source):
    with wave.open(source, "rb") as wav_file:
        frame_count = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        frame_size = wav_file.getnchannels() * wav_file.getsampwidth()
        frames = wav_file.readframes(frame_count)
        if frame_count <= 0 or sample_rate <= 0:
            raise ValueError("Empty or invalid WAV audio")
        if len(frames) != frame_count * frame_size:
            raise ValueError("Truncated WAV audio")
        return frame_count / sample_rate


def generate_clip(narrator, text, audio_file):
    """Synthesize `text` into `audio_file` with the local VieNeu voice; return seconds."""
    return narrator.synthesize(text, audio_file)


def main():
    segments = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    if not isinstance(segments, list) or not segments:
        raise ValueError("Expected a non-empty list of translated segments")

    # Validate all input before spending quota on any segment.
    for index, segment in enumerate(segments, start=1):
        start, end = segment["start"], segment["end"]
        if not (math.isfinite(start) and math.isfinite(end) and end > start):
            raise ValueError(f"Segment {index:03d}: invalid timestamp slot")
        text = segment["translated_text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Segment {index:03d}: empty translated_text")


    OUTPUT_DIR.mkdir(exist_ok=True)
    results = []
    metadata_file = OUTPUT_DIR / "cache_metadata.json"
    metadata = {}
    if metadata_file.exists():
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            print("Invalid cache metadata; existing audio will be regenerated")
        if not isinstance(metadata, dict):
            metadata = {}

    # Rebuild from current input and verified audio, never from old result rows.
    # Include all valid cached segments so interruption cannot drop later results.
    checkpoint = {}
    fingerprints = {}
    for index, segment in enumerate(segments, start=1):
        audio_file = OUTPUT_DIR / f"segment_{index:03d}.wav"
        fingerprint = content_fingerprint(segment["translated_text"])
        fingerprints[index] = fingerprint
        entry = metadata.get(audio_file.name)
        if not audio_file.exists():
            continue
        if not isinstance(entry, dict) or entry.get("fingerprint") != fingerprint:
            print(f"[{index:03d}] Cache mismatch - regenerating")
            continue
        try:
            duration = measure_wav(str(audio_file))
        except (OSError, EOFError, wave.Error, ValueError):
            print(f"[{index:03d}] Invalid cached audio - regenerating")
            continue
        checkpoint[index] = make_result(segment, audio_file, duration, fingerprint)
    write_json_atomic(RESULTS_FILE, [checkpoint[key] for key in sorted(checkpoint)])
    print(f"Model: {MODEL} | Voice: {VOICE} | Segments: {len(segments)}")
    with Narrator() as narrator:
        for index, segment in enumerate(segments, start=1):
            start, end = segment["start"], segment["end"]
            slot_duration = end - start
            audio_file = OUTPUT_DIR / f"segment_{index:03d}.wav"
            print(f"\n[{index:03d}] {start:.2f}-{end:.2f}", flush=True)
            try:
                if index in checkpoint:
                    tts_duration = checkpoint[index]["tts_duration"]
                    print(f"[{index:03d}] Using cached audio")
                else:
                    # Invalidate before overwriting audio, including if interrupted
                    # between the WAV replacement and the new metadata write.
                    metadata.pop(audio_file.name, None)
                    write_json_atomic(metadata_file, metadata)
                    tts_duration = generate_clip(
                        narrator, segment["translated_text"], audio_file
                    )
                    metadata[audio_file.name] = {"fingerprint": fingerprints[index]}
                    write_json_atomic(metadata_file, metadata)
            except Exception as error:
                print(
                    f"Segment {index:03d} TTS failed: {error}. "
                    "Benchmark stopped; completed checkpoints are preserved.",
                    flush=True,
                )
                raise

            ratio = tts_duration / slot_duration
            classification = classify(ratio)
            results.append(make_result(
                segment, audio_file, tts_duration, fingerprints[index]
            ))
            checkpoint[index] = results[-1]
            write_json_atomic(
                RESULTS_FILE, [checkpoint[key] for key in sorted(checkpoint)]
            )
            print(
                f"Slot: {slot_duration:.2f}s | TTS: {tts_duration:.2f}s | "
                f"Ratio: {ratio:.2f} | {classification}"
            )

    ratios = [result["duration_ratio"] for result in results]
    print(f"\nTotal segments: {len(results)}")
    for classification in ("FIT", "SPEED_UP", "TOO_LONG"):
        count = sum(result["classification"] == classification for result in results)
        print(f"{classification}: {count}")
    print(f"Average ratio: {sum(ratios) / len(ratios):.2f}")
    print(f"Maximum ratio: {max(ratios):.2f}")
    print(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
