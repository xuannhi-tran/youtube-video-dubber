"""Read the same Vietnamese sample with every VieNeu preset voice, to compare them by ear.

Writes one WAV per voice plus an index.html with labelled audio players.
Runs locally: no API key and no quota involved.

    python compare_voices.py                  # male voices, built-in sample text
    python compare_voices.py --gender all
    python compare_voices.py --text "First sentence." --text "Second sentence."
"""

import argparse
import html
from pathlib import Path
import re
import time
import unicodedata

import numpy as np

import hf_cache_fix  # keep before vieneu: it must set its env var before huggingface_hub loads


BASE_DIR = Path(__file__).resolve().parent
# Matches the "*_output/" rule in .gitignore, so the samples are never committed.
DEFAULT_OUT = BASE_DIR / "voice_samples_output"
GAP_SECONDS = 0.6

DEFAULT_TEXTS = [
    # Lecture-style narration, like a knowledge video.
    "Hôm nay chúng ta sẽ tìm hiểu về một trong những bộ sách cổ nhất của Trung Quốc, "
    "và vì sao nó vẫn còn ảnh hưởng đến chúng ta cho đến ngày nay.",
    # Stress test: a year, a decimal, a percentage and an English word.
    "Năm 2024, có khoảng 3,5 triệu người xem video này, "
    "và 80% trong số họ đến từ YouTube Shorts.",
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def slugify(name):
    """ASCII file name from a Vietnamese voice name: 'Thiện Minh' -> 'thien_minh'."""
    text = unicodedata.normalize("NFKD", name.replace("đ", "d").replace("Đ", "D"))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def pick_voices(tts, gender):
    """Return [(voice_id, info)] for the preset voices that match the gender filter."""
    voices = []
    for _label, voice_id in tts.list_preset_voices():
        info = tts.get_preset_voice(voice_id)
        if gender == "all" or info.get("gender") == gender:
            voices.append((voice_id, info))
    return voices


def synthesize(tts, voice_id, texts):
    """One waveform with every text read in order, separated by a short pause."""
    gap = np.zeros(int(GAP_SECONDS * tts.sample_rate), dtype=np.float32)
    pieces = []
    for index, text in enumerate(texts):
        if index:
            pieces.append(gap)
        pieces.append(np.asarray(tts.infer(text, voice=voice_id), dtype=np.float32))
    return np.concatenate(pieces)


def write_index(out_dir, rows, texts):
    items = "\n".join(
        f"<h3>{html.escape(name)} <small>{html.escape(description)}</small></h3>\n"
        f'<audio controls preload="none" src="{html.escape(file)}"></audio>'
        for name, description, file in rows
    )
    sample_list = "".join(f"<li>{html.escape(text)}</li>" for text in texts)
    page = f"""<!doctype html>
<meta charset="utf-8">
<title>VieNeu voice comparison</title>
<body style="font-family: sans-serif; max-width: 640px; margin: 2rem auto;">
<h1>VieNeu voice comparison</h1>
<ol>{sample_list}</ol>
{items}
</body>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def run_local_checks():
    assert slugify("Thiện Minh") == "thien_minh"
    assert slugify("Đức Trí") == "duc_tri"
    assert slugify("Adam bựa") == "adam_bua"
    assert slugify("Thiền Tâm Đức") == "thien_tam_duc"


# ---------------------------------------------------------
# Main flow
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gender", choices=["male", "female", "all"], default="male")
    parser.add_argument("--text", action="append",
                        help="sample text; repeat for several (default: two built-in sentences)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main():
    run_local_checks()
    args = parse_args()
    texts = args.text or DEFAULT_TEXTS

    try:
        from vieneu import Vieneu
    except ImportError:
        raise SystemExit(
            "vieneu is not installed. Run: pip install vieneu "
            "(on Windows with an NVIDIA GPU, install CUDA PyTorch first)."
        )

    fixed = hf_cache_fix.materialize_symlinks()
    if fixed:
        print(f"Converted {fixed} cached model files from symlinks to real files "
              "(ONNX Runtime rejects the symlinks).")

    print("Loading VieNeu v3 Turbo (the first run downloads the model)...")
    tts = Vieneu()
    voices = pick_voices(tts, args.gender)
    if not voices:
        raise SystemExit(f"No preset voices match --gender {args.gender}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(voices)} voices ({args.gender}) -> {args.out}\n")

    rows = []
    for voice_id, info in voices:
        file = f"{slugify(voice_id)}.wav"
        start = time.perf_counter()
        audio = synthesize(tts, voice_id, texts)
        tts.save(audio, args.out / file)
        print(f"{voice_id}: {len(audio) / tts.sample_rate:.1f}s of audio "
              f"in {time.perf_counter() - start:.1f}s")
        rows.append((voice_id, info.get("description", ""), file))

    write_index(args.out, rows, texts)
    print(f"\nOpen {args.out / 'index.html'} in a browser to compare the voices.")


if __name__ == "__main__":
    main()