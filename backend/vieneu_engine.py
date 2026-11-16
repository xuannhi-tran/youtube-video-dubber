"""Local VieNeu TTS for the dub pipeline: one engine, one fixed voice, no API quota.

    with Narrator() as narrator:
        duration = narrator.synthesize("Xin chào.", Path("clip.wav"))

The output is a mono 16-bit PCM WAV at the engine's sample rate (48 kHz for
VieNeu v3 Turbo), which is what assemble_dub.py expects.
"""

import os
from pathlib import Path
import re
import unicodedata
import wave

from dotenv import load_dotenv

import hf_cache_fix  # keep before vieneu: it must set its env var before huggingface_hub loads
from workdir import SCRIPT_DIR


load_dotenv(SCRIPT_DIR / ".env")

def read_device(value):
    """'cpu' (default) or 'cuda' from TTS_DEVICE; anything else is an error."""
    device = (value or "").strip().lower() or "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError(f"TTS_DEVICE must be cpu or cuda, got {value!r}")
    return device


# TTS_DEVICE=cuda reads with PyTorch on the GPU (needs torch with CUDA, see the README); the default
# keeps the ONNX CPU engine. The GPU reads slightly differently, so its clips and speaking-rate
# calibration are kept apart from the CPU ones (MODEL is part of both).
DEVICE = read_device(os.getenv("TTS_DEVICE"))
MODEL = "vieneu-v3-turbo" + ("" if DEVICE == "cpu" else f"-{DEVICE}")
# Slug of the preset voice name (see slugify()). TTS_VOICE in .env overrides it, e.g. TTS_VOICE=thien_minh;
# `python compare_voices.py` lists the voices and lets you listen to them. Clips are stored with the voice
# they were read with, so changing it reads every block again.
VOICE = os.getenv("TTS_VOICE", "").strip() or "xuan_vinh"


def slugify(name):
    """ASCII slug of a Vietnamese voice name: 'Hải Đăng' -> 'hai_dang'."""
    text = unicodedata.normalize("NFKD", name.replace("đ", "d").replace("Đ", "D"))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def find_voice(voices, wanted):
    """voices is [(label, voice_id)]; return the voice_id whose slug matches `wanted`."""
    for label, voice_id in voices:
        if wanted in (slugify(voice_id), slugify(label)):
            return voice_id
    available = ", ".join(slugify(voice_id) for _label, voice_id in voices)
    raise ValueError(f"No preset voice matches {wanted!r}. Available: {available}")


def device_problem(device=DEVICE):
    """None when `device` can be used; otherwise what to fix (checked before a long run)."""
    if device != "cuda":
        return None
    try:
        import torch
    except ImportError:
        return ("TTS_DEVICE=cuda needs PyTorch with CUDA, which is not installed. "
                "See 'Reading on the GPU' in the README, or remove TTS_DEVICE from .env.")
    if not torch.cuda.is_available():
        return ("TTS_DEVICE=cuda but PyTorch cannot see a CUDA GPU (is this the CPU build of torch?). "
                "Install the CUDA build (README, 'Reading on the GPU') or remove TTS_DEVICE from .env.")
    return None


def measure_wav(path):
    """Duration in seconds of a PCM WAV; raises ValueError for empty audio."""
    with wave.open(str(path), "rb") as wav_file:
        frames, rate = wav_file.getnframes(), wav_file.getframerate()
    if frames <= 0 or rate <= 0:
        raise ValueError("Empty or invalid WAV audio")
    return frames / rate


class Narrator:
    def __init__(self, voice=VOICE):
        try:
            from vieneu import Vieneu
        except ImportError:
            raise SystemExit("vieneu is not installed. Run: pip install vieneu")
        # Cached model files that are symlinks are rejected by ONNX Runtime.
        hf_cache_fix.materialize_symlinks()
        problem = device_problem()
        if problem:
            raise SystemExit(problem)
        self.engine = Vieneu(device="cuda") if DEVICE == "cuda" else Vieneu()
        self.voice_id = find_voice(self.engine.list_preset_voices(), voice)
        self.sample_rate = self.engine.sample_rate

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def synthesize(self, text, audio_file):
        """Write `text` read aloud to `audio_file` (atomically) and return its duration."""
        audio = self.engine.infer(text, voice=self.voice_id)
        # soundfile picks the format from the extension, so the temp name must end in .wav.
        temporary_file = audio_file.with_name(audio_file.stem + ".tmp.wav")
        self.engine.save(audio, temporary_file)
        duration = measure_wav(temporary_file)
        temporary_file.replace(audio_file)
        return duration


def run_local_checks():
    assert read_device(None) == "cpu" and read_device("") == "cpu" and read_device(" CUDA ") == "cuda"
    try:
        read_device("gpu")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown TTS_DEVICE must be rejected")
    assert device_problem("cpu") is None
    assert MODEL.startswith("vieneu-v3-turbo")
    assert slugify("Hải Đăng") == "hai_dang"
    assert slugify("hai_dang") == "hai_dang"
    voices = [("Hải Đăng (Nam)", "Hải Đăng"), ("Thiện Minh (Nam)", "Thiện Minh")]
    assert find_voice(voices, "hai_dang") == "Hải Đăng"
    try:
        find_voice(voices, "nobody")
    except ValueError as error:
        assert "hai_dang, thien_minh" in str(error)
        return
    raise AssertionError("find_voice accepted an unknown voice")


if __name__ == "__main__":
    run_local_checks()
    print("vieneu_engine local checks passed")
