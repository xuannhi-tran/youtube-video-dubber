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
MAX_BATCH = 32  # vieneu reads at most this many sentences per GPU pass


def read_batch(value):
    """How many blocks are read in one pass: TTS_BATCH, default 1 (one block at a time).

    More only helps on the GPU (the CPU engine reads them one after another anyway) and only
    when several blocks are waiting, e.g. when a finished job is read again.
    """
    text = (value or "").strip()
    if not text:
        return 1
    try:
        number = int(text)
    except ValueError:
        number = 0
    if not 1 <= number <= MAX_BATCH:
        raise ValueError(f"TTS_BATCH must be a whole number from 1 to {MAX_BATCH}, got {value!r}")
    return number


BATCH = read_batch(os.getenv("TTS_BATCH"))
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
        # Always name the device: a bare Vieneu() would switch to the GPU by itself as soon as
        # PyTorch with CUDA is installed, while MODEL (clip cache, calibration) still said "cpu".
        self.engine = Vieneu(device=DEVICE)
        self.voice_id = find_voice(self.engine.list_preset_voices(), voice)
        self.sample_rate = self.engine.sample_rate

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def synthesize_many(self, items):
        """items = [(text, audio_file)]; read them in one pass and return the durations in order.

        If the batch fails (typically CUDA out of memory next to Ollama), the blocks are read
        one by one instead of stopping the run.
        """
        if len(items) == 1:
            return [self.synthesize(*items[0])]
        try:
            audios = self.engine.infer_batch([text for text, _ in items], voice=self.voice_id)
            if len(audios) != len(items):
                raise RuntimeError(f"infer_batch returned {len(audios)} clips for {len(items)} texts")
        except RuntimeError as error:
            print(f"Batch of {len(items)} failed ({error.__class__.__name__}: {str(error)[:120]}); "
                  "reading one by one.", flush=True)
            return [self.synthesize(text, audio_file) for text, audio_file in items]
        return [self.save_audio(audio, audio_file) for audio, (_, audio_file) in zip(audios, items)]

    def synthesize(self, text, audio_file):
        """Write `text` read aloud to `audio_file` (atomically) and return its duration."""
        return self.save_audio(self.engine.infer(text, voice=self.voice_id), audio_file)

    def save_audio(self, audio, audio_file):
        # soundfile picks the format from the extension, so the temp name must end in .wav.
        temporary_file = audio_file.with_name(audio_file.stem + ".tmp.wav")
        self.engine.save(audio, temporary_file)
        duration = measure_wav(temporary_file)
        temporary_file.replace(audio_file)
        return duration


def _check_batching():
    import tempfile

    class Engine:
        def __init__(self, fail=False):
            self.fail, self.batches, self.singles = fail, [], []

        def infer_batch(self, texts, voice):
            self.batches.append(list(texts))
            if self.fail:
                raise RuntimeError("CUDA out of memory")
            return [[0.0] * (100 * (i + 1)) for i in range(len(texts))]

        def infer(self, text, voice):
            self.singles.append(text)
            return [0.0] * 100

        def save(self, audio, path):
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(100)
                out.writeframes(b"\0\0" * len(audio))

    with tempfile.TemporaryDirectory() as folder:
        files = [Path(folder) / f"b{i}.wav" for i in range(3)]
        for fail in (False, True):
            narrator = object.__new__(Narrator)
            narrator.engine, narrator.voice_id = Engine(fail), "v"
            durations = narrator.synthesize_many([(f"t{i}", f) for i, f in enumerate(files)])
            assert all(f.exists() for f in files) and not list(Path(folder).glob("*.tmp.wav"))
            if fail:  # out of memory: every block is still read, one by one
                assert narrator.engine.singles == ["t0", "t1", "t2"] and durations == [1.0, 1.0, 1.0]
            else:     # one pass, durations follow the order of the texts
                assert narrator.engine.batches == [["t0", "t1", "t2"]] and durations == [1.0, 2.0, 3.0]
        one = object.__new__(Narrator)
        one.engine, one.voice_id = Engine(), "v"
        assert one.synthesize_many([("solo", files[0])]) == [1.0] and not one.engine.batches


def run_local_checks():
    assert read_device(None) == "cpu" and read_device("") == "cpu" and read_device(" CUDA ") == "cuda"
    try:
        read_device("gpu")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown TTS_DEVICE must be rejected")
    assert read_batch(None) == 1 and read_batch("") == 1 and read_batch(" 16 ") == 16
    for bad in ("0", "33", "x", "2.5"):
        try:
            read_batch(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted TTS_BATCH {bad!r}")
    _check_batching()
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
