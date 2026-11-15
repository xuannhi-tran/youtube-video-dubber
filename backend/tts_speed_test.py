"""Measure how fast VieNeu reads on CPU and on GPU (if PyTorch with CUDA is installed).

    python tts_speed_test.py              # CPU and GPU
    python tts_speed_test.py --only cuda  # one device

Prints, per device, the load time and the "speed" = seconds of audio per second of
compute (above 1.0 means faster than real time). Nothing is written to disk.
"""

import argparse
import time

import hf_cache_fix  # keep before vieneu: it must set its env var before huggingface_hub loads
from vieneu_engine import VOICE, find_voice


SENTENCES = [
    "Hôm nay chúng ta sẽ cùng nhau tìm hiểu một câu chuyện rất thú vị đã xảy ra từ rất lâu rồi.",
    "Có một người đàn ông sống trên núi, mỗi ngày ông đều xuống suối lấy nước.",
    "Điều quan trọng không phải là bạn đi nhanh đến mức nào, mà là bạn có đi đúng hướng hay không.",
    "Vì vậy, khi gặp khó khăn, hãy bình tĩnh nhìn lại và tự hỏi mình thật sự muốn gì.",
]


def cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def seconds_of(audio, rate):
    return len(audio) / rate


def measure(device):
    from vieneu import Vieneu

    hf_cache_fix.materialize_symlinks()
    started = time.perf_counter()
    engine = Vieneu(device=device)
    load = time.perf_counter() - started
    voice = find_voice(engine.list_preset_voices(), VOICE)
    rate = engine.sample_rate
    print(f"\n[{device}] backend={getattr(engine, 'backend', '?')} | loaded in {load:.1f}s | voice {voice}")

    engine.infer(SENTENCES[0], voice=voice)  # warm-up (first call builds caches / CUDA graphs)

    audio_total, compute_total = 0.0, 0.0
    for text in SENTENCES:
        started = time.perf_counter()
        audio = engine.infer(text, voice=voice)
        took = time.perf_counter() - started
        length = seconds_of(audio, rate)
        audio_total += length
        compute_total += took
        print(f"  one by one: {length:5.2f}s audio in {took:5.2f}s ({length / took:5.2f}x)")
    print(f"  one by one total: {audio_total / compute_total:.2f}x real time")

    if hasattr(engine, "infer_batch"):
        texts = SENTENCES * 2
        started = time.perf_counter()
        audios = engine.infer_batch(texts, voice=voice)
        took = time.perf_counter() - started
        length = sum(seconds_of(audio, rate) for audio in audios)
        print(f"  batch of {len(texts)}: {length:5.2f}s audio in {took:5.2f}s ({length / took:5.2f}x)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["cpu", "cuda"], help="measure just this device")
    args = parser.parse_args()

    devices = [args.only] if args.only else ["cpu", "cuda"]
    if "cuda" in devices and not cuda_available():
        print("PyTorch with CUDA is not available: skipping the GPU.")
        devices.remove("cuda")
    print(f"Voice: {VOICE}")
    for device in devices:
        measure(device)


if __name__ == "__main__":
    main()
