# YouTube Video Dubber (Chinese → Vietnamese)

A personal prototype that dubs a Chinese video into Vietnamese. I built it because I wanted to watch Chinese knowledge videos without reading subtitles the whole time.

It's an offline pipeline, not real-time (the repo name says "live", but it isn't). You give it a video, wait a few minutes, and get back an MP4 with a Vietnamese voice track.

**Status:** works end to end on a 72-second demo clip. It has not been tested on long videos or other languages.

## How it works

```
video -> transcribe -> translate -> group into dub blocks -> TTS -> fit to timing -> assemble track -> mux into video
```

| Stage                  | Script                               | What it does                                                                                                                                                   |
| ---------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Transcribe + translate | `backend/pipeline.py`                | faster-whisper (large-v3) transcribes the Chinese speech. Gemini translates it in ~20s context windows. Terms come from `glossary.py` so they stay consistent. |
| Group                  | `backend/dub_block_builder.py`       | Merges segments into blocks of roughly 4-10s, cutting at sentence boundaries.                                                                                  |
| TTS benchmark          | `backend/test_block_tts_duration.py` | Generates speech for each block with Gemini TTS and measures how long it takes compared with the original time slot. Caches audio by text hash.                |
| Silence analysis       | `backend/analyze_tts_silence.py`     | Measures leading/trailing silence in each clip.                                                                                                                |
| Fit                    | `backend/fit_blocks.py`              | Rewrites translations that are too long for their slot, using a word budget based on measured timings. Dry run by default.                                     |
| Assemble               | `backend/assemble_dub.py`            | Trims silence, speeds clips up slightly (capped), and places them on the original timeline.                                                                    |
| Mux                    | `backend/mux_dub.py`                 | Mixes the dub into the video with ffmpeg.                                                                                                                      |

## Why the "fit" step exists

Vietnamese speech tends to run longer than Chinese in the same time slot. On the demo clip, the first translation overran its slots by 35% on average, so the dub would drift out of sync.

Two things helped:

- **Shorter rewrites.** A word budget per block (calculated from the measured TTS timings) is given to Gemini, which rewrites only the blocks that are too long.
- **Silence trimming and a capped speed-up** (1.1x base, 1.3x max).

Results on the demo clip (13 blocks, one video, Gemini voice "Charon"):

|                   | FIT / SPEED_UP / TOO_LONG | Average ratio (TTS length / slot) |
| ----------------- | ------------------------- | --------------------------------- |
| First translation | 1 / 4 / 8                 | 1.35                              |
| After rewriting   | 1 / 7 / 5                 | 1.18                              |

After trimming and speed-up, 1 of 13 blocks hit the speed cap, the largest drift was 0.09s, and the final drift was 0.00s (track length 71.86s vs 71.96s original).

This is a small sample. I wouldn't expect the same numbers on a different video or voice.

## Setup

You need Python 3.10+, ffmpeg and ffprobe on your PATH, a Gemini API key, and a CUDA GPU (`pipeline.py` runs Whisper on `cuda`; change `device` in the script if you don't have one).

```
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install faster-whisper google-genai python-dotenv
```

Create `backend/.env`:

```
GEMINI_API_KEY=your_key_here
```

`.env`, audio, video and generated output are git-ignored. You supply your own input video.

## Running it

From `backend/`. Put the audio at `xiaowulang_original.mp3` and the video at `xiaowulang_original.mp4` (the filename is currently hardcoded in `pipeline.py`, and only the first 90 seconds are processed).

```
python pipeline.py
python dub_block_builder.py
python test_block_tts_duration.py
python fit_blocks.py --target 1.2 --blocks 1-13            # dry run, no API calls
python fit_blocks.py --target 1.2 --blocks 1-13 --apply    # rewrites long blocks with Gemini
python test_block_tts_duration.py dub_blocks_fitted.json
python assemble_dub.py --original xiaowulang_original.mp3
python mux_dub.py --video xiaowulang_original.mp4
```

The output goes to `backend/dub_output/`: the dub track, a preview mix, and `dubbed.mp4`.

## Using a local LLM (Ollama) instead of Groq

Groq's free tier has a daily token limit. Translation (`pipeline.py`) and shortening (`fit_blocks.py`, and the LLM worker in `speech.py`) can instead use a model on your own computer through Ollama or LM Studio. Local models have no quota and no key. Nothing changes unless you set the variables below: the default is still Groq.

**1. Install Ollama and pull a model** (Windows: download the installer from https://ollama.com/download, then in a terminal):

```
ollama pull qwen3:8b             # recommended, about 5 GB
ollama pull qwen2.5:7b-instruct  # alternative, a bit faster, less careful with long context
ollama list                      # check it is there
```

Both are Q4 builds that fit an 8 GB GPU. Whisper large-v3 also uses the GPU, but only at the start, and Ollama unloads its model after a few idle minutes, so the stages mostly take turns. If you run out of VRAM, set `OLLAMA_KEEP_ALIVE=0` or pick a smaller model such as `qwen3:4b`.

**2. Choose it in `backend/.env`** (keep your keys, add the lines you need):

```
# everything local
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b

# or mix: translate with Groq, shorten locally (shortening is many short calls)
FIT_LLM_PROVIDER=ollama
FIT_LLM_MODEL=qwen3:8b

# or the other way round
TRANSLATE_LLM_PROVIDER=ollama
TRANSLATE_LLM_MODEL=qwen2.5:7b-instruct
```

`TRANSLATE_LLM_*` only affects translation, `FIT_LLM_*` only shortening, and `LLM_*` is the fallback for both. Setting a provider without a model uses that provider's default (`qwen3:8b` for Ollama).

| Variable | Meaning |
| --- | --- |
| `LLM_PROVIDER` | `groq`, `gemini`, `ollama` or `lmstudio` |
| `LOCAL_LLM_URL` | default `http://localhost:11434/v1` (Ollama) or `http://localhost:1234/v1` (LM Studio) |
| `LOCAL_LLM_TIMEOUT` | seconds to wait for one answer (default 600) |
| `LOCAL_LLM_THINK` | `1` lets Qwen3 reason before answering; off by default because it is slow |

LM Studio: load a model, start the server in the Developer tab, set `LLM_PROVIDER=lmstudio` and `LLM_MODEL` to the model identifier shown there.

**3. Run it**

```
python llm.py                                   # checks the logic and makes one real call per job
python dub.py "https://www.youtube.com/watch?v=..."
python dub.py URL --redo transcribe             # translate everything again with the current model
python compare_translators.py 2ZA7QtaFo4Q --provider ollama --models qwen3:8b,qwen2.5:7b-instruct
```

`dub.py` checks before starting that Ollama is running and that the model has been pulled.

**What to expect from 7-8B models**

- Their JSON is less reliable. The parser accepts code fences, `<think>` blocks, chatter around the array, trailing commas and an answer cut off before `]`. A bad answer is asked again with a note on what was wrong, and a chunk that still fails is split in two halves. If it still fails, the run stops and the finished chunks are kept.
- Translation quality is lower than gpt-oss-120b, especially for names and terms. Put them in the job's `glossary.json`.
- Chunks already translated by another model are kept when you switch (same transcript, glossary and prompt version). Add `--retranslate` to `pipeline.py` to translate them again.

## Limitations

- One demo clip, hardcoded filenames, first 90 seconds only.
- Free-tier Gemini quotas limit how much TTS you can generate per day. The scripts cache audio and stop cleanly on a daily quota error, and you can resume the next day.
- The word budget depends on the voice and the speaking rate I measured, so it needs recalibrating for another voice.
- Pronunciation and tone of the TTS voice are as good as Gemini's, and I haven't tuned them.
- Some wording differs between blocks (for example, "Kitô giáo" vs "Cơ Đốc giáo"). The glossary fixes this for new translations but I haven't regenerated the old ones.

## Notes on content

The demo uses a video by someone else for testing only.
