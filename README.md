# YouTube Video Dubber (Chinese → Vietnamese)

A personal tool that dubs a Chinese YouTube video into Vietnamese. You give it a link and get back an MP4 with a Vietnamese voice-over. I built it because I wanted to watch Chinese knowledge videos without reading subtitles the whole time.

It is an offline pipeline, not real-time (the repo name says "live", but it isn't). Everything runs on your own computer except the translation, which can use Groq, Gemini or a local model (Ollama / LM Studio).

**Status:** works end to end on real videos (Groq, and a 2-minute clip with a local Ollama model). It has not been tested on other languages, on long videos with a local model, or on anything but Windows. Translation quality with a 7-8B local model is clearly below Groq's `gpt-oss-120b`.

## Quick start

```
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install yt-dlp faster-whisper vieneu httpx python-dotenv numpy
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12     # CUDA libraries for Whisper (see Setup)
```

Create `backend/.env` with a key (see [Settings](#settings)), then:

```
python dub.py "https://www.youtube.com/watch?v=VIDEO_ID" --max-seconds 120   # try the first 2 minutes
python dub.py "https://www.youtube.com/watch?v=VIDEO_ID"                      # the whole video
```

The result is `backend/dub_jobs_output/<video id>/dub_output/dubbed.mp4`.

## How it works

`dub.py` runs five stages for one video. Each video gets its own folder, `backend/dub_jobs_output/<video id>/`, so jobs never mix. Finished stages are skipped on a re-run, so after an interruption (a quota limit, a crash, Ctrl+C) you run the same command and it continues.

```
download -> transcribe + translate  ─┐
                                      ├─ (at the same time) -> assemble -> mux
            speech (shorten + TTS)  ─┘
```

| Stage | Script | What it does |
| --- | --- | --- |
| 1. download | `dub.py` | `yt-dlp` downloads the video (up to 720p by default) to `source.mp4`; `ffmpeg` extracts 16 kHz mono `source.wav`. |
| 2. transcribe + translate | `pipeline.py` | faster-whisper (`large-v3`, on the GPU) transcribes the Chinese speech. The segments are grouped into windows of about 20 seconds and the LLM translates each window into spoken Vietnamese, with the previous 8 translated segments as context. Every translated window is saved, so an interrupted run does not redo it. Glossary terms (`glossary.py`) are given to both Whisper (as hotwords) and the LLM. |
| 3. speech | `speech.py` | Starts while translation is still running (`--follow`) and reads blocks as soon as they are translated. Segments are first merged into blocks of about 4-10 s (`dub_block_builder.py`). If a block will not fit its time slot, the LLM shortens it **before** it is read (`fit_blocks.py`); then VieNeu v3 Turbo (voice `xuan_vinh`, runs locally on CPU via ONNX) reads it and the clip length is measured; a clip that is still too long is shortened once more and read again. The LLM and VieNeu work in parallel, one thread each. |
| 4. assemble | `assemble_dub.py` | Places the clips on the original timeline into `dub_output/dub_track.wav`: trims silence, keeps at least 0.25 s between clips, lets the dub drift up to 3 s from the original, reads slightly faster (up to 1.1x) when more than 1.5 s behind and up to `--max-speed` (1.3x) only to avoid falling more than 3 s behind, and brings every clip to -20 dBFS. |
| 5. mux | `mux_dub.py` | Mixes the dub track with the original audio (original at volume 0.12 by default) into `dub_output/dubbed.mp4`. If the video runs longer than the dub, the original audio returns to full volume after the dub ends. |

Each script also has its own checks that run when it starts (`run_local_checks`), and most can be run alone from inside a job folder by setting `DUB_WORKDIR` to the job folder.

### Redoing a stage

```
python dub.py URL --redo transcribe      # translate again (Whisper's saved transcript is reused)
python dub.py URL --redo speech          # forget earlier rewrites and read again
python dub.py URL --redo mux             # only the last stage
python dub.py URL --original-volume 0.04 # remix with quieter original voice (0 drops it); redoes only the mux
python dub.py URL --subs                 # burn Vietnamese subtitles in (see below); redoes only the mux
python dub.py URL --glossary-only        # create the job folder and glossary.json, then stop
```

`--redo X` redoes stage X and everything after it. Changing `--max-seconds` redoes everything after the download, because the transcript depends on it.

### Files in a job folder

| File | What it is |
| --- | --- |
| `source.mp4`, `source.wav` | Downloaded video and its audio. |
| `glossary.json` | Terms to keep fixed (see [Glossary](#glossary)). Empty by default. |
| `transcript_raw.json` | Whisper output (reused while the audio and glossary hotwords are unchanged). |
| `translate_cache.json` | Every translated window, so quota stops cost nothing. |
| `translated_partial.json`, `translated_segments.json` | Growing / final translation. |
| `dub_blocks.json`, `dub_blocks_fitted.json` | Blocks before and after shortening. |
| `block_tts_output/`, `block_tts_duration_results_fitted.json` | The clips and their measured lengths. |
| `speech_state.json`, `dub_state.json` | Progress, so every stage can resume. |
| `dub_output/` | `dub_track.wav`, per-block files, `dub_timeline.json` (when each line is spoken), `dub_subs.ass` (with `--subs`) and `dubbed.mp4`. |

Everything under `*_output/`, plus `.env`, audio and video, is git-ignored.

## Setup

You need:

- **Python 3.10+**, **ffmpeg** and **ffprobe** on your PATH.
- A **CUDA GPU** for Whisper (`pipeline.py` loads `large-v3` on `cuda` in float16; the first run downloads about 3 GB). On Windows, `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` from pip provide the libraries, and `cuda_dlls.py` makes them visible to Windows before Whisper loads.
- **VieNeu** for speech. It runs on the CPU, so there is no TTS quota, but it is slow. `hf_cache_fix.py` works around ONNX Runtime refusing Hugging Face cache symlinks.
- An **LLM** for translation and shortening: a Groq or Gemini key, or a local Ollama / LM Studio model.

`dub.py` checks all of this (packages, ffmpeg, LLM key, local server and model) before it starts, and lists what is missing.

### Vietnamese subtitles

```
python dub.py URL --subs                       # add a black bar under the video with Vietnamese subtitles
python dub.py URL --subs --sub-bar 160         # taller bar (pixels)
python dub.py URL --subs --sub-font-size 40    # bigger text (pixels of the video)
python dub.py URL --subs --sub-font Tahoma     # another installed font
```

The video gets a black bar added **below** the picture, so the original (Chinese) subtitles stay fully visible and the Vietnamese text sits right under them. The text is exactly what the voice reads (after shortening) and appears when that line is actually played in the final dub, taken from `dub_output/dub_timeline.json`. Long sentences are split over several cues of at most two lines.

- Without `--subs` nothing changes: the video is copied as before (fast). With `--subs` the video is re-encoded once (libx264, crf 18), so the mux takes longer and the file is a bit larger.
- On a finished job `--subs` only redoes the last stage. A job made before this option has no `dub_timeline.json`, so the assemble stage is redone too (quick, no LLM or voice).
- Without `--sub-bar` / `--sub-font-size` the sizes are automatic: the bar is about 17% of the video height and the text fits two lines. Vertical (TikTok) videos get a width-limited font size so lines still fit.
- The font must be installed and contain Vietnamese diacritics. The default is Arial (fine on Windows and macOS); if the letters look wrong, set another font.
- Run again without `--subs` and `--redo mux` to get the plain video back.

### Settings

Put them in `backend/.env`. Only a key is required.

```
GROQ_API_KEY=...          # default provider when this is set
GEMINI_API_KEY=...        # used when LLM_PROVIDER=gemini
```

| Variable | Meaning |
| --- | --- |
| `LLM_PROVIDER` | `groq` (default), `gemini`, `ollama` or `lmstudio`. |
| `LLM_MODEL` | Model name; each provider has a default (`openai/gpt-oss-120b` for Groq, `gemini-3.5-flash-lite` for Gemini, `qwen3:8b` for Ollama). |
| `TRANSLATE_LLM_PROVIDER`, `TRANSLATE_LLM_MODEL` | Override the two above for translation only. |
| `FIT_LLM_PROVIDER`, `FIT_LLM_MODEL` | Override them for shortening only (`fit_blocks.py` and `speech.py`). |
| `LLM_MAX_TOKENS` | Cap on the answer length (some Groq models allow very little per minute). |
| `GROQ_API_URL` | Any OpenAI-compatible chat endpoint instead of Groq's. |
| `GEMINI_RPM` | Requests per minute this process may send to Gemini (default 10, the free Flash Lite limit is 15). |
| `LOCAL_LLM_URL`, `LOCAL_LLM_TIMEOUT`, `LOCAL_LLM_THINK` | Local server settings, see below. |
| `SUB_FONT`, `SUB_FONT_SIZE`, `SUB_BAR` | Defaults for `--subs`: font name (default `Arial`), text size and bar height in pixels (0 = automatic). The command-line options win. |
| `TTS_VOICE` | VieNeu preset voice, default `xuan_vinh` (e.g. `thien_minh`). `python compare_voices.py` lets you listen to every voice. Changing it reads all blocks again (clips are keyed by voice); the speech rate is calibrated per voice. |

`python llm.py` checks the logic and makes one real call for translation and one for shortening, which is the quickest way to test your settings. `python llm.py --list` lists the models your Groq key can use, `python llm.py --list gemini` the ones your Gemini key can use (copy the name exactly). `backend/.env.example` has every setting with a short comment.

### Which LLM for what

Translation needs the model to understand context (names, terms, sentence boundaries) and is the stage where small local models struggle most; shortening is many short calls and uses quota quickly. A setup that spreads the free quotas:

```
TRANSLATE_LLM_PROVIDER=gemini
TRANSLATE_LLM_MODEL=gemini-3.5-flash-lite
FIT_LLM_PROVIDER=groq
FIT_LLM_MODEL=openai/gpt-oss-120b
```

with a local Ollama model as the fallback: if a stage stops on a daily limit, run again with `TRANSLATE_LLM_PROVIDER=ollama` (or `FIT_LLM_PROVIDER=ollama`); windows already translated by another model are kept. Gemini limits differ a lot between models and accounts (a lite model usually allows far more requests per minute and per day than a full one); check yours in AI Studio and pick the model accordingly. A per-minute limit is waited out automatically; a per-day limit stops the run with a message, keeping everything finished (the free tier resets at midnight Pacific time). Requests are spaced to `GEMINI_RPM`; translation and speech run as separate processes, so keep that in mind if both use Gemini.

### Glossary

`dub.py` creates an empty `glossary.json` (a JSON list of strings) in each job folder. Edit it **before** the transcribe stage to pin names and terms:

```json
["khí chất", "tạp chí Time", "MBTI", "Fe (cảm xúc hướng ngoại)"]
```

An entry may carry its meaning in parentheses: the LLM explains it once in words and afterwards uses only the term.

An entry can also say what it stands for in the original speech, with `=>`. This is how Chinese names get their Sino-Vietnamese (Hán Việt) reading and how a word gets one fixed translation:

```json
["黄晓明 => Hoàng Hiểu Minh", "吴亦凡|Chris Wu => Ngô Diệc Phàm", "酷盖 => người ngầu", "态度 => thái độ"]
```

`|` separates several spellings of the same source. Such an entry is only given to the LLM for the passages whose original text contains its source (so one entry's term cannot leak into a passage about something else), and its sources are given to Whisper as hotwords, which helps it write the name correctly. Plain entries in Latin letters (MBTI, Fe) are used only where they occur in the original; a plain Vietnamese entry is always given, and the prompt tells the LLM to use it only where the speaker really means it. Without an entry, Chinese names are asked to be written in Hán Việt and foreign names in their original spelling, which a small model does unreliably, so add the names that matter.

If a job has no `glossary.json`, a default list of Bible-related terms from the first demo video is used. Only the glossary entries that apply to a window are part of its cache key, so adding a name does not translate windows again that do not contain it; but adding a source also changes Whisper's hotwords, which re-runs the transcription once.

### Pronunciation

An optional `pronunciations.json` (in `backend/`, or in a job folder to override it) maps text to how it should sound, e.g. `{"ENFJ": "e n f j"}`. It is applied just before reading.

## Using a local LLM (Ollama) instead of Groq

Groq's free tier has a daily token limit. Translation and shortening can instead use a model on your own computer through Ollama or LM Studio. Local models have no quota and no key. Nothing changes unless you set the variables below: the default is still Groq.

**1. Install Ollama and pull a model** (Windows: installer from https://ollama.com/download, then in a terminal):

```
ollama pull qwen3:8b             # recommended, about 5 GB
ollama pull qwen2.5:7b-instruct  # alternative
ollama list                      # check it is there
```

To keep models on another drive, set `OLLAMA_MODELS` (e.g. `setx OLLAMA_MODELS "D:\ollama\models"`), restart Ollama and open a new terminal before pulling.

Both are Q4 builds that fit an 8 GB GPU. Whisper also uses the GPU, but only at the start, and Ollama unloads its model after a few idle minutes, so the stages mostly take turns. If you run out of VRAM, pick a smaller model such as `qwen3:4b`.

**2. Choose it in `backend/.env`:**

```
# everything local
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b

# or mix: translate with Groq, shorten locally (shortening is many short calls)
FIT_LLM_PROVIDER=ollama
FIT_LLM_MODEL=qwen3:8b
```

| Variable | Meaning |
| --- | --- |
| `LOCAL_LLM_URL` | Default `http://localhost:11434/v1` (Ollama) or `http://localhost:1234/v1` (LM Studio). |
| `LOCAL_LLM_TIMEOUT` | Seconds to wait for one answer (default 600). |
| `LOCAL_LLM_THINK` | `1` lets Qwen3 reason before answering; off by default because it is slow. |

LM Studio: load a model, start the server in the Developer tab, set `LLM_PROVIDER=lmstudio` and `LLM_MODEL` to the identifier shown there.

**3. Run it**

```
python llm.py
python dub.py "https://www.youtube.com/watch?v=VIDEO_ID" --max-seconds 120
python compare_translators.py JOB_ID --provider ollama --models qwen3:8b,qwen2.5:7b-instruct
```

**What to expect from 7-8B models**

- Their JSON is less reliable. The parser accepts code fences, `<think>` blocks, chatter around the array, trailing commas and an answer cut off before `]`. A bad answer is asked again with a note on what was wrong, and a window that still fails is split in two halves. If it still fails, the run stops and the finished windows are kept.
- A translation that still contains Chinese characters is treated as wrong and asked again (the voice cannot read them); a single segment that still has some after all retries has those characters removed.
- Windows already translated by another model are kept when you switch (same transcript, glossary and prompt version). `pipeline.py --retranslate` translates them again with the current model.
- Names and terms are weaker than with Groq: use the glossary. Speech-recognition mistakes in names are sometimes translated literally.

## Other scripts

| Script | Use |
| --- | --- |
| `compare_translators.py JOB_ID` | Translates the same window of a job with several models, side by side, into `translator_compare_output/`. |
| `compare_voices.py` | Reads a sample with every VieNeu preset voice and writes an `index.html` to listen and compare. |
| `tts_speed_test.py` | Measures VieNeu speed on CPU vs GPU. |
| `fit_blocks.py` | The shortening step on its own. Dry run by default; `--apply` calls the LLM. `speech.py` uses it too. |
| `test_block_tts_duration.py`, `test_tts_duration.py`, `analyze_tts_silence.py` | Benchmark and silence analysis of clips; `speech.py` reuses their helpers. |
| `test_tts.py`, `test_translate.py`, `test_elevenlabs.py`, `test_transcribe.py` | Early prototypes (Gemini, ElevenLabs, Whisper). Not part of the pipeline; some need extra packages (`google-genai`, `elevenlabs`) and keys. |

## Limitations

- Chinese → Vietnamese only, one VieNeu voice per run (`xuan_vinh` by default, see `TTS_VOICE`).
- The original audio is only turned down, not removed: voices are not separated from music. Use `--original-volume` to taste (0 drops it).
- VieNeu on CPU is slow, so a long video takes a long time to read. GPU speech is not implemented.
- Whisper needs a CUDA GPU. Without one, change `device` in `pipeline.py` (slower).
- Vietnamese is longer than Chinese in the same time, so some blocks are shortened by the LLM, which can lose detail; some blocks over about 12 s have no sentence break to split at. Timing is kept by letting the dub drift (up to 3 s) and by small speed-ups rather than exact sync.
- The Groq free tier stops at its daily limit; the run keeps what it has and continues when you run the same command later.
- Quality depends on the LLM and on speech-recognition errors. Check the result before sharing it.

## Notes on content

Dub only videos you have the right to use. The demo videos used for testing belong to someone else and are not included.
