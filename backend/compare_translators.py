"""Translate the same chunk of a job with several models, side by side.

    python compare_translators.py 2ZA7QtaFo4Q                # chunk 5, default models
    python compare_translators.py 2ZA7QtaFo4Q --chunk 12
    python compare_translators.py 2ZA7QtaFo4Q --models qwen/qwen3.8-27b --max-tokens 900

Needs the job's transcript (dub.py makes it in the transcribe stage) and the key of
the chosen provider in .env. Writes translator_compare_output/<job>_chunk<N>.md.
"""

import argparse
import html
import json
import os
from pathlib import Path
import time

from workdir import SCRIPT_DIR


JOBS_DIR = SCRIPT_DIR / "dub_jobs_output"
DEFAULT_OUT = SCRIPT_DIR / "translator_compare_output"  # matches "*_output/" in .gitignore
DEFAULT_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]
DEFAULT_CHUNK = 5


# ---------------------------------------------------------
# Pure helpers (covered by run_local_checks)
# ---------------------------------------------------------

def count_words(text):
    return len(text.split())


def cell(text):
    """Text safe inside a Markdown table cell."""
    return html.escape(" ".join(text.split()), quote=False).replace("|", "\\|")


def render_markdown(job, number, chunk, results):
    """results: {model: {'seconds': float, 'texts': [str]} or {'error': str}}"""
    start, end = chunk[0]["start"], chunk[-1]["end"]
    models = list(results)
    lines = [
        f"# Translator comparison: {job}, chunk {number}",
        "",
        f"Time range {start:.1f}s to {end:.1f}s, {len(chunk)} segments.",
        "",
        "| # | Chinese | " + " | ".join(models) + " |",
        "|---|---|" + "---|" * len(models),
    ]
    for index, segment in enumerate(chunk):
        cells = []
        for model in models:
            result = results[model]
            cells.append(cell(result["texts"][index]) if "texts" in result else "(failed)")
        lines.append(f"| {index + 1} | {cell(segment['source_text'])} | " + " | ".join(cells) + " |")
    lines += ["", "## Summary", "", "| Model | Seconds | Words | Words per second of speech |", "|---|---|---|---|"]
    duration = max(end - start, 0.001)
    for model in models:
        result = results[model]
        if "texts" in result:
            words = sum(count_words(text) for text in result["texts"])
            lines.append(f"| {model} | {result['seconds']:.1f} | {words} | {words / duration:.2f} |")
        else:
            lines.append(f"| {model} | - | - | failed: {cell(result['error'])} |")
    return "\n".join(lines) + "\n"


def run_local_checks():
    assert count_words("  một hai\tba ") == 3
    assert cell("a | b\nc") == "a \\| b c"
    chunk = [{"start": 0.0, "end": 2.0, "source_text": "你好"},
             {"start": 2.0, "end": 4.0, "source_text": "再见"}]
    results = {
        "good": {"seconds": 1.5, "texts": ["Xin chào", "Tạm biệt các bạn"]},
        "bad": {"error": "model not found"},
    }
    page = render_markdown("job1", 3, chunk, results)
    assert "| 1 | 你好 | Xin chào | (failed) |" in page
    assert "| good | 1.5 | 6 | 1.50 |" in page
    assert "failed: model not found" in page


# ---------------------------------------------------------
# Main flow
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", help="video id, i.e. the folder name inside dub_jobs_output")
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help="1-based chunk number")
    parser.add_argument("--models", help="comma-separated model ids (default: gpt-oss-120b, qwen3.8-27b, gpt-oss-20b)")
    parser.add_argument("--provider", choices=["groq", "gemini"], default="groq")
    parser.add_argument("--max-tokens", type=int, help="cap on each answer's length (sets LLM_MAX_TOKENS)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.chunk < 1:
        parser.error("--chunk starts at 1")
    return args


def main():
    run_local_checks()
    args = parse_args()
    job = JOBS_DIR / args.job
    raw_file = job / "transcript_raw.json"
    if not raw_file.exists():
        raise SystemExit(f"No transcript at {raw_file}. Run dub.py for this video first.")

    # Both settings must be in place before pipeline and llm are imported.
    os.environ["DUB_WORKDIR"] = str(job)
    os.environ["LLM_PROVIDER"] = args.provider
    if args.max_tokens:
        os.environ["LLM_MAX_TOKENS"] = str(args.max_tokens)
    import llm
    import pipeline

    segments = json.loads(raw_file.read_text(encoding="utf-8"))["segments"]
    chunks = pipeline.create_chunks(segments)
    if args.chunk > len(chunks):
        raise SystemExit(f"This video has {len(chunks)} chunks; --chunk {args.chunk} is too high.")
    chunk = chunks[args.chunk - 1]
    models = [name.strip() for name in args.models.split(",")] if args.models else DEFAULT_MODELS

    results = {}
    for model in models:
        print(f"\n>>> {args.provider}:{model}", flush=True)
        began = time.perf_counter()
        try:
            translated = pipeline.translate_chunk(chunk, model=model)
        except (llm.QuotaExhausted, llm.LLMError, RuntimeError) as error:  # limits are per model
            print(f"Failed: {error}")
            results[model] = {"error": str(error)}
            continue
        seconds = time.perf_counter() - began
        results[model] = {"seconds": seconds, "texts": [item["text"] for item in translated]}
        print(f"Done in {seconds:.1f}s")

    if not results:
        raise SystemExit("No model produced anything.")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.job}_chunk{args.chunk}.md"
    path.write_text(render_markdown(args.job, args.chunk, chunk, results), encoding="utf-8")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
