"""Fit Vietnamese dub blocks to their time slots by word budget.

Default is a DRY RUN with no API calls. Pass --apply to rewrite over-budget
blocks with the LLM from llm.py (text only). dub_blocks.json is never modified.
"""

import argparse
import json
import math
from pathlib import Path
import re
import statistics

import llm
from workdir import WORKDIR


BASE_DIR = WORKDIR
INPUT_FILE = BASE_DIR / "dub_blocks.json"
RESULTS_FILE = BASE_DIR / "block_tts_duration_results.json"
OUTPUT_FILE = BASE_DIR / "dub_blocks_fitted.json"
DEFAULT_WORDS_PER_SECOND = 4.0
# How much longer than its slot a block may be read before it is shortened.
# The dub does not have to follow the original exactly: assemble_dub.py lets it
# run a few seconds behind or ahead, and speeds up only when it falls too far
# behind. So a block can overrun a little, and shortening (which loses meaning)
# is kept for the clear cases.
TARGET = 1.15
STRICT_FACTOR = 0.9
# A rewrite far below its budget lost content (or was cut off mid-answer).
MIN_USEFUL_FRACTION = 0.5
MIN_CHECKED_BUDGET = 20  # tiny budgets are legitimately met by very short texts


# ---------------------------------------------------------
# Pure helpers (covered by run_local_checks)
# ---------------------------------------------------------

def word_count(text):
    return len(text.split())


def budget_words(slot, words_per_second, target=TARGET):
    return max(1, math.floor(slot * words_per_second * target))


def parse_block_ids(spec):
    """Parse '1-13' or '1,3,5-7' into a set of block ids; None means all."""
    if spec is None:
        return None
    ids = set()
    for part in spec.split(","):
        low, _, high = part.strip().partition("-")
        low, high = int(low), int(high or low)
        if low > high:
            raise ValueError(f"Invalid range: {part}")
        ids.update(range(low, high + 1))
    return ids


def predicted_ratio(words, words_per_second, slot):
    return words / words_per_second / slot


def calibrate(results):
    """Return (median over all blocks, median over FIT+SPEED_UP) in words/s.

    Either value is None when no usable rows exist.
    """
    all_rates, fit_rates = [], []
    for row in results if isinstance(results, list) else []:
        try:
            rate = word_count(row["translated_text"]) / row["tts_duration"]
            classification = row["classification"]
        except (KeyError, TypeError, ZeroDivisionError, AttributeError):
            continue
        if not math.isfinite(rate) or rate <= 0:
            continue
        all_rates.append(rate)
        if classification in ("FIT", "SPEED_UP"):
            fit_rates.append(rate)
    return (
        statistics.median(all_rates) if all_rates else None,
        statistics.median(fit_rates) if fit_rates else None,
    )


def load_results(path=RESULTS_FILE):
    if not path.exists():
        return []
    try:
        results = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        return []
    return results if isinstance(results, list) else []


def load_calibration(path=RESULTS_FILE):
    return calibrate(load_results(path))


def measured_ratios(results, blocks):
    """Map block_id -> measured duration_ratio, only where the benchmarked text
    is identical to the block's current text (a stale measurement is ignored)."""
    texts = {block["block_id"]: block["translated_text"] for block in blocks}
    ratios = {}
    for row in results:
        try:
            ratio = row["duration_ratio"]
            if texts.get(row["block_id"]) == row["translated_text"] and math.isfinite(ratio) and ratio > 0:
                ratios[row["block_id"]] = ratio
        except (KeyError, TypeError):
            continue
    return ratios


def measured_budget(words, ratio, target=TARGET):
    return max(1, math.floor(words * target / ratio))


def clean_rewrite(text):
    text = (text or "").strip()
    text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text).strip()
    return text.strip("\"'“”‘’«» \n")


def within_budget(text, budget):
    return bool(text) and word_count(text) <= budget


def plausible_rewrite(text, budget):
    """False for an empty rewrite or one so short that it must have dropped most of the meaning."""
    if not text:
        return False
    return budget < MIN_CHECKED_BUDGET or word_count(text) >= budget * MIN_USEFUL_FRACTION


def block_status(words, budget):
    return "unchanged" if words <= budget else "needs_rewrite"


def pick_best(attempts, budget):
    """Return (text, status) from rewrite attempts; ignores empty ones.

    Prefers the longest attempt that fits the budget (it keeps the most meaning);
    if none fits, the shortest one.
    """
    attempts = [text for text in attempts if text]
    if not attempts:
        return None, "still_over_budget"
    fitting = [text for text in attempts if within_budget(text, budget)]
    if fitting:
        return max(fitting, key=word_count), "rewritten"
    return min(attempts, key=word_count), "still_over_budget"


# ---------------------------------------------------------
# LLM (only used with --apply)
# ---------------------------------------------------------

def build_prompt(block, limit):
    from glossary import GLOSSARY, prompt_lines

    glossary_rule = (
        f"- Glossary, use these terms exactly where relevant:\n{prompt_lines(GLOSSARY)}\n"
        if GLOSSARY else ""
    )
    return f"""
You are editing Vietnamese dubbing text so it fits a fixed time slot.

Rewrite the Vietnamese translation below in AT MOST {limit} words
(words are separated by spaces).

Requirements:
- Keep the key meaning; drop minor details, qualifiers and repetition first.
- Write complete, grammatical sentences, never a list of fragments or a telegram:
  a voice-over reads it aloud in one go, so it must flow like one person talking.
- Keep the connecting words ("vì vậy", "nhưng", "tức là"...) that carry the
  logic; join short clauses with commas instead of cutting them into pieces.
- Use natural spoken Vietnamese suitable for a voice-over.
- Use the limit well: aim for 80-100% of it, not far below.
- No parentheses, quotation marks or symbols: everything is read aloud.
- Do not invent facts that are not in the Chinese source.
{glossary_rule}- Return ONLY the rewritten Vietnamese text, no quotes, no explanation.

Chinese source (reference):
{block["source_text"]}

Current Vietnamese ({word_count(block["translated_text"])} words):
{block["translated_text"]}
"""


def rewrite_once(block, limit):
    return clean_rewrite(llm.generate(build_prompt(block, limit)))


def fit_block(block, budget):
    attempts = []
    for limit in (budget, max(1, math.floor(budget * STRICT_FACTOR))):
        text = rewrite_once(block, limit)
        if not plausible_rewrite(text, budget):
            print(f"  [{block['block_id']:02d}] rewrite of {word_count(text)} words for a "
                  f"budget of {budget} looks cut off; ignoring it")
            continue
        attempts.append(text)
        if within_budget(text, budget):
            break
        print(f"  [{block['block_id']:02d}] {word_count(text)} words > {budget}; retrying stricter")
    return pick_best(attempts, budget)


# ---------------------------------------------------------
# Main flow
# ---------------------------------------------------------

def load_blocks(path=INPUT_FILE):
    blocks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Expected a non-empty block list")
    return blocks


def plan(blocks, words_per_second, target=TARGET, measured=None):
    measured = measured or {}
    rows = []
    for block in blocks:
        slot = block["end"] - block["start"]
        words = word_count(block["translated_text"])
        predicted_budget = budget_words(slot, words_per_second, target)
        ratio = measured.get(block["block_id"])
        if ratio is None:
            source, budget = "predicted", predicted_budget
            over = words > budget
        else:
            source = "measured"
            budget = measured_budget(words, ratio, target) if ratio > target else words
            over = ratio > target
        rows.append({
            "block": block,
            "slot": slot,
            "words": words,
            "predicted_budget": predicted_budget,
            "budget": budget,
            "source": source,
            "measured_ratio": ratio,
            "ratio": predicted_ratio(words, words_per_second, slot),
            "over": over,
        })
    return rows


def print_plan(rows):
    print(f"\n{'id':>3} {'slot':>6} {'words':>5} {'old_bud':>7} {'new_bud':>7} "
          f"{'pred':>5} {'meas':>5} {'source':>9}  action")
    for row in rows:
        measured = row["measured_ratio"]
        meas = f"{measured:.2f}" if measured is not None else "-"
        action = f"rewrite to <= {row['budget']}" if row["over"] else "keep"
        print(f"{row['block']['block_id']:>3} {row['slot']:>5.2f}s {row['words']:>5} "
              f"{row['predicted_budget']:>7} {row['budget']:>7} {row['ratio']:>5.2f} "
              f"{meas:>5} {row['source']:>9}  {action}")
    over = sum(row["over"] for row in rows)
    print(f"\nOver budget: {over}/{len(rows)} blocks (= LLM calls with --apply, "
          f"plus up to {over} retries)")


def apply_fits(rows):
    fitted = []
    stopped = False
    for row in rows:
        block, words, budget = row["block"], row["words"], row["budget"]
        out = {**block, "original_translated_text": block["translated_text"]}
        status = "unchanged"
        if row["over"]:
            status = "still_over_budget"
            if not stopped:
                print(f"[{block['block_id']:02d}] {words} -> <= {budget} words...", flush=True)
                try:
                    text, status = fit_block(block, budget)
                except llm.QuotaExhausted as error:
                    print(f"Quota reached ({error}). Stopping; remaining blocks keep original text.")
                    stopped = True
                    text = None
                if text:
                    out["translated_text"] = text
        out["word_count"] = word_count(out["translated_text"])
        out["budget_words"] = budget
        out["status"] = status
        fitted.append(out)
    return fitted


def save_fitted(fitted, path=OUTPUT_FILE):
    path.write_text(json.dumps(fitted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_fitted_summary(fitted):
    print("\n--------------------------------")
    for status in ("unchanged", "rewritten", "still_over_budget"):
        print(f"{status}: {sum(block['status'] == status for block in fitted)}")
    print(f"Saved: {OUTPUT_FILE.name}")
    print("--------------------------------")
    for block in fitted:
        print(f"[{block['block_id']:02d}] {block['status']} "
              f"({block['word_count']}/{block['budget_words']} words)")


def run_local_checks():
    assert word_count("  a  b\tc\n") == 3 and word_count("") == 0
    assert budget_words(10, 4.0, 1.1) == 44
    assert budget_words(0.1, 4.0, 1.1) == 1
    assert budget_words(5.5, 4.0, 1.0) == 22
    assert budget_words(10, 4.0, 1.2) == 48
    assert parse_block_ids(None) is None
    assert parse_block_ids("1-3") == {1, 2, 3}
    assert parse_block_ids("1,4-5") == {1, 4, 5}
    assert abs(predicted_ratio(40, 4.0, 10) - 1.0) < 1e-9
    assert measured_budget(40, 2.0, 1.2) == 24
    assert measured_budget(1, 50, 1.2) == 1

    blocks = [
        {"block_id": 1, "start": 0, "end": 10, "translated_text": "a b c d"},
        {"block_id": 2, "start": 10, "end": 20, "translated_text": "a b c d"},
        {"block_id": 3, "start": 20, "end": 30, "translated_text": "a b c d"},
    ]
    results = [
        {"block_id": 1, "translated_text": "a b c d", "duration_ratio": 2.0},
        {"block_id": 2, "translated_text": "stale text", "duration_ratio": 2.0},
    ]
    measured = measured_ratios(results, blocks)
    assert measured == {1: 2.0}
    rows = plan(blocks, 4.0, 1.0, measured)
    assert [row["source"] for row in rows] == ["measured", "predicted", "predicted"]
    assert rows[0]["over"] and rows[0]["budget"] == 2
    assert not rows[1]["over"] and rows[1]["budget"] == 40
    # Measured ratio at or below target means no rewrite, whatever words/s says.
    assert not plan(blocks, 4.0, 1.2, {1: 1.2})[0]["over"]
    assert not plan(blocks, 0.1, 1.2, {1: 1.0})[0]["over"]
    assert plan(blocks, 4.0, 1.2, {1: 1.21})[0]["over"]

    rows = [
        {"translated_text": "a b c d", "tts_duration": 1.0, "classification": "FIT"},
        {"translated_text": "a b c d e f", "tts_duration": 2.0, "classification": "SPEED_UP"},
        {"translated_text": "a b", "tts_duration": 1.0, "classification": "TOO_LONG"},
    ]
    assert calibrate(rows) == (3.0, 3.5)
    assert calibrate([rows[2]]) == (2.0, None)
    assert calibrate([]) == (None, None)
    assert calibrate([{"bad": 1}, {"translated_text": "a", "tts_duration": 0, "classification": "FIT"}]) == (None, None)
    assert calibrate(None) == (None, None)
    assert load_calibration(BASE_DIR / "does_not_exist.json") == (None, None)

    assert clean_rewrite('  "Xin chào."  ') == "Xin chào."
    assert clean_rewrite("```\nXin chào\n```") == "Xin chào"
    assert clean_rewrite(None) == ""
    assert within_budget("a b c", 3) and not within_budget("a b c d", 3)
    assert not within_budget("", 3)
    assert block_status(5, 5) == "unchanged" and block_status(6, 5) == "needs_rewrite"

    assert pick_best(["a b", "a"], 2) == ("a b", "rewritten")
    assert pick_best(["a", "a b c", "a b"], 2) == ("a b", "rewritten")
    assert pick_best(["a b c d", "a b c"], 2) == ("a b c", "still_over_budget")
    assert pick_best(["", ""], 2) == (None, "still_over_budget")

    assert plausible_rewrite("a b c", 5) and not plausible_rewrite("", 5)
    assert plausible_rewrite("a", 10)  # budgets under 20 words are not checked
    assert plausible_rewrite(" ".join(["w"] * 15), 30)
    assert not plausible_rewrite(" ".join(["w"] * 14), 30)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="call the LLM and write the fitted file")
    parser.add_argument("--target", type=float, default=TARGET, help=f"budget multiplier (default {TARGET})")
    parser.add_argument("--blocks", help="only these block ids, e.g. 1-13 or 1,3,5-7")
    args = parser.parse_args()
    if not math.isfinite(args.target) or args.target <= 0:
        parser.error("--target must be positive")
    return args


def main():
    run_local_checks()
    args = parse_args()
    apply = args.apply
    blocks = load_blocks()
    selected = parse_block_ids(args.blocks)
    if selected is not None:
        blocks = [block for block in blocks if block["block_id"] in selected]
        if not blocks:
            raise ValueError("--blocks matched no blocks")

    all_wps, fit_wps = load_calibration()
    words_per_second = all_wps or DEFAULT_WORDS_PER_SECOND
    source = "all blocks" if all_wps else "default (no results file)"
    print(f"Calibrated words/s: {words_per_second:.2f} (median over {source})")
    if fit_wps:
        print(f"For comparison, median over FIT+SPEED_UP only: {fit_wps:.2f}")
    print(f"TARGET: {args.target}")
    if selected is not None:
        print(f"Blocks: {args.blocks} ({len(blocks)} selected; only these are written)")

    rows = plan(blocks, words_per_second, args.target, measured_ratios(load_results(), blocks))
    print_plan(rows)

    if not apply:
        print("\nDRY RUN: no API calls made. Pass --apply to rewrite.")
        return

    fitted = apply_fits(rows)
    save_fitted(fitted)
    print_fitted_summary(fitted)


if __name__ == "__main__":
    main()
