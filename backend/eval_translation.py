"""Score a job's Vietnamese translation against a reference you corrected by hand.

    python eval_translation.py JOB_ID --make-reference --from 60 --to 180
        copies the translated lines between 60 s and 180 s into dub_jobs_output/JOB_ID/reference.json;
        open it and fix "translated_text" until every line is how it should be said
    python eval_translation.py JOB_ID
        scores the job's current translated_segments.json against reference.json
    python eval_translation.py JOB_ID --candidate old.json --candidate new.json
        scores several versions side by side (e.g. saved copies before and after a prompt change)

The score is chrF (character n-gram F-score, 0 to 100, higher is closer to the reference), a
standard translation metric that works well for Vietnamese because it does not need word
segmentation. It also lists the lines furthest from the reference, so you can see what changed.
Nothing is sent anywhere; no LLM is called.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from workdir import SCRIPT_DIR


JOBS_DIR = SCRIPT_DIR / "dub_jobs_output"
REFERENCE_NAME = "reference.json"
CHAR_ORDER = 6
BETA = 2  # chrF2: recall counts twice as much as precision


def normalize(text):
    return " ".join(str(text or "").lower().split())


def char_ngrams(text, n):
    text = text.replace(" ", "")
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def chrf(candidates, references, order=CHAR_ORDER, beta=BETA):
    """Corpus chrF (0-100) of parallel lists of strings."""
    precisions, recalls = [], []
    for n in range(1, order + 1):
        matched = total_candidate = total_reference = 0
        for candidate, reference in zip(candidates, references):
            hyp, ref = char_ngrams(normalize(candidate), n), char_ngrams(normalize(reference), n)
            matched += sum((hyp & ref).values())
            total_candidate += sum(hyp.values())
            total_reference += sum(ref.values())
        if total_candidate and total_reference:
            precisions.append(matched / total_candidate)
            recalls.append(matched / total_reference)
    if not precisions:
        return 0.0
    precision, recall = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    if precision + recall == 0:
        return 0.0
    return 100 * (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)


def load_segments(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        data = data["segments"]
    if not isinstance(data, list):
        raise SystemExit(f"{path} is not a list of segments")
    return data


def aligned(reference, candidate):
    """(reference text, candidate text, segment) for every reference line, matched by start time."""
    by_start = {round(float(item["start"]), 2): item for item in candidate}
    result = []
    for item in reference:
        other = by_start.get(round(float(item["start"]), 2))
        if other is not None:
            result.append((item["translated_text"], other.get("translated_text", ""), item))
    return result


def score(reference, candidate):
    pairs = aligned(reference, candidate)
    if not pairs:
        return None
    total = chrf([c for _, c, _ in pairs], [r for r, _, _ in pairs])
    lines = sorted(((chrf([c], [r]), r, c, item) for r, c, item in pairs), key=lambda row: row[0])
    words = sum(len(c.split()) for _, c, _ in pairs) / max(1, sum(len(r.split()) for r, _, _ in pairs))
    return {"chrf": total, "lines": len(pairs), "missing": len(reference) - len(pairs),
            "length": words, "worst": lines}


def make_reference(job, start, end):
    source = job / "translated_segments.json"
    if not source.exists():
        raise SystemExit(f"{source} not found: translate the video first (dub.py).")
    picked = [{"start": s["start"], "end": s["end"], "source_text": s["source_text"],
               "translated_text": s["translated_text"]}
              for s in load_segments(source) if start <= float(s["start"]) < end]
    if not picked:
        raise SystemExit("No line in that time range.")
    path = job / REFERENCE_NAME
    if path.exists():
        raise SystemExit(f"{path} already exists; delete it first if you want a new one.")
    path.write_text(json.dumps(picked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(picked)} lines to {path}.\nEdit every \"translated_text\" until it is how the line "
          "should be said, then run: python eval_translation.py " + job.name)


def run_local_checks():
    assert round(chrf(["xin chào các bạn"], ["xin chào các bạn"]), 6) == 100.0
    assert chrf(["abc"], ["xyz"]) == 0.0
    close = chrf(["thiên tài Nietzsche cũng từng nói"], ["thiên tài Nietzsche từng nói"])
    far = chrf(["thiên tài của chúng ta Nietzsche cũng đã kể"], ["thiên tài Nietzsche từng nói"])
    assert 0 < far < close < 100
    reference = [{"start": 1.0, "end": 2.0, "translated_text": "Trời ạ."},
                 {"start": 2.0, "end": 3.0, "translated_text": "Con ếch đó."}]
    candidate = [{"start": 1.0, "translated_text": "Thật đấy."}, {"start": 2.0, "translated_text": "Con ếch đó."}]
    result = score(reference, candidate)
    assert result["lines"] == 2 and result["missing"] == 0 and result["worst"][0][1] == "Trời ạ."
    assert score(reference, [{"start": 9.0, "translated_text": "x"}]) is None


def main():
    run_local_checks()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job", help="video id, i.e. the folder name inside dub_jobs_output")
    parser.add_argument("--make-reference", action="store_true", help="write reference.json to edit by hand")
    parser.add_argument("--from", dest="start", type=float, default=0.0, help="first second (with --make-reference)")
    parser.add_argument("--to", dest="end", type=float, default=float("inf"), help="last second (with --make-reference)")
    parser.add_argument("--candidate", action="append", type=Path,
                        help="translated_segments-style JSON to score (repeatable; default: the job's own)")
    parser.add_argument("--worst", type=int, default=8, help="how many of the furthest lines to show")
    args = parser.parse_args()

    job = JOBS_DIR / args.job
    if not job.is_dir():
        raise SystemExit(f"No job folder {job}")
    if args.make_reference:
        make_reference(job, args.start, args.end)
        return
    reference_path = job / REFERENCE_NAME
    if not reference_path.exists():
        raise SystemExit(f"No {reference_path}; make one with --make-reference first.")
    reference = load_segments(reference_path)
    candidates = args.candidate or [job / "translated_segments.json"]
    for path in candidates:
        result = score(reference, load_segments(path))
        print(f"\n=== {path.name}")
        if result is None:
            print("No line matches the reference (different video or time range?)")
            continue
        print(f"chrF {result['chrf']:.1f} on {result['lines']} lines"
              + (f" ({result['missing']} reference lines not found)" if result["missing"] else "")
              + f" | length {result['length']:.2f}x the reference")
        for value, wanted, got, item in result["worst"][:args.worst]:
            print(f"  {value:5.1f}  {item.get('source_text', '')}\n         want: {wanted}\n         got:  {got}")


if __name__ == "__main__":
    main()
