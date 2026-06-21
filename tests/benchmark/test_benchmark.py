"""
Benchmark evaluation suite for DocuFlow.

Metrics
-------
- WER   : Word Error Rate on Whisper transcription vs ground-truth transcript
- ROUGE-L : F1 on generated steps (flattened) vs ground-truth steps (flattened)

Sample layout expected under tests/benchmark/samples/<sample_id>/
    transcript_ground_truth.txt   – reference spoken text
    transcript_whisper.txt        – Whisper output to evaluate
    steps_ground_truth.json       – {"steps": [{"step":int,"title":str,"instruction":str}, ...]}
    steps_generated.json          – {"title":str, "steps": [{"title":str,"instruction":str,...}, ...]}
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from rouge_score import rouge_scorer
import jiwer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "samples"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkSample:
    sample_id: str
    transcript_ground_truth: str
    transcript_whisper: str
    steps_ground_truth: list[dict]
    steps_generated: list[dict]

    @classmethod
    def load(cls, sample_dir: Path) -> "BenchmarkSample":
        def read(name: str) -> str:
            return (sample_dir / name).read_text(encoding="utf-8").strip()

        def read_json(name: str) -> dict:
            return json.loads((sample_dir / name).read_text(encoding="utf-8"))

        return cls(
            sample_id=sample_dir.name,
            transcript_ground_truth=read("transcript_ground_truth.txt"),
            transcript_whisper=read("transcript_whisper.txt"),
            steps_ground_truth=read_json("steps_ground_truth.json")["steps"],
            steps_generated=read_json("steps_generated.json")["steps"],
        )

    def _flatten_gt_steps(self) -> str:
        return " ".join(f"{s.get('title', '')} {s['instruction']}" for s in self.steps_ground_truth)

    def _flatten_gen_steps(self) -> str:
        return " ".join(f"{s.get('title', '')} {s['instruction']}" for s in self.steps_generated)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

_WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    if not reference or not hypothesis:
        return 0.0
    return _SCORER.score(reference, hypothesis)["rougeL"].fmeasure


def compute_wer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer.wer(
        reference, hypothesis,
        reference_transform=_WER_TRANSFORM,
        hypothesis_transform=_WER_TRANSFORM,
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_samples() -> list[BenchmarkSample]:
    if not SAMPLES_DIR.exists():
        return []
    dirs = sorted(p for p in SAMPLES_DIR.iterdir() if p.is_dir() and p.name.startswith("sample_"))
    return [BenchmarkSample.load(d) for d in dirs]


# ---------------------------------------------------------------------------
# Rating Buckets
# ---------------------------------------------------------------------------
def get_acc_rating(acc: float) -> str:
    if acc >= 0.95: return "PERFECT"
    if acc >= 0.85: return "GOOD"
    if acc >= 0.75: return "MODERATE"
    return "BAD"


def get_rouge_rating(rouge: float) -> str:
    if rouge >= 0.70: return "PERFECT"
    if rouge >= 0.50: return "GOOD"
    if rouge >= 0.30: return "MODERATE"
    return "BAD"


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    samples = discover_samples()
    if not samples:
        print(f"No samples found under {SAMPLES_DIR}")
        sys.exit(0)

    results = []

    for s in samples:
        wer = compute_wer(s.transcript_ground_truth, s.transcript_whisper)
        acc = 1.0 - wer
        rouge_l = compute_rouge_l(s._flatten_gt_steps(), s._flatten_gen_steps())

        results.append({
            "sample_id": s.sample_id,
            "acc": acc,
            "wer": wer,
            "rouge_l": rouge_l,
            "acc_rating": get_acc_rating(acc),
            "rouge_rating": get_rouge_rating(rouge_l)
        })

    # Print Table 1: Transcription Accuracy
    print("\n" + "=" * 65)
    print(" TABLE 1: Transcription Accuracy (WER)")
    print("=" * 65)
    print(f"{'Sample':<20} {'Accuracy':>10} {'WER':>10} {'Rating':>15}")
    print("-" * 65)
    for r in results:
        print(f"{r['sample_id']:<20} {r['acc']:>10.3f} {r['wer']:>10.3f} {r['acc_rating']:>15}")
    print("-" * 65)

    # Calculate Averages
    avg_acc = sum(r['acc'] for r in results) / len(results)
    avg_wer = sum(r['wer'] for r in results) / len(results)
    print(f"{'AVERAGE':<20} {avg_acc:>10.3f} {avg_wer:>10.3f}")
    print("=" * 65)

    # Print Table 2: Documentation Quality (ROUGE-L)
    print("\n" + "=" * 65)
    print(" TABLE 2: Documentation Quality (ROUGE-L)")
    print("=" * 65)
    print(f"{'Sample':<20} {'ROUGE-L':>10} {'Rating':>15}")
    print("-" * 65)
    for r in results:
        print(f"{r['sample_id']:<20} {r['rouge_l']:>10.3f} {r['rouge_rating']:>15}")
    print("-" * 65)

    avg_rouge = sum(r['rouge_l'] for r in results) / len(results)
    print(f"{'AVERAGE':<20} {avg_rouge:>10.3f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
