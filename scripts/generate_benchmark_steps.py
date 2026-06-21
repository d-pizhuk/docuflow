# scripts/generate_benchmark_steps.py

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.step_structurer import StepStructurer
from ai.api_gateway import ApiGateway, ApiGatewayError

SAMPLES_DIR = PROJECT_ROOT / "tests" / "benchmark" / "samples"


def process_sample(sample_dir: Path, structurer: StepStructurer):
    transcript_path = sample_dir / "transcript_whisper.txt"
    output_path = sample_dir / "steps_generated.json"

    if not transcript_path.exists():
        print(f"Skipping {sample_dir.name}: Missing transcript_whisper.txt")
        return

    print(f"Processing {sample_dir.name}...")
    transcript = transcript_path.read_text(encoding="utf-8").strip()

    if not transcript:
        print(f"  -> Skipping {sample_dir.name}: Transcript is empty.")
        return

    try:
        doc = structurer.structure(
            annotated_transcript=transcript,
            valid_screenshots=None,
            session_dir=sample_dir,
        )
        output_data = doc.to_json()
        output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
        print(f"  -> Saved {output_path.name} ({len(doc.steps)} steps generated)")

    except ApiGatewayError as e:
        print(f"  !! FAILED for {sample_dir.name}: {e}")
    except Exception as e:
        print(f"  !! Unexpected error for {sample_dir.name}: {e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate steps_generated.json for benchmark samples using the LLM Step Structurer."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--sample",
        metavar="SAMPLE_ID",
        help="Process only this sample (e.g. --sample sample_001). "
             "Mutually exclusive with --all.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Process all samples. Mutually exclusive with --sample.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing steps_generated.json files. "
             "Without this flag, samples that already have output are skipped.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    if not SAMPLES_DIR.exists():
        print(f"Samples directory not found: {SAMPLES_DIR}")
        sys.exit(1)

    all_samples = sorted(
        d for d in SAMPLES_DIR.iterdir()
        if d.is_dir() and d.name.startswith("sample_")
    )

    if not all_samples:
        print(f"No sample_XXX directories found in {SAMPLES_DIR}")
        sys.exit(1)

    # --- Sample selection ---
    if args.sample:
        matched = [d for d in all_samples if d.name == args.sample]
        if not matched:
            available = ", ".join(d.name for d in all_samples)
            print(f"Sample '{args.sample}' not found. Available: {available}")
            sys.exit(1)
        samples = matched
    elif args.all:
        samples = all_samples
    else:
        # Default: process all, but remind the user about the flags
        print(
            "Tip: use --sample <id> to process a single sample, "
            "or --all to suppress this message.\n"
        )
        samples = all_samples

    # --- Skip already-generated samples unless --force ---
    if not args.force:
        pending = [d for d in samples if not (d / "steps_generated.json").exists()]
        skipped = len(samples) - len(pending)
        if skipped:
            print(f"Skipping {skipped} sample(s) that already have steps_generated.json "
                  f"(use --force to regenerate).")
        samples = pending

    if not samples:
        print("Nothing to do.")
        sys.exit(0)

    print(f"Initializing LLM Step Structurer...")
    structurer = StepStructurer(gateway=ApiGateway())
    print(f"Processing {len(samples)} sample(s)...\n")

    for sample_dir in samples:
        process_sample(sample_dir, structurer)

    print("\nDone! Run the benchmark test to evaluate.")


if __name__ == "__main__":
    main()