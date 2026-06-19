from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_REPO = "Systran/faster-distil-whisper-large-v3"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "whisper-distil-large-v3"

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{MODEL_REPO}' → {MODEL_DIR} ...")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(MODEL_DIR),
    )
    print(f"\nDone. Model files saved to:\n  {MODEL_DIR}")
    print("\nFiles downloaded:")
    for f in sorted(MODEL_DIR.iterdir()):
        size_mb = f.stat().st_size / 1_048_576
        print(f"  {f.name:<40} {size_mb:>8.1f} MB")


if __name__ == "__main__":
    main()