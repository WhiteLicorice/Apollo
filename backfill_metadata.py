"""One-time backfill: copies metadata from input\\ files to their already-
converted output\\ counterparts, for files processed before copy_metadata()
existed in inference2.py's pipeline. Safe to re-run (copy_metadata()
overwrites the same ID3 frames each time) and safe to run while a batch
conversion is still in progress: it only ever touches an output file that
already exists, and inference2.py writes each output file in one shot, so an
existing file on disk is always a complete one.
"""

from pathlib import Path

from metadata_utils import copy_metadata

EXTENSIONS = {".mp3", ".m4a", ".wav", ".aiff", ".aif", ".flac", ".opus"}
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")


def main():
    files = sorted(p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in EXTENSIONS)
    tagged = 0
    pending = 0
    failed = 0

    for in_path in files:
        rel = in_path.relative_to(INPUT_DIR)
        out_path = (OUTPUT_DIR / rel).with_suffix(".wav")
        if not out_path.exists():
            pending += 1
            continue
        try:
            copy_metadata(in_path, out_path)
            tagged += 1
            print(f"Tagged: {out_path}")
        except Exception as e:
            failed += 1
            print(f"! Failed to tag {out_path}: {e}")

    print(f"Done: {tagged} tagged, {pending} not yet converted, {failed} failed")


if __name__ == "__main__":
    main()
