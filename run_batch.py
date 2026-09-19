"""Batch driver for inference2.py.

Recursively walks input\\, mirrors its subfolder structure into output\\,
skips files already processed, and runs inference2.py once per file via
subprocess -- unchanged from the prior run.bat-only design, just with the
file-discovery and path bookkeeping done in Python instead of batch.

Batch treats many characters common in real track names -- parentheses,
"!", "&", "%", "^", quotes -- as syntax, not text. Two of this library's own
real filenames already broke that assumption: "Florida!!!.opus" had its "!"s
silently stripped by delayed expansion, and "Habits (Stay High).opus" made
the whole script crash with a parse error. Python has no equivalent failure
mode: a filename is always just a string.
"""

import subprocess
import sys
import time
from pathlib import Path

EXTENSIONS = {".mp3", ".m4a", ".wav", ".aiff", ".aif", ".flac", ".opus"}
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    files = sorted(p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in EXTENSIONS)
    total = len(files)

    batch_start = time.time()
    for i, in_path in enumerate(files, start=1):
        rel = in_path.relative_to(INPUT_DIR)
        out_path = (OUTPUT_DIR / rel).with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            print(f"[{i}/{total}] Skipping {in_path}, {out_path} already exists")
            continue

        print(f"[{i}/{total}] Processing: {in_path}")
        result = subprocess.run(
            [sys.executable, "inference2.py",
             "--in_wav", str(in_path), "--out_wav", str(out_path)]
        )
        if result.returncode != 0:
            print(f"  ! inference2.py failed for {in_path} (exit code {result.returncode}), continuing")

    elapsed = time.time() - batch_start
    print(f"All files processed in {elapsed:.0f}s!")


if __name__ == "__main__":
    main()
