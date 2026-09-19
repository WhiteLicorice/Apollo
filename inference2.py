"""Windows inference entry point for Apollo.

Source: https://github.com/starinspace/Apollo-Windows/blob/main/inference2.py
Adapts: https://github.com/JusperLee/Apollo (the Apollo audio restoration model)

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
Apollo is licensed under CC BY-SA 4.0; see the LICENSE file at the root of this
repository for the full text, or https://creativecommons.org/licenses/by-sa/4.0/
for the human-readable summary. This file is Adapted Material under that license
and is shared under the same CC BY-SA 4.0 terms.
"""

import os
import subprocess
import tempfile
import time
import torch
import torchaudio
import soundfile as sf
import argparse
import look2hear.models
import warnings
from rich import print
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from metadata_utils import copy_metadata

SEGMENT_SECONDS = 10  # length of each segment in seconds
OVERLAP_SECONDS = 1   # overlap to ensure no gaps (kan justeras)

# Apollo was trained on 44.1 kHz music, so every input is resampled to that
# rate before inference regardless of its native rate -- feeding it anything
# else silently distorts the output rather than erroring.
TARGET_SAMPLE_RATE = 44100

# ffmpeg's libsoxr isn't compiled into the available build (verified: selecting
# it raises "Requested resampling engine is unavailable"), so this uses
# ffmpeg's native swr engine tuned to its highest-quality settings instead:
# max filter size, max phase shift, and a near-Nyquist cutoff.
RESAMPLE_FILTER = "aresample=resampler=swr:filter_size=256:phase_shift=24:cutoff=0.98"

# Suppress Torchaudio backend warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchaudio._backend.utils"
)

# Resample to 44.1 kHz via ffmpeg, then load
def load_audio(file_path, device="cuda"):
    # Written to the OS temp dir, never next to the source file: a leftover
    # from an interrupted run must never be mistaken for a new input by a
    # later run's directory scan.
    fd, temp_file = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", file_path,
             "-af", RESAMPLE_FILTER,
             "-ar", str(TARGET_SAMPLE_RATE),
             "-c:a", "pcm_f32le",
             temp_file],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg resample failed for {file_path}:\n{result.stderr}")

        audio, samplerate = torchaudio.load(temp_file)
    except Exception:
        os.remove(temp_file)
        raise

    audio = audio.unsqueeze(0)  # [1, 1, samples]
    audio = audio.to(device)

    return audio, samplerate, temp_file  # return temp file path for cleanup

def save_audio(file_path, audio, samplerate=44100):
    audio = audio.squeeze(0).cpu()  # [channels, samples]
    # 32-bit float WAV: preserves the model's native float32 output exactly,
    # with no lossy encoding, no bit-depth quantization, and no further
    # resampling. torchaudio.save() can't be used for this -- verified that as
    # of the TorchCodec-backed save(), its `encoding`/`bits_per_sample` args
    # are silently ignored and it always writes 16-bit PCM regardless of what
    # is requested -- so this writes directly via soundfile instead.
    sf.write(file_path, audio.numpy().T, samplerate, subtype="FLOAT")

def process_segments(model, audio, samplerate, track_name, overlap=OVERLAP_SECONDS):
    segment_length = SEGMENT_SECONDS * samplerate
    overlap_length = int(overlap * samplerate)
    hop_length = segment_length - overlap_length
    total_samples = audio.shape[-1]
    output_chunks = []
    cursor = 0
    total_chunks = -(-total_samples // hop_length)  # ceiling division

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} chunks"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(track_name, total=total_chunks)
        with torch.no_grad():
            while cursor < total_samples:
                end = min(cursor + segment_length, total_samples)
                segment = audio[:, :, cursor:end]

                # Pad sista segmentet om det är för kort
                if end - cursor < segment_length:
                    pad_size = segment_length - (end - cursor)
                    segment = torch.nn.functional.pad(segment, (0, pad_size))

                out = model(segment)

                # Trimma bort paddingen på sista segmentet
                if end - cursor < segment_length:
                    out = out[:, :, :end - cursor]

                # Hantera överlappning
                if cursor == 0:
                    # Första segmentet: ta allt utom sista overlappen
                    output_chunks.append(out[:, :, :-overlap_length])
                elif end >= total_samples:
                    # Sista segmentet: ta allt
                    output_chunks.append(out)
                else:
                    # Mellansegment: ta allt utom sista overlappen
                    output_chunks.append(out[:, :, :-overlap_length])

                cursor += hop_length
                progress.advance(task)

    return torch.cat(output_chunks, dim=-1)

def main(input_file, output_file):
    start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained model
    model_file = r"Apollo\pytorch_model.bin"
    model = look2hear.models.BaseModel.from_pretrain(
        model_file,
        sr=44100,
        win=20,
        feature_dim=256,
        layer=6
    ).to(device)
    model.eval()

    temp_file = None
    try:
        # Load audio
        audio, samplerate, temp_file = load_audio(input_file, device=device)
        # Process audio with overlap-and-crop
        track_name = os.path.basename(input_file)
        output_audio = process_segments(model, audio, samplerate, track_name)
        # Save output
        save_audio(output_file, output_audio, samplerate)
        # Persist tags (title/artist/album/...) so restoring audio doesn't
        # erase what a music library sorts by; a failure here shouldn't
        # discard an already-successful restoration.
        try:
            copy_metadata(input_file, output_file)
        except Exception as e:
            print(f"[yellow]![/yellow] Could not copy metadata for {os.path.basename(output_file)}: {e}")
    finally:
        # Always delete the temporary WAV file, even if inference raised,
        # so a crash never leaves it behind for a later run to pick up.
        if temp_file is not None and os.path.exists(temp_file):
            os.remove(temp_file)

    elapsed = time.time() - start_time
    print(f"[green]✓[/green] {os.path.basename(output_file)} restored in {elapsed:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Inference Script")
    parser.add_argument("--in_wav", type=str, required=True, help="Path to input wav/mp3 file")
    parser.add_argument("--out_wav", type=str, required=True, help="Path to output wav file (32-bit float)")
    args = parser.parse_args()

    main(args.in_wav, args.out_wav)
