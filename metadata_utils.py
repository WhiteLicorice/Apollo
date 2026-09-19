"""Copies common tags (title, artist, album, ...) from a source audio file
into a restored WAV's ID3 frames, so restoring audio doesn't erase what a
music library sorts by. Kept separate from inference2.py so a metadata-only
pass (see backfill_metadata.py) doesn't have to pay for torch/CUDA imports.
"""

import mutagen
from mutagen.id3 import TIT2, TPE1, TPE2, TALB, TDRC, TCON, TRCK

# Vorbis-comment-style key (what mutagen's easy mode gives for every
# supported input format) -> the ID3 frame class WAV output tags need.
TAG_FRAMES = {
    "title": TIT2,
    "artist": TPE1,
    "albumartist": TPE2,
    "album": TALB,
    "date": TDRC,
    "genre": TCON,
    "tracknumber": TRCK,
}


def copy_metadata(src_path, dst_path):
    try:
        src = mutagen.File(str(src_path), easy=True)
    except Exception:
        return
    if src is None or not src.tags:
        return

    dst = mutagen.File(str(dst_path))
    if dst.tags is None:
        dst.add_tags()

    wrote_any = False
    for key, frame_cls in TAG_FRAMES.items():
        values = src.tags.get(key)
        if values:
            dst.tags.add(frame_cls(encoding=3, text=values))
            wrote_any = True

    if wrote_any:
        dst.save()
