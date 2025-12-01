import argparse
import json
import os
from pathlib import Path
from random import random
from typing import List, Optional

import soundfile as sf
import numpy as np

from crop import crop_audio
from crop_silence import crop_silence


def _ensure_three_words(text: str) -> None:
    words = [w for w in text.strip().split() if w]
    if len(words) < 3:
        raise ValueError("normalized_text must contain at least 3 words.")


def _next_out_path(base_dir: Path, stem: str, ext: str = ".wav", start_idx: int = 0) -> Path:
    """
        Generate the path to the output file
        (we need to find a path that doesn't already exist)
    """
    i = start_idx
    while True:
        p = base_dir / f"{stem}_aug_{i:04d}{ext}"
        if not p.exists():
            return p
        i += 1


def augment_one(
    wav_path: str,
    normalized_text: str,
    *,
    keyphrase: Optional[str] = None,
    p_trim_start: float = 0.5,
    p_trim_end: float = 0.5,
    p_crop_left: float = 0.5,
    p_crop_right: float = 0.5,
    top_db: str = "auto",
    fade_ms: int = 10,
    frame_jitter: int = 12,
    max_inword_ms: int = 80,
) -> np.ndarray:
    """
    Randomly crop the audio to start/end mid-word while keeping the key phrase intact.
    In the future we can use the transcript to crop the audio more precisely.
    (more randomness but still keep the key phrase intact)
    """
    _ensure_three_words(normalized_text)

    # Sometimes we trim silence first, sometimes we don't
    do_trim_start = (random() < p_trim_start)
    do_trim_end = (random() < p_trim_end)

    if do_trim_start or do_trim_end:
        y_sil, sr = crop_silence(
            wav_path,
            remove_left=do_trim_start,
            remove_right=do_trim_end,
            top_db=top_db if top_db != "auto" else "auto",
            fade_ms=fade_ms,
        )
        # Save to an in-memory temporary holder
        import tempfile
        with tempfile.NamedTemporaryFile(prefix="aug_trim_", suffix=".wav", delete=False) as tf:
            tmp_path = tf.name
        try:
            sf.write(tmp_path, y_sil, sr)
            y_crop, sr2 = crop_audio(
                tmp_path,
                normalized_text,
                crop_left=(random() < p_crop_left),
                crop_right=(random() < p_crop_right),
                fade_ms=fade_ms,
                frame_jitter=frame_jitter,
                max_inword_ms=max_inword_ms,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return y_crop.astype(np.float32)
    else:
        # Directly crop words from original
        y_crop, sr = crop_audio(
            wav_path,
            normalized_text,
            crop_left=(random() < p_crop_left),
            crop_right=(random() < p_crop_right),
            fade_ms=fade_ms,
            frame_jitter=frame_jitter,
            max_inword_ms=max_inword_ms,
        )
        return y_crop.astype(np.float32)


def augment_batch(
    wav_path: str,
    normalized_text: str,
    *,
    keyphrase: Optional[str],
    num_clips: int,
    output_dir: Optional[str],
    p_trim_start: float,
    p_trim_end: float,
    p_crop_left: float,
    p_crop_right: float,
    top_db: str = "auto",
    fade_ms: int = 10,
    frame_jitter: int = 12,
    max_inword_ms: int = 80,
) -> List[str]:
    """Generate N augmented clips and save them next to the input (or to output_dir)."""
    _ensure_three_words(normalized_text)
    in_path = Path(wav_path).resolve()
    out_dir = Path(output_dir).resolve() if output_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = in_path.stem

    saved: List[str] = []
    # Read original sr to ensure consistent sample rate
    _, sr = sf.read(str(in_path), dtype="float32", always_2d=False)
    for i in range(num_clips):
        y = augment_one(
            wav_path=str(in_path),
            normalized_text=normalized_text,
            keyphrase=keyphrase,
            p_trim_start=p_trim_start,
            p_trim_end=p_trim_end,
            p_crop_left=p_crop_left,
            p_crop_right=p_crop_right,
            top_db=top_db,
            fade_ms=fade_ms,
            frame_jitter=frame_jitter,
            max_inword_ms=max_inword_ms,
        )
        out_path = _next_out_path(out_dir, stem, ".wav")
        sf.write(str(out_path), y, sr)
        saved.append(str(out_path))
    return saved


# Simple test made by AI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio augmentation pipeline for a single wav.")
    parser.add_argument("--guidance", action="store_true", help="Interactive prompts for inputs")
    parser.add_argument("--wav", required=False, help="Path to source WAV")
    parser.add_argument("--text", required=False, help="Normalized transcript (>= 3 words)")
    parser.add_argument("--keyphrase", default=None, help="Key phrase (optional, currently unused)")
    parser.add_argument("--num", type=int, default=5, help="Number of augmented clips to generate")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: alongside input)")
    parser.add_argument("--p-trim-start", type=float, default=0.5, help="Prob. to remove leading silence")
    parser.add_argument("--p-trim-end", type=float, default=0.5, help="Prob. to remove trailing silence")
    parser.add_argument("--p-crop-left", type=float, default=0.5, help="Prob. to crop first word region")
    parser.add_argument("--p-crop-right", type=float, default=0.5, help="Prob. to crop last word region")
    parser.add_argument("--top-db", default="auto", help='Silence threshold (dB) or "auto"')
    parser.add_argument("--fade-ms", type=int, default=10, help="Fade at boundaries (ms)")
    parser.add_argument("--frame-jitter", type=int, default=12, help="Target jitter in frames for word-boundary cuts")
    parser.add_argument("--max-inword-ms", type=int, default=80, help="Max depth (ms) to cut into first/last word")
    args = parser.parse_args()

    if args.guidance:
        wav = args.wav or input("Enter path to WAV file: ").strip()
        text = args.text or input("Enter normalized transcript (>= 3 words): ").strip()
        keyphrase = args.keyphrase or (input("Enter keyphrase (optional): ").strip() or None)
        try:
            num = int(input(f"Number of augmented clips [{args.num}]: ").strip() or str(args.num))
        except Exception:
            num = args.num
        out_dir = args.out_dir or (input("Output directory (blank=alongside input): ").strip() or None)
        try:
            p_ts = float(input(f"Prob trim start [0-1] [{args.p_trim_start}]: ").strip() or str(args.p_trim_start))
        except Exception:
            p_ts = args.p_trim_start
        try:
            p_te = float(input(f"Prob trim end [0-1] [{args.p_trim_end}]: ").strip() or str(args.p_trim_end))
        except Exception:
            p_te = args.p_trim_end
        try:
            p_cl = float(input(f"Prob crop left [0-1] [{args.p_crop_left}]: ").strip() or str(args.p_crop_left))
        except Exception:
            p_cl = args.p_crop_left
        try:
            p_cr = float(input(f"Prob crop right [0-1] [{args.p_crop_right}]: ").strip() or str(args.p_crop_right))
        except Exception:
            p_cr = args.p_crop_right
        top_db = (input(f'Silence threshold top_db or "auto" [{args.top_db}]: ').strip() or str(args.top_db))
        try:
            fade_ms = int(input(f"Fade (ms) [{args.fade_ms}]: ").strip() or str(args.fade_ms))
        except Exception:
            fade_ms = args.fade_ms
        try:
            frame_jitter = int(input(f"Frame jitter (±frames) [{args.frame_jitter}]: ").strip() or str(args.frame_jitter))
        except Exception:
            frame_jitter = args.frame_jitter
        try:
            max_inword_ms = int(input(f"Max in-word crop (ms) [{args.max_inword_ms}]: ").strip() or str(args.max_inword_ms))
        except Exception:
            max_inword_ms = args.max_inword_ms
    else:
        if not args.wav or not args.text:
            raise SystemExit("Error: --wav and --text are required when not using --guidance.")
        wav = args.wav
        text = args.text
        keyphrase = args.keyphrase
        num = args.num
        out_dir = args.out_dir
        p_ts = args.p_trim_start
        p_te = args.p_trim_end
        p_cl = args.p_crop_left
        p_cr = args.p_crop_right
        top_db = args.top_db
        fade_ms = args.fade_ms
        frame_jitter = args.frame_jitter
        max_inword_ms = args.max_inword_ms

    results = augment_batch(
        wav_path=wav,
        normalized_text=text,
        keyphrase=keyphrase,
        num_clips=num,
        output_dir=out_dir,
        p_trim_start=p_ts,
        p_trim_end=p_te,
        p_crop_left=p_cl,
        p_crop_right=p_cr,
        top_db=top_db,
        fade_ms=fade_ms,
        frame_jitter=frame_jitter,
        max_inword_ms=max_inword_ms,
    )
    print(json.dumps(results, indent=2))


