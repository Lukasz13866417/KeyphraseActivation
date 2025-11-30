import numpy as np
import librosa
import soundfile as sf
from typing import Tuple

def estimate_top_db(
    y: np.ndarray,
    sr: int,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    percentile: float = 0.10,   # bottom 10%
    margin_db: float = 6.0,     # how much above the floor we gate
    region: str = "global",     # "global" | "sides"
) -> float:
    # Compute RMS
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms = rms[np.isfinite(rms)]
    rms = rms[rms > 0]
    if rms.size == 0:
        return 60.0

    # Optionally bias the floor from edges (leading/trailing), which are more often silent
    if region == "sides":
        n = len(rms)
        k = max(1, int(0.2 * n))  # 20% head/tail
        candidate = np.concatenate([rms[:k], rms[-k:]])
    else:
        candidate = rms

    floor = np.percentile(candidate, percentile * 100.0)
    if not np.isfinite(floor) or floor <= 0:
        return 60.0

    # Convert floor to dB relative to peak (peak -> 0 dB)
    floor_db_rel = librosa.amplitude_to_db(np.array([floor]), ref=np.max(rms))[0]  # negative
    # librosa.split keeps frames with dB > -top_db
    # Choose top_db so that we keep everything above (floor + margin)
    top_db = max(6.0, min(80.0, float(-floor_db_rel - margin_db)))
    return top_db
    
def _nearest_zero_cross(y: np.ndarray, sidx: int, max_search: int) -> int:
    lo = max(1, sidx - max_search)
    hi = min(len(y) - 1, sidx + max_search)
    best = sidx
    best_dist = max_search + 1
    for i in range(lo, hi):
        if (y[i - 1] <= 0.0 <= y[i]) or (y[i - 1] >= 0.0 >= y[i]):
            d = abs(i - sidx)
            if d < best_dist:
                best = i
                best_dist = d
    return best


def crop_silence(
    wav_path: str,
    *,
    remove_left: bool = True,
    remove_right: bool = True,
    top_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    fade_ms: int = 10,
) -> Tuple[np.ndarray, int]:
    """
    Remove leading and/or trailing silence from a WAV file.
    Snap cut points to nearest zero crossing and add a short fade in/out.
    """
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    if isinstance(top_db, str) and top_db.lower() == "auto":
        top_db_val = estimate_top_db(
            y, sr,
            frame_length=frame_length,
            hop_length=hop_length,
            percentile=0.10,    #0.05–0.15
            margin_db=6.0,      # 3–10
            region="sides",     # other option - "global"
        )
    else:
        top_db_val = float(top_db)

    intervals = librosa.effects.split(
        y, top_db=top_db_val, frame_length=frame_length, hop_length=hop_length
    )
    if intervals.size == 0:
        # nothing considered non-silent, return original
        return y, sr

    start = 0
    end = len(y)
    if remove_left:
        start = int(intervals[0, 0])
    if remove_right:
        end = int(intervals[-1, 1])

    # snap to nearest zero crossing (+-8 ms)
    zmax = int(sr * 0.008)
    start = _nearest_zero_cross(y, start, zmax)
    end = _nearest_zero_cross(y, end, zmax)
    if end <= start:
        end = min(len(y), start + int(0.05 * sr))  # must be non-empty

    y_crop = y[start:end]

    # apply fade in/out
    f = max(1, int(sr * (fade_ms / 1000.0)))
    if len(y_crop) > 2 * f:
        fi = np.linspace(0.0, 1.0, f, dtype=np.float32)
        fo = 1.0 - fi
        y_crop[:f] *= fi
        y_crop[-f:] *= fo

    return y_crop, sr


if __name__ == "__main__":
    wav = input("Enter path to WAV file: ").strip()
    ans_l = (input("Trim start? [Y/n]: ").strip() or "y").lower()
    ans_r = (input("Trim end? [Y/n]: ").strip() or "y").lower()
    try:
        top_db = float(input("Silence threshold top_db (default 30): ").strip() or "30")
    except Exception:
        top_db = 30
    y_out, sr_out = crop_silence(
        wav,
        remove_left=ans_l != "n",
        remove_right=ans_r != "n",
        top_db=top_db,
    )
    base = wav.rsplit(".", 1)[0] if "." in wav else wav
    out_path = base + "_nosil.wav"
    sf.write(out_path, y_out, sr_out)
    print(f"Saved: {out_path} | sr={sr_out} | dur={len(y_out)/sr_out:.2f}s")


