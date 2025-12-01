import numpy as np
import librosa
from typing import Tuple


def _features(y: np.ndarray, sr: int, hop: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """ Compute useful values for audio signal:
        onset strength, energy, and zero-crossing rate features.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=hop)[0]
    frames = len(energy)
    return onset_env, energy, zcr, frames


def _peak_pick(envelope: np.ndarray) -> np.ndarray:
    """ Find local maxima in an audio envelope 
    (envelope = discrete sequence that says how strong the audio is at each point in time).
    Local maxima are likely to be starts of syllables in synthetic speech.
    """
    if envelope.size == 0:
        return np.array([], dtype=int)
    return librosa.util.peak_pick(
        x=envelope,
        pre_max=3, post_max=3,
        pre_avg=3, post_avg=3,
        delta=0.3 * float(envelope.max()) if envelope.max() > 0 else 0.0,
        wait=5,
    )


def _nearest_zero_cross(y: np.ndarray, sidx: int, max_search: int) -> int:
    """ Simple binary search to find first zero crossing in either direction in a continuous function."""
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


def _natural_cut_sample(
    y: np.ndarray,
    sr: int,
    target_sample: int,
    energy: np.ndarray,
    hop: int,
    energy_search_win_frames: int = 8,
    zero_search_ms: int = 8,
    target_jitter_frames: int = 12,
) -> int:
    """ Cut an audio sample at a point around the provided timestamp, but with enhancements:
    - jitter the timestamp a bit to allow for some randomness (useful in training)
    - pick a nearby point which is very quiet (energy-wise)
    - snap to zero crossing to decrease likelihood of cutting into a word AND to make the sample start from 0 pressure
    """
    # Jitter the target a bit (+- target_jitter_frames frames) to avoid locking onto the same valley every time
    target_frame = int(np.clip(round(target_sample / hop), 0, len(energy) - 1))
    jf = max(0, int(target_jitter_frames))
    if jf > 0:
        # Directional bias handled by caller via target_sample; keep symmetric here
        target_frame = int(np.clip(target_frame + np.random.randint(-jf, jf + 1), 0, len(energy) - 1))
    # Sample one of the low-energy frames nearby (window size with slight randomness)
    half_win = energy_search_win_frames + np.random.randint(0, max(1, energy_search_win_frames // 2) + 1)
    a = max(0, target_frame - half_win)
    b = min(len(energy), target_frame + half_win + 1)
    if b <= a + 1:
        e_idx = target_frame
    else:
        window = energy[a:b]
        # Find local minima indices in the window
        local_min = []
        for k in range(1, len(window) - 1):
            if window[k] <= window[k - 1] and window[k] <= window[k + 1]:
                local_min.append(a + k)
        if not local_min:
            # fallback: global min in window
            e_idx = a + int(np.argmin(window))
        else:
            # Weight minima by inverse energy to prefer deeper valleys but allow variety
            vals = np.array([energy[i] for i in local_min], dtype=np.float32)
            w = 1.0 / (vals + 1e-8)
            w = w / w.sum()
            pick = int(np.random.choice(len(local_min), p=w))
            e_idx = int(local_min[pick])
    s_idx = int(np.clip(e_idx * hop, 0, len(y) - 1))
    # Use deterministic zero-cross snapping (direction supplied by caller)
    zms = max(1.0, float(zero_search_ms))
    # The caller chooses direction by the side (left/right) they are cropping
    s_idx = _nearest_zero_cross(y, s_idx, max_search=int(sr * zms / 1000))
    return s_idx

def crop_audio(
    wav_path: str,
    spoken_text: str,
    *,
    crop_left: bool = True,
    crop_right: bool = True,
    min_crop_ms: int = 50,
    max_crop_ms: int = 250,
    fade_ms: int = 10,
    frame_jitter: int = 12,
    max_inword_ms: int = 80,
) -> Tuple[np.ndarray, int]:
    """
    Smartly crop a WAV to start/end mid-word while keeping the key phrase intact.
    !!!We assume the first and last words of the text dont belong to the keyphrase.!!!
    This will be used in positive tests that contain keyphrase plus some words before and after it.
    - spoken_text must have at least 3 words 
    - If crop_left: remove a random prefix from the first word (do not extend into the second word).
    - If crop_right: remove a random suffix from the last word (do not extend into the last word of keyphrase).
    We use onset/energy and snap to low-energy + zero crossings to find the best cut points.
    (zero crossings are used to make the sample start from 0 pressure, so no "clicks" are heard)
    We assume that words correspond to peaks in the onset envelope (rapid changes in pressure).
    """
    words = [w for w in spoken_text.strip().split() if w]
    if len(words) < 3:
        raise ValueError("spoken_text must contain at least 3 words.")

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    n = len(y)
    hop = 512
    onset_env, energy, frames = _features(y, sr, hop=hop)
    peaks = _peak_pick(onset_env)

    # Compute conservative limits so we don't cut into neighboring words too far
    # Left: limit to before the second onset peak or first 25% of frames
    if peaks.size >= 2:
        left_limit_frame = max(1, min(int(0.25 * frames), int(peaks[1])))
    elif peaks.size == 1:
        left_limit_frame = max(1, min(int(0.25 * frames), int(peaks[0] + 5)))
    else:
        left_limit_frame = max(1, int(0.2 * frames))

    # Right: limit from end to after second-to-last peak or last 25% window
    if peaks.size >= 2:
        right_limit_from_end_frame = max(1, min(int(0.25 * frames), int(frames - 1 - peaks[-2])))
    elif peaks.size == 1:
        right_limit_from_end_frame = max(1, min(int(0.25 * frames), int(frames - 1 - (peaks[0] + 5))))
    else:
        right_limit_from_end_frame = max(1, int(0.2 * frames))

    # Convert limits and min/max to samples
    left_limit_samples = left_limit_frame * hop
    right_limit_from_end_samples = right_limit_from_end_frame * hop
    min_crop_samples = int(sr * (min_crop_ms / 1000.0))
    max_crop_samples = int(sr * (max_crop_ms / 1000.0))

    # Random crop lengths within safe bounds
    rng_left_max = max(min(left_limit_samples, max_crop_samples), 0)
    rng_right_max = max(min(right_limit_from_end_samples, max_crop_samples), 0)

    # Safety caps: do not cut deeper than max_inword_ms into the first/last word (approx by onset)
    inword_cap_samples = int(sr * (max_inword_ms / 1000.0))
    first_onset_sample = int(peaks[0] * hop) if peaks.size >= 1 else int(0.15 * n)
    last_onset_sample = int(peaks[-1] * hop) if peaks.size >= 1 else int(0.85 * n)

    # Default no-crop positions
    start_sample = 0
    end_sample = n

    # Left crop
    if crop_left and rng_left_max > min_crop_samples:
        left_cap = min(rng_left_max, first_onset_sample + inword_cap_samples)
        raw_cut = np.random.randint(min_crop_samples, max(min_crop_samples + 1, left_cap + 1))
        # We subtract extra random frames to make better randomization
        # (left crop is more likely to be at the start of the word than at the end)
        raw_cut_frames = raw_cut // hop
        bias_frames = np.random.randint(0, frame_jitter + 1)
        raw_cut_biased = max(0, (raw_cut_frames - bias_frames) * hop)
        start_sample = _natural_cut_sample(
            y, sr, raw_cut_biased, energy=energy, hop=hop,
            energy_search_win_frames=8, zero_search_ms=8, target_jitter_frames=frame_jitter
        )

    # Right crop
    if crop_right and rng_right_max > min_crop_samples:
        max_remove_allowed = max(0, n - (last_onset_sample + inword_cap_samples))
        right_cap = min(rng_right_max, max_remove_allowed)
        raw_cut_from_end = np.random.randint(min_crop_samples, max(min_crop_samples + 1, right_cap + 1))
        # We add extra random frames to make better randomization
        # (right crop is more likely to be at the end of the word than at the start)
        raw_end_frames = raw_cut_from_end // hop
        bias_frames = np.random.randint(0, frame_jitter + 1)
        raw_end_biased = (raw_end_frames + bias_frames) * hop
        target_end = max(0, n - raw_end_biased)
        end_sample = _natural_cut_sample(
            y, sr, target_end, energy=energy, hop=hop,
            energy_search_win_frames=8, zero_search_ms=8, target_jitter_frames=frame_jitter
        )
        # ensure end after start
        if end_sample <= start_sample:
            end_sample = min(n, start_sample + int(0.2 * sr))

    y_crop = y[start_sample:end_sample]

    # Apply short fade-in/out to hide discontinuities
    f = max(1, int(sr * (fade_ms / 1000.0)))
    if len(y_crop) > 2 * f:
        fi = np.linspace(0.0, 1.0, f, dtype=np.float32)
        fo = 1.0 - fi
        y_crop[:f] *= fi
        y_crop[-f:] *= fo

    return y_crop, sr


if __name__ == "__main__":
    import soundfile as sf
    wav = input("Enter path to WAV file: ").strip()
    text = input("Enter transcript (>= 3 words): ").strip()
    try:
        fj = int(input("Frame jitter (±frames, default 12): ").strip() or "12")
    except Exception:
        fj = 12
    try:
        miw = int(input("Max in-word crop (ms, default 80): ").strip() or "80")
    except Exception:
        miw = 80
    y_out, sr_out = crop_audio(
        wav, text,
        crop_left=True, crop_right=True,
        frame_jitter=fj,
        max_inword_ms=miw,
    )
    base = wav.rsplit(".", 1)[0] if "." in wav else wav
    out_path = base + "_cropped.wav"
    sf.write(out_path, y_out, sr_out)
    print(f"Saved: {out_path} | sr={sr_out} | dur={len(y_out)/sr_out:.2f}s")
