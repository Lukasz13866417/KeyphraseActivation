from __future__ import annotations

import argparse
import numpy as np
import soundfile as sf
import librosa
import pyworld as pw


def add_question_intonation(
    input_path: str,
    output_path: str,
    rise_fraction: float = 0.35,
    rise_semitones: float = 4.0,
    trim_silence: bool = True,
) -> None:
    """
    Convert a short spoken word/phrase into something that sounds like a question
    by adding a rising pitch contour at the end.
    """
    # 1) Load audio (mono, keep original sr)
    y, sr = librosa.load(input_path, sr=None, mono=True)
    if trim_silence:
        # Trim leading/trailing silence
        y, _ = librosa.effects.trim(y, top_db=30)
        if y.size == 0:
            raise RuntimeError("Audio became empty after trimming silence")

    # WORLD prefers float64
    y = y.astype(np.float64, copy=False)

    # 2) WORLD analysis
    f0, sp, ap = pw.wav2world(y, sr)  # f0: (T,), sp/ap: (T, freq_bins)
    n_frames = len(f0)
    if n_frames < 10:
        raise RuntimeError("Too few WORLD frames; input likely too short or silent.")

    # 3) Rising region start
    start_idx = int((1.0 - rise_fraction) * n_frames)
    start_idx = max(0, min(start_idx, n_frames - 1))

    # 4) Pitch factor ramp
    max_factor = 2.0 ** (rise_semitones / 12.0)
    n_rise_frames = n_frames - start_idx
    if n_rise_frames <= 0:
        n_rise_frames = 1
        start_idx = n_frames - 1
    factors = np.linspace(1.0, max_factor, n_rise_frames)

    new_f0 = f0.copy()
    region = new_f0[start_idx:]
    voiced_mask = region > 0
    region[voiced_mask] = region[voiced_mask] * factors[voiced_mask]
    new_f0[start_idx:] = region

    # 5) Resynthesize with modified F0
    y_out = pw.synthesize(new_f0, sp, ap, sr)  # float64

    # 6) Normalize
    peak = np.max(np.abs(y_out))
    if peak > 0:
        y_out = (y_out / peak) * 0.99

    # 7) Write
    sf.write(output_path, y_out, sr)


def add_exclamation_emphasis(
    input_path: str,
    output_path: str,
    rise_fraction: float = 0.25,
    rise_semitones: float = 3.0,
    emphasis_gain: float = 1.15,
    trim_silence: bool = True,
) -> None:
    """
    Add a slight rising pitch near the end and a gentle emphasis to simulate an exclamation.
    """
    y, sr = librosa.load(input_path, sr=None, mono=True)
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=30)
        if y.size == 0:
            raise RuntimeError("Audio became empty after trimming silence")
    y = y.astype(np.float64, copy=False)

    f0, sp, ap = pw.wav2world(y, sr)
    n_frames = len(f0)
    if n_frames < 10:
        raise RuntimeError("Too few WORLD frames; input likely too short or silent.")

    start_idx = int((1.0 - rise_fraction) * n_frames)
    start_idx = max(0, min(start_idx, n_frames - 1))
    max_factor = 2.0 ** (rise_semitones / 12.0)
    n_rise_frames = max(1, n_frames - start_idx)
    factors = np.linspace(1.0, max_factor, n_rise_frames)

    new_f0 = f0.copy()
    region = new_f0[start_idx:]
    voiced_mask = region > 0
    region[voiced_mask] = region[voiced_mask] * factors[voiced_mask]
    new_f0[start_idx:] = region

    y_out = pw.synthesize(new_f0, sp, ap, sr)

    # Gentle end emphasis: apply gain envelope on last ~150ms
    tail_ms = 150
    tail_samples = int(sr * tail_ms / 1000.0)
    if tail_samples > 0 and tail_samples < len(y_out):
        env = np.ones_like(y_out)
        ramp = np.linspace(1.0, emphasis_gain, tail_samples)
        env[-tail_samples:] *= ramp
        y_out = y_out * env

    peak = np.max(np.abs(y_out))
    if peak > 0:
        y_out = (y_out / peak) * 0.99
    sf.write(output_path, y_out, sr)


def _append_silence(y: np.ndarray, sr: int, ms: float) -> np.ndarray:
    if ms <= 0:
        return y
    pad = np.zeros(int(sr * ms / 1000.0), dtype=y.dtype)
    return np.concatenate([y, pad])


def add_period_cadence(
    input_path: str,
    output_path: str,
    fall_fraction: float = 0.25,
    fall_semitones: float = 3.0,
    extra_silence_ms: float = 150.0,
    trim_silence: bool = True,
) -> None:
    """
    Add a slight falling pitch at the end and a short pause to simulate a period cadence.
    """
    y, sr = librosa.load(input_path, sr=None, mono=True)
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=30)
        if y.size == 0:
            raise RuntimeError("Audio became empty after trimming silence")
    y = y.astype(np.float64, copy=False)

    f0, sp, ap = pw.wav2world(y, sr)
    n_frames = len(f0)
    if n_frames < 10:
        raise RuntimeError("Too few WORLD frames; input likely too short or silent.")

    start_idx = int((1.0 - fall_fraction) * n_frames)
    start_idx = max(0, min(start_idx, n_frames - 1))
    min_factor = 2.0 ** (-abs(fall_semitones) / 12.0)
    n_fall_frames = max(1, n_frames - start_idx)
    factors = np.linspace(1.0, min_factor, n_fall_frames)

    new_f0 = f0.copy()
    region = new_f0[start_idx:]
    voiced_mask = region > 0
    region[voiced_mask] = region[voiced_mask] * factors[voiced_mask]
    new_f0[start_idx:] = region

    y_out = pw.synthesize(new_f0, sp, ap, sr)
    y_out = _append_silence(y_out, sr, extra_silence_ms)
    peak = np.max(np.abs(y_out))
    if peak > 0:
        y_out = (y_out / peak) * 0.99
    sf.write(output_path, y_out, sr)


def add_comma_pause(
    input_path: str,
    output_path: str,
    fall_fraction: float = 0.15,
    fall_semitones: float = 2.0,
    extra_silence_ms: float = 80.0,
    trim_silence: bool = True,
) -> None:
    """
    Add a tiny falling pitch and a short pause to simulate a comma break.
    """
    y, sr = librosa.load(input_path, sr=None, mono=True)
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=30)
        if y.size == 0:
            raise RuntimeError("Audio became empty after trimming silence")
    y = y.astype(np.float64, copy=False)

    f0, sp, ap = pw.wav2world(y, sr)
    n_frames = len(f0)
    if n_frames < 10:
        raise RuntimeError("Too few WORLD frames; input likely too short or silent.")

    start_idx = int((1.0 - fall_fraction) * n_frames)
    start_idx = max(0, min(start_idx, n_frames - 1))
    min_factor = 2.0 ** (-abs(fall_semitones) / 12.0)
    n_fall_frames = max(1, n_frames - start_idx)
    factors = np.linspace(1.0, min_factor, n_fall_frames)

    new_f0 = f0.copy()
    region = new_f0[start_idx:]
    voiced_mask = region > 0
    region[voiced_mask] = region[voiced_mask] * factors[voiced_mask]
    new_f0[start_idx:] = region

    y_out = pw.synthesize(new_f0, sp, ap, sr)
    y_out = _append_silence(y_out, sr, extra_silence_ms)
    peak = np.max(np.abs(y_out))
    if peak > 0:
        y_out = (y_out / peak) * 0.99
    sf.write(output_path, y_out, sr)


def _main():
    parser = argparse.ArgumentParser(description="Add rising question intonation to a short spoken WAV.")
    parser.add_argument("input", help="Input WAV file (mono speech)")
    parser.add_argument("output", help="Output WAV file")
    parser.add_argument("--rise-fraction", type=float, default=0.35, help="Fraction over which pitch rises")
    parser.add_argument("--rise-semitones", type=float, default=4.0, help="Total pitch rise in semitones")
    parser.add_argument("--no-trim", action="store_true", help="Disable trimming leading/trailing silence")
    args = parser.parse_args()
    add_question_intonation(
        input_path=args.input,
        output_path=args.output,
        rise_fraction=args.rise_fraction,
        rise_semitones=args.rise_semitones,
        trim_silence=not args.no_trim,
    )


if __name__ == "__main__":
    _main()


