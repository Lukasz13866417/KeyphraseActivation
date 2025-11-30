import random
from pathlib import Path
from typing import Dict, List, Optional

import itertools
import sys
import time

from datasets import load_dataset
import soundfile as sf


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_wavs_from_tps(
    num_clips: int,
    *,
    output_dir: Optional[str] = None,
    max_attempt_multiplier: int = 250,
    shuffle_buffer: int = 10_000,
) -> List[Dict[str, object]]:
    """
    Stream short audio clips from The People's Speech dataset, save them as WAV files,
    and return Piper-style metadata records describing each clip.

    Selects clips 2-10 words long and up to 6 seconds duration.
    """
    if num_clips <= 0:
        return []

    start_time = time.time()
    print(f"[TPS] Requesting {num_clips} clip(s) from MLCommons/peoples_speech...", flush=True)
    try:
        dataset = load_dataset("MLCommons/peoples_speech", "clean", split="train", streaming=True)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=random.randrange(10**6))
    except Exception as exc:
        print(f"[TPS] Failed to initialize dataset: {exc}", file=sys.stderr, flush=True)
        return []

    samples_dir = Path(output_dir).resolve() if output_dir else (_project_root() / "samples").resolve()
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    clips_saved = 0
    records: List[Dict[str, object]] = []
    max_attempts = max(num_clips * max_attempt_multiplier, num_clips)
    attempts = 0
    for sample in itertools.islice(dataset, max_attempts):
        attempts += 1
        text = sample["text"]
        duration_ms = sample["duration_ms"]
        # Count words in the transcript
        word_count = len(text.split())
        # Filter criteria: 2-10 words and <= 6000 ms (6 seconds)
        if 2 <= word_count <= 10 and duration_ms <= 6000:
            # Extract audio array and sampling rate (16 kHz):contentReference[oaicite:9]{index=9}:contentReference[oaicite:10]{index=10}
            audio_array = sample["audio"]["array"]  # numpy array of audio samples
            sample_rate = sample["audio"]["sampling_rate"]  # should be 16000 Hz
            # Construct a filename for the clip (e.g., tps_clip_1.wav, tps_clip_2.wav, ...)
            filename = samples_dir / f"tps_peoples_speech_clean_{clips_saved:04d}.wav"
            # Save the audio clip to disk as a WAV file:contentReference[oaicite:11]{index=11}
            sf.write(filename, audio_array, sample_rate)

            duration_sec = duration_ms / 1000.0 if duration_ms else len(audio_array) / sample_rate
            records.append(
                {
                    "path": str(filename.resolve()),
                    "sample_rate": sample_rate,
                    "duration_sec": duration_sec,
                    "model_name": "peoples_speech_clean",
                    "api_name": "tps",
                    "text": text,
                    "source": "MLCommons/peoples_speech",
                }
            )
            clips_saved += 1
            print(f"[TPS] Saved clip {clips_saved}/{num_clips} -> {filename.name}", flush=True)
            if clips_saved >= num_clips:
                break
        if attempts % 500 == 0:
            elapsed = time.time() - start_time
            print(
                f"[TPS] Attempts={attempts}, saved={clips_saved}, elapsed={elapsed:.1f}s "
                f"(target={num_clips}).",
                flush=True,
            )

    elapsed_total = time.time() - start_time
    if clips_saved < num_clips:
        print(
            f"[TPS] Only {clips_saved}/{num_clips} clips retrieved after {attempts} attempts "
            f"in {elapsed_total:.1f}s. Consider loosening filters or increasing max_attempt_multiplier.",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(f"[TPS] Completed {clips_saved} clip(s) in {elapsed_total:.1f}s.", flush=True)

    return records
