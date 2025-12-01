import json
from typing import Dict, List, Optional
from pathlib import Path

# Import drivers from sibling subprojects
from piper_side.driver import synthesize as piper_synthesize
from suno_side.driver import synthesize as bark_synthesize
from elevenlabs_side.driver import synthesize as eleven_synthesize
from kokoro_side.driver import synthesize as kokoro_synthesize


# Default proportions for how many samples to generate from each TTS engine.
PROPORTION_PIPER = 0.4
PROPORTION_BARK = 0.1
PROPORTION_KOKORO = 0.2
PROPORTION_ELEVEN = 1.0 - PROPORTION_PIPER - PROPORTION_BARK - PROPORTION_KOKORO

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _distribute_counts(total: int, proportions: Dict[str, float]) -> Dict[str, int]:
    """Distribute the total number of samples to each TTS engine according to the proportions."""
    keys = list(proportions.keys())
    counts = {k: int(total * max(0.0, proportions[k])) for k in keys}
    allocated = sum(counts.values())
    # Distribute remainder by largest fractional part
    remainders = sorted(
        ((k, (total * proportions[k]) - counts[k]) for k in keys),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for k, _ in remainders:
        if allocated >= total:
            break
        counts[k] += 1
        allocated += 1
    return counts


def generate_for_phrase(
    phrase: str,
    num_samples: int,
    *,
    output_dir: Optional[str] = None,
    piper_kwargs: Optional[dict] = None,
    bark_kwargs: Optional[dict] = None,
    kokoro_kwargs: Optional[dict] = None,
    eleven_kwargs: Optional[dict] = None,
) -> List[dict]:
    """
    Generate samples for the same phrase using all TTS engines. Returns a combined list of JSON records from drivers.
    """
    if not phrase or not phrase.strip():
        raise ValueError("Phrase must be non-empty.")
    if num_samples < 1:
        return []

    
    if output_dir is None:
        output_dir = str((_project_root() / "samples").resolve())

    piper_kwargs = piper_kwargs or {}
    bark_kwargs = bark_kwargs or {}
    kokoro_kwargs = kokoro_kwargs or {}
    eleven_kwargs = eleven_kwargs or {}

    proportions = {
        "piper": PROPORTION_PIPER,
        "bark": PROPORTION_BARK,
        "kokoro": PROPORTION_KOKORO,
        "eleven": PROPORTION_ELEVEN,
    }
    counts = _distribute_counts(num_samples, proportions)

    all_records: List[dict] = []

    # Piper
    if counts["piper"] > 0:
        try:
            recs = piper_synthesize(
                phrase=phrase,
                num_samples=counts["piper"],
                output_dir=output_dir,
                **piper_kwargs,
            )
            all_records.extend(recs or [])
        except Exception as e:
            print(f"[audio_generation] Piper failed: {e}")

    # Bark
    if counts["bark"] > 0:
        try:
            recs = bark_synthesize(
                phrase=phrase,
                num_samples=counts["bark"],
                output_dir=output_dir,
                **bark_kwargs,
            )
            all_records.extend(recs or [])
        except Exception as e:
            print(f"[audio_generation] Bark failed: {e}")

    # Kokoro
    if counts["kokoro"] > 0:
        try:
            recs = kokoro_synthesize(
                text=phrase,
                num_samples=counts["kokoro"],
                output_dir=output_dir,
                **kokoro_kwargs,
            )
            all_records.extend(recs or [])
        except Exception as e:
            print(f"[audio_generation] Kokoro failed: {e}")

    print(f"Counts: {counts}")

    # 11
    if counts["eleven"] > 0:
        try:
            recs = eleven_synthesize(
                phrase=phrase,
                num_samples=counts["eleven"],
                output_dir=output_dir,
                **eleven_kwargs,
            )
            all_records.extend(recs or [])
        except Exception as e:
            print(f"[audio_generation] ElevenLabs failed: {e}")

    return all_records


if __name__ == "__main__":
    records = generate_for_phrase(
        phrase="AI money machine",
        num_samples=6,
    )
    print(json.dumps(records, indent=2))


