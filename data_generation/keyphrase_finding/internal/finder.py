"""
Finds the timestamps of a keyphrase in an audio file, given the transcript (making this much easier).
Based on torchaudio CTC segmentation tutorial.
This will be necessary to reuse existing audio samples as positives for training (those that contain the keyphrase).
""" 

# TODO use the _natural_cut_sample function to crop cleanly (will help account for model's imperfections).
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torchaudio

# Use GPU if available, otherwise CPU.
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use torchaudio's pretrained CTC model (Wav2Vec2 on LibriSpeech).
_BUNDLE = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
_MODEL = None
_LABELS = None
_DICTIONARY = None
_BLANK_ID = None


def _load_model():
    """Lazy-load the ASR model and associated label dictionary."""
    global _MODEL, _LABELS, _DICTIONARY, _BLANK_ID
    if _MODEL is None:
        _MODEL = _BUNDLE.get_model().to(_DEVICE)
        _LABELS = _BUNDLE.get_labels()  # tuple like ('-', '|', 'E', 'T', 'A', ...)
        _DICTIONARY = {c: i for i, c in enumerate(_LABELS)}
        _BLANK_ID = _DICTIONARY["-"]
    return _MODEL, _LABELS, _DICTIONARY, _BLANK_ID


def _normalize_transcript(text: str) -> str:
    """
    Normalize text to the character set used by the model:

    - Uppercase letters A–Z and apostrophe.
    - Any non [A-Z'] becomes a space.
    - Collapse to words and join with '|' as word separator.

    Example:
        "I had that curiosity, beside me at this moment."
        -> "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT"
    """
    text = text.upper()
    text = re.sub(r"[^A-Z']+", " ", text)  # keep only A-Z and '
    words = [w for w in text.split() if w]
    return "|".join(words)


def _get_emission(wav_path: str) -> Tuple[torch.Tensor, int, int]:
    """
    Load audio and compute log-prob emissions from the CTC model.

    Returns:
        emission: Tensor [num_frames, num_labels]
        num_samples: number of samples in waveform
        sample_rate: sample rate used for the model (typically 16000)
    """
    model, _, _, _ = _load_model()
    waveform, sr = torchaudio.load(wav_path)

    target_sr = _BUNDLE.sample_rate
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    num_samples = waveform.shape[1]

    with torch.inference_mode():
        emissions, _ = model(waveform.to(_DEVICE))
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu()  # [num_frames, num_labels]
    return emission, num_samples, target_sr


@dataclass
class Point:
    token_index: int  # index in transcript string
    time_index: int   # frame index in emission
    score: float      # probability of this step


@dataclass
class Segment:
    label: str        # character or word
    start: int        # start frame index (inclusive)
    end: int          # end frame index (exclusive)
    score: float      # average probability over the segment

    @property
    def length(self) -> int:
        return self.end - self.start


def _get_trellis(
    emission: torch.Tensor,
    tokens: List[int],
    blank_id: int,
) -> torch.Tensor:
    """
    Build the trellis matrix as in the tutorial.

    emission: [num_frames, num_labels]
    tokens: list of label indices corresponding to each character in normalized transcript
    """
    num_frames = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.full((num_frames + 1, num_tokens + 1), -float("inf"))
    trellis[:, 0] = 0  # starting from blank

    for t in range(num_frames):
        # Staying at the same token vs moving to next token.
        stay = trellis[t, 1:] + emission[t, blank_id]
        change = trellis[t, :-1] + emission[t, tokens]
        trellis[t + 1, 1:] = torch.maximum(stay, change)

    return trellis


def _backtrack(
    trellis: torch.Tensor,
    emission: torch.Tensor,
    tokens: List[int],
    blank_id: int,
) -> List[Point]:
    """
    Backtrack on the trellis to find the most likely alignment path.
    """
    j = trellis.size(1) - 1  # last token index (in trellis coordinates)
    # Find time index where last token is most likely
    t_start = torch.argmax(trellis[:, j]).item()

    path: List[Point] = []
    for t in range(t_start, 0, -1):
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        if changed > stayed:
            prob = emission[t - 1, tokens[j - 1]].exp().item()
            path.append(Point(token_index=j - 1, time_index=t - 1, score=prob))
            j -= 1
            if j == 0:
                break
        else:
            prob = emission[t - 1, blank_id].exp().item()
            path.append(Point(token_index=j - 1, time_index=t - 1, score=prob))

    else:
        raise ValueError("Failed to align transcript to audio (backtracking did not reach start).")

    return list(reversed(path))


def _merge_repeats(path: List[Point], transcript: str) -> List[Segment]:
    """
    Merge consecutive points belonging to the same transcript index into char-level segments.
    """
    segments: List[Segment] = []
    i1 = 0
    while i1 < len(path):
        i2 = i1
        while i2 < len(path) and path[i2].token_index == path[i1].token_index:
            i2 += 1

        scores = [path[k].score for k in range(i1, i2)]
        avg_score = sum(scores) / len(scores)
        label = transcript[path[i1].token_index]
        start = path[i1].time_index
        end = path[i2 - 1].time_index + 1

        segments.append(Segment(label=label, start=start, end=end, score=avg_score))
        i1 = i2

    return segments


def _merge_words(segments: List[Segment], separator: str = "|") -> List[Segment]:
    """
    Merge character-level segments into word-level segments using 'separator' as the boundary.
    """
    words: List[Segment] = []
    i1, i2 = 0, 0

    while i1 < len(segments):
        # Move i2 until we hit a separator or end.
        while i2 < len(segments) and segments[i2].label != separator:
            i2 += 1

        if i1 != i2:
            # characters for this word are segments[i1:i2]
            char_segments = segments[i1:i2]
            word_label = "".join(seg.label for seg in char_segments)
            total_len = sum(seg.length for seg in char_segments)
            avg_score = sum(seg.score * seg.length for seg in char_segments) / total_len
            words.append(
                Segment(
                    label=word_label,
                    start=char_segments[0].start,
                    end=char_segments[-1].end,
                    score=avg_score,
                )
            )

        # Skip the separator itself
        i2 += 1
        i1 = i2

    return words


def _find_subsequence(haystack: List[str], needle: List[str]) -> int:
    """
    Find the index of the first occurrence of 'needle' as a contiguous subsequence of 'haystack'.
    Assumes needle appears exactly once; raises ValueError otherwise.
    """
    n = len(needle)
    if n == 0:
        raise ValueError("Keyphrase is empty after normalization.")

    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i

    raise ValueError("Keyphrase not found in transcript (after normalization).")


def find_keyphrase_timestamps(
    wav_path: str,
    transcript: str,
    keyphrase: str,
) -> Tuple[float, float]:
    """
    Align a known transcript to audio and return (start_sec, end_sec) of a keyphrase.

    Args:
        wav_path: path to .wav file
        transcript: full transcript (string)
        keyphrase: phrase that appears exactly once in the transcript

    Returns:
        (start_time_seconds, end_time_seconds)
    """
    # Normalize transcript and keyphrase into model's char space
    norm_transcript = _normalize_transcript(transcript)
    norm_keyphrase = _normalize_transcript(keyphrase)

    full_words = norm_transcript.split("|")
    key_words = norm_keyphrase.split("|")

    # Get emissions from audio
    emission, num_samples, sample_rate = _get_emission(wav_path)
    model, labels, dictionary, blank_id = _load_model()

    # Convert normalized transcript to label IDs
    try:
        tokens = [dictionary[c] for c in norm_transcript]
    except KeyError as e:
        raise ValueError(f"Character {e} in normalized transcript is not in model vocabulary.") from e

    # CTC segmentation
    trellis = _get_trellis(emission, tokens, blank_id=blank_id)
    path = _backtrack(trellis, emission, tokens, blank_id=blank_id)

    # Merge to char segments then word segments
    char_segments = _merge_repeats(path, norm_transcript)
    word_segments = _merge_words(char_segments, separator="|")

    # Sanity check: words from alignment match normalized transcript words
    aligned_words = [seg.label for seg in word_segments]
    if aligned_words != full_words:
        # Not necessarily fatal, but useful to know if it goes weird
        raise RuntimeError(
            "Aligned words differ from normalized transcript words.\n"
            f"Transcript words: {full_words}\n"
            f"Aligned words:    {aligned_words}"
        )

    # Locate keyphrase as contiguous word subsequence
    start_word_idx = _find_subsequence(aligned_words, key_words)
    end_word_idx = start_word_idx + len(key_words) - 1

    first_seg = word_segments[start_word_idx]
    last_seg = word_segments[end_word_idx]

    # Convert frame indices to seconds
    num_frames = emission.size(0)
    audio_duration_sec = num_samples / float(sample_rate)
    time_per_frame = audio_duration_sec / num_frames

    start_time_sec = first_seg.start * time_per_frame
    end_time_sec = last_seg.end * time_per_frame

    return start_time_sec, end_time_sec


def find_keyphrase_segment_with_margin(
    wav_path: str,
    transcript: str,
    keyphrase: str,
    *,
    pre_padding_sec: float = 0.15,
    post_padding_sec: float = 0.15,
    min_duration_sec: float = 0.6,
    max_duration_sec: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Return a slightly longer segment that contains the keyphrase plus some part of surrounding words
    (to resemble other positives more).

    Args:
        wav_path: Path to the audio file.
        transcript: Transcript matching the audio.
        keyphrase: Phrase to locate.
        pre_padding_sec: Seconds to include before the detected start.
        post_padding_sec: Seconds to include after the detected end.
        min_duration_sec: Ensure the final window is at least this long (if possible).
        max_duration_sec: Cap the window length (if provided).
    """
    start, end = find_keyphrase_timestamps(wav_path, transcript, keyphrase)
    if end <= start:
        return start, end

    info = torchaudio.info(wav_path)
    total_duration = info.num_frames / float(info.sample_rate)

    start = max(0.0, start - max(0.0, pre_padding_sec))
    end = min(total_duration, end + max(0.0, post_padding_sec))

    if min_duration_sec and (end - start) < min_duration_sec:
        deficit = min_duration_sec - (end - start)
        shift = deficit / 2.0
        start = max(0.0, start - shift)
        end = min(total_duration, end + shift)
        # If still too short due to hitting boundaries, stretch to min duration where possible.
        if (end - start) < min_duration_sec and total_duration >= min_duration_sec:
            if start == 0.0:
                end = min(total_duration, min_duration_sec)
            elif end == total_duration:
                start = max(0.0, total_duration - min_duration_sec)

    if max_duration_sec and (end - start) > max_duration_sec:
        center = 0.5 * (start + end)
        half = max_duration_sec / 2.0
        start = max(0.0, center - half)
        end = min(total_duration, center + half)
        if (end - start) > max_duration_sec + 1e-6:
            end = min(total_duration, start + max_duration_sec)

    return start, end


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python find_keyphrase_timestamps.py audio.wav \"full transcript\" \"keyphrase\"")
        sys.exit(1)

    wav_path = sys.argv[1]
    transcript = sys.argv[2]
    keyphrase = sys.argv[3]

    s, e = find_keyphrase_timestamps(wav_path, transcript, keyphrase)
    print(f"Keyphrase timestamps: start={s:.3f}s, end={e:.3f}s")
