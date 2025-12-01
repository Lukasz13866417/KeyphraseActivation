import os, sys, time, random, json, string, unicodedata
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict, Any

import requests
import wave
import soundfile as sf


ELEVEN_API_BASE = "https://api.elevenlabs.io/v1"


def _log(msg: str, *, quiet: bool):
    if not quiet:
        print(msg, flush=True)


def _safe_component(text: str) -> str:
    """ Replace non-alphanumeric characters with underscores to get predictable file names. """
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def _unicode_to_ascii_punct(text: str) -> str:
    """ Map common Unicode punctuation to ASCII equivalents without adding new punctuation semantics.
    (to get predictable behavior from TTS models)"""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "＂": '"',
        "‘": "'", "’": "'", "‚": "'", "＇": "'",
        "—": "-", "–": "-", "−": "-", "‑": "-", "‒": "-", "―": "-",
        "…": ".",
        "，": ",", "。": ".", "！": "!", "？": "?", "：": ":", "；": ";",
        "（": "(", "）": ")",
        "·": ".",
    }
    ZW_CHARS = {"\u200b", "\u200c", "\u200d", "\ufeff"}
    out = []
    for ch in text:
        if ch in replacements:
            out.append(replacements[ch]); continue
        if ch in ZW_CHARS:
            continue
        if ch == "\u00a0":
            out.append(" "); continue
        out.append(ch)
    return "".join(out)


def _normalize_punctuation(text: str) -> str:
    """ Normalize punctuation for a specific model's allowed set.
    - Map Unicode punctuation to ASCII equivalents
    - Drop ASCII punctuation not supported by the model
    - Do not insert new punctuation or change spacing/grammar
    """
    if not text:
        return text
    mapped = _unicode_to_ascii_punct(text)
    allowed = set(string.printable)  # ElevenLabs tokenizer is robust, so we keep printable ASCII
    return "".join(ch for ch in mapped if ch in allowed)


def _require_api_key() -> str:
    api_key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("XI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ELEVENLABS_API_KEY (or XI_API_KEY) in environment.")
    return api_key


def _list_voices(api_key: str) -> List[Tuple[str, str]]:
    url = f"{ELEVEN_API_BASE}/voices"
    headers = {"xi-api-key": api_key}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    voices = []
    for v in data.get("voices", []):
        vid = v.get("voice_id") or ""
        name = v.get("name") or vid
        if vid:
            voices.append((vid, name))
    return voices


def _synthesize_wav(api_key: str, voice_id: str, text: str, *, stability: float, similarity_boost: float) -> bytes:
    url = f"{ELEVEN_API_BASE}/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "accept": "audio/wav",
        "content-type": "application/json",
    }
    body = {
        "text": text,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
        },
    }
    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    return r.content


def generate(
    phrase: str,
    num_samples: int,
    output_dir: str = "samples",
    *,
    models: Optional[List[str]] = None,  # voice IDs
    num_models: Optional[int] = None,
    quiet: bool = False,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    jitter: float = 0.0,
    api_name: str = "elevenlabs",
) -> List[Dict[str, Any]]:
    if not phrase or not phrase.strip():
        raise ValueError("Phrase must be non-empty.")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1.")

    api_key = _require_api_key()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose voices
    if models and len(models) > 0:
        available = [(m, m) for m in models]
    else:
        voices = _list_voices(api_key)
        if not voices:
            raise RuntimeError("No ElevenLabs voices available for this API key.")
        available = voices
    if num_models is None:
        num_models = max(1, len(available))
    num_models = max(1, min(num_models, len(available)))
    random.shuffle(available)
    selected = available[:num_models]
    if not quiet:
        pretty = ", ".join([f"{name}({vid[:8]}...)" for vid, name in selected])
        print(f"[11LABS] [voices] using {len(selected)}/{len(available)}: {pretty}")

    normalized_phrase = _normalize_punctuation(phrase)

    t0 = time.time()
    generated_records: List[Dict[str, Any]] = []
    for i in range(num_samples):
        voice_id, voice_name = selected[i % len(selected)]
        safe_api = _safe_component(api_name)
        # Prefer name if present, else voice_id prefix
        model_tag = voice_name if voice_name else voice_id[:8]
        safe_model = _safe_component(model_tag)
        out_path = out_dir / f"{safe_api}_{safe_model}_{i:04d}.wav"
        try:
            if jitter > 0:
                def j(x: float) -> float:
                    return max(0.0, min(1.0, x * (1.0 + random.uniform(-jitter, jitter))))
                st = j(stability)
                sb = j(similarity_boost)
            else:
                st = stability
                sb = similarity_boost
            audio_bytes = _synthesize_wav(
                api_key, voice_id, normalized_phrase,
                stability=st, similarity_boost=sb
            )
            with open(out_path, "wb") as f:
                f.write(audio_bytes)
            _log(f"[11LABS] [ok] {out_path.name} | stab={st:.2f} sim={sb:.2f}", quiet=quiet)
            # Prefer robust libsndfile for reading metadata; fall back to wave
            sr = None
            duration = None
            try:
                info = sf.info(str(out_path))
                if info is not None and info.samplerate:
                    sr = int(info.samplerate)
                    duration = (info.frames / float(info.samplerate)) if info.samplerate else None
            except Exception:
                try:
                    with wave.open(str(out_path), "rb") as rf:
                        sr = rf.getframerate()
                        nframes = rf.getnframes()
                    duration = (nframes / float(sr)) if sr else 0.0
                except Exception:
                    sr = None
                    duration = None
            generated_records.append({
                "path": str(out_path.resolve()),
                "sample_rate": sr,
                "duration_sec": duration,
                "model_name": (voice_name or voice_id),
                "api_name": api_name,
            })
        except Exception as e:
            print(f"[11LABS] [err] sample {i} failed: {e}", file=sys.stderr)
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass

    _log(f"[11LABS] [done] wrote up to {num_samples} files in {out_dir} in {time.time()-t0:.1f}s", quiet=quiet)
    try:
        out_file = os.environ.get("RETURN_PATHS_FILE")
        if out_file:
            try:
                Path(out_file).write_text(json.dumps(generated_records), encoding="utf-8")
            except Exception as werr:
                print(f"[11LABS] [warn] failed to write RETURN_PATHS_FILE: {werr}", file=sys.stderr)
        elif os.environ.get("RETURN_PATHS") == "1":
            print(json.dumps(generated_records), flush=True)
    except Exception:
        pass
    return generated_records


if __name__ == "__main__":
    generate(
        phrase="AI money machine.",
        num_samples=3,
        output_dir="../../samples",
        quiet=False,
        stability=0.5,
        similarity_boost=0.75,
        api_name="elevenlabs",
    )


