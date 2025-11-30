# piper_batch_generate.py
import os, sys, time, random, requests, wave, json, string, unicodedata
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict, Any
from piper.voice import PiperVoice, SynthesisConfig 

BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

DEFAULT_VOICES: List[str] = [
    "en_US-lessac-high",
    "en_US-libritts-high",
    "en_US-ljspeech-high",
    "en_US-amy-medium",
    "en_US-bryce-medium",
    "en_US-joe-medium",
    "en_US-john-medium",
    "en_US-kristin-medium",
    "en_US-norman-medium",
    "en_US-ryan-medium",
    "en_US-sam-medium",
    "en_GB-cori-high",
    "en_GB-semaine-medium",
    "en_GB-alan-medium",
    "en_GB-alba-medium",
    "en_GB-aru-medium",
    "en_GB-jenny_dioco-medium",
    "en_GB-northern_english_male-medium",
    "en_GB-southern_english_female-medium",
]

def _log(msg: str, *, quiet: bool):
    if not quiet:
        print(msg, flush=True)

def _voice_paths(model_id: str) -> tuple[str, str, str, str]:
    try:
        locale_voice, quality = model_id.rsplit("-", 1)
        locale, voice_name = locale_voice.split("-", 1)
    except ValueError:
        raise RuntimeError(f"Model identifier '{model_id}' must look like 'en_US-lessac-medium'")
    base_lang = locale.split("_")[0]
    onnx_filename = f"{model_id}.onnx"
    json_filename = f"{model_id}.onnx.json"
    onnx_url = f"{BASE_URL}/{base_lang}/{locale}/{voice_name}/{quality}/{onnx_filename}?download=true"
    json_url = f"{BASE_URL}/{base_lang}/{locale}/{voice_name}/{quality}/{json_filename}?download=true"
    return onnx_filename, json_filename, onnx_url, json_url

def download_voice_model(model_id: str, models_dir: Path, *, quiet: bool=False, timeout: int=120) -> tuple[Path, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_filename, json_filename, onnx_url, json_url = _voice_paths(model_id)
    onnx_path = models_dir / onnx_filename
    json_path = models_dir / json_filename

    if onnx_path.exists() and json_path.exists():
        _log(f"[PIPER] [cache] {model_id} already present.", quiet=quiet)
        return onnx_path, json_path

    _log(f"[PIPER] [download] {model_id}…", quiet=quiet)
    try:
        r = requests.get(onnx_url, timeout=timeout)
        r.raise_for_status()
        onnx_path.write_bytes(r.content)
    except Exception as e:
        raise RuntimeError(f"Failed to download ONNX: {onnx_url}\n{e}")

    try:
        r = requests.get(json_url, timeout=timeout//2)
        r.raise_for_status()
        json_path.write_bytes(r.content)
    except Exception as e:
        if onnx_path.exists():
            onnx_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download JSON: {json_url}\n{e}")

    return onnx_path, json_path

def _compute_allowed_punctuation(json_path: Path) -> Set[str]:
    """Return ASCII punctuation characters supported by the model (from phoneme_id_map)."""
    try:
        cfg = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    id_map = cfg.get("phoneme_id_map", {})
    ascii_punct = set(string.punctuation)
    # Keep only single-character ASCII punctuation present in the model's token map
    allowed = {ch for ch in id_map.keys() if isinstance(ch, str) and len(ch) == 1 and ch in ascii_punct}
    return allowed

def _safe_filename_component(text: str) -> str:
    """Return a filesystem-friendly version of a string.
    Keeps alphanumerics, dash and underscore; replaces others with underscore.
    """
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)

def _unicode_to_ascii_punct(text: str) -> str:
    """Map common Unicode punctuation to ASCII equivalents without adding new punctuation semantics."""
    # Normalize compatibility forms
    text = unicodedata.normalize("NFKC", text)
    # Explicit mappings for punctuation commonly seen in inputs
    replacements = {
        # Quotes
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "＂": '"',
        "‘": "'", "’": "'", "‚": "'", "＇": "'",
        # Dashes/Hypens
        "—": "-", "–": "-", "−": "-", "‑": "-", "‒": "-", "―": "-",
        # Ellipsis → single period to avoid amplifying punctuation
        "…": ".",
        # Fullwidth/CJK punctuation to ASCII
        "，": ",", "。": ".", "！": "!", "？": "?", "：": ":", "；": ";",
        "（": "(", "）": ")",
        # Misc
        "·": ".",
    }
    # Zero-width and BOM characters removed
    ZW_CHARS = {
        "\u200b", # zero width space
        "\u200c", # zero width non-joiner
        "\u200d", # zero width joiner
        "\ufeff", # BOM
    }
    out_chars = []
    for ch in text:
        if ch in replacements:
            out_chars.append(replacements[ch])
            continue
        if ch in ZW_CHARS:
            continue
        # NBSP -> space
        if ch == "\u00a0":
            out_chars.append(" ")
            continue
        out_chars.append(ch)
    return "".join(out_chars)

def _normalize_punctuation_for_voice(text: str, allowed_punct: Set[str]) -> str:
    """Normalize punctuation for a specific model's allowed set.
    - Map Unicode punctuation to ASCII equivalents
    - Drop ASCII punctuation not supported by the model
    - Do not insert new punctuation or change spacing/grammar
    """
    if not text:
        return text
    mapped = _unicode_to_ascii_punct(text)
    ascii_punct = set(string.punctuation)
    out_chars = []
    for ch in mapped:
        if ch in ascii_punct:
            if ch in allowed_punct:
                out_chars.append(ch)
            else:
                # drop unsupported punctuation
                continue
        else:
            out_chars.append(ch)
    return "".join(out_chars)

def _load_voice(model_id: str, models_dir: Path, *, quiet: bool=False) -> tuple[PiperVoice, Path, Set[str]]:
    onnx_path, json_path = download_voice_model(model_id, models_dir, quiet=quiet)
    try:
        voice = PiperVoice.load(model_path=str(onnx_path), config_path=str(json_path))
        allowed_punct = _compute_allowed_punctuation(json_path)
        _log(f"[PIPER] [loaded] {model_id} @ {onnx_path.name}", quiet=quiet)
        return voice, json_path, allowed_punct
    except Exception as e:
        raise RuntimeError(f"Failed to load {model_id}: {e}")

def generate(
    phrase: str,
    num_samples: int,
    output_dir: str = "samples",
    *,
    models: Optional[List[str]] = None,
    num_models: Optional[int] = None,
    models_dir: str = "models",
    quiet: bool = False,
    length_scale: float = 1.0,    # baseline speech speed control
    noise_scale: float = 0.667,   # overall non-determinism
    noise_w_scale: float = 0.333, # non-determinism timing and length of phonemes
    jitter: float = 0.0,          # how these params vary from sample to sample
    api_name: str = "piper",
) -> List[Dict[str, Any]]:
    """
    Generate WAV samples of `phrase` cycling through selected Piper voices.

    - If `models` is None, picks voices from DEFAULT_VOICES.
    - If `num_models` is None, uses ALL available in `models` (or ALL defaults).
    - Downloads missing (.onnx + .onnx.json) into `models_dir`.
    - Uses Piper 1.3.0 API (AudioChunk iterator).
    """
    if not phrase or not phrase.strip():
        raise ValueError("Phrase must be non-empty.")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1.")

    models_dir = Path(models_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose voices
    pool = list(models) if models else list(DEFAULT_VOICES)
    if not pool:
        raise RuntimeError("No model IDs provided and DEFAULT_VOICES is empty.")
    if num_models is None:
        num_models = len(pool)               # <- use ALL, not 1
    num_models = max(1, min(num_models, len(pool)))
    random.shuffle(pool)
    selected = pool[:num_models]
    if not quiet:
        print(f"[PIPER] [voices] using {len(selected)}/{len(pool)} voice(s): {', '.join(selected)}")

    # Load voices
    voices: List[tuple[PiperVoice, Set[str], str]] = []  # (voice, allowed_punct, model_id)
    for mid in selected:
        try:
            v, cfg_path, allowed_punct = _load_voice(mid, models_dir, quiet=quiet)
            voices.append((v, allowed_punct, mid))
        except Exception as e:
            print(f"[PIPER] [warn] skipping {mid}: {e}", file=sys.stderr)

    if not voices:
        raise RuntimeError("No voices loaded. Aborting.")

    # Synthesize
    t0 = time.time()
    generated_records: List[Dict[str, Any]] = []
    for i in range(num_samples):
        v, allowed_punct, model_id = voices[i % len(voices)]
        # small per-sample jitter if requested
        if jitter > 0:
            def j(x):  # +/- jitter fraction
                return max(0.05, x * (1.0 + random.uniform(-jitter, jitter)))
            ls = j(length_scale)
            ns = j(noise_scale)
            nws = j(noise_w_scale)
        else:
            ls, ns, nws = length_scale, noise_scale, noise_w_scale

        cfg = SynthesisConfig(length_scale=ls, noise_scale=ns, noise_w_scale=nws)

        # Per-voice punctuation normalization (no augmentation)
        normalized_phrase = _normalize_punctuation_for_voice(phrase, allowed_punct)

        safe_model = _safe_filename_component(model_id)
        safe_api = _safe_filename_component(api_name)
        out_path = out_dir / f"{safe_api}_{safe_model}_{i:04d}.wav"
        try:
            with wave.open(str(out_path), "wb") as wf:
                first = True
                frames = 0
                for chunk in v.synthesize(normalized_phrase, cfg):
                    if first:
                        wf.setnchannels(chunk.sample_channels)
                        wf.setsampwidth(chunk.sample_width)    # 2 bytes (PCM16)
                        wf.setframerate(chunk.sample_rate)     # model-native SR
                        first = False
                    wf.writeframes(chunk.audio_int16_bytes)
                    frames += len(chunk.audio_int16_bytes) // chunk.sample_width
            if frames == 0:
                out_path.unlink(missing_ok=True)
                raise RuntimeError("no audio produced (0 frames)")
            _log(f"[PIPER] [ok] {out_path.name} | ls={ls:.2f} ns={ns:.2f} nw={nws:.2f}", quiet=quiet)
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
                "model_name": model_id,
                "api_name": api_name,
            })
        except Exception as e:
            print(f"[PIPER] [err] sample {i} failed: {e}", file=sys.stderr)
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass

    _log(f"[PIPER] [done] wrote up to {num_samples} files in {out_dir} in {time.time()-t0:.1f}s", quiet=quiet)
    try:
        out_file = os.environ.get("RETURN_PATHS_FILE")
        if out_file:
            try:
                Path(out_file).write_text(json.dumps(generated_records), encoding="utf-8")
            except Exception as werr:
                print(f"[PIPER] [warn] failed to write RETURN_PATHS_FILE: {werr}", file=sys.stderr)
        elif os.environ.get("RETURN_PATHS") == "1":
            print(json.dumps(generated_records), flush=True)
    except Exception:
        pass
    return generated_records

if __name__ == "__main__":
    # Example:
    generate(
        phrase="AI money machine",
        num_samples=20,
        output_dir="../../samples",
        models_dir="models",
        quiet=False,               # set True to silence logs
        num_models=None,           # None → use all in DEFAULT_VOICES
        jitter=0.05,               # small variety per sample
        length_scale=1.05,
        noise_scale=0.7,
        noise_w_scale=0.4,
    )
