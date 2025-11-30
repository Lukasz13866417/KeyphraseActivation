import os, sys, time, random, json, string, unicodedata, traceback, gc
from pathlib import Path
from typing import List, Optional, Set, Dict, Any

import soundfile as sf
from bark import SAMPLE_RATE, generate_audio, preload_models


DEFAULT_SPEAKERS: List[str] = [
    "v2/en_speaker_0",
    "v2/en_speaker_1",
    "v2/en_speaker_2",
    "v2/en_speaker_3",
    "v2/en_speaker_4",
    "v2/en_speaker_5",
]

# Conservative punctuation set that Bark handles naturally
BARK_ALLOWED_PUNCT: Set[str] = set(".,!?;:'\"()- ")
MAX_TEXT_LEN = 300  # Bark degrades/hallucinates with long text; cap to prevent OOM/hangs


def _log(msg: str, *, quiet: bool):
    if not quiet:
        print(msg, flush=True)


def _unicode_to_ascii_punct(text: str) -> str:
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
            out.append(replacements[ch])
            continue
        if ch in ZW_CHARS:
            continue
        if ch == "\u00a0":
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out)


def _normalize_punctuation_for_speaker(text: str, allowed_punct: Set[str]) -> str:
    if not text:
        return text
    mapped = _unicode_to_ascii_punct(text)
    ascii_punct = set(string.punctuation) | {" "}
    out_chars = []
    for ch in mapped:
        if ch in ascii_punct:
            if ch in allowed_punct:
                out_chars.append(ch)
            else:
                continue
        else:
            out_chars.append(ch)
    return "".join(out_chars)


def _preload_bark_models_with_safe_globals() -> None:
    """Preload Bark models using non-safe loading (trusted checkpoints).

    Force torch.load(..., weights_only=False) for all loads during preload.
    """
    import torch  # type: ignore

    original_load = torch.load
    try:
        original_serialization_load = torch.serialization.load  # type: ignore
    except Exception:
        original_serialization_load = None  # type: ignore

    def _wrapped_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    def _wrapped_serialization_load(*args, **kwargs):  # type: ignore
        kwargs["weights_only"] = False
        return original_serialization_load(*args, **kwargs)  # type: ignore

    torch.load = _wrapped_load  # type: ignore
    if original_serialization_load is not None:
        torch.serialization.load = _wrapped_serialization_load  # type: ignore

    # Also patch bark.generation's cached torch if present
    bark_gen = None
    try:
        import bark.generation as bark_gen  # type: ignore
    except Exception:
        bark_gen = None
    gen_torch_patched = False
    gen_ser_patched = False
    if bark_gen is not None and hasattr(bark_gen, "torch"):
        try:
            if getattr(bark_gen.torch, "load", None) is not None:
                bark_gen.torch.load = _wrapped_load  # type: ignore
                gen_torch_patched = True
            if original_serialization_load is not None and getattr(bark_gen.torch, "serialization", None) is not None:
                if getattr(bark_gen.torch.serialization, "load", None) is not None:
                    bark_gen.torch.serialization.load = _wrapped_serialization_load  # type: ignore
                    gen_ser_patched = True
        except Exception:
            pass

    try:
        preload_models()
    finally:
        torch.load = original_load  # type: ignore
        if original_serialization_load is not None:
            torch.serialization.load = original_serialization_load  # type: ignore
        if bark_gen is not None and hasattr(bark_gen, "torch"):
            try:
                if gen_torch_patched:
                    bark_gen.torch.load = original_load  # type: ignore
                if gen_ser_patched and original_serialization_load is not None:
                    bark_gen.torch.serialization.load = original_serialization_load  # type: ignore
            except Exception:
                pass


def _preload_guarded(*, quiet: bool) -> None:
    try:
        _preload_bark_models_with_safe_globals()
    except MemoryError:
        print("[BARK] [fatal] Ran out of memory while loading models.", file=sys.stderr)
        raise
    except Exception:
        print("[BARK] [fatal] Failed to preload Bark models:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise

def generate(
    phrase: str,
    num_samples: int,
    output_dir: str = "samples",
    *,
    models: Optional[List[str]] = None,
    num_models: Optional[int] = None,
    quiet: bool = False,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    jitter: float = 0.0,
    api_name: str = "bark",
) -> List[Dict[str, Any]]:
    if not phrase or not phrase.strip():
        raise ValueError("Phrase must be non-empty.")
    if len(phrase) > MAX_TEXT_LEN:
        print(f"[BARK] [warn] Phrase too long ({len(phrase)} chars), truncating to {MAX_TEXT_LEN} to prevent crash.", file=sys.stderr)
        phrase = phrase[:MAX_TEXT_LEN]

    if num_samples < 1:
        raise ValueError("num_samples must be >= 1.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = list(models) if models else list(DEFAULT_SPEAKERS)
    if not pool:
        raise RuntimeError("No speaker models provided and DEFAULT_SPEAKERS is empty.")
    if num_models is None:
        num_models = len(pool)
    num_models = max(1, min(num_models, len(pool)))
    random.shuffle(pool)
    selected = pool[:num_models]
    if not quiet:
        print(f"[BARK] [voices] using {len(selected)}/{len(pool)} preset(s): {', '.join(selected)}")

    # Load Bark models once
    t0_load = time.time()
    _preload_guarded(quiet=quiet)
    if not quiet:
        print(f"[BARK] [loaded] models in {time.time()-t0_load:.1f}s")

    def _safe_component(text: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_", "/"} else "_" for ch in text)

    generated_records: List[Dict[str, Any]] = []
    for i in range(num_samples):
        speaker = selected[i % len(selected)]
        if jitter > 0:
            def j(x: float) -> float:
                return max(0.05, x * (1.0 + random.uniform(-jitter, jitter)))
            tt = j(text_temp)
            wt = j(waveform_temp)
        else:
            tt, wt = text_temp, waveform_temp

        normalized_phrase = _normalize_punctuation_for_speaker(phrase, BARK_ALLOWED_PUNCT)

        safe_api = _safe_component(api_name).replace('/', '_')
        safe_model = _safe_component(speaker).replace('/', '_')
        out_path = out_dir / f"{safe_api}_{safe_model}_{i:04d}.wav"

        try:
            audio_array = generate_audio(
                normalized_phrase,
                history_prompt=speaker,
                text_temp=tt,
                waveform_temp=wt,
            )
            if audio_array is None or len(audio_array) == 0:
                raise RuntimeError("no audio produced (empty array)")
            sf.write(str(out_path), audio_array, SAMPLE_RATE)
            _log(f"[BARK] [ok] {out_path.name} | text_temp={tt:.2f} wave_temp={wt:.2f}", quiet=quiet)
            duration = float(len(audio_array)) / float(SAMPLE_RATE) if SAMPLE_RATE else 0.0
            generated_records.append({
                "path": str(out_path.resolve()),
                "sample_rate": SAMPLE_RATE,
                "duration_sec": duration,
                "model_name": speaker,
                "api_name": api_name,
            })
            # Explicit cleanup to prevent memory creep
            del audio_array
            gc.collect()
        except MemoryError:
            print(
                f"[BARK] [fatal] Ran out of memory while synthesizing sample {i}. Halting generation.",
                file=sys.stderr,
            )
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            break
        except Exception as e:
            print(f"[BARK] [err] sample {i} failed: {e}", file=sys.stderr)
            if not quiet:
                print(traceback.format_exc(), file=sys.stderr)
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass

    _log(f"[BARK] [done] wrote up to {num_samples} files in {out_dir}", quiet=quiet)
    try:
        out_file = os.environ.get("RETURN_PATHS_FILE")
        if out_file:
            try:
                Path(out_file).write_text(json.dumps(generated_records), encoding="utf-8")
            except Exception as werr:
                print(f"[BARK] [warn] failed to write RETURN_PATHS_FILE: {werr}", file=sys.stderr)
        elif os.environ.get("RETURN_PATHS") == "1":
            print(json.dumps(generated_records), flush=True)
    except Exception:
        pass
    return generated_records


if __name__ == "__main__":
    generate(
        phrase="AI money machine",
        num_samples=3,
        output_dir="../../samples",
        quiet=False,
        text_temp=0.7,
        waveform_temp=0.7,
        api_name="bark",
    )


