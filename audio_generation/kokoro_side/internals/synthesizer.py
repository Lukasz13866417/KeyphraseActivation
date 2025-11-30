from __future__ import annotations

import json
import os
import re
import sys
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import tempfile
import shutil

API_NAME = "kokoro"
DEFAULT_SAMPLE_RATE = 24000  # typical for many TTS engines; updated if we can detect


def _eprint(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _read_args() -> Dict[str, Any]:
    raw = os.environ.get("ARGS_JSON", "")
    if not raw:
        raise RuntimeError("ARGS_JSON not provided to kokoro internals")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid ARGS_JSON: {e}") from e


def _naturalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text, flags=re.MULTILINE)
    return text.strip()


_UNSUPPORTED_PUNCT = re.compile(r"[^A-Za-z0-9\s\.\,\!\?]+")


def normalize_text_for_kokoro(text: str) -> str:
    """
    Conservative normalization:
      - strip zero-width/tabs
      - collapse whitespace
      - normalize punctuation: (?!|!?) -> ?, collapse repeats, remove others
      - keep only . , ! ?
    """
    text = text.replace("\u200b", " ").replace("\t", " ")
    text = _naturalize_whitespace(text)
    # Map mixed terminal combos to '?'
    text = re.sub(r"(\?\!|\!\?)", "?", text)
    # Collapse repeated allowed punctuation
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\!{2,}", "!", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r",{2,}", ",", text)
    # Remove all but . , ! ?
    text = _UNSUPPORTED_PUNCT.sub(" ", text)
    text = _naturalize_whitespace(text)
    # Ensure sentence ends cleanly
    if text and text[-1] not in ".!?":
        text += "."
    return text


_ENGINE_CACHE = None
_VOICE_POOL_CACHE: Optional[List[str]] = None


def _default_models_dir() -> Path:
    return Path(__file__).resolve().parent / "models"


def _resolve_model_files(raw: Dict[str, Any]) -> tuple[Path, Path]:
    """
    Determine model and voices file paths from args or defaults.
    Defaults:
      models/kokoro-v1.0.onnx
      models/voices-v1.0.bin
    """
    model_path = raw.get("model_path")
    voices_path = raw.get("voices_path")
    models_dir = _default_models_dir()
    if not model_path:
        model_path = str(models_dir / "kokoro-v1.0.onnx")
    if not voices_path:
        voices_path = str(models_dir / "voices-v1.0.bin")
    return Path(model_path), Path(voices_path)


def _load_engine(model_path: Path, voices_path: Path):
    """
    Try to import and construct the Kokoro ONNX TTS engine.
    """
    global _ENGINE_CACHE
    if _ENGINE_CACHE is not None:
        return _ENGINE_CACHE

    # Validate files early with actionable guidance
    if not voices_path.exists() or not model_path.exists():
        missing = []
        if not voices_path.exists():
            missing.append(f"voices at {voices_path}")
        if not model_path.exists():
            missing.append(f"model at {model_path}")
        hint = (
            "Required Kokoro files are missing: "
            + ", ".join(missing)
            + "\nDownload into the internals models directory, e.g.:\n"
            + "  mkdir -p models\n"
            + "  wget -O models/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin\n"
            + "  wget -O models/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
        )
        raise FileNotFoundError(hint)

    try:
        from kokoro_onnx import Kokoro  # type: ignore
        engine = Kokoro(str(model_path), str(voices_path))
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize Kokoro ONNX TTS engine. "
            "Ensure 'kokoro-onnx' and 'onnxruntime' are installed in this internals env."
        ) from e

    _ENGINE_CACHE = engine
    return engine


@dataclass
class GenerationArgs:
    api_name: str
    text: str
    num_samples: int
    output_dir: str
    voice: Optional[str]
    normalize_punctuation: bool = True
    speed_jitter: float = 0.1
    p_no_trim: float = 0.15
    voice_pool: Optional[List[str]] = None


def _parse_generation_args(raw: Dict[str, Any]) -> GenerationArgs:
    return GenerationArgs(
        api_name=str(raw.get("api_name") or API_NAME),
        text=str(raw.get("text") or ""),
        num_samples=int(raw.get("num_samples") or 1),
        output_dir=str(raw.get("output_dir") or ""),
        voice=(raw.get("voice") or None),
        normalize_punctuation=bool(raw.get("normalize_punctuation", True)),
        speed_jitter=float(raw.get("speed_jitter", 0.1)),
        p_no_trim=float(raw.get("p_no_trim", 0.15)),
        voice_pool=list(raw.get("voice_pool") or []) or None,
    )


def _tts_to_file(engine, text: str, out_path: Path, voice: Optional[str]) -> None:
    """
    Attempt multiple likely APIs to synthesize to file.
    """
    # Prefer 'save' signature commonly used by some ONNX TTS wrappers
    if hasattr(engine, "save"):
        try:
            # Common forms: save(text, voice="name", file="out.wav") or save(text, speaker="name", path="out.wav")
            try:
                engine.save(text=text, voice=voice, file=str(out_path))
                return
            except TypeError:
                engine.save(text=text, speaker=voice, path=str(out_path))
                return
        except Exception as e:
            _eprint(f"[kokoro internals] engine.save failed: {e}")
    # Fall back to 'tts_to_file'
    if hasattr(engine, "tts_to_file"):
        try:
            # Common forms: tts_to_file(text, speaker_id/voice, output_path)
            engine.tts_to_file(text, voice, str(out_path))  # type: ignore[arg-type]
            return
        except Exception as e:
            _eprint(f"[kokoro internals] engine.tts_to_file failed: {e}")
    # Last resort: synthesize to ndarray and write via soundfile
    if hasattr(engine, "synthesize"):
        try:
            wav, sr = engine.synthesize(text, voice)  # type: ignore[call-arg]
            import soundfile as sf  # local import to avoid import if not needed

            sf.write(str(out_path), wav, int(sr))
            return
        except Exception as e:
            _eprint(f"[kokoro internals] engine.synthesize failed: {e}")
    raise RuntimeError("No supported synthesis API found on Kokoro engine (save/tts_to_file/synthesize).")


def _get_voice_pool(engine, explicit_pool: Optional[List[str]]) -> List[str]:
    global _VOICE_POOL_CACHE
    if _VOICE_POOL_CACHE is not None:
        return _VOICE_POOL_CACHE
    try:
        available = set(engine.get_voices() if hasattr(engine, "get_voices") else [])
    except Exception:
        available = set()
    pool: List[str] = []
    if explicit_pool:
        pool = [v for v in explicit_pool if v in available] if available else explicit_pool
    if not pool:
        # Fallback to available voices
        pool = sorted(list(available)) if available else []
    # Final fallback to a generic token if pool is still empty
    if not pool:
        pool = ["af"]
    _VOICE_POOL_CACHE = pool
    return pool


def generate(raw_args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate audio files using Kokoro TTS.
    Returns list of dicts: { path, sample_rate, duration_sec, model_name, api_name, text, text_normalized }
    """
    args = _parse_generation_args(raw_args)
    if not args.text:
        raise ValueError("Empty text provided")
    if not args.output_dir:
        raise ValueError("Missing output_dir")

    text_norm = normalize_text_for_kokoro(args.text) if args.normalize_punctuation else _naturalize_whitespace(args.text)
    model_path, voices_path = _resolve_model_files(raw_args)
    engine = _load_engine(model_path, voices_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    ts_base = int(time.time() * 1000)

    voice_pool = _get_voice_pool(engine, args.voice_pool)
    jitter = max(0.0, float(args.speed_jitter))
    p_no_trim = min(1.0, max(0.0, float(args.p_no_trim)))

    for i in range(args.num_samples):
        # Choose voice randomly from pool, or fallback
        try:
            chosen_voice = args.voice or random.choice(voice_pool)
        except Exception:
            chosen_voice = args.voice or "af"

        fname = f"kokoro_{chosen_voice.replace('/', '-').replace(' ', '_')}_{ts_base}_{i:04d}.wav"
        out_path = out_dir / fname

        # Per-sample params
        speed = 1.0 + random.uniform(-jitter, jitter)
        speed = max(0.5, min(2.0, speed))
        trim_flag = not (random.random() < p_no_trim)
        # Synthesize via Kokoro API; fall back to generic writers on failure
        sample_rate_detected = None
        try:
            audio, sr = engine.create(text_norm, chosen_voice, speed=speed, lang="en-us", is_phonemes=False, trim=trim_flag)
            import soundfile as sf
            sf.write(str(out_path), audio, int(sr))
            sample_rate_detected = int(sr)
        except Exception as e:
            _eprint(f"[kokoro internals] engine.create failed, falling back: {e}")
            _tts_to_file(engine, text_norm, out_path, chosen_voice)

        # Post-process for punctuation-driven prosody
        try:
            orig = args.text.strip()
            m = re.search(r"([!\?\.,]+)\s*$", orig)
            eff = None
            if m:
                tail = m.group(1)
                if tail.endswith("?") or tail.endswith("?!") or tail.endswith("!?"):
                    eff = "?"
                elif tail.endswith("!"):
                    eff = "!"
                elif tail.endswith("."):
                    eff = "."
                elif tail.endswith(","):
                    eff = ","

            if eff == "?":
                from prosody import add_question_intonation
                with tempfile.TemporaryDirectory() as td:
                    temp_out = Path(td) / "prosody_tmp.wav"
                    add_question_intonation(
                        input_path=str(out_path),
                        output_path=str(temp_out),
                        rise_fraction=0.5,
                        rise_semitones=8.0,
                        trim_silence=True,
                    )
                    shutil.move(str(temp_out), str(out_path))
                # _eprint("[kokoro internals] Applied question intonation")
            elif eff == "!":
                from prosody import add_exclamation_emphasis
                with tempfile.TemporaryDirectory() as td:
                    temp_out = Path(td) / "prosody_tmp.wav"
                    add_exclamation_emphasis(
                        input_path=str(out_path),
                        output_path=str(temp_out),
                        rise_fraction=0.25,
                        rise_semitones=3.0,
                        emphasis_gain=1.15,
                        trim_silence=True,
                    )
                    shutil.move(str(temp_out), str(out_path))
                # _eprint("[kokoro internals] Applied exclamation emphasis")
            elif eff == ".":
                from prosody import add_period_cadence
                with tempfile.TemporaryDirectory() as td:
                    temp_out = Path(td) / "prosody_tmp.wav"
                    add_period_cadence(
                        input_path=str(out_path),
                        output_path=str(temp_out),
                        fall_fraction=0.25,
                        fall_semitones=3.0,
                        extra_silence_ms=150.0,
                        trim_silence=True,
                    )
                    shutil.move(str(temp_out), str(out_path))
                # _eprint("[kokoro internals] Applied period cadence")
            elif eff == ",":
                from prosody import add_comma_pause
                with tempfile.TemporaryDirectory() as td:
                    temp_out = Path(td) / "prosody_tmp.wav"
                    add_comma_pause(
                        input_path=str(out_path),
                        output_path=str(temp_out),
                        fall_fraction=0.15,
                        fall_semitones=2.0,
                        extra_silence_ms=80.0,
                        trim_silence=True,
                    )
                    shutil.move(str(temp_out), str(out_path))
                # _eprint("[kokoro internals] Applied comma pause")
        except Exception as e:
            _eprint(f"[kokoro internals] prosody step failed: {e}")

        # Probe audio info
        sample_rate = sample_rate_detected or DEFAULT_SAMPLE_RATE
        duration_sec = None
        try:
            import soundfile as sf

            info = sf.info(str(out_path))
            sample_rate = int(info.samplerate)
            if info.frames and info.samplerate:
                duration_sec = float(info.frames) / float(info.samplerate)
        except Exception as e:
            _eprint(f"[kokoro internals] soundfile info failed for {out_path}: {e}")
            try:
                import wave

                with wave.open(str(out_path), "rb") as wf:
                    sample_rate = wf.getframerate()
                    nframes = wf.getnframes()
                    duration_sec = float(nframes) / float(sample_rate)
            except Exception as e2:
                _eprint(f"[kokoro internals] wave probe failed for {out_path}: {e2}")

        model_name = "kokoro-onnx"
        model_voice = args.voice or chosen_voice
        if model_voice:
            model_name = f"{model_name}:{model_voice}"

        _eprint(f"[ok] {fname} | voice={chosen_voice} speed={speed:.2f} trim={trim_flag} dur={duration_sec:.2f}s")

        records.append(
            {
                "path": str(out_path.resolve()),
                "sample_rate": int(sample_rate),
                "duration_sec": float(duration_sec) if duration_sec is not None else None,
                "model_name": model_name,
                "api_name": API_NAME,
                "text": args.text,
                "text_normalized": text_norm,
                "voice_used": model_voice,
                "speed": speed,
                "trim": trim_flag,
            }
        )

    return records


def _write_or_print(records: List[Dict[str, Any]]) -> None:
    payload = json.dumps(records, ensure_ascii=False)
    out_file = os.environ.get("RETURN_PATHS_FILE")
    if out_file:
        Path(out_file).write_text(payload, encoding="utf-8")
    else:
        print(payload)
        sys.stdout.flush()


def main() -> None:
    try:
        raw_args = _read_args()
        res = generate(raw_args)
        _write_or_print(res)
    except Exception as e:
        _eprint(f"[kokoro internals] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


