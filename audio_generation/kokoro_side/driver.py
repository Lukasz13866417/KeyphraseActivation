from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
from typing import Any, Dict, List, Optional


API_NAME = "kokoro"


def _project_root() -> Path:
    # repo root: .../KeyphraseActivation
    return Path(__file__).resolve().parents[2]


def _stream_pipe(prefix: str, pipe, to_stderr: bool = False) -> None:
    target = sys.stderr if to_stderr else sys.stdout
    for raw in iter(pipe.readline, b""):
        try:
            line = raw.decode("utf-8", errors="replace")
        except Exception:
            line = raw.decode("utf-8", errors="replace")
        target.write(f"{prefix}{line}")
        target.flush()


def run_internals_synthesizer(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Launch the isolated uv environment in internals/ and run synthesizer.py.
    Streams logs live and reads the final JSON result from a temp file.
    """
    internals_dir = _project_root() / "audio_generation" / "kokoro_side" / "internals"
    if not internals_dir.exists():
        raise FileNotFoundError(f"Internals directory not found: {internals_dir}")

    # Create dir if nonexistent
    out_dir = args.get("output_dir")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(prefix="kokoro_return_", suffix=".json", delete=False) as tf:
        return_file = Path(tf.name)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["ARGS_JSON"] = json.dumps(args, ensure_ascii=False)
    env["RETURN_PATHS_FILE"] = str(return_file)

    cmd = ["uv", "run", "python", "-u", "synthesizer.py"]
    proc = Popen(cmd, cwd=str(internals_dir), env=env, stdout=PIPE, stderr=STDOUT, bufsize=1)

    t = threading.Thread(target=_stream_pipe, args=("[kokoro] ", proc.stdout))
    t.daemon = True
    t.start()
    code = proc.wait()
    t.join(timeout=1.0)

    if code != 0:
        # Attempt to read error details if any were printed already
        raise RuntimeError(f"kokoro internals exited with code {code}")

    if not return_file.exists():
        raise FileNotFoundError(f"No result file produced by internals: {return_file}")

    raw = return_file.read_text(encoding="utf-8")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by kokoro internals at {return_file}: {e}\nRaw:\n{raw}") from e
    finally:
        try:
            return_file.unlink(missing_ok=True)
        except Exception:
            pass

    if not isinstance(result, list):
        raise TypeError(f"kokoro internals returned non-list result: {type(result)}")
    return result


def synthesize(
    text: str,
    *,
    num_samples: int = 1,
    voice: Optional[str] = None,
    output_dir: Optional[Path] = None,
    normalize_punctuation: bool = True,
    voice_pool: Optional[List[str]] = None,
    speed_jitter: float = 0.1,
    p_no_trim: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Public entry point: generate one or more WAVs via kokoro internals.
    Returns a list of JSON-able dict records:
      { path, sample_rate, duration_sec, model_name, api_name, text, text_normalized }
    """
    repo_root = _project_root()
    out_dir = str(output_dir or (repo_root / "samples"))
    args: Dict[str, Any] = {
        "api_name": API_NAME,
        "text": text,
        "num_samples": int(num_samples),
        "voice": voice,
        "output_dir": out_dir,
        "normalize_punctuation": bool(normalize_punctuation),
        "voice_pool": list(voice_pool or []),
        "speed_jitter": float(speed_jitter),
        "p_no_trim": float(p_no_trim),
    }
    return run_internals_synthesizer(args)


def clear_samples(remove_from_db: bool = True) -> int:
    """
    Remove kokoro-generated WAV files from project_root/samples and optionally
    delete DB rows for api_name='kokoro'.
    Returns number of files removed.
    """
    repo_root = _project_root()
    samples_dir = repo_root / "samples"
    removed = 0
    if samples_dir.exists():
        for p in samples_dir.glob("kokoro_*.wav"):
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass

    if remove_from_db:
        try:
            # best-effort DB cleanup by api_name
            from db import db_api
            try:
                db_api.init_db()
            except Warning:
                pass
            with db_api.get_db_connection() as conn:
                conn.execute("DELETE FROM audio_sample WHERE api_name = ?", (API_NAME,))
                conn.commit()
        except Exception:
            # DB may not be initialized/available
            pass
    return removed


def main() -> None:
    """
    Simple test for the driver: runs one sample and prints the JSON list.
    """
    print("[kokoro driver] Running test...")
    phrase = "Hello from Kokoro! Is this a test?"
    try:
        # With punctuation (default normalization: True)
        records_with = synthesize(phrase, num_samples=1, voice=None, normalize_punctuation=True)
        print("[kokoro driver] With punctuation:")
        print(json.dumps(records_with, indent=2))

        # Without punctuation (strip punctuation characters, disable normalization)
        phrase_no_punct = "".join(ch for ch in phrase if ch.isalnum() or ch.isspace())
        records_without = synthesize(phrase_no_punct, num_samples=1, voice=None, normalize_punctuation=False)
        print("[kokoro driver] Without punctuation:")
        print(json.dumps(records_without, indent=2))
    except Exception as e:
        print(f"[kokoro driver] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


