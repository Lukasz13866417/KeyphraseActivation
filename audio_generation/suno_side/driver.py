"""Driver for the Suno TTS engine. Made by AI.
TODO: review this
"""
import json
import os
import shutil
import subprocess
import sys
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import List, Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _internals_dir() -> Path:
    return Path(__file__).resolve().parent / "internals"


def _internals_venv_python() -> Path:
    return _internals_dir() / ".venv" / "bin" / "python"


def _mem_available_bytes() -> Optional[int]:
    """Parse /proc/meminfo for MemAvailable."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = int(parts[1])
                        return kb * 1024
    except OSError:
        return None
    except ValueError:
        return None
    return None


def _assert_memory_budget(num_samples: int, num_models: Optional[int]) -> None:
    """Best-effort guard rail against Bark OOMs."""
    if os.environ.get("BARK_SKIP_MEMCHECK") == "1":
        return
    available = _mem_available_bytes()
    if available is None:
        return
    # Empirical: Bark needs ~2.0 GB baseline (safety margin) + 0.8 GB per active voice in RAM.
    active_voices = max(1, num_models or 1)
    required = int(2.0 * (1024**3) + active_voices * 0.8 * (1024**3))
    if available < required:
        raise RuntimeError(
            "Insufficient free memory for Bark synthesis "
            f"(have ~{available/1024**3:.1f} GB, need ~{required/1024**3:.1f} GB). "
            "Lower the Bark sample count or voices, or add swap, or set BARK_SKIP_MEMCHECK=1 to override."
        )


def _stream_process_output(proc: subprocess.Popen) -> threading.Thread:
    def _reader():
        if proc.stdout is None:
            return
        for line in proc.stdout:
            print(line.rstrip("\n"), flush=True)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return thread


def _read_records_file(path: str) -> List[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def run_internals_synthesizer(
    phrase: str,
    num_samples: int,
    *,
    output_dir: Optional[str] = None,
    models: Optional[List[str]] = None,
    num_models: Optional[int] = None,
    quiet: bool = False,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    jitter: float = 0.1,
    check: bool = True,
    capture_output: bool = True,
    timeout_seconds: int = 900,
) -> List[dict]:
    """Launch the Bark internals synthesizer using the internals venv if available.

    Preference order:
    1) Use internals/.venv/bin/python directly (most robust).
    2) Fallback to `uv run` in the internals directory if venv python is missing.
    """
    internals_cwd = _internals_dir()
    if not internals_cwd.is_dir():
        raise FileNotFoundError(f"internals directory not found: {internals_cwd}")

    default_output = (_project_root() / "samples").resolve()
    args_payload = {
        "phrase": phrase,
        "num_samples": int(num_samples),
        "output_dir": str(Path(output_dir).resolve()) if output_dir else str(default_output),
        "models": models,
        "num_models": num_models,
        "quiet": bool(quiet),
        "text_temp": float(text_temp),
        "waveform_temp": float(waveform_temp),
        "jitter": float(jitter),
        "api_name": "bark",
    }

    _assert_memory_budget(num_samples, num_models)

    env = os.environ.copy()
    # Avoid leaking the parent venv into the child; let internals control its env
    env.pop("VIRTUAL_ENV", None)
    # Force CPU for Bark unless the caller intentionally overrides
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    env.setdefault("BARK_FORCE_CPU", "1")
    env.setdefault("BARK_DISABLE_LRU_CACHE", "1")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["SYNTH_ARGS"] = json.dumps(args_payload)
    # Use a temp file for results to allow live log streaming
    with tempfile.NamedTemporaryFile(prefix="bark_records_", suffix=".json", delete=False) as tf:
        results_path = tf.name
    env["RETURN_PATHS_FILE"] = results_path
    env["PYTHONUNBUFFERED"] = "1"

    program = (
        "import os,json; a=json.loads(os.environ['SYNTH_ARGS']); "
        "import synthesizer as s; s.generate(**a)"
    )

    venv_python = _internals_venv_python()
    if venv_python.exists():
        cmd = [str(venv_python), "-c", program]
    else:
        uv_exec = shutil.which("uv")
        if uv_exec is None:
            raise FileNotFoundError(
                "Neither internals venv python nor 'uv' found. Set up internals or install uv."
            )
        cmd = [uv_exec, "run", "python", "-c", program]

    # Stream logs live with watchdog
    proc = subprocess.Popen(
        cmd,
        cwd=str(internals_cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    reader_thread = _stream_process_output(proc)
    try:
        proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        reader_thread.join(timeout=1)
        raise TimeoutError(
            f"Bark synthesis exceeded timeout ({timeout_seconds}s). "
            "Try fewer samples or check Bark internals for hangs."
        )
    reader_thread.join(timeout=1)
    # Read records
    records = _read_records_file(results_path)
    try:
        os.unlink(results_path)
    except Exception:
        pass
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return records


def synthesize(
    phrase: str,
    num_samples: int = 3,
    *,
    output_dir: Optional[str] = None,
    models: Optional[List[str]] = None,
    num_models: Optional[int] = 2,
    quiet: bool = False,
    text_temp: float = 0.5,
    waveform_temp: float = 0.5,
    jitter: float = 0.0,
) -> List[dict]:
    return run_internals_synthesizer(
        phrase=phrase,
        num_samples=num_samples,
        output_dir=output_dir,
        models=models,
        num_models=num_models,
        quiet=quiet,
        text_temp=text_temp,
        waveform_temp=waveform_temp,
        jitter=jitter,
    )


if __name__ == "__main__":
    paths = synthesize(
        phrase="AI money machine",
        num_samples=3,
        output_dir=str((_project_root() / "samples").resolve()),
        quiet=False,
        num_models=2,
        text_temp=0.4,
        waveform_temp=0.4,
        jitter=0.0,
    )
    print(json.dumps(paths), flush=True)
    sys.exit(0)


def clear_samples(
    *,
    output_dir: Optional[str] = None,
    remove_files: bool = True,
) -> tuple[int, int]:
    """
    Delete this driver's generated samples from disk and DB.
    Returns (files_removed, db_rows_deleted).
    """
    api_name = "bark"
    samples_dir = Path(output_dir).resolve() if output_dir else (_project_root() / "samples").resolve()
    files_removed = 0
    if remove_files and samples_dir.is_dir():
        for p in samples_dir.glob(f"{api_name}_*.wav"):
            try:
                p.unlink()
                files_removed += 1
            except Exception:
                pass
    # delete db rows for this api under this samples dir prefix
    db_rows = 0
    try:
        db_path = (_project_root() / "db" / "db.sqlite3").resolve()
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            try:
                prefix = str(samples_dir) + os.sep
                cur = conn.execute(
                    "DELETE FROM audio_sample WHERE api_name=? AND path LIKE ?",
                    (api_name, prefix + "%"),
                )
                db_rows = cur.rowcount if cur.rowcount is not None else 0
                conn.commit()
            finally:
                conn.close()
    except Exception:
        pass
    return files_removed, db_rows


