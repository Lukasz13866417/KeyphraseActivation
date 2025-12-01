"""Driver for the Piper TTS engine. Made by AI.
TODO: review this
"""
import json
import os
import shutil
import subprocess
import sys
import sqlite3
import tempfile
from pathlib import Path
from typing import List, Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _internals_dir() -> Path:
    return Path(__file__).resolve().parent / "internals"


def _internals_venv_python() -> Path:
    """Return the internals' venv python if it exists (uv-managed .venv)."""
    return _internals_dir() / ".venv" / "bin" / "python"


def _read_records_file(path: str) -> List[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                    return data
    except Exception as exc:
        print(f"Error reading records file: {exc}")
    return []


def run_internals_synthesizer(
    phrase: str,
    num_samples: int,
    *,
    output_dir: Optional[str] = None,
    models: Optional[List[str]] = None,
    num_models: Optional[int] = None,
    models_dir: str = "models",
    quiet: bool = False,
    length_scale: float = 1.0,
    noise_scale: float = 0.667,
    noise_w_scale: float = 0.333,
    jitter: float = 0.0,
    check: bool = True,
    capture_output: bool = True,
) -> List[dict]:
    """Launch the internals synthesizer using the internals venv if available.

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
        "models_dir": models_dir,
        "quiet": bool(quiet),
        "length_scale": float(length_scale),
        "noise_scale": float(noise_scale),
        "noise_w_scale": float(noise_w_scale),
        "jitter": float(jitter),
        "api_name": "piper",
    }

    env = os.environ.copy()
    # Avoid leaking the parent venv into the child; let internals control its env
    env.pop("VIRTUAL_ENV", None)
    env["SYNTH_ARGS"] = json.dumps(args_payload)
    with tempfile.NamedTemporaryFile(prefix="piper_records_", suffix=".json", delete=False) as tf:
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
    proc = subprocess.Popen(
        cmd,
        cwd=str(internals_cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line.rstrip("\n"), flush=True)
    finally:
        proc.wait()
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
    length_scale: float = 1.05,
    noise_scale: float = 0.7,
    noise_w_scale: float = 0.6,
    jitter: float = 0.2,
) -> List[dict]:
    return run_internals_synthesizer(
        phrase=phrase,
        num_samples=num_samples,
        output_dir=output_dir,
        models=models,
        num_models=num_models,
        models_dir="models",
        quiet=quiet,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
        jitter=jitter,
    )


if __name__ == "__main__":
    paths = synthesize(
        phrase="AI money machine",
        num_samples=3,
        output_dir=str((_project_root() / "samples").resolve()),
        quiet=False,
        num_models=2,
        length_scale=1.05,
        noise_scale=0.9,
        noise_w_scale=0.5,
        jitter=0.25,
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
    api_name = "piper"
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