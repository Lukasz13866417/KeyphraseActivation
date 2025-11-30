import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Tuple, Optional

import soundfile as sf


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _internals_dir() -> Path:
    return Path(__file__).resolve().parent / "internal"


def _internals_venv_python() -> Path:
    return _internals_dir() / ".venv" / "bin" / "python"


def find_keyphrase(
    audio_path: str,
    transcript: str,
    key_phrase: str,
    *,
    check: bool = True,
    capture_output: bool = True,
) -> List[Tuple[float, float]]:
    """
    Run the keyphrase finder in an isolated env (venv-first, fallback to uv run).
    Returns a list of (start_sec, end_sec) tuples.
    """
    internals_cwd = _internals_dir()
    if not internals_cwd.is_dir():
        raise FileNotFoundError(f"internals directory not found: {internals_cwd}")

    audio_abs = str(Path(audio_path).resolve())
    args_payload = {
        "audio_path": audio_abs,
        "transcript": transcript,
        "key_phrase": key_phrase,
    }

    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["SYNTH_ARGS"] = json.dumps(args_payload)
    env.setdefault("PYTHONUNBUFFERED", "1")

    program = (
        "import os,json; a=json.loads(os.environ['SYNTH_ARGS']); "
        "import finder as f; "
        "res=f.find_keyphrase_timestamps(a['audio_path'], a['transcript'], a['key_phrase']); "
        "print(json.dumps([res]), flush=True)"
    )

    venv_python = _internals_venv_python()
    if venv_python.exists():
        cmd = [str(venv_python), "-c", program]
    else:
        uv_exec = shutil.which("uv")
        if uv_exec is None:
            raise FileNotFoundError("Neither internals venv python nor 'uv' found.")
        cmd = [uv_exec, "run", "python", "-c", program]

    try:
        # Stream child's stdout and stderr separately so debug is visible immediately
        proc = subprocess.Popen(
        cmd,
        cwd=str(internals_cwd),
        env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        text=True,
            bufsize=1,
        )
        lines: List[str] = []
        assert proc.stdout is not None
        assert proc.stderr is not None
        
        # Read both streams concurrently
        stderr_lines: List[str] = []
        
        def read_stderr():
            for line in proc.stderr:
                s = line.rstrip("\n")
                if s:
                    stderr_lines.append(s)
                    print(s, file=sys.stderr, flush=True)
        
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        
        for line in proc.stdout:
            s = line.rstrip("\n")
            if s:
                lines.append(s)
                print(s, flush=True)
        
        proc.wait()
        stderr_thread.join(timeout=1.0)
        
        matches_json = None
        for ln in reversed(lines):
            if ln.startswith("[") and ln.endswith("]"):
                matches_json = ln
                break
        if matches_json:
            try:
                return json.loads(matches_json)
    except Exception:
                pass
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        return []
    except subprocess.CalledProcessError as e:
        # Surface child stderr/stdout for debugging
        if e.stdout:
            print(e.stdout, flush=True)
        if e.stderr:
            print(e.stderr, file=sys.stderr, flush=True)
        raise


def extract_keyphrase_audio(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
) -> None:
    """
    Extract a segment from an audio file and save it as a new WAV file.
    
    Args:
        audio_path: Path to input audio file
        start_sec: Start time in seconds
        end_sec: End time in seconds
        output_path: Path to save the extracted segment
    """
    audio, sample_rate = sf.read(audio_path)
    
    # Convert seconds to sample indices
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    
    # Ensure indices are within bounds
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    
    # Extract segment
    segment = audio[start_sample:end_sample]
    
    # Save to output file
    sf.write(output_path, segment, sample_rate)
    print(f"Extracted keyphrase segment saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        audio, trx, key = sys.argv[1:4]
    else:
        audio = input("Path to audio WAV: ").strip()
        trx = input("Transcript: ").strip()
        key = input("Key phrase: ").strip()
    spans = find_keyphrase(audio, trx, key)
    print(json.dumps(spans, indent=2))
    
    # Prompt user to extract keyphrase segment
    if spans:
        start_sec, end_sec = spans[0]
        response = input("\nExtract keyphrase segment to a new WAV file? (y/n): ").strip().lower()
        if response == 'y' or response == 'yes':
            # Generate output filename
            audio_path = Path(audio)
            keyphrase_safe = re.sub(r'[^\w\s-]', '', key).strip().replace(' ', '_')
            output_path = audio_path.parent / f"{audio_path.stem}_keyphrase_{keyphrase_safe}.wav"
            
            extract_keyphrase_audio(str(audio_path), start_sec, end_sec, str(output_path))


