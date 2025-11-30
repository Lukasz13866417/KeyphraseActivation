ElevenLabs TTS internals

Setup:
- Set the environment variable ELEVENLABS_API_KEY (or XI_API_KEY).
- Create a venv here (or use uv) and install dependencies per pyproject.toml.

Usage:
- From the project root, run the driver: `python ../driver.py` (from this folder) or `python elevenlabs_side/driver.py`.
- The synthesizer will pick the first available voice by default or use the provided voice IDs.

Notes:
- Outputs are written as `<api_name>_<model>_<id>.wav` into the repo `samples/` folder by default.
- To target a specific voice, pass its voice ID via `models=[\"<voice_id>\"]` in the driver call.


