PRAGMA journal_mode=WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS audio_sample (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  api_name TEXT NOT NULL,
  model_name TEXT NOT NULL,
  text_original TEXT NOT NULL,
    text_normalized TEXT NOT NULL,
  audio_sha256 TEXT NOT NULL,
  duration_sec REAL,
  sample_rate INTEGER,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_phrase_voice
  ON audio_sample (text_normalized, api_name, model_name);

CREATE UNIQUE INDEX IF NOT EXISTS uq_audio_sha256
  ON audio_sample (audio_sha256); 

