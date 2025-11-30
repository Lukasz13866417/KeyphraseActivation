# Phrase recognition service Django MLaaS

### User enters phrase. The app then trains a lightweight AI model to detect the phrase in audio. Once the training is finished, the user can download the model and run inference locally due to the small size.
![screenshot.png](How the django app looks)
# The model

### Setup

- Make sql database for reusable audio samples:
  
  - `sudo apt install sqlite3`
  
  - `sqlite3 db/db.sqlite3 < db/db.sql`

- Useful but not necessary: the project uses ElevenLabs TTS to make better datasets,
  so to use this (paid) TTS, u need to specify API key:
  
  - `cd elevenlabs_side/internals`
  
  - `nano .env`
  
  - There, write `ELEVENLABS_API_KEY=<your key>`


