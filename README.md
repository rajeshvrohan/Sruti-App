# Sruti: A Carnatic Music Analyzer

A Flask web app that analyzes a recording, detects the swaras present, ranks
candidate Carnatic raagas, and generates a short educational analysis of the
top match with the Anthropic Claude API.

Live at [sruti.io](https://sruti.io).

## What it does

- Accepts a WAV, MP3, M4A, or MP4 upload (up to 1 GB, up to 30 minutes long).
- Extracts pitch with librosa, folds it against a user-supplied or
  auto-detected tonic (Sa), and produces the set of swaras heard.
- Matches those swaras against a database of common Carnatic raagas
  (`raaga_database.py`) and returns the best candidates by Jaccard overlap.
- Sends the top raaga name to Claude for a short educational write-up.
- Deletes the upload immediately after analysis, whether it succeeds or fails.

## Running locally

Requires Python 3 and `ffmpeg` on the PATH (bundled `imageio-ffmpeg` also
works). Redis is optional; without it, rate limiting falls back to an
in-memory store.

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY
python app.py          # dev server on :5000
```

Production is served by gunicorn (see `Procfile` and `gunicorn.conf.py`):

```
gunicorn app:app
```

## CLI

`identify_raaga.py` runs the same pipeline against a local file without the
web layer:

```
python identify_raaga.py path/to/recording.wav
```

## Layout

- `app.py` — Flask routes, upload validation, rate limiting.
- `audio_processor.py` — pitch extraction and swara detection (numba-accelerated).
- `raaga_database.py`, `swara_database.py` — scale definitions and matching.
- `claude_client.py` — Anthropic API wrapper for the educational analysis.
- `templates/`, `static/` — the web UI.
- `render.yaml` — Render.com service definition.

## Environment

- `ANTHROPIC_API_KEY` — required, for the raaga analysis.
- `FLASK_KEY` — Flask secret key.
- `RATELIMIT_STORAGE_URI` — Redis URL for shared rate-limit counters
  (default `redis://localhost:6379`).
- `PORT` — bind port, set by the host.
- `RENDER_EXTERNAL_URL` — set by Render; used to route uploads past the
  Cloudflare 100 MB body cap on sruti.io.

## Disclaimer

The analysis is for educational purposes only. It is not medical or
therapeutic advice. Only upload recordings you own or have the rights to.

## License

Released under the [MIT License](LICENSE).

## Credits

`Local Forecast - Elevator.mp3` — "Local Forecast - Elevator" by Kevin
MacLeod ([incompetech.com](https://incompetech.com)), licensed under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
Used as a sample input for testing.
