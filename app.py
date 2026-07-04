# Test the production (gunicorn) setup locally:
#     gunicorn app:app --bind 0.0.0.0:5000
# To confirm the PORT env var is read, bind to it explicitly, e.g.:
#     PORT=8080 gunicorn app:app --bind 0.0.0.0:$PORT   # serves on :8080
# (Under `python app.py` the __main__ block reads PORT itself; see the bottom.)

import os
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.errors import RateLimitExceeded
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from dotenv import load_dotenv

from audio_processor import AudioProcessor
from claude_client import get_raaga_analysis

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
# Cloudflare (fronting sruti.io) rejects request bodies over 100 MB, so the
# page uploads straight to the Render host (RENDER_EXTERNAL_URL) instead,
# where this is the only limit. Keep it in lockstep with the client-side
# check in index.html.
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB limit

# Origins allowed to make the cross-origin upload described above.
ALLOWED_UPLOAD_ORIGINS = {"https://sruti.io", "https://www.sruti.io"}

ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "mp4"}

# Max clip length; longer uploads are rejected before the costly pitch analysis.
MAX_AUDIO_DURATION_SECONDS = 30 * 60  # 30 minutes

# Per-IP rate limiting for /analyze. Shared Redis keeps counts consistent across
# workers; set RATELIMIT_STORAGE_URI to point at it.
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri=os.environ.get("RATELIMIT_STORAGE_URI", "redis://localhost:6379"),
    in_memory_fallback_enabled=True,  # keep serving if Redis is down
)

processor = AudioProcessor()

# Compile the numba-backed analysis path at import (worker boot) so no request
# pays the cold-compile cost. On Render's slow CPU that compile can exceed the
# gunicorn timeout and get the worker killed mid-compile, corrupting numba's
# cache and failing later requests with "no compiled object yet". Best-effort:
# if it fails, boot proceeds and the path compiles lazily on first request.
try:
    processor.warmup()
    print("Numba analysis warmup complete.", flush=True)
except Exception as exc:
    print(f"Numba warmup failed (will compile lazily): {exc}", flush=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Plausible tonic (Sa) range in Hz: roughly the lowest bass to a high soprano.
MIN_TONIC_HZ = 60.0
MAX_TONIC_HZ = 600.0

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

def parse_tonic(raw):
    """Parse an optional user-supplied tonic (Sa) in Hz.

    Returns ``None`` when blank/absent (auto-detect). Raises ``ValueError`` with
    a user-facing message when the value is non-numeric or out of range.
    """
    if raw is None or str(raw).strip() == "":
        return None
    try:
        tonic = float(raw)
    except (TypeError, ValueError):
        raise ValueError("Tonic (Sa) must be a number in Hz, or left blank.")
    if not MIN_TONIC_HZ <= tonic <= MAX_TONIC_HZ:
        raise ValueError(
            f"Tonic (Sa) must be between {MIN_TONIC_HZ:.0f} and {MAX_TONIC_HZ:.0f} Hz."
        )
    return tonic


# Enough to cover the furthest signature: the ISO-BMFF "ftyp" box at offset 4.
_MAGIC_HEADER_BYTES = 16


def has_audio_signature(header):
    """True if `header` matches a WAV/MP3/M4A/MP4 signature (extensions are spoofable)."""
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":  # WAV
        return True
    if header[:3] == b"ID3":  # MP3 with ID3 tag
        return True
    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:  # MP3 frame sync
        return True
    if header[4:8] == b"ftyp":  # M4A / MP4 (ISO-BMFF)
        return True
    return False


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(exc):
    """JSON error for uploads over MAX_CONTENT_LENGTH."""
    limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return jsonify(
        {"error": f"File is too large. The maximum upload size is {limit_mb} MB."}
    ), 413


@app.errorhandler(RateLimitExceeded)
def handle_rate_limit(exc):
    """JSON error when a client exceeds the rate limit."""
    return jsonify(
        {
            "error": (
                "You're uploading too quickly. Please wait a little while before "
                "uploading again."
            ),
            "limit": exc.description,
        }
    ), 429


@app.errorhandler(HTTPException)
def handle_http_exception(exc):
    """JSON (not an HTML page) for HTTP errors like 404/405."""
    return jsonify({"error": exc.description}), exc.code


@app.after_request
def allow_cross_origin_uploads(response):
    """Let the sruti.io page read responses from direct-to-Render uploads.

    A multipart POST is a "simple" CORS request, so no preflight handling is
    needed — only this response header.
    """
    origin = request.headers.get("Origin")
    if origin in ALLOWED_UPLOAD_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.vary.add("Origin")
    return response


@app.route("/")
def index():
    # Absolute base URL for uploads, sidestepping Cloudflare's 100 MB body
    # cap on sruti.io. Render sets RENDER_EXTERNAL_URL; empty locally, which
    # leaves the page using the relative /analyze path.
    return render_template(
        "index.html",
        analyze_base=os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/"),
    )


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/terms")
def terms():
    return render_template("terms.html")


@app.route("/dmca")
def dmca():
    return render_template("dmca.html")


@app.route("/analyze", methods=["POST"])
@limiter.limit("5 per minute")
@limiter.limit("20 per hour")
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only WAV, MP3, M4A, and MP4 files are accepted"}), 400

    # Verify content by magic bytes before saving; extensions are spoofable.
    header = file.stream.read(_MAGIC_HEADER_BYTES)
    file.stream.seek(0)
    if not has_audio_signature(header):
        return jsonify(
            {"error": "File content is not a valid WAV, MP3, M4A, or MP4 audio file"}
        ), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        file.save(filepath)
    except OSError as exc:
        return jsonify({"error": f"Could not save the uploaded file: {exc}"}), 500

    print(f"Received upload, saved to: {os.path.abspath(filepath)}", flush=True)

    # finally removes the upload whether analysis succeeds or fails.
    try:
        # Reject over-long clips from metadata alone, before the heavy analysis.
        try:
            duration = processor.get_duration(filepath)
        except FileNotFoundError:
            return jsonify(
                {"error": "ffmpeg is required to process this file but was not found on the server."}
            ), 500
        except (ValueError, OSError, RuntimeError) as exc:
            return jsonify({"error": f"Could not read the audio file: {exc}"}), 422

        if duration > MAX_AUDIO_DURATION_SECONDS:
            limit_min = MAX_AUDIO_DURATION_SECONDS // 60
            return jsonify(
                {
                    "error": (
                        f"Audio is too long ({duration / 60:.1f} minutes). "
                        f"The maximum length is {limit_min} minutes."
                    )
                }
            ), 400

        # Optional user-supplied tonic (Sa) in Hz; blank means auto-detect.
        try:
            tonic_hz = parse_tonic(request.form.get("tonic"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            result = processor.extract_swaras(filepath, tonic_hz=tonic_hz)
        except FileNotFoundError:
            return jsonify(
                {"error": "ffmpeg is required to process MP4 files but was not found on the server."}
            ), 500
        except ValueError as exc:
            # ffmpeg ran but found no audio stream.
            return jsonify({"error": str(exc)}), 422
        except Exception as exc:
            return jsonify({"error": f"Could not analyze audio: {exc}"}), 500

        raagas = result["raagas"]
        analysis = None
        if raagas:
            top_raaga = raagas[0]
            try:
                analysis = get_raaga_analysis(top_raaga["name"])
            except Exception as exc:
                return jsonify({"error": f"Could not generate raaga analysis: {exc}"}), 502

        return jsonify(
            {
                "filename": filename,
                "swaras": result["swaras"],
                "raagas": raagas,
                "analysis": analysis,
                "tonic_hz": result.get("tonic_hz"),
                "tonic_source": "manual" if tonic_hz else "auto",
            }
        )
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
