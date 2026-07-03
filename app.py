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
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB limit

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


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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
        {"error": f"File is too large. The maximum upload size is {limit_mb} MB (1 GB)."}
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


@app.route("/")
def index():
    return render_template("index.html")


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

        try:
            result = processor.extract_swaras(filepath)
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
            }
        )
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass


if __name__ == "__main__":
    app.run(debug=True)
