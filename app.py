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

# Reject recordings longer than this before running the (expensive) pitch
# analysis, so nobody can tie up the server by uploading hours of audio.
MAX_AUDIO_DURATION_SECONDS = 30 * 60  # 30 minutes

# Rate limit uploads per client IP address. Keyed on the remote address so each
# IP gets its own budget for the /analyze route. A shared Redis backend keeps the
# counters consistent across multiple worker processes; set RATELIMIT_STORAGE_URI
# (e.g. redis://localhost:6379) to point at your instance.
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri=os.environ.get("RATELIMIT_STORAGE_URI", "redis://localhost:6379"),
    # If Redis is unreachable, degrade to per-process in-memory limiting instead
    # of failing the request, so uploads keep working (locally, or if Redis blips).
    in_memory_fallback_enabled=True,
)

processor = AudioProcessor()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Number of header bytes to inspect. The furthest signature we check is the
# ISO-BMFF "ftyp" box, which sits at offset 4, so 16 bytes is comfortably enough.
_MAGIC_HEADER_BYTES = 16


def has_audio_signature(header):
    """Return True if `header` starts with a known WAV/MP3/M4A/MP4 signature.

    Guards against someone renaming e.g. a .exe to .mp3: the extension check
    alone trusts the client-supplied name, whereas this inspects the actual
    leading bytes of the file's content.
    """
    # WAV: RIFF container tagged as WAVE -> "RIFF....WAVE"
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return True

    # MP3: either an ID3v2 tag ("ID3") or a raw MPEG audio frame. A frame begins
    # with an 11-bit sync word: 0xFF followed by a byte whose top 3 bits are set.
    if header[:3] == b"ID3":
        return True
    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return True

    # M4A / MP4: ISO Base Media File Format. The first box is "ftyp" at offset 4
    # (the preceding 4 bytes are the box size).
    if header[4:8] == b"ftyp":
        return True

    return False


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(exc):
    """Return a clear JSON error when an upload exceeds MAX_CONTENT_LENGTH."""
    limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return jsonify(
        {"error": f"File is too large. The maximum upload size is {limit_mb} MB (1 GB)."}
    ), 413


@app.errorhandler(RateLimitExceeded)
def handle_rate_limit(exc):
    """Return a clear JSON message when a client exceeds the upload rate limit."""
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
    """Return JSON (never an HTML error page) for HTTP errors such as 404/405."""
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

    # The extension is client-supplied and easily spoofed (e.g. a .exe renamed to
    # .mp3), so verify the actual file content matches a real audio signature
    # before anything is written to disk or handed to the processor.
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

    # The uploaded file is removed in the finally block, so the uploads folder is
    # cleaned up whether analysis succeeds or fails. For MP4 inputs, AudioProcessor
    # demuxes the audio to its own temp file and cleans that up internally.
    try:
        # Reject over-long recordings up front, reading only file metadata, so we
        # never kick off the heavy pitch analysis on hours of audio.
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
            # ffmpeg ran but could not extract an audio stream (e.g. no audio track).
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
