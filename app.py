import os
from flask import Flask, request, jsonify, render_template
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

processor = AudioProcessor()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(exc):
    """Return a clear JSON error when an upload exceeds MAX_CONTENT_LENGTH."""
    limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return jsonify(
        {"error": f"File is too large. The maximum upload size is {limit_mb} MB (1 GB)."}
    ), 413


@app.errorhandler(HTTPException)
def handle_http_exception(exc):
    """Return JSON (never an HTML error page) for HTTP errors such as 404/405."""
    return jsonify({"error": exc.description}), exc.code


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only WAV, MP3, M4A, and MP4 files are accepted"}), 400

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
