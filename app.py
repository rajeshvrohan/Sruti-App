import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from audio_processor import AudioProcessor

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB limit

ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a"}

processor = AudioProcessor()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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
        return jsonify({"error": "Only WAV, MP3, and M4A files are accepted"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        result = processor.extract_swaras(filepath)
    except Exception as exc:
        return jsonify({"error": f"Could not analyze audio: {exc}"}), 500

    return jsonify(
        {
            "filename": filename,
            "swaras": result["swaras"],
            "raagas": result["raagas"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
