import glob
import sys
import requests

URL = "http://127.0.0.1:5000/analyze"

MIME_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg"}

audio_files = glob.glob("*.wav") + glob.glob("*.mp3")

if not audio_files:
    print("Warning: no WAV or MP3 file found in the current directory.")
    print("Place a .wav or .mp3 file here and re-run.")
    sys.exit(1)

audio_path = audio_files[0]
extension = "." + audio_path.rsplit(".", 1)[-1].lower()
mime_type = MIME_TYPES[extension]

print(f"Sending: {audio_path}")

with open(audio_path, "rb") as f:
    response = requests.post(URL, files={"audio": (audio_path, f, mime_type)})

print(f"Status: {response.status_code}")
try:
    print(response.json())
except ValueError:
    print(response.text)
