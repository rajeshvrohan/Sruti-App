import glob
import sys
import requests

URL = "http://127.0.0.1:5000/analyze"

MIME_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4"}

audio_files = glob.glob("*.wav") + glob.glob("*.mp3") + glob.glob("*.m4a")

if not audio_files:
    print("Warning: no WAV, MP3, or M4A file found in the current directory.")
    print("Place a .wav, .mp3, or .m4a file here and re-run.")
    sys.exit(1)

audio_path = audio_files[0]
extension = "." + audio_path.rsplit(".", 1)[-1].lower()
mime_type = MIME_TYPES[extension]

print(f"Sending: {audio_path}")

with open(audio_path, "rb") as f:
    response = requests.post(URL, files={"audio": (audio_path, f, mime_type)})

print(f"Status: {response.status_code}\n")

try:
    data = response.json()
except ValueError:
    print(response.text)
    sys.exit(1)

if "error" in data:
    print(f"Error: {data['error']}")
    sys.exit(1)


def rule(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


rule("DETECTED SWARAS")
swaras = data.get("swaras") or []
print(", ".join(swaras) if swaras else "(none detected)")

rule("TOP RAAGA MATCHES")
raagas = data.get("raagas") or []
if not raagas:
    print("(no matching raagas)")
for i, raaga in enumerate(raagas, 1):
    print(f"{i}. {raaga['name']:20} {raaga['match_percentage']:5.1f}%")
    if raaga.get("matched"):
        print(f"     matched:    {', '.join(raaga['matched'])}")
    if raaga.get("missing"):
        print(f"     missing:    {', '.join(raaga['missing'])}")
    if raaga.get("unexpected"):
        print(f"     unexpected: {', '.join(raaga['unexpected'])}")

analysis = data.get("analysis")
if analysis:
    top_name = raagas[0]["name"] if raagas else "top match"
    rule(f"CLAUDE ANALYSIS — {top_name}")
    for key in ("psychological", "physiological", "meaning"):
        print(f"\n{key.capitalize()}:")
        print(analysis.get(key, "(missing)"))
else:
    rule("CLAUDE ANALYSIS")
    print("(no analysis returned)")
