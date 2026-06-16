import os
import sys

from audio_processor import AudioProcessor

TEST_FILE = "S.wav"


def main() -> None:
    audio_path = sys.argv[1] if len(sys.argv) > 1 else TEST_FILE

    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        sys.exit(1)

    processor = AudioProcessor()
    print(f"Analyzing: {audio_path}")

    result = processor.extract_swaras(audio_path)
    swaras = result["swaras"]
    if not swaras:
        print("No swaras detected.")
        return

    print(f"Detected swaras: {', '.join(swaras)}")
    print("Top 3 raaga matches:")
    for raaga in result["raagas"]:
        print(f"  {raaga['name']:18} {raaga['match_percentage']:5.1f}%")


if __name__ == "__main__":
    main()
