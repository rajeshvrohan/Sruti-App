import sys

from audio_processor import AudioProcessor


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python identify_raaga.py <audio file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    processor = AudioProcessor()

    result = processor.extract_swaras(audio_path)
    swaras = result["swaras"]
    print(f"Detected swaras: {', '.join(swaras) if swaras else '(none)'}")
    if not swaras:
        return

    if not result["raagas"]:
        print("No matching raaga found.")
        return

    print("Candidate raagas:")
    for raaga in result["raagas"]:
        missing = "+".join(raaga["missing"]) if raaga["missing"] else "none"
        unexpected = "+".join(raaga["unexpected"]) if raaga["unexpected"] else "none"
        print(
            f"  {raaga['name']:18} {raaga['match_percentage']:5.1f}%"
            f"  missing: {missing}  unexpected: {unexpected}"
        )


if __name__ == "__main__":
    main()
