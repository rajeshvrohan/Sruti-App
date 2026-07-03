import os
import anthropic
from pydantic import BaseModel

from raaga_database import RAAGAS, raaga_swaras

# Min distinct swaras for a raaga to be "dense" enough that janyas blur together;
# pentatonic ragas fall below this and skip the ambiguity note.
MIN_DENSE_SWARAS = 6

# Min fraction of the identified raga's swaras a related raga must share.
MIN_SHARED_FRACTION = 0.6

# Shown with every analysis result.
ANALYSIS_DISCLAIMER = (
    "This analysis is provided for educational purposes only. Uploaded audio is "
    "deleted immediately after processing and is never stored. Please only upload "
    "recordings that you own or have permission to analyze."
)


class RaagaAnalysis(BaseModel):
    psychological: str
    physiological: str
    meaning: str


def _related_janya_ragas(raaga_name: str) -> list[str]:
    """DB ragas whose swaras are a subset of raaga_name's and share >= MIN_SHARED_FRACTION, closest first.

    Empty for sparse (pentatonic) ragas, where this ambiguity is unlikely.
    """
    parent = RAAGAS.get(raaga_name)
    if parent is None:
        return []

    parent_swaras = raaga_swaras(parent)
    if len(parent_swaras) < MIN_DENSE_SWARAS:
        return []

    threshold = MIN_SHARED_FRACTION * len(parent_swaras)
    related = []
    for name, raaga in RAAGAS.items():
        if name == raaga_name:
            continue
        swaras = raaga_swaras(raaga)
        if swaras <= parent_swaras and len(swaras) >= threshold:
            related.append((name, len(swaras)))

    # Closest (largest shared subset) first, then alphabetical.
    related.sort(key=lambda item: (-item[1], item[0]))
    return [name for name, _shared in related]


def _ambiguity_note(raaga_name: str, related: list[str], max_examples: int = 3) -> str:
    """Note acknowledging parent/janya ambiguity under swara detection."""
    shown = related[:max_examples]
    names = ", ".join(shown)
    if len(related) > len(shown):
        names += ", among others"
    return (
        f"{raaga_name} has a dense swara set that it shares closely with related "
        f"janya (derived) ragas such as {names}. Ragas with denser note sets are "
        f"harder to distinguish from their closely related janya ragas using swara "
        f"detection alone, because the difference often lies in characteristic phrase "
        f"patterns (prayogas) and gamakams rather than in which notes are used. This "
        f"result may therefore represent {raaga_name} itself or one of its closely "
        f"related janya ragas."
    )


def get_raaga_analysis(raaga_name: str) -> dict:
    """Ask Claude to explain a raaga's effects and compositions.

    Returns keys `psychological`, `physiological`, `meaning`, always a
    `disclaimer`, and — for dense ragas with close janyas — a `note` flagging
    that swara detection can't separate parent from janya.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = f"""You are an expert in Carnatic classical music, music psychology, and \
music therapy. For the raaga "{raaga_name}", explain three things:

1. The psychological effects this raaga has on listeners (mood, emotion, mental state).
2. The physiological effects this raaga has on listeners (bodily/nervous-system responses).
3. The common themes or meanings found in compositions written in this raaga.

Base your answer on traditional Carnatic understanding of the raaga's rasa (aesthetic \
mood) and any recognized music-therapy associations."""

    message = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        output_format=RaagaAnalysis,
    )

    result = message.parsed_output.model_dump()

    related = _related_janya_ragas(raaga_name)
    if related:
        result["note"] = _ambiguity_note(raaga_name, related)

    result["disclaimer"] = ANALYSIS_DISCLAIMER

    return result


class ClaudeClient:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-6"

    def analyze_carnatic(self, features: dict) -> str:
        prompt = self._build_prompt(features)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text

    def _build_prompt(self, features: dict) -> str:
        return f"""You are an expert in Carnatic classical music. Analyze the following audio features
extracted from a recording and provide insights about the music.

Audio Features:
- Duration: {features.get('duration', 'N/A'):.2f} seconds
- Tempo: {features.get('tempo', 'N/A'):.1f} BPM
- Sample Rate: {features.get('sample_rate', 'N/A')} Hz
- Spectral Centroid Mean: {features.get('spectral_centroid_mean', 'N/A'):.2f} Hz
- Dominant Pitches (Hz): {features.get('dominant_pitches', [])}
- MFCC Means: {features.get('mfcc_means', [])}

Based on these features, please analyze:
1. Possible Raga identification or characteristics
2. Tala (rhythmic cycle) inference from tempo
3. General musical characteristics (mood, style)
4. Any notable observations about the tonal structure

Provide a concise, informative analysis suitable for both students and enthusiasts of Carnatic music."""
